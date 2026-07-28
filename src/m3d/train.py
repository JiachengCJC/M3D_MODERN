"""Distributed training entry point for M3D-Modernized.

Run this module with ``torchrun`` after the deterministic training manifest has
been created::

    torchrun --standalone --nproc_per_node=2 \
        -m m3d.train \
        --config configs/m3d_joint_finetune.yaml

This file deliberately owns only orchestration.  Model architecture, dataset
semantics, distributed wrapping, optimization, checkpointing, and the training
loop remain in their dedicated modules.  Keeping the entry point thin makes the
startup order auditable and prevents subtle mistakes such as constructing an
optimizer before FSDP2 has sharded the parameters.

The required order is:

1. load and validate the resolved YAML configuration;
2. initialize the rank-local CUDA/NCCL runtime;
3. build and cross-rank validate the tokenizer;
4. build the deterministic task-homogeneous training DataLoader;
5. construct the complete M3D model under a common initialization seed;
6. apply DDP or FSDP2;
7. create AdamW *after* distributed wrapping;
8. create the optimizer-step scheduler;
9. create the distributed checkpoint manager;
10. start :class:`m3d.trainer.M3DTrainer`.

Two independent image encoders are preserved throughout this sequence.  The
entry point never aliases, merges, or shares their parameters.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .config import ExperimentConfig, config_fingerprint, load_config


_ENTRYPOINT_STATE_VERSION = 1
_DEFAULT_CONFIG = "configs/m3d_joint_finetune.yaml"


class TrainingEntrypointError(RuntimeError):
    """Raised when startup wiring is invalid before the first microbatch."""


@dataclass(frozen=True, slots=True)
class TrainCLIOptions:
    """Resolved command-line options that are not training hyperparameters."""

    config_path: Path
    overrides: tuple[str, ...]
    cache_dir: Path | None
    local_files_only: bool
    verify_paths: bool | None
    verbose_all_ranks: bool
    startup_only: bool
    save_tokenizer: bool
    allow_unfused_ddp_fallback: bool

    def __post_init__(self) -> None:
        config_path = self.config_path.expanduser().resolve()
        if not config_path.is_file():
            raise FileNotFoundError(f"Configuration file does not exist: {config_path}")
        object.__setattr__(self, "config_path", config_path)

        if self.cache_dir is not None:
            cache_dir = self.cache_dir.expanduser().resolve()
            object.__setattr__(self, "cache_dir", cache_dir)

        for expression in self.overrides:
            if not isinstance(expression, str) or "=" not in expression:
                raise ValueError(
                    "Every --override value must use dotted.path=YAML_VALUE syntax; "
                    f"got {expression!r}."
                )

    def consistency_payload(self) -> dict[str, Any]:
        """Return options that must be identical on every distributed rank."""

        return {
            "config_path": str(self.config_path),
            "overrides": list(self.overrides),
            "local_files_only": self.local_files_only,
            "verify_paths": self.verify_paths,
            "startup_only": self.startup_only,
            "save_tokenizer": self.save_tokenizer,
            "allow_unfused_ddp_fallback": self.allow_unfused_ddp_fallback,
        }

    def to_dict(self) -> dict[str, Any]:
        result = self.consistency_payload()
        result.update(
            {
                "cache_dir": None if self.cache_dir is None else str(self.cache_dir),
                "verbose_all_ranks": self.verbose_all_ranks,
            }
        )
        return result


@dataclass(frozen=True, slots=True)
class StartupReport:
    """Serializable record of the fully constructed training stack."""

    created_at_utc: str
    hostname: str
    process_id: int
    experiment_name: str
    config_path: str
    config_sha256: str
    entrypoint_options: Mapping[str, Any]
    software: Mapping[str, Any]
    git: Mapping[str, Any]
    environment: Mapping[str, Any]
    ranks: tuple[Mapping[str, Any], ...]
    tokenizer: Mapping[str, Any]
    data_pipeline: Mapping[str, Any]
    model: Mapping[str, Any]
    distributed: Mapping[str, Any]
    optimizer: Mapping[str, Any]
    scheduler: Mapping[str, Any]
    startup_only: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_version": _ENTRYPOINT_STATE_VERSION,
            "created_at_utc": self.created_at_utc,
            "hostname": self.hostname,
            "process_id": self.process_id,
            "experiment_name": self.experiment_name,
            "config_path": self.config_path,
            "config_sha256": self.config_sha256,
            "entrypoint_options": dict(self.entrypoint_options),
            "software": dict(self.software),
            "git": dict(self.git),
            "environment": dict(self.environment),
            "ranks": [dict(item) for item in self.ranks],
            "tokenizer": dict(self.tokenizer),
            "data_pipeline": dict(self.data_pipeline),
            "model": dict(self.model),
            "distributed": dict(self.distributed),
            "optimizer": dict(self.optimizer),
            "scheduler": dict(self.scheduler),
            "startup_only": self.startup_only,
        }


@dataclass(frozen=True, slots=True)
class EntrypointResult:
    """Small process-local result returned by :func:`run_training`."""

    status: str
    output_dir: str
    startup_report: str
    training_result: Mapping[str, Any] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "output_dir": self.output_dir,
            "startup_report": self.startup_report,
            "training_result": (
                None if self.training_result is None else dict(self.training_result)
            ),
        }


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m m3d.train",
        description=(
            "Launch task-balanced distributed M3D training. Use torchrun with "
            "two or more CUDA processes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default=_DEFAULT_CONFIG,
        help="Path to the resolved experiment YAML.",
    )
    parser.add_argument(
        "-o",
        "--override",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override a YAML field using dotted.path=YAML_VALUE. Repeat this "
            "option for multiple overrides."
        ),
    )
    parser.add_argument(
        "--resume-from",
        default=None,
        help=(
            "Convenience alias for --override checkpoint.resume_from=PATH. "
            "Exact resume remains controlled by checkpoint compatibility checks."
        ),
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory.",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Disallow network access when loading the tokenizer and Phi-3.",
    )

    verification = parser.add_mutually_exclusive_group()
    verification.add_argument(
        "--verify-paths",
        dest="verify_paths",
        action="store_true",
        help="Force source data/checkpoint path verification during config load.",
    )
    verification.add_argument(
        "--skip-path-verification",
        dest="verify_paths",
        action="store_false",
        help="Skip config-time source path verification.",
    )
    parser.set_defaults(verify_paths=None)

    parser.add_argument(
        "--verbose-all-ranks",
        action="store_true",
        help="Emit normal INFO logs from every rank instead of rank 0 only.",
    )
    parser.add_argument(
        "--startup-only",
        action="store_true",
        help=(
            "Build and validate tokenizer, data, model, distributed wrapper, "
            "optimizer, scheduler, and checkpoint manager, then exit before "
            "the first batch."
        ),
    )
    parser.add_argument(
        "--no-save-tokenizer",
        dest="save_tokenizer",
        action="store_false",
        help="Do not copy tokenizer files into checkpoint.output_dir/tokenizer.",
    )
    parser.set_defaults(save_tokenizer=True)
    parser.add_argument(
        "--allow-unfused-ddp-fallback",
        action="store_true",
        help=(
            "Diagnostic-only escape hatch. By default the primary DDP+A100 path "
            "fails startup unless fused AdamW is active."
        ),
    )
    return parser


def parse_cli_options(argv: Sequence[str] | None = None) -> TrainCLIOptions:
    args = _argument_parser().parse_args(argv)
    overrides = list(args.overrides)
    if args.resume_from is not None:
        # json.dumps produces a valid YAML scalar and safely preserves spaces,
        # colons, and other path characters.
        overrides.append(
            "checkpoint.resume_from=" + json.dumps(str(args.resume_from))
        )
    return TrainCLIOptions(
        config_path=Path(args.config), # "configs/m3d_joint_finetune.yaml"
        overrides=tuple(overrides),
        cache_dir=None if args.cache_dir is None else Path(args.cache_dir),
        local_files_only=bool(args.local_files_only),
        verify_paths=args.verify_paths,
        verbose_all_ranks=bool(args.verbose_all_ranks),
        startup_only=bool(args.startup_only),
        save_tokenizer=bool(args.save_tokenizer),
        allow_unfused_ddp_fallback=bool(args.allow_unfused_ddp_fallback),
    )


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write JSON atomically without relying on trainer/checkpoint internals."""

    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                payload,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                default=str,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            temporary.unlink(missing_ok=True)
        finally:
            raise


def _package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def _software_versions() -> dict[str, Any]:
    cudnn_version = torch.backends.cudnn.version()
    return {
        "python": sys.version.split()[0],
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cudnn": None if cudnn_version is None else int(cudnn_version),
        "transformers": _package_version("transformers"),
        "peft": _package_version("peft"),
        "monai": _package_version("monai"),
        "safetensors": _package_version("safetensors"),
        "numpy": _package_version("numpy"),
    }


def _repository_root() -> Path:
    # <repo>/src/m3d/train.py -> parents[2] == <repo>
    return Path(__file__).resolve().parents[2]


def _git_metadata() -> dict[str, Any]:
    root = _repository_root()

    def run(*arguments: str) -> str | None:
        try:
            completed = subprocess.run(
                ["git", "-C", str(root), *arguments],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=5,
            )
        except (FileNotFoundError, subprocess.SubprocessError):
            return None
        value = completed.stdout.strip()
        return value or None

    revision = run("rev-parse", "HEAD")
    branch = run("rev-parse", "--abbrev-ref", "HEAD")
    dirty_text = run("status", "--porcelain")
    return {
        "repository_root": str(root),
        "revision": revision,
        "branch": branch,
        "dirty": None if dirty_text is None else bool(dirty_text),
    }


def _selected_environment() -> dict[str, Any]:
    """Capture useful launch metadata while excluding secrets and credentials."""

    names = (
        "PBS_JOBID",
        "PBS_JOBNAME",
        "PBS_O_WORKDIR",
        "TMPDIR",
        "CUDA_VISIBLE_DEVICES",
        "TORCH_EXTENSIONS_DIR",
        "NCCL_DEBUG",
        "NCCL_SOCKET_IFNAME",
        "OMP_NUM_THREADS",
        "TOKENIZERS_PARALLELISM",
        "HF_HOME",
        "TRANSFORMERS_CACHE",
    )
    return {name: os.environ.get(name) for name in names if name in os.environ}


def _runtime_rank_payload(runtime: Any) -> dict[str, Any]:
    if runtime.device.type == "cuda":
        properties = torch.cuda.get_device_properties(runtime.device)
        device_name = properties.name
        total_memory = int(properties.total_memory)
        capability = [int(properties.major), int(properties.minor)]
    else:
        device_name = "CPU distributed smoke"
        total_memory = 0
        capability = [0, 0]
    return {
        "rank": runtime.rank,
        "local_rank": runtime.local_rank,
        "world_size": runtime.world_size,
        "local_world_size": runtime.local_world_size,
        "node_rank": runtime.node_rank,
        "hostname": socket.gethostname(),
        "process_id": os.getpid(),
        "device": str(runtime.device),
        "gpu_name": device_name,
        "gpu_total_memory_bytes": total_memory,
        "gpu_compute_capability": capability,
        "rank_seed": int(runtime.seed),
    }


def _validate_entrypoint_config(config: ExperimentConfig) -> None:
    """Reject settings that this entry point would otherwise silently ignore."""

    config.validate()
    if config.optimization.compile_model:
        raise TrainingEntrypointError(
            "optimization.compile_model=true is not yet wired into the stable "
            "training entry point. Keep it false until the dedicated compile "
            "integration file has passed eager/compiled equivalence tests."
        )
    if config.distributed.strategy not in {"ddp", "fsdp2"}:
        raise TrainingEntrypointError(
            f"Unsupported distributed strategy {config.distributed.strategy!r}."
        )
    if not config.data.task_sampling.homogeneous_batches:
        raise TrainingEntrypointError(
            "M3D conditional distributed training requires task-homogeneous batches."
        )
    if not config.data.task_sampling.enabled:
        raise TrainingEntrypointError(
            "The modernized trainer requires the explicit task sampler."
        )


def _load_config(options: TrainCLIOptions) -> ExperimentConfig:
    config = load_config(
        options.config_path,
        options.overrides,
        resolve_paths=True,
        verify_paths=options.verify_paths,
    )
    _validate_entrypoint_config(config)
    return config


def _prepare_output_directory(config: ExperimentConfig, runtime: Any) -> Path:
    # 规范化输出路径
    output_dir = Path(config.checkpoint.output_dir).expanduser().resolve()
    # 只有 rank 0 创建目录
    if runtime.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        _atomic_write_json(output_dir / "resolved_config.json", config.to_dict()) # 未来排查实验时，应该优先看resolved_config.json
    # 所有 rank 等待输出目录准备好
    runtime.barrier()
    if not output_dir.is_dir():
        raise TrainingEntrypointError(
            f"Training output directory is unavailable: {output_dir}"
        )
    return output_dir


def _build_tokenizer_rank_safely(
    *,
    config: ExperimentConfig,
    runtime: Any,
    options: TrainCLIOptions,
) -> Any:
    """Let rank 0 warm the shared cache before other ranks load the tokenizer."""

    from .tokenization import build_tokenizer

    # 让 rank 0 先准备共享缓存，其他 rank 稍后再加载。
    with runtime.main_process_first():
        bundle = build_tokenizer(
            config, # config 告诉 tokenizer：使用哪个语言模型；需要哪些特殊 token；visual token 数量；segmentation token；padding 配置等。。
            cache_dir=options.cache_dir, # cache_dir: Hugging Face 缓存位置。
            local_files_only=options.local_files_only, # local_files_only: 是否允许联网。
        )
    # 提取 tokenizer metadata: 可能包含vocabulary size；special token ID; image token ID...
    metadata_payload = bundle.metadata.to_dict()
    # 每个 rank 比较 metadata fingerprint。
    runtime.assert_all_ranks_equal(
        _sha256_text(json.dumps(metadata_payload, sort_keys=True)),
        label="tokenizer metadata fingerprint",
    )
    return bundle


def _save_tokenizer_once(
    *,
    tokenizer_bundle: Any,
    output_dir: Path,
    runtime: Any,
    enabled: bool,
) -> None:
    if enabled and runtime.is_main_process:
        tokenizer_bundle.save_pretrained(output_dir / "tokenizer")
    runtime.barrier()


def _startup_report(
    *,
    config: ExperimentConfig,
    options: TrainCLIOptions,
    runtime: Any,
    tokenizer_bundle: Any,
    data_pipeline: Any,
    model_report: Any,
    distributed_report: Any,
    optimizer_report: Any,
    scheduler_report: Any,
) -> StartupReport:
    rank_payloads = tuple(runtime.all_gather_object(_runtime_rank_payload(runtime)))
    return StartupReport(
        created_at_utc=_utc_now(),
        hostname=socket.gethostname(),
        process_id=os.getpid(),
        experiment_name=config.experiment_name,
        config_path=str(options.config_path),
        config_sha256=_sha256_text(config_fingerprint(config)),
        entrypoint_options=options.to_dict(),
        software=_software_versions(),
        git=_git_metadata(),
        environment=_selected_environment(),
        ranks=rank_payloads,
        tokenizer=tokenizer_bundle.metadata.to_dict(),
        data_pipeline=dict(data_pipeline.summary()),
        model=model_report.to_dict(),
        distributed=distributed_report.to_dict(),
        optimizer=optimizer_report.to_dict(include_parameter_names=False),
        scheduler=scheduler_report.to_dict(),
        startup_only=options.startup_only,
    )


def _validate_optimizer_backend(
    *,
    strategy: str,
    report: Any,
    allow_unfused_ddp_fallback: bool,
) -> None:
    if strategy == "ddp" and not report.fused_enabled:
        message = (
            "The primary DDP path did not enable fused AdamW. "
            f"Reason: {report.fused_fallback_reason!r}."
        )
        if not allow_unfused_ddp_fallback:
            raise TrainingEntrypointError(
                message
                + " Use the supported ASPIRE 2A PyTorch/CUDA environment, or pass "
                "--allow-unfused-ddp-fallback only for diagnostics."
            )


def run_training(options: TrainCLIOptions) -> EntrypointResult:
    """Construct the complete stack and execute the configured training plan."""

    if not isinstance(options, TrainCLIOptions):
        raise TypeError("options must be TrainCLIOptions")

    # Heavy optional dependencies are imported lazily so ``--help``, py_compile,
    # and the dependency-light self-test work outside the cluster environment.
    from .checkpointing import CheckpointManager
    from .data.loader import build_training_data_pipeline
    from .distributed import build_model_synchronously, prepare_distributed_model
    from .model.m3d import build_m3d_model
    from .optim import build_optimizer
    from .runtime import distributed_runtime
    from .scheduler import build_scheduler
    from .tokenization import M3DTextProcessor
    from .trainer import M3DTrainer

    config = _load_config(options)
    checkpoint_manager: Any | None = None # 一开始还没有创建 CheckpointManager。但是后面无论在哪一步失败，都需要知道是否已经创建，以便 cleanup。
    training_succeeded = False

    with distributed_runtime(
        config,
        verbose_all_ranks=options.verbose_all_ranks,
    ) as runtime:
        # 检查所有 rank 的启动参数一致
        runtime.assert_all_ranks_equal(
            _sha256_text(
                json.dumps(options.consistency_payload(), sort_keys=True)
            ),
            label="training entrypoint options",
        )
        # 准备输出目录,进入 prepare_output_directory
        output_dir = _prepare_output_directory(config, runtime)
        # 记录训练构建开始，默认通常只有 rank 0 输出。
        runtime.logger.info(
            "Starting M3D construction: experiment=%s config=%s strategy=%s",
            config.experiment_name,
            options.config_path,
            config.distributed.strategy,
        )

        try:
            # 安全构建 tokenizer
            tokenizer_bundle = _build_tokenizer_rank_safely(
                config=config,
                runtime=runtime,
                options=options,
            )
            _save_tokenizer_once(
                tokenizer_bundle=tokenizer_bundle,
                output_dir=output_dir,
                runtime=runtime,
                enabled=options.save_tokenizer,
            )
            text_processor = M3DTextProcessor(tokenizer_bundle, config)

            # Build data before allocating the 4B language model. Missing or
            # invalid manifests therefore fail without consuming most GPU memory.
            data_pipeline = build_training_data_pipeline(
                config=config,
                runtime=runtime,
                tokenizer_bundle=tokenizer_bundle,
                text_processor=text_processor,
            )

            model, model_report = build_model_synchronously(
                runtime,
                lambda: build_m3d_model(
                    config=config,
                    tokenizer_bundle=tokenizer_bundle,
                    cache_dir=options.cache_dir,
                    local_files_only=options.local_files_only,
                    torch_dtype=torch.bfloat16,
                    load_pretrained_components=True,
                    strict_pretrained=True,
                ),
            )

            distributed_model, distributed_report = prepare_distributed_model(
                model,
                runtime,
            )

            allow_optimizer_fallback = bool(
                config.distributed.strategy == "fsdp2"
                or options.allow_unfused_ddp_fallback
            )
            optimizer, optimizer_report = build_optimizer(
                distributed_model.unwrapped_model,
                config,
                distributed_strategy=distributed_model.strategy,
                allow_unfused_fallback=allow_optimizer_fallback,
            )
            _validate_optimizer_backend(
                strategy=distributed_model.strategy,
                report=optimizer_report,
                allow_unfused_ddp_fallback=options.allow_unfused_ddp_fallback,
            )

            scheduler, scheduler_report = build_scheduler(
                optimizer,
                config,
                steps_per_epoch=data_pipeline.steps_per_epoch,
            )

            checkpoint_manager = CheckpointManager(
                config=config,
                runtime=runtime,
                distributed_model=distributed_model,
                optimizer=optimizer,
                scheduler=scheduler,
                data_pipeline=data_pipeline,
            )

            startup = _startup_report(
                config=config,
                options=options,
                runtime=runtime,
                tokenizer_bundle=tokenizer_bundle,
                data_pipeline=data_pipeline,
                model_report=model_report,
                distributed_report=distributed_report,
                optimizer_report=optimizer_report,
                scheduler_report=scheduler_report,
            )
            startup_path = output_dir / "startup_report.json"
            if runtime.is_main_process:
                _atomic_write_json(startup_path, startup.to_dict())
            runtime.barrier()

            runtime.logger.info(
                "M3D startup validation complete: optimizer_steps=%d "
                "microbatches=%d visual_tokens=%d",
                scheduler.total_optimizer_steps,
                scheduler.plan.total_microbatches,
                tokenizer_bundle.metadata.visual_token_count,
            )

            if options.startup_only:
                checkpoint_manager.close()
                checkpoint_manager = None
                runtime.logger.info(
                    "Startup-only mode complete; no training batch was consumed."
                )
                return EntrypointResult(
                    status="startup_validated",
                    output_dir=str(output_dir),
                    startup_report=str(startup_path),
                    training_result=None,
                )

            trainer = M3DTrainer(
                config=config,
                runtime=runtime,
                distributed_model=distributed_model,
                optimizer=optimizer,
                scheduler=scheduler,
                data_pipeline=data_pipeline,
                checkpoint_manager=checkpoint_manager,
            )
            result = trainer.train(resume=True)
            training_succeeded = True

            runtime.logger.info(
                "M3D training complete: optimizer_steps=%d/%d elapsed=%.2fs "
                "checkpoint=%s",
                result.completed_optimizer_steps,
                result.total_optimizer_steps,
                result.elapsed_seconds,
                result.final_checkpoint,
            )
            if config.checkpoint.export_safetensors_at_end:
                runtime.logger.info(
                    "The distributed checkpoint is complete. Final consolidated "
                    "safetensors export is intentionally handled by the separate "
                    "export entry point so FSDP2 never gathers a full model during "
                    "the training process."
                )

            return EntrypointResult(
                status="training_complete",
                output_dir=str(output_dir),
                startup_report=str(startup_path),
                training_result=result.to_dict(),
            )

        finally:
            # On normal completion M3DTrainer already closes the manager. Calling
            # close again is harmless because the auxiliary group is set to None.
            # On startup or training failure this releases any pending DCP writer
            # and auxiliary Gloo process group before the default NCCL group exits.
            if checkpoint_manager is not None:
                try:
                    checkpoint_manager.close()
                except Exception:
                    runtime.logger.exception(
                        "Checkpoint manager cleanup failed after success=%s",
                        training_succeeded,
                    )
            if runtime.device.type == "cuda":
                torch.cuda.empty_cache()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point used by ``python -m m3d.train``."""

    try:
        # 解析 train.py 自己的启动参数
        options = parse_cli_options(argv)
        result = run_training(options)
        # Only rank 0 logs normally, but printing here occurs after the runtime
        # closes on every rank. Keep the payload compact and machine-readable.
        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
        return 0
    except KeyboardInterrupt:
        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            print("M3D training interrupted by user.", file=sys.stderr)
        return 130
    except Exception as exc:
        rank = int(os.environ.get("RANK", "0"))
        if rank == 0:
            print(
                f"M3D training entry point failed: {type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
            traceback.print_exc()
        return 1


# ---------------------------------------------------------------------------
# Dependency-light self-test
# ---------------------------------------------------------------------------


def _self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="m3d-train-self-test-") as directory:
        root = Path(directory)
        config_path = root / "config.yaml"
        config_path.write_text("schema_version: 1\n", encoding="utf-8")

        parsed = parse_cli_options(
            [
                "--config",
                str(config_path),
                "--override",
                "optimization.epochs=2.5",
                "--resume-from",
                str(root / "checkpoint step 10"),
                "--cache-dir",
                str(root / "cache"),
                "--local-files-only",
                "--startup-only",
                "--no-save-tokenizer",
            ]
        )
        assert parsed.config_path == config_path.resolve()
        assert parsed.local_files_only
        assert parsed.startup_only
        assert not parsed.save_tokenizer
        assert len(parsed.overrides) == 2
        assert parsed.overrides[0] == "optimization.epochs=2.5"
        assert parsed.overrides[1].startswith("checkpoint.resume_from=")

        latest = parse_cli_options(
            ["--config", str(config_path), "--resume-from", "latest"]
        )
        latest_config = load_config(
            config_path,
            latest.overrides,
            resolve_paths=True,
            verify_paths=False,
        )
        assert latest_config.checkpoint.resume_from == "latest"

        payload = {
            "state_version": _ENTRYPOINT_STATE_VERSION,
            "sha256": _sha256_text("m3d"),
            "options": parsed.to_dict(),
        }
        output = root / "nested" / "report.json"
        _atomic_write_json(output, payload)
        loaded = json.loads(output.read_text(encoding="utf-8"))
        assert loaded["sha256"] == hashlib.sha256(b"m3d").hexdigest()
        assert not list(output.parent.glob("*.tmp"))

        consistency_a = _sha256_text(
            json.dumps(parsed.consistency_payload(), sort_keys=True)
        )
        consistency_b = _sha256_text(
            json.dumps(parsed.consistency_payload(), sort_keys=True)
        )
        assert consistency_a == consistency_b

        return {
            "status": "passed",
            "override_count": len(parsed.overrides),
            "resume_path_yaml_quoted": "checkpoint step 10" in parsed.overrides[1],
            "latest_resume_sentinel_preserved": (
                latest_config.checkpoint.resume_from == "latest"
            ),
            "atomic_json_roundtrip": loaded == payload,
            "consistency_fingerprint_stable": consistency_a == consistency_b,
            "entrypoint_state_version": _ENTRYPOINT_STATE_VERSION,
        }


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        print(json.dumps(_self_test(), indent=2, sort_keys=True))
        raise SystemExit(0)
    # main() 会返回整数,0 成功,1 普通失败,130 用户中断
    raise SystemExit(main())


__all__ = [
    "EntrypointResult",
    "StartupReport",
    "TrainCLIOptions",
    "TrainingEntrypointError",
    "main",
    "parse_cli_options",
    "run_training",
]
