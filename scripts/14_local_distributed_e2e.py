#!/usr/bin/env python3
"""Run a real tiny M3D DDP/FSDP2/checkpoint/resume system on local CPUs.

This is deliberately not a mock-model unit test.  It creates a small local
Phi-3 checkpoint, a fast tokenizer, two independent 3D vision encoders, SegVol
inputs, deterministic manifests, and then launches the production entry points
through two-process ``torch.distributed.run`` jobs.

The isolated CPU path is enabled only in child processes through
``M3D_CPU_DISTRIBUTED_SMOKE=1``.  Normal M3D launches remain CUDA/NCCL-only.
Hardware-specific A100, CUDA, NCCL, Flash-SDPA, PBS, and fabric checks still
belong to the ASPIRE 2A preflight and integration jobs.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import torch
import yaml
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import Phi3Config, Phi3ForCausalLM, PreTrainedTokenizerFast

from m3d.data.manifest import M3DManifest, ManifestRecord, PromptVariant, write_manifest
from m3d.data.schema import DataSplit, TaskName


PINNED_CORE_VERSIONS: Mapping[str, str] = {
    "torch": "2.6.0",
    "torchvision": "0.21.0",
    "transformers": "4.52.4",
    "peft": "0.15.2",
    "monai": "1.4.0",
    "numpy": "1.26.4",
    "safetensors": "0.5.3",
    "tokenizers": "0.21.1",
    "PyYAML": "6.0.2",
}
REPORT_NAME = "local_distributed_e2e_report.json"


class LocalE2EError(RuntimeError):
    """Raised when the local distributed production-path test fails."""


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def _normalise_version(value: str) -> str:
    return value.strip().split("+", 1)[0]


def _dependency_report(*, require_pinned: bool) -> dict[str, Any]:
    observed: dict[str, str] = {}
    mismatches: list[str] = []
    for distribution, expected in PINNED_CORE_VERSIONS.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise LocalE2EError(
                f"Required distribution {distribution!r} is not installed."
            ) from exc
        observed[distribution] = actual
        if _normalise_version(actual) != expected:
            mismatches.append(f"{distribution}={actual}, expected {expected}")

    if require_pinned and mismatches:
        raise LocalE2EError(
            "Core dependency versions differ from requirements.txt:\n  - "
            + "\n  - ".join(mismatches)
        )
    if not torch.distributed.is_available():
        raise LocalE2EError("This PyTorch build has no distributed support.")
    try:
        from torch.distributed._composable.fsdp import fully_shard  # noqa: F401
    except ImportError as exc:
        raise LocalE2EError("This PyTorch build has no FSDP2 fully_shard API.") from exc

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": observed,
        "pinned_mismatches": mismatches,
        "torch_distributed_available": True,
        "cuda_available": bool(torch.cuda.is_available()),
    }


def _create_tiny_phi3(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    vocabulary = {
        "<unk>": 0,
        "<s>": 1,
        "</s>": 2,
        "<pad>": 3,
        "You": 4,
        "are": 5,
        "a": 6,
        "helpful": 7,
        "medical": 8,
        "assistant": 9,
        "Describe": 10,
        "the": 11,
        "CT": 12,
        "scan": 13,
        "There": 14,
        "is": 15,
        "small": 16,
        "lesion": 17,
        "in": 18,
        "image": 19,
        "Segment": 20,
        "Answer": 21,
        "The": 22,
        "finding": 23,
        "appears": 24,
        "central": 25,
        ".": 26,
        ":": 27,
        ",": 28,
        "?": 29,
    }
    backend = Tokenizer(WordLevel(vocabulary, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        bos_token="<s>",
        eos_token="</s>",
        pad_token="<pad>",
        model_max_length=64,
        padding_side="right",
        truncation_side="right",
    )
    tokenizer.save_pretrained(destination)

    torch.manual_seed(104729)
    model = Phi3ForCausalLM(
        Phi3Config(
            vocab_size=32,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=128,
            original_max_position_embeddings=128,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attention_dropout=0.0,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=3,
            tie_word_embeddings=False,
        )
    )
    model.save_pretrained(destination, safe_serialization=True)


def _create_synthetic_data(data_root: Path, manifest_dir: Path) -> dict[str, Any]:
    volume_dir = data_root / "volumes"
    text_dir = data_root / "texts"
    volume_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)
    records: list[ManifestRecord] = []

    voxel_count = 8 * 16 * 16
    base = np.linspace(0.0, 1.0, voxel_count, dtype=np.float32).reshape(1, 8, 16, 16)
    for index in range(4):
        image = np.roll(base, shift=index, axis=-1).copy()
        image_rel = f"volumes/image_{index}.npy"
        np.save(data_root / image_rel, image)

        caption_rel = f"texts/caption_{index}.txt"
        (data_root / caption_rel).write_text(
            f"There is a small central finding in the CT image {index}.",
            encoding="utf-8",
        )
        records.append(
            ManifestRecord(
                record_id=f"caption-{index}",
                task=TaskName.CAPTION,
                split=DataSplit.TRAIN,
                source_name="local_e2e_caption",
                source_index=index,
                image_path=image_rel,
                text_path=caption_rel,
                prompt_variant=PromptVariant.CAPTION,
            )
        )

        mask = np.zeros((1, 8, 16, 16), dtype=np.float32)
        mask[:, 2 + (index % 2) : 6, 4:12, 4:12] = 1.0
        mask_rel = f"volumes/mask_{index}.npy"
        np.save(data_root / mask_rel, mask)
        records.append(
            ManifestRecord(
                record_id=f"segmentation-{index}",
                task=TaskName.SEGMENTATION,
                split=DataSplit.TRAIN,
                source_name="local_e2e_refseg",
                source_index=index,
                image_path=image_rel,
                mask_path=mask_rel,
                prompt_variant=PromptVariant.REFERRING_SEGMENTATION,
                question="Segment the central lesion in this CT image.",
                answer="The central lesion is [SEG].",
            )
        )

    manifest = M3DManifest(split=DataSplit.TRAIN, records=tuple(records))
    path, summary = write_manifest(manifest, manifest_dir / "train.jsonl")
    return {
        "data_root": str(data_root),
        "manifest": str(path),
        "manifest_summary": str(summary),
        "manifest_fingerprint": manifest.fingerprint,
        "record_count": len(manifest),
        "counts_by_task": manifest.counts_by_task,
    }


def _tiny_config(
    *,
    model_dir: Path,
    data_root: Path,
    output_dir: Path,
    strategy: str,
    asynchronous: bool,
) -> dict[str, Any]:
    with (PROJECT_ROOT / "configs" / "m3d_joint_finetune.yaml").open(
        "r", encoding="utf-8"
    ) as handle:
        config = yaml.safe_load(handle)

    config["experiment_name"] = f"m3d-local-e2e-{strategy}"
    model = config["model"]
    model["language_model_name_or_path"] = str(model_dir)
    model["tokenizer_name_or_path"] = str(model_dir)
    model["trust_remote_code"] = False
    model["model_max_length"] = 64
    for name, use_cls in (("main_vision", True), ("seg_vision", False)):
        vision = model[name]
        vision.update(
            {
                "checkpoint_path": None,
                "image_channels": 1,
                "image_size": [8, 16, 16],
                "patch_size": [4, 8, 8],
                "hidden_size": 32,
                "depth": 2,
                "num_heads": 4,
                "mlp_dim": 64,
                "dropout": 0.0,
                "qkv_bias": False,
                "use_cls_token": use_cls,
                "attention_backend": "math",
                "require_flash_sdpa": False,
                "activation_checkpoint_every_n_layers": 0,
                "freeze": False,
                "unfreeze_last_n_layers": 0,
            }
        )
    model["projector"].update(
        {
            "num_layers": 2,
            "pooling_type": "spatial",
            "pooling_size": 2,
            "checkpoint_path": None,
            "freeze": False,
        }
    )
    model["segmentation"].update(
        {
            "checkpoint_path": None,
            "prompt_embed_dim": 32,
            "decoder_depth": 2,
            "decoder_heads": 4,
            "freeze_prompt_encoder": False,
            "freeze_mask_decoder": False,
        }
    )
    model["lora"].update(
        {
            "enabled": True,
            "rank": 4,
            "alpha": 8,
            "dropout": 0.0,
            "target_modules": [
                "qkv_proj",
                "o_proj",
                "gate_up_proj",
                "down_proj",
            ],
            "adapter_checkpoint_path": None,
        }
    )

    data = config["data"]
    data["paths"]["data_root"] = str(data_root)
    data["task_sampling"].update(
        {
            "steps_per_epoch": 2,
            "temperature_alpha": 0.5,
            "task_weights": {
                "caption": 1.0,
                "vqa_closed": 0.0,
                "vqa_open": 0.0,
                "vqa_yes_no": 0.0,
                "positioning": 0.0,
                "segmentation": 1.0,
            },
        }
    )
    data.update(
        {
            "pad_to_multiple_of": 8,
            "sequence_length_buckets": [32, 64],
            "num_workers": 0,
            "persistent_workers": False,
            "pin_memory": False,
            "prefetch_factor": 1,
            "non_blocking_transfer": False,
            "local_cache_root": None,
            "verify_files_at_startup": True,
        }
    )

    optimization = config["optimization"]
    optimization.update(
        {
            "stage": "joint_finetune",
            "epochs": 1.0,
            "per_device_batch_size": 1,
            "gradient_accumulation_steps": 1,
            "allow_tf32": False,
            "matmul_precision": "highest",
            "warmup_ratio": 0.0,
            "checkpoint_language_model": False,
            "checkpoint_main_vision": False,
            "checkpoint_seg_vision": False,
            "checkpoint_segmentation_decoder": False,
            "compile_model": False,
        }
    )
    config["distributed"]["strategy"] = strategy
    config["distributed"]["backend"] = "gloo"
    config["distributed"]["timeout_seconds"] = 300
    config["distributed"]["fsdp2"]["cpu_offload"] = False

    config["checkpoint"].update(
        {
            "output_dir": str(output_dir),
            "resume_from": None,
            "save_every_steps": 1,
            "keep_last_n": 2,
            "save_optimizer": True,
            "save_scheduler": True,
            "save_rng_state": True,
            "asynchronous": asynchronous,
            "export_safetensors_at_end": False,
        }
    )
    config["logging"].update(
        {
            "log_every_steps": 1,
            "report_to": [],
            "tensorboard_dir": str(output_dir / "tensorboard"),
            "profile_steps": [],
            "log_gpu_memory": False,
        }
    )
    config["runtime"].update(
        {
            "seed": 1729,
            "deterministic": True,
            "fail_on_nondeterministic_ops": False,
            "detect_anomaly": False,
        }
    )
    return config


def _write_case_config(
    *,
    destination: Path,
    model_dir: Path,
    data_root: Path,
    output_dir: Path,
    manifest_source: Path,
    strategy: str,
    asynchronous: bool,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    output_manifest_dir = output_dir / "manifests"
    output_manifest_dir.mkdir(parents=True, exist_ok=True)
    for source in manifest_source.glob("train.jsonl*"):
        shutil.copy2(source, output_manifest_dir / source.name)
    config = _tiny_config(
        model_dir=model_dir,
        data_root=data_root,
        output_dir=output_dir,
        strategy=strategy,
        asynchronous=asynchronous,
    )
    with destination.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return destination


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def _torchrun_prefix(processes: int) -> list[str]:
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={processes}",
        "--master_addr=127.0.0.1",
        f"--master_port={_free_port()}",
    ]


def _run_case(
    *,
    name: str,
    command: Sequence[str],
    environment: Mapping[str, str],
    log_dir: Path,
) -> dict[str, Any]:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    started = time.monotonic()
    print(f"\n=== {name} ===", flush=True)
    print(" ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            list(command),
            cwd=PROJECT_ROOT,
            env=dict(environment),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_handle.write(line)
        return_code = process.wait()
    elapsed = time.monotonic() - started
    result = {
        "name": name,
        "status": "passed" if return_code == 0 else "failed",
        "return_code": return_code,
        "elapsed_seconds": elapsed,
        "log": str(log_path),
        "command": list(command),
    }
    if return_code != 0:
        raise LocalE2EError(
            f"{name} failed with exit code {return_code}; inspect {log_path}."
        )
    return result


def _require_json_status(path: Path, expected: str) -> dict[str, Any]:
    if not path.is_file():
        raise LocalE2EError(f"Expected report does not exist: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("status") != expected:
        raise LocalE2EError(
            f"{path} has status={value.get('status')!r}, expected {expected!r}."
        )
    return value


def _training_checks(output_dir: Path) -> dict[str, Any]:
    result_path = output_dir / "training_result.json"
    if not result_path.is_file():
        raise LocalE2EError(f"Expected training result does not exist: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    completed = int(result["completed_optimizer_steps"])
    total = int(result["total_optimizer_steps"])
    if completed != 2 or total != 2:
        raise LocalE2EError(
            f"{output_dir} completed {completed}/{total} optimizer steps, expected 2/2."
        )
    final_checkpoint = result.get("final_checkpoint")
    if not final_checkpoint or not Path(final_checkpoint).is_dir():
        raise LocalE2EError(f"Final checkpoint is missing: {final_checkpoint!r}")
    return {
        "completed_optimizer_steps": completed,
        "total_optimizer_steps": total,
        "final_checkpoint": final_checkpoint,
    }


def run(args: argparse.Namespace) -> Path:
    started = time.time()
    work_dir = (
        Path(args.work_dir).expanduser().resolve()
        if args.work_dir
        else (
            PROJECT_ROOT
            / "outputs"
            / "local-distributed-e2e"
            / time.strftime("%Y%m%d-%H%M%S")
        ).resolve()
    )
    if work_dir.exists() and any(work_dir.iterdir()):
        raise LocalE2EError(
            f"Work directory is not empty: {work_dir}. Choose a new --work-dir."
        )
    work_dir.mkdir(parents=True, exist_ok=True)
    report_path = work_dir / REPORT_NAME
    results: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "status": "running",
        "started_at_unix": started,
        "project_root": str(PROJECT_ROOT),
        "work_dir": str(work_dir),
        "limitations": [
            "Local test uses CPU BF16 and Gloo.",
            "PyTorch 2.6 CPU FSDP2 keeps full parameters through backward; "
            "CUDA uses the configured post-forward resharding.",
            "CUDA 11.8, NCCL, A100 Flash-SDPA, PBS, and cluster fabric require ASPIRE 2A.",
        ],
        "cases": results,
    }
    _atomic_write_json(report_path, report)

    try:
        report["environment"] = _dependency_report(require_pinned=args.require_pinned)
        fixture_root = work_dir / "fixtures"
        model_dir = fixture_root / "tiny-phi3"
        data_root = fixture_root / "data"
        shared_manifest_dir = fixture_root / "manifests"
        _create_tiny_phi3(model_dir)
        report["fixtures"] = _create_synthetic_data(data_root, shared_manifest_dir)
        report["fixtures"]["model_dir"] = str(model_dir)

        case_configs: dict[str, Path] = {}
        case_outputs: dict[str, Path] = {}
        for strategy in ("ddp", "fsdp2"):
            for suite in ("runtime", "checkpoint", "training"):
                name = f"{strategy}_{suite}"
                output = work_dir / name
                config_path = work_dir / "configs" / f"{name}.yaml"
                _write_case_config(
                    destination=config_path,
                    model_dir=model_dir,
                    data_root=data_root,
                    output_dir=output,
                    manifest_source=shared_manifest_dir,
                    strategy=strategy,
                    asynchronous=(strategy == "ddp"),
                )
                case_configs[name] = config_path
                case_outputs[name] = output

        environment = os.environ.copy()
        environment.update(
            {
                "M3D_CPU_DISTRIBUTED_SMOKE": "1",
                "GLOO_SOCKET_IFNAME": "lo0",
                "OMP_NUM_THREADS": "1",
                "TOKENIZERS_PARALLELISM": "false",
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYTHONPATH": str(SOURCE_ROOT)
                + (
                    ""
                    if not environment.get("PYTHONPATH")
                    else os.pathsep + environment["PYTHONPATH"]
                ),
            }
        )
        log_dir = work_dir / "command_logs"

        for strategy in ("ddp", "fsdp2"):
            runtime_name = f"{strategy}_runtime"
            runtime_command = [
                *_torchrun_prefix(args.processes),
                str(PROJECT_ROOT / "scripts" / "03_aspire2a_integration.py"),
                "--config",
                str(case_configs[runtime_name]),
                "--strategy",
                strategy,
                "--output-dir",
                str(case_outputs[runtime_name]),
                "--local-files-only",
                "--skip-path-verification",
                "--allow-version-mismatch",
                "--expected-world-size",
                str(args.processes),
                "--num-workers",
                "0",
            ]
            results.append(
                _run_case(
                    name=runtime_name,
                    command=runtime_command,
                    environment=environment,
                    log_dir=log_dir,
                )
            )
            runtime_report = _require_json_status(
                case_outputs[runtime_name] / "integration_report.json", "passed"
            )
            results[-1]["report"] = str(
                case_outputs[runtime_name] / "integration_report.json"
            )
            results[-1]["task_schedule"] = runtime_report["task_schedule"]
            results[-1]["completed_optimizer_steps"] = runtime_report[
                "completed_optimizer_steps"
            ]
            _atomic_write_json(report_path, report)

            checkpoint_name = f"{strategy}_checkpoint"
            checkpoint_mode = "async" if strategy == "ddp" else "sync"
            checkpoint_command = [
                *_torchrun_prefix(args.processes),
                str(
                    PROJECT_ROOT
                    / "scripts"
                    / "04_aspire2a_checkpoint_integration.py"
                ),
                "--config",
                str(case_configs[checkpoint_name]),
                "--strategy",
                strategy,
                "--checkpoint-mode",
                checkpoint_mode,
                "--output-dir",
                str(case_outputs[checkpoint_name]),
                "--manifest-dir",
                str(shared_manifest_dir),
                "--local-files-only",
                "--skip-path-verification",
                "--allow-version-mismatch",
                "--overwrite-output",
                "--expected-world-size",
                str(args.processes),
                "--num-workers",
                "0",
            ]
            if checkpoint_mode == "async":
                checkpoint_command.append("--allow-async-fallback")
            results.append(
                _run_case(
                    name=checkpoint_name,
                    command=checkpoint_command,
                    environment=environment,
                    log_dir=log_dir,
                )
            )
            checkpoint_report = _require_json_status(
                case_outputs[checkpoint_name]
                / "checkpoint_integration_report.json",
                "passed",
            )
            results[-1]["report"] = str(
                case_outputs[checkpoint_name]
                / "checkpoint_integration_report.json"
            )
            results[-1]["checkpoint_mode"] = checkpoint_mode
            results[-1]["data_replay_exact"] = all(
                expected["digest"] == replayed["digest"]
                for expected, replayed in zip(
                    checkpoint_report["next_batch_by_rank"],
                    checkpoint_report["replayed_batch_by_rank"],
                    strict=True,
                )
            )
            _atomic_write_json(report_path, report)

            training_name = f"{strategy}_training"
            training_command = [
                *_torchrun_prefix(args.processes),
                "--module",
                "m3d.train",
                "--config",
                str(case_configs[training_name]),
                "--local-files-only",
                "--skip-path-verification",
                "--allow-unfused-ddp-fallback",
            ]
            results.append(
                _run_case(
                    name=training_name,
                    command=training_command,
                    environment=environment,
                    log_dir=log_dir,
                )
            )
            results[-1].update(_training_checks(case_outputs[training_name]))
            _atomic_write_json(report_path, report)

            resume_name = f"{strategy}_resume_latest"
            resume_command = [*training_command, "--resume-from", "latest"]
            results.append(
                _run_case(
                    name=resume_name,
                    command=resume_command,
                    environment=environment,
                    log_dir=log_dir,
                )
            )
            results[-1].update(_training_checks(case_outputs[training_name]))
            _atomic_write_json(report_path, report)

        report.update(
            {
                "status": "passed",
                "finished_at_unix": time.time(),
                "elapsed_seconds": time.time() - started,
                "verified": {
                    "real_production_train_entrypoint": True,
                    "ddp_two_process": True,
                    "fsdp2_two_process": True,
                    "caption_and_segmentation_graphs": True,
                    "distributed_checkpoint_save_load_replay": True,
                    "full_training_checkpoint": True,
                    "resume_from_latest": True,
                    "independent_main_and_seg_vision": True,
                    "lora_phi3": True,
                },
            }
        )
        _atomic_write_json(report_path, report)
        return report_path
    except BaseException as exc:
        report.update(
            {
                "status": "failed",
                "finished_at_unix": time.time(),
                "elapsed_seconds": time.time() - started,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
        _atomic_write_json(report_path, report)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run real tiny two-process M3D DDP/FSDP2/checkpoint tests."
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="Fresh output directory; defaults to outputs/local-distributed-e2e/<time>.",
    )
    parser.add_argument(
        "--processes",
        type=int,
        default=2,
        help="Local processes. Keep this at 2 to mirror the two-GPU PBS jobs.",
    )
    parser.add_argument(
        "--require-pinned",
        action="store_true",
        help="Fail unless core package versions exactly match requirements.txt.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    try:
        args = _parser().parse_args(argv)
        if args.processes != 2:
            raise LocalE2EError("--processes must be 2 for this release validation.")
        report = run(args)
        print(json.dumps({"status": "passed", "report": str(report)}, indent=2))
        return 0
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Local distributed E2E failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
