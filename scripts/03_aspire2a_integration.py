#!/usr/bin/env python3
"""Two-GPU ASPIRE 2A integration test for M3D-Modernized.

This script is deliberately stronger than ``m3d.train --startup-only``.  It
constructs the real tokenizer, the published checkpoints, both independent 3D
image encoders, Phi-3, LoRA, DDP/FSDP2, the component-aware optimizer and the
cosine scheduler.  It then consumes exactly two task-homogeneous distributed
microbatches from the real training manifest:

1. one caption microbatch, which must skip the complete SegVol branch; and
2. one segmentation microbatch, which must execute the independent SegVol 3D
   ViT, prompt encoder and mask decoder.

Both microbatches run a real BF16 forward, globally normalised backward,
gradient clipping, optimizer update and scheduler update.  The test also
restricts a standalone SDPA probe to the Flash backend, verifies that DDP uses
fused AdamW, and checks branch-specific gradients before each optimizer step.

The integration run is intentionally tiny and does not save the multi-gigabyte
model/optimizer checkpoint.  Distributed checkpoint save/resume is tested by a
separate checkpoint integration file so a basic compute smoke test remains
fast enough to run before every long PBS job.

Recommended ASPIRE 2A launch::

    export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

    torchrun \
      --standalone \
      --nproc_per_node=2 \
      scripts/03_aspire2a_integration.py \
      --config configs/m3d_joint_finetune.yaml \
      --strategy ddp

FSDP2 memory-fallback validation::

    torchrun \
      --standalone \
      --nproc_per_node=2 \
      scripts/03_aspire2a_integration.py \
      --config configs/m3d_joint_finetune.yaml \
      --strategy fsdp2

The final machine-readable report is written to::

    <output-dir>/integration_report.json
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import socket
import sys
import tempfile
import time
import traceback
from typing import Any, Iterable, Mapping, Sequence

# Make the source tree importable when this file is executed directly from the
# repository without requiring an editable installation.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import torch
from torch import Tensor, nn


_INTEGRATION_STATE_VERSION = 1
_EXPECTED_PACKAGE_VERSIONS: Mapping[str, str] = {
    "torch": "2.6.0",
    "torchvision": "0.21.0",
    "transformers": "4.52.4",
    "peft": "0.15.2",
    "monai": "1.4.0",
    "numpy": "1.26.4",
}
_EXPECTED_PYTHON = (3, 10, 9)
_EXPECTED_TORCH_CUDA = "11.8"
_REQUIRED_TASKS = ("caption", "segmentation")


class IntegrationError(RuntimeError):
    """Base error for failed cluster integration contracts."""


class EnvironmentContractError(IntegrationError):
    """Raised when the compute-node software or GPU contract is wrong."""


class GradientContractError(IntegrationError):
    """Raised when the observed branch gradients do not match the task graph."""


class IntegrationProgressError(IntegrationError):
    """Raised when the two-step data/scheduler plan diverges."""


@dataclasses.dataclass(frozen=True, slots=True)
class IntegrationOptions:
    """Resolved command-line settings shared by every distributed rank."""

    config_path: str
    strategy: str
    output_dir: str
    cache_dir: str | None
    local_files_only: bool
    overrides: tuple[str, ...]
    expected_world_size: int
    num_workers: int
    allow_non_a100: bool
    allow_version_mismatch: bool
    verify_paths: bool
    verbose_all_ranks: bool

    def consistency_payload(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class EnvironmentReport:
    """One rank's validated software, topology and GPU description."""

    rank: int
    local_rank: int
    world_size: int
    hostname: str
    process_id: int
    python_version: str
    torch_version: str
    torch_cuda_version: str | None
    cudnn_version: int | None
    nccl_version: str | None
    distributed_backend: str
    gpu_name: str
    gpu_compute_capability: tuple[int, int]
    gpu_total_memory_bytes: int
    bf16_supported: bool
    flash_sdpa_enabled: bool
    package_versions: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["gpu_compute_capability"] = list(self.gpu_compute_capability)
        payload["package_versions"] = dict(self.package_versions)
        return payload


@dataclasses.dataclass(frozen=True, slots=True)
class FlashProbeReport:
    """Result of an explicitly Flash-only BF16 SDPA forward/backward."""

    qkv_shape: tuple[int, int, int, int]
    output_shape: tuple[int, ...]
    output_dtype: str
    loss: float
    q_gradient_finite: bool
    k_gradient_finite: bool
    v_gradient_finite: bool
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["qkv_shape"] = list(self.qkv_shape)
        payload["output_shape"] = list(self.output_shape)
        return payload


@dataclasses.dataclass(frozen=True, slots=True)
class GradientSummary:
    """Aggregate local-shard gradient status for one logical model component."""

    trainable_parameter_tensors: int
    gradient_parameter_tensors: int
    finite_gradient_parameter_tensors: int
    nonzero_gradient_parameter_tensors: int
    local_gradient_elements: int
    local_nonzero_gradient_elements: int
    local_max_abs_gradient: float

    @property
    def has_gradient(self) -> bool:
        return self.gradient_parameter_tensors > 0

    @property
    def all_gradients_finite(self) -> bool:
        return (
            self.gradient_parameter_tensors
            == self.finite_gradient_parameter_tensors
        )

    @property
    def has_nonzero_gradient(self) -> bool:
        return self.nonzero_gradient_parameter_tensors > 0

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["has_gradient"] = self.has_gradient
        payload["all_gradients_finite"] = self.all_gradients_finite
        payload["has_nonzero_gradient"] = self.has_nonzero_gradient
        return payload


@dataclasses.dataclass(frozen=True, slots=True)
class MicrobatchReport:
    """One rank's result for one real task-homogeneous optimizer update."""

    step: int
    task: str
    sample_ids: tuple[str, ...]
    sequence_length: int
    unpadded_lengths: tuple[int, ...]
    image_shape: tuple[int, ...]
    segmentation_target_shape: tuple[int, ...] | None
    segmentation_logit_shape: tuple[int, ...] | None
    foreground_voxel_count: int | None
    language_token_count: int
    local_total_loss: float
    global_backward_loss: float
    language_loss: float
    dice_loss: float | None
    bce_loss: float | None
    gradient_norm: float
    learning_rates: Mapping[str, float]
    gradients: Mapping[str, GradientSummary]
    elapsed_seconds: float
    cuda_memory: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": self.step,
            "task": self.task,
            "sample_ids": list(self.sample_ids),
            "sequence_length": self.sequence_length,
            "unpadded_lengths": list(self.unpadded_lengths),
            "image_shape": list(self.image_shape),
            "segmentation_target_shape": (
                None
                if self.segmentation_target_shape is None
                else list(self.segmentation_target_shape)
            ),
            "segmentation_logit_shape": (
                None
                if self.segmentation_logit_shape is None
                else list(self.segmentation_logit_shape)
            ),
            "foreground_voxel_count": self.foreground_voxel_count,
            "language_token_count": self.language_token_count,
            "local_total_loss": self.local_total_loss,
            "global_backward_loss": self.global_backward_loss,
            "language_loss": self.language_loss,
            "dice_loss": self.dice_loss,
            "bce_loss": self.bce_loss,
            "gradient_norm": self.gradient_norm,
            "learning_rates": dict(self.learning_rates),
            "gradients": {
                name: summary.to_dict()
                for name, summary in self.gradients.items()
            },
            "elapsed_seconds": self.elapsed_seconds,
            "cuda_memory": dict(self.cuda_memory),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class IntegrationReport:
    """Rank-0 aggregate proving the real two-task training graph executed."""

    state_version: int
    status: str
    strategy: str
    config_path: str
    output_dir: str
    started_at_unix: float
    finished_at_unix: float
    elapsed_seconds: float
    expected_world_size: int
    task_schedule: tuple[str, ...]
    environment_by_rank: tuple[EnvironmentReport, ...]
    flash_probe_by_rank: tuple[FlashProbeReport, ...]
    tokenizer: Mapping[str, Any]
    data_pipeline: Mapping[str, Any]
    model: Mapping[str, Any]
    distributed: Mapping[str, Any]
    optimizer: Mapping[str, Any]
    scheduler: Mapping[str, Any]
    microbatches_by_rank: tuple[tuple[MicrobatchReport, ...], ...]
    completed_optimizer_steps: int
    committed_microbatches: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_version": self.state_version,
            "status": self.status,
            "strategy": self.strategy,
            "config_path": self.config_path,
            "output_dir": self.output_dir,
            "started_at_unix": self.started_at_unix,
            "finished_at_unix": self.finished_at_unix,
            "elapsed_seconds": self.elapsed_seconds,
            "expected_world_size": self.expected_world_size,
            "task_schedule": list(self.task_schedule),
            "environment_by_rank": [
                item.to_dict() for item in self.environment_by_rank
            ],
            "flash_probe_by_rank": [
                item.to_dict() for item in self.flash_probe_by_rank
            ],
            "tokenizer": dict(self.tokenizer),
            "data_pipeline": dict(self.data_pipeline),
            "model": dict(self.model),
            "distributed": dict(self.distributed),
            "optimizer": dict(self.optimizer),
            "scheduler": dict(self.scheduler),
            "microbatches_by_rank": [
                [item.to_dict() for item in rank_items]
                for rank_items in self.microbatches_by_rank
            ],
            "completed_optimizer_steps": self.completed_optimizer_steps,
            "committed_microbatches": self.committed_microbatches,
        }


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _normalise_version(value: str) -> str:
    """Drop local build suffixes such as ``+cu118`` for exact base checks."""

    return value.strip().split("+", 1)[0]


def _package_version(distribution_name: str) -> str:
    if distribution_name == "torch":
        return str(torch.__version__)
    try:
        return importlib.metadata.version(distribution_name)
    except importlib.metadata.PackageNotFoundError as exc:
        raise EnvironmentContractError(
            f"Required Python package {distribution_name!r} is not installed."
        ) from exc


def _nccl_version_string() -> str | None:
    try:
        version = torch.cuda.nccl.version()
    except Exception:
        return None
    if version is None:
        return None
    if isinstance(version, tuple):
        return ".".join(str(int(part)) for part in version)
    return str(version)


def _tensor_local_shard(tensor: Tensor) -> Tensor:
    """Return a local Tensor for regular tensors and FSDP2 DTensors."""

    to_local = getattr(tensor, "to_local", None)
    if callable(to_local):
        local = to_local()
        if not isinstance(local, Tensor):
            raise TypeError(
                f"DTensor.to_local() returned {type(local).__name__}, not Tensor."
            )
        return local
    return tensor


def _safe_scalar(value: Tensor | float | int) -> float:
    if isinstance(value, Tensor):
        if value.numel() != 1:
            raise ValueError(f"Expected scalar tensor, got {tuple(value.shape)}.")
        local = _tensor_local_shard(value.detach())
        if local.numel() != 1:
            raise ValueError(
                "Expected a scalar local shard, got " f"{tuple(local.shape)}."
            )
        return float(local.float().cpu().item())
    return float(value)


def _gradient_summary(module: nn.Module) -> GradientSummary:
    trainable = 0
    with_grad = 0
    finite = 0
    nonzero_tensors = 0
    local_elements = 0
    nonzero_elements = 0
    max_abs = 0.0

    seen: set[int] = set()
    for parameter in module.parameters(recurse=True):
        identity = id(parameter)
        if identity in seen:
            continue
        seen.add(identity)
        if not parameter.requires_grad:
            continue
        trainable += 1
        gradient = parameter.grad
        if gradient is None:
            continue
        with_grad += 1
        local = _tensor_local_shard(gradient.detach())
        local_elements += int(local.numel())
        if local.numel() == 0:
            finite += 1
            continue
        finite_mask = torch.isfinite(local)
        if bool(finite_mask.all().item()):
            finite += 1
        absolute = local.float().abs()
        local_nonzero = int(torch.count_nonzero(absolute).item())
        nonzero_elements += local_nonzero
        if local_nonzero > 0:
            nonzero_tensors += 1
        max_abs = max(max_abs, float(absolute.max().cpu().item()))

    return GradientSummary(
        trainable_parameter_tensors=trainable,
        gradient_parameter_tensors=with_grad,
        finite_gradient_parameter_tensors=finite,
        nonzero_gradient_parameter_tensors=nonzero_tensors,
        local_gradient_elements=local_elements,
        local_nonzero_gradient_elements=nonzero_elements,
        local_max_abs_gradient=max_abs,
    )


def _learning_rates_by_role(optimizer: torch.optim.Optimizer) -> dict[str, float]:
    by_role: dict[str, set[float]] = {}
    for group in optimizer.param_groups:
        role = group.get("role")
        if not isinstance(role, str) or not role:
            raise IntegrationError("Optimizer group is missing stable role metadata.")
        by_role.setdefault(role, set()).add(float(group["lr"]))
    inconsistent = {
        role: sorted(values)
        for role, values in by_role.items()
        if len(values) != 1
    }
    if inconsistent:
        raise IntegrationError(
            "Decay/no-decay groups disagree on role learning rate: "
            f"{inconsistent}."
        )
    return {role: next(iter(values)) for role, values in sorted(by_role.items())}


def _task_weights() -> dict[str, float]:
    return {
        "caption": 1.0,
        "vqa_closed": 0.0,
        "vqa_open": 0.0,
        "vqa_yes_no": 0.0,
        "positioning": 0.0,
        "segmentation": 1.0,
    }


# ---------------------------------------------------------------------------
# Environment and Flash-SDPA probes
# ---------------------------------------------------------------------------


def _validate_versions(*, allow_mismatch: bool) -> dict[str, str]:
    observed = {
        name: _package_version(name)
        for name in _EXPECTED_PACKAGE_VERSIONS
    }
    mismatches: list[str] = []

    python_tuple = tuple(sys.version_info[:3])
    if python_tuple != _EXPECTED_PYTHON:
        mismatches.append(
            f"Python {platform.python_version()} != "
            f"{'.'.join(map(str, _EXPECTED_PYTHON))}"
        )

    for name, expected in _EXPECTED_PACKAGE_VERSIONS.items():
        actual = _normalise_version(observed[name])
        if actual != expected:
            mismatches.append(f"{name} {actual} != {expected}")

    torch_cuda = None if torch.version.cuda is None else str(torch.version.cuda)
    if torch_cuda != _EXPECTED_TORCH_CUDA:
        mismatches.append(
            f"torch.version.cuda {torch_cuda!r} != {_EXPECTED_TORCH_CUDA!r}"
        )

    if mismatches and not allow_mismatch:
        formatted = "\n".join(f"  - {item}" for item in mismatches)
        raise EnvironmentContractError(
            "ASPIRE 2A software contract mismatch:\n" + formatted
        )
    return observed


def _environment_report(runtime: Any, options: IntegrationOptions) -> EnvironmentReport:
    import torch.distributed as dist

    if runtime.world_size != options.expected_world_size:
        raise EnvironmentContractError(
            f"Expected world_size={options.expected_world_size}, got "
            f"{runtime.world_size}. Launch with torchrun --nproc_per_node="
            f"{options.expected_world_size}."
        )
    if runtime.world_size <= 1 or not runtime.process_group_initialized:
        raise EnvironmentContractError(
            "This integration test requires an initialized multi-process group."
        )
    if dist.get_backend() != "nccl":
        raise EnvironmentContractError(
            f"Expected NCCL default process group, got {dist.get_backend()!r}."
        )
    if not torch.cuda.is_available():
        raise EnvironmentContractError("CUDA is unavailable on this process.")
    if runtime.device.type != "cuda":
        raise EnvironmentContractError(
            f"Runtime device must be CUDA, got {runtime.device}."
        )
    if torch.cuda.current_device() != runtime.local_rank:
        raise EnvironmentContractError(
            "The process is not bound to its torchrun local rank: "
            f"current_device={torch.cuda.current_device()}, "
            f"local_rank={runtime.local_rank}."
        )

    properties = torch.cuda.get_device_properties(runtime.device)
    capability = (int(properties.major), int(properties.minor))
    is_a100 = "A100" in properties.name.upper() and capability == (8, 0)
    if not is_a100 and not options.allow_non_a100:
        raise EnvironmentContractError(
            "Expected an NVIDIA A100 (compute capability 8.0), got "
            f"{properties.name!r} capability={capability}."
        )
    bf16_supported = bool(torch.cuda.is_bf16_supported())
    if not bf16_supported:
        raise EnvironmentContractError(
            f"GPU {properties.name!r} does not report BF16 support."
        )
    flash_enabled = bool(torch.backends.cuda.flash_sdp_enabled())
    if not flash_enabled:
        raise EnvironmentContractError(
            "torch.backends.cuda.flash_sdp_enabled() is false."
        )

    package_versions = _validate_versions(
        allow_mismatch=options.allow_version_mismatch
    )
    return EnvironmentReport(
        rank=int(runtime.rank),
        local_rank=int(runtime.local_rank),
        world_size=int(runtime.world_size),
        hostname=socket.gethostname(),
        process_id=os.getpid(),
        python_version=platform.python_version(),
        torch_version=str(torch.__version__),
        torch_cuda_version=(
            None if torch.version.cuda is None else str(torch.version.cuda)
        ),
        cudnn_version=torch.backends.cudnn.version(),
        nccl_version=_nccl_version_string(),
        distributed_backend=str(dist.get_backend()),
        gpu_name=properties.name,
        gpu_compute_capability=capability,
        gpu_total_memory_bytes=int(properties.total_memory),
        bf16_supported=bf16_supported,
        flash_sdpa_enabled=flash_enabled,
        package_versions=package_versions,
    )


def _run_flash_probe(runtime: Any) -> FlashProbeReport:
    from m3d.model.attention import (
        AttentionPolicy,
        scaled_dot_product_attention,
    )

    shape = (1, 8, 128, 64)
    generator = torch.Generator(device=runtime.device)
    generator.manual_seed(1729 + int(runtime.rank))
    q = torch.randn(
        shape,
        device=runtime.device,
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=True,
    )
    k = torch.randn(
        shape,
        device=runtime.device,
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=True,
    )
    v = torch.randn(
        shape,
        device=runtime.device,
        dtype=torch.bfloat16,
        generator=generator,
        requires_grad=True,
    )
    torch.cuda.synchronize(runtime.device)
    started = time.perf_counter()
    output = scaled_dot_product_attention(
        q,
        k,
        v,
        policy=AttentionPolicy(backend="sdpa", require_flash=True),
        training=True,
        dropout_p=0.0,
    )
    loss = output.float().square().mean()
    loss.backward()
    torch.cuda.synchronize(runtime.device)
    elapsed = time.perf_counter() - started

    gradients = (q.grad, k.grad, v.grad)
    finite = [
        gradient is not None and bool(torch.isfinite(gradient).all().item())
        for gradient in gradients
    ]
    if not all(finite):
        raise EnvironmentContractError(
            "Flash-SDPA probe produced missing or non-finite Q/K/V gradients."
        )

    report = FlashProbeReport(
        qkv_shape=shape,
        output_shape=tuple(int(item) for item in output.shape),
        output_dtype=str(output.dtype),
        loss=float(loss.detach().cpu().item()),
        q_gradient_finite=finite[0],
        k_gradient_finite=finite[1],
        v_gradient_finite=finite[2],
        elapsed_seconds=elapsed,
    )
    del q, k, v, output, loss
    torch.cuda.empty_cache()
    return report


# ---------------------------------------------------------------------------
# Integration-specific configuration and model checks
# ---------------------------------------------------------------------------


def _build_integration_config(options: IntegrationOptions) -> Any:
    from m3d.config import load_config

    config = load_config(
        options.config_path,
        options.overrides,
        resolve_paths=True,
        verify_paths=options.verify_paths,
    )
    config.experiment_name = f"{config.experiment_name}-integration-{options.strategy}"
    config.distributed.strategy = options.strategy

    # Exactly two deterministic task-homogeneous microbatches: one text graph
    # and one segmentation graph. The sampler guarantees each active task once.
    config.data.task_sampling.task_weights = _task_weights()
    config.data.task_sampling.steps_per_epoch = 2
    config.data.num_workers = int(options.num_workers)
    config.data.persistent_workers = bool(options.num_workers > 0)
    if options.num_workers == 0:
        config.data.prefetch_factor = 1  # ignored by WorkerSettings/DataLoader

    config.optimization.epochs = 1.0
    config.optimization.per_device_batch_size = 1
    config.optimization.gradient_accumulation_steps = 1
    config.optimization.warmup_ratio = 0.0
    config.optimization.compile_model = False

    config.checkpoint.output_dir = options.output_dir
    config.checkpoint.resume_from = None
    config.checkpoint.asynchronous = False
    config.checkpoint.keep_last_n = 1
    config.checkpoint.save_every_steps = 2
    config.checkpoint.export_safetensors_at_end = False

    config.logging.log_every_steps = 1
    config.logging.report_to = ()
    config.logging.profile_steps = ()
    config.logging.tensorboard_dir = str(Path(options.output_dir) / "tensorboard")
    config.validate()
    return config


def _validate_schedule(data_pipeline: Any, runtime: Any) -> tuple[str, ...]:
    schedule = tuple(task.value for task in data_pipeline.batch_sampler.schedule.tasks)
    counts = {
        task.value: int(count)
        for task, count in data_pipeline.batch_sampler.schedule.counts.items()
    }
    if len(schedule) != 2 or sorted(schedule) != sorted(_REQUIRED_TASKS):
        raise IntegrationProgressError(
            "Integration sampler must schedule exactly one caption and one "
            f"segmentation microbatch, got schedule={schedule}, counts={counts}."
        )
    if counts != {"caption": 1, "segmentation": 1}:
        raise IntegrationProgressError(
            f"Unexpected integration task counts: {counts}."
        )
    runtime.assert_all_ranks_equal(
        schedule,
        label="integration two-task schedule",
    )
    return schedule


def _component_modules(model: Any) -> dict[str, nn.Module]:
    if not model.seg_enable or model.seg_module is None or model.seg_projector is None:
        raise IntegrationError(
            "Integration requires segmentation-enabled M3D with both image encoders."
        )
    return {
        "main_vision": model.vision_tower,
        "multimodal_projector": model.mm_projector,
        "language_model": model.language_model,
        "segmentation_projector": model.seg_projector,
        "segvol_vision": model.seg_module.image_encoder,
        "segvol_prompt_encoder": model.seg_module.prompt_encoder,
        "segvol_mask_decoder": model.seg_module.mask_decoder,
    }


def _validate_gradient_contract(
    *,
    task: str,
    summaries: Mapping[str, GradientSummary],
) -> None:
    main_required = (
        "main_vision",
        "multimodal_projector",
        "language_model",
    )
    segmentation_required = (
        "segmentation_projector",
        "segvol_vision",
        "segvol_prompt_encoder",
        "segvol_mask_decoder",
    )

    for name, summary in summaries.items():
        if summary.has_gradient and not summary.all_gradients_finite:
            raise GradientContractError(
                f"Component {name!r} contains non-finite gradients on task {task!r}: "
                f"{summary.to_dict()}."
            )

    for name in main_required:
        summary = summaries[name]
        if not summary.has_nonzero_gradient:
            raise GradientContractError(
                f"Task {task!r} did not produce a non-zero gradient in required "
                f"main-path component {name!r}: {summary.to_dict()}."
            )

    if task == "caption":
        unexpected = {
            name: summaries[name].to_dict()
            for name in segmentation_required
            if summaries[name].has_gradient
        }
        if unexpected:
            raise GradientContractError(
                "Caption microbatch touched the SegVol-only branch: "
                f"{unexpected}."
            )
    elif task == "segmentation":
        missing = {
            name: summaries[name].to_dict()
            for name in segmentation_required
            if not summaries[name].has_nonzero_gradient
        }
        if missing:
            raise GradientContractError(
                "Segmentation microbatch failed to train one or more SegVol "
                f"components: {missing}."
            )
    else:
        raise GradientContractError(f"Unexpected integration task {task!r}.")


def _validate_model_independence(model: Any) -> None:
    summary = model.parameter_summary()
    if summary.shared_image_encoder_parameters != 0:
        raise IntegrationError(
            "Main ViT and SegVol ViT share Parameter objects: "
            f"{summary.shared_image_encoder_parameters}."
        )
    if summary.shared_image_encoder_storages != 0:
        raise IntegrationError(
            "Main ViT and SegVol ViT share parameter storage: "
            f"{summary.shared_image_encoder_storages}."
        )


# ---------------------------------------------------------------------------
# Two real distributed optimizer updates
# ---------------------------------------------------------------------------


def _microbatch_metrics(output: Any) -> tuple[float, float | None, float | None]:
    if output.loss_output is None:
        raise IntegrationError("Training forward returned no loss_output.")
    language = _safe_scalar(output.loss_output.language.detach())
    if output.loss_output.segmentation is None:
        return language, None, None
    return (
        language,
        _safe_scalar(output.loss_output.segmentation.dice.detach()),
        _safe_scalar(output.loss_output.segmentation.bce.detach()),
    )


def _run_two_microbatches(
    *,
    config: Any,
    runtime: Any,
    data_pipeline: Any,
    distributed_model: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
) -> tuple[MicrobatchReport, ...]:
    from m3d.data.sampler import sampler_position_from_batch
    from m3d.trainer import _compose_backward_loss

    model = distributed_model.unwrapped_model
    modules = _component_modules(model)
    reports: list[MicrobatchReport] = []
    observed_tasks: list[str] = []

    distributed_model.train()
    optimizer.zero_grad(set_to_none=True)
    data_pipeline.set_epoch(0, committed_step=0)

    for cpu_batch in data_pipeline.loader:
        epoch, step, sampler_task = sampler_position_from_batch(cpu_batch)
        if epoch != 0:
            raise IntegrationProgressError(
                f"Integration DataLoader yielded epoch {epoch}, expected 0."
            )
        if step != data_pipeline.committed_step:
            raise IntegrationProgressError(
                "DataLoader cursor mismatch: "
                f"step={step}, committed={data_pipeline.committed_step}."
            )
        if sampler_task is not cpu_batch.task:
            raise IntegrationProgressError(
                "Sampler task and batch task differ: "
                f"{sampler_task.value} != {cpu_batch.task.value}."
            )
        runtime.assert_all_ranks_equal(
            cpu_batch.task.value,
            label=f"integration task at step {step}",
        )

        gpu_batch = cpu_batch.to(
            runtime.device,
            non_blocking=data_pipeline.non_blocking_transfer,
        )
        torch.cuda.reset_peak_memory_stats(runtime.device)
        torch.cuda.synchronize(runtime.device)
        started = time.perf_counter()

        with distributed_model.gradient_sync(enabled=True):
            with runtime.autocast():
                output = distributed_model.forward_batch(
                    gpu_batch,
                    logits_mode="none",
                    return_intermediates=False,
                )
                if output.loss_output is None:
                    raise IntegrationError(
                        f"Task {gpu_batch.task.value!r} produced no loss output."
                    )
                backward_loss = _compose_backward_loss(
                    runtime=runtime,
                    batch=gpu_batch,
                    loss_output=output.loss_output,
                )
            backward_loss.backward()

        summaries = {
            name: _gradient_summary(module)
            for name, module in modules.items()
        }
        _validate_gradient_contract(
            task=gpu_batch.task.value,
            summaries=summaries,
        )

        gradient_norm = distributed_model.clip_grad_norm_(
            config.optimization.max_grad_norm
        )
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        data_pipeline.commit_batch(cpu_batch)

        torch.cuda.synchronize(runtime.device)
        elapsed = time.perf_counter() - started
        language_loss, dice_loss, bce_loss = _microbatch_metrics(output)

        target_shape = (
            None
            if gpu_batch.segmentation_targets is None
            else tuple(int(item) for item in gpu_batch.segmentation_targets.shape)
        )
        logit_shape = (
            None
            if output.segmentation_logits is None
            else tuple(int(item) for item in output.segmentation_logits.shape)
        )
        if gpu_batch.task.requires_segmentation_target:
            if output.segmentation_logits is None:
                raise IntegrationError(
                    "Segmentation task returned no segmentation logits."
                )
            if tuple(output.segmentation_logits.shape) != tuple(
                gpu_batch.segmentation_targets.shape
            ):
                raise IntegrationError(
                    "Segmentation logits/target shape mismatch: "
                    f"logits={tuple(output.segmentation_logits.shape)}, "
                    f"target={tuple(gpu_batch.segmentation_targets.shape)}."
                )
        elif output.segmentation_logits is not None:
            raise IntegrationError(
                "Caption task unexpectedly returned segmentation logits."
            )

        foreground = (
            None
            if gpu_batch.segmentation_targets is None
            else int(gpu_batch.segmentation_targets.detach().sum().cpu().item())
        )
        report = MicrobatchReport(
            step=int(step),
            task=gpu_batch.task.value,
            sample_ids=tuple(gpu_batch.sample_ids),
            sequence_length=int(gpu_batch.text.sequence_length),
            unpadded_lengths=tuple(
                int(item)
                for item in gpu_batch.text.unpadded_lengths.detach().cpu().tolist()
            ),
            image_shape=tuple(int(item) for item in gpu_batch.images.shape),
            segmentation_target_shape=target_shape,
            segmentation_logit_shape=logit_shape,
            foreground_voxel_count=foreground,
            language_token_count=int(
                output.loss_output.language_token_count.detach().cpu().item()
            ),
            local_total_loss=_safe_scalar(output.loss_output.total.detach()),
            global_backward_loss=_safe_scalar(backward_loss.detach()),
            language_loss=language_loss,
            dice_loss=dice_loss,
            bce_loss=bce_loss,
            gradient_norm=_safe_scalar(gradient_norm),
            learning_rates=_learning_rates_by_role(optimizer),
            gradients=summaries,
            elapsed_seconds=elapsed,
            cuda_memory=runtime.cuda_memory_snapshot(),
        )
        reports.append(report)
        observed_tasks.append(gpu_batch.task.value)

        del gpu_batch, output, backward_loss, gradient_norm
        torch.cuda.empty_cache()

        if len(reports) == 2:
            break

    if len(reports) != 2:
        raise IntegrationProgressError(
            f"Expected two integration microbatches, consumed {len(reports)}."
        )
    if sorted(observed_tasks) != sorted(_REQUIRED_TASKS):
        raise IntegrationProgressError(
            f"Observed tasks {observed_tasks}, expected {_REQUIRED_TASKS}."
        )
    if data_pipeline.committed_step != 2:
        raise IntegrationProgressError(
            "Data pipeline did not commit both microbatches: "
            f"{data_pipeline.committed_step}."
        )
    if scheduler.completed_optimizer_steps != 2 or not scheduler.is_finished:
        raise IntegrationProgressError(
            "Scheduler did not finish exactly two optimizer updates: "
            f"completed={scheduler.completed_optimizer_steps}, "
            f"total={scheduler.total_optimizer_steps}."
        )
    runtime.assert_all_ranks_equal(
        tuple(observed_tasks),
        label="observed integration task order",
    )
    return tuple(reports)


# ---------------------------------------------------------------------------
# Main distributed integration entry point
# ---------------------------------------------------------------------------


def run_integration(options: IntegrationOptions) -> IntegrationReport | None:
    from m3d.config import config_fingerprint
    from m3d.data.loader import build_training_data_pipeline
    from m3d.distributed import build_model_synchronously, prepare_distributed_model
    from m3d.model.m3d import build_m3d_model
    from m3d.optim import build_optimizer
    from m3d.runtime import distributed_runtime
    from m3d.scheduler import build_scheduler
    from m3d.tokenization import M3DTextProcessor, build_tokenizer

    config = _build_integration_config(options)
    started_at = time.time()

    with distributed_runtime(
        config,
        verbose_all_ranks=options.verbose_all_ranks,
    ) as runtime:
        runtime.assert_all_ranks_equal(
            _json_sha256(options.consistency_payload()),
            label="integration CLI options",
        )
        runtime.assert_all_ranks_equal(
            _json_sha256(config_fingerprint(config)),
            label="resolved integration config",
        )

        output_dir = Path(options.output_dir).expanduser().resolve()
        if runtime.is_main_process:
            output_dir.mkdir(parents=True, exist_ok=True)
            _atomic_write_json(output_dir / "resolved_integration_config.json", config.to_dict())
        runtime.barrier()

        environment = _environment_report(runtime, options)
        environments = tuple(
            EnvironmentReport(**item)
            for item in runtime.all_gather_object(dataclasses.asdict(environment))
        )
        flash_probe = _run_flash_probe(runtime)
        flash_probes = tuple(
            FlashProbeReport(
                qkv_shape=tuple(item["qkv_shape"]),
                output_shape=tuple(item["output_shape"]),
                output_dtype=item["output_dtype"],
                loss=float(item["loss"]),
                q_gradient_finite=bool(item["q_gradient_finite"]),
                k_gradient_finite=bool(item["k_gradient_finite"]),
                v_gradient_finite=bool(item["v_gradient_finite"]),
                elapsed_seconds=float(item["elapsed_seconds"]),
            )
            for item in runtime.all_gather_object(dataclasses.asdict(flash_probe))
        )

        # Rank 0 populates the shared Hugging Face cache first; all ranks then
        # independently construct identical tokenizer objects.
        with runtime.main_process_first():
            tokenizer_bundle = build_tokenizer(
                config,
                cache_dir=options.cache_dir,
                local_files_only=options.local_files_only,
            )
        runtime.assert_all_ranks_equal(
            _json_sha256(tokenizer_bundle.metadata.to_dict()),
            label="integration tokenizer metadata",
        )
        text_processor = M3DTextProcessor(tokenizer_bundle, config)

        data_pipeline = build_training_data_pipeline(
            config=config,
            runtime=runtime,
            tokenizer_bundle=tokenizer_bundle,
            text_processor=text_processor,
        )
        schedule = _validate_schedule(data_pipeline, runtime)

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
        _validate_model_independence(model)

        distributed_model, distributed_report = prepare_distributed_model(
            model,
            runtime,
        )
        optimizer, optimizer_report = build_optimizer(
            distributed_model.unwrapped_model,
            config,
            distributed_strategy=distributed_model.strategy,
            allow_unfused_fallback=(options.strategy == "fsdp2"),
        )
        if options.strategy == "ddp" and not optimizer_report.fused_enabled:
            raise EnvironmentContractError(
                "DDP integration requires fused AdamW, but it was not enabled: "
                f"{optimizer_report.fused_fallback_reason!r}."
            )
        scheduler, scheduler_report = build_scheduler(
            optimizer,
            config,
            steps_per_epoch=data_pipeline.steps_per_epoch,
        )
        if scheduler.total_optimizer_steps != 2:
            raise IntegrationProgressError(
                "Integration scheduler must contain exactly two optimizer steps, "
                f"got {scheduler.total_optimizer_steps}."
            )

        local_microbatches = _run_two_microbatches(
            config=config,
            runtime=runtime,
            data_pipeline=data_pipeline,
            distributed_model=distributed_model,
            optimizer=optimizer,
            scheduler=scheduler,
        )
        gathered_microbatches_raw = runtime.all_gather_object(
            tuple(item.to_dict() for item in local_microbatches)
        )

        # Verify all ranks agree on task order and tensor contracts while still
        # allowing sample IDs and local GPU memory values to differ.
        rank_signatures = []
        for rank_items in gathered_microbatches_raw:
            rank_signatures.append(
                tuple(
                    (
                        item["step"],
                        item["task"],
                        tuple(item["image_shape"]),
                        item["segmentation_target_shape"],
                        item["segmentation_logit_shape"],
                    )
                    for item in rank_items
                )
            )
        if any(signature != rank_signatures[0] for signature in rank_signatures[1:]):
            raise IntegrationProgressError(
                f"Ranks disagree on integration tensor/task contracts: {rank_signatures}."
            )

        finished_at = time.time()
        if not runtime.is_main_process:
            runtime.barrier()
            return None

        microbatches_by_rank: list[tuple[MicrobatchReport, ...]] = []
        for rank_items in gathered_microbatches_raw:
            parsed_items: list[MicrobatchReport] = []
            for item in rank_items:
                parsed_items.append(
                    MicrobatchReport(
                        step=int(item["step"]),
                        task=str(item["task"]),
                        sample_ids=tuple(item["sample_ids"]),
                        sequence_length=int(item["sequence_length"]),
                        unpadded_lengths=tuple(item["unpadded_lengths"]),
                        image_shape=tuple(item["image_shape"]),
                        segmentation_target_shape=(
                            None
                            if item["segmentation_target_shape"] is None
                            else tuple(item["segmentation_target_shape"])
                        ),
                        segmentation_logit_shape=(
                            None
                            if item["segmentation_logit_shape"] is None
                            else tuple(item["segmentation_logit_shape"])
                        ),
                        foreground_voxel_count=item["foreground_voxel_count"],
                        language_token_count=int(item["language_token_count"]),
                        local_total_loss=float(item["local_total_loss"]),
                        global_backward_loss=float(item["global_backward_loss"]),
                        language_loss=float(item["language_loss"]),
                        dice_loss=(
                            None
                            if item["dice_loss"] is None
                            else float(item["dice_loss"])
                        ),
                        bce_loss=(
                            None
                            if item["bce_loss"] is None
                            else float(item["bce_loss"])
                        ),
                        gradient_norm=float(item["gradient_norm"]),
                        learning_rates=dict(item["learning_rates"]),
                        gradients={
                            name: GradientSummary(
                                trainable_parameter_tensors=int(value["trainable_parameter_tensors"]),
                                gradient_parameter_tensors=int(value["gradient_parameter_tensors"]),
                                finite_gradient_parameter_tensors=int(value["finite_gradient_parameter_tensors"]),
                                nonzero_gradient_parameter_tensors=int(value["nonzero_gradient_parameter_tensors"]),
                                local_gradient_elements=int(value["local_gradient_elements"]),
                                local_nonzero_gradient_elements=int(value["local_nonzero_gradient_elements"]),
                                local_max_abs_gradient=float(value["local_max_abs_gradient"]),
                            )
                            for name, value in item["gradients"].items()
                        },
                        elapsed_seconds=float(item["elapsed_seconds"]),
                        cuda_memory=dict(item["cuda_memory"]),
                    )
                )
            microbatches_by_rank.append(tuple(parsed_items))

        report = IntegrationReport(
            state_version=_INTEGRATION_STATE_VERSION,
            status="passed",
            strategy=options.strategy,
            config_path=str(Path(options.config_path).expanduser().resolve()),
            output_dir=str(output_dir),
            started_at_unix=started_at,
            finished_at_unix=finished_at,
            elapsed_seconds=finished_at - started_at,
            expected_world_size=options.expected_world_size,
            task_schedule=schedule,
            environment_by_rank=environments,
            flash_probe_by_rank=flash_probes,
            tokenizer=tokenizer_bundle.metadata.to_dict(),
            data_pipeline=dict(data_pipeline.summary()),
            model=model_report.to_dict(),
            distributed=distributed_report.to_dict(),
            optimizer=optimizer_report.to_dict(include_parameter_names=False),
            scheduler=scheduler_report.to_dict(),
            microbatches_by_rank=tuple(microbatches_by_rank),
            completed_optimizer_steps=int(scheduler.completed_optimizer_steps),
            committed_microbatches=int(data_pipeline.committed_step),
        )
        _atomic_write_json(output_dir / "integration_report.json", report.to_dict())
        runtime.logger.info(
            "ASPIRE 2A integration passed: strategy=%s tasks=%s elapsed=%.2fs report=%s",
            options.strategy,
            schedule,
            report.elapsed_seconds,
            output_dir / "integration_report.json",
        )
        runtime.barrier()
        return report


# ---------------------------------------------------------------------------
# CLI and dependency-light self-test
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run two real distributed M3D optimizer updates on ASPIRE 2A: "
            "one caption batch and one segmentation batch."
        )
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "m3d_joint_finetune.yaml"),
        help="Path to the resolved-source YAML configuration.",
    )
    parser.add_argument(
        "--strategy",
        choices=("ddp", "fsdp2"),
        default="ddp",
        help="Distributed wrapper to validate.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Report directory. Defaults to outputs/aspire2a-integration-<strategy>."
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
        help="Disallow network access while loading tokenizer/Phi-3.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional dotted.path=value YAML override applied before test settings.",
    )
    parser.add_argument(
        "--expected-world-size",
        type=int,
        default=2,
        help="Expected torchrun world size. ASPIRE 2A default is 2 GPUs.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader workers per rank for the two-batch integration run.",
    )
    parser.add_argument(
        "--allow-non-a100",
        action="store_true",
        help="Diagnostic escape hatch; production validation requires A100.",
    )
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="Report but do not fail on pinned Python/package/CUDA mismatches.",
    )
    parser.add_argument(
        "--skip-path-verification",
        action="store_true",
        help="Skip config path checks; model/data construction may still fail later.",
    )
    parser.add_argument(
        "--verbose-all-ranks",
        action="store_true",
        help="Emit runtime logs from every rank instead of rank 0 only.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run dependency-light local helper tests and exit.",
    )
    return parser


def _parse_options(argv: Sequence[str] | None = None) -> tuple[IntegrationOptions | None, bool]:
    args = _build_parser().parse_args(argv)
    if args.self_test:
        return None, True
    if args.expected_world_size <= 1:
        raise ValueError("--expected-world-size must be greater than one.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else (PROJECT_ROOT / "outputs" / f"aspire2a-integration-{args.strategy}").resolve()
    )
    options = IntegrationOptions(
        config_path=str(Path(args.config).expanduser().resolve()),
        strategy=str(args.strategy),
        output_dir=str(output_dir),
        cache_dir=(
            None
            if args.cache_dir is None
            else str(Path(args.cache_dir).expanduser().resolve())
        ),
        local_files_only=bool(args.local_files_only),
        overrides=tuple(str(item) for item in args.override),
        expected_world_size=int(args.expected_world_size),
        num_workers=int(args.num_workers),
        allow_non_a100=bool(args.allow_non_a100),
        allow_version_mismatch=bool(args.allow_version_mismatch),
        verify_paths=not bool(args.skip_path_verification),
        verbose_all_ranks=bool(args.verbose_all_ranks),
    )
    return options, False


def _run_self_test() -> dict[str, Any]:
    assert _normalise_version("2.6.0+cu118") == "2.6.0"
    assert _normalise_version("4.52.4") == "4.52.4"
    weights = _task_weights()
    assert set(weights) == {
        "caption",
        "vqa_closed",
        "vqa_open",
        "vqa_yes_no",
        "positioning",
        "segmentation",
    }
    assert [name for name, value in weights.items() if value > 0] == [
        "caption",
        "segmentation",
    ]

    toy = nn.Sequential(nn.Linear(4, 8), nn.GELU(), nn.Linear(8, 1))
    input_tensor = torch.randn(3, 4)
    toy(input_tensor).sum().backward()
    summary = _gradient_summary(toy)
    assert summary.trainable_parameter_tensors == 4
    assert summary.gradient_parameter_tensors == 4
    assert summary.all_gradients_finite
    assert summary.has_nonzero_gradient

    empty = nn.Linear(2, 2)
    empty_summary = _gradient_summary(empty)
    assert not empty_summary.has_gradient

    parser_options, self_test = _parse_options(
        [
            "--strategy",
            "fsdp2",
            "--expected-world-size",
            "4",
            "--num-workers",
            "2",
        ]
    )
    assert not self_test and parser_options is not None
    assert parser_options.strategy == "fsdp2"
    assert parser_options.expected_world_size == 4
    assert parser_options.num_workers == 2

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "report.json"
        payload = {"status": "passed", "weights": weights}
        _atomic_write_json(path, payload)
        loaded = json.loads(path.read_text(encoding="utf-8"))
        assert loaded == payload
        assert not list(Path(directory).glob("*.tmp-*"))

    return {
        "status": "passed",
        "state_version": _INTEGRATION_STATE_VERSION,
        "version_normalisation": True,
        "task_weights": weights,
        "gradient_summary": summary.to_dict(),
        "empty_gradient_detected": True,
        "cli_parse": True,
        "atomic_json_roundtrip": True,
    }


def main(argv: Sequence[str] | None = None) -> int:
    options, self_test = _parse_options(argv)
    if self_test:
        print(json.dumps(_run_self_test(), indent=2, sort_keys=True))
        return 0
    assert options is not None

    try:
        report = run_integration(options)
    except BaseException as exc:
        # Every rank logs its own failure to stderr. The distributed runtime's
        # finally block performs best-effort process-group cleanup.
        payload = {
            "status": "failed",
            "rank": int(os.environ.get("RANK", "0")),
            "local_rank": int(os.environ.get("LOCAL_RANK", "0")),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        return 1

    if report is not None:
        print(
            json.dumps(
                {
                    "status": report.status,
                    "strategy": report.strategy,
                    "task_schedule": list(report.task_schedule),
                    "completed_optimizer_steps": report.completed_optimizer_steps,
                    "committed_microbatches": report.committed_microbatches,
                    "elapsed_seconds": report.elapsed_seconds,
                    "report": str(Path(report.output_dir) / "integration_report.json"),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
