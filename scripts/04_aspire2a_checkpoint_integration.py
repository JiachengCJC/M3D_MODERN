#!/usr/bin/env python3
"""Exact distributed-checkpoint integration test for M3D on ASPIRE 2A.

This script validates the part that the compute-only integration test
(``03_aspire2a_integration.py``) intentionally does not exercise: durable
PyTorch Distributed Checkpoint save/load and exact continuation.

The real test sequence on every rank is:

1. build the real tokenizer, task-homogeneous DataLoader, M3D model, both
   independent 3D image encoders, LoRA, DDP/FSDP2, optimizer and scheduler;
2. execute the first real optimizer update;
3. save a complete distributed checkpoint at the valid accumulation boundary;
4. record model/optimizer sentinels and the next Python/NumPy/CPU/CUDA RNG draws;
5. fetch and execute the second real task batch, deliberately changing model,
   optimizer, scheduler and sampler state;
6. load the step-1 checkpoint into those already wrapped objects;
7. prove that model, optimizer, scheduler, sampler and rank-local RNG return to
   the saved state;
8. fetch the next batch again and prove its sample IDs, prompt tokens, image
   augmentation and segmentation target exactly match the uninterrupted batch;
9. replay the second update and compare the resulting sampled model and
   optimizer states with the uninterrupted result.

The two-step integration schedule contains exactly one caption batch and one
segmentation batch.  Consequently, resume crosses from one execution graph to
another regardless of which task the deterministic sampler places first.

Recommended ASPIRE 2A launch::

    export PYTHONPATH="$(pwd)/src:${PYTHONPATH:-}"

    torchrun \
      --standalone \
      --nproc_per_node=2 \
      scripts/04_aspire2a_checkpoint_integration.py \
      --config configs/m3d_joint_finetune.yaml \
      --strategy ddp \
      --checkpoint-mode async

FSDP2 validation::

    torchrun \
      --standalone \
      --nproc_per_node=2 \
      scripts/04_aspire2a_checkpoint_integration.py \
      --config configs/m3d_joint_finetune.yaml \
      --strategy fsdp2 \
      --checkpoint-mode async

A successful machine-readable report is written to::

    <output-dir>/checkpoint_integration_report.json

The large temporary checkpoint is deleted after a successful test unless
``--keep-checkpoint`` is supplied.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import shutil
import struct
import sys
import tempfile
import time
import traceback
from types import ModuleType
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
COMPUTE_INTEGRATION_PATH = (
    PROJECT_ROOT / "scripts" / "03_aspire2a_integration.py"
)
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import numpy as np
import torch
from torch import Tensor, nn


_CHECKPOINT_INTEGRATION_STATE_VERSION = 1
_DEFAULT_SAMPLE_COUNT_PER_TENSOR = 8
_MODEL_REPLAY_ATOL = 2.0e-5
_MODEL_REPLAY_RTOL = 2.0e-4
_OPTIMIZER_REPLAY_ATOL = 2.0e-6
_OPTIMIZER_REPLAY_RTOL = 2.0e-5


class CheckpointIntegrationError(RuntimeError):
    """Base error for an exact-resume integration failure."""


class StateRoundTripError(CheckpointIntegrationError):
    """Raised when model/optimizer/scheduler state does not restore."""


class DataReplayError(CheckpointIntegrationError):
    """Raised when the next batch after resume differs from uninterrupted data."""


class RNGReplayError(CheckpointIntegrationError):
    """Raised when rank-local random streams do not resume exactly."""


@dataclasses.dataclass(frozen=True, slots=True)
class CheckpointIntegrationOptions:
    """Resolved command-line settings shared by all ranks."""

    config_path: str
    strategy: str
    checkpoint_mode: str
    output_dir: str
    cache_dir: str | None
    local_files_only: bool
    overrides: tuple[str, ...]
    expected_world_size: int
    num_workers: int
    allow_non_a100: bool
    allow_version_mismatch: bool
    allow_async_fallback: bool
    verify_paths: bool
    verbose_all_ranks: bool
    overwrite_output: bool
    keep_checkpoint: bool
    samples_per_tensor: int

    def consistency_payload(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class SampledTensorEntry:
    """Small deterministic sample from one tensor or local DTensor shard."""

    key: str
    dtype: str
    global_shape: tuple[int, ...]
    local_shape: tuple[int, ...]
    indices: tuple[int, ...]
    values: tuple[float, ...]
    raw_sha256: str

    def metadata_payload(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "dtype": self.dtype,
            "global_shape": list(self.global_shape),
            "local_shape": list(self.local_shape),
            "indices": list(self.indices),
            "values": list(self.values),
            "raw_sha256": self.raw_sha256,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class SampledState:
    """Rank-local compact fingerprint used without gathering full 4B weights."""

    kind: str
    digest: str
    tensor_count: int
    sampled_element_count: int
    entries: Mapping[str, SampledTensorEntry]

    def summary(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "digest": self.digest,
            "tensor_count": self.tensor_count,
            "sampled_element_count": self.sampled_element_count,
        }


@dataclasses.dataclass(frozen=True, slots=True)
class StateDistance:
    """Numerical difference between two sampled states with identical layout."""

    compared_entries: int
    compared_values: int
    max_absolute_difference: float
    max_relative_difference: float
    allclose: bool
    first_mismatch: str | None

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True, slots=True)
class RNGProbe:
    """Values drawn from every rank-local random stream saved by checkpointing."""

    python_values: tuple[float, ...]
    numpy_values: tuple[float, ...]
    torch_cpu_values: tuple[float, ...]
    torch_cuda_values: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "python_values": list(self.python_values),
            "numpy_values": list(self.numpy_values),
            "torch_cpu_values": list(self.torch_cpu_values),
            "torch_cuda_values": list(self.torch_cuda_values),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class BatchFingerprint:
    """Exact CPU-batch digest, including augmentation and prompt tokens."""

    digest: str
    task: str
    sample_ids: tuple[str, ...]
    image_shape: tuple[int, ...]
    sequence_length: int
    segmentation_target_shape: tuple[int, ...] | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "digest": self.digest,
            "task": self.task,
            "sample_ids": list(self.sample_ids),
            "image_shape": list(self.image_shape),
            "sequence_length": self.sequence_length,
            "segmentation_target_shape": (
                None
                if self.segmentation_target_shape is None
                else list(self.segmentation_target_shape)
            ),
        }


@dataclasses.dataclass(frozen=True, slots=True)
class OptimizerUpdateReport:
    """One real optimizer update before save, perturbation or replay."""

    phase: str
    step_before: int
    step_after: int
    task: str
    sample_ids: tuple[str, ...]
    batch_digest: str
    local_total_loss: float
    global_backward_loss: float
    gradient_norm: float
    elapsed_seconds: float
    cuda_memory: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["sample_ids"] = list(self.sample_ids)
        payload["cuda_memory"] = dict(self.cuda_memory)
        return payload


@dataclasses.dataclass(frozen=True, slots=True)
class CheckpointIntegrationReport:
    """Rank-0 aggregate proving exact DCP continuation."""

    state_version: int
    status: str
    strategy: str
    checkpoint_mode_requested: str
    checkpoint_mode_used: str
    config_path: str
    output_dir: str
    checkpoint_path: str
    checkpoint_retained: bool
    elapsed_seconds: float
    task_schedule: tuple[str, ...]
    environment_by_rank: tuple[Mapping[str, Any], ...]
    flash_probe_by_rank: tuple[Mapping[str, Any], ...]
    tokenizer: Mapping[str, Any]
    data_pipeline: Mapping[str, Any]
    model: Mapping[str, Any]
    distributed: Mapping[str, Any]
    optimizer: Mapping[str, Any]
    scheduler: Mapping[str, Any]
    save_report: Mapping[str, Any]
    resume_report: Mapping[str, Any]
    first_update_by_rank: tuple[Mapping[str, Any], ...]
    uninterrupted_update_by_rank: tuple[Mapping[str, Any], ...]
    replayed_update_by_rank: tuple[Mapping[str, Any], ...]
    checkpoint_model_state_by_rank: tuple[Mapping[str, Any], ...]
    restored_model_state_by_rank: tuple[Mapping[str, Any], ...]
    checkpoint_optimizer_state_by_rank: tuple[Mapping[str, Any], ...]
    restored_optimizer_state_by_rank: tuple[Mapping[str, Any], ...]
    next_batch_by_rank: tuple[Mapping[str, Any], ...]
    replayed_batch_by_rank: tuple[Mapping[str, Any], ...]
    rng_exact_by_rank: tuple[bool, ...]
    model_restore_exact_by_rank: tuple[bool, ...]
    optimizer_restore_exact_by_rank: tuple[bool, ...]
    scheduler_restore_exact_by_rank: tuple[bool, ...]
    data_cursor_restore_exact_by_rank: tuple[bool, ...]
    replay_model_distance_by_rank: tuple[Mapping[str, Any], ...]
    replay_optimizer_distance_by_rank: tuple[Mapping[str, Any], ...]
    completed_optimizer_steps: int
    committed_microbatches: int

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["task_schedule"] = list(self.task_schedule)
        for key in (
            "environment_by_rank",
            "flash_probe_by_rank",
            "first_update_by_rank",
            "uninterrupted_update_by_rank",
            "replayed_update_by_rank",
            "checkpoint_model_state_by_rank",
            "restored_model_state_by_rank",
            "checkpoint_optimizer_state_by_rank",
            "restored_optimizer_state_by_rank",
            "next_batch_by_rank",
            "replayed_batch_by_rank",
            "replay_model_distance_by_rank",
            "replay_optimizer_distance_by_rank",
        ):
            payload[key] = [dict(item) for item in getattr(self, key)]
        for key in (
            "rng_exact_by_rank",
            "model_restore_exact_by_rank",
            "optimizer_restore_exact_by_rank",
            "scheduler_restore_exact_by_rank",
            "data_cursor_restore_exact_by_rank",
        ):
            payload[key] = list(getattr(self, key))
        return payload


# ---------------------------------------------------------------------------
# Shared compute-integration helpers
# ---------------------------------------------------------------------------


def _load_compute_support() -> ModuleType:
    """Load script 03 under a normal module name.

    The checkpoint test intentionally reuses the exact ASPIRE 2A environment,
    Flash-SDPA and branch-gradient checks already reviewed in the preceding
    file.  Registering the module before execution is required by dataclasses.
    """

    if not COMPUTE_INTEGRATION_PATH.is_file():
        raise FileNotFoundError(
            f"Missing prerequisite integration script: {COMPUTE_INTEGRATION_PATH}"
        )
    module_name = "m3d_aspire2a_compute_integration_support"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(
        module_name,
        COMPUTE_INTEGRATION_PATH,
    )
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Could not create import specification for {COMPUTE_INTEGRATION_PATH}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# Generic hashing and distributed assertions
# ---------------------------------------------------------------------------


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    )


def _json_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, path)


def _distributed_assert(
    runtime: Any,
    condition: bool,
    *,
    label: str,
    detail: str,
) -> None:
    """Make every rank fail together instead of deadlocking later."""

    local_error = None if condition else f"rank={runtime.rank}: {detail}"
    gathered = runtime.all_gather_object(local_error)
    failures = [item for item in gathered if item is not None]
    if failures:
        raise CheckpointIntegrationError(
            f"Distributed assertion failed for {label!r}: {failures}"
        )


def _tensor_local_shard(tensor: Tensor) -> Tensor:
    to_local = getattr(tensor, "to_local", None)
    if callable(to_local):
        result = to_local()
        if not isinstance(result, Tensor):
            raise TypeError(
                f"DTensor.to_local() returned {type(result).__name__}, not Tensor."
            )
        return result
    return tensor


def _safe_scalar(value: Tensor | float | int) -> float:
    if isinstance(value, Tensor):
        local = _tensor_local_shard(value.detach())
        if local.numel() != 1:
            raise ValueError(f"Expected scalar, got local shape {tuple(local.shape)}.")
        return float(local.float().cpu().item())
    return float(value)


def _sample_indices(key: str, numel: int, count: int) -> tuple[int, ...]:
    if numel <= 0 or count <= 0:
        return ()
    desired = min(numel, count)
    indices: set[int] = {0, numel - 1, numel // 2}
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=32).digest()
    cursor = 0
    while len(indices) < desired:
        if cursor + 8 > len(digest):
            digest = hashlib.blake2b(digest, digest_size=32).digest()
            cursor = 0
        candidate = int.from_bytes(digest[cursor : cursor + 8], "little") % numel
        indices.add(candidate)
        cursor += 8
    return tuple(sorted(indices)[:desired])


def _sample_tensor(
    key: str,
    tensor: Tensor,
    *,
    global_shape: Iterable[int] | None = None,
    sample_count: int,
) -> SampledTensorEntry:
    local = _tensor_local_shard(tensor.detach())
    if local.layout != torch.strided:
        raise TypeError(f"State tensor {key!r} is not strided: {local.layout}.")
    if not local.is_contiguous():
        # Parameters and AdamW states should be contiguous.  Rejecting avoids a
        # hidden full-size copy merely to build a tiny checkpoint sentinel.
        raise StateRoundTripError(
            f"State tensor {key!r} is unexpectedly non-contiguous."
        )
    flat = local.view(-1)
    indices = _sample_indices(key, int(flat.numel()), sample_count)
    if indices:
        index_tensor = torch.tensor(indices, dtype=torch.long, device=flat.device)
        selected = flat.index_select(0, index_tensor).contiguous()
        values = tuple(float(item) for item in selected.float().cpu().tolist())
        raw = selected.view(torch.uint8).cpu().numpy().tobytes()
    else:
        values = ()
        raw = b""
    return SampledTensorEntry(
        key=key,
        dtype=str(tensor.dtype),
        global_shape=tuple(
            int(item) for item in (tensor.shape if global_shape is None else global_shape)
        ),
        local_shape=tuple(int(item) for item in local.shape),
        indices=indices,
        values=values,
        raw_sha256=hashlib.sha256(raw).hexdigest(),
    )


def _finalise_sampled_state(
    kind: str,
    entries: MutableMapping[str, SampledTensorEntry],
) -> SampledState:
    ordered = {key: entries[key] for key in sorted(entries)}
    digest = _json_sha256(
        [entry.metadata_payload() for entry in ordered.values()]
    )
    return SampledState(
        kind=kind,
        digest=digest,
        tensor_count=len(ordered),
        sampled_element_count=sum(len(item.values) for item in ordered.values()),
        entries=ordered,
    )


def _snapshot_model(model: nn.Module, *, sample_count: int) -> SampledState:
    entries: dict[str, SampledTensorEntry] = {}
    for name, parameter in sorted(model.named_parameters(), key=lambda item: item[0]):
        entries[f"parameter:{name}"] = _sample_tensor(
            f"parameter:{name}",
            parameter,
            sample_count=sample_count,
        )
    return _finalise_sampled_state("model_parameters", entries)


def _snapshot_optimizer(
    optimizer: torch.optim.Optimizer,
    *,
    sample_count: int,
) -> SampledState:
    entries: dict[str, SampledTensorEntry] = {}
    for group_index, group in enumerate(optimizer.param_groups):
        group_name = group.get("group_name", f"group-{group_index}")
        param_names = group.get("param_names")
        if not isinstance(param_names, list) or len(param_names) != len(group["params"]):
            raise StateRoundTripError(
                f"Optimizer group {group_name!r} has invalid param_names metadata."
            )
        for parameter, parameter_name in zip(group["params"], param_names):
            state = optimizer.state.get(parameter, {})
            for state_name, value in sorted(state.items(), key=lambda item: str(item[0])):
                key = f"{group_name}:{parameter_name}:{state_name}"
                if isinstance(value, Tensor):
                    entries[key] = _sample_tensor(
                        key,
                        value,
                        sample_count=sample_count,
                    )
                elif isinstance(value, (bool, int, float)):
                    scalar = torch.tensor(value, dtype=torch.float64)
                    entries[key] = _sample_tensor(
                        key,
                        scalar,
                        sample_count=1,
                    )
                else:
                    raise StateRoundTripError(
                        "Unsupported optimizer state value for checkpoint sentinel: "
                        f"key={key!r}, type={type(value).__name__}."
                    )
    return _finalise_sampled_state("optimizer_state", entries)


def _state_distance(
    first: SampledState,
    second: SampledState,
    *,
    atol: float,
    rtol: float,
) -> StateDistance:
    if set(first.entries) != set(second.entries):
        missing = sorted(set(first.entries) - set(second.entries))[:8]
        extra = sorted(set(second.entries) - set(first.entries))[:8]
        return StateDistance(
            compared_entries=0,
            compared_values=0,
            max_absolute_difference=math.inf,
            max_relative_difference=math.inf,
            allclose=False,
            first_mismatch=f"layout differs; missing={missing}, extra={extra}",
        )

    max_abs = 0.0
    max_rel = 0.0
    compared = 0
    mismatch: str | None = None
    for key in sorted(first.entries):
        left = first.entries[key]
        right = second.entries[key]
        layout_left = (
            left.dtype,
            left.global_shape,
            left.local_shape,
            left.indices,
        )
        layout_right = (
            right.dtype,
            right.global_shape,
            right.local_shape,
            right.indices,
        )
        if layout_left != layout_right:
            mismatch = f"layout mismatch at {key!r}"
            return StateDistance(
                compared_entries=compared,
                compared_values=0,
                max_absolute_difference=math.inf,
                max_relative_difference=math.inf,
                allclose=False,
                first_mismatch=mismatch,
            )
        for left_value, right_value in zip(left.values, right.values):
            compared += 1
            absolute = abs(left_value - right_value)
            relative = absolute / max(abs(left_value), abs(right_value), 1.0e-30)
            max_abs = max(max_abs, absolute)
            max_rel = max(max_rel, relative)
            if mismatch is None and absolute > atol + rtol * abs(left_value):
                mismatch = (
                    f"{key}: checkpoint={left_value}, replay={right_value}, "
                    f"abs={absolute}, rel={relative}"
                )
    return StateDistance(
        compared_entries=len(first.entries),
        compared_values=compared,
        max_absolute_difference=max_abs,
        max_relative_difference=max_rel,
        allclose=mismatch is None,
        first_mismatch=mismatch,
    )


def _tensor_bytes(tensor: Tensor) -> bytes:
    value = tensor.detach().cpu().contiguous()
    header = _canonical_json(
        {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
        }
    ).encode("utf-8")
    raw = value.view(torch.uint8).numpy().tobytes()
    return struct.pack("<Q", len(header)) + header + raw


def _batch_fingerprint(batch: Any) -> BatchFingerprint:
    hasher = hashlib.sha256()
    provenance = [
        {
            "sample_id": item.sample_id,
            "source_name": item.source_name,
            "source_index": int(item.source_index),
            "split": item.split.value,
            "image_path": str(item.image_path),
            "mask_path": None if item.mask_path is None else str(item.mask_path),
            "metadata": dict(item.metadata),
        }
        for item in batch.provenance
    ]
    metadata = {
        "task": batch.task.value,
        "sample_ids": list(batch.sample_ids),
        "sequence_length": int(batch.text.sequence_length),
        "provenance": provenance,
    }
    hasher.update(_canonical_json(metadata).encode("utf-8"))
    tensors = (
        ("images", batch.images),
        ("input_ids", batch.text.input_ids),
        ("labels", batch.text.labels),
        ("attention_mask", batch.text.attention_mask),
        ("unpadded_lengths", batch.text.unpadded_lengths),
    )
    for name, tensor in tensors:
        hasher.update(name.encode("utf-8"))
        hasher.update(_tensor_bytes(tensor))
    if batch.segmentation_targets is not None:
        hasher.update(b"segmentation_targets")
        hasher.update(_tensor_bytes(batch.segmentation_targets))
    target_shape = (
        None
        if batch.segmentation_targets is None
        else tuple(int(item) for item in batch.segmentation_targets.shape)
    )
    return BatchFingerprint(
        digest=hasher.hexdigest(),
        task=batch.task.value,
        sample_ids=tuple(batch.sample_ids),
        image_shape=tuple(int(item) for item in batch.images.shape),
        sequence_length=int(batch.text.sequence_length),
        segmentation_target_shape=target_shape,
    )


# ---------------------------------------------------------------------------
# RNG replay and one-update execution
# ---------------------------------------------------------------------------


def _draw_rng_probe(device: torch.device) -> RNGProbe:
    return RNGProbe(
        python_values=tuple(random.random() for _ in range(4)),
        numpy_values=tuple(float(item) for item in np.random.random(4).tolist()),
        torch_cpu_values=tuple(float(item) for item in torch.rand(4).tolist()),
        torch_cuda_values=tuple(
            float(item) for item in torch.rand(4, device=device).cpu().tolist()
        ),
    )


def _run_optimizer_update(
    *,
    phase: str,
    config: Any,
    runtime: Any,
    data_pipeline: Any,
    distributed_model: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    cpu_batch: Any,
    compute_support: ModuleType,
) -> OptimizerUpdateReport:
    from m3d.data.sampler import sampler_position_from_batch
    from m3d.trainer import _compose_backward_loss

    epoch, step, sampler_task = sampler_position_from_batch(cpu_batch)
    if epoch != data_pipeline.epoch:
        raise CheckpointIntegrationError(
            f"{phase}: batch epoch={epoch}, pipeline epoch={data_pipeline.epoch}."
        )
    if step != data_pipeline.committed_step:
        raise CheckpointIntegrationError(
            f"{phase}: batch step={step}, committed={data_pipeline.committed_step}."
        )
    if sampler_task is not cpu_batch.task:
        raise CheckpointIntegrationError(
            f"{phase}: sampler task={sampler_task.value}, batch task={cpu_batch.task.value}."
        )
    runtime.assert_all_ranks_equal(
        cpu_batch.task.value,
        label=f"checkpoint integration task at step {step}",
    )

    fingerprint = _batch_fingerprint(cpu_batch)
    gpu_batch = cpu_batch.to(
        runtime.device,
        non_blocking=data_pipeline.non_blocking_transfer,
    )
    optimizer.zero_grad(set_to_none=True)
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
                raise CheckpointIntegrationError(
                    f"{phase}: model returned no loss output."
                )
            backward_loss = _compose_backward_loss(
                runtime=runtime,
                batch=gpu_batch,
                loss_output=output.loss_output,
            )
        backward_loss.backward()

    modules = compute_support._component_modules(  # noqa: SLF001
        distributed_model.unwrapped_model
    )
    gradients = {
        name: compute_support._gradient_summary(module)  # noqa: SLF001
        for name, module in modules.items()
    }
    compute_support._validate_gradient_contract(  # noqa: SLF001
        task=gpu_batch.task.value,
        summaries=gradients,
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
    report = OptimizerUpdateReport(
        phase=phase,
        step_before=int(step),
        step_after=int(scheduler.completed_optimizer_steps),
        task=gpu_batch.task.value,
        sample_ids=tuple(gpu_batch.sample_ids),
        batch_digest=fingerprint.digest,
        local_total_loss=_safe_scalar(output.loss_output.total.detach()),
        global_backward_loss=_safe_scalar(backward_loss.detach()),
        gradient_norm=_safe_scalar(gradient_norm),
        elapsed_seconds=elapsed,
        cuda_memory=runtime.cuda_memory_snapshot(),
    )

    del gpu_batch, output, backward_loss, gradient_norm
    torch.cuda.empty_cache()
    return report


# ---------------------------------------------------------------------------
# Configuration and main distributed test
# ---------------------------------------------------------------------------


def _support_options(
    compute_support: ModuleType,
    options: CheckpointIntegrationOptions,
) -> Any:
    return compute_support.IntegrationOptions(
        config_path=options.config_path,
        strategy=options.strategy,
        output_dir=options.output_dir,
        cache_dir=options.cache_dir,
        local_files_only=options.local_files_only,
        overrides=options.overrides,
        expected_world_size=options.expected_world_size,
        num_workers=options.num_workers,
        allow_non_a100=options.allow_non_a100,
        allow_version_mismatch=options.allow_version_mismatch,
        verify_paths=options.verify_paths,
        verbose_all_ranks=options.verbose_all_ranks,
    )


def _build_checkpoint_config(
    compute_support: ModuleType,
    options: CheckpointIntegrationOptions,
) -> Any:
    support_options = _support_options(compute_support, options)
    config = compute_support._build_integration_config(support_options)  # noqa: SLF001
    output_dir = Path(options.output_dir).expanduser().resolve()
    config.experiment_name = (
        f"{config.experiment_name}-checkpoint-{options.checkpoint_mode}"
    )
    config.checkpoint.output_dir = str(output_dir / "checkpoints")
    config.checkpoint.resume_from = None
    config.checkpoint.save_every_steps = 1
    config.checkpoint.keep_last_n = 1
    config.checkpoint.save_optimizer = True
    config.checkpoint.save_scheduler = True
    config.checkpoint.save_rng_state = True
    config.checkpoint.asynchronous = options.checkpoint_mode == "async"
    config.checkpoint.export_safetensors_at_end = False
    config.logging.tensorboard_dir = str(output_dir / "tensorboard")
    config.validate()
    return config


def _prepare_output_directory(
    runtime: Any,
    output_dir: Path,
    *,
    overwrite: bool,
) -> None:
    local_error: str | None = None
    if runtime.is_main_process:
        try:
            if output_dir.exists():
                populated = any(output_dir.iterdir())
                if populated and not overwrite:
                    raise FileExistsError(
                        f"Output directory is not empty: {output_dir}. "
                        "Use --overwrite-output for a new integration run."
                    )
                if overwrite:
                    shutil.rmtree(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"
    payload = runtime.broadcast_object(
        {"error": local_error} if runtime.is_main_process else None
    )
    if not isinstance(payload, Mapping):
        raise CheckpointIntegrationError(
            "Output-directory status broadcast returned an invalid payload."
        )

    output_error = payload.get("error")
    if output_error is not None:
        raise CheckpointIntegrationError(str(output_error))

    runtime.barrier()


def _gather_summaries(runtime: Any, state: SampledState) -> tuple[Mapping[str, Any], ...]:
    return tuple(dict(item) for item in runtime.all_gather_object(state.summary()))


def run_checkpoint_integration(
    options: CheckpointIntegrationOptions,
) -> CheckpointIntegrationReport | None:
    compute_support = _load_compute_support()

    from m3d.checkpointing import CheckpointManager
    from m3d.config import config_fingerprint
    from m3d.data.loader import build_training_data_pipeline
    from m3d.distributed import build_model_synchronously, prepare_distributed_model
    from m3d.model.m3d import build_m3d_model
    from m3d.optim import build_optimizer
    from m3d.runtime import distributed_runtime
    from m3d.scheduler import build_scheduler
    from m3d.tokenization import M3DTextProcessor, build_tokenizer

    config = _build_checkpoint_config(compute_support, options)
    started_at = time.time()
    manager: CheckpointManager | None = None
    checkpoint_path: Path | None = None
    report_payload: CheckpointIntegrationReport | None = None

    with distributed_runtime(
        config,
        verbose_all_ranks=options.verbose_all_ranks,
    ) as runtime:
        runtime.assert_all_ranks_equal(
            _json_sha256(options.consistency_payload()),
            label="checkpoint integration CLI options",
        )
        runtime.assert_all_ranks_equal(
            _json_sha256(config_fingerprint(config)),
            label="checkpoint integration resolved config",
        )

        output_dir = Path(options.output_dir).expanduser().resolve()
        _prepare_output_directory(
            runtime,
            output_dir,
            overwrite=options.overwrite_output,
        )
        if runtime.is_main_process:
            _atomic_write_json(
                output_dir / "resolved_checkpoint_integration_config.json",
                config.to_dict(),
            )
        runtime.barrier()

        support_options = _support_options(compute_support, options)
        environment = compute_support._environment_report(  # noqa: SLF001
            runtime,
            support_options,
        )
        environment_by_rank = tuple(
            dict(item)
            for item in runtime.all_gather_object(environment.to_dict())
        )
        flash_probe = compute_support._run_flash_probe(runtime)  # noqa: SLF001
        flash_probe_by_rank = tuple(
            dict(item)
            for item in runtime.all_gather_object(flash_probe.to_dict())
        )

        with runtime.main_process_first():
            tokenizer_bundle = build_tokenizer(
                config,
                cache_dir=options.cache_dir,
                local_files_only=options.local_files_only,
            )
        runtime.assert_all_ranks_equal(
            _json_sha256(tokenizer_bundle.metadata.to_dict()),
            label="checkpoint integration tokenizer metadata",
        )
        text_processor = M3DTextProcessor(tokenizer_bundle, config)
        data_pipeline = build_training_data_pipeline(
            config=config,
            runtime=runtime,
            tokenizer_bundle=tokenizer_bundle,
            text_processor=text_processor,
        )
        schedule = compute_support._validate_schedule(  # noqa: SLF001
            data_pipeline,
            runtime,
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
        compute_support._validate_model_independence(model)  # noqa: SLF001
        distributed_model, distributed_report = prepare_distributed_model(
            model,
            runtime,
        )
        optimizer, optimizer_report = build_optimizer(
            distributed_model.unwrapped_model,
            config,
            distributed_strategy=distributed_model.strategy,
            allow_unfused_fallback=(distributed_model.strategy == "fsdp2"),
        )
        if distributed_model.strategy == "ddp" and not optimizer_report.fused_enabled:
            raise CheckpointIntegrationError(
                "DDP checkpoint integration requires fused AdamW on A100."
            )
        scheduler, scheduler_report = build_scheduler(
            optimizer,
            config,
            steps_per_epoch=data_pipeline.steps_per_epoch,
        )
        manager = CheckpointManager(
            config=config,
            runtime=runtime,
            distributed_model=distributed_model,
            optimizer=optimizer,
            scheduler=scheduler,
            data_pipeline=data_pipeline,
        )

        distributed_model.train()
        data_pipeline.set_epoch(0, committed_step=0)
        iterator = iter(data_pipeline.loader)
        try:
            first_batch = next(iterator)
        except StopIteration as exc:
            raise CheckpointIntegrationError(
                "Training DataLoader produced no first integration batch."
            ) from exc

        first_update = _run_optimizer_update(
            phase="before_checkpoint",
            config=config,
            runtime=runtime,
            data_pipeline=data_pipeline,
            distributed_model=distributed_model,
            optimizer=optimizer,
            scheduler=scheduler,
            cpu_batch=first_batch,
            compute_support=compute_support,
        )
        if scheduler.completed_optimizer_steps != 1 or data_pipeline.committed_step != 1:
            raise CheckpointIntegrationError(
                "First integration update did not finish at checkpoint cursor 1."
            )

        pending = manager.save(
            force_synchronous=options.checkpoint_mode == "sync"
        )
        save_report = pending.wait()
        checkpoint_path = Path(save_report.checkpoint_path).resolve()
        if options.checkpoint_mode == "async" and not save_report.asynchronous:
            if not options.allow_async_fallback:
                raise CheckpointIntegrationError(
                    "Asynchronous checkpoint was requested but DCP used the "
                    "synchronous fallback. Re-run with --allow-async-fallback only "
                    "for diagnosis."
                )

        checkpoint_model = _snapshot_model(
            distributed_model.unwrapped_model,
            sample_count=options.samples_per_tensor,
        )
        checkpoint_optimizer = _snapshot_optimizer(
            optimizer,
            sample_count=options.samples_per_tensor,
        )
        checkpoint_scheduler_state = scheduler.state_dict()
        checkpoint_data_state = data_pipeline.state_dict()

        # These draws intentionally occur after save.  Loading the checkpoint
        # must recreate them exactly, and consuming the same probe on both paths
        # leaves the following model update on the same random stream position.
        expected_rng_probe = _draw_rng_probe(runtime.device)

        try:
            uninterrupted_batch = next(iterator)
        except StopIteration as exc:
            raise CheckpointIntegrationError(
                "Training DataLoader produced no second integration batch."
            ) from exc
        uninterrupted_batch_fingerprint = _batch_fingerprint(uninterrupted_batch)
        uninterrupted_update = _run_optimizer_update(
            phase="uninterrupted_after_checkpoint",
            config=config,
            runtime=runtime,
            data_pipeline=data_pipeline,
            distributed_model=distributed_model,
            optimizer=optimizer,
            scheduler=scheduler,
            cpu_batch=uninterrupted_batch,
            compute_support=compute_support,
        )
        uninterrupted_model = _snapshot_model(
            distributed_model.unwrapped_model,
            sample_count=options.samples_per_tensor,
        )
        uninterrupted_optimizer = _snapshot_optimizer(
            optimizer,
            sample_count=options.samples_per_tensor,
        )
        uninterrupted_scheduler_state = scheduler.state_dict()
        uninterrupted_data_state = data_pipeline.state_dict()

        if scheduler.completed_optimizer_steps != 2 or data_pipeline.committed_step != 2:
            raise CheckpointIntegrationError(
                "Uninterrupted path did not complete the second optimizer update."
            )

        resume_report = manager.load(checkpoint_path, exact=True)
        restored_model = _snapshot_model(
            distributed_model.unwrapped_model,
            sample_count=options.samples_per_tensor,
        )
        restored_optimizer = _snapshot_optimizer(
            optimizer,
            sample_count=options.samples_per_tensor,
        )
        restored_scheduler_state = scheduler.state_dict()
        restored_data_state = data_pipeline.state_dict()

        model_restore_exact = restored_model.digest == checkpoint_model.digest
        optimizer_restore_exact = (
            restored_optimizer.digest == checkpoint_optimizer.digest
        )
        scheduler_restore_exact = (
            restored_scheduler_state == checkpoint_scheduler_state
        )
        data_restore_exact = restored_data_state == checkpoint_data_state
        _distributed_assert(
            runtime,
            model_restore_exact,
            label="model checkpoint round-trip",
            detail=(
                f"saved={checkpoint_model.digest}, restored={restored_model.digest}"
            ),
        )
        _distributed_assert(
            runtime,
            optimizer_restore_exact,
            label="optimizer checkpoint round-trip",
            detail=(
                f"saved={checkpoint_optimizer.digest}, "
                f"restored={restored_optimizer.digest}"
            ),
        )
        _distributed_assert(
            runtime,
            scheduler_restore_exact,
            label="scheduler checkpoint round-trip",
            detail="scheduler state_dict differs after exact resume",
        )
        _distributed_assert(
            runtime,
            data_restore_exact,
            label="data cursor checkpoint round-trip",
            detail="training data state_dict differs after exact resume",
        )

        actual_rng_probe = _draw_rng_probe(runtime.device)
        rng_exact = actual_rng_probe == expected_rng_probe
        _distributed_assert(
            runtime,
            rng_exact,
            label="rank-local RNG replay",
            detail=(
                f"expected={expected_rng_probe.to_dict()}, "
                f"actual={actual_rng_probe.to_dict()}"
            ),
        )

        resumed_iterator = iter(data_pipeline.loader)
        try:
            replayed_batch = next(resumed_iterator)
        except StopIteration as exc:
            raise DataReplayError(
                "Resumed DataLoader produced no batch at committed step 1."
            ) from exc
        replayed_batch_fingerprint = _batch_fingerprint(replayed_batch)
        batch_exact = (
            replayed_batch_fingerprint.digest
            == uninterrupted_batch_fingerprint.digest
        )
        _distributed_assert(
            runtime,
            batch_exact,
            label="next batch exact replay",
            detail=(
                f"uninterrupted={uninterrupted_batch_fingerprint.to_dict()}, "
                f"replayed={replayed_batch_fingerprint.to_dict()}"
            ),
        )

        replayed_update = _run_optimizer_update(
            phase="replayed_after_resume",
            config=config,
            runtime=runtime,
            data_pipeline=data_pipeline,
            distributed_model=distributed_model,
            optimizer=optimizer,
            scheduler=scheduler,
            cpu_batch=replayed_batch,
            compute_support=compute_support,
        )
        replayed_model = _snapshot_model(
            distributed_model.unwrapped_model,
            sample_count=options.samples_per_tensor,
        )
        replayed_optimizer = _snapshot_optimizer(
            optimizer,
            sample_count=options.samples_per_tensor,
        )
        replayed_scheduler_state = scheduler.state_dict()
        replayed_data_state = data_pipeline.state_dict()

        model_distance = _state_distance(
            uninterrupted_model,
            replayed_model,
            atol=_MODEL_REPLAY_ATOL,
            rtol=_MODEL_REPLAY_RTOL,
        )
        optimizer_distance = _state_distance(
            uninterrupted_optimizer,
            replayed_optimizer,
            atol=_OPTIMIZER_REPLAY_ATOL,
            rtol=_OPTIMIZER_REPLAY_RTOL,
        )
        scheduler_replay_exact = (
            replayed_scheduler_state == uninterrupted_scheduler_state
        )
        data_replay_exact = replayed_data_state == uninterrupted_data_state
        _distributed_assert(
            runtime,
            model_distance.allclose,
            label="model continuation replay",
            detail=str(model_distance.to_dict()),
        )
        _distributed_assert(
            runtime,
            optimizer_distance.allclose,
            label="optimizer continuation replay",
            detail=str(optimizer_distance.to_dict()),
        )
        _distributed_assert(
            runtime,
            scheduler_replay_exact,
            label="scheduler continuation replay",
            detail="replayed scheduler differs from uninterrupted scheduler",
        )
        _distributed_assert(
            runtime,
            data_replay_exact,
            label="data continuation replay",
            detail="replayed sampler state differs from uninterrupted state",
        )

        if scheduler.completed_optimizer_steps != 2 or data_pipeline.committed_step != 2:
            raise CheckpointIntegrationError(
                "Replayed path did not finish at optimizer/data cursor 2."
            )

        manager.close()
        manager = None
        runtime.barrier()

        first_updates = tuple(
            dict(item)
            for item in runtime.all_gather_object(first_update.to_dict())
        )
        uninterrupted_updates = tuple(
            dict(item)
            for item in runtime.all_gather_object(uninterrupted_update.to_dict())
        )
        replayed_updates = tuple(
            dict(item)
            for item in runtime.all_gather_object(replayed_update.to_dict())
        )
        checkpoint_models = _gather_summaries(runtime, checkpoint_model)
        restored_models = _gather_summaries(runtime, restored_model)
        checkpoint_optimizers = _gather_summaries(runtime, checkpoint_optimizer)
        restored_optimizers = _gather_summaries(runtime, restored_optimizer)
        next_batches = tuple(
            dict(item)
            for item in runtime.all_gather_object(
                uninterrupted_batch_fingerprint.to_dict()
            )
        )
        replayed_batches = tuple(
            dict(item)
            for item in runtime.all_gather_object(
                replayed_batch_fingerprint.to_dict()
            )
        )
        rng_exact_by_rank = tuple(
            bool(item) for item in runtime.all_gather_object(rng_exact)
        )
        model_restore_by_rank = tuple(
            bool(item) for item in runtime.all_gather_object(model_restore_exact)
        )
        optimizer_restore_by_rank = tuple(
            bool(item) for item in runtime.all_gather_object(optimizer_restore_exact)
        )
        scheduler_restore_by_rank = tuple(
            bool(item) for item in runtime.all_gather_object(scheduler_restore_exact)
        )
        data_restore_by_rank = tuple(
            bool(item) for item in runtime.all_gather_object(data_restore_exact)
        )
        model_distances = tuple(
            dict(item)
            for item in runtime.all_gather_object(model_distance.to_dict())
        )
        optimizer_distances = tuple(
            dict(item)
            for item in runtime.all_gather_object(optimizer_distance.to_dict())
        )

        elapsed = time.time() - started_at
        report_payload = CheckpointIntegrationReport(
            state_version=_CHECKPOINT_INTEGRATION_STATE_VERSION,
            status="passed",
            strategy=options.strategy,
            checkpoint_mode_requested=options.checkpoint_mode,
            checkpoint_mode_used=(
                "async" if save_report.asynchronous else "sync"
            ),
            config_path=str(Path(options.config_path).expanduser().resolve()),
            output_dir=str(output_dir),
            checkpoint_path=str(checkpoint_path),
            checkpoint_retained=options.keep_checkpoint,
            elapsed_seconds=elapsed,
            task_schedule=tuple(schedule),
            environment_by_rank=environment_by_rank,
            flash_probe_by_rank=flash_probe_by_rank,
            tokenizer=tokenizer_bundle.metadata.to_dict(),
            data_pipeline=dict(data_pipeline.summary()),
            model=model_report.to_dict(),
            distributed=distributed_report.to_dict(),
            optimizer=optimizer_report.to_dict(include_parameter_names=False),
            scheduler=scheduler_report.to_dict(),
            save_report=save_report.to_dict(),
            resume_report=resume_report.to_dict(),
            first_update_by_rank=first_updates,
            uninterrupted_update_by_rank=uninterrupted_updates,
            replayed_update_by_rank=replayed_updates,
            checkpoint_model_state_by_rank=checkpoint_models,
            restored_model_state_by_rank=restored_models,
            checkpoint_optimizer_state_by_rank=checkpoint_optimizers,
            restored_optimizer_state_by_rank=restored_optimizers,
            next_batch_by_rank=next_batches,
            replayed_batch_by_rank=replayed_batches,
            rng_exact_by_rank=rng_exact_by_rank,
            model_restore_exact_by_rank=model_restore_by_rank,
            optimizer_restore_exact_by_rank=optimizer_restore_by_rank,
            scheduler_restore_exact_by_rank=scheduler_restore_by_rank,
            data_cursor_restore_exact_by_rank=data_restore_by_rank,
            replay_model_distance_by_rank=model_distances,
            replay_optimizer_distance_by_rank=optimizer_distances,
            completed_optimizer_steps=int(scheduler.completed_optimizer_steps),
            committed_microbatches=int(data_pipeline.committed_step),
        )
        if runtime.is_main_process:
            _atomic_write_json(
                output_dir / "checkpoint_integration_report.json",
                report_payload.to_dict(),
            )
        runtime.barrier()

        if not options.keep_checkpoint:
            local_cleanup_error: str | None = None
            if runtime.is_main_process:
                try:
                    shutil.rmtree(Path(config.checkpoint.output_dir), ignore_errors=False)
                except FileNotFoundError:
                    pass
                except Exception as exc:
                    local_cleanup_error = f"{type(exc).__name__}: {exc}"
            cleanup_payload = runtime.broadcast_object(
                {"error": local_cleanup_error} if runtime.is_main_process else None
            )
            if not isinstance(cleanup_payload, Mapping):
                raise CheckpointIntegrationError(
                    "Checkpoint cleanup status broadcast returned an invalid payload."
                )

            cleanup_error = cleanup_payload.get("error")
            if cleanup_error is not None:
                raise CheckpointIntegrationError(
                    f"Checkpoint passed but cleanup failed: {cleanup_error}"
                )

            runtime.barrier()

        runtime.logger.info(
            "Checkpoint integration passed: strategy=%s mode=%s schedule=%s "
            "elapsed=%.2fs report=%s",
            options.strategy,
            "async" if save_report.asynchronous else "sync",
            schedule,
            elapsed,
            output_dir / "checkpoint_integration_report.json",
        )

    return report_payload


# ---------------------------------------------------------------------------
# CLI and dependency-light self-test
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run an exact M3D distributed-checkpoint save/load/replay test on "
            "two ASPIRE 2A GPUs."
        )
    )
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" / "m3d_joint_finetune.yaml"),
        help="Path to the M3D YAML configuration.",
    )
    parser.add_argument(
        "--strategy",
        choices=("ddp", "fsdp2"),
        default="ddp",
        help="Distributed wrapper whose checkpoint path should be validated.",
    )
    parser.add_argument(
        "--checkpoint-mode",
        choices=("async", "sync"),
        default="async",
        help="Use asynchronous DCP with Gloo staging or synchronous DCP.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help=(
            "Report/checkpoint root. Defaults to "
            "outputs/aspire2a-checkpoint-integration-<strategy>-<mode>."
        ),
    )
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Additional dotted.path=value configuration override.",
    )
    parser.add_argument("--expected-world-size", type=int, default=2)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help=(
            "DataLoader workers per rank. Zero is the strictest checkpoint "
            "replay baseline; deterministic per-sample augmentation also allows "
            "later worker-count tests."
        ),
    )
    parser.add_argument("--allow-non-a100", action="store_true")
    parser.add_argument("--allow-version-mismatch", action="store_true")
    parser.add_argument(
        "--allow-async-fallback",
        action="store_true",
        help="Do not fail if async mode falls back to synchronous DCP.",
    )
    parser.add_argument("--skip-path-verification", action="store_true")
    parser.add_argument("--verbose-all-ranks", action="store_true")
    parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="Delete a previous integration output directory before starting.",
    )
    parser.add_argument(
        "--keep-checkpoint",
        action="store_true",
        help="Keep the large DCP directory after the successful test.",
    )
    parser.add_argument(
        "--samples-per-tensor",
        type=int,
        default=_DEFAULT_SAMPLE_COUNT_PER_TENSOR,
        help="Deterministic sampled values per model/optimizer state tensor.",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run dependency-light local helper tests and exit.",
    )
    return parser


def _parse_options(
    argv: Sequence[str] | None = None,
) -> tuple[CheckpointIntegrationOptions | None, bool]:
    args = _build_parser().parse_args(argv)
    if args.self_test:
        return None, True
    if args.expected_world_size <= 1:
        raise ValueError("--expected-world-size must be greater than one.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.samples_per_tensor <= 0:
        raise ValueError("--samples-per-tensor must be positive.")
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else (
            PROJECT_ROOT
            / "outputs"
            / (
                "aspire2a-checkpoint-integration-"
                f"{args.strategy}-{args.checkpoint_mode}"
            )
        ).resolve()
    )
    return (
        CheckpointIntegrationOptions(
            config_path=str(Path(args.config).expanduser().resolve()),
            strategy=str(args.strategy),
            checkpoint_mode=str(args.checkpoint_mode),
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
            allow_async_fallback=bool(args.allow_async_fallback),
            verify_paths=not bool(args.skip_path_verification),
            verbose_all_ranks=bool(args.verbose_all_ranks),
            overwrite_output=bool(args.overwrite_output),
            keep_checkpoint=bool(args.keep_checkpoint),
            samples_per_tensor=int(args.samples_per_tensor),
        ),
        False,
    )


def _run_self_test() -> dict[str, Any]:
    support = _load_compute_support()
    if not hasattr(support, "IntegrationOptions"):
        raise AssertionError("Compute-integration support did not load correctly.")

    torch.manual_seed(123)
    model = nn.Sequential(nn.Linear(4, 6), nn.GELU(), nn.Linear(6, 2))
    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.parameters()),
                "lr": 1.0e-3,
                "group_name": "toy/decay",
                "role": "language_model",
                "decay_kind": "decay",
                "param_names": [name for name, _ in model.named_parameters()],
            }
        ]
    )
    input_tensor = torch.randn(3, 4)
    model(input_tensor).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    model_state = _snapshot_model(model, sample_count=4)
    optimizer_state = _snapshot_optimizer(optimizer, sample_count=4)
    saved_model = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    saved_optimizer = copy.deepcopy(optimizer.state_dict())

    model(torch.randn(3, 4)).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    disturbed_model = _snapshot_model(model, sample_count=4)
    if disturbed_model.digest == model_state.digest:
        raise AssertionError("Toy optimizer update did not disturb sampled model state.")

    model.load_state_dict(saved_model)
    optimizer.load_state_dict(saved_optimizer)
    restored_model = _snapshot_model(model, sample_count=4)
    restored_optimizer = _snapshot_optimizer(optimizer, sample_count=4)
    if restored_model.digest != model_state.digest:
        raise AssertionError("Toy model state did not restore exactly.")
    if restored_optimizer.digest != optimizer_state.digest:
        raise AssertionError("Toy optimizer state did not restore exactly.")

    distance = _state_distance(
        model_state,
        restored_model,
        atol=0.0,
        rtol=0.0,
    )
    if not distance.allclose:
        raise AssertionError(distance.first_mismatch)

    random.seed(55)
    np.random.seed(55)
    torch.manual_seed(55)
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    first_probe = RNGProbe(
        python_values=tuple(random.random() for _ in range(4)),
        numpy_values=tuple(float(item) for item in np.random.random(4).tolist()),
        torch_cpu_values=tuple(float(item) for item in torch.rand(4).tolist()),
        torch_cuda_values=(),
    )
    random.setstate(python_state)
    np.random.set_state(numpy_state)
    torch.set_rng_state(torch_state)
    second_probe = RNGProbe(
        python_values=tuple(random.random() for _ in range(4)),
        numpy_values=tuple(float(item) for item in np.random.random(4).tolist()),
        torch_cpu_values=tuple(float(item) for item in torch.rand(4).tolist()),
        torch_cuda_values=(),
    )
    if first_probe != second_probe:
        raise AssertionError("CPU RNG replay helper is not exact.")

    options, self_test = _parse_options(
        [
            "--strategy",
            "fsdp2",
            "--checkpoint-mode",
            "sync",
            "--expected-world-size",
            "4",
            "--samples-per-tensor",
            "3",
        ]
    )
    if self_test or options is None:
        raise AssertionError("CLI parser unexpectedly selected self-test.")
    if options.strategy != "fsdp2" or options.checkpoint_mode != "sync":
        raise AssertionError("CLI parser did not preserve strategy/mode.")

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "report.json"
        payload = {"status": "passed", "digest": model_state.digest}
        _atomic_write_json(path, payload)
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise AssertionError("Atomic JSON round-trip failed.")
        if list(Path(directory).glob("*.tmp-*")):
            raise AssertionError("Atomic JSON helper left temporary files.")

    return {
        "status": "passed",
        "state_version": _CHECKPOINT_INTEGRATION_STATE_VERSION,
        "compute_support_loaded": True,
        "model_snapshot_roundtrip": True,
        "optimizer_snapshot_roundtrip": True,
        "rng_replay": True,
        "state_distance": distance.to_dict(),
        "model_snapshot": model_state.summary(),
        "optimizer_snapshot": optimizer_state.summary(),
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
        report = run_checkpoint_integration(options)
    except BaseException as exc:
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
                    "checkpoint_mode_used": report.checkpoint_mode_used,
                    "task_schedule": list(report.task_schedule),
                    "completed_optimizer_steps": report.completed_optimizer_steps,
                    "committed_microbatches": report.committed_microbatches,
                    "checkpoint_retained": report.checkpoint_retained,
                    "elapsed_seconds": report.elapsed_seconds,
                    "report": str(
                        Path(report.output_dir)
                        / "checkpoint_integration_report.json"
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
