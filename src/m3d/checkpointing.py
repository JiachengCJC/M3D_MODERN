"""Distributed, checkpoint-exact persistence for M3D-Modernized.

This module is used after the complete model has been wrapped with DDP or
composable FSDP2 and after the optimizer and scheduler have been constructed.
It uses :mod:`torch.distributed.checkpoint` (DCP) for model and optimizer state
so the same code path can save replicated DDP tensors or sharded FSDP2 DTensors.
Small control state is stored in validated JSON sidecars.

A durable training checkpoint contains::

    checkpoint-step-00001000/
    ├── COMPLETED.json
    ├── trainer_state.json
    ├── resolved_config.json
    ├── rank_state/
    │   ├── rank-00000.json
    │   └── rank-00001.json
    └── dcp/
        ├── .metadata
        └── __*.distcp

The directory is first written under an ``.incomplete-*`` name.  It is renamed
and made discoverable through ``latest.json`` only after every rank has
finished DCP upload and all sidecars are present.  A crashed write therefore
cannot be mistaken for a resumable checkpoint.

Checkpoints are valid only at a gradient-accumulation boundary, after::

    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    data_pipeline.commit_batch(batch)

The sampler cursor represents consumed microbatches, while the scheduler
counter represents successful optimizer updates.  Both are validated before
save and after load.
"""

from __future__ import annotations

import base64
import copy
import dataclasses
import hashlib
import json
import os
import random
import re
import shutil
import socket
import time
import uuid
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from .config import (
    LATEST_CHECKPOINT_SENTINEL,
    CheckpointConfig,
    ExperimentConfig,
)
from .distributed import DistributedM3DModel
from .optim import (
    optimizer_groups_by_name,
    restore_optimizer_group_metadata,
    validate_optimizer_parameter_coverage,
)
from .runtime import RuntimeContext, atomic_write_json
from .scheduler import CosineWarmupScheduler, validate_resume_progress

if TYPE_CHECKING:
    from .data.loader import TrainingDataPipeline


_CHECKPOINT_STATE_VERSION = 1
_RNG_STATE_VERSION = 1
_COMPLETION_FILE = "COMPLETED.json"
_TRAINER_STATE_FILE = "trainer_state.json"
_CONFIG_FILE = "resolved_config.json"
_LATEST_FILE = "latest.json"
_RANK_STATE_DIR = "rank_state"
_DCP_DIR = "dcp"
_CHECKPOINT_PATTERN = re.compile(r"^checkpoint-step-(\d{8,})$")


class CheckpointError(RuntimeError):
    """Base class for checkpoint save/load failures."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised when a checkpoint cannot exactly resume the current run."""


class CheckpointBoundaryError(CheckpointError):
    """Raised when save is requested inside an accumulation window."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _torch_major_minor(version: str) -> tuple[int, int]:
    match = re.match(r"^(\d+)\.(\d+)", version)
    if match is None:
        raise CheckpointCompatibilityError(
            f"Cannot parse PyTorch version {version!r}."
        )
    return int(match.group(1)), int(match.group(2))


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _as_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointCompatibilityError(
            f"{name} must be a mapping, got {type(value).__name__}."
        )
    return cast(Mapping[str, Any], value)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise CheckpointCompatibilityError(
            f"Required checkpoint file does not exist: {path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise CheckpointCompatibilityError(
            f"Checkpoint JSON is malformed: {path}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise CheckpointCompatibilityError(
            f"Checkpoint JSON root must be an object: {path}"
        )
    return payload


def _encode_bytes(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _decode_bytes(value: Any, *, name: str) -> bytes:
    if not isinstance(value, str):
        raise CheckpointCompatibilityError(f"{name} must be a base64 string.")
    try:
        return base64.b64decode(value.encode("ascii"), validate=True)
    except Exception as exc:
        raise CheckpointCompatibilityError(f"{name} is not valid base64.") from exc


def _tensor_to_base64(tensor: Tensor) -> dict[str, Any]:
    cpu = tensor.detach().to(device="cpu", dtype=torch.uint8).contiguous()
    return {
        "numel": int(cpu.numel()),
        "data": _encode_bytes(cpu.numpy().tobytes()),
    }


def _tensor_from_base64(payload: Any, *, name: str) -> Tensor:
    mapping = _as_mapping(payload, name=name)
    numel = mapping.get("numel")
    if not isinstance(numel, int) or isinstance(numel, bool) or numel < 0:
        raise CheckpointCompatibilityError(f"{name}.numel is invalid: {numel!r}.")
    raw = _decode_bytes(mapping.get("data"), name=f"{name}.data")
    if len(raw) != numel:
        raise CheckpointCompatibilityError(
            f"{name} byte count differs: declared={numel}, actual={len(raw)}."
        )
    array = np.frombuffer(raw, dtype=np.uint8).copy()
    return torch.from_numpy(array)


def _jsonify_python_random_state(state: tuple[Any, ...]) -> dict[str, Any]:
    if len(state) != 3:
        raise CheckpointError("Unexpected Python random state structure.")
    version, internal, gaussian = state
    if not isinstance(internal, tuple):
        raise CheckpointError("Python random internal state must be a tuple.")
    return {
        "version": int(version),
        "internal": [int(item) for item in internal],
        "gaussian": None if gaussian is None else float(gaussian),
    }


def _restore_python_random_state(payload: Any) -> None:
    mapping = _as_mapping(payload, name="python_random")
    version = mapping.get("version")
    internal = mapping.get("internal")
    gaussian = mapping.get("gaussian")
    if not isinstance(version, int) or isinstance(version, bool):
        raise CheckpointCompatibilityError("python_random.version must be an integer.")
    if not isinstance(internal, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in internal
    ):
        raise CheckpointCompatibilityError(
            "python_random.internal must be a list of integers."
        )
    if gaussian is not None and not isinstance(gaussian, (int, float)):
        raise CheckpointCompatibilityError(
            "python_random.gaussian must be numeric or null."
        )
    random.setstate(
        (
            int(version),
            tuple(int(item) for item in internal),
            None if gaussian is None else float(gaussian),
        )
    )


def _jsonify_numpy_random_state(state: tuple[Any, ...]) -> dict[str, Any]:
    if len(state) != 5:
        raise CheckpointError("Unexpected NumPy random state structure.")
    algorithm, keys, position, has_gaussian, cached_gaussian = state
    array = np.asarray(keys, dtype=np.uint32)
    return {
        "algorithm": str(algorithm),
        "keys": [int(item) for item in array.tolist()],
        "position": int(position),
        "has_gaussian": int(has_gaussian),
        "cached_gaussian": float(cached_gaussian),
    }


def _restore_numpy_random_state(payload: Any) -> None:
    mapping = _as_mapping(payload, name="numpy_random")
    algorithm = mapping.get("algorithm")
    keys = mapping.get("keys")
    position = mapping.get("position")
    has_gaussian = mapping.get("has_gaussian")
    cached_gaussian = mapping.get("cached_gaussian")
    if not isinstance(algorithm, str) or not algorithm:
        raise CheckpointCompatibilityError("numpy_random.algorithm is invalid.")
    if not isinstance(keys, list) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in keys
    ):
        raise CheckpointCompatibilityError(
            "numpy_random.keys must be a list of integers."
        )
    if not isinstance(position, int) or isinstance(position, bool):
        raise CheckpointCompatibilityError("numpy_random.position is invalid.")
    if not isinstance(has_gaussian, int) or isinstance(has_gaussian, bool):
        raise CheckpointCompatibilityError("numpy_random.has_gaussian is invalid.")
    if not isinstance(cached_gaussian, (int, float)):
        raise CheckpointCompatibilityError("numpy_random.cached_gaussian is invalid.")
    np.random.set_state(
        (
            algorithm,
            np.asarray(keys, dtype=np.uint32),
            int(position),
            int(has_gaussian),
            float(cached_gaussian),
        )
    )


@dataclass(frozen=True, slots=True)
class RNGSnapshot:
    """Safe JSON representation of one process's random-number streams."""

    python_random: Mapping[str, Any]
    numpy_random: Mapping[str, Any]
    torch_cpu: Mapping[str, Any]
    torch_cuda: Mapping[str, Any] | None
    cuda_device_index: int | None

    @classmethod
    def capture(cls, device: torch.device) -> "RNGSnapshot":
        cuda_payload: Mapping[str, Any] | None = None
        cuda_index: int | None = None
        if device.type == "cuda":
            if not torch.cuda.is_available():
                raise CheckpointError("Runtime device is CUDA but CUDA is unavailable.")
            if device.index is None:
                raise CheckpointError("CUDA runtime device must have an explicit index.")
            cuda_index = int(device.index)
            cuda_payload = _tensor_to_base64(torch.cuda.get_rng_state(device))
        return cls(
            python_random=_jsonify_python_random_state(random.getstate()),
            numpy_random=_jsonify_numpy_random_state(np.random.get_state()),
            torch_cpu=_tensor_to_base64(torch.get_rng_state()),
            torch_cuda=cuda_payload,
            cuda_device_index=cuda_index,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_version": _RNG_STATE_VERSION,
            "python_random": dict(self.python_random),
            "numpy_random": dict(self.numpy_random),
            "torch_cpu": dict(self.torch_cpu),
            "torch_cuda": None if self.torch_cuda is None else dict(self.torch_cuda),
            "cuda_device_index": self.cuda_device_index,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RNGSnapshot":
        if payload.get("state_version") != _RNG_STATE_VERSION:
            raise CheckpointCompatibilityError(
                "Unsupported RNG state version "
                f"{payload.get('state_version')!r}."
            )
        cuda_index = payload.get("cuda_device_index")
        if cuda_index is not None and (
            not isinstance(cuda_index, int) or isinstance(cuda_index, bool)
        ):
            raise CheckpointCompatibilityError("cuda_device_index is invalid.")
        torch_cuda = payload.get("torch_cuda")
        if torch_cuda is not None:
            torch_cuda = _as_mapping(torch_cuda, name="torch_cuda")
        return cls(
            python_random=_as_mapping(
                payload.get("python_random"), name="python_random"
            ),
            numpy_random=_as_mapping(
                payload.get("numpy_random"), name="numpy_random"
            ),
            torch_cpu=_as_mapping(payload.get("torch_cpu"), name="torch_cpu"),
            torch_cuda=torch_cuda,
            cuda_device_index=cuda_index,
        )

    def restore(self, device: torch.device) -> None:
        _restore_python_random_state(self.python_random)
        _restore_numpy_random_state(self.numpy_random)
        torch.set_rng_state(_tensor_from_base64(self.torch_cpu, name="torch_cpu"))
        if self.torch_cuda is None:
            if device.type == "cuda":
                raise CheckpointCompatibilityError(
                    "Checkpoint has no CUDA RNG state for a CUDA resume."
                )
            return
        if device.type != "cuda" or device.index is None:
            raise CheckpointCompatibilityError(
                "Checkpoint contains CUDA RNG state but current runtime is not CUDA."
            )
        # Rank-local CUDA RNG belongs to the current local device.  The saved
        # numeric index is diagnostic; jobs may be relaunched with a different
        # CUDA_VISIBLE_DEVICES ordering while keeping the same local rank.
        torch.cuda.set_rng_state(
            _tensor_from_base64(self.torch_cuda, name="torch_cuda"),
            device=device,
        )


def _training_compatibility_payload(config: ExperimentConfig) -> dict[str, Any]:
    """Return semantic training config while excluding operational paths.

    Resume location, checkpoint retention, logging cadence, and DDP-vs-FSDP2
    wrapping do not alter model/data/optimization semantics.  Published
    initialization checkpoint paths are also irrelevant after the complete
    model state has been restored.
    """

    payload = copy.deepcopy(config.to_dict())
    payload.pop("checkpoint", None)
    payload.pop("logging", None)
    payload.pop("distributed", None)

    model = payload.get("model")
    if isinstance(model, MutableMapping):
        for key in ("main_vision", "seg_vision", "projector", "segmentation"):
            section = model.get(key)
            if isinstance(section, MutableMapping):
                section.pop("checkpoint_path", None)
        lora = model.get("lora")
        if isinstance(lora, MutableMapping):
            lora.pop("adapter_checkpoint_path", None)
    return payload


def training_compatibility_sha256(config: ExperimentConfig) -> str:
    return _sha256_payload(_training_compatibility_payload(config))


def _state_dict_options() -> StateDictOptions:
    # Distributed/sharded state avoids gathering the complete 4B model on one
    # rank. DCP handles canonical FQNs across DDP and FSDP2.
    return StateDictOptions(
        full_state_dict=False,
        cpu_offload=False,
        strict=True,
        keep_submodule_prefixes=True,
        flatten_optimizer_state_dict=False,
    )


def _materialise_empty_lazy_optimizer_states(
    optimizer_state: MutableMapping[str, Any],
) -> None:
    """Represent parameters AdamW has not seen yet with empty state mappings.

    AdamW creates ``exp_avg``/``exp_avg_sq`` lazily on the first gradient.
    Conditional M3D branches therefore have valid optimizer parameters with no
    state at an early checkpoint, and the unsupervised SegVol IoU head may
    remain lazy permanently. PyTorch 2.6 ``set_state_dict`` indexes every FQN
    in each param group, so explicit empty mappings are required for an exact
    load. ``Optimizer.load_state_dict`` preserves these as lazy empty states.
    """

    state = optimizer_state.get("state")
    param_groups = optimizer_state.get("param_groups")
    if not isinstance(state, MutableMapping) or not isinstance(param_groups, list):
        raise CheckpointCompatibilityError(
            "Optimizer state dictionary has an unexpected DCP structure."
        )
    for group in param_groups:
        if not isinstance(group, Mapping):
            raise CheckpointCompatibilityError(
                "Optimizer param group is not a mapping."
            )
        parameters = group.get("params")
        if not isinstance(parameters, list):
            raise CheckpointCompatibilityError(
                "Optimizer param group does not contain an FQN list."
            )
        for fqn in parameters:
            if not isinstance(fqn, str) or not fqn:
                raise CheckpointCompatibilityError(
                    f"Optimizer parameter FQN is invalid: {fqn!r}."
                )
            state.setdefault(fqn, {})


class _ModelOptimizerState(Stateful):
    """DCP Stateful wrapper using parallelism-aware state-dict APIs."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer | None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.options = _state_dict_options()

    def state_dict(self) -> dict[str, Any]:
        if self.optimizer is None:
            return {
                "model": get_model_state_dict(
                    self.model,
                    options=self.options,
                )
            }
        model_state, optimizer_state = get_state_dict(
            self.model,
            self.optimizer,
            options=self.options,
        )
        return {
            "model": model_state,
            "optimizer": optimizer_state,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        model_state = state_dict.get("model")
        if not isinstance(model_state, dict):
            raise CheckpointCompatibilityError(
                "DCP checkpoint does not contain a model state dictionary."
            )
        if self.optimizer is None:
            incompatible = set_model_state_dict(
                self.model,
                model_state,
                options=self.options,
            )
        else:
            optimizer_state = state_dict.get("optimizer")
            if not isinstance(optimizer_state, dict):
                raise CheckpointCompatibilityError(
                    "Exact resume requested optimizer state, but DCP checkpoint "
                    "does not contain it."
                )
            _materialise_empty_lazy_optimizer_states(optimizer_state)
            incompatible = set_state_dict(
                self.model,
                self.optimizer,
                model_state_dict=model_state,
                optim_state_dict=optimizer_state,
                options=self.options,
            )
        missing = list(getattr(incompatible, "missing_keys", ()))
        unexpected = list(getattr(incompatible, "unexpected_keys", ()))
        if missing or unexpected:
            raise CheckpointCompatibilityError(
                "Model state is not strict-compatible: "
                f"missing={missing}, unexpected={unexpected}."
            )


@dataclass(frozen=True, slots=True)
class CheckpointSaveReport:
    checkpoint_path: str
    optimizer_step: int
    epoch: int
    committed_microbatches: int
    asynchronous: bool
    saved_optimizer: bool
    saved_scheduler: bool
    saved_rng_state: bool
    exact_resume_capable: bool
    duration_seconds: float
    completed_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class CheckpointResumeReport:
    checkpoint_path: str
    optimizer_step: int
    epoch: int
    committed_microbatches: int
    restored_optimizer: bool
    restored_scheduler: bool
    restored_rng_state: bool
    saved_world_size: int
    current_world_size: int
    saved_strategy: str
    current_strategy: str
    loaded_at_utc: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _wait_async_result(result: Any, *, staging_only: bool = False) -> None:
    """Handle PyTorch 2.6 Future and newer AsyncSaveResponse objects."""

    staging = getattr(result, "staging_completion", None)
    upload = getattr(result, "upload_completion", None)
    if staging_only and staging is not None:
        staging.result()
        return
    if not staging_only and upload is not None:
        upload.result()
        return
    if staging_only:
        # In the PyTorch 2.6 Future API, CPU staging is completed before the
        # Future is returned; waiting here would make the whole save sync.
        return
    result_method = getattr(result, "result", None)
    if not callable(result_method):
        raise CheckpointError(
            "torch.distributed.checkpoint.async_save returned an unsupported "
            f"object: {type(result).__name__}."
        )
    result_method()


@dataclass(slots=True)
class PendingCheckpoint:
    """One asynchronous checkpoint whose upload may still be running."""

    manager: "CheckpointManager"
    temporary_path: Path
    final_path: Path
    optimizer_step: int
    epoch: int
    committed_microbatches: int
    started_at: float
    asynchronous: bool
    async_result: Any | None
    finalized: bool = False
    report: CheckpointSaveReport | None = None

    def wait(self) -> CheckpointSaveReport:
        if self.finalized:
            if self.report is None:
                raise CheckpointError("Finalized checkpoint has no save report.")
            return self.report
        self.report = self.manager._finalize_pending(self)
        self.finalized = True
        return self.report


class CheckpointManager:
    """Own checkpoint lifecycle for one distributed training process."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        runtime: RuntimeContext,
        distributed_model: DistributedM3DModel,
        optimizer: Optimizer,
        scheduler: CosineWarmupScheduler,
        data_pipeline: "TrainingDataPipeline",
    ) -> None:
        if not isinstance(config, ExperimentConfig):
            raise TypeError("config must be ExperimentConfig.")
        if not isinstance(runtime, RuntimeContext):
            raise TypeError("runtime must be RuntimeContext.")
        if not isinstance(distributed_model, DistributedM3DModel):
            raise TypeError("distributed_model must be DistributedM3DModel.")
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be torch.optim.Optimizer.")
        if not isinstance(scheduler, CosineWarmupScheduler):
            raise TypeError("scheduler must be CosineWarmupScheduler.")
        required_pipeline_attributes = (
            "epoch",
            "committed_step",
            "steps_per_epoch",
            "state_dict",
            "load_state_dict",
        )
        missing_pipeline_attributes = [
            name for name in required_pipeline_attributes if not hasattr(data_pipeline, name)
        ]
        if missing_pipeline_attributes:
            raise TypeError(
                "data_pipeline does not satisfy the TrainingDataPipeline contract; "
                f"missing={missing_pipeline_attributes}."
            )
        if config.checkpoint.format != "distributed_checkpoint":
            raise CheckpointError(
                f"Unsupported checkpoint format {config.checkpoint.format!r}."
            )

        self.config = config
        self.settings: CheckpointConfig = config.checkpoint
        self.runtime = runtime
        self.distributed_model = distributed_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_pipeline = data_pipeline
        self.output_dir = Path(self.settings.output_dir).expanduser().resolve()
        self._pending: PendingCheckpoint | None = None
        self._compatibility_sha256 = training_compatibility_sha256(config)
        self._async_process_group: Any | None = None
        self._async_fallback_reason: str | None = None

        validate_optimizer_parameter_coverage(
            optimizer,
            distributed_model.unwrapped_model,
        )
        optimizer_groups_by_name(optimizer)
        self._prepare_output_directory()
        self._prepare_async_process_group()

    @property
    def pending(self) -> PendingCheckpoint | None:
        return self._pending

    @property
    def exact_resume_capable(self) -> bool:
        return bool(
            self.settings.save_optimizer
            and self.settings.save_scheduler
            and self.settings.save_rng_state
        )

    @property
    def _state_dict_model(self) -> nn.Module:
        """Return the canonical model passed to PyTorch DCP state-dict APIs.

        FSDP2 mutates the original M3D model in place, so its sharded module is
        already ``unwrapped_model``. DDP, however, uses a reducer-only forward
        adapter whose extra ``model.`` namespace must never become part of
        durable model or optimizer FQNs.
        """

        if self.distributed_model.is_ddp:
            return self.distributed_model.unwrapped_model
        return self.distributed_model.wrapped_model

    def _prepare_output_directory(self) -> None:
        if self.runtime.is_main_process:
            self.output_dir.mkdir(parents=True, exist_ok=True)
        self.runtime.barrier()
        if not self.output_dir.is_dir():
            raise CheckpointError(
                f"Checkpoint output directory is unavailable: {self.output_dir}"
            )

    def _prepare_async_process_group(self) -> None:
        """Create the CPU/Gloo group required by PyTorch 2.6 async DCP.

        The training default group is NCCL-only. PyTorch 2.6 ``async_save``
        stages tensors to CPU and explicitly requires a process group with a CPU
        backend. A separate all-rank Gloo group keeps training collectives on
        NCCL while enabling asynchronous checkpoint upload.
        """

        if not self.settings.asynchronous or not self.runtime.process_group_initialized:
            return
        local_error: str | None = None
        try:
            self._async_process_group = dist.new_group(
                ranks=list(range(self.runtime.world_size)),
                backend="gloo",
            )
        except Exception as exc:  # pragma: no cover - cluster dependent
            local_error = f"{type(exc).__name__}: {exc}"
            self._async_process_group = None
        gathered = self.runtime.all_gather_object(local_error)
        failures = [item for item in gathered if item is not None]
        if failures:
            self._async_fallback_reason = (
                "Could not create the all-rank Gloo checkpoint group; "
                f"falling back to synchronous DCP: {failures}"
            )
            self.runtime.logger.warning(self._async_fallback_reason)
            self._async_process_group = None
        self.runtime.barrier()

    def _validate_boundary(self) -> None:
        try:
            self.scheduler.validate_training_position(
                epoch=self.data_pipeline.epoch,
                committed_microbatches=self.data_pipeline.committed_step,
                require_update_boundary=True,
            )
        except Exception as exc:
            raise CheckpointBoundaryError(
                "Checkpoint may only be saved after a complete optimizer update "
                "and committed sampler step."
            ) from exc

        gradients = [
            name
            for name, parameter in self.distributed_model.unwrapped_model.named_parameters()
            if parameter.grad is not None
        ]
        if gradients:
            preview = gradients[:8]
            raise CheckpointBoundaryError(
                "Checkpoint was requested before optimizer.zero_grad(set_to_none=True); "
                f"parameters still holding gradients include {preview}."
            )

        expected_step = self.scheduler.completed_optimizer_steps
        self.runtime.assert_all_ranks_equal(
            (
                expected_step,
                self.data_pipeline.epoch,
                self.data_pipeline.committed_step,
            ),
            label="checkpoint training cursor",
        )

    def _new_paths(self, optimizer_step: int) -> tuple[Path, Path]:
        final_name = f"checkpoint-step-{optimizer_step:08d}"
        temporary_name: str | None = None
        if self.runtime.is_main_process:
            temporary_name = f".{final_name}.incomplete-{uuid.uuid4().hex}"
        temporary_name = self.runtime.broadcast_object(temporary_name)
        final_path = self.output_dir / final_name
        temporary_path = self.output_dir / temporary_name

        if self.runtime.is_main_process:
            if final_path.exists():
                raise CheckpointError(
                    f"Refusing to overwrite completed checkpoint {final_path}."
                )
            temporary_path.mkdir(parents=False, exist_ok=False)
            (temporary_path / _DCP_DIR).mkdir()
            (temporary_path / _RANK_STATE_DIR).mkdir()
        self.runtime.barrier()
        return temporary_path, final_path

    def _common_state(self) -> dict[str, Any]:
        scheduler_state = (
            self.scheduler.state_dict() if self.settings.save_scheduler else None
        )
        data_state = self.data_pipeline.state_dict()
        optimizer_layout = self.scheduler.optimizer_group_layout_sha256
        common = {
            "state_version": _CHECKPOINT_STATE_VERSION,
            "experiment_name": self.config.experiment_name,
            "created_at_utc": _utc_now(),
            "host": socket.gethostname(),
            "torch_version": torch.__version__,
            "saved_world_size": self.runtime.world_size,
            "saved_strategy": self.distributed_model.strategy,
            "optimizer_step": self.scheduler.completed_optimizer_steps,
            "epoch": self.data_pipeline.epoch,
            "committed_microbatches": self.data_pipeline.committed_step,
            "steps_per_epoch": self.data_pipeline.steps_per_epoch,
            "training_compatibility_sha256": self._compatibility_sha256,
            "optimizer_group_layout_sha256": optimizer_layout,
            "lazy_optimizer_parameter_names": (
                self._lazy_optimizer_parameter_names()
                if self.settings.save_optimizer
                else []
            ),
            "scheduler": scheduler_state,
            "data_pipeline": data_state,
            "save_options": {
                "optimizer": bool(self.settings.save_optimizer),
                "scheduler": bool(self.settings.save_scheduler),
                "rng_state": bool(self.settings.save_rng_state),
                "asynchronous_requested": bool(self.settings.asynchronous),
                "asynchronous_available": bool(
                    not self.runtime.process_group_initialized
                    or self._async_process_group is not None
                ),
                "asynchronous_fallback_reason": self._async_fallback_reason,
            },
            "exact_resume_capable": self.exact_resume_capable,
        }
        return common

    def _optimizer_parameters_by_name(self) -> dict[str, nn.Parameter]:
        result: dict[str, nn.Parameter] = {}
        for group in self.optimizer.param_groups:
            names = group.get("param_names")
            parameters = group.get("params")
            if not isinstance(names, list) or not isinstance(parameters, list):
                raise CheckpointCompatibilityError(
                    "Optimizer groups must retain param_names metadata."
                )
            if len(names) != len(parameters):
                raise CheckpointCompatibilityError(
                    "Optimizer param_names and params lengths differ."
                )
            for name, parameter in zip(names, parameters, strict=True):
                if not isinstance(name, str) or not isinstance(parameter, nn.Parameter):
                    raise CheckpointCompatibilityError(
                        "Optimizer parameter metadata is malformed."
                    )
                if name in result and result[name] is not parameter:
                    raise CheckpointCompatibilityError(
                        f"Optimizer parameter name {name!r} is duplicated."
                    )
                result[name] = parameter
        return result

    def _lazy_optimizer_parameter_names(self) -> list[str]:
        by_name = self._optimizer_parameters_by_name()
        return sorted(
            name
            for name, parameter in by_name.items()
            if not self.optimizer.state.get(parameter)
        )

    def _write_sidecars(self, temporary_path: Path) -> dict[str, Any]:
        # Rank 0 creates timestamp/host metadata once, then every rank uses the
        # exact same object. Computing those fields independently would make a
        # valid multi-node checkpoint fail its cross-rank fingerprint check.
        common_local = self._common_state() if self.runtime.is_main_process else None
        common = self.runtime.broadcast_object(common_local)
        if not isinstance(common, dict):
            raise CheckpointError("Broadcast checkpoint common state is not a dictionary.")
        common_hash = _sha256_payload(common)
        self.runtime.assert_all_ranks_equal(
            common_hash,
            label="checkpoint common-state fingerprint",
        )

        rank_payload: dict[str, Any] = {
            "state_version": _CHECKPOINT_STATE_VERSION,
            "rank": self.runtime.rank,
            "local_rank": self.runtime.local_rank,
            "world_size": self.runtime.world_size,
            "optimizer_step": self.scheduler.completed_optimizer_steps,
            "common_state_sha256": common_hash,
            "rng_state": (
                RNGSnapshot.capture(self.runtime.device).to_dict()
                if self.settings.save_rng_state
                else None
            ),
        }
        rank_path = (
            temporary_path
            / _RANK_STATE_DIR
            / f"rank-{self.runtime.rank:05d}.json"
        )
        atomic_write_json(rank_path, rank_payload)

        if self.runtime.is_main_process:
            atomic_write_json(temporary_path / _TRAINER_STATE_FILE, common)
            atomic_write_json(temporary_path / _CONFIG_FILE, self.config.to_dict())
        self.runtime.barrier()
        return common

    def save(
        self,
        *,
        force_synchronous: bool = False,
    ) -> PendingCheckpoint:
        """Start one durable checkpoint at the current optimizer boundary.

        At most one upload is allowed in flight.  Starting another checkpoint
        first finalizes the previous one to prevent unbounded CPU staging memory.
        """

        self.wait_for_pending()
        self._validate_boundary()
        optimizer_step = self.scheduler.completed_optimizer_steps
        temporary_path, final_path = self._new_paths(optimizer_step)
        self._write_sidecars(temporary_path)

        app_state = _ModelOptimizerState(
            self._state_dict_model,
            self.optimizer if self.settings.save_optimizer else None,
        )
        state: dict[str, Any] = {"application": app_state}
        started = time.monotonic()
        use_async = bool(
            self.settings.asynchronous
            and not force_synchronous
            and (
                not self.runtime.process_group_initialized
                or self._async_process_group is not None
            )
        )
        async_result: Any | None = None
        try:
            if use_async:
                async_result = dcp.async_save(
                    state,
                    checkpoint_id=temporary_path / _DCP_DIR,
                    process_group=self._async_process_group,
                )
                # Newer DCP may return before staging completes.  Training must
                # not mutate parameters until the immutable CPU snapshot exists.
                _wait_async_result(async_result, staging_only=True)
            else:
                dcp.save(
                    state,
                    checkpoint_id=temporary_path / _DCP_DIR,
                )
        except Exception as exc:
            if self.runtime.is_main_process:
                atomic_write_json(
                    temporary_path / "FAILED.json",
                    {
                        "failed_at_utc": _utc_now(),
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
            raise CheckpointError(
                f"Distributed checkpoint save failed at step {optimizer_step}."
            ) from exc

        pending = PendingCheckpoint(
            manager=self,
            temporary_path=temporary_path,
            final_path=final_path,
            optimizer_step=optimizer_step,
            epoch=self.data_pipeline.epoch,
            committed_microbatches=self.data_pipeline.committed_step,
            started_at=started,
            asynchronous=use_async,
            async_result=async_result,
        )
        self._pending = pending
        if not use_async:
            pending.wait()
        return pending

    def _validate_completed_files(self, temporary_path: Path) -> None:
        required = [
            temporary_path / _TRAINER_STATE_FILE,
            temporary_path / _CONFIG_FILE,
            temporary_path / _DCP_DIR,
        ]
        required.extend(
            temporary_path
            / _RANK_STATE_DIR
            / f"rank-{rank:05d}.json"
            for rank in range(self.runtime.world_size)
        )
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise CheckpointError(
                "Checkpoint upload finished but required files are missing: "
                + ", ".join(missing)
            )
        dcp_entries = list((temporary_path / _DCP_DIR).iterdir())
        if not dcp_entries:
            raise CheckpointError("DCP directory is empty after save completion.")

    def _finalize_pending(self, pending: PendingCheckpoint) -> CheckpointSaveReport:
        local_error: str | None = None
        try:
            if pending.async_result is not None:
                _wait_async_result(pending.async_result, staging_only=False)
        except Exception as exc:
            local_error = f"{type(exc).__name__}: {exc}"

        errors = self.runtime.all_gather_object(local_error)
        failures = {
            rank: error for rank, error in enumerate(errors) if error is not None
        }
        if failures:
            if self.runtime.is_main_process:
                atomic_write_json(
                    pending.temporary_path / "FAILED.json",
                    {
                        "failed_at_utc": _utc_now(),
                        "rank_errors": failures,
                    },
                )
            self.runtime.barrier()
            raise CheckpointError(
                f"Asynchronous checkpoint failed on ranks: {failures}."
            )

        self.runtime.barrier()
        duration = time.monotonic() - pending.started_at
        report = CheckpointSaveReport(
            checkpoint_path=str(pending.final_path),
            optimizer_step=pending.optimizer_step,
            epoch=pending.epoch,
            committed_microbatches=pending.committed_microbatches,
            asynchronous=pending.asynchronous,
            saved_optimizer=bool(self.settings.save_optimizer),
            saved_scheduler=bool(self.settings.save_scheduler),
            saved_rng_state=bool(self.settings.save_rng_state),
            exact_resume_capable=self.exact_resume_capable,
            duration_seconds=float(duration),
            completed_at_utc=_utc_now(),
        )

        finalize_error: str | None = None
        if self.runtime.is_main_process:
            try:
                self._validate_completed_files(pending.temporary_path)
                atomic_write_json(
                    pending.temporary_path / _COMPLETION_FILE,
                    {
                        "state_version": _CHECKPOINT_STATE_VERSION,
                        "status": "complete",
                        "report": report.to_dict(),
                    },
                )
                os.replace(pending.temporary_path, pending.final_path)
                atomic_write_json(
                    self.output_dir / _LATEST_FILE,
                    {
                        "state_version": _CHECKPOINT_STATE_VERSION,
                        "checkpoint": pending.final_path.name,
                        "optimizer_step": pending.optimizer_step,
                        "updated_at_utc": _utc_now(),
                    },
                )
                self._prune_completed_checkpoints()
            except Exception as exc:
                finalize_error = f"{type(exc).__name__}: {exc}"
                failure_path = (
                    pending.temporary_path
                    if pending.temporary_path.exists()
                    else pending.final_path
                )
                if failure_path.exists():
                    atomic_write_json(
                        failure_path / "FAILED.json",
                        {
                            "failed_at_utc": _utc_now(),
                            "error": finalize_error,
                        },
                    )
        finalize_payload = self.runtime.broadcast_object(
            {"error": finalize_error} if self.runtime.is_main_process else None
        )
        if not isinstance(finalize_payload, Mapping):
            raise CheckpointError("Checkpoint finalization status broadcast is invalid.")
        observed_finalize_error = finalize_payload.get("error")
        if observed_finalize_error is not None:
            raise CheckpointError(
                "Checkpoint files were uploaded but atomic finalization failed: "
                f"{observed_finalize_error}"
            )
        self.runtime.barrier()

        if self._pending is pending:
            self._pending = None
        return report

    def wait_for_pending(self) -> CheckpointSaveReport | None:
        if self._pending is None:
            return None
        return self._pending.wait()

    def close(self) -> CheckpointSaveReport | None:
        """Finish uploads and release the auxiliary Gloo process group."""

        report = self.wait_for_pending()
        if self._async_process_group is not None:
            self.runtime.barrier()
            dist.destroy_process_group(self._async_process_group)
            self._async_process_group = None
        return report

    def _prune_completed_checkpoints(self) -> None:
        completed = list_completed_checkpoints(self.output_dir)
        excess = len(completed) - int(self.settings.keep_last_n)
        if excess <= 0:
            return
        for path in completed[:excess]:
            shutil.rmtree(path)

    def resolve_resume_path(self, value: str | os.PathLike[str] | None = None) -> Path:
        if (
            value is not None
            and str(value).strip() == LATEST_CHECKPOINT_SENTINEL
        ):
            value = None
        return resolve_checkpoint_path(
            self.output_dir if value is None else value,
        )

    def load(
        self,
        checkpoint: str | os.PathLike[str] | None = None,
        *,
        exact: bool = True,
    ) -> CheckpointResumeReport:
        """Load a completed checkpoint into already wrapped model/optimizer.

        Exact resume restores optimizer, scheduler, data cursor, and rank-local
        RNG state.  Model-only warm-start is intentionally not conflated with
        training resume; set ``exact=False`` only for controlled diagnostics.
        """

        self.wait_for_pending()
        path = self.resolve_resume_path(
            self.settings.resume_from if checkpoint is None else checkpoint
        )
        completion = _read_json(path / _COMPLETION_FILE)
        if completion.get("status") != "complete":
            raise CheckpointCompatibilityError(
                f"Checkpoint is not marked complete: {path}"
            )
        common = _read_json(path / _TRAINER_STATE_FILE)
        self._validate_common_state(common, exact=exact)
        save_options = _as_mapping(common.get("save_options"), name="save_options")
        has_optimizer = bool(save_options.get("optimizer"))
        has_scheduler = bool(save_options.get("scheduler"))
        has_rng = bool(save_options.get("rng_state"))

        if exact and not (has_optimizer and has_scheduler and has_rng):
            raise CheckpointCompatibilityError(
                "Checkpoint is not exact-resume capable: "
                f"optimizer={has_optimizer}, scheduler={has_scheduler}, rng={has_rng}."
            )

        lazy_names_raw = common.get("lazy_optimizer_parameter_names")
        if has_optimizer and not isinstance(lazy_names_raw, list):
            raise CheckpointCompatibilityError(
                "Checkpoint is missing lazy optimizer parameter metadata."
            )
        lazy_names = (
            []
            if not has_optimizer
            else [str(item) for item in cast(list[Any], lazy_names_raw)]
        )
        if len(lazy_names) != len(set(lazy_names)) or any(
            not item for item in lazy_names
        ):
            raise CheckpointCompatibilityError(
                "Checkpoint lazy optimizer parameter metadata is malformed."
            )
        optimizer_parameters = self._optimizer_parameters_by_name()
        unknown_lazy_names = sorted(set(lazy_names) - set(optimizer_parameters))
        if unknown_lazy_names:
            raise CheckpointCompatibilityError(
                "Checkpoint names optimizer parameters absent from this run: "
                f"{unknown_lazy_names[:20]}."
            )

        app_state = _ModelOptimizerState(
            self._state_dict_model,
            self.optimizer if has_optimizer else None,
        )
        try:
            dcp.load(
                {"application": app_state},
                checkpoint_id=path / _DCP_DIR,
                planner=dcp.DefaultLoadPlanner(allow_partial_load=True),
            )
        except Exception as exc:
            raise CheckpointCompatibilityError(
                f"DCP model/optimizer load failed from {path}."
            ) from exc

        if has_optimizer:
            # PyTorch 2.6 initializes zero Adam states for every destination
            # parameter before planning a load. Remove states that were still
            # lazy when saved so their future first update retains step=1
            # semantics instead of inheriting a fabricated zero step.
            for name in lazy_names:
                self.optimizer.state.pop(optimizer_parameters[name], None)
            restore_optimizer_group_metadata(self.optimizer)
            validate_optimizer_parameter_coverage(
                self.optimizer,
                self.distributed_model.unwrapped_model,
            )
        elif exact:
            raise CheckpointCompatibilityError(
                "Exact resume requires optimizer state."
            )

        scheduler_state = common.get("scheduler")
        if has_scheduler:
            if not isinstance(scheduler_state, Mapping):
                raise CheckpointCompatibilityError(
                    "Checkpoint declares scheduler state but none is present."
                )
            self.scheduler.load_state_dict(scheduler_state)
        elif exact:
            raise CheckpointCompatibilityError(
                "Exact resume requires scheduler state."
            )

        data_state = common.get("data_pipeline")
        if not isinstance(data_state, Mapping):
            raise CheckpointCompatibilityError(
                "Checkpoint has no training data-pipeline state."
            )
        self.data_pipeline.load_state_dict(data_state)
        validate_resume_progress(
            self.scheduler,
            epoch=self.data_pipeline.epoch,
            committed_microbatches=self.data_pipeline.committed_step,
        )

        restored_rng = False
        if has_rng:
            rank_payload = _read_json(
                path
                / _RANK_STATE_DIR
                / f"rank-{self.runtime.rank:05d}.json"
            )
            self._validate_rank_state(rank_payload, common)
            rng_payload = rank_payload.get("rng_state")
            if not isinstance(rng_payload, Mapping):
                raise CheckpointCompatibilityError(
                    "Checkpoint declares RNG state but rank sidecar has none."
                )
            RNGSnapshot.from_dict(rng_payload).restore(self.runtime.device)
            restored_rng = True
        elif exact:
            raise CheckpointCompatibilityError("Exact resume requires RNG state.")

        self.optimizer.zero_grad(set_to_none=True)
        self.runtime.barrier()
        return CheckpointResumeReport(
            checkpoint_path=str(path),
            optimizer_step=int(common["optimizer_step"]),
            epoch=int(common["epoch"]),
            committed_microbatches=int(common["committed_microbatches"]),
            restored_optimizer=has_optimizer,
            restored_scheduler=has_scheduler,
            restored_rng_state=restored_rng,
            saved_world_size=int(common["saved_world_size"]),
            current_world_size=self.runtime.world_size,
            saved_strategy=str(common["saved_strategy"]),
            current_strategy=self.distributed_model.strategy,
            loaded_at_utc=_utc_now(),
        )

    def _validate_common_state(
        self,
        common: Mapping[str, Any],
        *,
        exact: bool,
    ) -> None:
        if common.get("state_version") != _CHECKPOINT_STATE_VERSION:
            raise CheckpointCompatibilityError(
                "Unsupported checkpoint state version "
                f"{common.get('state_version')!r}."
            )
        saved_torch_version = common.get("torch_version")
        if not isinstance(saved_torch_version, str):
            raise CheckpointCompatibilityError("Checkpoint torch_version is invalid.")
        if exact and _torch_major_minor(saved_torch_version) != _torch_major_minor(
            torch.__version__
        ):
            raise CheckpointCompatibilityError(
                "Exact DCP resume requires the same PyTorch major/minor version: "
                f"checkpoint={saved_torch_version!r}, current={torch.__version__!r}."
            )

        saved_world = common.get("saved_world_size")
        if not isinstance(saved_world, int) or isinstance(saved_world, bool):
            raise CheckpointCompatibilityError("saved_world_size is invalid.")
        # DCP itself can reshard across world sizes, but the current task sampler
        # promises exact sample order and intentionally rejects topology changes.
        if exact and saved_world != self.runtime.world_size:
            raise CheckpointCompatibilityError(
                "Exact M3D resume currently requires the same world size because "
                "the task-balanced sampler's global-batch schedule depends on it: "
                f"checkpoint={saved_world}, current={self.runtime.world_size}."
            )
        checkpoint_compat = common.get("training_compatibility_sha256")
        if checkpoint_compat != self._compatibility_sha256:
            raise CheckpointCompatibilityError(
                "Training configuration changed since checkpoint: "
                f"checkpoint={checkpoint_compat!r}, "
                f"current={self._compatibility_sha256!r}."
            )
        checkpoint_layout = common.get("optimizer_group_layout_sha256")
        if checkpoint_layout != self.scheduler.optimizer_group_layout_sha256:
            raise CheckpointCompatibilityError(
                "Optimizer parameter-group layout changed: "
                f"checkpoint={checkpoint_layout!r}, "
                f"current={self.scheduler.optimizer_group_layout_sha256!r}."
            )
        for key in ("optimizer_step", "epoch", "committed_microbatches"):
            value = common.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise CheckpointCompatibilityError(
                    f"Checkpoint field {key!r} is invalid: {value!r}."
                )

    def _validate_rank_state(
        self,
        rank_state: Mapping[str, Any],
        common: Mapping[str, Any],
    ) -> None:
        expected = {
            "state_version": _CHECKPOINT_STATE_VERSION,
            "rank": self.runtime.rank,
            "world_size": self.runtime.world_size,
            "optimizer_step": common.get("optimizer_step"),
            "common_state_sha256": _sha256_payload(common),
        }
        mismatches = {
            key: {"checkpoint": rank_state.get(key), "current": value}
            for key, value in expected.items()
            if rank_state.get(key) != value
        }
        if mismatches:
            raise CheckpointCompatibilityError(
                f"Rank-local checkpoint sidecar is incompatible: {mismatches}."
            )


def list_completed_checkpoints(output_dir: str | os.PathLike[str]) -> list[Path]:
    root = Path(output_dir).expanduser().resolve()
    if not root.exists():
        return []
    candidates: list[tuple[int, Path]] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        match = _CHECKPOINT_PATTERN.fullmatch(child.name)
        if match is None or not (child / _COMPLETION_FILE).is_file():
            continue
        candidates.append((int(match.group(1)), child))
    candidates.sort(key=lambda item: item[0])
    return [path for _, path in candidates]


def resolve_checkpoint_path(
    value: str | os.PathLike[str],
) -> Path:
    """Resolve an explicit checkpoint directory or an output directory's latest."""

    path = Path(value).expanduser().resolve()
    if path.is_dir() and (path / _COMPLETION_FILE).is_file():
        return path
    if path.is_file() and path.name == _LATEST_FILE:
        root = path.parent
        latest = _read_json(path)
    elif path.is_dir():
        root = path
        latest_path = root / _LATEST_FILE
        if latest_path.is_file():
            latest = _read_json(latest_path)
        else:
            completed = list_completed_checkpoints(root)
            if not completed:
                raise CheckpointCompatibilityError(
                    f"No completed checkpoints found under {root}."
                )
            return completed[-1]
    else:
        raise CheckpointCompatibilityError(
            f"Checkpoint path does not exist or is incomplete: {path}"
        )

    name = latest.get("checkpoint")
    if not isinstance(name, str) or not name:
        raise CheckpointCompatibilityError(
            f"Latest checkpoint pointer is malformed: {root / _LATEST_FILE}"
        )
    resolved = (root / name).resolve()
    if resolved.parent != root.resolve():
        raise CheckpointCompatibilityError(
            "Latest checkpoint pointer escapes the output directory."
        )
    if not (resolved / _COMPLETION_FILE).is_file():
        raise CheckpointCompatibilityError(
            f"Latest pointer targets an incomplete checkpoint: {resolved}"
        )
    return resolved


def should_save_checkpoint(
    scheduler: CosineWarmupScheduler,
    settings: CheckpointConfig,
    *,
    final_update: bool = False,
) -> bool:
    """Return whether the current completed optimizer step should be persisted."""

    step = scheduler.completed_optimizer_steps
    if step <= 0:
        return False
    if final_update or scheduler.is_finished:
        return True
    return step % int(settings.save_every_steps) == 0


# ---------------------------------------------------------------------------
# Dependency-light self-test
# ---------------------------------------------------------------------------


def _dcp_roundtrip_test(root: Path) -> bool:
    torch.manual_seed(17)

    class ConditionalModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.active = nn.Sequential(
                nn.Linear(4, 8),
                nn.GELU(),
                nn.Linear(8, 2),
            )
            self.lazy_branch = nn.Linear(4, 2)

        def forward(self, value: Tensor) -> Tensor:
            return self.active(value)

    model = ConditionalModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    x = torch.randn(3, 4)
    model(x).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    expected = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}

    save_dir = root / "dcp-roundtrip"
    dcp.save(
        {"application": _ModelOptimizerState(model, optimizer)},
        checkpoint_id=save_dir,
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(10.0)
    dcp.load(
        {"application": _ModelOptimizerState(model, optimizer)},
        checkpoint_id=save_dir,
    )
    return all(
        torch.equal(model.state_dict()[name], tensor)
        for name, tensor in expected.items()
    )


def _rng_roundtrip_test() -> bool:
    random.seed(101)
    np.random.seed(102)
    torch.manual_seed(103)
    snapshot = RNGSnapshot.capture(torch.device("cpu"))
    expected = (
        random.random(),
        float(np.random.rand()),
        torch.rand(4),
    )
    random.random()
    np.random.rand()
    torch.rand(7)
    RNGSnapshot.from_dict(snapshot.to_dict()).restore(torch.device("cpu"))
    observed = (
        random.random(),
        float(np.random.rand()),
        torch.rand(4),
    )
    return (
        expected[0] == observed[0]
        and expected[1] == observed[1]
        and torch.equal(expected[2], observed[2])
    )


def run_self_test() -> Mapping[str, Any]:
    import tempfile

    with tempfile.TemporaryDirectory(prefix="m3d-checkpoint-test-") as temporary:
        root = Path(temporary)
        dcp_roundtrip = _dcp_roundtrip_test(root)
        if not dcp_roundtrip:
            raise AssertionError("DCP model/optimizer roundtrip failed.")
        rng_roundtrip = _rng_roundtrip_test()
        if not rng_roundtrip:
            raise AssertionError("RNG JSON roundtrip failed.")

        completed_steps = (3, 11, 27)
        for step in completed_steps:
            path = root / f"checkpoint-step-{step:08d}"
            path.mkdir()
            atomic_write_json(
                path / _COMPLETION_FILE,
                {"state_version": 1, "status": "complete"},
            )
        incomplete = root / ".checkpoint-step-00000099.incomplete-deadbeef"
        incomplete.mkdir()
        discovered = list_completed_checkpoints(root)
        if [path.name for path in discovered] != [
            "checkpoint-step-00000003",
            "checkpoint-step-00000011",
            "checkpoint-step-00000027",
        ]:
            raise AssertionError("Completed checkpoint discovery is incorrect.")
        atomic_write_json(
            root / _LATEST_FILE,
            {
                "state_version": 1,
                "checkpoint": "checkpoint-step-00000027",
                "optimizer_step": 27,
            },
        )
        latest = resolve_checkpoint_path(root)
        if latest.name != "checkpoint-step-00000027":
            raise AssertionError("Latest checkpoint resolution failed.")

        return {
            "status": "passed",
            "dcp_model_optimizer_roundtrip": dcp_roundtrip,
            "rng_roundtrip": rng_roundtrip,
            "completed_checkpoint_count": len(discovered),
            "incomplete_checkpoint_ignored": incomplete not in discovered,
            "latest_checkpoint": latest.name,
            "checkpoint_state_version": _CHECKPOINT_STATE_VERSION,
        }


def main() -> None:
    print(json.dumps(run_self_test(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
