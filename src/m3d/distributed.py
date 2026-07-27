"""Distributed model preparation for M3D-Modernized.

This module runs after the complete :class:`m3d.model.m3d.M3DModel` has been
constructed and all published component checkpoints have been loaded, but
before optimizer creation.

Two execution strategies are supported:

* ``ddp`` is the primary high-throughput path when one complete model replica
  fits on each A100;
* ``fsdp2`` is the memory fallback and shards parameters, gradients, and
  optimizer state across the data-parallel mesh.

The complete M3D model has task-dependent control flow: text tasks skip the
SegVol branch while segmentation tasks execute it.  The wrapper therefore
keeps DDP ``find_unused_parameters=True`` and ``static_graph=False``.  FSDP2 is
applied bottom-up so the Main ViT, Phi-3 decoder layers, independent SegVol ViT,
and SegVol decoder form separate communication groups instead of one enormous
root all-gather.

Important construction order::

    with synchronized_model_initialization(runtime):
        model, build_report = build_m3d_model(...)

    distributed_model, distributed_report = prepare_distributed_model(
        model,
        runtime,
    )

    optimizer = build_optimizer(distributed_model.unwrapped_model, ...)

For FSDP2, the common model-initialization context is mandatory.  The runtime
uses rank-offset random seeds for data augmentation, so constructing random
model components directly under those seeds would give every rank a different
unsharded model before FSDP2 slices it.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import random
import weakref
from collections.abc import Callable, Generator, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, Literal, TypeVar, cast

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel

from .config import ExperimentConfig
from .data.schema import M3DBatch, TaskName
from .model.m3d import M3DModel, M3DModelOutput
from .runtime import RuntimeContext


DistributedStrategy = Literal["ddp", "fsdp2"]
T = TypeVar("T")
BuildT = TypeVar("BuildT")


class DistributedModelError(RuntimeError):
    """Raised when model placement, wrapping, or rank state is inconsistent."""


@dataclass(frozen=True, slots=True)
class ParameterLayoutSummary:
    """Stable pre-wrap description shared by every rank."""

    parameter_count: int
    trainable_parameter_count: int
    parameter_tensor_count: int
    trainable_tensor_count: int
    persistent_buffer_count: int
    layout_sha256: str

    def to_dict(self) -> dict[str, int | str]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class FSDPWrapGroup:
    """One bottom-up FSDP2 communication group."""

    name: str
    module_type: str
    parameter_tensor_count: int
    parameter_element_count: int
    trainable_parameter_count: int
    reshard_after_forward: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class DistributedModelReport:
    """Serializable report produced before optimizer construction."""

    strategy: DistributedStrategy
    world_size: int
    rank: int
    local_rank: int
    device: str
    model_type: str
    parameter_layout: ParameterLayoutSummary
    wrapped_parameter_tensor_count: int
    wrapped_trainable_tensor_count: int
    ddp_find_unused_parameters: bool | None
    ddp_gradient_as_bucket_view: bool | None
    ddp_static_graph: bool | None
    ddp_bucket_cap_mb: int | None
    ddp_broadcast_buffers_each_forward: bool | None
    persistent_buffers_synchronised_once: int
    fsdp_groups: tuple[FSDPWrapGroup, ...]
    fsdp_mixed_precision: str | None
    fsdp_reduce_dtype: str | None
    fsdp_cpu_offload: bool | None
    requires_even_evaluation_steps: bool
    synchronous_initialization_seed: int | None

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["fsdp_groups"] = [group.to_dict() for group in self.fsdp_groups]
        return result


@dataclass(frozen=True, slots=True)
class _NamedModule:
    name: str
    module: nn.Module


@dataclass(frozen=True, slots=True)
class _PersistentBuffer:
    name: str
    tensor: Tensor


class _ModelBuildMarker:
    """Private immutable marker attached after common-seed construction."""

    __slots__ = ("seed", "world_size")

    def __init__(self, seed: int, world_size: int) -> None:
        self.seed = int(seed)
        self.world_size = int(world_size)


_MODEL_BUILD_MARKER = "_m3d_synchronous_model_build"


def _normalise_seed(seed: int) -> int:
    if seed < 0:
        raise DistributedModelError(f"Model initialization seed must be >= 0, got {seed}.")
    return int(seed) % (2**32)


@contextlib.contextmanager
def synchronized_model_initialization(
    runtime: RuntimeContext,
    *,
    seed: int | None = None,
) -> Generator[None, None, None]:
    """Temporarily give every rank the same model-construction RNG stream.

    The process-wide rank-offset RNG streams are restored on exit, so data
    augmentation and dropout continue to use process-specific randomness.
    """

    common_seed = _normalise_seed(
        runtime.config.runtime.seed if seed is None else int(seed)
    )
    python_state = random.getstate()
    numpy_state = np.random.get_state()

    cuda_devices: list[int] = []
    if runtime.device.type == "cuda" and torch.cuda.is_available():
        if runtime.device.index is None:
            raise DistributedModelError("CUDA runtime device must have an explicit index.")
        cuda_devices = [int(runtime.device.index)]

    try:
        with torch.random.fork_rng(devices=cuda_devices, enabled=True):
            random.seed(common_seed)
            np.random.seed(common_seed)
            torch.manual_seed(common_seed)
            if cuda_devices:
                torch.cuda.manual_seed(common_seed)
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def _extract_built_model(value: Any) -> M3DModel:
    if isinstance(value, M3DModel):
        return value
    if isinstance(value, tuple) and value and isinstance(value[0], M3DModel):
        return value[0]
    raise DistributedModelError(
        "The synchronized builder must return M3DModel or a tuple whose first "
        f"element is M3DModel, got {type(value).__name__}."
    )


def build_model_synchronously(
    runtime: RuntimeContext,
    builder: Callable[[], BuildT],
    *,
    seed: int | None = None,
) -> BuildT:
    """Run a model builder under a common seed and mark the resulting model."""

    if not callable(builder):
        raise TypeError("builder must be callable.")
    common_seed = _normalise_seed(
        runtime.config.runtime.seed if seed is None else int(seed)
    )
    with synchronized_model_initialization(runtime, seed=common_seed):
        result = builder()
    model = _extract_built_model(result)
    setattr(
        model,
        _MODEL_BUILD_MARKER,
        _ModelBuildMarker(common_seed, runtime.world_size),
    )
    return result


def _model_build_marker(model: M3DModel) -> _ModelBuildMarker | None:
    marker = getattr(model, _MODEL_BUILD_MARKER, None)
    return marker if isinstance(marker, _ModelBuildMarker) else None


def _iter_persistent_buffers(module: nn.Module) -> Iterator[_PersistentBuffer]:
    """Yield persistent buffers only, excluding runtime caches."""

    for module_name, child in module.named_modules():
        non_persistent = getattr(child, "_non_persistent_buffers_set", set())
        for buffer_name, tensor in child.named_buffers(recurse=False):
            if buffer_name in non_persistent:
                continue
            qualified = f"{module_name}.{buffer_name}" if module_name else buffer_name
            yield _PersistentBuffer(qualified, tensor)


def _layout_payload(model: M3DModel) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for name, parameter in model.named_parameters():
        payload.append(
            {
                "kind": "parameter",
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "requires_grad": bool(parameter.requires_grad),
            }
        )
    for item in _iter_persistent_buffers(model):
        payload.append(
            {
                "kind": "buffer",
                "name": item.name,
                "shape": list(item.tensor.shape),
                "dtype": str(item.tensor.dtype),
                "requires_grad": False,
            }
        )
    return payload


def parameter_layout_summary(model: M3DModel) -> ParameterLayoutSummary:
    payload = _layout_payload(model)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    parameters = list(model.parameters())
    buffers = list(_iter_persistent_buffers(model))
    return ParameterLayoutSummary(
        parameter_count=int(sum(parameter.numel() for parameter in parameters)),
        trainable_parameter_count=int(
            sum(parameter.numel() for parameter in parameters if parameter.requires_grad)
        ),
        parameter_tensor_count=len(parameters),
        trainable_tensor_count=sum(parameter.requires_grad for parameter in parameters),
        persistent_buffer_count=len(buffers),
        layout_sha256=hashlib.sha256(encoded).hexdigest(),
    )


def _assert_layout_equal_across_ranks(
    runtime: RuntimeContext,
    summary: ParameterLayoutSummary,
) -> None:
    runtime.assert_all_ranks_equal(
        summary.layout_sha256,
        label="pre-wrap model parameter layout",
    )


def _validate_parameter_devices_before_wrap(
    model: M3DModel,
    *,
    strategy: DistributedStrategy,
    device: torch.device,
) -> None:
    invalid_meta = [name for name, parameter in model.named_parameters() if parameter.is_meta]
    if invalid_meta:
        raise DistributedModelError(
            "Meta parameters remain before distributed wrapping: "
            + ", ".join(invalid_meta[:20])
        )

    devices = {parameter.device for parameter in model.parameters()}
    if strategy == "ddp":
        allowed = {torch.device("cpu"), device}
        unexpected = [item for item in devices if item not in allowed]
        if unexpected:
            raise DistributedModelError(
                "DDP model parameters must be on CPU or the process-local CUDA "
                f"device {device}; found {sorted(map(str, unexpected))}."
            )
    else:
        unexpected = [
            item
            for item in devices
            if item.type not in {"cpu", device.type}
            or (item.type == "cuda" and item != device)
        ]
        if unexpected:
            raise DistributedModelError(
                "FSDP2 input model has parameters on an unexpected device: "
                + ", ".join(sorted(map(str, unexpected)))
            )


def _synchronise_persistent_buffers_once(
    model: nn.Module,
    runtime: RuntimeContext,
) -> int:
    """Broadcast persistent buffers once so DDP can disable per-forward sync."""

    buffers = list(_iter_persistent_buffers(model))
    if not runtime.process_group_initialized:
        return len(buffers)
    with torch.no_grad():
        for item in buffers:
            if item.tensor.device != runtime.device:
                raise DistributedModelError(
                    f"Persistent buffer {item.name!r} is on {item.tensor.device}, "
                    f"expected {runtime.device}."
                )
            dist.broadcast(item.tensor, src=0)
    return len(buffers)


def _validate_ddp_contract(model: M3DModel, config: ExperimentConfig) -> None:
    ddp = config.distributed.ddp
    if model.seg_enable and not ddp.find_unused_parameters:
        raise DistributedModelError(
            "M3D text steps skip the SegVol branch, so DDP requires "
            "find_unused_parameters=true while segmentation is enabled."
        )
    if model.seg_enable and ddp.static_graph:
        raise DistributedModelError(
            "M3D alternates text and segmentation execution graphs; DDP "
            "static_graph must remain false."
        )
    if ddp.bucket_cap_mb <= 0:
        raise DistributedModelError("DDP bucket_cap_mb must be positive.")


def _wrap_ddp(
    model: M3DModel,
    runtime: RuntimeContext,
) -> tuple[DistributedDataParallel, int]:
    _validate_ddp_contract(model, runtime.config)
    model.to(runtime.device)

    ddp_config = runtime.config.distributed.ddp
    wrapped = DistributedDataParallel(
        model,
        device_ids=[runtime.local_rank],
        output_device=runtime.local_rank,
        broadcast_buffers=False,
        bucket_cap_mb=int(ddp_config.bucket_cap_mb),
        find_unused_parameters=bool(ddp_config.find_unused_parameters),
        gradient_as_bucket_view=bool(ddp_config.gradient_as_bucket_view),
        static_graph=bool(ddp_config.static_graph),
    )
    # PyTorch 2.6 DDP always performs its initial parameter sync during
    # construction; the configurable ``init_sync`` keyword only exists in newer
    # releases. Since DDP buffer broadcasting is disabled, synchronize persistent
    # buffers exactly once after construction.
    synchronised_buffers = _synchronise_persistent_buffers_once(model, runtime)
    return wrapped, synchronised_buffers


def _unassigned_parameter_stats(
    module: nn.Module,
    assigned_parameter_ids: set[int],
) -> tuple[int, int, int, set[int]]:
    """Describe parameters that an upcoming bottom-up FSDP group will own."""

    unique: dict[int, nn.Parameter] = {}
    for parameter in module.parameters(recurse=True):
        identity = id(parameter)
        if identity not in assigned_parameter_ids:
            unique.setdefault(identity, parameter)
    new_ids = set(unique)
    parameters = list(unique.values())
    return (
        len(parameters),
        int(sum(parameter.numel() for parameter in parameters)),
        int(sum(parameter.numel() for parameter in parameters if parameter.requires_grad)),
        new_ids,
    )


def _module_list(value: Any, *, label: str) -> Sequence[nn.Module]:
    if not isinstance(value, (nn.ModuleList, list, tuple)):
        raise DistributedModelError(
            f"Expected {label} to be a module sequence, got {type(value).__name__}."
        )
    modules = list(value)
    if not modules or not all(isinstance(module, nn.Module) for module in modules):
        raise DistributedModelError(f"{label} must contain at least one nn.Module.")
    return cast(Sequence[nn.Module], modules)


def _language_decoder_layers(model: M3DModel) -> Sequence[nn.Module]:
    decoder = model.language_model.get_decoder()
    candidates = (
        getattr(decoder, "layers", None),
        getattr(getattr(decoder, "model", None), "layers", None),
        getattr(getattr(decoder, "decoder", None), "layers", None),
    )
    for candidate in candidates:
        if isinstance(candidate, (nn.ModuleList, list, tuple)) and len(candidate) > 0:
            return _module_list(candidate, label="Phi-3 decoder layers")
    raise DistributedModelError(
        f"Cannot locate Phi-3 decoder layers inside {type(decoder).__name__}."
    )


def _append_unique(
    destination: list[_NamedModule],
    seen: set[int],
    *,
    name: str,
    module: nn.Module,
) -> None:
    identity = id(module)
    if identity in seen:
        return
    destination.append(_NamedModule(name=name, module=module))
    seen.add(identity)


def build_fsdp2_wrap_plan(
    model: M3DModel,
    config: ExperimentConfig,
) -> tuple[_NamedModule, ...]:
    """Return modules in the exact bottom-up order required by ``fully_shard``."""

    plan: list[_NamedModule] = []
    seen: set[int] = set()
    fsdp = config.distributed.fsdp2

    if fsdp.wrap_main_vision_layers:
        blocks = _module_list(
            model.main_image_encoder.blocks,
            label="Main ViT blocks",
        )
        for index, block in enumerate(blocks):
            _append_unique(
                plan,
                seen,
                name=f"main_vision.blocks.{index}",
                module=block,
            )
    _append_unique(
        plan,
        seen,
        name="main_vision.encoder",
        module=model.main_image_encoder,
    )

    if fsdp.wrap_language_layers:
        for index, layer in enumerate(_language_decoder_layers(model)):
            _append_unique(
                plan,
                seen,
                name=f"language.decoder.layers.{index}",
                module=layer,
            )
    # The causal-LM root groups embeddings, final norm, LM head, PEFT wrappers,
    # and any remaining parameters while excluding already-sharded layers.
    _append_unique(
        plan,
        seen,
        name="language.causal_lm",
        module=model.language_model.causal_lm,
    )

    _append_unique(
        plan,
        seen,
        name="multimodal_projector",
        module=model.mm_projector,
    )

    if model.seg_enable:
        assert model.seg_module is not None
        assert model.seg_projector is not None
        if fsdp.wrap_seg_vision_layers:
            seg_blocks = _module_list(
                model.seg_module.image_encoder.blocks,
                label="SegVol ViT blocks",
            )
            for index, block in enumerate(seg_blocks):
                _append_unique(
                    plan,
                    seen,
                    name=f"segvol.image_encoder.blocks.{index}",
                    module=block,
                )
        _append_unique(
            plan,
            seen,
            name="segvol.image_encoder",
            module=model.seg_module.image_encoder,
        )
        _append_unique(
            plan,
            seen,
            name="segmentation_projector",
            module=model.seg_projector,
        )
        _append_unique(
            plan,
            seen,
            name="segvol.prompt_encoder",
            module=model.seg_module.prompt_encoder,
        )

        transformer = model.seg_module.mask_decoder.transformer
        if fsdp.wrap_segmentation_decoder_layers:
            transformer_layers = _module_list(
                transformer.layers,
                label="SegVol Two-Way Transformer layers",
            )
            for index, layer in enumerate(transformer_layers):
                _append_unique(
                    plan,
                    seen,
                    name=f"segvol.mask_decoder.transformer.layers.{index}",
                    module=layer,
                )
        _append_unique(
            plan,
            seen,
            name="segvol.mask_decoder.transformer",
            module=transformer,
        )
        _append_unique(
            plan,
            seen,
            name="segvol.mask_decoder",
            module=model.seg_module.mask_decoder,
        )

    # Root must be last. It picks up only parameters not assigned to child groups.
    _append_unique(plan, seen, name="m3d.root", module=model)
    return tuple(plan)


def _import_fsdp2() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from torch.distributed.fsdp import (
            CPUOffloadPolicy,
            FSDPModule,
            MixedPrecisionPolicy,
            OffloadPolicy,
            fully_shard,
        )
    except ImportError as exc:  # pragma: no cover - depends on installed torch.
        raise DistributedModelError(
            "The installed PyTorch build does not provide the FSDP2 fully_shard API."
        ) from exc
    return fully_shard, FSDPModule, MixedPrecisionPolicy, OffloadPolicy, CPUOffloadPolicy


def _wrap_fsdp2(
    model: M3DModel,
    runtime: RuntimeContext,
) -> tuple[nn.Module, tuple[FSDPWrapGroup, ...]]:
    marker = _model_build_marker(model)
    if marker is None:
        raise DistributedModelError(
            "FSDP2 requires model construction through "
            "build_model_synchronously(runtime, builder). Rank-offset random "
            "initialization cannot be sharded safely."
        )
    if marker.world_size != runtime.world_size:
        raise DistributedModelError(
            "The model-build marker world size differs from the active runtime: "
            f"{marker.world_size} != {runtime.world_size}."
        )
    if runtime.device_mesh is None:
        raise DistributedModelError("FSDP2 requires RuntimeContext.device_mesh.")

    fully_shard, _, MixedPrecisionPolicy, OffloadPolicy, CPUOffloadPolicy = _import_fsdp2()
    policy = MixedPrecisionPolicy(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        output_dtype=None,
        cast_forward_inputs=True,
    )
    fsdp_config = runtime.config.distributed.fsdp2
    offload_policy = (
        CPUOffloadPolicy(pin_memory=True)
        if fsdp_config.cpu_offload
        else OffloadPolicy()
    )

    records: list[FSDPWrapGroup] = []
    assigned_parameter_ids: set[int] = set()
    for item in build_fsdp2_wrap_plan(model, runtime.config):
        (
            tensor_count,
            element_count,
            trainable_count,
            newly_assigned_ids,
        ) = _unassigned_parameter_stats(item.module, assigned_parameter_ids)
        fully_shard(
            item.module,
            mesh=runtime.device_mesh,
            reshard_after_forward=bool(fsdp_config.reshard_after_forward),
            mp_policy=policy,
            offload_policy=offload_policy,
        )
        assigned_parameter_ids.update(newly_assigned_ids)
        records.append(
            FSDPWrapGroup(
                name=item.name,
                module_type=type(item.module).__name__,
                parameter_tensor_count=tensor_count,
                parameter_element_count=element_count,
                trainable_parameter_count=trainable_count,
                reshard_after_forward=bool(fsdp_config.reshard_after_forward),
            )
        )

    # ``fully_shard`` mutates the original module in place and preserves fully
    # qualified parameter names while exposing FSDPModule methods dynamically.
    return model, tuple(records)


def _wrapped_parameter_counts(module: nn.Module) -> tuple[int, int]:
    parameters = list(module.parameters())
    return len(parameters), sum(parameter.requires_grad for parameter in parameters)


class DistributedM3DModel(nn.Module):
    """Small strategy-neutral facade around DDP or composable FSDP2.

    Calling ``forward_batch`` always routes through the distributed wrapper's
    normal ``forward`` method.  Calling ``ddp.module.forward_batch`` directly
    would bypass DDP's forward bookkeeping and is intentionally not exposed as
    the recommended path.
    """

    def __init__(
        self,
        wrapped_model: nn.Module,
        *,
        unwrapped_model: M3DModel,
        runtime: RuntimeContext,
        strategy: DistributedStrategy,
    ) -> None:
        super().__init__()
        self.wrapped_model = wrapped_model
        # Keep only a weak reference to the original model. Registering the same
        # nn.Module a second time would duplicate state-dict paths under this
        # facade even though both attributes point to the same parameters.
        object.__setattr__(self, "_unwrapped_model_ref", weakref.ref(unwrapped_model))
        self.runtime = runtime
        self.strategy = strategy

    @property
    def unwrapped_model(self) -> M3DModel:
        reference = cast(weakref.ReferenceType[M3DModel], self._unwrapped_model_ref)
        model = reference()
        if model is None:
            raise DistributedModelError("The underlying M3D model was released.")
        return model

    @property
    def is_ddp(self) -> bool:
        return self.strategy == "ddp"

    @property
    def is_fsdp2(self) -> bool:
        return self.strategy == "fsdp2"

    @property
    def requires_even_evaluation_steps(self) -> bool:
        # FSDP2 forward executes all-gathers. Every rank must therefore enter
        # the same number of evaluation forwards; a later evaluator will pad
        # exhausted ranks without counting those examples in metrics.
        return self.is_fsdp2

    def forward(self, *args: Any, **kwargs: Any) -> M3DModelOutput:
        output = self.wrapped_model(*args, **kwargs)
        if not isinstance(output, M3DModelOutput):
            raise DistributedModelError(
                "Distributed M3D forward returned an unexpected type: "
                f"{type(output).__name__}."
            )
        return output

    def forward_batch(
        self,
        batch: M3DBatch,
        *,
        logits_mode: Literal["none", "supervised", "full"] = "none",
        return_intermediates: bool = False,
    ) -> M3DModelOutput:
        if not isinstance(batch, M3DBatch):
            raise TypeError(f"batch must be M3DBatch, got {type(batch).__name__}.")
        return self(
            task=batch.task,
            **batch.model_inputs(),
            logits_mode=logits_mode,
            return_intermediates=return_intermediates,
        )

    @contextlib.contextmanager
    def gradient_sync(self, *, enabled: bool) -> Generator[None, None, None]:
        """Control gradient communication for one forward/backward microstep."""

        if self.is_ddp:
            if not isinstance(self.wrapped_model, DistributedDataParallel):
                raise DistributedModelError("DDP strategy does not hold a DDP wrapper.")
            if enabled:
                yield
            else:
                with self.wrapped_model.no_sync():
                    yield
            return

        _, FSDPModule, _, _, _ = _import_fsdp2()
        root = self.wrapped_model
        if not isinstance(root, FSDPModule):
            raise DistributedModelError(
                f"FSDP2 root is not FSDPModule: {type(root).__name__}."
            )
        root.set_requires_gradient_sync(bool(enabled), recurse=True)
        set_last_backward = getattr(root, "set_is_last_backward", None)
        if callable(set_last_backward):
            set_last_backward(bool(enabled))
        try:
            yield
        finally:
            # Always restore communication and final-backward cleanup, including
            # exception paths.
            root.set_requires_gradient_sync(True, recurse=True)
            if callable(set_last_backward):
                set_last_backward(True)

    def clip_grad_norm_(self, max_norm: float, norm_type: float = 2.0) -> Tensor:
        if max_norm <= 0:
            raise ValueError(f"max_norm must be positive, got {max_norm}.")
        result = torch.nn.utils.clip_grad_norm_(
            self.unwrapped_model.parameters(),
            max_norm=float(max_norm),
            norm_type=float(norm_type),
            error_if_nonfinite=True,
        )
        return result if isinstance(result, Tensor) else torch.as_tensor(result)

    def assert_task_consistent(self, task: TaskName | str) -> None:
        parsed = TaskName.parse(task)
        self.runtime.assert_all_ranks_equal(
            parsed.value,
            label="current M3D task graph",
        )


def prepare_distributed_model(
    model: M3DModel,
    runtime: RuntimeContext,
) -> tuple[DistributedM3DModel, DistributedModelReport]:
    """Move/wrap a complete M3D model and return it ready for optimizer creation."""

    if not isinstance(model, M3DModel):
        raise TypeError(f"model must be M3DModel, got {type(model).__name__}.")
    strategy = cast(DistributedStrategy, runtime.config.distributed.strategy)
    if strategy not in {"ddp", "fsdp2"}:
        raise DistributedModelError(f"Unsupported distributed strategy {strategy!r}.")
    if not runtime.process_group_initialized or runtime.world_size <= 1:
        raise DistributedModelError(
            f"Strategy {strategy!r} requires an initialized multi-process group."
        )

    model._validate_component_contracts()
    _validate_parameter_devices_before_wrap(
        model,
        strategy=strategy,
        device=runtime.device,
    )
    layout = parameter_layout_summary(model)
    _assert_layout_equal_across_ranks(runtime, layout)

    fsdp_groups: tuple[FSDPWrapGroup, ...] = ()
    synchronized_buffers = 0
    marker = _model_build_marker(model)
    if strategy == "ddp":
        wrapped, synchronized_buffers = _wrap_ddp(model, runtime)
    else:
        wrapped, fsdp_groups = _wrap_fsdp2(model, runtime)

    facade = DistributedM3DModel(
        wrapped,
        unwrapped_model=model,
        runtime=runtime,
        strategy=strategy,
    )
    wrapped_tensors, wrapped_trainable = _wrapped_parameter_counts(model)

    ddp_config = runtime.config.distributed.ddp
    fsdp_config = runtime.config.distributed.fsdp2
    report = DistributedModelReport(
        strategy=strategy,
        world_size=runtime.world_size,
        rank=runtime.rank,
        local_rank=runtime.local_rank,
        device=str(runtime.device),
        model_type=type(model).__name__,
        parameter_layout=layout,
        wrapped_parameter_tensor_count=wrapped_tensors,
        wrapped_trainable_tensor_count=wrapped_trainable,
        ddp_find_unused_parameters=(
            bool(ddp_config.find_unused_parameters) if strategy == "ddp" else None
        ),
        ddp_gradient_as_bucket_view=(
            bool(ddp_config.gradient_as_bucket_view) if strategy == "ddp" else None
        ),
        ddp_static_graph=(bool(ddp_config.static_graph) if strategy == "ddp" else None),
        ddp_bucket_cap_mb=(int(ddp_config.bucket_cap_mb) if strategy == "ddp" else None),
        ddp_broadcast_buffers_each_forward=(False if strategy == "ddp" else None),
        persistent_buffers_synchronised_once=synchronized_buffers,
        fsdp_groups=fsdp_groups,
        fsdp_mixed_precision=("bfloat16" if strategy == "fsdp2" else None),
        fsdp_reduce_dtype=("float32" if strategy == "fsdp2" else None),
        fsdp_cpu_offload=(bool(fsdp_config.cpu_offload) if strategy == "fsdp2" else None),
        requires_even_evaluation_steps=facade.requires_even_evaluation_steps,
        synchronous_initialization_seed=(None if marker is None else marker.seed),
    )

    runtime.logger.info(
        "Distributed model prepared: strategy=%s parameters=%d trainable=%d "
        "parameter_tensors=%d FSDP_groups=%d",
        strategy,
        layout.parameter_count,
        layout.trainable_parameter_count,
        layout.parameter_tensor_count,
        len(fsdp_groups),
    )
    runtime.barrier()
    return facade, report


# ---------------------------------------------------------------------------
# Dependency-light CPU self-test
# ---------------------------------------------------------------------------


def _test_synchronised_initialization() -> dict[str, Any]:
    class _Runtime:
        def __init__(self) -> None:
            self.device = torch.device("cpu")
            self.world_size = 2
            self.config = type(
                "Config",
                (),
                {"runtime": type("Runtime", (), {"seed": 123})()},
            )()

    runtime = cast(RuntimeContext, _Runtime())
    torch.manual_seed(999)
    before = torch.rand(3)
    with synchronized_model_initialization(runtime):
        first = torch.rand(5)
    middle = torch.rand(3)

    torch.manual_seed(999)
    expected_before = torch.rand(3)
    expected_middle = torch.rand(3)
    with synchronized_model_initialization(runtime):
        second = torch.rand(5)

    if not torch.equal(before, expected_before):
        raise AssertionError("CPU RNG stream changed before synchronized context.")
    if not torch.equal(middle, expected_middle):
        raise AssertionError("CPU RNG stream was not restored after context.")
    if not torch.equal(first, second):
        raise AssertionError("Common-seed model initialization is not deterministic.")
    return {
        "common_seed_reproducible": True,
        "rank_rng_stream_restored": True,
    }


def _test_persistent_buffer_filter() -> dict[str, Any]:
    module = nn.Module()
    module.register_buffer("persistent", torch.ones(2), persistent=True)
    module.register_buffer("cache", torch.zeros(3), persistent=False)
    names = [item.name for item in _iter_persistent_buffers(module)]
    if names != ["persistent"]:
        raise AssertionError(f"Persistent-buffer filter returned {names!r}.")
    return {
        "persistent_buffers": names,
        "nonpersistent_cache_excluded": True,
    }


def _test_layout_summary() -> dict[str, Any]:
    class _TinyM3D(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 3)
            self.register_buffer("scale", torch.ones(1))

    tiny = cast(M3DModel, _TinyM3D())
    first = parameter_layout_summary(tiny)
    second = parameter_layout_summary(tiny)
    if first.layout_sha256 != second.layout_sha256:
        raise AssertionError("Parameter layout fingerprint is not stable.")
    return {
        "layout_sha256_stable": True,
        "parameter_tensor_count": first.parameter_tensor_count,
        "persistent_buffer_count": first.persistent_buffer_count,
    }


def run_self_test() -> Mapping[str, Any]:
    result: dict[str, Any] = {"status": "passed"}
    result.update(_test_synchronised_initialization())
    result.update(_test_persistent_buffer_filter())
    result.update(_test_layout_summary())
    return result


def main() -> None:
    print(json.dumps(run_self_test(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
