"""Deterministic, component-aware optimizer construction for modernised M3D.

The complete M3D model contains two independent 3-D image encoders, a
multimodal projector, a Phi-3/PEFT language stack, a language-to-segmentation
projector, and the remaining SegVol prompt/mask decoder.  Those components use
separate learning rates, while every trainable parameter must appear in exactly
one optimizer group.

This module deliberately runs *after* DDP or composable FSDP2 wrapping.  Under
FSDP2, wrapping replaces parameters with sharded DTensor parameters, so an
optimizer created before wrapping would hold stale full-parameter references.

The public entry point is::

    optimizer, report = build_optimizer(
        distributed_model.unwrapped_model,
        config,
        distributed_strategy=distributed_model.strategy,
    )

The returned AdamW parameter groups contain stable ``group_name``, ``role``,
``decay_kind``, and ``param_names`` metadata.  This makes optimizer checkpoint
state deterministic and makes later scheduler/reporting code independent of
integer group positions.
"""

from __future__ import annotations

import dataclasses
import hashlib
import inspect
import json
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from typing import Any, Iterable, Mapping, MutableMapping, Sequence, cast

import torch
from torch import Tensor, nn
from torch.optim import AdamW, Optimizer

from m3d.config import ExperimentConfig, OptimizationConfig
from m3d.model.m3d import M3DModel


class OptimizerConfigurationError(ValueError):
    """Raised when optimizer configuration is incompatible with the model."""


class ParameterGroupingError(RuntimeError):
    """Raised when trainable parameters cannot be partitioned unambiguously."""


class ParameterRole(str, Enum):
    """Semantic optimizer roles with independently configurable learning rates."""

    LANGUAGE_MODEL = "language_model"
    MAIN_VISION = "main_vision"
    SEG_VISION = "seg_vision"
    PROJECTOR = "projector"
    SEGMENTATION_PROJECTOR = "segmentation_projector"
    SEGMENTATION_DECODER = "segmentation_decoder"
    TOKEN_EMBEDDINGS = "token_embeddings"


class DecayKind(str, Enum):
    DECAY = "decay"
    NO_DECAY = "no_decay"


# Keep this order equal to LearningRateConfig's public order.  Optimizer state
# uses group order, so changing it is a checkpoint compatibility decision.
ROLE_ORDER: tuple[ParameterRole, ...] = (
    ParameterRole.LANGUAGE_MODEL,
    ParameterRole.MAIN_VISION,
    ParameterRole.SEG_VISION,
    ParameterRole.PROJECTOR,
    ParameterRole.SEGMENTATION_PROJECTOR,
    ParameterRole.SEGMENTATION_DECODER,
    ParameterRole.TOKEN_EMBEDDINGS,
)

DECAY_ORDER: tuple[DecayKind, ...] = (
    DecayKind.DECAY,
    DecayKind.NO_DECAY,
)


@dataclass(frozen=True, slots=True)
class ParameterRecord:
    """One canonical trainable parameter and its optimizer classification."""

    name: str
    parameter: nn.Parameter
    role: ParameterRole
    decay_kind: DecayKind
    learning_rate: float
    weight_decay: float

    @property
    def numel(self) -> int:
        return int(self.parameter.numel())

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(value) for value in self.parameter.shape)


@dataclass(frozen=True, slots=True)
class ParameterGroupReport:
    group_name: str
    role: str
    decay_kind: str
    learning_rate: float
    weight_decay: float
    parameter_tensor_count: int
    parameter_element_count: int
    parameter_names_sha256: str
    parameter_names: tuple[str, ...]

    def to_dict(self, *, include_parameter_names: bool = True) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "group_name": self.group_name,
            "role": self.role,
            "decay_kind": self.decay_kind,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "parameter_tensor_count": self.parameter_tensor_count,
            "parameter_element_count": self.parameter_element_count,
            "parameter_names_sha256": self.parameter_names_sha256,
        }
        if include_parameter_names:
            payload["parameter_names"] = list(self.parameter_names)
        return payload


@dataclass(frozen=True, slots=True)
class OptimizerBuildReport:
    optimizer_name: str
    training_stage: str
    requested_fused: bool
    fused_enabled: bool
    fused_fallback_reason: str | None
    distributed_strategy: str | None
    betas: tuple[float, float]
    epsilon: float
    configured_weight_decay: float
    trainable_parameter_tensor_count: int
    trainable_parameter_element_count: int
    optimized_parameter_tensor_count: int
    optimized_parameter_element_count: int
    frozen_parameter_tensor_count: int
    zero_learning_rate_roles: tuple[str, ...]
    active_roles: tuple[str, ...]
    parameter_layout_sha256: str
    groups: tuple[ParameterGroupReport, ...]

    def to_dict(self, *, include_parameter_names: bool = True) -> dict[str, Any]:
        return {
            "optimizer_name": self.optimizer_name,
            "training_stage": self.training_stage,
            "requested_fused": self.requested_fused,
            "fused_enabled": self.fused_enabled,
            "fused_fallback_reason": self.fused_fallback_reason,
            "distributed_strategy": self.distributed_strategy,
            "betas": list(self.betas),
            "epsilon": self.epsilon,
            "configured_weight_decay": self.configured_weight_decay,
            "trainable_parameter_tensor_count": self.trainable_parameter_tensor_count,
            "trainable_parameter_element_count": self.trainable_parameter_element_count,
            "optimized_parameter_tensor_count": self.optimized_parameter_tensor_count,
            "optimized_parameter_element_count": self.optimized_parameter_element_count,
            "frozen_parameter_tensor_count": self.frozen_parameter_tensor_count,
            "zero_learning_rate_roles": list(self.zero_learning_rate_roles),
            "active_roles": list(self.active_roles),
            "parameter_layout_sha256": self.parameter_layout_sha256,
            "groups": [
                group.to_dict(include_parameter_names=include_parameter_names)
                for group in self.groups
            ],
        }

    def to_json(
        self,
        *,
        include_parameter_names: bool = True,
        indent: int = 2,
    ) -> str:
        return json.dumps(
            self.to_dict(include_parameter_names=include_parameter_names),
            indent=indent,
            sort_keys=True,
        )


@dataclass(frozen=True, slots=True)
class _GroupingResult:
    records: tuple[ParameterRecord, ...]
    groups: tuple[dict[str, Any], ...]
    reports: tuple[ParameterGroupReport, ...]
    layout_sha256: str


# ---------------------------------------------------------------------------
# Configuration and parameter discovery
# ---------------------------------------------------------------------------


def _optimization_config(
    config: ExperimentConfig | OptimizationConfig,
) -> OptimizationConfig:
    if isinstance(config, OptimizationConfig):
        return config
    if isinstance(config, ExperimentConfig):
        return config.optimization
    raise TypeError(
        "config must be ExperimentConfig or OptimizationConfig, got "
        f"{type(config).__name__}."
    )


def _validate_optimization_config(config: OptimizationConfig) -> None:
    if config.optimizer != "adamw_fused":
        raise OptimizerConfigurationError(
            f"Only optimizer='adamw_fused' is supported, got {config.optimizer!r}."
        )
    if len(config.betas) != 2:
        raise OptimizerConfigurationError(
            f"AdamW betas must contain two values, got {config.betas!r}."
        )
    beta1, beta2 = (float(config.betas[0]), float(config.betas[1]))
    if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
        raise OptimizerConfigurationError(
            f"AdamW betas must be in [0,1), got {(beta1, beta2)}."
        )
    if float(config.epsilon) <= 0.0:
        raise OptimizerConfigurationError(
            f"AdamW epsilon must be positive, got {config.epsilon}."
        )
    if float(config.weight_decay) < 0.0:
        raise OptimizerConfigurationError(
            f"weight_decay cannot be negative, got {config.weight_decay}."
        )

    rates = dataclasses.asdict(config.learning_rates)
    expected = {role.value for role in ROLE_ORDER}
    if set(rates) != expected:
        raise OptimizerConfigurationError(
            "LearningRateConfig fields differ from optimizer roles: "
            f"expected={sorted(expected)}, got={sorted(rates)}."
        )
    invalid = {
        name: value
        for name, value in rates.items()
        if not isinstance(value, (int, float)) or float(value) < 0.0
    }
    if invalid:
        raise OptimizerConfigurationError(
            f"Learning rates must be finite non-negative numbers: {invalid}."
        )
    non_finite = {
        name: value
        for name, value in rates.items()
        if not torch.isfinite(torch.tensor(float(value)))
    }
    if non_finite:
        raise OptimizerConfigurationError(
            f"Learning rates must be finite: {non_finite}."
        )


def _parameter_ids(module: nn.Module | None) -> frozenset[int]:
    if module is None:
        return frozenset()
    return frozenset(id(parameter) for parameter in module.parameters())


def _direct_parameter_owners(model: nn.Module) -> Mapping[int, tuple[nn.Module, ...]]:
    owners: MutableMapping[int, list[nn.Module]] = defaultdict(list)
    for _, module in model.named_modules():
        # recurse=False means tied/shared parameters may have several direct
        # owners, which is useful for no-decay classification.
        for parameter in module.parameters(recurse=False):
            owners[id(parameter)].append(module)
    return {key: tuple(value) for key, value in owners.items()}


def _component_parameter_sets(model: M3DModel) -> Mapping[ParameterRole, frozenset[int]]:
    token_modules = (
        model.language_model.get_input_embeddings(),
        model.language_model.get_output_embeddings(),
    )
    token_ids = frozenset(
        parameter_id
        for module in token_modules
        for parameter_id in _parameter_ids(module)
    )

    seg_decoder_ids: set[int] = set()
    if model.seg_module is not None:
        seg_decoder_ids.update(_parameter_ids(model.seg_module.prompt_encoder))
        seg_decoder_ids.update(_parameter_ids(model.seg_module.mask_decoder))

    language_ids = set(_parameter_ids(model.language_model))
    language_ids.difference_update(token_ids)

    return {
        ParameterRole.TOKEN_EMBEDDINGS: token_ids,
        ParameterRole.LANGUAGE_MODEL: frozenset(language_ids),
        ParameterRole.MAIN_VISION: _parameter_ids(model.vision_tower),
        ParameterRole.PROJECTOR: _parameter_ids(model.mm_projector),
        ParameterRole.SEGMENTATION_PROJECTOR: _parameter_ids(model.seg_projector),
        ParameterRole.SEG_VISION: _parameter_ids(
            None if model.seg_module is None else model.seg_module.image_encoder
        ),
        ParameterRole.SEGMENTATION_DECODER: frozenset(seg_decoder_ids),
    }


def _canonical_named_parameters(model: nn.Module) -> tuple[tuple[str, nn.Parameter], ...]:
    pairs = tuple(model.named_parameters(remove_duplicate=True))
    names = [name for name, _ in pairs]
    if len(names) != len(set(names)):
        raise ParameterGroupingError("Canonical model parameter names are not unique.")
    ids = [id(parameter) for _, parameter in pairs]
    if len(ids) != len(set(ids)):
        raise ParameterGroupingError(
            "named_parameters(remove_duplicate=True) returned duplicate Parameter objects."
        )
    return pairs


def _role_for_parameter(
    *,
    name: str,
    parameter: nn.Parameter,
    component_sets: Mapping[ParameterRole, frozenset[int]],
) -> ParameterRole:
    parameter_id = id(parameter)
    memberships = [
        role for role, identifiers in component_sets.items() if parameter_id in identifiers
    ]
    if len(memberships) == 1:
        return memberships[0]
    if not memberships:
        raise ParameterGroupingError(
            "A trainable parameter is outside every known M3D optimizer component: "
            f"name={name!r}, shape={tuple(parameter.shape)}.  Add an explicit role "
            "instead of silently assigning the base learning rate."
        )
    raise ParameterGroupingError(
        "A trainable parameter belongs to multiple optimizer components: "
        f"name={name!r}, roles={[role.value for role in memberships]}."
    )


def _is_norm_module(module: nn.Module) -> bool:
    if isinstance(module, nn.LayerNorm):
        return True
    class_name = type(module).__name__.lower()
    # Covers Phi-3 RMSNorm and compatible custom norm classes without importing
    # Transformers into this dependency-light optimizer module.
    return class_name.endswith("norm") or "rmsnorm" in class_name


def _decay_kind_for_parameter(
    *,
    name: str,
    parameter: nn.Parameter,
    owners: Sequence[nn.Module],
    explicit_no_decay_names: frozenset[str],
) -> DecayKind:
    if name in explicit_no_decay_names:
        return DecayKind.NO_DECAY
    if name.endswith(".bias"):
        return DecayKind.NO_DECAY
    if parameter.ndim <= 1:
        return DecayKind.NO_DECAY
    if any(isinstance(owner, nn.Embedding) for owner in owners):
        return DecayKind.NO_DECAY
    if any(_is_norm_module(owner) for owner in owners):
        return DecayKind.NO_DECAY
    return DecayKind.DECAY


def _learning_rate_for_role(
    config: OptimizationConfig,
    role: ParameterRole,
) -> float:
    try:
        value = getattr(config.learning_rates, role.value)
    except AttributeError as error:  # pragma: no cover - protected by validation
        raise OptimizerConfigurationError(
            f"Missing learning rate for role {role.value!r}."
        ) from error
    return float(value)


# ---------------------------------------------------------------------------
# Alias validation and deterministic grouping
# ---------------------------------------------------------------------------


def _is_dtensor(value: Any) -> bool:
    try:
        from torch.distributed.tensor import DTensor
    except (ImportError, RuntimeError):  # pragma: no cover - build dependent
        return False
    return isinstance(value, DTensor)


def _parameter_contains_dtensor(parameter: nn.Parameter) -> bool:
    if _is_dtensor(parameter):
        return True
    try:
        return _is_dtensor(parameter.data)
    except RuntimeError:
        return False


def _storage_identity(parameter: nn.Parameter) -> tuple[Any, ...] | None:
    """Return a conservative local storage identity for alias detection.

    FSDP2 DTensors are intentionally skipped: each rank sees local shards and
    DTensor ownership is already managed by FSDP2.  Ordinary DDP parameters are
    checked to prevent two distinct optimizer parameters from updating the same
    underlying storage.
    """

    if _parameter_contains_dtensor(parameter):
        return None
    if parameter.device.type == "meta" or parameter.numel() == 0:
        return None
    try:
        storage = parameter.untyped_storage()
        return (
            parameter.device.type,
            parameter.device.index,
            int(storage.data_ptr()),
        )
    except (AttributeError, RuntimeError):
        return None


def _validate_no_distinct_storage_aliases(
    named_parameters: Sequence[tuple[str, nn.Parameter]],
) -> None:
    by_storage: MutableMapping[tuple[Any, ...], list[tuple[str, nn.Parameter]]] = (
        defaultdict(list)
    )
    for name, parameter in named_parameters:
        if not parameter.requires_grad:
            continue
        identity = _storage_identity(parameter)
        if identity is not None:
            by_storage[identity].append((name, parameter))

    aliases: list[str] = []
    for entries in by_storage.values():
        unique_ids = {id(parameter) for _, parameter in entries}
        if len(unique_ids) <= 1:
            continue
        aliases.append(", ".join(name for name, _ in entries))
    if aliases:
        raise ParameterGroupingError(
            "Distinct trainable Parameter objects share storage and would be updated "
            "more than once by AdamW:\n  - " + "\n  - ".join(aliases)
        )


def _hash_lines(lines: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for line in lines:
        digest.update(line.encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _parameter_layout_hash(records: Sequence[ParameterRecord]) -> str:
    return _hash_lines(
        (
            f"{record.name}|{record.role.value}|{record.decay_kind.value}|"
            f"{record.shape}|{record.parameter.dtype}|{record.learning_rate:.17g}|"
            f"{record.weight_decay:.17g}"
        )
        for record in records
    )


def partition_trainable_parameters(
    model: M3DModel,
    config: ExperimentConfig | OptimizationConfig,
) -> _GroupingResult:
    """Partition every trainable M3D parameter exactly once.

    This function does not create an optimizer and is useful for startup
    validation, reports, and tests.
    """

    if not isinstance(model, M3DModel):
        raise TypeError(f"model must be M3DModel, got {type(model).__name__}.")
    optimization = _optimization_config(config)
    _validate_optimization_config(optimization)

    canonical = _canonical_named_parameters(model)
    _validate_no_distinct_storage_aliases(canonical)
    component_sets = _component_parameter_sets(model)
    owners = _direct_parameter_owners(model)
    explicit_no_decay = model.no_weight_decay_parameter_names()

    records: list[ParameterRecord] = []
    for name, parameter in canonical:
        if not parameter.requires_grad:
            continue
        role = _role_for_parameter(
            name=name,
            parameter=parameter,
            component_sets=component_sets,
        )
        decay_kind = _decay_kind_for_parameter(
            name=name,
            parameter=parameter,
            owners=owners.get(id(parameter), ()),
            explicit_no_decay_names=explicit_no_decay,
        )
        records.append(
            ParameterRecord(
                name=name,
                parameter=parameter,
                role=role,
                decay_kind=decay_kind,
                learning_rate=_learning_rate_for_role(optimization, role),
                weight_decay=(
                    float(optimization.weight_decay)
                    if decay_kind is DecayKind.DECAY
                    else 0.0
                ),
            )
        )

    if not records:
        raise ParameterGroupingError(
            "The complete M3D model has no trainable parameters; optimizer creation "
            "would be meaningless."
        )

    buckets: MutableMapping[
        tuple[ParameterRole, DecayKind], list[ParameterRecord]
    ] = defaultdict(list)
    for record in records:
        buckets[(record.role, record.decay_kind)].append(record)

    groups: list[dict[str, Any]] = []
    reports: list[ParameterGroupReport] = []
    optimized_ids: list[int] = []

    for role in ROLE_ORDER:
        for decay_kind in DECAY_ORDER:
            bucket = buckets.get((role, decay_kind), [])
            if not bucket:
                continue
            # Canonical named_parameters order is deterministic. Sorting names
            # makes the contract explicit even if component registration order
            # changes in a future refactor.
            bucket = sorted(bucket, key=lambda record: record.name)
            names = tuple(record.name for record in bucket)
            parameters = [record.parameter for record in bucket]
            group_name = f"{role.value}/{decay_kind.value}"
            learning_rate = bucket[0].learning_rate
            weight_decay = bucket[0].weight_decay

            if any(record.learning_rate != learning_rate for record in bucket):
                raise ParameterGroupingError(
                    f"Group {group_name!r} contains inconsistent learning rates."
                )
            if any(record.weight_decay != weight_decay for record in bucket):
                raise ParameterGroupingError(
                    f"Group {group_name!r} contains inconsistent weight decay."
                )

            groups.append(
                {
                    "params": parameters,
                    # Extra metadata is intentionally retained in optimizer
                    # state_dict() and consumed by the scheduler/reporting code.
                    "param_names": list(names),
                    "group_name": group_name,
                    "role": role.value,
                    "decay_kind": decay_kind.value,
                    "lr": float(learning_rate),
                    "initial_lr": float(learning_rate),
                    "weight_decay": float(weight_decay),
                }
            )
            optimized_ids.extend(id(parameter) for parameter in parameters)
            reports.append(
                ParameterGroupReport(
                    group_name=group_name,
                    role=role.value,
                    decay_kind=decay_kind.value,
                    learning_rate=float(learning_rate),
                    weight_decay=float(weight_decay),
                    parameter_tensor_count=len(parameters),
                    parameter_element_count=sum(
                        int(parameter.numel()) for parameter in parameters
                    ),
                    parameter_names_sha256=_hash_lines(names),
                    parameter_names=names,
                )
            )

    trainable_ids = [id(record.parameter) for record in records]
    if len(optimized_ids) != len(set(optimized_ids)):
        raise ParameterGroupingError(
            "At least one trainable parameter was inserted into multiple optimizer groups."
        )
    if set(optimized_ids) != set(trainable_ids):
        missing = set(trainable_ids).difference(optimized_ids)
        extra = set(optimized_ids).difference(trainable_ids)
        raise ParameterGroupingError(
            "Optimizer group coverage mismatch: "
            f"missing_parameter_count={len(missing)}, extra_parameter_count={len(extra)}."
        )

    return _GroupingResult(
        records=tuple(records),
        groups=tuple(groups),
        reports=tuple(reports),
        layout_sha256=_parameter_layout_hash(records),
    )


# ---------------------------------------------------------------------------
# AdamW backend selection and construction
# ---------------------------------------------------------------------------


def _adamw_supports_fused() -> bool:
    try:
        return "fused" in inspect.signature(torch.optim.AdamW).parameters
    except (TypeError, ValueError):  # pragma: no cover - unusual builds
        return False


def _select_fused_backend(
    parameters: Sequence[nn.Parameter],
    *,
    distributed_strategy: str | None,
    allow_unfused_fallback: bool,
) -> tuple[bool, str | None]:
    reason: str | None = None
    if not _adamw_supports_fused():
        reason = "This PyTorch AdamW build does not expose fused=True."
    elif distributed_strategy == "fsdp2" or any(
        _parameter_contains_dtensor(parameter) for parameter in parameters
    ):
        # The safe FSDP2 path uses regular AdamW over sharded DTensor parameters.
        # DDP remains the high-throughput fused path.
        reason = "FSDP2/DTensor parameters use the conservative unfused AdamW path."
    elif not parameters:
        reason = "No parameters were supplied."
    elif any(parameter.device.type != "cuda" for parameter in parameters):
        devices = sorted({str(parameter.device) for parameter in parameters})
        reason = f"Fused AdamW requires CUDA parameters; found devices={devices}."

    if reason is None:
        return True, None
    if not allow_unfused_fallback:
        raise OptimizerConfigurationError(
            "optimizer='adamw_fused' could not enable the fused backend: " + reason
        )
    return False, reason


def _all_group_parameters(groups: Sequence[Mapping[str, Any]]) -> list[nn.Parameter]:
    parameters: list[nn.Parameter] = []
    for group in groups:
        raw = group.get("params")
        if not isinstance(raw, Sequence):
            raise ParameterGroupingError("Optimizer group 'params' is not a sequence.")
        for parameter in raw:
            if not isinstance(parameter, nn.Parameter):
                raise ParameterGroupingError(
                    f"Optimizer group contains {type(parameter).__name__}, not Parameter."
                )
            parameters.append(parameter)
    return parameters


def validate_optimizer_parameter_coverage(
    optimizer: Optimizer,
    model: M3DModel,
) -> None:
    """Ensure an existing optimizer covers each trainable parameter exactly once."""

    optimizer_parameters: list[nn.Parameter] = []
    for group in optimizer.param_groups:
        optimizer_parameters.extend(cast(Sequence[nn.Parameter], group["params"]))
    optimizer_ids = [id(parameter) for parameter in optimizer_parameters]
    trainable_ids = [
        id(parameter)
        for parameter in model.parameters()
        if parameter.requires_grad
    ]
    if len(optimizer_ids) != len(set(optimizer_ids)):
        raise ParameterGroupingError("Optimizer contains duplicate Parameter objects.")
    if set(optimizer_ids) != set(trainable_ids):
        raise ParameterGroupingError(
            "Optimizer no longer matches model trainability.  Build the optimizer only "
            "after all freezing, LoRA injection, vocabulary resize, and DDP/FSDP2 wrapping."
        )


def build_optimizer(
    model: M3DModel,
    config: ExperimentConfig | OptimizationConfig,
    *,
    distributed_strategy: str | None = None,
    allow_unfused_fallback: bool = True,
) -> tuple[AdamW, OptimizerBuildReport]:
    """Build component-aware AdamW after distributed model preparation.

    ``allow_unfused_fallback`` is useful for CPU tests and the conservative
    FSDP2/DTensor path.  On the primary DDP+A100 path, the report must show
    ``fused_enabled=true``; startup integration tests will enforce that.
    """

    if not isinstance(model, M3DModel):
        raise TypeError(f"model must be M3DModel, got {type(model).__name__}.")
    if distributed_strategy not in {None, "ddp", "fsdp2"}:
        raise OptimizerConfigurationError(
            "distributed_strategy must be None, 'ddp', or 'fsdp2', got "
            f"{distributed_strategy!r}."
        )

    optimization = _optimization_config(config)
    _validate_optimization_config(optimization)
    grouping = partition_trainable_parameters(model, optimization)
    mutable_groups = [dict(group) for group in grouping.groups]
    parameters = _all_group_parameters(mutable_groups)
    fused, fallback_reason = _select_fused_backend(
        parameters,
        distributed_strategy=distributed_strategy,
        allow_unfused_fallback=allow_unfused_fallback,
    )

    kwargs: dict[str, Any] = {
        "lr": 0.0,  # Every group has its own explicit lr.
        "betas": (float(optimization.betas[0]), float(optimization.betas[1])),
        "eps": float(optimization.epsilon),
        "weight_decay": 0.0,  # Every group has its own explicit decay.
    }
    if _adamw_supports_fused():
        kwargs["fused"] = bool(fused)
    # Explicitly avoid foreach for the conservative FSDP2/DTensor path.  The
    # regular single-tensor implementation has the broadest sharded support.
    if distributed_strategy == "fsdp2":
        kwargs["foreach"] = False

    optimizer = AdamW(mutable_groups, **kwargs)
    validate_optimizer_parameter_coverage(optimizer, model)

    canonical = _canonical_named_parameters(model)
    frozen_tensor_count = sum(
        1 for _, parameter in canonical if not parameter.requires_grad
    )
    trainable_tensor_count = sum(
        1 for _, parameter in canonical if parameter.requires_grad
    )
    trainable_elements = sum(
        int(parameter.numel())
        for _, parameter in canonical
        if parameter.requires_grad
    )
    optimized_tensor_count = sum(
        report.parameter_tensor_count for report in grouping.reports
    )
    optimized_elements = sum(
        report.parameter_element_count for report in grouping.reports
    )
    zero_lr_roles = tuple(
        role.value
        for role in ROLE_ORDER
        if any(record.role is role for record in grouping.records)
        and _learning_rate_for_role(optimization, role) == 0.0
    )
    active_roles = tuple(
        role.value
        for role in ROLE_ORDER
        if any(record.role is role for record in grouping.records)
    )

    report = OptimizerBuildReport(
        optimizer_name="torch.optim.AdamW",
        training_stage=str(optimization.stage),
        requested_fused=True,
        fused_enabled=bool(fused),
        fused_fallback_reason=fallback_reason,
        distributed_strategy=distributed_strategy,
        betas=(float(optimization.betas[0]), float(optimization.betas[1])),
        epsilon=float(optimization.epsilon),
        configured_weight_decay=float(optimization.weight_decay),
        trainable_parameter_tensor_count=trainable_tensor_count,
        trainable_parameter_element_count=trainable_elements,
        optimized_parameter_tensor_count=optimized_tensor_count,
        optimized_parameter_element_count=optimized_elements,
        frozen_parameter_tensor_count=frozen_tensor_count,
        zero_learning_rate_roles=zero_lr_roles,
        active_roles=active_roles,
        parameter_layout_sha256=grouping.layout_sha256,
        groups=grouping.reports,
    )
    return optimizer, report


# ---------------------------------------------------------------------------
# Stable metadata helpers used by schedulers and checkpoints
# ---------------------------------------------------------------------------


def optimizer_groups_by_name(optimizer: Optimizer) -> Mapping[str, Mapping[str, Any]]:
    """Return a validated mapping from stable group name to parameter group."""

    result: dict[str, Mapping[str, Any]] = {}
    for index, group in enumerate(optimizer.param_groups):
        name = group.get("group_name")
        if not isinstance(name, str) or not name:
            raise ParameterGroupingError(
                f"Optimizer group {index} has no stable 'group_name' metadata."
            )
        if name in result:
            raise ParameterGroupingError(f"Duplicate optimizer group name {name!r}.")
        result[name] = group
    return result


def optimizer_role_learning_rates(optimizer: Optimizer) -> Mapping[str, float]:
    """Return one current LR per role, validating decay/no-decay consistency."""

    by_role: MutableMapping[str, set[float]] = defaultdict(set)
    for group in optimizer.param_groups:
        role = group.get("role")
        if not isinstance(role, str) or not role:
            raise ParameterGroupingError("Optimizer group is missing role metadata.")
        by_role[role].add(float(group["lr"]))
    inconsistent = {
        role: sorted(values) for role, values in by_role.items() if len(values) != 1
    }
    if inconsistent:
        raise ParameterGroupingError(
            "Decay and no-decay groups for the same role have different learning "
            f"rates: {inconsistent}."
        )
    return {role: next(iter(values)) for role, values in by_role.items()}


def restore_optimizer_group_metadata(optimizer: Optimizer) -> None:
    """Validate metadata after ``optimizer.load_state_dict``.

    PyTorch copies saved parameter-group dictionaries into the live optimizer.
    This helper catches checkpoints made by an incompatible optimizer layout
    before a scheduler starts indexing groups.
    """

    groups = optimizer_groups_by_name(optimizer)
    for name, group in groups.items():
        role = group.get("role")
        decay_kind = group.get("decay_kind")
        param_names = group.get("param_names")
        if role not in {item.value for item in ParameterRole}:
            raise ParameterGroupingError(
                f"Optimizer group {name!r} has invalid role metadata {role!r}."
            )
        if decay_kind not in {item.value for item in DecayKind}:
            raise ParameterGroupingError(
                f"Optimizer group {name!r} has invalid decay_kind {decay_kind!r}."
            )
        if not isinstance(param_names, list) or not all(
            isinstance(item, str) for item in param_names
        ):
            raise ParameterGroupingError(
                f"Optimizer group {name!r} has invalid param_names metadata."
            )
        if len(param_names) != len(group["params"]):
            raise ParameterGroupingError(
                f"Optimizer group {name!r} parameter-name count does not match params."
            )


# ---------------------------------------------------------------------------
# Dependency-light self-test
# ---------------------------------------------------------------------------


def run_self_test() -> Mapping[str, Any]:
    # Private toy builders are imported only for this module self-test; the
    # production optimizer has no dependency on those helpers.
    from m3d.config import LearningRateConfig
    from m3d.data.schema import TaskName
    from m3d.model.m3d import _tiny_components, _toy_text_tensors

    torch.manual_seed(101)
    model, metadata = _tiny_components()
    model.train()

    optimization = OptimizationConfig(
        stage="joint_finetune",
        optimizer="adamw_fused",
        learning_rates=LearningRateConfig(
            language_model=1.0e-5,
            main_vision=2.0e-5,
            seg_vision=3.0e-5,
            projector=4.0e-5,
            segmentation_projector=5.0e-5,
            segmentation_decoder=6.0e-5,
            token_embeddings=7.0e-5,
        ),
        betas=(0.9, 0.95),
        epsilon=1.0e-8,
        weight_decay=0.1,
        checkpoint_language_model=False,
        checkpoint_main_vision=False,
        checkpoint_seg_vision=False,
        checkpoint_segmentation_decoder=False,
    )

    optimizer, report = build_optimizer(
        model,
        optimization,
        distributed_strategy=None,
        allow_unfused_fallback=True,
    )
    if report.fused_enabled:
        raise AssertionError("CPU self-test unexpectedly enabled fused AdamW.")
    if report.fused_fallback_reason is None:
        raise AssertionError("CPU fused fallback reason was not reported.")
    if (
        report.optimized_parameter_tensor_count
        != report.trainable_parameter_tensor_count
    ):
        raise AssertionError("Optimizer tensor coverage is incomplete.")
    if (
        report.optimized_parameter_element_count
        != report.trainable_parameter_element_count
    ):
        raise AssertionError("Optimizer element coverage is incomplete.")

    groups = optimizer_groups_by_name(optimizer)
    expected_roles = {role.value for role in ROLE_ORDER}
    actual_roles = {str(group["role"]) for group in groups.values()}
    if actual_roles != expected_roles:
        raise AssertionError(
            f"Toy model role coverage differs: {actual_roles} vs {expected_roles}."
        )

    role_lrs = optimizer_role_learning_rates(optimizer)
    for role in ROLE_ORDER:
        expected_lr = _learning_rate_for_role(optimization, role)
        if role_lrs[role.value] != expected_lr:
            raise AssertionError(f"Incorrect LR for role {role.value}.")

    # Explicit ViT position and CLS embeddings must be no-decay despite being
    # rank-3 tensors.
    main_position = "vision_tower.vision_tower.patch_embedding.position_embeddings"
    main_cls = "vision_tower.vision_tower.cls_token"
    no_decay_names = {
        name
        for group in optimizer.param_groups
        if group["decay_kind"] == DecayKind.NO_DECAY.value
        for name in group["param_names"]
    }
    if main_position not in no_decay_names or main_cls not in no_decay_names:
        raise AssertionError("Main ViT positional/CLS parameters received weight decay.")

    # The two image encoders must land in different optimizer roles.
    name_to_role = {
        name: str(group["role"])
        for group in optimizer.param_groups
        for name in group["param_names"]
    }
    if not all(
        role == ParameterRole.MAIN_VISION.value
        for name, role in name_to_role.items()
        if name.startswith("vision_tower.")
    ):
        raise AssertionError("Main vision parameter escaped the main_vision role.")
    if not all(
        role == ParameterRole.SEG_VISION.value
        for name, role in name_to_role.items()
        if name.startswith("seg_module.image_encoder.")
    ):
        raise AssertionError("SegVol image parameter escaped the seg_vision role.")

    # Run one real toy M3D step so AdamW creates state for used parameters.
    images = torch.randn(2, 1, 8, 16, 16)
    input_ids, attention_mask, labels = _toy_text_tensors(metadata)
    output = model(
        task=TaskName.CAPTION,
        images=images,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
    )
    if output.loss is None:
        raise AssertionError("Toy text step did not produce a loss.")
    output.loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if not optimizer.state:
        raise AssertionError("AdamW did not create any optimizer state.")

    saved = optimizer.state_dict()
    if not all("group_name" in group for group in saved["param_groups"]):
        raise AssertionError("Stable group metadata is absent from optimizer state.")

    # Rebuild the same architecture and verify state-dict compatibility.
    torch.manual_seed(101)
    restored_model, _ = _tiny_components()
    restored_optimizer, restored_report = build_optimizer(
        restored_model,
        optimization,
        distributed_strategy=None,
        allow_unfused_fallback=True,
    )
    if restored_report.parameter_layout_sha256 != report.parameter_layout_sha256:
        raise AssertionError("Equivalent models produced different optimizer layouts.")
    restored_optimizer.load_state_dict(saved)
    restore_optimizer_group_metadata(restored_optimizer)
    if set(optimizer_groups_by_name(restored_optimizer)) != set(groups):
        raise AssertionError("Optimizer group names changed after state restore.")

    # Unknown trainable parameters must never be silently assigned a generic LR.
    model.register_parameter(
        "unexpected_trainable_parameter",
        nn.Parameter(torch.ones(3, 3)),
    )
    unknown_parameter_detected = False
    try:
        partition_trainable_parameters(model, optimization)
    except ParameterGroupingError:
        unknown_parameter_detected = True
    if not unknown_parameter_detected:
        raise AssertionError("Unknown trainable parameter was silently optimized.")

    return {
        "status": "passed",
        "fused_enabled_on_cpu": report.fused_enabled,
        "fused_fallback_reported": report.fused_fallback_reason is not None,
        "optimizer_group_count": len(report.groups),
        "active_roles": list(report.active_roles),
        "trainable_parameter_tensor_count": report.trainable_parameter_tensor_count,
        "trainable_parameter_element_count": report.trainable_parameter_element_count,
        "parameter_layout_sha256": report.parameter_layout_sha256,
        "main_and_seg_vision_roles_separate": True,
        "position_embeddings_no_decay": True,
        "optimizer_state_restore": True,
        "unknown_parameter_detected": unknown_parameter_detected,
    }


def _main() -> None:
    print(json.dumps(run_self_test(), indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
