"""Deterministic warmup/cosine scheduling for modernized M3D training.

The M3D data sampler counts *microbatches*, while the learning-rate scheduler
must count successful optimizer updates.  Gradient accumulation and partial
final epochs mean those counters are not interchangeable.  This module builds
an explicit immutable training-step plan first, then applies one shared scalar
multiplier to every component-specific AdamW learning rate.

The schedule deliberately preserves the relative learning rates created by
:mod:`m3d.optim`::

    language_model             1.0e-5
    main_vision                5.0e-6
    seg_vision                 5.0e-6
    projector                  5.0e-5
    segmentation_projector     5.0e-5
    segmentation_decoder       1.0e-5
    token_embeddings           5.0e-5

Only the common multiplier changes.  Main 3D ViT and SegVol 3D ViT therefore
remain separate optimizer roles throughout warmup, cosine decay, checkpoint
save, and exact resume.

Public construction::

    scheduler, report = build_scheduler(
        optimizer,
        config,
        steps_per_epoch=training_pipeline.steps_per_epoch,
    )

Training order::

    optimizer.step()
    scheduler.step()  # exactly once after every successful optimizer update

Do not call ``scheduler.step()`` for skipped/non-finite updates.  Checkpoints
should normally be written only at optimizer-update boundaries; the plan can
validate the sampler cursor against the scheduler counter before saving or
resuming.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from decimal import Decimal, ROUND_CEILING
from types import MappingProxyType
from typing import Any, Mapping, MutableMapping, Sequence

import torch
from torch.optim import Optimizer

from m3d.config import ExperimentConfig, OptimizationConfig
from m3d.optim import (
    optimizer_groups_by_name,
    optimizer_role_learning_rates,
    restore_optimizer_group_metadata,
)


class SchedulerConfigurationError(ValueError):
    """Raised when schedule configuration or optimizer metadata is invalid."""


class SchedulerStateError(RuntimeError):
    """Raised when a scheduler checkpoint cannot be resumed exactly."""


_SCHEDULER_STATE_VERSION = 1


def _as_positive_int(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if value <= 0:
        raise SchedulerConfigurationError(f"{name} must be positive, got {value}.")
    return value


def _as_nonnegative_int(value: Any, *, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
    if value < 0:
        raise SchedulerConfigurationError(f"{name} cannot be negative, got {value}.")
    return value


def _optimization_config(
    config: ExperimentConfig | OptimizationConfig,
) -> OptimizationConfig:
    if isinstance(config, ExperimentConfig):
        return config.optimization
    if isinstance(config, OptimizationConfig):
        return config
    raise TypeError(
        "config must be ExperimentConfig or OptimizationConfig, got "
        f"{type(config).__name__}."
    )


def _decimal_epochs(value: Any) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise TypeError("optimization.epochs must be numeric")
    decimal_value = Decimal(str(value))
    if not decimal_value.is_finite() or decimal_value <= 0:
        raise SchedulerConfigurationError(
            f"optimization.epochs must be finite and positive, got {value!r}."
        )
    return decimal_value


def _ceil_decimal(value: Decimal) -> int:
    return int(value.to_integral_value(rounding=ROUND_CEILING))


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True, slots=True)
class TrainingStepPlan:
    """Exact mapping between epoch microbatches and optimizer updates.

    Every epoch flushes a partial gradient-accumulation window at its own end.
    Gradients are never carried across a sampler epoch boundary.  For example,
    ten microbatches with accumulation four produce windows ``4, 4, 2`` and
    therefore three optimizer updates.

    Fractional epochs consume a prefix of the final sampler epoch.  The prefix
    length is ``ceil(fraction * steps_per_epoch)`` so a positive requested
    fraction always executes at least one microbatch.
    """

    configured_epochs: str
    steps_per_epoch: int
    gradient_accumulation_steps: int
    epoch_microbatch_limits: tuple[int, ...]
    optimizer_steps_per_epoch: tuple[int, ...]
    total_microbatches: int
    total_optimizer_steps: int
    warmup_ratio: float
    warmup_steps: int
    plan_sha256: str

    def __post_init__(self) -> None:
        _as_positive_int(self.steps_per_epoch, name="steps_per_epoch")
        _as_positive_int(
            self.gradient_accumulation_steps,
            name="gradient_accumulation_steps",
        )
        if not self.epoch_microbatch_limits:
            raise SchedulerConfigurationError(
                "epoch_microbatch_limits cannot be empty"
            )
        if len(self.epoch_microbatch_limits) != len(
            self.optimizer_steps_per_epoch
        ):
            raise SchedulerConfigurationError(
                "epoch microbatch and optimizer-step plans have different lengths"
            )
        if any(
            limit <= 0 or limit > self.steps_per_epoch
            for limit in self.epoch_microbatch_limits
        ):
            raise SchedulerConfigurationError(
                "Every epoch microbatch limit must be in "
                f"[1, {self.steps_per_epoch}]."
            )
        expected_updates = tuple(
            math.ceil(limit / self.gradient_accumulation_steps)
            for limit in self.epoch_microbatch_limits
        )
        if expected_updates != self.optimizer_steps_per_epoch:
            raise SchedulerConfigurationError(
                "optimizer_steps_per_epoch does not match accumulation windows: "
                f"expected={expected_updates}, got={self.optimizer_steps_per_epoch}."
            )
        if sum(self.epoch_microbatch_limits) != self.total_microbatches:
            raise SchedulerConfigurationError("total_microbatches is inconsistent")
        if sum(self.optimizer_steps_per_epoch) != self.total_optimizer_steps:
            raise SchedulerConfigurationError("total_optimizer_steps is inconsistent")
        if not 0.0 <= self.warmup_ratio < 1.0:
            raise SchedulerConfigurationError("warmup_ratio must be in [0, 1)")
        if not 0 <= self.warmup_steps <= self.total_optimizer_steps:
            raise SchedulerConfigurationError(
                "warmup_steps must be between zero and total optimizer steps"
            )
        if not isinstance(self.plan_sha256, str) or len(self.plan_sha256) != 64:
            raise SchedulerConfigurationError("plan_sha256 must be a SHA-256 hex digest")

    @property
    def epoch_count(self) -> int:
        return len(self.epoch_microbatch_limits)

    @property
    def decay_steps(self) -> int:
        return self.total_optimizer_steps - self.warmup_steps

    @property
    def is_fractional_final_epoch(self) -> bool:
        return self.epoch_microbatch_limits[-1] != self.steps_per_epoch

    def microbatches_for_epoch(self, epoch: int) -> int:
        self._validate_epoch(epoch)
        return self.epoch_microbatch_limits[epoch]

    def optimizer_steps_for_epoch(self, epoch: int) -> int:
        self._validate_epoch(epoch)
        return self.optimizer_steps_per_epoch[epoch]

    def global_microbatch_offset(self, epoch: int) -> int:
        self._validate_epoch_or_end(epoch)
        return sum(self.epoch_microbatch_limits[:epoch])

    def global_optimizer_step_offset(self, epoch: int) -> int:
        self._validate_epoch_or_end(epoch)
        return sum(self.optimizer_steps_per_epoch[:epoch])

    def accumulation_window_bounds(
        self,
        epoch: int,
        microbatch_index: int,
    ) -> tuple[int, int]:
        """Return ``[start, end)`` for one epoch-local accumulation window."""

        limit = self.microbatches_for_epoch(epoch)
        if not isinstance(microbatch_index, int) or isinstance(microbatch_index, bool):
            raise TypeError("microbatch_index must be an integer")
        if not 0 <= microbatch_index < limit:
            raise SchedulerConfigurationError(
                f"microbatch_index must be in [0, {limit}), got {microbatch_index}."
            )
        accumulation = self.gradient_accumulation_steps
        start = (microbatch_index // accumulation) * accumulation
        end = min(start + accumulation, limit)
        return start, end

    def accumulation_window_size(self, epoch: int, microbatch_index: int) -> int:
        start, end = self.accumulation_window_bounds(epoch, microbatch_index)
        return end - start

    def is_optimizer_update_microbatch(
        self,
        epoch: int,
        microbatch_index: int,
    ) -> bool:
        _, end = self.accumulation_window_bounds(epoch, microbatch_index)
        return microbatch_index + 1 == end

    def completed_updates_at_position(
        self,
        epoch: int,
        committed_microbatches: int,
        *,
        require_update_boundary: bool = True,
    ) -> int:
        """Return expected completed optimizer updates at a sampler cursor.

        ``committed_microbatches`` is the number of successfully consumed
        microbatches in ``epoch``.  Exact checkpoint resume normally requires
        it to lie at an accumulation-window boundary because gradients from a
        partially accumulated window are not represented by optimizer or
        scheduler state.
        """

        limit = self.microbatches_for_epoch(epoch)
        committed = _as_nonnegative_int(
            committed_microbatches,
            name="committed_microbatches",
        )
        if committed > limit:
            raise SchedulerStateError(
                f"Epoch {epoch} permits only {limit} microbatches, but the sampler "
                f"cursor is {committed}."
            )
        accumulation = self.gradient_accumulation_steps
        is_boundary = (
            committed == 0
            or committed == limit
            or committed % accumulation == 0
        )
        if require_update_boundary and not is_boundary:
            window_start = (committed // accumulation) * accumulation
            window_end = min(window_start + accumulation, limit)
            raise SchedulerStateError(
                "Checkpoint cursor is inside a gradient-accumulation window: "
                f"epoch={epoch}, committed_microbatches={committed}, "
                f"window=[{window_start}, {window_end}). Save checkpoints only "
                "after optimizer updates unless gradient buffers are also saved."
            )
        local_updates = math.ceil(committed / accumulation) if committed else 0
        return self.global_optimizer_step_offset(epoch) + local_updates

    def validate_scheduler_position(
        self,
        *,
        epoch: int,
        committed_microbatches: int,
        completed_optimizer_steps: int,
        require_update_boundary: bool = True,
    ) -> None:
        expected = self.completed_updates_at_position(
            epoch,
            committed_microbatches,
            require_update_boundary=require_update_boundary,
        )
        completed = _as_nonnegative_int(
            completed_optimizer_steps,
            name="completed_optimizer_steps",
        )
        if completed != expected:
            raise SchedulerStateError(
                "Sampler and scheduler progress disagree: "
                f"epoch={epoch}, committed_microbatches={committed_microbatches}, "
                f"expected_optimizer_steps={expected}, "
                f"scheduler_optimizer_steps={completed}."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "configured_epochs": self.configured_epochs,
            "steps_per_epoch": self.steps_per_epoch,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "epoch_microbatch_limits": list(self.epoch_microbatch_limits),
            "optimizer_steps_per_epoch": list(self.optimizer_steps_per_epoch),
            "total_microbatches": self.total_microbatches,
            "total_optimizer_steps": self.total_optimizer_steps,
            "warmup_ratio": self.warmup_ratio,
            "warmup_steps": self.warmup_steps,
            "decay_steps": self.decay_steps,
            "epoch_count": self.epoch_count,
            "is_fractional_final_epoch": self.is_fractional_final_epoch,
            "plan_sha256": self.plan_sha256,
        }

    def _validate_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or isinstance(epoch, bool):
            raise TypeError("epoch must be an integer")
        if not 0 <= epoch < self.epoch_count:
            raise SchedulerConfigurationError(
                f"epoch must be in [0, {self.epoch_count}), got {epoch}."
            )

    def _validate_epoch_or_end(self, epoch: int) -> None:
        if not isinstance(epoch, int) or isinstance(epoch, bool):
            raise TypeError("epoch must be an integer")
        if not 0 <= epoch <= self.epoch_count:
            raise SchedulerConfigurationError(
                f"epoch must be in [0, {self.epoch_count}], got {epoch}."
            )


def build_training_step_plan(
    *,
    steps_per_epoch: int,
    epochs: int | float | Decimal,
    gradient_accumulation_steps: int,
    warmup_ratio: float,
) -> TrainingStepPlan:
    """Create the exact microbatch/update schedule used by training."""

    steps = _as_positive_int(steps_per_epoch, name="steps_per_epoch")
    accumulation = _as_positive_int(
        gradient_accumulation_steps,
        name="gradient_accumulation_steps",
    )
    epoch_decimal = _decimal_epochs(epochs)
    if isinstance(warmup_ratio, bool) or not isinstance(warmup_ratio, (int, float)):
        raise TypeError("warmup_ratio must be numeric")
    warmup = float(warmup_ratio)
    if not math.isfinite(warmup) or not 0.0 <= warmup < 1.0:
        raise SchedulerConfigurationError(
            f"warmup_ratio must be finite and in [0, 1), got {warmup_ratio!r}."
        )

    full_epochs = int(epoch_decimal)
    fractional = epoch_decimal - Decimal(full_epochs)
    epoch_limits: list[int] = [steps] * full_epochs
    if fractional > 0:
        fractional_steps = _ceil_decimal(fractional * Decimal(steps))
        epoch_limits.append(max(1, min(steps, fractional_steps)))
    if not epoch_limits:
        # This branch is reachable only for a positive epochs value below 1.
        epoch_limits.append(max(1, _ceil_decimal(epoch_decimal * Decimal(steps))))

    optimizer_steps = [math.ceil(limit / accumulation) for limit in epoch_limits]
    total_microbatches = sum(epoch_limits)
    total_updates = sum(optimizer_steps)
    warmup_steps = min(total_updates, math.ceil(total_updates * warmup))

    fingerprint_payload = {
        "configured_epochs": str(epoch_decimal),
        "steps_per_epoch": steps,
        "gradient_accumulation_steps": accumulation,
        "epoch_microbatch_limits": epoch_limits,
        "optimizer_steps_per_epoch": optimizer_steps,
        "total_microbatches": total_microbatches,
        "total_optimizer_steps": total_updates,
        "warmup_ratio": warmup,
        "warmup_steps": warmup_steps,
        "schedule": "linear_warmup_cosine_decay",
    }
    fingerprint = _sha256_json(fingerprint_payload)

    return TrainingStepPlan(
        configured_epochs=str(epoch_decimal),
        steps_per_epoch=steps,
        gradient_accumulation_steps=accumulation,
        epoch_microbatch_limits=tuple(epoch_limits),
        optimizer_steps_per_epoch=tuple(optimizer_steps),
        total_microbatches=total_microbatches,
        total_optimizer_steps=total_updates,
        warmup_ratio=warmup,
        warmup_steps=warmup_steps,
        plan_sha256=fingerprint,
    )


def cosine_warmup_multiplier(
    completed_optimizer_steps: int,
    *,
    total_optimizer_steps: int,
    warmup_steps: int,
) -> float:
    """Return LR scale for the *next* optimizer update.

    This follows the common linear-warmup/cosine convention used by modern
    Transformer training:

    * during warmup: ``completed_steps / warmup_steps``;
    * after warmup: ``0.5 * (1 + cos(pi * progress))``;
    * after all planned updates: exactly zero.

    Because the scheduler is stepped after ``optimizer.step()``, a non-zero
    warmup begins with an LR of zero for the first update and reaches the base
    LR immediately after the warmup counter is completed.
    """

    completed = _as_nonnegative_int(
        completed_optimizer_steps,
        name="completed_optimizer_steps",
    )
    total = _as_positive_int(total_optimizer_steps, name="total_optimizer_steps")
    warmup = _as_nonnegative_int(warmup_steps, name="warmup_steps")
    if warmup > total:
        raise SchedulerConfigurationError(
            f"warmup_steps={warmup} exceeds total_optimizer_steps={total}."
        )
    if completed >= total:
        return 0.0
    if warmup > 0 and completed < warmup:
        return float(completed) / float(warmup)
    decay_steps = total - warmup
    if decay_steps <= 0:
        return 0.0
    progress = float(completed - warmup) / float(decay_steps)
    progress = min(max(progress, 0.0), 1.0)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass(frozen=True, slots=True)
class SchedulerGroupReport:
    group_name: str
    role: str
    base_learning_rate: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "group_name": self.group_name,
            "role": self.role,
            "base_learning_rate": self.base_learning_rate,
        }


@dataclass(frozen=True, slots=True)
class SchedulerBuildReport:
    scheduler_name: str
    scheduler_type: str
    plan: TrainingStepPlan
    optimizer_group_layout_sha256: str
    groups: tuple[SchedulerGroupReport, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheduler_name": self.scheduler_name,
            "scheduler_type": self.scheduler_type,
            "plan": self.plan.to_dict(),
            "optimizer_group_layout_sha256": self.optimizer_group_layout_sha256,
            "groups": [group.to_dict() for group in self.groups],
        }

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


def _validated_optimizer_group_records(
    optimizer: Optimizer,
) -> tuple[tuple[SchedulerGroupReport, ...], str]:
    restore_optimizer_group_metadata(optimizer)
    groups_by_name = optimizer_groups_by_name(optimizer)
    records: list[SchedulerGroupReport] = []
    fingerprint_groups: list[dict[str, Any]] = []

    for group_name in sorted(groups_by_name):
        group = groups_by_name[group_name]
        role = group.get("role")
        if not isinstance(role, str) or not role:
            raise SchedulerConfigurationError(
                f"Optimizer group {group_name!r} has no role metadata."
            )
        initial_lr = group.get("initial_lr")
        if isinstance(initial_lr, bool) or not isinstance(initial_lr, (int, float)):
            raise SchedulerConfigurationError(
                f"Optimizer group {group_name!r} has invalid initial_lr "
                f"metadata {initial_lr!r}."
            )
        base_lr = float(initial_lr)
        if not math.isfinite(base_lr) or base_lr < 0.0:
            raise SchedulerConfigurationError(
                f"Optimizer group {group_name!r} has invalid base LR {base_lr}."
            )
        param_names = group.get("param_names")
        if not isinstance(param_names, list) or not all(
            isinstance(name, str) for name in param_names
        ):
            raise SchedulerConfigurationError(
                f"Optimizer group {group_name!r} has invalid param_names metadata."
            )
        record = SchedulerGroupReport(
            group_name=group_name,
            role=role,
            base_learning_rate=base_lr,
        )
        records.append(record)
        fingerprint_groups.append(
            {
                "group_name": group_name,
                "role": role,
                "initial_lr": base_lr,
                "decay_kind": group.get("decay_kind"),
                "param_names": param_names,
            }
        )

    if not records:
        raise SchedulerConfigurationError("Optimizer has no parameter groups")

    # Decay and no-decay groups belonging to one role must start from the same
    # LR before applying the common schedule multiplier.
    role_to_base_lrs: MutableMapping[str, set[float]] = {}
    for record in records:
        role_to_base_lrs.setdefault(record.role, set()).add(record.base_learning_rate)
    inconsistent = {
        role: sorted(values)
        for role, values in role_to_base_lrs.items()
        if len(values) != 1
    }
    if inconsistent:
        raise SchedulerConfigurationError(
            "Optimizer decay/no-decay groups disagree on initial LR: "
            f"{inconsistent}."
        )

    fingerprint = _sha256_json({"groups": fingerprint_groups})
    return tuple(records), fingerprint


class CosineWarmupScheduler:
    """Checkpoint-exact scheduler independent of optimizer group positions."""

    def __init__(
        self,
        optimizer: Optimizer,
        plan: TrainingStepPlan,
        *,
        optimizer_group_layout_sha256: str,
    ) -> None:
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be a torch.optim.Optimizer")
        if not isinstance(plan, TrainingStepPlan):
            raise TypeError("plan must be TrainingStepPlan")
        if (
            not isinstance(optimizer_group_layout_sha256, str)
            or len(optimizer_group_layout_sha256) != 64
        ):
            raise SchedulerConfigurationError(
                "optimizer_group_layout_sha256 must be a SHA-256 hex digest"
            )

        self.optimizer = optimizer
        self.plan = plan
        self.optimizer_group_layout_sha256 = optimizer_group_layout_sha256
        self.completed_optimizer_steps = 0
        self._groups, observed_layout = _validated_optimizer_group_records(optimizer)
        if observed_layout != optimizer_group_layout_sha256:
            raise SchedulerConfigurationError(
                "Optimizer group layout changed while constructing scheduler: "
                f"expected={optimizer_group_layout_sha256}, observed={observed_layout}."
            )
        self._base_lrs = {
            group.group_name: group.base_learning_rate for group in self._groups
        }
        self._apply_learning_rates()

    @property
    def total_optimizer_steps(self) -> int:
        return self.plan.total_optimizer_steps

    @property
    def warmup_steps(self) -> int:
        return self.plan.warmup_steps

    @property
    def is_finished(self) -> bool:
        return self.completed_optimizer_steps >= self.total_optimizer_steps

    @property
    def current_multiplier(self) -> float:
        return cosine_warmup_multiplier(
            self.completed_optimizer_steps,
            total_optimizer_steps=self.total_optimizer_steps,
            warmup_steps=self.warmup_steps,
        )

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def learning_rates_by_group(self) -> Mapping[str, float]:
        result = {
            str(group["group_name"]): float(group["lr"])
            for group in self.optimizer.param_groups
        }
        return MappingProxyType(result)

    def learning_rates_by_role(self) -> Mapping[str, float]:
        return MappingProxyType(dict(optimizer_role_learning_rates(self.optimizer)))

    def step(self) -> None:
        """Advance after exactly one successful ``optimizer.step()`` call."""

        if self.completed_optimizer_steps >= self.total_optimizer_steps:
            raise SchedulerStateError(
                "scheduler.step() was called after the complete training plan: "
                f"completed={self.completed_optimizer_steps}, "
                f"total={self.total_optimizer_steps}."
            )
        self.completed_optimizer_steps += 1
        self._apply_learning_rates()

    def set_completed_optimizer_steps(self, completed_steps: int) -> None:
        """Set progress explicitly, primarily for validated checkpoint resume."""

        completed = _as_nonnegative_int(completed_steps, name="completed_steps")
        if completed > self.total_optimizer_steps:
            raise SchedulerStateError(
                f"completed_steps={completed} exceeds total_optimizer_steps="
                f"{self.total_optimizer_steps}."
            )
        self.completed_optimizer_steps = completed
        self._apply_learning_rates()

    def validate_training_position(
        self,
        *,
        epoch: int,
        committed_microbatches: int,
        require_update_boundary: bool = True,
    ) -> None:
        self.plan.validate_scheduler_position(
            epoch=epoch,
            committed_microbatches=committed_microbatches,
            completed_optimizer_steps=self.completed_optimizer_steps,
            require_update_boundary=require_update_boundary,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_version": _SCHEDULER_STATE_VERSION,
            "scheduler_type": "linear_warmup_cosine_decay",
            "completed_optimizer_steps": self.completed_optimizer_steps,
            "total_optimizer_steps": self.total_optimizer_steps,
            "warmup_steps": self.warmup_steps,
            "plan_sha256": self.plan.plan_sha256,
            "optimizer_group_layout_sha256": self.optimizer_group_layout_sha256,
            "base_lrs": dict(self._base_lrs),
            "current_multiplier": self.current_multiplier,
            "current_lrs": dict(self.learning_rates_by_group()),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("Scheduler state must be a mapping")
        if state.get("state_version") != _SCHEDULER_STATE_VERSION:
            raise SchedulerStateError(
                "Unsupported scheduler state version "
                f"{state.get('state_version')!r}."
            )
        expected_scalars = {
            "scheduler_type": "linear_warmup_cosine_decay",
            "total_optimizer_steps": self.total_optimizer_steps,
            "warmup_steps": self.warmup_steps,
            "plan_sha256": self.plan.plan_sha256,
            "optimizer_group_layout_sha256": self.optimizer_group_layout_sha256,
        }
        mismatches = {
            key: {"checkpoint": state.get(key), "current": current}
            for key, current in expected_scalars.items()
            if state.get(key) != current
        }
        if mismatches:
            raise SchedulerStateError(
                "Scheduler checkpoint is incompatible with the current training "
                f"plan or optimizer layout: {mismatches}."
            )

        checkpoint_base_lrs = state.get("base_lrs")
        if not isinstance(checkpoint_base_lrs, Mapping):
            raise SchedulerStateError("Scheduler checkpoint has no base_lrs mapping")
        normalized_base_lrs: dict[str, float] = {}
        for name, value in checkpoint_base_lrs.items():
            if not isinstance(name, str):
                raise SchedulerStateError("Scheduler base_lrs keys must be strings")
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise SchedulerStateError(
                    f"Scheduler base LR for {name!r} is invalid: {value!r}."
                )
            normalized_base_lrs[name] = float(value)
        if normalized_base_lrs != self._base_lrs:
            raise SchedulerStateError(
                "Scheduler base learning rates changed: "
                f"checkpoint={normalized_base_lrs}, current={self._base_lrs}."
            )

        completed = state.get("completed_optimizer_steps")
        self.set_completed_optimizer_steps(
            _as_nonnegative_int(completed, name="completed_optimizer_steps")
        )

        # Never trust serialized current LRs.  They are redundant state and may
        # have been written between optimizer.step() and scheduler.step().
        # Recomputing from the durable counter guarantees exact recovery.
        expected_current = dict(self.learning_rates_by_group())
        checkpoint_current = state.get("current_lrs")
        if isinstance(checkpoint_current, Mapping):
            normalized_current = {
                str(name): float(value)
                for name, value in checkpoint_current.items()
            }
            for name, expected in expected_current.items():
                observed = normalized_current.get(name)
                if observed is None or not math.isclose(
                    observed,
                    expected,
                    rel_tol=1e-12,
                    abs_tol=0.0,
                ):
                    raise SchedulerStateError(
                        "Serialized scheduler LR does not match the recomputed "
                        f"state for group {name!r}: checkpoint={observed}, "
                        f"recomputed={expected}."
                    )

    def _apply_learning_rates(self) -> None:
        multiplier = self.current_multiplier
        groups_by_name = optimizer_groups_by_name(self.optimizer)
        if set(groups_by_name) != set(self._base_lrs):
            raise SchedulerStateError(
                "Optimizer groups changed after scheduler construction: "
                f"expected={sorted(self._base_lrs)}, "
                f"observed={sorted(groups_by_name)}."
            )
        for group_name, group in groups_by_name.items():
            group["lr"] = self._base_lrs[group_name] * multiplier



def build_scheduler(
    optimizer: Optimizer,
    config: ExperimentConfig | OptimizationConfig,
    *,
    steps_per_epoch: int,
) -> tuple[CosineWarmupScheduler, SchedulerBuildReport]:
    """Build the configured scheduler from the full sampler epoch length."""

    optimization = _optimization_config(config)
    if optimization.scheduler != "cosine":
        raise SchedulerConfigurationError(
            f"Unsupported scheduler {optimization.scheduler!r}; expected 'cosine'."
        )
    plan = build_training_step_plan(
        steps_per_epoch=steps_per_epoch,
        epochs=optimization.epochs,
        gradient_accumulation_steps=optimization.gradient_accumulation_steps,
        warmup_ratio=optimization.warmup_ratio,
    )
    group_reports, layout_sha256 = _validated_optimizer_group_records(optimizer)
    scheduler = CosineWarmupScheduler(
        optimizer,
        plan,
        optimizer_group_layout_sha256=layout_sha256,
    )
    report = SchedulerBuildReport(
        scheduler_name="CosineWarmupScheduler",
        scheduler_type="linear_warmup_cosine_decay",
        plan=plan,
        optimizer_group_layout_sha256=layout_sha256,
        groups=group_reports,
    )
    return scheduler, report


def validate_resume_progress(
    scheduler: CosineWarmupScheduler,
    *,
    epoch: int,
    committed_microbatches: int,
) -> None:
    """Validate sampler/scheduler agreement after both states are loaded."""

    if not isinstance(scheduler, CosineWarmupScheduler):
        raise TypeError("scheduler must be CosineWarmupScheduler")
    scheduler.validate_training_position(
        epoch=epoch,
        committed_microbatches=committed_microbatches,
        require_update_boundary=True,
    )


# ---------------------------------------------------------------------------
# Dependency-light self-test
# ---------------------------------------------------------------------------


def _toy_optimizer() -> Optimizer:
    first = torch.nn.Parameter(torch.tensor([1.0]))
    second = torch.nn.Parameter(torch.tensor([2.0]))
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [first],
                "lr": 1.0e-5,
                "initial_lr": 1.0e-5,
                "weight_decay": 0.0,
                "group_name": "language_model/decay",
                "role": "language_model",
                "decay_kind": "decay",
                "param_names": ["language.weight"],
            },
            {
                "params": [second],
                "lr": 5.0e-6,
                "initial_lr": 5.0e-6,
                "weight_decay": 0.0,
                "group_name": "main_vision/decay",
                "role": "main_vision",
                "decay_kind": "decay",
                "param_names": ["vision.weight"],
            },
        ],
        lr=0.0,
    )
    return optimizer


def run_self_test() -> Mapping[str, Any]:
    plan = build_training_step_plan(
        steps_per_epoch=10,
        epochs=2.5,
        gradient_accumulation_steps=4,
        warmup_ratio=0.25,
    )
    assert plan.epoch_microbatch_limits == (10, 10, 5)
    assert plan.optimizer_steps_per_epoch == (3, 3, 2)
    assert plan.total_microbatches == 25
    assert plan.total_optimizer_steps == 8
    assert plan.warmup_steps == 2

    first_epoch_update_indices = tuple(
        index
        for index in range(plan.microbatches_for_epoch(0))
        if plan.is_optimizer_update_microbatch(0, index)
    )
    assert first_epoch_update_indices == (3, 7, 9)
    assert plan.accumulation_window_size(0, 8) == 2
    assert plan.accumulation_window_size(2, 4) == 1
    assert plan.completed_updates_at_position(1, 8) == 5

    mid_window_detected = False
    try:
        plan.completed_updates_at_position(0, 3)
    except SchedulerStateError:
        mid_window_detected = True
    assert mid_window_detected

    optimizer = _toy_optimizer()
    records, layout = _validated_optimizer_group_records(optimizer)
    scheduler = CosineWarmupScheduler(
        optimizer,
        plan,
        optimizer_group_layout_sha256=layout,
    )
    assert scheduler.current_multiplier == 0.0
    assert scheduler.get_last_lr() == [0.0, 0.0]

    multipliers = [scheduler.current_multiplier]
    role_ratios: list[float] = []
    for _ in range(plan.total_optimizer_steps):
        # The optimizer update would occur here with the currently assigned LR.
        scheduler.step()
        multipliers.append(scheduler.current_multiplier)
        by_role = scheduler.learning_rates_by_role()
        if by_role["main_vision"] > 0:
            role_ratios.append(
                by_role["language_model"] / by_role["main_vision"]
            )
    assert scheduler.is_finished
    assert multipliers[0] == 0.0
    assert math.isclose(multipliers[1], 0.5)
    assert math.isclose(multipliers[2], 1.0)
    assert multipliers[-1] == 0.0
    assert all(math.isclose(ratio, 2.0) for ratio in role_ratios)

    saved = scheduler.state_dict()
    restored_optimizer = _toy_optimizer()
    restored_records, restored_layout = _validated_optimizer_group_records(
        restored_optimizer
    )
    assert records == restored_records
    assert layout == restored_layout
    restored = CosineWarmupScheduler(
        restored_optimizer,
        plan,
        optimizer_group_layout_sha256=restored_layout,
    )
    restored.load_state_dict(saved)
    assert restored.completed_optimizer_steps == scheduler.completed_optimizer_steps
    assert restored.get_last_lr() == scheduler.get_last_lr()

    # Resume at epoch 1 after eight microbatches means three completed updates
    # from epoch 0 and two from epoch 1.
    progress_optimizer = _toy_optimizer()
    _, progress_layout = _validated_optimizer_group_records(progress_optimizer)
    progress_scheduler = CosineWarmupScheduler(
        progress_optimizer,
        plan,
        optimizer_group_layout_sha256=progress_layout,
    )
    progress_scheduler.set_completed_optimizer_steps(5)
    progress_scheduler.validate_training_position(
        epoch=1,
        committed_microbatches=8,
    )

    disagreement_detected = False
    progress_scheduler.set_completed_optimizer_steps(4)
    try:
        progress_scheduler.validate_training_position(
            epoch=1,
            committed_microbatches=8,
        )
    except SchedulerStateError:
        disagreement_detected = True
    assert disagreement_detected

    return MappingProxyType(
        {
            "status": "passed",
            "epoch_microbatch_limits": list(plan.epoch_microbatch_limits),
            "optimizer_steps_per_epoch": list(plan.optimizer_steps_per_epoch),
            "total_microbatches": plan.total_microbatches,
            "total_optimizer_steps": plan.total_optimizer_steps,
            "warmup_steps": plan.warmup_steps,
            "first_epoch_update_indices": list(first_epoch_update_indices),
            "partial_accumulation_window_size": plan.accumulation_window_size(0, 8),
            "fractional_final_window_size": plan.accumulation_window_size(2, 4),
            "mid_window_checkpoint_detected": mid_window_detected,
            "sampler_scheduler_disagreement_detected": disagreement_detected,
            "relative_role_learning_rates_preserved": True,
            "state_restore_exact": True,
            "final_multiplier": scheduler.current_multiplier,
            "plan_sha256": plan.plan_sha256,
        }
    )


def main() -> None:
    print(json.dumps(dict(run_self_test()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
