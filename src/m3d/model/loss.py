"""Loss functions and task-aware objective composition for M3D-Modernized.

This module reproduces the original M3D segmentation objective while exposing
sufficient numerators and normalisers for correct distributed reduction.

Original M3D objective
----------------------
For every task, Phi-3 contributes its causal language-model loss.  A
segmentation batch additionally contributes binary Dice loss and
BCE-with-logits loss::

    text task:
        total = language

    segmentation task:
        total = language + dice_weight * dice + bce_weight * bce

Task identity is explicit.  A segmentation branch is selected because
``task is TaskName.SEGMENTATION`` -- never because a target contains foreground
voxels.  An all-zero target is therefore a valid segmentation target.

Numerical policy
----------------
The model may run under BF16 autocast, but reductions are accumulated in FP32.
This keeps the original formulas while avoiding low-precision sums over large
3D volumes.  Raw mask logits are consumed directly; callers must not apply
``sigmoid`` or thresholding before this module.

Distributed policy
------------------
``M3DLossOutput.total`` is the exact local-mean objective and reproduces the
original single-process behaviour.  ``compose_data_parallel_backward_loss``
can instead form a scalar from local differentiable sums and globally reduced
counts.  When DDP/FSDP averages gradients across ``world_size`` replicas, that
helper produces the exact global mean for language tokens, Dice samples, and
BCE voxels without placing collectives inside model forward.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Mapping, TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from m3d.config import SegmentationConfig
from m3d.data.schema import TaskName
from m3d.model.language import LanguageModelOutput
from m3d.model.segvol import SegVolOutput


__all__ = [
    "BCELoss",
    "BinaryBCELoss",
    "BinaryDiceLoss",
    "LossConfigurationError",
    "LossContractError",
    "LossReduction",
    "M3DLoss",
    "M3DLossOutput",
    "SegmentationLoss",
    "SegmentationLossOutput",
    "build_m3d_loss",
    "compose_data_parallel_backward_loss",
    "compute_segmentation_loss",
]


_IGNORE_MASK_VALUE: Final[float] = -1.0
_DEFAULT_DICE_SMOOTH: Final[float] = 1.0

CountLike: TypeAlias = int | Tensor


class LossConfigurationError(ValueError):
    """Raised when static loss settings are invalid."""


class LossContractError(RuntimeError):
    """Raised when runtime tensors violate the objective contract."""


class LossReduction(str, Enum):
    """Supported reduction modes for standalone binary losses."""

    NONE = "none"
    MEAN = "mean"
    SUM = "sum"

    @classmethod
    def parse(cls, value: str | "LossReduction") -> "LossReduction":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise LossConfigurationError(
                f"Unknown loss reduction {value!r}; expected one of: {allowed}."
            ) from exc


@dataclass(frozen=True, slots=True)
class SegmentationLossOutput:
    """Structured binary segmentation objective.

    ``dice_sum`` is the sum of per-sample Dice losses and is normalised by
    ``sample_count``.  ``bce_sum`` is the sum over every mask element and is
    normalised by ``voxel_count``.  Both sums remain attached to autograd.
    Count and diagnostic tensors do not require gradients.
    """

    total: Tensor
    dice: Tensor
    bce: Tensor
    dice_sum: Tensor
    bce_sum: Tensor
    sample_count: Tensor
    voxel_count: Tensor
    foreground_voxel_count: Tensor
    empty_target_count: Tensor
    legacy_minus_one_voxel_count: Tensor
    dice_weight: float
    bce_weight: float

    def __post_init__(self) -> None:
        scalar_tensors = {
            "total": self.total,
            "dice": self.dice,
            "bce": self.bce,
            "dice_sum": self.dice_sum,
            "bce_sum": self.bce_sum,
            "sample_count": self.sample_count,
            "voxel_count": self.voxel_count,
            "foreground_voxel_count": self.foreground_voxel_count,
            "empty_target_count": self.empty_target_count,
            "legacy_minus_one_voxel_count": self.legacy_minus_one_voxel_count,
        }
        for name, value in scalar_tensors.items():
            if not isinstance(value, Tensor) or value.ndim != 0:
                raise LossContractError(
                    f"Segmentation loss field {name} must be a scalar tensor."
                )
        if self.dice_weight < 0.0 or self.bce_weight < 0.0:
            raise LossConfigurationError("Segmentation loss weights cannot be negative.")

    @property
    def device(self) -> torch.device:
        return self.total.device

    def detached_metrics(self, *, prefix: str = "loss") -> dict[str, Tensor]:
        """Return synchronisation-free scalar tensors for logging.

        The trainer may batch these tensors and transfer them to CPU later;
        this method deliberately does not call ``item()`` inside the training
        step.
        """

        return {
            f"{prefix}/segmentation": self.total.detach(),
            f"{prefix}/dice": self.dice.detach(),
            f"{prefix}/bce": self.bce.detach(),
            "count/segmentation_samples": self.sample_count.detach(),
            "count/segmentation_voxels": self.voxel_count.detach(),
            "count/foreground_voxels": self.foreground_voxel_count.detach(),
            "count/empty_targets": self.empty_target_count.detach(),
            "count/legacy_minus_one_voxels": (
                self.legacy_minus_one_voxel_count.detach()
            ),
        }


@dataclass(frozen=True, slots=True)
class M3DLossOutput:
    """Task-aware objective returned by :class:`M3DLoss`.

    ``total`` is the local-mean scalar suitable for the legacy training
    behaviour.  ``language_sum`` and the optional segmentation sums retain
    gradients so the trainer can construct a globally normalised scalar with
    :func:`compose_data_parallel_backward_loss`.
    """

    task: TaskName
    total: Tensor
    language: Tensor
    language_sum: Tensor
    language_token_count: Tensor
    segmentation: SegmentationLossOutput | None

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", TaskName.parse(self.task))
        for name in ("total", "language", "language_sum", "language_token_count"):
            value = getattr(self, name)
            if not isinstance(value, Tensor) or value.ndim != 0:
                raise LossContractError(f"{name} must be a scalar tensor.")
        if self.task.requires_segmentation_target != (self.segmentation is not None):
            raise LossContractError(
                "Segmentation loss presence does not match explicit task identity: "
                f"task={self.task.value}, has_segmentation={self.segmentation is not None}."
            )

    @property
    def device(self) -> torch.device:
        return self.total.device

    def detached_metrics(self) -> dict[str, Tensor]:
        metrics = {
            "loss/total": self.total.detach(),
            "loss/language": self.language.detach(),
            "count/language_tokens": self.language_token_count.detach(),
        }
        if self.segmentation is not None:
            metrics.update(self.segmentation.detached_metrics())
        return metrics


# ---------------------------------------------------------------------------
# Validation and reusable primitives
# ---------------------------------------------------------------------------


def _validate_non_negative_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise LossConfigurationError(
            f"{name} must be finite and non-negative, got {value!r}."
        )
    return value


def _validate_positive_finite(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise LossConfigurationError(
            f"{name} must be finite and positive, got {value!r}."
        )
    return value


def _validate_binary_segmentation_pair(logits: Tensor, target: Tensor) -> None:
    if not isinstance(logits, Tensor) or not isinstance(target, Tensor):
        raise TypeError("Segmentation logits and target must be torch.Tensor objects.")
    if logits.ndim != 5:
        raise LossContractError(
            "Segmentation logits must have shape [B,1,D,H,W], got "
            f"{tuple(logits.shape)}."
        )
    if target.ndim != 5:
        raise LossContractError(
            "Segmentation target must have shape [B,1,D,H,W], got "
            f"{tuple(target.shape)}."
        )
    if tuple(logits.shape) != tuple(target.shape):
        raise LossContractError(
            "Segmentation logits and target shapes differ: "
            f"logits={tuple(logits.shape)}, target={tuple(target.shape)}."
        )
    if int(logits.shape[0]) <= 0:
        raise LossContractError("Segmentation batch cannot be empty.")
    if int(logits.shape[1]) != 1:
        raise LossContractError(
            "M3D binary segmentation training expects exactly one selected mask "
            f"channel, got {int(logits.shape[1])}. Set multimask_output=False."
        )
    if not logits.is_floating_point() or not target.is_floating_point():
        raise TypeError("Segmentation logits and targets must use floating dtypes.")
    if logits.device != target.device:
        raise LossContractError(
            "Segmentation logits and targets are on different devices: "
            f"{logits.device} vs {target.device}."
        )
    if target.requires_grad:
        raise LossContractError("Segmentation targets must not require gradients.")


def _normalise_legacy_target(target: Tensor) -> tuple[Tensor, Tensor]:
    """Convert the original M3D ``-1`` sentinel to background.

    Modern M3D datasets are validated as binary ``0/1`` before device transfer,
    so the returned legacy count should normally be zero.  Keeping this rule
    reproduces old checkpoints and ad-hoc batches that used ``-1``.
    """

    target_fp32 = target.float()
    legacy_mask = target_fp32.eq(_IGNORE_MASK_VALUE)
    clean_target = torch.where(
        legacy_mask,
        torch.zeros((), dtype=target_fp32.dtype, device=target_fp32.device),
        target_fp32,
    )
    return clean_target, legacy_mask.sum(dtype=torch.int64)


def _reduce(values: Tensor, reduction: LossReduction) -> Tensor:
    if reduction is LossReduction.NONE:
        return values
    if reduction is LossReduction.SUM:
        return values.sum()
    return values.mean()


def _binary_dice_components(
    logits: Tensor,
    clean_target: Tensor,
    *,
    smooth: float,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return per-sample Dice loss, probabilities, and foreground counts."""

    probabilities = torch.sigmoid(logits.float())
    target_fp32 = clean_target.float()
    batch_size = int(probabilities.shape[0])
    probability_flat = probabilities.reshape(batch_size, -1)
    target_flat = target_fp32.reshape(batch_size, -1)

    intersection = (probability_flat * target_flat).sum(dim=1)
    denominator = (
        probability_flat.sum(dim=1)
        + target_flat.sum(dim=1)
        + float(smooth)
    )
    dice_score = (2.0 * intersection) / denominator
    per_sample_loss = 1.0 - dice_score
    foreground_per_sample = target_flat.sum(dim=1)
    return per_sample_loss, probabilities, foreground_per_sample


# ---------------------------------------------------------------------------
# Original-compatible standalone losses
# ---------------------------------------------------------------------------


class BinaryDiceLoss(nn.Module):
    """Original M3D binary Dice loss with stable FP32 reductions.

    The original constructor exposed ``p`` but its forward formula never used
    it.  The argument is retained for source compatibility and validated, while
    the reproduced formula remains::

        1 - 2 * sum(sigmoid(logits) * target)
              / (sum(sigmoid(logits)) + sum(target) + smooth)

    For an all-zero target, this exact legacy formula returns ``1`` and its Dice
    component has zero gradient; BCE still supplies the background gradient.
    """

    def __init__(
        self,
        smooth: float = _DEFAULT_DICE_SMOOTH,
        p: float = 2.0,
        reduction: LossReduction | str = LossReduction.MEAN,
    ) -> None:
        super().__init__()
        self.smooth = _validate_positive_finite("smooth", smooth)
        self.p = _validate_positive_finite("p", p)
        self.reduction = LossReduction.parse(reduction)

    def forward(self, predict: Tensor, target: Tensor) -> Tensor:
        _validate_binary_segmentation_pair(predict, target)
        clean_target, _ = _normalise_legacy_target(target)
        per_sample, _, _ = _binary_dice_components(
            predict,
            clean_target,
            smooth=self.smooth,
        )
        return _reduce(per_sample, self.reduction)


class BinaryBCELoss(nn.Module):
    """Original M3D BCE-with-logits loss with optional reduction."""

    def __init__(
        self,
        reduction: LossReduction | str = LossReduction.MEAN,
    ) -> None:
        super().__init__()
        self.reduction = LossReduction.parse(reduction)

    def forward(self, predict: Tensor, target: Tensor) -> Tensor:
        _validate_binary_segmentation_pair(predict, target)
        clean_target, _ = _normalise_legacy_target(target)
        per_voxel = F.binary_cross_entropy_with_logits(
            predict.float(),
            clean_target,
            reduction="none",
        )
        return _reduce(per_voxel, self.reduction)


class BCELoss(BinaryBCELoss):
    """Legacy class name retained for original M3D imports."""

    def __init__(self) -> None:
        super().__init__(reduction=LossReduction.MEAN)


# ---------------------------------------------------------------------------
# Structured segmentation objective
# ---------------------------------------------------------------------------


def compute_segmentation_loss(
    *,
    logits: Tensor,
    target: Tensor,
    dice_weight: float = 1.0,
    bce_weight: float = 1.0,
    dice_smooth: float = _DEFAULT_DICE_SMOOTH,
) -> SegmentationLossOutput:
    """Compute weighted Dice + BCE from raw binary mask logits.

    The loss is deliberately computed at the final target resolution.  The
    lower-resolution decoder logits remain available for diagnostics but are
    not supervised by the original M3D objective.
    """

    _validate_binary_segmentation_pair(logits, target)
    dice_weight = _validate_non_negative_finite("dice_weight", dice_weight)
    bce_weight = _validate_non_negative_finite("bce_weight", bce_weight)
    dice_smooth = _validate_positive_finite("dice_smooth", dice_smooth)
    if dice_weight == 0.0 and bce_weight == 0.0:
        raise LossConfigurationError(
            "At least one of dice_weight and bce_weight must be positive."
        )

    clean_target, legacy_count = _normalise_legacy_target(target)
    per_sample_dice, _, foreground_per_sample = _binary_dice_components(
        logits,
        clean_target,
        smooth=dice_smooth,
    )

    dice_sum = per_sample_dice.sum()
    sample_count = torch.tensor(
        int(logits.shape[0]),
        dtype=torch.int64,
        device=logits.device,
    )
    dice_mean = dice_sum / sample_count.to(dtype=dice_sum.dtype)

    per_voxel_bce = F.binary_cross_entropy_with_logits(
        logits.float(),
        clean_target,
        reduction="none",
    )
    bce_sum = per_voxel_bce.sum()
    voxel_count = torch.tensor(
        target.numel(),
        dtype=torch.int64,
        device=target.device,
    )
    bce_mean = bce_sum / voxel_count.to(dtype=bce_sum.dtype)

    foreground_count = clean_target.sum(dtype=torch.float32)
    empty_target_count = foreground_per_sample.eq(0.0).sum(dtype=torch.int64)
    total = dice_mean * dice_weight + bce_mean * bce_weight

    return SegmentationLossOutput(
        total=total,
        dice=dice_mean,
        bce=bce_mean,
        dice_sum=dice_sum,
        bce_sum=bce_sum,
        sample_count=sample_count,
        voxel_count=voxel_count,
        foreground_voxel_count=foreground_count,
        empty_target_count=empty_target_count,
        legacy_minus_one_voxel_count=legacy_count,
        dice_weight=dice_weight,
        bce_weight=bce_weight,
    )


class SegmentationLoss(nn.Module):
    """Module wrapper around :func:`compute_segmentation_loss`."""

    def __init__(
        self,
        *,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        dice_smooth: float = _DEFAULT_DICE_SMOOTH,
    ) -> None:
        super().__init__()
        self.dice_weight = _validate_non_negative_finite(
            "dice_weight", dice_weight
        )
        self.bce_weight = _validate_non_negative_finite("bce_weight", bce_weight)
        self.dice_smooth = _validate_positive_finite("dice_smooth", dice_smooth)
        if self.dice_weight == 0.0 and self.bce_weight == 0.0:
            raise LossConfigurationError(
                "At least one segmentation loss weight must be positive."
            )

    def forward(self, logits: Tensor, target: Tensor) -> SegmentationLossOutput:
        return compute_segmentation_loss(
            logits=logits,
            target=target,
            dice_weight=self.dice_weight,
            bce_weight=self.bce_weight,
            dice_smooth=self.dice_smooth,
        )

    def extra_repr(self) -> str:
        return (
            f"dice_weight={self.dice_weight}, bce_weight={self.bce_weight}, "
            f"dice_smooth={self.dice_smooth}"
        )


# ---------------------------------------------------------------------------
# Task-aware complete M3D objective
# ---------------------------------------------------------------------------


def _extract_language_loss(
    output: LanguageModelOutput,
) -> tuple[Tensor, Tensor, Tensor]:
    if not isinstance(output, LanguageModelOutput):
        raise TypeError(
            "language_output must be LanguageModelOutput, got "
            f"{type(output).__name__}."
        )
    if (
        output.loss is None
        or output.loss_sum is None
        or output.supervised_token_count is None
    ):
        raise LossContractError(
            "LanguageModelOutput does not contain a training loss. Pass labels to "
            "M3DLanguageModel before invoking the objective."
        )
    for name, value in (
        ("language loss", output.loss),
        ("language loss_sum", output.loss_sum),
        ("language token count", output.supervised_token_count),
    ):
        if not isinstance(value, Tensor) or value.ndim != 0:
            raise LossContractError(f"{name} must be a scalar tensor.")
    if output.loss.device != output.loss_sum.device:
        raise LossContractError("Language mean and summed losses are on different devices.")
    if output.supervised_token_count.device != output.loss.device:
        raise LossContractError("Language token count is on a different device.")
    return output.loss, output.loss_sum, output.supervised_token_count


def _extract_segmentation_logits(output: SegVolOutput | Tensor) -> Tensor:
    if isinstance(output, SegVolOutput):
        return output.logits
    if isinstance(output, Tensor):
        return output
    raise TypeError(
        "segmentation_output must be SegVolOutput or Tensor, got "
        f"{type(output).__name__}."
    )


class M3DLoss(nn.Module):
    """Compose language and segmentation objectives from explicit task identity."""

    def __init__(
        self,
        *,
        dice_weight: float = 1.0,
        bce_weight: float = 1.0,
        dice_smooth: float = _DEFAULT_DICE_SMOOTH,
    ) -> None:
        super().__init__()
        self.segmentation_loss = SegmentationLoss(
            dice_weight=dice_weight,
            bce_weight=bce_weight,
            dice_smooth=dice_smooth,
        )

    @property
    def dice_weight(self) -> float:
        return self.segmentation_loss.dice_weight

    @property
    def bce_weight(self) -> float:
        return self.segmentation_loss.bce_weight

    def forward(
        self,
        *,
        task: TaskName | str,
        language_output: LanguageModelOutput,
        segmentation_output: SegVolOutput | Tensor | None = None,
        segmentation_targets: Tensor | None = None,
    ) -> M3DLossOutput:
        task = TaskName.parse(task)
        language_mean, language_sum, language_count = _extract_language_loss(
            language_output
        )

        if not task.requires_segmentation_target:
            if segmentation_output is not None or segmentation_targets is not None:
                raise LossContractError(
                    f"Task {task.value!r} must not carry segmentation output or targets. "
                    "Execution routing must follow explicit task identity."
                )
            return M3DLossOutput(
                task=task,
                total=language_mean,
                language=language_mean,
                language_sum=language_sum,
                language_token_count=language_count,
                segmentation=None,
            )

        if segmentation_output is None or segmentation_targets is None:
            raise LossContractError(
                "A segmentation task requires both segmentation_output and "
                "segmentation_targets, including valid all-zero targets."
            )
        logits = _extract_segmentation_logits(segmentation_output)
        segmentation = self.segmentation_loss(logits, segmentation_targets)
        if segmentation.total.device != language_mean.device:
            raise LossContractError(
                "Language and segmentation losses are on different devices: "
                f"{language_mean.device} vs {segmentation.total.device}."
            )
        return M3DLossOutput(
            task=task,
            total=language_mean + segmentation.total,
            language=language_mean,
            language_sum=language_sum,
            language_token_count=language_count,
            segmentation=segmentation,
        )

    def extra_repr(self) -> str:
        return (
            f"dice_weight={self.dice_weight}, "
            f"bce_weight={self.bce_weight}, "
            f"dice_smooth={self.segmentation_loss.dice_smooth}"
        )


def build_m3d_loss(segmentation_config: SegmentationConfig) -> M3DLoss:
    """Build the objective from the validated project configuration."""

    if not isinstance(segmentation_config, SegmentationConfig):
        raise TypeError(
            "segmentation_config must be SegmentationConfig, got "
            f"{type(segmentation_config).__name__}."
        )
    return M3DLoss(
        dice_weight=segmentation_config.dice_loss_weight,
        bce_weight=segmentation_config.bce_loss_weight,
        dice_smooth=_DEFAULT_DICE_SMOOTH,
    )


# ---------------------------------------------------------------------------
# Exact data-parallel normalisation
# ---------------------------------------------------------------------------


def _count_tensor(
    value: CountLike,
    *,
    name: str,
    device: torch.device,
) -> Tensor:
    if isinstance(value, bool):
        raise TypeError(f"{name} cannot be bool.")
    if isinstance(value, int):
        if value <= 0:
            raise LossContractError(f"{name} must be positive, got {value}.")
        return torch.tensor(value, dtype=torch.int64, device=device)
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be int or scalar Tensor.")
    if value.ndim != 0:
        raise LossContractError(f"{name} must be a scalar Tensor.")
    if value.device != device:
        value = value.to(device=device)
    return value


def compose_data_parallel_backward_loss(
    output: M3DLossOutput,
    *,
    world_size: int,
    global_language_token_count: CountLike,
    global_segmentation_sample_count: CountLike | None = None,
    global_segmentation_voxel_count: CountLike | None = None,
) -> Tensor:
    """Compose an exact global-mean scalar before DDP/FSDP backward.

    This function performs no collective communication.  The trainer should
    all-reduce the three count tensors first.  PyTorch data parallel wrappers
    normally average gradients across ``world_size`` ranks, so every local sum
    is multiplied by ``world_size / global_count`` before backward.

    All ranks must execute the same task for the current step.  The project's
    task-homogeneous distributed sampler guarantees that contract.
    """

    if not isinstance(output, M3DLossOutput):
        raise TypeError("output must be M3DLossOutput.")
    if isinstance(world_size, bool) or not isinstance(world_size, int) or world_size <= 0:
        raise LossConfigurationError(
            f"world_size must be a positive integer, got {world_size!r}."
        )

    device = output.device
    global_language = _count_tensor(
        global_language_token_count,
        name="global_language_token_count",
        device=device,
    )
    world = torch.tensor(float(world_size), dtype=torch.float32, device=device)
    backward_loss = (
        output.language_sum
        * world
        / global_language.to(dtype=output.language_sum.dtype)
    )

    if output.segmentation is None:
        if (
            global_segmentation_sample_count is not None
            or global_segmentation_voxel_count is not None
        ):
            raise LossContractError(
                "Text-only task received segmentation global normalisers."
            )
        return backward_loss

    if (
        global_segmentation_sample_count is None
        or global_segmentation_voxel_count is None
    ):
        raise LossContractError(
            "Segmentation task requires global sample and voxel counts."
        )
    global_samples = _count_tensor(
        global_segmentation_sample_count,
        name="global_segmentation_sample_count",
        device=device,
    )
    global_voxels = _count_tensor(
        global_segmentation_voxel_count,
        name="global_segmentation_voxel_count",
        device=device,
    )
    segmentation = output.segmentation
    backward_loss = backward_loss + (
        segmentation.dice_sum
        * (world * segmentation.dice_weight)
        / global_samples.to(dtype=segmentation.dice_sum.dtype)
    )
    backward_loss = backward_loss + (
        segmentation.bce_sum
        * (world * segmentation.bce_weight)
        / global_voxels.to(dtype=segmentation.bce_sum.dtype)
    )
    return backward_loss


# ---------------------------------------------------------------------------
# Dependency-free self-test
# ---------------------------------------------------------------------------


def _legacy_dice_reference(logits: Tensor, target: Tensor) -> Tensor:
    predict = torch.sigmoid(logits)
    target_copy = target.clone().float()
    target_copy[target == -1] = 0
    predict = predict.contiguous().view(predict.shape[0], -1)
    target_copy = target_copy.contiguous().view(target_copy.shape[0], -1)
    numerator = torch.sum(predict * target_copy, dim=1)
    denominator = (
        torch.sum(predict, dim=1)
        + torch.sum(target_copy, dim=1)
        + 1.0
    )
    return (1.0 - (2.0 * numerator / denominator)).sum() / predict.shape[0]


def _legacy_bce_reference(logits: Tensor, target: Tensor) -> Tensor:
    target_copy = target.clone()
    target_copy[target == -1] = 0
    return F.binary_cross_entropy_with_logits(logits, target_copy.float())


def _toy_language_output(
    *,
    loss_sum: Tensor,
    token_count: int,
    batch_size: int,
) -> LanguageModelOutput:
    count = torch.tensor(token_count, dtype=torch.int64, device=loss_sum.device)
    return LanguageModelOutput(
        loss=loss_sum / count.to(dtype=loss_sum.dtype),
        loss_sum=loss_sum,
        supervised_token_count=count,
        last_hidden_state=torch.zeros(
            batch_size,
            4,
            8,
            dtype=loss_sum.dtype,
            device=loss_sum.device,
        ),
        logits=None,
        supervised_labels=torch.zeros(
            token_count,
            dtype=torch.long,
            device=loss_sum.device,
        ),
        image_token_counts=torch.full(
            (batch_size,),
            2,
            dtype=torch.int64,
            device=loss_sum.device,
        ),
    )


def _run_self_test() -> Mapping[str, Any]:
    torch.manual_seed(20260725)

    logits = torch.randn(2, 1, 2, 3, 4, dtype=torch.float32, requires_grad=True)
    target = torch.zeros_like(logits)
    target[0, 0, 0, 0, 0] = 1.0
    # The second sample is a legal all-zero segmentation target.

    output = compute_segmentation_loss(
        logits=logits,
        target=target,
        dice_weight=1.0,
        bce_weight=1.0,
    )
    reference_dice = _legacy_dice_reference(logits, target)
    reference_bce = _legacy_bce_reference(logits, target)
    torch.testing.assert_close(output.dice, reference_dice, rtol=0.0, atol=1e-7)
    torch.testing.assert_close(output.bce, reference_bce, rtol=0.0, atol=1e-7)
    torch.testing.assert_close(
        output.total,
        reference_dice + reference_bce,
        rtol=0.0,
        atol=1e-7,
    )

    output.total.backward()
    if logits.grad is None or not bool(torch.isfinite(logits.grad).all()):
        raise AssertionError("Combined segmentation loss did not produce finite gradients.")

    empty_logits = torch.randn(
        1, 1, 2, 2, 2, dtype=torch.float32, requires_grad=True
    )
    empty_target = torch.zeros_like(empty_logits)
    empty_dice = BinaryDiceLoss()(empty_logits, empty_target)
    torch.testing.assert_close(empty_dice, torch.ones_like(empty_dice))
    empty_dice.backward()
    if empty_logits.grad is None:
        raise AssertionError("All-zero Dice test did not create a gradient tensor.")
    torch.testing.assert_close(empty_logits.grad, torch.zeros_like(empty_logits.grad))

    legacy_target = target.clone()
    legacy_target[0, 0, 0, 0, 1] = -1.0
    legacy = compute_segmentation_loss(logits=logits.detach(), target=legacy_target)
    if int(legacy.legacy_minus_one_voxel_count) != 1:
        raise AssertionError("Legacy -1 target was not counted and mapped to background.")

    objective = M3DLoss(dice_weight=1.5, bce_weight=0.25)
    language_sum = torch.tensor(6.0, dtype=torch.float32, requires_grad=True)
    language_output = _toy_language_output(
        loss_sum=language_sum,
        token_count=3,
        batch_size=2,
    )
    seg_logits = logits.detach().clone().requires_grad_(True)
    segvol_output = SegVolOutput(
        logits=seg_logits,
        low_resolution_logits=seg_logits,
        iou_predictions=torch.zeros(2, 1),
        output_spatial_size=(2, 3, 4),
        image_embedding_shape=(2, 8, 1, 1, 1),
        sparse_prompt_count=1,
        multimask_output=False,
    )
    joint = objective(
        task=TaskName.SEGMENTATION,
        language_output=language_output,
        segmentation_output=segvol_output,
        segmentation_targets=target,
    )
    assert joint.segmentation is not None
    expected_total = (
        language_output.loss
        + 1.5 * joint.segmentation.dice
        + 0.25 * joint.segmentation.bce
    )
    torch.testing.assert_close(joint.total, expected_total)

    text_only = objective(
        task=TaskName.CAPTION,
        language_output=language_output,
    )
    torch.testing.assert_close(text_only.total, language_output.loss)

    route_error = False
    try:
        objective(
            task=TaskName.CAPTION,
            language_output=language_output,
            segmentation_output=seg_logits,
            segmentation_targets=target,
        )
    except LossContractError:
        route_error = True
    if not route_error:
        raise AssertionError("Text task accepted segmentation tensors.")

    # Emulate two ranks and verify the DDP-scaled local objectives average to
    # the exact global mean of each independently normalised term.
    rank0_language_sum = torch.tensor(4.0, requires_grad=True)
    rank1_language_sum = torch.tensor(8.0, requires_grad=True)
    rank0_language = _toy_language_output(
        loss_sum=rank0_language_sum, token_count=2, batch_size=1
    )
    rank1_language = _toy_language_output(
        loss_sum=rank1_language_sum, token_count=4, batch_size=1
    )
    one_target = target[:1]
    rank0_logits = torch.randn_like(one_target, requires_grad=True)
    rank1_logits = torch.randn_like(one_target, requires_grad=True)
    rank0 = objective(
        task=TaskName.SEGMENTATION,
        language_output=rank0_language,
        segmentation_output=rank0_logits,
        segmentation_targets=one_target,
    )
    rank1 = objective(
        task=TaskName.SEGMENTATION,
        language_output=rank1_language,
        segmentation_output=rank1_logits,
        segmentation_targets=one_target,
    )
    global_language_count = (
        rank0.language_token_count + rank1.language_token_count
    )
    assert rank0.segmentation is not None and rank1.segmentation is not None
    global_sample_count = (
        rank0.segmentation.sample_count + rank1.segmentation.sample_count
    )
    global_voxel_count = rank0.segmentation.voxel_count + rank1.segmentation.voxel_count
    scaled0 = compose_data_parallel_backward_loss(
        rank0,
        world_size=2,
        global_language_token_count=global_language_count,
        global_segmentation_sample_count=global_sample_count,
        global_segmentation_voxel_count=global_voxel_count,
    )
    scaled1 = compose_data_parallel_backward_loss(
        rank1,
        world_size=2,
        global_language_token_count=global_language_count,
        global_segmentation_sample_count=global_sample_count,
        global_segmentation_voxel_count=global_voxel_count,
    )
    averaged_scaled = (scaled0 + scaled1) / 2.0
    exact_global = (
        (rank0.language_sum + rank1.language_sum)
        / global_language_count.to(torch.float32)
        + objective.dice_weight
        * (rank0.segmentation.dice_sum + rank1.segmentation.dice_sum)
        / global_sample_count.to(torch.float32)
        + objective.bce_weight
        * (rank0.segmentation.bce_sum + rank1.segmentation.bce_sum)
        / global_voxel_count.to(torch.float32)
    )
    torch.testing.assert_close(averaged_scaled, exact_global, rtol=0.0, atol=1e-7)

    return {
        "status": "passed",
        "legacy_dice_equivalence": True,
        "legacy_bce_equivalence": True,
        "all_zero_target_is_valid": True,
        "all_zero_target_dice": float(empty_dice.detach()),
        "all_zero_dice_gradient_is_zero": True,
        "legacy_minus_one_compatibility": True,
        "task_routing_error_detected": route_error,
        "distributed_global_mean_equivalence": True,
        "segmentation_batch_size": int(output.sample_count),
        "segmentation_voxel_count": int(output.voxel_count),
        "empty_target_count": int(output.empty_target_count),
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run dependency-free numerical and contract tests.",
    )
    args = parser.parse_args()
    if not args.self_test:
        parser.print_help()
        return
    print(json.dumps(_run_self_test(), indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
