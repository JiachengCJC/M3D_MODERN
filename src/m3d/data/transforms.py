"""Synchronous 3D augmentations for every M3D task.

The original M3D data pipeline applies the following training transforms:

* random 90-degree rotation in the H/W plane with probability 0.5;
* independent flips along D, H, and W with probability 0.1 each;
* random image-only intensity scaling by a factor sampled from [-0.1, 0.1]
  with probability 0.5;
* random image-only intensity shifting by an offset sampled from [-0.1, 0.1]
  with probability 0.5.

This module reproduces those semantics using native PyTorch tensor operations.
For segmentation samples, one augmentation plan is sampled once and the same
spatial operations are applied to both image and target mask.  Intensity
operations are never applied to masks.

Unlike the legacy implementation, task identity is not inferred from target
contents.  A valid all-zero segmentation target remains a segmentation target
and is transformed normally.

Tensor conventions
------------------
* image: ``[C, D, H, W]``, CPU, ``torch.float32``;
* segmentation target: ``[1, D, H, W]``, CPU, ``torch.float32``;
* spatial axis 0/1/2 means D/H/W respectively;
* validation and test transforms are identity transforms plus validation;
* outputs are contiguous CPU tensors.

Reproducibility
---------------
A training dataset should pass an :class:`AugmentationContext` containing the
sample ID and epoch.  The random plan is then derived from a stable hash and is
independent of DataLoader worker scheduling.  This makes augmentation replay
stable when the worker count changes or a checkpoint is resumed.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from enum import IntEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

import torch
from torch import Tensor

from .schema import DataSplit, M3DSample, SampleProvenance

if TYPE_CHECKING:
    from m3d.config import ExperimentConfig


class TransformError(ValueError):
    """Raised when an augmentation input or result violates the data contract."""


class SpatialAxis(IntEnum):
    """Spatial axis numbering used by MONAI and the original M3D pipeline."""

    DEPTH = 0
    HEIGHT = 1
    WIDTH = 2

    @property
    def tensor_dimension(self) -> int:
        """Map D/H/W to dimensions 1/2/3 of a ``[C,D,H,W]`` tensor."""

        return int(self) + 1


@dataclass(frozen=True, slots=True)
class AugmentationPolicy:
    """Immutable augmentation probabilities and magnitudes.

    Defaults exactly match the repeated transform definitions in the original
    ``multi_dataset.py`` implementation.
    """

    expected_spatial_shape: tuple[int, int, int] = (32, 256, 256)

    rotate90_probability: float = 0.5
    rotate90_axes: tuple[SpatialAxis, SpatialAxis] = (
        SpatialAxis.HEIGHT,
        SpatialAxis.WIDTH,
    )
    rotate90_max_k: int = 3

    flip_probabilities: tuple[float, float, float] = (0.1, 0.1, 0.1)

    scale_intensity_probability: float = 0.5
    scale_intensity_factor: float = 0.1

    shift_intensity_probability: float = 0.5
    shift_intensity_offset: float = 0.1

    # The legacy pipeline did not clamp after RandScaleIntensity/RandShiftIntensity.
    clamp_after_intensity: bool = False
    clamp_range: tuple[float, float] = (0.0, 1.0)

    validate_binary_target: bool = True
    reject_nonfinite: bool = True

    def __post_init__(self) -> None:
        if len(self.expected_spatial_shape) != 3:
            raise TransformError("expected_spatial_shape must contain D, H, and W")
        if any(size <= 0 for size in self.expected_spatial_shape):
            raise TransformError("expected_spatial_shape values must be positive")

        _validate_probability(
            self.rotate90_probability,
            name="rotate90_probability",
        )
        if len(self.rotate90_axes) != 2:
            raise TransformError("rotate90_axes must contain exactly two axes")
        axis_a = SpatialAxis(self.rotate90_axes[0])
        axis_b = SpatialAxis(self.rotate90_axes[1])
        if axis_a == axis_b:
            raise TransformError("rotate90_axes must refer to two different axes")
        object.__setattr__(self, "rotate90_axes", (axis_a, axis_b))

        if self.rotate90_max_k < 1 or self.rotate90_max_k > 3:
            raise TransformError("rotate90_max_k must be between 1 and 3")

        if len(self.flip_probabilities) != 3:
            raise TransformError("flip_probabilities must contain D, H, and W")
        for axis, probability in zip(SpatialAxis, self.flip_probabilities):
            _validate_probability(
                probability,
                name=f"flip_probability_{axis.name.lower()}",
            )

        _validate_probability(
            self.scale_intensity_probability,
            name="scale_intensity_probability",
        )
        _validate_probability(
            self.shift_intensity_probability,
            name="shift_intensity_probability",
        )
        if self.scale_intensity_factor < 0:
            raise TransformError("scale_intensity_factor cannot be negative")
        if self.shift_intensity_offset < 0:
            raise TransformError("shift_intensity_offset cannot be negative")

        clamp_low, clamp_high = self.clamp_range
        if not (math.isfinite(clamp_low) and math.isfinite(clamp_high)):
            raise TransformError("clamp_range values must be finite")
        if not clamp_low < clamp_high:
            raise TransformError("clamp_range must satisfy low < high")

        # Odd k values swap the two rotated dimensions.  M3D rotates H/W, both
        # of which are 256, so the output shape remains fixed.  Reject a policy
        # that could silently change the configured spatial shape.
        dim_a = self.expected_spatial_shape[int(axis_a)]
        dim_b = self.expected_spatial_shape[int(axis_b)]
        if dim_a != dim_b and self.rotate90_probability > 0:
            raise TransformError(
                "The selected rotate90 axes have unequal sizes "
                f"({dim_a} and {dim_b}); odd rotations would change the fixed "
                "M3D tensor shape"
            )

    @classmethod
    def original_m3d(
        cls,
        *,
        expected_spatial_shape: tuple[int, int, int] = (32, 256, 256),
    ) -> "AugmentationPolicy":
        """Return the exact augmentation policy used by original M3D training."""

        return cls(expected_spatial_shape=expected_spatial_shape)

    @classmethod
    def identity(
        cls,
        *,
        expected_spatial_shape: tuple[int, int, int] = (32, 256, 256),
    ) -> "AugmentationPolicy":
        """Return a validation/test policy with no random augmentation."""

        return cls(
            expected_spatial_shape=expected_spatial_shape,
            rotate90_probability=0.0,
            flip_probabilities=(0.0, 0.0, 0.0),
            scale_intensity_probability=0.0,
            shift_intensity_probability=0.0,
        )


@dataclass(frozen=True, slots=True)
class AugmentationContext:
    """Stable information used to derive one sample's augmentation seed."""

    sample_id: str
    epoch: int
    base_seed: int
    view_index: int = 0

    def __post_init__(self) -> None:
        sample_id = self.sample_id.strip()
        if not sample_id:
            raise TransformError("AugmentationContext.sample_id cannot be empty")
        if self.epoch < 0:
            raise TransformError("AugmentationContext.epoch cannot be negative")
        if self.view_index < 0:
            raise TransformError("AugmentationContext.view_index cannot be negative")
        object.__setattr__(self, "sample_id", sample_id)

    def stable_seed(self) -> int:
        """Return a deterministic non-negative 63-bit seed.

        Python's built-in ``hash`` is intentionally randomised between
        processes, so it must not be used for distributed reproducibility.
        """

        payload = (
            f"m3d-augmentation-v1\0{self.base_seed}\0{self.epoch}\0"
            f"{self.view_index}\0{self.sample_id}"
        ).encode("utf-8")
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False) & (
            (1 << 63) - 1
        )

    def make_generator(self) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.stable_seed())
        return generator


@dataclass(frozen=True, slots=True)
class AugmentationPlan:
    """All random choices for one image/mask pair."""

    seed: int | None
    rotate90_k: int
    flip_depth: bool
    flip_height: bool
    flip_width: bool
    intensity_scale_delta: float
    intensity_shift: float

    def __post_init__(self) -> None:
        if self.seed is not None and self.seed < 0:
            raise TransformError("plan seed cannot be negative")
        if self.rotate90_k not in (0, 1, 2, 3):
            raise TransformError("rotate90_k must be 0, 1, 2, or 3")
        if not math.isfinite(self.intensity_scale_delta):
            raise TransformError("intensity_scale_delta must be finite")
        if not math.isfinite(self.intensity_shift):
            raise TransformError("intensity_shift must be finite")

    @property
    def has_spatial_change(self) -> bool:
        return bool(
            self.rotate90_k
            or self.flip_depth
            or self.flip_height
            or self.flip_width
        )

    @property
    def has_intensity_change(self) -> bool:
        return bool(self.intensity_scale_delta or self.intensity_shift)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "rotate90_k": self.rotate90_k,
            "flip_depth": self.flip_depth,
            "flip_height": self.flip_height,
            "flip_width": self.flip_width,
            "intensity_scale_delta": self.intensity_scale_delta,
            "intensity_multiplier": 1.0 + self.intensity_scale_delta,
            "intensity_shift": self.intensity_shift,
        }


@dataclass(frozen=True, slots=True)
class TransformResult:
    """Transformed tensors plus a replayable record of the sampled choices."""

    image: Tensor
    segmentation_target: Tensor | None
    plan: AugmentationPlan

    def __post_init__(self) -> None:
        _validate_volume_tensor(
            self.image,
            name="transformed image",
            channels=None,
            expected_spatial_shape=tuple(self.image.shape[1:]),
            reject_nonfinite=True,
        )
        if self.segmentation_target is not None:
            _validate_volume_tensor(
                self.segmentation_target,
                name="transformed segmentation target",
                channels=1,
                expected_spatial_shape=tuple(self.image.shape[1:]),
                reject_nonfinite=True,
            )


class M3DVolumeTransform:
    """Apply one synchronised transform plan to an image and optional mask."""

    def __init__(
        self,
        policy: AugmentationPolicy,
        *,
        training: bool,
    ) -> None:
        self.policy = policy
        self.training = bool(training)

    def sample_plan(
        self,
        *,
        context: AugmentationContext | None = None,
        generator: torch.Generator | None = None,
    ) -> AugmentationPlan:
        """Sample all random choices once.

        Pass ``context`` for scheduling-independent deterministic augmentation,
        or pass an explicit CPU generator for controlled tests.  They are
        mutually exclusive.  With neither supplied, the DataLoader worker's
        globally seeded CPU RNG is used.
        """

        if context is not None and generator is not None:
            raise TransformError("context and generator are mutually exclusive")

        if not self.training:
            return AugmentationPlan(
                seed=None if context is None else context.stable_seed(),
                rotate90_k=0,
                flip_depth=False,
                flip_height=False,
                flip_width=False,
                intensity_scale_delta=0.0,
                intensity_shift=0.0,
            )

        if context is not None:
            active_generator = context.make_generator()
            seed: int | None = context.stable_seed()
        else:
            active_generator = generator
            seed = None

        rotate90_k = 0
        if _bernoulli(
            self.policy.rotate90_probability,
            generator=active_generator,
        ):
            rotate90_k = int(
                torch.randint(
                    low=1,
                    high=self.policy.rotate90_max_k + 1,
                    size=(),
                    generator=active_generator,
                    device="cpu",
                ).item()
            )

        flips = tuple(
            _bernoulli(probability, generator=active_generator)
            for probability in self.policy.flip_probabilities
        )

        scale_delta = 0.0
        if _bernoulli(
            self.policy.scale_intensity_probability,
            generator=active_generator,
        ):
            scale_delta = _uniform_symmetric(
                self.policy.scale_intensity_factor,
                generator=active_generator,
            )

        shift = 0.0
        if _bernoulli(
            self.policy.shift_intensity_probability,
            generator=active_generator,
        ):
            shift = _uniform_symmetric(
                self.policy.shift_intensity_offset,
                generator=active_generator,
            )

        return AugmentationPlan(
            seed=seed,
            rotate90_k=rotate90_k,
            flip_depth=flips[0],
            flip_height=flips[1],
            flip_width=flips[2],
            intensity_scale_delta=scale_delta,
            intensity_shift=shift,
        )

    def __call__(
        self,
        image: Tensor,
        segmentation_target: Tensor | None = None,
        *,
        context: AugmentationContext | None = None,
        generator: torch.Generator | None = None,
        plan: AugmentationPlan | None = None,
    ) -> TransformResult:
        """Validate and transform one image with an optional dense target."""

        if plan is not None and (context is not None or generator is not None):
            raise TransformError(
                "A supplied plan cannot be combined with context or generator"
            )

        self._validate_inputs(image, segmentation_target)
        active_plan = plan or self.sample_plan(
            context=context,
            generator=generator,
        )

        transformed_image = image
        transformed_target = segmentation_target

        if active_plan.rotate90_k:
            axis_a, axis_b = self.policy.rotate90_axes
            dims = (axis_a.tensor_dimension, axis_b.tensor_dimension)
            transformed_image = torch.rot90(
                transformed_image,
                k=active_plan.rotate90_k,
                dims=dims,
            )
            if transformed_target is not None:
                transformed_target = torch.rot90(
                    transformed_target,
                    k=active_plan.rotate90_k,
                    dims=dims,
                )

        flip_tensor_dims = tuple(
            axis.tensor_dimension
            for axis, enabled in zip(
                SpatialAxis,
                (
                    active_plan.flip_depth,
                    active_plan.flip_height,
                    active_plan.flip_width,
                ),
            )
            if enabled
        )
        if flip_tensor_dims:
            transformed_image = torch.flip(
                transformed_image,
                dims=flip_tensor_dims,
            )
            if transformed_target is not None:
                transformed_target = torch.flip(
                    transformed_target,
                    dims=flip_tensor_dims,
                )

        if active_plan.intensity_scale_delta:
            transformed_image = transformed_image * (
                1.0 + active_plan.intensity_scale_delta
            )
        if active_plan.intensity_shift:
            transformed_image = transformed_image + active_plan.intensity_shift

        if self.policy.clamp_after_intensity:
            low, high = self.policy.clamp_range
            transformed_image = transformed_image.clamp(min=low, max=high)

        # torch.rot90 and torch.flip can return non-contiguous views.  The two
        # Conv3d patch embedders consume contiguous tensors more predictably,
        # and pinned-memory transfer also benefits from a standard layout.
        transformed_image = transformed_image.to(dtype=torch.float32).contiguous()
        if transformed_target is not None:
            transformed_target = transformed_target.to(
                dtype=torch.float32
            ).contiguous()

        self._validate_outputs(transformed_image, transformed_target)
        return TransformResult(
            image=transformed_image,
            segmentation_target=transformed_target,
            plan=active_plan,
        )

    def apply_sample(
        self,
        sample: M3DSample,
        *,
        epoch: int,
        base_seed: int,
        view_index: int = 0,
        record_plan_in_metadata: bool = False,
    ) -> M3DSample:
        """Return an immutable sample copy with synchronised augmentation.

        The augmentation key is based on ``sample.provenance.sample_id``.  The
        caller should provide the current epoch from the shared epoch state used
        by persistent DataLoader workers.
        """

        context = AugmentationContext(
            sample_id=sample.provenance.sample_id,
            epoch=epoch,
            base_seed=base_seed,
            view_index=view_index,
        )
        result = self(
            sample.image,
            sample.segmentation_target,
            context=context,
        )

        provenance = sample.provenance
        if record_plan_in_metadata:
            metadata = dict(provenance.metadata)
            metadata["augmentation"] = result.plan.to_metadata()
            provenance = replace(
                provenance,
                metadata=MappingProxyType(metadata),
            )

        return replace(
            sample,
            provenance=provenance,
            image=result.image,
            segmentation_target=result.segmentation_target,
        )

    def _validate_inputs(
        self,
        image: Tensor,
        segmentation_target: Tensor | None,
    ) -> None:
        _validate_volume_tensor(
            image,
            name="image",
            channels=None,
            expected_spatial_shape=self.policy.expected_spatial_shape,
            reject_nonfinite=self.policy.reject_nonfinite,
        )
        if segmentation_target is not None:
            _validate_volume_tensor(
                segmentation_target,
                name="segmentation_target",
                channels=1,
                expected_spatial_shape=self.policy.expected_spatial_shape,
                reject_nonfinite=self.policy.reject_nonfinite,
            )
            if self.policy.validate_binary_target:
                _validate_binary_mask(segmentation_target)

    def _validate_outputs(
        self,
        image: Tensor,
        segmentation_target: Tensor | None,
    ) -> None:
        _validate_volume_tensor(
            image,
            name="transformed image",
            channels=None,
            expected_spatial_shape=self.policy.expected_spatial_shape,
            reject_nonfinite=self.policy.reject_nonfinite,
        )
        if segmentation_target is not None:
            _validate_volume_tensor(
                segmentation_target,
                name="transformed segmentation_target",
                channels=1,
                expected_spatial_shape=self.policy.expected_spatial_shape,
                reject_nonfinite=self.policy.reject_nonfinite,
            )
            if self.policy.validate_binary_target:
                _validate_binary_mask(segmentation_target)


def build_volume_transform(
    config: "ExperimentConfig",
    split: DataSplit | str,
) -> M3DVolumeTransform:
    """Build the original-M3D transform policy from the experiment config."""

    parsed_split = DataSplit.parse(split)
    main_shape = tuple(config.model.main_vision.image_size)
    seg_shape = tuple(config.model.seg_vision.image_size)
    if main_shape != seg_shape:
        raise TransformError(
            "The two independent image encoders must accept the same input "
            f"volume shape, got main={main_shape} and seg={seg_shape}"
        )

    if parsed_split is DataSplit.TRAIN:
        policy = AugmentationPolicy.original_m3d(
            expected_spatial_shape=main_shape,
        )
        training = True
    else:
        policy = AugmentationPolicy.identity(
            expected_spatial_shape=main_shape,
        )
        training = False

    return M3DVolumeTransform(policy, training=training)


def provenance_with_augmentation(
    provenance: SampleProvenance,
    plan: AugmentationPlan,
) -> SampleProvenance:
    """Return a provenance copy containing a JSON-safe augmentation trace."""

    metadata: dict[str, Any] = dict(provenance.metadata)
    metadata["augmentation"] = plan.to_metadata()
    return replace(
        provenance,
        metadata=MappingProxyType(metadata),
    )


def _validate_probability(value: float, *, name: str) -> None:
    if not math.isfinite(value):
        raise TransformError(f"{name} must be finite")
    if not 0.0 <= value <= 1.0:
        raise TransformError(f"{name} must be in [0,1], got {value}")


def _bernoulli(
    probability: float,
    *,
    generator: torch.Generator | None,
) -> bool:
    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    return bool(
        torch.rand(
            (),
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).item()
        < probability
    )


def _uniform_symmetric(
    magnitude: float,
    *,
    generator: torch.Generator | None,
) -> float:
    if magnitude <= 0.0:
        return 0.0
    unit = float(
        torch.rand(
            (),
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).item()
    )
    return (2.0 * unit - 1.0) * magnitude


def _validate_volume_tensor(
    tensor: Tensor,
    *,
    name: str,
    channels: int | None,
    expected_spatial_shape: tuple[int, int, int],
    reject_nonfinite: bool,
) -> None:
    if not isinstance(tensor, Tensor):
        raise TransformError(f"{name} must be torch.Tensor")
    if tensor.device.type != "cpu":
        raise TransformError(f"{name} must remain on CPU before collation")
    if tensor.dtype != torch.float32:
        raise TransformError(
            f"{name} must use torch.float32, got {tensor.dtype}"
        )
    if tensor.ndim != 4:
        raise TransformError(
            f"{name} must have shape [C,D,H,W], got {tuple(tensor.shape)}"
        )
    if channels is not None and tensor.shape[0] != channels:
        raise TransformError(
            f"{name} must have {channels} channel(s), got {tensor.shape[0]}"
        )
    actual_spatial = tuple(int(value) for value in tensor.shape[1:])
    if actual_spatial != tuple(expected_spatial_shape):
        raise TransformError(
            f"{name} spatial shape must be {tuple(expected_spatial_shape)}, "
            f"got {actual_spatial}"
        )
    if reject_nonfinite and not bool(torch.isfinite(tensor).all()):
        bad_count = int((~torch.isfinite(tensor)).sum().item())
        raise TransformError(f"{name} contains {bad_count} NaN/Inf values")


def _validate_binary_mask(mask: Tensor) -> None:
    valid = torch.logical_or(mask == 0.0, mask == 1.0)
    if not bool(valid.all()):
        invalid = mask[~valid]
        preview = invalid[: min(8, invalid.numel())].tolist()
        raise TransformError(
            "segmentation_target must remain binary after transform; "
            f"invalid values include {preview}"
        )


__all__ = [
    "AugmentationContext",
    "AugmentationPlan",
    "AugmentationPolicy",
    "M3DVolumeTransform",
    "SpatialAxis",
    "TransformError",
    "TransformResult",
    "build_volume_transform",
    "provenance_with_augmentation",
]
