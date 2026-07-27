"""Task-homogeneous collation and dynamic text padding for M3D.

The distributed task sampler emits one local microbatch whose samples all have
one task.  This module validates that contract, stacks CPU volumes, applies the
smallest configured text-length bucket, and constructs :class:`M3DBatch`.

Important behaviour
-------------------
* Caption/VQA/positioning batches carry ``segmentation_targets=None``.
* Segmentation batches stack every target, including valid all-zero masks.
* Task routing never depends on ``mask.sum()`` or any foreground statistic.
* Text is padded only at collate time.  Padding labels use ``IGNORE_INDEX``.
* The collator stores only tokenizer metadata, not the full Hugging Face
  tokenizer, so DataLoader workers do not receive another large tokenizer
  object merely to obtain ``pad_token_id``.
* No CUDA or distributed collective is called here.  With ``num_workers > 0``,
  the collator runs inside DataLoader worker processes and must remain a pure
  CPU operation.

DataLoader integration::

    collator = build_training_collator(
        config=config,
        tokenizer_bundle=tokenizer_bundle,
        rank=runtime.rank,
    )

    loader = DataLoader(
        multiplexed_dataset,
        batch_sampler=batch_sampler,
        collate_fn=collator,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
        prefetch_factor=config.data.prefetch_factor,
    )

The returned custom batch implements ``pin_memory()`` and ``to()`` in
:mod:`m3d.data.schema`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import torch
from torch import Tensor

from m3d.config import DataConfig, ExperimentConfig
from m3d.data.sampler import (
    SAMPLER_EPOCH_METADATA_KEY,
    SAMPLER_GLOBAL_SLOT_METADATA_KEY,
    SAMPLER_LOCAL_SLOT_METADATA_KEY,
    SAMPLER_RANK_METADATA_KEY,
    SAMPLER_STEP_METADATA_KEY,
    SAMPLER_TASK_OCCURRENCE_METADATA_KEY,
    sampler_position_from_samples,
)
from m3d.data.schema import (
    DataContractError,
    M3DBatch,
    M3DSample,
    TaskName,
    assert_homogeneous_samples,
    ensure_unique_sample_ids,
)
from m3d.tokenization import (
    IGNORE_INDEX,
    PaddedTextBatch,
    TokenizerBundle,
    TokenizerMetadata,
    choose_padded_sequence_length,
)


class CollationError(DataContractError):
    """Raised when samples cannot form one valid M3D microbatch."""


@dataclass(frozen=True, slots=True)
class SamplerBatchPosition:
    """Validated local sampler coordinates attached to a collated batch."""

    epoch: int
    step: int
    task: TaskName
    task_occurrence: int
    rank: int
    local_slots: tuple[int, ...]
    global_slots: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", TaskName.parse(self.task))
        integer_values = (
            self.epoch,
            self.step,
            self.task_occurrence,
            self.rank,
            *self.local_slots,
            *self.global_slots,
        )
        if any(not isinstance(value, int) or isinstance(value, bool) for value in integer_values):
            raise TypeError("SamplerBatchPosition values must be integers")
        if self.epoch < 0 or self.step < 0 or self.task_occurrence < 0 or self.rank < 0:
            raise CollationError("Sampler epoch, step, occurrence, and rank cannot be negative")
        if not self.local_slots:
            raise CollationError("SamplerBatchPosition must contain at least one local slot")
        if len(self.local_slots) != len(self.global_slots):
            raise CollationError("local_slots and global_slots must have equal lengths")
        if len(set(self.local_slots)) != len(self.local_slots):
            raise CollationError("local sampler slots must be unique")
        if len(set(self.global_slots)) != len(self.global_slots):
            raise CollationError("global sampler slots must be unique")

    @property
    def batch_size(self) -> int:
        return len(self.local_slots)

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "task": self.task.value,
            "task_occurrence": self.task_occurrence,
            "rank": self.rank,
            "local_slots": list(self.local_slots),
            "global_slots": list(self.global_slots),
        }


@dataclass(frozen=True, slots=True)
class CollatorSettings:
    """Lightweight, picklable settings copied into DataLoader workers."""

    pad_token_id: int
    model_max_length: int
    dynamic_padding: bool
    pad_to_multiple_of: int
    sequence_length_buckets: tuple[int, ...]
    expected_image_shape: tuple[int, int, int, int]
    expected_batch_size: int | None
    require_sampler_metadata: bool
    expected_rank: int | None

    def __post_init__(self) -> None:
        if self.pad_token_id < 0:
            raise CollationError("pad_token_id cannot be negative")
        if self.model_max_length <= 0:
            raise CollationError("model_max_length must be positive")
        if self.pad_to_multiple_of <= 0:
            raise CollationError("pad_to_multiple_of must be positive")
        if len(self.expected_image_shape) != 4:
            raise CollationError("expected_image_shape must be [C,D,H,W]")
        if any(value <= 0 for value in self.expected_image_shape):
            raise CollationError("expected image dimensions must be positive")
        if self.expected_batch_size is not None and self.expected_batch_size <= 0:
            raise CollationError("expected_batch_size must be positive or None")
        if self.expected_rank is not None and self.expected_rank < 0:
            raise CollationError("expected_rank cannot be negative")

        buckets = tuple(int(value) for value in self.sequence_length_buckets)
        if self.dynamic_padding:
            if not buckets:
                raise CollationError("dynamic padding requires sequence-length buckets")
            if tuple(sorted(set(buckets))) != buckets:
                raise CollationError("sequence-length buckets must be unique and increasing")
            if buckets[-1] != self.model_max_length:
                raise CollationError(
                    "largest sequence-length bucket must equal model_max_length"
                )
            for bucket in buckets:
                if bucket % self.pad_to_multiple_of != 0:
                    raise CollationError(
                        f"Sequence bucket {bucket} is not divisible by "
                        f"pad_to_multiple_of={self.pad_to_multiple_of}"
                    )
        object.__setattr__(self, "sequence_length_buckets", buckets)

    def data_config(self) -> DataConfig:
        """Create the small DataConfig view used by the shared bucket chooser."""

        return DataConfig(
            dynamic_padding=self.dynamic_padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            sequence_length_buckets=self.sequence_length_buckets,
            # The remaining fields are irrelevant to padding and keep defaults.
        )


@dataclass(frozen=True, slots=True)
class M3DCollator:
    """Collate one CPU task-homogeneous microbatch.

    Instances are immutable and safe to pickle into persistent DataLoader
    workers.  The full tokenizer is deliberately absent.
    """

    settings: CollatorSettings

    def __post_init__(self) -> None:
        if not isinstance(self.settings, CollatorSettings):
            raise TypeError("settings must be CollatorSettings")

    def __call__(self, samples: Sequence[M3DSample]) -> M3DBatch:
        sample_tuple = tuple(samples)
        task = assert_homogeneous_samples(sample_tuple)
        ensure_unique_sample_ids(sample_tuple)
        self._validate_batch_size(sample_tuple)
        self._validate_images(sample_tuple)

        if self.settings.require_sampler_metadata:
            self._validate_sampler_metadata(sample_tuple, task=task)

        images = self._stack_images(sample_tuple)
        text = self._pad_text(sample_tuple)
        segmentation_targets = self._stack_segmentation_targets(
            sample_tuple,
            task=task,
        )

        batch = M3DBatch(
            task=task,
            sample_ids=tuple(sample.sample_id for sample in sample_tuple),
            images=images,
            text=text,
            segmentation_targets=segmentation_targets,
            provenance=tuple(sample.provenance for sample in sample_tuple),
        )
        self._validate_constructed_batch(batch)
        return batch

    def _validate_batch_size(self, samples: Sequence[M3DSample]) -> None:
        expected = self.settings.expected_batch_size
        if expected is not None and len(samples) != expected:
            identifiers = ", ".join(sample.sample_id for sample in samples)
            raise CollationError(
                f"Expected local batch size {expected}, received {len(samples)}; "
                f"samples=[{identifiers}]"
            )

    def _validate_images(self, samples: Sequence[M3DSample]) -> None:
        expected = self.settings.expected_image_shape
        problems: list[str] = []
        for index, sample in enumerate(samples):
            image = sample.image
            if tuple(image.shape) != expected:
                problems.append(
                    f"index={index} sample={sample.sample_id!r} "
                    f"shape={tuple(image.shape)} expected={expected}"
                )
            if image.device.type != "cpu":
                problems.append(
                    f"index={index} sample={sample.sample_id!r} "
                    f"device={image.device} expected=cpu"
                )
            if image.dtype != torch.float32:
                problems.append(
                    f"index={index} sample={sample.sample_id!r} "
                    f"dtype={image.dtype} expected=torch.float32"
                )
        if problems:
            raise CollationError("Invalid image tensors:\n  - " + "\n  - ".join(problems))

    def _validate_sampler_metadata(
        self,
        samples: Sequence[M3DSample],
        *,
        task: TaskName,
    ) -> SamplerBatchPosition:
        try:
            epoch, step, observed_task = sampler_position_from_samples(samples)
        except Exception as exc:
            raise CollationError(f"Invalid sampler metadata: {exc}") from exc
        if observed_task is not task:
            raise CollationError(
                f"Sampler task {observed_task.value!r} differs from sample task "
                f"{task.value!r}"
            )

        first_metadata = samples[0].provenance.metadata
        task_occurrence = _required_int_metadata(
            first_metadata,
            SAMPLER_TASK_OCCURRENCE_METADATA_KEY,
            sample_id=samples[0].sample_id,
        )
        rank = _required_int_metadata(
            first_metadata,
            SAMPLER_RANK_METADATA_KEY,
            sample_id=samples[0].sample_id,
        )

        expected_rank = self.settings.expected_rank
        if expected_rank is not None and rank != expected_rank:
            raise CollationError(
                f"Sampler rank {rank} differs from collator rank {expected_rank}"
            )

        local_slots: list[int] = []
        global_slots: list[int] = []
        for local_position, sample in enumerate(samples):
            metadata = sample.provenance.metadata
            values = {
                "epoch": _required_int_metadata(
                    metadata,
                    SAMPLER_EPOCH_METADATA_KEY,
                    sample_id=sample.sample_id,
                ),
                "step": _required_int_metadata(
                    metadata,
                    SAMPLER_STEP_METADATA_KEY,
                    sample_id=sample.sample_id,
                ),
                "task_occurrence": _required_int_metadata(
                    metadata,
                    SAMPLER_TASK_OCCURRENCE_METADATA_KEY,
                    sample_id=sample.sample_id,
                ),
                "rank": _required_int_metadata(
                    metadata,
                    SAMPLER_RANK_METADATA_KEY,
                    sample_id=sample.sample_id,
                ),
                "local_slot": _required_int_metadata(
                    metadata,
                    SAMPLER_LOCAL_SLOT_METADATA_KEY,
                    sample_id=sample.sample_id,
                ),
                "global_slot": _required_int_metadata(
                    metadata,
                    SAMPLER_GLOBAL_SLOT_METADATA_KEY,
                    sample_id=sample.sample_id,
                ),
            }
            expected_values = {
                "epoch": epoch,
                "step": step,
                "task_occurrence": task_occurrence,
                "rank": rank,
                "local_slot": local_position,
                "global_slot": rank * len(samples) + local_position,
            }
            disagreements = {
                name: (values[name], expected_value)
                for name, expected_value in expected_values.items()
                if values[name] != expected_value
            }
            if disagreements:
                formatted = ", ".join(
                    f"{name}={actual} expected={expected}"
                    for name, (actual, expected) in disagreements.items()
                )
                raise CollationError(
                    f"Sampler metadata mismatch for sample {sample.sample_id!r}: "
                    f"{formatted}"
                )
            local_slots.append(values["local_slot"])
            global_slots.append(values["global_slot"])

        return SamplerBatchPosition(
            epoch=epoch,
            step=step,
            task=task,
            task_occurrence=task_occurrence,
            rank=rank,
            local_slots=tuple(local_slots),
            global_slots=tuple(global_slots),
        )

    def _stack_images(self, samples: Sequence[M3DSample]) -> Tensor:
        try:
            images = torch.stack([sample.image for sample in samples], dim=0)
        except RuntimeError as exc:
            details = ", ".join(
                f"{sample.sample_id}:{tuple(sample.image.shape)}" for sample in samples
            )
            raise CollationError(f"Could not stack images: {details}") from exc
        if images.dtype != torch.float32 or images.device.type != "cpu":
            raise CollationError(
                f"Collated images must be CPU float32, got {images.device}/{images.dtype}"
            )
        return images.contiguous()

    def _pad_text(self, samples: Sequence[M3DSample]) -> PaddedTextBatch:
        examples = tuple(sample.text for sample in samples)
        lengths = tuple(int(example.length) for example in examples)
        data_config = self.settings.data_config()
        target_length = choose_padded_sequence_length(
            lengths,
            data_config=data_config,
            model_max_length=self.settings.model_max_length,
        )

        batch_size = len(examples)
        input_ids = torch.full(
            (batch_size, target_length),
            fill_value=self.settings.pad_token_id,
            dtype=torch.long,
        )
        labels = torch.full(
            (batch_size, target_length),
            fill_value=IGNORE_INDEX,
            dtype=torch.long,
        )
        attention_mask = torch.zeros(
            (batch_size, target_length),
            dtype=torch.bool,
        )

        for row, (sample, example) in enumerate(zip(samples, examples)):
            length = int(example.length)
            if length > target_length:
                raise CollationError(
                    f"Sample {sample.sample_id!r} has {length} tokens, exceeding "
                    f"selected bucket {target_length}"
                )
            if example.input_ids.device.type != "cpu":
                raise CollationError(
                    f"Text tensors for {sample.sample_id!r} must remain on CPU"
                )
            input_ids[row, :length].copy_(example.input_ids)
            labels[row, :length].copy_(example.labels)
            attention_mask[row, :length].copy_(example.attention_mask)

        text = PaddedTextBatch(
            input_ids=input_ids.contiguous(),
            labels=labels.contiguous(),
            attention_mask=attention_mask.contiguous(),
            sequence_length=target_length,
            unpadded_lengths=torch.tensor(lengths, dtype=torch.int32),
        )
        _validate_padding_contract(
            text,
            pad_token_id=self.settings.pad_token_id,
        )
        return text

    def _stack_segmentation_targets(
        self,
        samples: Sequence[M3DSample],
        *,
        task: TaskName,
    ) -> Tensor | None:
        if task is not TaskName.SEGMENTATION:
            unexpected = [
                sample.sample_id
                for sample in samples
                if sample.segmentation_target is not None
            ]
            if unexpected:
                raise CollationError(
                    f"Non-segmentation task {task.value!r} contains dense targets: "
                    + ", ".join(unexpected)
                )
            return None

        missing = [
            sample.sample_id
            for sample in samples
            if sample.segmentation_target is None
        ]
        if missing:
            raise CollationError(
                "Segmentation batch is missing targets for: " + ", ".join(missing)
            )

        targets = [sample.segmentation_target for sample in samples]
        # Static type narrowing is not available after the explicit check above.
        materialized = [target for target in targets if target is not None]
        try:
            stacked = torch.stack(materialized, dim=0)
        except RuntimeError as exc:
            details = ", ".join(
                f"{sample.sample_id}:"
                f"{None if sample.segmentation_target is None else tuple(sample.segmentation_target.shape)}"
                for sample in samples
            )
            raise CollationError(
                f"Could not stack segmentation targets: {details}"
            ) from exc

        if stacked.dtype != torch.float32 or stacked.device.type != "cpu":
            raise CollationError(
                "Collated segmentation targets must be CPU float32, got "
                f"{stacked.device}/{stacked.dtype}"
            )
        # Do not inspect stacked.sum() for task routing.  All-zero targets are
        # valid and intentionally preserved.
        valid_values = torch.logical_or(stacked == 0.0, stacked == 1.0)
        if not bool(valid_values.all()):
            invalid = stacked[~valid_values].flatten()[:8].tolist()
            raise CollationError(
                f"Segmentation targets must remain binary; invalid values={invalid}"
            )
        return stacked.contiguous()

    def _validate_constructed_batch(self, batch: M3DBatch) -> None:
        expected_shape = (
            batch.batch_size,
            *self.settings.expected_image_shape,
        )
        if tuple(batch.images.shape) != expected_shape:
            raise CollationError(
                f"Constructed image batch has shape {tuple(batch.images.shape)}, "
                f"expected {expected_shape}"
            )
        if batch.task is TaskName.SEGMENTATION:
            expected_target_shape = (
                batch.batch_size,
                1,
                *self.settings.expected_image_shape[1:],
            )
            if batch.segmentation_targets is None or tuple(
                batch.segmentation_targets.shape
            ) != expected_target_shape:
                actual = (
                    None
                    if batch.segmentation_targets is None
                    else tuple(batch.segmentation_targets.shape)
                )
                raise CollationError(
                    f"Constructed segmentation targets have shape {actual}, "
                    f"expected {expected_target_shape}"
                )
        elif batch.segmentation_targets is not None:
            raise CollationError("Text/positioning batch unexpectedly contains masks")


def _required_int_metadata(
    metadata: Mapping[str, Any],
    key: str,
    *,
    sample_id: str,
) -> int:
    try:
        value = metadata[key]
    except KeyError as exc:
        raise CollationError(
            f"Sample {sample_id!r} lacks sampler metadata key {key!r}"
        ) from exc
    if not isinstance(value, int) or isinstance(value, bool):
        raise CollationError(
            f"Sampler metadata {key!r} for sample {sample_id!r} must be int, "
            f"got {type(value).__name__}"
        )
    if value < 0:
        raise CollationError(
            f"Sampler metadata {key!r} for sample {sample_id!r} cannot be negative"
        )
    return value


def _validate_padding_contract(
    text: PaddedTextBatch,
    *,
    pad_token_id: int,
) -> None:
    """Verify right padding, ignored labels, and true unpadded lengths."""

    if text.input_ids.ndim != 2:
        raise CollationError("Padded input_ids must be [B,L]")
    batch_size, sequence_length = text.input_ids.shape
    if text.sequence_length != sequence_length:
        raise CollationError(
            f"PaddedTextBatch sequence_length={text.sequence_length} but tensor "
            f"width={sequence_length}"
        )
    if text.unpadded_lengths.shape != (batch_size,):
        raise CollationError("unpadded_lengths must contain one value per row")

    for row in range(batch_size):
        length = int(text.unpadded_lengths[row].item())
        if not 0 < length <= sequence_length:
            raise CollationError(
                f"Row {row} has invalid unpadded length {length} for width "
                f"{sequence_length}"
            )
        if not bool(text.attention_mask[row, :length].all()):
            raise CollationError(f"Row {row} has masked tokens inside its true length")
        if bool(text.attention_mask[row, length:].any()):
            raise CollationError(f"Row {row} has active attention in right padding")
        if length < sequence_length:
            if not bool(text.input_ids[row, length:].eq(pad_token_id).all()):
                raise CollationError(f"Row {row} contains non-pad IDs after true length")
            if not bool(text.labels[row, length:].eq(IGNORE_INDEX).all()):
                raise CollationError(f"Row {row} padding labels are not IGNORE_INDEX")
        if not bool(text.labels[row, :length].ne(IGNORE_INDEX).any()):
            raise CollationError(f"Row {row} contains no supervised target token")


def build_training_collator(
    *,
    config: ExperimentConfig,
    tokenizer_bundle: TokenizerBundle,
    rank: int,
) -> M3DCollator:
    """Build the strict collator used by the distributed training loader."""

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be ExperimentConfig")
    if not isinstance(tokenizer_bundle, TokenizerBundle):
        raise TypeError("tokenizer_bundle must be TokenizerBundle")
    if not isinstance(rank, int) or isinstance(rank, bool) or rank < 0:
        raise ValueError("rank must be a non-negative integer")

    main = config.model.main_vision
    expected_image_shape = (
        main.image_channels,
        *main.image_size,
    )
    settings = CollatorSettings(
        pad_token_id=tokenizer_bundle.metadata.pad_token_id,
        model_max_length=config.model.model_max_length,
        dynamic_padding=config.data.dynamic_padding,
        pad_to_multiple_of=config.data.pad_to_multiple_of,
        sequence_length_buckets=tuple(config.data.sequence_length_buckets),
        expected_image_shape=expected_image_shape,
        expected_batch_size=config.optimization.per_device_batch_size,
        require_sampler_metadata=True,
        expected_rank=rank,
    )
    return M3DCollator(settings=settings)


def build_evaluation_collator(
    *,
    config: ExperimentConfig,
    tokenizer_bundle: TokenizerBundle,
    expected_batch_size: int | None = None,
) -> M3DCollator:
    """Build a collator for sequential validation/test loaders.

    Evaluation does not require sampler-reserved metadata and may retain a
    smaller final batch.
    """

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be ExperimentConfig")
    if not isinstance(tokenizer_bundle, TokenizerBundle):
        raise TypeError("tokenizer_bundle must be TokenizerBundle")
    main = config.model.main_vision
    return M3DCollator(
        settings=CollatorSettings(
            pad_token_id=tokenizer_bundle.metadata.pad_token_id,
            model_max_length=config.model.model_max_length,
            dynamic_padding=config.data.dynamic_padding,
            pad_to_multiple_of=config.data.pad_to_multiple_of,
            sequence_length_buckets=tuple(config.data.sequence_length_buckets),
            expected_image_shape=(main.image_channels, *main.image_size),
            expected_batch_size=expected_batch_size,
            require_sampler_metadata=False,
            expected_rank=None,
        )
    )


def batch_padding_statistics(batch: M3DBatch) -> Mapping[str, float | int]:
    """Return lightweight padding metrics for trainer logging.

    This helper reads CPU or CUDA metadata tensors but should be called only at
    configured logging intervals.  It is diagnostic and never affects routing.
    """

    if not isinstance(batch, M3DBatch):
        raise TypeError("batch must be M3DBatch")
    lengths = batch.text.unpadded_lengths.to(device="cpu", dtype=torch.int64)
    real_tokens = int(lengths.sum().item())
    allocated_tokens = batch.batch_size * batch.text.sequence_length
    padding_tokens = allocated_tokens - real_tokens
    ratio = 0.0 if allocated_tokens == 0 else padding_tokens / allocated_tokens
    return MappingProxyType(
        {
            "batch_size": batch.batch_size,
            "sequence_bucket": batch.text.sequence_length,
            "minimum_length": int(lengths.min().item()),
            "maximum_length": int(lengths.max().item()),
            "real_tokens": real_tokens,
            "allocated_tokens": allocated_tokens,
            "padding_tokens": padding_tokens,
            "padding_ratio": float(ratio),
        }
    )


__all__ = [
    "CollationError",
    "CollatorSettings",
    "M3DCollator",
    "SamplerBatchPosition",
    "batch_padding_statistics",
    "build_evaluation_collator",
    "build_training_collator",
]
