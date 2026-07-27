"""Shared data contracts for every M3D task.

This module is the first file imported by the data pipeline.  It defines the
canonical task names and the exact objects exchanged between dataset workers,
the collator, the distributed task sampler, and the model.

The most important contract is explicit task identity.  A sample is a
segmentation sample because ``sample.task is TaskName.SEGMENTATION`` -- never
because its target mask contains non-zero voxels.  A valid all-zero target mask
therefore remains a real segmentation example and contributes to the loss.

Tensor conventions
------------------
* One image: ``[C, D, H, W]``.
* One segmentation target: ``[1, D, H, W]``.
* One batch of images: ``[B, C, D, H, W]``.
* One batch of masks: ``[B, 1, D, H, W]``.
* Dataset workers return CPU tensors.  Device transfer happens after collation.
* Images and masks use ``torch.float32`` before BF16 autocast.  Masks remain
  binary-valued, but an all-zero mask is explicitly allowed.
* Every training batch is task-homogeneous across both its local samples and
  all distributed ranks.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum, IntEnum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Sequence

import torch
from torch import Tensor

if TYPE_CHECKING:
    from m3d.tokenization import EncodedText, PaddedTextBatch


class DataContractError(ValueError):
    """Raised when a dataset sample or batch violates the M3D contract."""


class TaskId(IntEnum):
    """Stable numeric task IDs stored in checkpoints and sampler state."""

    CAPTION = 0
    VQA_CLOSED = 1
    VQA_OPEN = 2
    VQA_YES_NO = 3
    POSITIONING = 4
    SEGMENTATION = 5


class TaskName(str, Enum):
    """Canonical names used in YAML, logs, dataset registries, and metrics."""

    CAPTION = "caption"
    VQA_CLOSED = "vqa_closed"
    VQA_OPEN = "vqa_open"
    VQA_YES_NO = "vqa_yes_no"
    POSITIONING = "positioning"
    SEGMENTATION = "segmentation"

    @property
    def task_id(self) -> TaskId:
        return _TASK_IDS[self]

    @property
    def requires_segmentation_target(self) -> bool:
        return self is TaskName.SEGMENTATION

    @property
    def uses_box_tokens(self) -> bool:
        return self is TaskName.POSITIONING

    @property
    def is_text_only_objective(self) -> bool:
        """Whether the task has no dense segmentation-loss branch."""

        return not self.requires_segmentation_target

    @classmethod
    def parse(cls, value: str | "TaskName") -> "TaskName":
        """Parse canonical names and a small set of legacy M3D aliases."""

        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError(
                f"Task name must be str or TaskName, got {type(value).__name__}"
            )

        normalised = value.strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "cap": cls.CAPTION,
            "caption": cls.CAPTION,
            "vqa": cls.VQA_OPEN,
            "vqa_open": cls.VQA_OPEN,
            "open_vqa": cls.VQA_OPEN,
            "vqa_closed": cls.VQA_CLOSED,
            "closed_vqa": cls.VQA_CLOSED,
            "vqa_close": cls.VQA_CLOSED,
            "vqa_yes_no": cls.VQA_YES_NO,
            "vqa_yn": cls.VQA_YES_NO,
            "yes_no": cls.VQA_YES_NO,
            "rec": cls.POSITIONING,
            "reg": cls.POSITIONING,
            "position": cls.POSITIONING,
            "positioning": cls.POSITIONING,
            "seg": cls.SEGMENTATION,
            "refseg": cls.SEGMENTATION,
            "segmentation": cls.SEGMENTATION,
        }
        try:
            return aliases[normalised]
        except KeyError as exc:
            allowed = ", ".join(task.value for task in cls)
            raise DataContractError(
                f"Unknown task name {value!r}. Canonical names: {allowed}"
            ) from exc


_TASK_IDS: Mapping[TaskName, TaskId] = MappingProxyType(
    {
        TaskName.CAPTION: TaskId.CAPTION,
        TaskName.VQA_CLOSED: TaskId.VQA_CLOSED,
        TaskName.VQA_OPEN: TaskId.VQA_OPEN,
        TaskName.VQA_YES_NO: TaskId.VQA_YES_NO,
        TaskName.POSITIONING: TaskId.POSITIONING,
        TaskName.SEGMENTATION: TaskId.SEGMENTATION,
    }
)

TASK_ORDER: tuple[TaskName, ...] = tuple(
    sorted(TaskName, key=lambda task: int(task.task_id))
)


class DataSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

    @classmethod
    def parse(cls, value: str | "DataSplit") -> "DataSplit":
        if isinstance(value, cls):
            return value
        if not isinstance(value, str):
            raise TypeError(
                f"Data split must be str or DataSplit, got {type(value).__name__}"
            )

        normalised = value.strip().lower()
        aliases = {
            "train": cls.TRAIN,
            "training": cls.TRAIN,
            "val": cls.VALIDATION,
            "valid": cls.VALIDATION,
            "validation": cls.VALIDATION,
            "test": cls.TEST,
            "testing": cls.TEST,
        }
        try:
            return aliases[normalised]
        except KeyError as exc:
            allowed = ", ".join(split.value for split in cls)
            raise DataContractError(
                f"Unknown data split {value!r}. Allowed values: {allowed}"
            ) from exc


@dataclass(frozen=True, slots=True)
class SampleProvenance:
    """Stable identity and source information for one logical sample."""

    sample_id: str
    source_name: str
    source_index: int
    split: DataSplit
    image_path: Path
    mask_path: Path | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        sample_id = self.sample_id.strip()
        source_name = self.source_name.strip()
        if not sample_id:
            raise DataContractError("sample_id cannot be empty")
        if not source_name:
            raise DataContractError("source_name cannot be empty")
        if self.source_index < 0:
            raise DataContractError("source_index cannot be negative")

        image_path = Path(self.image_path)
        mask_path = None if self.mask_path is None else Path(self.mask_path)
        metadata = MappingProxyType(dict(self.metadata))

        object.__setattr__(self, "sample_id", sample_id)
        object.__setattr__(self, "source_name", source_name)
        object.__setattr__(self, "split", DataSplit.parse(self.split))
        object.__setattr__(self, "image_path", image_path)
        object.__setattr__(self, "mask_path", mask_path)
        object.__setattr__(self, "metadata", metadata)

    def with_resolved_paths(self, root: str | Path) -> "SampleProvenance":
        """Return a copy whose relative paths are resolved below ``root``."""

        root_path = Path(root).expanduser().resolve()

        def resolve_one(path: Path | None) -> Path | None:
            if path is None:
                return None
            candidate = path.expanduser()
            if not candidate.is_absolute():
                candidate = root_path / candidate
            return candidate.resolve()

        return replace(
            self,
            image_path=resolve_one(self.image_path),
            mask_path=resolve_one(self.mask_path),
        )


@dataclass(frozen=True, slots=True)
class M3DSample:
    """One fully prepared, unpadded training example returned by a dataset.

    ``segmentation_target`` is required only for ``TaskName.SEGMENTATION`` and
    forbidden for all other tasks.  Its values are deliberately not inspected
    to decide task identity; an entirely zero target is legal.
    """

    task: TaskName
    provenance: SampleProvenance
    image: Tensor
    text: EncodedText
    question: str
    answer: str
    segmentation_target: Tensor | None = None

    def __post_init__(self) -> None:
        task = TaskName.parse(self.task)
        object.__setattr__(self, "task", task)

        if not isinstance(self.provenance, SampleProvenance):
            raise TypeError("provenance must be a SampleProvenance")
        _validate_encoded_text(self.text)

        question = self.question.strip()
        answer = self.answer.strip()
        if not question:
            raise DataContractError(
                f"Question is empty for sample {self.provenance.sample_id!r}"
            )
        if not answer:
            raise DataContractError(
                f"Answer is empty for sample {self.provenance.sample_id!r}"
            )
        object.__setattr__(self, "question", question)
        object.__setattr__(self, "answer", answer)

        _validate_image_tensor(self.image, sample_id=self.provenance.sample_id)

        if task.requires_segmentation_target:
            if self.segmentation_target is None:
                raise DataContractError(
                    "Segmentation task requires segmentation_target for sample "
                    f"{self.provenance.sample_id!r}"
                )
            _validate_segmentation_tensor(
                self.segmentation_target,
                image=self.image,
                sample_id=self.provenance.sample_id,
            )
        elif self.segmentation_target is not None:
            raise DataContractError(
                f"Task {task.value!r} must not carry a segmentation_target; "
                f"sample={self.provenance.sample_id!r}"
            )

    @property
    def sample_id(self) -> str:
        return self.provenance.sample_id

    @property
    def has_segmentation_target(self) -> bool:
        """Explicit structural signal; never inferred from target values."""

        return self.task.requires_segmentation_target

    @property
    def sequence_length(self) -> int:
        return self.text.length


@dataclass(frozen=True, slots=True)
class M3DBatch:
    """One task-homogeneous batch ready for model execution."""

    task: TaskName
    sample_ids: tuple[str, ...]
    images: Tensor
    text: PaddedTextBatch
    segmentation_targets: Tensor | None = None
    provenance: tuple[SampleProvenance, ...] = ()

    def __post_init__(self) -> None:
        task = TaskName.parse(self.task)
        object.__setattr__(self, "task", task)

        _validate_padded_text_batch(self.text)
        if self.images.ndim != 5:
            raise DataContractError(
                "Batch images must have shape [B,C,D,H,W], got "
                f"{tuple(self.images.shape)}"
            )
        if not self.images.is_floating_point():
            raise TypeError("Batch images must use a floating-point dtype")

        batch_size = int(self.images.shape[0])
        if batch_size <= 0:
            raise DataContractError("A batch must contain at least one sample")
        if len(self.sample_ids) != batch_size:
            raise DataContractError(
                f"sample_ids has {len(self.sample_ids)} entries but batch size is "
                f"{batch_size}"
            )
        if len(set(self.sample_ids)) != len(self.sample_ids):
            raise DataContractError("sample_ids must be unique within a batch")
        if self.text.input_ids.shape[0] != batch_size:
            raise DataContractError(
                "Text batch size does not match image batch size: "
                f"text={self.text.input_ids.shape[0]}, images={batch_size}"
            )
        if self.provenance and len(self.provenance) != batch_size:
            raise DataContractError(
                "provenance length must be zero or equal to the batch size"
            )

        if task.requires_segmentation_target:
            targets = self.segmentation_targets
            if targets is None:
                raise DataContractError(
                    "A segmentation batch requires segmentation_targets"
                )
            if targets.ndim != 5:
                raise DataContractError(
                    "Batch segmentation targets must have shape [B,1,D,H,W], got "
                    f"{tuple(targets.shape)}"
                )
            if int(targets.shape[0]) != batch_size:
                raise DataContractError(
                    "Segmentation target batch size does not match images"
                )
            if int(targets.shape[1]) != 1:
                raise DataContractError(
                    "Segmentation targets must have exactly one channel"
                )
            if tuple(targets.shape[2:]) != tuple(self.images.shape[2:]):
                raise DataContractError(
                    "Segmentation spatial shape must match image spatial shape: "
                    f"target={tuple(targets.shape[2:])}, "
                    f"image={tuple(self.images.shape[2:])}"
                )
            if not targets.is_floating_point():
                raise TypeError(
                    "Batch segmentation targets must use a floating-point dtype"
                )
        elif self.segmentation_targets is not None:
            raise DataContractError(
                f"Task {task.value!r} must not carry segmentation_targets"
            )

    @property
    def batch_size(self) -> int:
        return int(self.images.shape[0])

    @property
    def task_id(self) -> int:
        return int(self.task.task_id)

    @property
    def has_segmentation_targets(self) -> bool:
        return self.task.requires_segmentation_target

    def to(
        self,
        device: torch.device | str,
        *,
        non_blocking: bool = False,
    ) -> "M3DBatch":
        """Move model-facing tensors while retaining CPU provenance metadata."""

        destination = torch.device(device)
        moved_text = type(self.text)(
            input_ids=self.text.input_ids.to(
                destination, non_blocking=non_blocking
            ),
            labels=self.text.labels.to(destination, non_blocking=non_blocking),
            attention_mask=self.text.attention_mask.to(
                destination, non_blocking=non_blocking
            ),
            sequence_length=self.text.sequence_length,
            unpadded_lengths=self.text.unpadded_lengths.to(
                destination, non_blocking=non_blocking
            ),
        )
        moved_targets = (
            None
            if self.segmentation_targets is None
            else self.segmentation_targets.to(
                destination, non_blocking=non_blocking
            )
        )
        return M3DBatch(
            task=self.task,
            sample_ids=self.sample_ids,
            images=self.images.to(destination, non_blocking=non_blocking),
            text=moved_text,
            segmentation_targets=moved_targets,
            provenance=self.provenance,
        )

    def pin_memory(self) -> "M3DBatch":
        """Pin CPU tensors for asynchronous host-to-device transfer."""

        if self.images.device.type != "cpu":
            return self

        pinned_text = type(self.text)(
            input_ids=self.text.input_ids.pin_memory(),
            labels=self.text.labels.pin_memory(),
            attention_mask=self.text.attention_mask.pin_memory(),
            sequence_length=self.text.sequence_length,
            unpadded_lengths=self.text.unpadded_lengths.pin_memory(),
        )
        pinned_targets = (
            None
            if self.segmentation_targets is None
            else self.segmentation_targets.pin_memory()
        )
        return M3DBatch(
            task=self.task,
            sample_ids=self.sample_ids,
            images=self.images.pin_memory(),
            text=pinned_text,
            segmentation_targets=pinned_targets,
            provenance=self.provenance,
        )

    def model_inputs(self) -> dict[str, Tensor | None]:
        """Return the tensor-only contract consumed by the future M3D model.

        The Python task enum remains on the batch so the trainer can select the
        correct compiled graph without converting a CUDA tensor to a Python
        scalar.  ``segmentation_targets`` is structurally present only for the
        segmentation task.
        """

        return {
            "images": self.images,
            "input_ids": self.text.input_ids,
            "labels": self.text.labels,
            "attention_mask": self.text.attention_mask,
            "segmentation_targets": self.segmentation_targets,
        }


@dataclass(frozen=True, slots=True)
class TaskDatasetInfo:
    """Small immutable description used by the task sampler and logs."""

    task: TaskName
    dataset_name: str
    size: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", TaskName.parse(self.task))
        name = self.dataset_name.strip()
        if not name:
            raise DataContractError("dataset_name cannot be empty")
        if self.size <= 0:
            raise DataContractError(
                f"Dataset {name!r} must contain at least one sample"
            )
        object.__setattr__(self, "dataset_name", name)


def canonical_task_mapping(
    values: Mapping[str | TaskName, Any],
    *,
    require_all: bool = False,
) -> dict[TaskName, Any]:
    """Normalise a user mapping and reject aliases that collide."""

    result: dict[TaskName, Any] = {}
    original_keys: dict[TaskName, str] = {}
    for raw_name, value in values.items():
        task = TaskName.parse(raw_name)
        if task in result:
            raise DataContractError(
                "Multiple mapping keys resolve to the same task "
                f"{task.value!r}: {original_keys[task]!r} and {raw_name!r}"
            )
        result[task] = value
        original_keys[task] = str(raw_name)

    if require_all:
        missing = [task.value for task in TASK_ORDER if task not in result]
        if missing:
            raise DataContractError(
                "Task mapping is missing: " + ", ".join(missing)
            )
    return result


def assert_homogeneous_samples(samples: Sequence[M3DSample]) -> TaskName:
    """Validate a local batch before stacking and return its one task."""

    if not samples:
        raise DataContractError("Cannot collate an empty sample sequence")
    task = samples[0].task
    mismatches = [
        (index, sample.task.value, sample.sample_id)
        for index, sample in enumerate(samples)
        if sample.task is not task
    ]
    if mismatches:
        details = ", ".join(
            f"index={index} task={name} sample={sample_id}"
            for index, name, sample_id in mismatches
        )
        raise DataContractError(
            f"Task-homogeneous batch required; expected {task.value!r}; {details}"
        )
    return task


def ensure_unique_sample_ids(samples: Iterable[M3DSample]) -> None:
    """Reject accidental duplicate indices inside one local batch."""

    seen: set[str] = set()
    duplicates: list[str] = []
    for sample in samples:
        if sample.sample_id in seen:
            duplicates.append(sample.sample_id)
        seen.add(sample.sample_id)
    if duplicates:
        duplicate_text = ", ".join(sorted(set(duplicates)))
        raise DataContractError(
            f"Duplicate sample IDs in local batch: {duplicate_text}"
        )



def _validate_encoded_text(text: Any) -> None:
    """Validate the lightweight tokenized-example protocol without importing
    Hugging Face in every dataset worker at schema-import time.
    """

    required = (
        "input_ids",
        "labels",
        "attention_mask",
        "prompt_token_count",
        "supervised_token_count",
        "was_truncated",
        "length",
    )
    missing = [name for name in required if not hasattr(text, name)]
    if missing:
        raise TypeError(
            "text does not satisfy the EncodedText contract; missing: "
            + ", ".join(missing)
        )
    if not isinstance(text.input_ids, Tensor) or text.input_ids.ndim != 1:
        raise DataContractError("Encoded text input_ids must be a 1D tensor")
    if not isinstance(text.labels, Tensor) or text.labels.shape != text.input_ids.shape:
        raise DataContractError("Encoded text labels must match input_ids")
    if (
        not isinstance(text.attention_mask, Tensor)
        or text.attention_mask.shape != text.input_ids.shape
    ):
        raise DataContractError("Encoded text attention_mask must match input_ids")


def _validate_padded_text_batch(text: Any) -> None:
    """Validate the collated text-batch protocol without a runtime import."""

    required = (
        "input_ids",
        "labels",
        "attention_mask",
        "sequence_length",
        "unpadded_lengths",
    )
    missing = [name for name in required if not hasattr(text, name)]
    if missing:
        raise TypeError(
            "text does not satisfy the PaddedTextBatch contract; missing: "
            + ", ".join(missing)
        )
    if not isinstance(text.input_ids, Tensor) or text.input_ids.ndim != 2:
        raise DataContractError("Padded input_ids must have shape [B,L]")
    if not isinstance(text.labels, Tensor) or text.labels.shape != text.input_ids.shape:
        raise DataContractError("Padded labels must match input_ids")
    if (
        not isinstance(text.attention_mask, Tensor)
        or text.attention_mask.shape != text.input_ids.shape
    ):
        raise DataContractError("Padded attention_mask must match input_ids")
    if (
        not isinstance(text.unpadded_lengths, Tensor)
        or text.unpadded_lengths.ndim != 1
        or text.unpadded_lengths.shape[0] != text.input_ids.shape[0]
    ):
        raise DataContractError(
            "unpadded_lengths must have one value per batch element"
        )


def _validate_image_tensor(image: Tensor, *, sample_id: str) -> None:
    if not isinstance(image, Tensor):
        raise TypeError(
            f"Image for sample {sample_id!r} must be torch.Tensor, "
            f"got {type(image).__name__}"
        )
    if image.ndim != 4:
        raise DataContractError(
            f"Image for sample {sample_id!r} must have shape [C,D,H,W], "
            f"got {tuple(image.shape)}"
        )
    if int(image.shape[0]) <= 0:
        raise DataContractError(f"Image {sample_id!r} has no channels")
    if any(int(dimension) <= 0 for dimension in image.shape[1:]):
        raise DataContractError(
            f"Image {sample_id!r} has invalid spatial shape {tuple(image.shape[1:])}"
        )
    if image.device.type != "cpu":
        raise DataContractError(
            f"Dataset image {sample_id!r} must remain on CPU, got {image.device}"
        )
    if image.dtype != torch.float32:
        raise TypeError(
            f"Dataset image {sample_id!r} must use torch.float32, got {image.dtype}"
        )
    if not bool(torch.isfinite(image).all()):
        raise DataContractError(
            f"Image {sample_id!r} contains NaN or infinite values"
        )


def _validate_segmentation_tensor(
    target: Tensor,
    *,
    image: Tensor,
    sample_id: str,
) -> None:
    if not isinstance(target, Tensor):
        raise TypeError(
            f"Segmentation target for {sample_id!r} must be torch.Tensor"
        )
    if target.ndim != 4:
        raise DataContractError(
            f"Segmentation target for {sample_id!r} must have shape [1,D,H,W], "
            f"got {tuple(target.shape)}"
        )
    if int(target.shape[0]) != 1:
        raise DataContractError(
            f"Segmentation target for {sample_id!r} must have one channel, "
            f"got {target.shape[0]}"
        )
    if tuple(target.shape[1:]) != tuple(image.shape[1:]):
        raise DataContractError(
            f"Segmentation target for {sample_id!r} has spatial shape "
            f"{tuple(target.shape[1:])}, expected {tuple(image.shape[1:])}"
        )
    if target.device.type != "cpu":
        raise DataContractError(
            f"Dataset target {sample_id!r} must remain on CPU, got {target.device}"
        )
    if target.dtype != torch.float32:
        raise TypeError(
            f"Segmentation target {sample_id!r} must use torch.float32, "
            f"got {target.dtype}"
        )
    if not bool(torch.isfinite(target).all()):
        raise DataContractError(
            f"Segmentation target {sample_id!r} contains NaN or infinite values"
        )

    # Validate binary supervision without using target.sum() to infer task type.
    valid_values = torch.logical_or(target == 0.0, target == 1.0)
    if not bool(valid_values.all()):
        invalid = target[~valid_values]
        preview = invalid.flatten()[:8].tolist()
        raise DataContractError(
            f"Segmentation target {sample_id!r} must be binary 0/1; "
            f"invalid values include {preview}"
        )
