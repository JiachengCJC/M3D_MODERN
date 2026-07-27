"""Typed dataset catalogue for the modernized M3D training pipeline.

This module sits between the deterministic JSONL manifests and the concrete
``torch.utils.data.Dataset`` implementation.  Its purpose is to make every
routing decision explicit before worker processes begin loading volumes.

The original M3D code decides behaviour through a mixture of dataset classes,
CSV names, and checks such as ``seg.sum() == 0``.  The modernized pipeline uses
three explicit pieces of information instead:

* :class:`~m3d.data.schema.TaskName` selects the model execution path;
* :class:`~m3d.data.manifest.PromptVariant` selects prompt construction;
* :class:`TaskRecordGroup` owns the records for one homogeneous task.

A positioning record is an important example.  It *reads* a dense mask to
construct a box answer after spatial augmentation, but it does not return that
mask to the dense segmentation-loss branch.  A segmentation record both reads
and returns a dense target.  Keeping those capabilities separate prevents the
old all-zero-mask ambiguity.

The catalogue is immutable and deterministic.  It preserves manifest order,
validates every prompt variant, exposes stable dataset sizes for the future
task-balanced sampler, and rejects positive training weights for missing tasks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

from m3d.config import ExperimentConfig, load_config
from m3d.data.manifest import (
    M3DManifest,
    ManifestRecord,
    PromptVariant,
    read_manifest,
)
from m3d.data.schema import DataSplit, TASK_ORDER, TaskDatasetInfo, TaskName


class DatasetCatalogError(RuntimeError):
    """Raised when manifest records cannot form a safe dataset catalogue."""


class PromptSource(str, Enum):
    """Where a concrete question/answer pair is constructed."""

    CAPTION_TEXT_FILE = "caption_text_file"
    MANIFEST = "manifest"
    AFTER_SPATIAL_TRANSFORM = "after_spatial_transform"


class ModelExecutionPath(str, Enum):
    """Conditional forward graph used by a task-homogeneous batch."""

    MAIN_VISION_AND_LANGUAGE = "main_vision_and_language"
    MAIN_VISION_LANGUAGE_AND_SEGMENTATION = (
        "main_vision_language_and_segmentation"
    )


@dataclass(frozen=True, slots=True)
class VariantSpec:
    """Capabilities and construction rules for one prompt variant."""

    variant: PromptVariant
    task: TaskName
    prompt_source: PromptSource
    reads_mask: bool
    returns_segmentation_target: bool
    uses_box_tokens: bool
    requires_segmentation_token: bool

    def __post_init__(self) -> None:
        if self.returns_segmentation_target != self.task.requires_segmentation_target:
            raise DatasetCatalogError(
                f"Variant {self.variant.value!r} has an invalid dense-target contract"
            )
        if self.uses_box_tokens != self.task.uses_box_tokens:
            raise DatasetCatalogError(
                f"Variant {self.variant.value!r} has an invalid box-token contract"
            )
        if self.task is TaskName.POSITIONING and not self.reads_mask:
            raise DatasetCatalogError(
                f"Positioning variant {self.variant.value!r} must read a mask"
            )
        if self.task is TaskName.SEGMENTATION and not self.reads_mask:
            raise DatasetCatalogError(
                f"Segmentation variant {self.variant.value!r} must read a mask"
            )
        if self.requires_segmentation_token and self.task is not TaskName.SEGMENTATION:
            raise DatasetCatalogError(
                "Only the segmentation task may require the [SEG] token"
            )


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Static model/data behaviour shared by all variants of one task."""

    task: TaskName
    dataset_name: str
    execution_path: ModelExecutionPath
    allowed_variants: tuple[PromptVariant, ...]
    reads_mask: bool
    returns_segmentation_target: bool
    uses_box_tokens: bool

    def __post_init__(self) -> None:
        if not self.dataset_name.strip():
            raise DatasetCatalogError("dataset_name cannot be empty")
        if not self.allowed_variants:
            raise DatasetCatalogError(
                f"Task {self.task.value!r} must allow at least one prompt variant"
            )
        if len(set(self.allowed_variants)) != len(self.allowed_variants):
            raise DatasetCatalogError(
                f"Task {self.task.value!r} contains duplicate prompt variants"
            )
        if self.returns_segmentation_target != self.task.requires_segmentation_target:
            raise DatasetCatalogError(
                f"Task {self.task.value!r} has an invalid dense-target contract"
            )
        if self.uses_box_tokens != self.task.uses_box_tokens:
            raise DatasetCatalogError(
                f"Task {self.task.value!r} has an invalid box-token contract"
            )
        expected_path = (
            ModelExecutionPath.MAIN_VISION_LANGUAGE_AND_SEGMENTATION
            if self.task is TaskName.SEGMENTATION
            else ModelExecutionPath.MAIN_VISION_AND_LANGUAGE
        )
        if self.execution_path is not expected_path:
            raise DatasetCatalogError(
                f"Task {self.task.value!r} must use {expected_path.value!r}"
            )

        for variant in self.allowed_variants:
            variant_spec = VARIANT_SPECS.get(variant)
            if variant_spec is None:
                raise DatasetCatalogError(
                    f"No VariantSpec is registered for {variant.value!r}"
                )
            if variant_spec.task is not self.task:
                raise DatasetCatalogError(
                    f"Variant {variant.value!r} belongs to "
                    f"{variant_spec.task.value!r}, not {self.task.value!r}"
                )


VARIANT_SPECS: Mapping[PromptVariant, VariantSpec] = MappingProxyType(
    {
        PromptVariant.CAPTION: VariantSpec(
            variant=PromptVariant.CAPTION,
            task=TaskName.CAPTION,
            prompt_source=PromptSource.CAPTION_TEXT_FILE,
            reads_mask=False,
            returns_segmentation_target=False,
            uses_box_tokens=False,
            requires_segmentation_token=False,
        ),
        PromptVariant.VQA_CLOSED: VariantSpec(
            variant=PromptVariant.VQA_CLOSED,
            task=TaskName.VQA_CLOSED,
            prompt_source=PromptSource.MANIFEST,
            reads_mask=False,
            returns_segmentation_target=False,
            uses_box_tokens=False,
            requires_segmentation_token=False,
        ),
        PromptVariant.VQA_OPEN: VariantSpec(
            variant=PromptVariant.VQA_OPEN,
            task=TaskName.VQA_OPEN,
            prompt_source=PromptSource.MANIFEST,
            reads_mask=False,
            returns_segmentation_target=False,
            uses_box_tokens=False,
            requires_segmentation_token=False,
        ),
        PromptVariant.VQA_YES_NO: VariantSpec(
            variant=PromptVariant.VQA_YES_NO,
            task=TaskName.VQA_YES_NO,
            prompt_source=PromptSource.MANIFEST,
            reads_mask=False,
            returns_segmentation_target=False,
            uses_box_tokens=False,
            requires_segmentation_token=False,
        ),
        PromptVariant.REC_CLASS: VariantSpec(
            variant=PromptVariant.REC_CLASS,
            task=TaskName.POSITIONING,
            prompt_source=PromptSource.AFTER_SPATIAL_TRANSFORM,
            reads_mask=True,
            returns_segmentation_target=False,
            uses_box_tokens=True,
            requires_segmentation_token=False,
        ),
        PromptVariant.REC_DESCRIPTION: VariantSpec(
            variant=PromptVariant.REC_DESCRIPTION,
            task=TaskName.POSITIONING,
            prompt_source=PromptSource.AFTER_SPATIAL_TRANSFORM,
            reads_mask=True,
            returns_segmentation_target=False,
            uses_box_tokens=True,
            requires_segmentation_token=False,
        ),
        PromptVariant.REG_CLASS: VariantSpec(
            variant=PromptVariant.REG_CLASS,
            task=TaskName.POSITIONING,
            prompt_source=PromptSource.AFTER_SPATIAL_TRANSFORM,
            reads_mask=True,
            returns_segmentation_target=False,
            uses_box_tokens=True,
            requires_segmentation_token=False,
        ),
        PromptVariant.REG_DESCRIPTION: VariantSpec(
            variant=PromptVariant.REG_DESCRIPTION,
            task=TaskName.POSITIONING,
            prompt_source=PromptSource.AFTER_SPATIAL_TRANSFORM,
            reads_mask=True,
            returns_segmentation_target=False,
            uses_box_tokens=True,
            requires_segmentation_token=False,
        ),
        PromptVariant.SEG_CLASS: VariantSpec(
            variant=PromptVariant.SEG_CLASS,
            task=TaskName.SEGMENTATION,
            prompt_source=PromptSource.AFTER_SPATIAL_TRANSFORM,
            reads_mask=True,
            returns_segmentation_target=True,
            uses_box_tokens=False,
            requires_segmentation_token=True,
        ),
        PromptVariant.SEG_DESCRIPTION: VariantSpec(
            variant=PromptVariant.SEG_DESCRIPTION,
            task=TaskName.SEGMENTATION,
            prompt_source=PromptSource.AFTER_SPATIAL_TRANSFORM,
            reads_mask=True,
            returns_segmentation_target=True,
            uses_box_tokens=False,
            requires_segmentation_token=True,
        ),
        PromptVariant.REFERRING_SEGMENTATION: VariantSpec(
            variant=PromptVariant.REFERRING_SEGMENTATION,
            task=TaskName.SEGMENTATION,
            prompt_source=PromptSource.MANIFEST,
            reads_mask=True,
            returns_segmentation_target=True,
            uses_box_tokens=False,
            requires_segmentation_token=True,
        ),
    }
)


TASK_SPECS: Mapping[TaskName, TaskSpec] = MappingProxyType(
    {
        TaskName.CAPTION: TaskSpec(
            task=TaskName.CAPTION,
            dataset_name="m3d_caption",
            execution_path=ModelExecutionPath.MAIN_VISION_AND_LANGUAGE,
            allowed_variants=(PromptVariant.CAPTION,),
            reads_mask=False,
            returns_segmentation_target=False,
            uses_box_tokens=False,
        ),
        TaskName.VQA_CLOSED: TaskSpec(
            task=TaskName.VQA_CLOSED,
            dataset_name="m3d_vqa_closed",
            execution_path=ModelExecutionPath.MAIN_VISION_AND_LANGUAGE,
            allowed_variants=(PromptVariant.VQA_CLOSED,),
            reads_mask=False,
            returns_segmentation_target=False,
            uses_box_tokens=False,
        ),
        TaskName.VQA_OPEN: TaskSpec(
            task=TaskName.VQA_OPEN,
            dataset_name="m3d_vqa_open",
            execution_path=ModelExecutionPath.MAIN_VISION_AND_LANGUAGE,
            allowed_variants=(PromptVariant.VQA_OPEN,),
            reads_mask=False,
            returns_segmentation_target=False,
            uses_box_tokens=False,
        ),
        TaskName.VQA_YES_NO: TaskSpec(
            task=TaskName.VQA_YES_NO,
            dataset_name="m3d_vqa_yes_no",
            execution_path=ModelExecutionPath.MAIN_VISION_AND_LANGUAGE,
            allowed_variants=(PromptVariant.VQA_YES_NO,),
            reads_mask=False,
            returns_segmentation_target=False,
            uses_box_tokens=False,
        ),
        TaskName.POSITIONING: TaskSpec(
            task=TaskName.POSITIONING,
            dataset_name="m3d_positioning",
            execution_path=ModelExecutionPath.MAIN_VISION_AND_LANGUAGE,
            allowed_variants=(
                PromptVariant.REC_CLASS,
                PromptVariant.REC_DESCRIPTION,
                PromptVariant.REG_CLASS,
                PromptVariant.REG_DESCRIPTION,
            ),
            reads_mask=True,
            returns_segmentation_target=False,
            uses_box_tokens=True,
        ),
        TaskName.SEGMENTATION: TaskSpec(
            task=TaskName.SEGMENTATION,
            dataset_name="m3d_segmentation",
            execution_path=ModelExecutionPath.MAIN_VISION_LANGUAGE_AND_SEGMENTATION,
            allowed_variants=(
                PromptVariant.SEG_CLASS,
                PromptVariant.SEG_DESCRIPTION,
                PromptVariant.REFERRING_SEGMENTATION,
            ),
            reads_mask=True,
            returns_segmentation_target=True,
            uses_box_tokens=False,
        ),
    }
)


@dataclass(frozen=True, slots=True)
class TaskRecordGroup:
    """Immutable manifest records for one task and one split."""

    spec: TaskSpec
    split: DataSplit
    records: tuple[ManifestRecord, ...]

    def __post_init__(self) -> None:
        split = DataSplit.parse(self.split)
        records = tuple(self.records)
        if not records:
            raise DatasetCatalogError(
                f"Task group {self.spec.task.value!r}/{split.value!r} is empty"
            )

        seen_ids: set[str] = set()
        for index, record in enumerate(records):
            if record.task is not self.spec.task:
                raise DatasetCatalogError(
                    f"Record {record.record_id!r} at group index {index} belongs "
                    f"to {record.task.value!r}, expected {self.spec.task.value!r}"
                )
            if record.split is not split:
                raise DatasetCatalogError(
                    f"Record {record.record_id!r} belongs to "
                    f"{record.split.value!r}, expected {split.value!r}"
                )
            if record.prompt_variant not in self.spec.allowed_variants:
                allowed = ", ".join(item.value for item in self.spec.allowed_variants)
                raise DatasetCatalogError(
                    f"Record {record.record_id!r} uses variant "
                    f"{record.prompt_variant.value!r}; allowed for "
                    f"{self.spec.task.value!r}: {allowed}"
                )
            _validate_record_against_variant(record)
            if record.record_id in seen_ids:
                raise DatasetCatalogError(
                    f"Duplicate record ID {record.record_id!r} in task group"
                )
            seen_ids.add(record.record_id)

        object.__setattr__(self, "split", split)
        object.__setattr__(self, "records", records)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> ManifestRecord:
        return self.records[index]

    @property
    def task(self) -> TaskName:
        return self.spec.task

    @property
    def dataset_name(self) -> str:
        return f"{self.spec.dataset_name}_{self.split.value}"

    @property
    def variant_counts(self) -> dict[str, int]:
        counts = Counter(record.prompt_variant.value for record in self.records)
        return dict(sorted(counts.items()))

    @property
    def source_counts(self) -> dict[str, int]:
        counts = Counter(record.source_name for record in self.records)
        return dict(sorted(counts.items()))

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.task.value.encode("utf-8"))
        digest.update(b"\0")
        digest.update(self.split.value.encode("utf-8"))
        digest.update(b"\0")
        for record in self.records:
            digest.update(record.record_id.encode("ascii"))
            digest.update(b"\n")
        return digest.hexdigest()

    def records_for_variant(
        self,
        variant: str | PromptVariant,
    ) -> tuple[ManifestRecord, ...]:
        canonical = PromptVariant.parse(variant)
        if canonical not in self.spec.allowed_variants:
            raise DatasetCatalogError(
                f"Variant {canonical.value!r} does not belong to task "
                f"{self.task.value!r}"
            )
        return tuple(
            record for record in self.records if record.prompt_variant is canonical
        )

    def dataset_info(self) -> TaskDatasetInfo:
        return TaskDatasetInfo(
            task=self.task,
            dataset_name=self.dataset_name,
            size=len(self),
        )

    def summary(self) -> dict[str, Any]:
        return {
            "task": self.task.value,
            "task_id": int(self.task.task_id),
            "split": self.split.value,
            "dataset_name": self.dataset_name,
            "record_count": len(self),
            "variant_counts": self.variant_counts,
            "source_counts": self.source_counts,
            "reads_mask": self.spec.reads_mask,
            "returns_segmentation_target": self.spec.returns_segmentation_target,
            "uses_box_tokens": self.spec.uses_box_tokens,
            "execution_path": self.spec.execution_path.value,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class DatasetCatalog:
    """All available homogeneous task groups for one manifest split."""

    manifest_path: Path
    manifest_fingerprint: str
    split: DataSplit
    groups: Mapping[TaskName, TaskRecordGroup]

    def __post_init__(self) -> None:
        path = Path(self.manifest_path).expanduser().resolve()
        split = DataSplit.parse(self.split)
        groups = dict(self.groups)
        if not groups:
            raise DatasetCatalogError(
                f"Dataset catalogue for {split.value!r} cannot be empty"
            )

        ordered: dict[TaskName, TaskRecordGroup] = {}
        for task in TASK_ORDER:
            group = groups.get(task)
            if group is None:
                continue
            if group.task is not task or group.split is not split:
                raise DatasetCatalogError(
                    f"Invalid group registered under task {task.value!r}"
                )
            ordered[task] = group

        unexpected = set(groups) - set(ordered)
        if unexpected:
            names = ", ".join(sorted(str(item) for item in unexpected))
            raise DatasetCatalogError(f"Unexpected task keys in catalogue: {names}")

        object.__setattr__(self, "manifest_path", path)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "groups", MappingProxyType(ordered))

    @classmethod
    def from_manifest(
        cls,
        manifest: M3DManifest,
        *,
        manifest_path: str | Path,
    ) -> "DatasetCatalog":
        grouped: dict[TaskName, list[ManifestRecord]] = defaultdict(list)
        for record in manifest.records:
            grouped[record.task].append(record)

        groups = {
            task: TaskRecordGroup(
                spec=get_task_spec(task),
                split=manifest.split,
                records=tuple(grouped[task]),
            )
            for task in TASK_ORDER
            if grouped.get(task)
        }
        return cls(
            manifest_path=Path(manifest_path),
            manifest_fingerprint=manifest.fingerprint,
            split=manifest.split,
            groups=groups,
        )

    def __contains__(self, task: object) -> bool:
        try:
            canonical = TaskName.parse(task)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        return canonical in self.groups

    def __len__(self) -> int:
        return sum(len(group) for group in self.groups.values())

    @property
    def available_tasks(self) -> tuple[TaskName, ...]:
        return tuple(self.groups)

    def group(self, task: str | TaskName) -> TaskRecordGroup:
        canonical = TaskName.parse(task)
        try:
            return self.groups[canonical]
        except KeyError as exc:
            available = ", ".join(item.value for item in self.available_tasks)
            raise DatasetCatalogError(
                f"Task {canonical.value!r} is absent from {self.split.value!r} "
                f"manifest. Available: {available}"
            ) from exc

    def task_infos(self) -> tuple[TaskDatasetInfo, ...]:
        return tuple(self.groups[task].dataset_info() for task in self.available_tasks)

    def active_training_groups(
        self,
        task_weights: Mapping[str, float],
    ) -> tuple[TaskRecordGroup, ...]:
        if self.split is not DataSplit.TRAIN:
            raise DatasetCatalogError(
                "Task-sampling weights may only select groups from the train split"
            )
        weights = canonical_task_weights(task_weights)
        missing = [
            task.value
            for task, weight in weights.items()
            if weight > 0.0 and task not in self.groups
        ]
        if missing:
            raise DatasetCatalogError(
                "Positive task weights reference tasks absent from train manifest: "
                + ", ".join(missing)
            )
        groups = tuple(
            self.groups[task]
            for task in TASK_ORDER
            if weights.get(task, 0.0) > 0.0 and task in self.groups
        )
        if not groups:
            raise DatasetCatalogError(
                "No train task remains after applying task weights"
            )
        return groups

    def summary(self) -> dict[str, Any]:
        return {
            "manifest_path": str(self.manifest_path),
            "manifest_fingerprint": self.manifest_fingerprint,
            "split": self.split.value,
            "record_count": len(self),
            "available_tasks": [task.value for task in self.available_tasks],
            "groups": {
                task.value: self.groups[task].summary()
                for task in self.available_tasks
            },
        }


def get_variant_spec(variant: str | PromptVariant) -> VariantSpec:
    """Return the immutable specification for a prompt variant."""

    canonical = PromptVariant.parse(variant)
    try:
        return VARIANT_SPECS[canonical]
    except KeyError as exc:  # Defensive: PromptVariant and registry changed apart.
        raise DatasetCatalogError(
            f"No VariantSpec registered for {canonical.value!r}"
        ) from exc


def get_task_spec(task: str | TaskName) -> TaskSpec:
    """Return the immutable specification for a canonical task."""

    canonical = TaskName.parse(task)
    try:
        return TASK_SPECS[canonical]
    except KeyError as exc:  # Defensive: TaskName and registry changed apart.
        raise DatasetCatalogError(
            f"No TaskSpec registered for {canonical.value!r}"
        ) from exc


def canonical_task_weights(
    values: Mapping[str, float],
) -> Mapping[TaskName, float]:
    """Canonicalise YAML task weights without silently merging aliases."""

    result: dict[TaskName, float] = {}
    original_names: dict[TaskName, str] = {}
    for raw_name, raw_weight in values.items():
        task = TaskName.parse(raw_name)
        if task in result:
            raise DatasetCatalogError(
                f"Task weight keys {original_names[task]!r} and {raw_name!r} "
                f"both resolve to {task.value!r}"
            )
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError) as exc:
            raise DatasetCatalogError(
                f"Task weight for {raw_name!r} is not numeric: {raw_weight!r}"
            ) from exc
        if not math.isfinite(weight) or weight < 0.0:
            raise DatasetCatalogError(
                f"Task weight for {raw_name!r} must be finite and non-negative"
            )
        result[task] = weight
        original_names[task] = raw_name

    ordered = {
        task: result.get(task, 0.0)
        for task in TASK_ORDER
    }
    if not any(weight > 0.0 for weight in ordered.values()):
        raise DatasetCatalogError("At least one canonical task weight must be positive")
    return MappingProxyType(ordered)


def default_manifest_directory(config: ExperimentConfig) -> Path:
    """Return ``<checkpoint.output_dir>/manifests`` as an absolute path."""

    return (Path(config.checkpoint.output_dir).expanduser().resolve() / "manifests")


def manifest_path_for_split(
    config: ExperimentConfig,
    split: str | DataSplit,
    *,
    manifest_directory: str | Path | None = None,
) -> Path:
    canonical = DataSplit.parse(split)
    directory = (
        default_manifest_directory(config)
        if manifest_directory is None
        else Path(manifest_directory).expanduser().resolve()
    )
    return directory / f"{canonical.value}.jsonl"


def load_dataset_catalog(
    config: ExperimentConfig,
    split: str | DataSplit,
    *,
    manifest_directory: str | Path | None = None,
    validate_training_weights: bool = True,
) -> DatasetCatalog:
    """Read one manifest, group records by task, and validate configuration."""

    canonical_split = DataSplit.parse(split)
    path = manifest_path_for_split(
        config,
        canonical_split,
        manifest_directory=manifest_directory,
    )
    if not path.is_file():
        raise DatasetCatalogError(
            f"Manifest does not exist: {path}. Build it first with "
            "`python -m m3d.data.manifest --config ...`."
        )

    manifest = read_manifest(path)
    if manifest.split is not canonical_split:
        raise DatasetCatalogError(
            f"Manifest {path} declares split {manifest.split.value!r}, "
            f"expected {canonical_split.value!r}"
        )
    catalog = DatasetCatalog.from_manifest(manifest, manifest_path=path)
    validate_catalog_for_config(
        catalog,
        config,
        validate_training_weights=validate_training_weights,
    )
    return catalog


def load_dataset_catalogs(
    config: ExperimentConfig,
    splits: Iterable[str | DataSplit],
    *,
    manifest_directory: str | Path | None = None,
) -> Mapping[DataSplit, DatasetCatalog]:
    """Load several split catalogues and reject duplicate requested splits."""

    loaded: dict[DataSplit, DatasetCatalog] = {}
    for value in splits:
        split = DataSplit.parse(value)
        if split in loaded:
            raise DatasetCatalogError(
                f"Split {split.value!r} was requested more than once"
            )
        loaded[split] = load_dataset_catalog(
            config,
            split,
            manifest_directory=manifest_directory,
            validate_training_weights=(split is DataSplit.TRAIN),
        )
    return MappingProxyType(loaded)


def validate_catalog_for_config(
    catalog: DatasetCatalog,
    config: ExperimentConfig,
    *,
    validate_training_weights: bool = True,
) -> None:
    """Check model capabilities and task weights against one catalogue."""

    has_segmentation = TaskName.SEGMENTATION in catalog.groups
    if has_segmentation and not config.model.segmentation.enabled:
        raise DatasetCatalogError(
            f"{catalog.split.value} manifest contains segmentation records, but "
            "model.segmentation.enabled=false"
        )
    if has_segmentation and not config.model.seg_vision.enabled:
        raise DatasetCatalogError(
            f"{catalog.split.value} manifest contains segmentation records, but "
            "model.seg_vision.enabled=false"
        )

    if validate_training_weights and catalog.split is DataSplit.TRAIN:
        catalog.active_training_groups(config.data.task_sampling.task_weights)


def _validate_record_against_variant(record: ManifestRecord) -> None:
    spec = get_variant_spec(record.prompt_variant)
    if spec.task is not record.task:
        raise DatasetCatalogError(
            f"Record {record.record_id!r} has task {record.task.value!r}, but "
            f"variant {record.prompt_variant.value!r} belongs to "
            f"{spec.task.value!r}"
        )

    if spec.reads_mask != (record.mask_path is not None):
        raise DatasetCatalogError(
            f"Record {record.record_id!r} mask contract does not match variant "
            f"{record.prompt_variant.value!r}"
        )

    if spec.prompt_source is PromptSource.CAPTION_TEXT_FILE:
        if record.text_path is None or record.question is not None or record.answer is not None:
            raise DatasetCatalogError(
                f"Caption-file record {record.record_id!r} must carry text_path "
                "and defer question/answer construction"
            )
    elif spec.prompt_source is PromptSource.MANIFEST:
        if record.question is None or record.answer is None:
            raise DatasetCatalogError(
                f"Manifest-prompt record {record.record_id!r} requires question and answer"
            )
    elif spec.prompt_source is PromptSource.AFTER_SPATIAL_TRANSFORM:
        if record.question is not None or record.answer is not None:
            raise DatasetCatalogError(
                f"Generated-prompt record {record.record_id!r} must defer question "
                "and answer construction until after spatial augmentation"
            )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and validate typed M3D dataset catalogues."
    )
    parser.add_argument("--config", required=True, help="Experiment YAML path")
    parser.add_argument(
        "--manifest-dir",
        default=None,
        help="Defaults to <checkpoint.output_dir>/manifests",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=[DataSplit.TRAIN.value],
        choices=[item.value for item in DataSplit],
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Configuration override such as data.task_sampling.temperature_alpha=0.5",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    config = load_config(args.config, args.overrides, verify_paths=False)
    catalogs = load_dataset_catalogs(
        config,
        args.splits,
        manifest_directory=args.manifest_dir,
    )
    payload = {
        split.value: catalog.summary()
        for split, catalog in catalogs.items()
    }
    print(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


# Fail at import time if an enum was extended without updating both registries.
if set(VARIANT_SPECS) != set(PromptVariant):
    missing = set(PromptVariant) - set(VARIANT_SPECS)
    extra = set(VARIANT_SPECS) - set(PromptVariant)
    raise DatasetCatalogError(
        "Prompt variant registry mismatch; "
        f"missing={[item.value for item in missing]}, "
        f"extra={[item.value for item in extra]}"
    )
if set(TASK_SPECS) != set(TaskName):
    missing = set(TaskName) - set(TASK_SPECS)
    extra = set(TASK_SPECS) - set(TaskName)
    raise DatasetCatalogError(
        "Task registry mismatch; "
        f"missing={[item.value for item in missing]}, "
        f"extra={[item.value for item in extra]}"
    )


__all__ = [
    "DatasetCatalog",
    "DatasetCatalogError",
    "ModelExecutionPath",
    "PromptSource",
    "TASK_SPECS",
    "TaskRecordGroup",
    "TaskSpec",
    "VARIANT_SPECS",
    "VariantSpec",
    "canonical_task_weights",
    "default_manifest_directory",
    "get_task_spec",
    "get_variant_spec",
    "load_dataset_catalog",
    "load_dataset_catalogs",
    "manifest_path_for_split",
    "validate_catalog_for_config",
]


if __name__ == "__main__":
    raise SystemExit(main())
