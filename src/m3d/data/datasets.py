"""PyTorch datasets for every modernized M3D training task.

This module is the point where a validated :class:`ManifestRecord` becomes a
fully prepared :class:`M3DSample`.  Every task follows the same sequence:

1. read the image through :mod:`m3d.data.io`;
2. read a mask only when the task contract requires it;
3. apply one deterministic, synchronised spatial transform to image and mask;
4. generate the exact M3D question/answer pair for the record variant;
5. tokenize once without fixed 512-token padding;
6. return one explicit task-labelled sample.

Important behavioural guarantees
--------------------------------
* A failed record raises an error containing its stable record ID.  The dataset
  never silently replaces it with a random sample.
* Caption/VQA samples never manufacture a full-size zero mask.
* Positioning samples may read a mask to construct a box, but do not return a
  dense segmentation target and therefore do not run SegVol.
* Segmentation identity is structural.  An all-zero mask is a valid dense
  target and still reaches the Dice/BCE branch.
* Generated negative segmentation prompts receive one trailing ``[SEG]``
  control token.  Original M3D omitted it because all-zero masks were skipped;
  the modernized pipeline retains the natural-language negative answer while
  adding the control token required to train a real empty-mask target.
* Prompt and augmentation choices depend only on seed, epoch, record ID, and
  prompt namespace.  DataLoader scheduling does not change them.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Iterable, Mapping, Protocol, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from m3d.config import ExperimentConfig
from m3d.data.anatomy_catalog import descriptions_for, resolve_class_name
from m3d.data.dataset_catalog import (
    DatasetCatalog,
    TaskRecordGroup,
    get_variant_spec,
)
from m3d.data.io import (
    DataIOError,
    MaskSelection,
    VolumeReader,
    build_volume_reader,
    read_utf8_text,
)
from m3d.data.manifest import ManifestRecord, PromptVariant
from m3d.data.prompt_templates import (
    PromptFamily,
    PromptTemplateError,
    get_prompt_set,
    render_template,
    select_caption_question,
    stable_choice,
)
from m3d.data.schema import (
    DataSplit,
    M3DSample,
    SampleProvenance,
    TaskDatasetInfo,
    TaskName,
)
from m3d.data.transforms import (
    AugmentationContext,
    M3DVolumeTransform,
    build_volume_transform,
    provenance_with_augmentation,
)

if TYPE_CHECKING:
    from m3d.tokenization import EncodedText, M3DTextProcessor


class DatasetConstructionError(RuntimeError):
    """Raised when one manifest record cannot become a valid M3D sample."""


class TextProcessorProtocol(Protocol):
    """Minimal tokenizer interface required inside DataLoader workers."""

    bundle: Any

    def encode_supervised(
        self,
        question: str,
        answer: str,
        *,
        prepend_image_tokens: bool = True,
        answer_separator: str = " ",
        required_answer_token_ids: Iterable[int] = (),
    ) -> Any: ...

    def encode_segmentation(
        self,
        question: str,
        answer: str,
        *,
        prepend_image_tokens: bool = True,
    ) -> Any: ...


@dataclass(frozen=True, slots=True)
class PromptResult:
    """Rendered natural-language pair and diagnostics for one record."""

    question: str
    answer: str
    class_name: str | None = None
    description: str | None = None
    has_foreground: bool | None = None
    normalized_box: tuple[float, float, float, float, float, float] | None = None
    appended_negative_segmentation_token: bool = False

    def __post_init__(self) -> None:
        question = self.question.strip()
        answer = self.answer.strip()
        if not question:
            raise DatasetConstructionError("Rendered question cannot be empty")
        if not answer:
            raise DatasetConstructionError("Rendered answer cannot be empty")
        object.__setattr__(self, "question", question)
        object.__setattr__(self, "answer", answer)

    def metadata(self) -> dict[str, Any]:
        return {
            "rendered_class_name": self.class_name,
            "rendered_description": self.description,
            "mask_has_foreground": self.has_foreground,
            "normalized_box": (
                None if self.normalized_box is None else list(self.normalized_box)
            ),
            "appended_negative_segmentation_token": (
                self.appended_negative_segmentation_token
            ),
        }


class SharedEpoch:
    """A tiny shared-memory epoch counter visible to persistent workers.

    The main process updates this value between epochs.  Workers only read it
    while fetching records, so no mutable per-worker epoch copy can become
    stale when ``persistent_workers=True``.
    """

    def __init__(self, initial_epoch: int = 0) -> None:
        if initial_epoch < 0:
            raise ValueError("initial_epoch cannot be negative")
        self._value = torch.tensor(initial_epoch, dtype=torch.int64).share_memory_()

    @property
    def epoch(self) -> int:
        return int(self._value.item())

    def set(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int):
            raise TypeError("epoch must be an integer")
        if epoch < 0:
            raise ValueError("epoch cannot be negative")
        self._value.fill_(epoch)

    def state_dict(self) -> dict[str, int]:
        return {"epoch": self.epoch}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if "epoch" not in state:
            raise KeyError("SharedEpoch state is missing 'epoch'")
        self.set(int(state["epoch"]))


class M3DTaskDataset(Dataset[M3DSample]):
    """One immutable manifest group exposed as a PyTorch map-style dataset."""

    def __init__(
        self,
        *,
        group: TaskRecordGroup,
        config: ExperimentConfig,
        text_processor: TextProcessorProtocol,
        volume_reader: VolumeReader,
        volume_transform: M3DVolumeTransform,
        epoch_state: SharedEpoch,
        text_cache: Mapping[tuple[str, int], Any] | None = None,
        record_augmentation_metadata: bool = True,
    ) -> None:
        if not isinstance(group, TaskRecordGroup):
            raise TypeError("group must be a TaskRecordGroup")
        if not isinstance(config, ExperimentConfig):
            raise TypeError("config must be an ExperimentConfig")
        if not isinstance(volume_reader, VolumeReader):
            raise TypeError("volume_reader must be a VolumeReader")
        if not isinstance(volume_transform, M3DVolumeTransform):
            raise TypeError("volume_transform must be an M3DVolumeTransform")
        if not isinstance(epoch_state, SharedEpoch):
            raise TypeError("epoch_state must be a SharedEpoch")

        self.group = group
        self.config = config
        self.text_processor = text_processor
        self.volume_reader = volume_reader
        self.volume_transform = volume_transform
        self.epoch_state = epoch_state
        self.text_cache = text_cache
        self.record_augmentation_metadata = bool(record_augmentation_metadata)

        expected_training = group.split is DataSplit.TRAIN
        if self.volume_transform.training != expected_training:
            raise DatasetConstructionError(
                "Transform mode does not match dataset split: "
                f"split={group.split.value}, transform.training="
                f"{self.volume_transform.training}"
            )

    def __len__(self) -> int:
        return len(self.group)

    @property
    def task(self) -> TaskName:
        return self.group.task

    @property
    def split(self) -> DataSplit:
        return self.group.split

    @property
    def dataset_name(self) -> str:
        return self.group.dataset_name

    @property
    def fingerprint(self) -> str:
        return self.group.fingerprint

    def dataset_info(self) -> TaskDatasetInfo:
        return self.group.dataset_info()

    def set_epoch(self, epoch: int) -> None:
        self.epoch_state.set(epoch)

    def __getitem__(self, index: int) -> M3DSample:
        normalized_index = self._normalise_index(index)
        record = self.group[normalized_index]
        try:
            return self._build_sample(record)
        except DatasetConstructionError:
            raise
        except (DataIOError, PromptTemplateError, ValueError, TypeError, OSError) as exc:
            raise DatasetConstructionError(
                self._record_error_message(record, normalized_index, exc)
            ) from exc
        except Exception as exc:  # noqa: BLE001 - enrich unexpected worker failures.
            raise DatasetConstructionError(
                self._record_error_message(record, normalized_index, exc)
            ) from exc

    def _normalise_index(self, index: int) -> int:
        if isinstance(index, bool) or not isinstance(index, int):
            raise TypeError(f"Dataset index must be int, got {type(index).__name__}")
        size = len(self)
        normalized = index + size if index < 0 else index
        if normalized < 0 or normalized >= size:
            raise IndexError(f"Dataset index {index} is outside [0, {size})")
        return normalized

    def _build_sample(self, record: ManifestRecord) -> M3DSample:
        variant_spec = get_variant_spec(record.prompt_variant)
        if variant_spec.task is not self.task:
            raise DatasetConstructionError(
                f"Record variant {record.prompt_variant.value!r} belongs to "
                f"{variant_spec.task.value!r}, dataset task is {self.task.value!r}"
            )

        image_volume = self.volume_reader.load_image(record.image_path)

        mask_volume = None
        raw_mask = None
        if variant_spec.reads_mask:
            if record.mask_path is None:
                raise DatasetConstructionError(
                    f"Variant {record.prompt_variant.value!r} requires mask_path"
                )
            selection = (
                MaskSelection(label_id=record.mask_label_id)
                if record.mask_label_id is not None
                else MaskSelection()
            )
            mask_volume = self.volume_reader.load_mask(
                record.mask_path,
                selection=selection,
            )
            raw_mask = mask_volume.tensor

        epoch = self.epoch_state.epoch
        context = AugmentationContext(
            sample_id=record.record_id,
            epoch=epoch,
            base_seed=self.config.runtime.seed,
        )
        transformed = self.volume_transform(
            image_volume.tensor,
            raw_mask,
            context=context,
        )

        transformed_mask = transformed.segmentation_target
        if variant_spec.reads_mask and transformed_mask is None:
            raise DatasetConstructionError(
                "Synchronized transform unexpectedly dropped the required mask"
            )

        prompt = self._render_prompt(record, transformed_mask, epoch=epoch)
        encoded = self._encode_text(record, prompt, epoch=epoch)

        metadata = dict(record.metadata)
        metadata.update(
            {
                "manifest_record_id": record.record_id,
                "prompt_variant": record.prompt_variant.value,
                "task": record.task.value,
                "epoch": epoch,
                "image_geometry": image_volume.geometry.to_jsonable(),
                "image_resolved_path": str(image_volume.resolved_path),
                "mask_geometry": (
                    None
                    if mask_volume is None
                    else mask_volume.geometry.to_jsonable()
                ),
                "mask_resolved_path": (
                    None if mask_volume is None else str(mask_volume.resolved_path)
                ),
            }
        )
        metadata.update(prompt.metadata())

        provenance = SampleProvenance(
            sample_id=record.record_id,
            source_name=record.source_name,
            source_index=record.source_index,
            split=record.split,
            image_path=image_volume.source_path,
            mask_path=None if mask_volume is None else mask_volume.source_path,
            metadata=MappingProxyType(metadata),
        )
        if self.record_augmentation_metadata:
            provenance = provenance_with_augmentation(
                provenance,
                transformed.plan,
            )

        dense_target = (
            transformed_mask
            if variant_spec.returns_segmentation_target
            else None
        )
        return M3DSample(
            task=record.task,
            provenance=provenance,
            image=transformed.image,
            text=encoded,
            question=prompt.question,
            answer=prompt.answer,
            segmentation_target=dense_target,
        )

    def _render_prompt(
        self,
        record: ManifestRecord,
        mask: Tensor | None,
        *,
        epoch: int,
    ) -> PromptResult:
        variant = record.prompt_variant

        if variant is PromptVariant.CAPTION:
            if record.text_path is None:
                raise DatasetConstructionError("Caption record is missing text_path")
            question = select_caption_question(
                base_seed=self.config.runtime.seed,
                sample_id=record.record_id,
                epoch=epoch,
            )
            answer = read_utf8_text(
                record.text_path,
                file_cache=self.volume_reader.file_cache,
            )
            return PromptResult(question=question, answer=answer)

        if variant in {
            PromptVariant.VQA_CLOSED,
            PromptVariant.VQA_OPEN,
            PromptVariant.VQA_YES_NO,
            PromptVariant.REFERRING_SEGMENTATION,
        }:
            return PromptResult(
                question=_required_manifest_text(record.question, "question", record),
                answer=_required_manifest_text(record.answer, "answer", record),
                has_foreground=(
                    None if mask is None else mask_has_foreground(mask)
                ),
            )

        if mask is None:
            raise DatasetConstructionError(
                f"Generated variant {variant.value!r} requires a transformed mask"
            )

        class_name, description = self._anatomy_terms(record, epoch=epoch)
        has_foreground = mask_has_foreground(mask)
        box = mask_to_normalized_box(mask) if has_foreground else None

        if variant is PromptVariant.REC_CLASS:
            return self._render_rec(
                record,
                epoch=epoch,
                class_name=class_name,
                description=None,
                box=box,
                has_foreground=has_foreground,
            )
        if variant is PromptVariant.REC_DESCRIPTION:
            return self._render_rec(
                record,
                epoch=epoch,
                class_name=class_name,
                description=description,
                box=box,
                has_foreground=has_foreground,
            )
        if variant is PromptVariant.REG_CLASS:
            return self._render_reg(
                record,
                epoch=epoch,
                class_name=class_name,
                description=None,
                box=box,
                has_foreground=has_foreground,
            )
        if variant is PromptVariant.REG_DESCRIPTION:
            return self._render_reg(
                record,
                epoch=epoch,
                class_name=class_name,
                description=description,
                box=box,
                has_foreground=has_foreground,
            )
        if variant is PromptVariant.SEG_CLASS:
            return self._render_segmentation(
                record,
                epoch=epoch,
                class_name=class_name,
                description=None,
                has_foreground=has_foreground,
            )
        if variant is PromptVariant.SEG_DESCRIPTION:
            return self._render_segmentation(
                record,
                epoch=epoch,
                class_name=class_name,
                description=description,
                has_foreground=has_foreground,
            )

        raise DatasetConstructionError(
            f"No dataset prompt renderer exists for {variant.value!r}"
        )

    def _anatomy_terms(
        self,
        record: ManifestRecord,
        *,
        epoch: int,
    ) -> tuple[str, str]:
        dataset_tag = record.metadata.get("dataset_tag")
        class_id = record.metadata.get("class_id")
        if dataset_tag is None or class_id is None:
            raise DatasetConstructionError(
                f"Record {record.record_id!r} is missing dataset_tag/class_id metadata"
            )
        if isinstance(class_id, bool):
            raise DatasetConstructionError("class_id cannot be boolean")
        try:
            numeric_class_id = int(class_id)
        except (TypeError, ValueError) as exc:
            raise DatasetConstructionError(
                f"Invalid class_id {class_id!r} for record {record.record_id!r}"
            ) from exc
        if isinstance(class_id, float) and not math.isclose(
            class_id,
            numeric_class_id,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise DatasetConstructionError(
                f"class_id must be integral, got {class_id!r}"
            )

        class_name = resolve_class_name(str(dataset_tag), numeric_class_id)
        description = stable_choice(
            descriptions_for(class_name),
            base_seed=self.config.runtime.seed,
            sample_id=record.record_id,
            epoch=epoch,
            namespace=f"{record.prompt_variant.value}:anatomy-description",
        )
        return class_name, description

    def _render_rec(
        self,
        record: ManifestRecord,
        *,
        epoch: int,
        class_name: str,
        description: str | None,
        box: tuple[float, float, float, float, float, float] | None,
        has_foreground: bool,
    ) -> PromptResult:
        prompts = get_prompt_set(PromptFamily.POSITION_REC)
        is_description = description is not None
        subject = description if is_description else class_name

        question_templates = (
            prompts.des_questions if is_description else prompts.cls_questions
        )
        question = render_template(
            stable_choice(
                question_templates,
                base_seed=self.config.runtime.seed,
                sample_id=record.record_id,
                epoch=epoch,
                namespace=f"{record.prompt_variant.value}:question",
            ),
            subject,
        )

        if has_foreground:
            if box is None:
                raise DatasetConstructionError(
                    "Foreground REC sample is missing its normalized box"
                )
            box_text = self._box_text(box)
            answer_templates = (
                prompts.des_answers if is_description else prompts.cls_answers
            )
            answer_template = stable_choice(
                answer_templates,
                base_seed=self.config.runtime.seed,
                sample_id=record.record_id,
                epoch=epoch,
                namespace=f"{record.prompt_variant.value}:positive-answer",
            )
            answer = (
                render_template(answer_template, class_name, box_text)
                if is_description
                else render_template(answer_template, box_text)
            )
        else:
            answer_templates = (
                prompts.des_no_answers
                if is_description
                else prompts.cls_no_answers
            )
            answer = render_template(
                stable_choice(
                    answer_templates,
                    base_seed=self.config.runtime.seed,
                    sample_id=record.record_id,
                    epoch=epoch,
                    namespace=f"{record.prompt_variant.value}:negative-answer",
                ),
                class_name,
            )

        return PromptResult(
            question=question,
            answer=answer,
            class_name=class_name,
            description=description,
            has_foreground=has_foreground,
            normalized_box=box,
        )

    def _render_reg(
        self,
        record: ManifestRecord,
        *,
        epoch: int,
        class_name: str,
        description: str | None,
        box: tuple[float, float, float, float, float, float] | None,
        has_foreground: bool,
    ) -> PromptResult:
        reg = get_prompt_set(PromptFamily.POSITION_REG)
        rec = get_prompt_set(PromptFamily.POSITION_REC)
        is_description = description is not None

        if has_foreground:
            if box is None:
                raise DatasetConstructionError(
                    "Foreground REG sample is missing its normalized box"
                )
            box_text = self._box_text(box)
            question_templates = (
                reg.des_questions if is_description else reg.cls_questions
            )
            question = render_template(
                stable_choice(
                    question_templates,
                    base_seed=self.config.runtime.seed,
                    sample_id=record.record_id,
                    epoch=epoch,
                    namespace=f"{record.prompt_variant.value}:positive-question",
                ),
                box_text,
            )
            answer_templates = (
                reg.des_answers if is_description else reg.cls_answers
            )
            answer_template = stable_choice(
                answer_templates,
                base_seed=self.config.runtime.seed,
                sample_id=record.record_id,
                epoch=epoch,
                namespace=f"{record.prompt_variant.value}:positive-answer",
            )
            answer = (
                render_template(answer_template, class_name, description)
                if is_description
                else render_template(answer_template, class_name)
            )
        else:
            # Original M3D deliberately falls back to REC-style questions when
            # a REG target has no box to put in the question.
            question_templates = (
                rec.des_questions if is_description else rec.cls_questions
            )
            question_subject = description if is_description else class_name
            question = render_template(
                stable_choice(
                    question_templates,
                    base_seed=self.config.runtime.seed,
                    sample_id=record.record_id,
                    epoch=epoch,
                    namespace=f"{record.prompt_variant.value}:negative-question",
                ),
                question_subject,
            )
            answer_templates = (
                reg.des_no_answers if is_description else reg.cls_no_answers
            )
            answer = render_template(
                stable_choice(
                    answer_templates,
                    base_seed=self.config.runtime.seed,
                    sample_id=record.record_id,
                    epoch=epoch,
                    namespace=f"{record.prompt_variant.value}:negative-answer",
                ),
                class_name,
            )

        return PromptResult(
            question=question,
            answer=answer,
            class_name=class_name,
            description=description,
            has_foreground=has_foreground,
            normalized_box=box,
        )

    def _render_segmentation(
        self,
        record: ManifestRecord,
        *,
        epoch: int,
        class_name: str,
        description: str | None,
        has_foreground: bool,
    ) -> PromptResult:
        prompts = get_prompt_set(PromptFamily.SEGMENTATION)
        is_description = description is not None
        subject = description if is_description else class_name

        question_templates = (
            prompts.des_questions if is_description else prompts.cls_questions
        )
        question = render_template(
            stable_choice(
                question_templates,
                base_seed=self.config.runtime.seed,
                sample_id=record.record_id,
                epoch=epoch,
                namespace=f"{record.prompt_variant.value}:question",
            ),
            subject,
        )

        appended_control = False
        if has_foreground:
            answer_templates = (
                prompts.des_answers if is_description else prompts.cls_answers
            )
            answer_template = stable_choice(
                answer_templates,
                base_seed=self.config.runtime.seed,
                sample_id=record.record_id,
                epoch=epoch,
                namespace=f"{record.prompt_variant.value}:positive-answer",
            )
            answer = (
                render_template(answer_template, class_name)
                if is_description
                else render_template(answer_template)
            )
        else:
            answer_templates = (
                prompts.des_no_answers
                if is_description
                else prompts.cls_no_answers
            )
            answer = render_template(
                stable_choice(
                    answer_templates,
                    base_seed=self.config.runtime.seed,
                    sample_id=record.record_id,
                    epoch=epoch,
                    namespace=f"{record.prompt_variant.value}:negative-answer",
                ),
                class_name,
            )
            answer, appended_control = self._append_segmentation_control(answer)

        return PromptResult(
            question=question,
            answer=answer,
            class_name=class_name,
            description=description,
            has_foreground=has_foreground,
            appended_negative_segmentation_token=appended_control,
        )

    def _append_segmentation_control(self, answer: str) -> tuple[str, bool]:
        token = str(self.text_processor.bundle.metadata.segmentation_token)
        if token in answer:
            return answer, False
        return f"{answer.rstrip()} {token}", True

    def _box_text(
        self,
        box: tuple[float, float, float, float, float, float],
    ) -> str:
        # Convert to list before str() to reproduce original M3D formatting:
        # "[0.1, 0.2, ...]" including spaces after commas.
        return self.text_processor.bundle.box_text(list(box)) if hasattr(
            self.text_processor.bundle, "box_text"
        ) else (
            f"{self.text_processor.bundle.metadata.box_start_token}"
            f"{list(box)}"
            f"{self.text_processor.bundle.metadata.box_end_token}"
        )

    def _encode_text(
        self,
        record: ManifestRecord,
        prompt: PromptResult,
        *,
        epoch: int,
    ) -> Any:
        cache_key = (record.record_id, epoch)
        if self.text_cache is not None:
            cached = self.text_cache.get(cache_key)
            if cached is not None:
                return cached

        if record.task is TaskName.SEGMENTATION:
            return self.text_processor.encode_segmentation(
                prompt.question,
                prompt.answer,
                prepend_image_tokens=True,
            )
        return self.text_processor.encode_supervised(
            prompt.question,
            prompt.answer,
            prepend_image_tokens=True,
        )

    @staticmethod
    def _record_error_message(
        record: ManifestRecord,
        index: int,
        exc: BaseException,
    ) -> str:
        return (
            "Failed to construct M3D sample: "
            f"task={record.task.value!r}, variant={record.prompt_variant.value!r}, "
            f"record_id={record.record_id!r}, group_index={index}, "
            f"source={record.source_name!r}, source_index={record.source_index}, "
            f"image={record.image_path!r}, mask={record.mask_path!r}. "
            f"Cause: {type(exc).__name__}: {exc}"
        )


@dataclass(frozen=True, slots=True)
class M3DDatasetCollection:
    """Task-indexed datasets that share one reader and one epoch counter."""

    datasets: Mapping[TaskName, M3DTaskDataset]
    epoch_state: SharedEpoch
    split: DataSplit
    manifest_fingerprint: str

    def __post_init__(self) -> None:
        parsed_split = DataSplit.parse(self.split)
        datasets = dict(self.datasets)
        if not datasets:
            raise DatasetConstructionError("Dataset collection cannot be empty")
        for task, dataset in datasets.items():
            canonical = TaskName.parse(task)
            if canonical is not dataset.task:
                raise DatasetConstructionError(
                    f"Dataset mapping key {canonical.value!r} does not match "
                    f"dataset task {dataset.task.value!r}"
                )
            if dataset.split is not parsed_split:
                raise DatasetConstructionError(
                    f"Dataset {dataset.dataset_name!r} belongs to "
                    f"{dataset.split.value!r}, expected {parsed_split.value!r}"
                )
            if dataset.epoch_state is not self.epoch_state:
                raise DatasetConstructionError(
                    "All task datasets must share the same SharedEpoch instance"
                )
        object.__setattr__(self, "datasets", MappingProxyType(datasets))
        object.__setattr__(self, "split", parsed_split)

    def __contains__(self, task: object) -> bool:
        try:
            canonical = TaskName.parse(task)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False
        return canonical in self.datasets

    def __getitem__(self, task: str | TaskName) -> M3DTaskDataset:
        canonical = TaskName.parse(task)
        try:
            return self.datasets[canonical]
        except KeyError as exc:
            available = ", ".join(item.value for item in self.datasets)
            raise KeyError(
                f"Task {canonical.value!r} is absent; available: {available}"
            ) from exc

    @property
    def tasks(self) -> tuple[TaskName, ...]:
        return tuple(sorted(self.datasets, key=lambda task: int(task.task_id)))

    def set_epoch(self, epoch: int) -> None:
        self.epoch_state.set(epoch)

    def task_infos(self) -> tuple[TaskDatasetInfo, ...]:
        return tuple(self.datasets[task].dataset_info() for task in self.tasks)

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch_state.epoch,
            "split": self.split.value,
            "manifest_fingerprint": self.manifest_fingerprint,
            "dataset_fingerprints": {
                task.value: self.datasets[task].fingerprint for task in self.tasks
            },
        }


def build_task_datasets(
    catalog: DatasetCatalog,
    config: ExperimentConfig,
    text_processor: TextProcessorProtocol,
    *,
    volume_reader: VolumeReader | None = None,
    epoch_state: SharedEpoch | None = None,
    text_cache: Mapping[tuple[str, int], Any] | None = None,
    record_augmentation_metadata: bool = True,
) -> M3DDatasetCollection:
    """Construct one map-style dataset per available task in ``catalog``."""

    if not isinstance(catalog, DatasetCatalog):
        raise TypeError("catalog must be a DatasetCatalog")
    config.validate()

    reader = volume_reader or build_volume_reader(config)
    shared_epoch = epoch_state or SharedEpoch(0)
    transform = build_volume_transform(config, catalog.split)

    datasets = {
        task: M3DTaskDataset(
            group=catalog.group(task),
            config=config,
            text_processor=text_processor,
            volume_reader=reader,
            volume_transform=transform,
            epoch_state=shared_epoch,
            text_cache=text_cache,
            record_augmentation_metadata=record_augmentation_metadata,
        )
        for task in catalog.available_tasks
    }
    return M3DDatasetCollection(
        datasets=datasets,
        epoch_state=shared_epoch,
        split=catalog.split,
        manifest_fingerprint=catalog.manifest_fingerprint,
    )


def mask_has_foreground(mask: Tensor) -> bool:
    """Return whether a CPU binary mask contains at least one foreground voxel."""

    _validate_mask_for_prompt(mask)
    return bool(torch.any(mask != 0).item())


def mask_to_normalized_box(
    mask: Tensor,
) -> tuple[float, float, float, float, float, float] | None:
    """Reproduce original M3D ``mask2box`` coordinates after augmentation.

    Input may be ``[1,D,H,W]`` or ``[D,H,W]``.  For a foreground mask, minima
    and maxima are divided by ``D``, ``H``, and ``W`` respectively and rounded
    to three decimal places, exactly matching the public implementation.  An
    empty mask returns ``None`` instead of asking ``torch.min`` to reduce an
    empty tensor.
    """

    _validate_mask_for_prompt(mask)
    spatial = mask[0] if mask.ndim == 4 else mask
    indices = torch.nonzero(spatial, as_tuple=False)
    if indices.numel() == 0:
        return None

    minima = indices.amin(dim=0)
    maxima = indices.amax(dim=0)
    sizes = spatial.shape
    values = (
        round(int(minima[0].item()) / sizes[0], 3),
        round(int(minima[1].item()) / sizes[1], 3),
        round(int(minima[2].item()) / sizes[2], 3),
        round(int(maxima[0].item()) / sizes[0], 3),
        round(int(maxima[1].item()) / sizes[1], 3),
        round(int(maxima[2].item()) / sizes[2], 3),
    )
    return values


def _validate_mask_for_prompt(mask: Tensor) -> None:
    if not isinstance(mask, Tensor):
        raise TypeError("mask must be a torch.Tensor")
    if mask.device.type != "cpu":
        raise DatasetConstructionError("Dataset prompt masks must remain on CPU")
    if mask.ndim == 4:
        if mask.shape[0] != 1:
            raise DatasetConstructionError(
                f"Prompt mask must have one channel, got {tuple(mask.shape)}"
            )
    elif mask.ndim != 3:
        raise DatasetConstructionError(
            f"Prompt mask must be [1,D,H,W] or [D,H,W], got {tuple(mask.shape)}"
        )
    if not bool(torch.isfinite(mask).all()):
        raise DatasetConstructionError("Prompt mask contains NaN or Inf")
    if not bool(torch.logical_or(mask == 0, mask == 1).all()):
        raise DatasetConstructionError("Prompt mask must be binary-valued")


def _required_manifest_text(
    value: str | None,
    field_name: str,
    record: ManifestRecord,
) -> str:
    if value is None or not value.strip():
        raise DatasetConstructionError(
            f"Record {record.record_id!r} has no non-empty {field_name}"
        )
    return value.strip()


__all__ = [
    "DatasetConstructionError",
    "M3DDatasetCollection",
    "M3DTaskDataset",
    "PromptResult",
    "SharedEpoch",
    "build_task_datasets",
    "mask_has_foreground",
    "mask_to_normalized_box",
]
