"""Deterministic manifest construction for every M3D training task.

The original repository lets each ``Dataset`` parse JSON/CSV files, open paths,
and silently retry a different random index when a row is malformed.  That
makes the effective training set hard to inspect and impossible to reproduce
exactly.  This module moves source parsing into a deterministic, auditable
pre-training step.

A manifest record describes one *logical* training example.  Multiple logical
records may intentionally reference the same physical CT volume, for example:

* one VQA source row becomes an open-ended and a closed-ended example;
* one Decathlon segmentation pair becomes REC, REG, and segmentation prompt
  variants;
* a referring-segmentation row keeps its supplied question and answer.

The manifest stores paths relative to ``data_root`` so it remains portable
between the shared filesystem and PBS node-local staging.  It never decides
whether a row is a segmentation example by inspecting mask values.  Task
identity is explicit in ``record.task``.

The file format is JSON Lines.  Line one is a small header and every remaining
line is one record.  Writes are atomic and accompanied by a summary JSON file.

完整调用流程为：
python -m m3d.data.manifest
        │
        ▼
加载 m3d package
        │
        ▼
加载 m3d.data package
        │
        ▼
从第一行开始执行 manifest.py
        │
        ├── 读取模块说明字符串
        ├── import 各种依赖
        ├── 创建常量
        ├── 定义 ManifestError
        ├── 定义 PromptVariant
        ├── 定义 ManifestRecord
        ├── 定义 M3DManifest
        ├── 定义 ManifestBuildOptions
        ├── 定义 ManifestBuilder
        ├── 定义所有辅助函数
        └── 这些函数此时大部分还没有运行
        │
        ▼
到达文件最底部
if __name__ == "__main__":
        │
        ▼
main()
        │
        ├── 解析命令行参数
        ├── 读取 YAML config
        ├── 创建 ManifestBuildOptions
        ├── 创建 ManifestBuilder
        ├── build_many(train, validation, test)
        │      ├── build_split(train)
        │      ├── build_split(validation)
        │      └── build_split(test)
        │
        ├── 检查 split overlap
        ├── 确定输出目录
        ├── write_manifest(train)
        ├── write_manifest(validation)
        ├── write_manifest(test)
        └── return 0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence

from m3d.config import ExperimentConfig, load_config
from m3d.data.schema import DataSplit, TaskName


LOGGER = logging.getLogger(__name__)
MANIFEST_SCHEMA_VERSION = 1
_SUPPORTED_VOLUME_SUFFIXES = (".npy", ".nii", ".nii.gz")


class ManifestError(RuntimeError):
    """Raised when a source table cannot be converted safely."""


class PromptVariant(str, Enum):
    """Stable logical variants reproduced from the original M3D datasets."""
    
    CAPTION = "caption"
    VQA_CLOSED = "vqa_closed"
    VQA_OPEN = "vqa_open"
    VQA_YES_NO = "vqa_yes_no"
    REC_CLASS = "rec_class" # 根据描述定位区域
    REC_DESCRIPTION = "rec_description"
    REG_CLASS = "reg_class" # 根据区域生成描述
    REG_DESCRIPTION = "reg_description"
    SEG_CLASS = "seg_class" # 生成 segmentation mask
    SEG_DESCRIPTION = "seg_description"
    REFERRING_SEGMENTATION = "referring_segmentation"

    @classmethod
    def parse(cls, value: str | "PromptVariant") -> "PromptVariant":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip())
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise ManifestError(
                f"Unknown prompt variant {value!r}; allowed: {allowed}"
            ) from exc


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    """One portable logical example before image loading and tokenisation."""

    record_id: str
    task: TaskName
    split: DataSplit
    source_name: str
    source_index: int
    image_path: str
    prompt_variant: PromptVariant

    question: str | None = None
    answer: str | None = None
    text_path: str | None = None
    mask_path: str | None = None
    mask_label_id: int | float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        task = TaskName.parse(self.task)
        split = DataSplit.parse(self.split)
        variant = PromptVariant.parse(self.prompt_variant)
        record_id = self.record_id.strip()
        source_name = self.source_name.strip()
        image_path = _normalise_manifest_path(self.image_path)
        question = _optional_text(self.question)
        answer = _optional_text(self.answer)
        text_path = (
            None
            if self.text_path is None
            else _normalise_manifest_path(self.text_path)
        )
        mask_path = (
            None
            if self.mask_path is None
            else _normalise_manifest_path(self.mask_path)
        )
        metadata = MappingProxyType(_json_safe_mapping(self.metadata))

        if not record_id:
            raise ManifestError("record_id cannot be empty")
        if not source_name:
            raise ManifestError(f"source_name is empty for {record_id!r}")
        if self.source_index < 0:
            raise ManifestError(
                f"source_index cannot be negative for {record_id!r}"
            )
        if not _looks_like_volume(image_path):
            raise ManifestError(
                f"Unsupported image path in {record_id!r}: {image_path!r}"
            )

        if task is TaskName.CAPTION:
            if text_path is None:
                raise ManifestError(
                    f"Caption record {record_id!r} requires text_path"
                )
            if question is not None or answer is not None:
                raise ManifestError(
                    f"Caption record {record_id!r} must defer prompt/text loading"
                )
            if mask_path is not None:
                raise ManifestError(
                    f"Caption record {record_id!r} cannot carry a mask"
                )

        elif task in {
            TaskName.VQA_CLOSED,
            TaskName.VQA_OPEN,
            TaskName.VQA_YES_NO,
        }:
            if question is None or answer is None:
                raise ManifestError(
                    f"VQA record {record_id!r} requires question and answer"
                )
            if text_path is not None or mask_path is not None:
                raise ManifestError(
                    f"VQA record {record_id!r} cannot carry text_path or mask"
                )

        elif task is TaskName.POSITIONING:
            if mask_path is None:
                raise ManifestError(
                    f"Positioning record {record_id!r} requires mask_path"
                )
            if variant not in {
                PromptVariant.REC_CLASS,
                PromptVariant.REC_DESCRIPTION,
                PromptVariant.REG_CLASS,
                PromptVariant.REG_DESCRIPTION,
            }:
                raise ManifestError(
                    f"Invalid positioning variant {variant.value!r}"
                )
            if question is not None or answer is not None or text_path is not None:
                raise ManifestError(
                    "Generated positioning prompts must not be frozen in the "
                    f"manifest; record={record_id!r}"
                )

        elif task is TaskName.SEGMENTATION:
            if mask_path is None:
                raise ManifestError(
                    f"Segmentation record {record_id!r} requires mask_path"
                )
            if variant is PromptVariant.REFERRING_SEGMENTATION:
                if question is None or answer is None:
                    raise ManifestError(
                        f"RefSeg record {record_id!r} requires question and answer"
                    )
            elif variant not in {
                PromptVariant.SEG_CLASS,
                PromptVariant.SEG_DESCRIPTION,
            }:
                raise ManifestError(
                    f"Invalid segmentation variant {variant.value!r}"
                )
            elif question is not None or answer is not None:
                raise ManifestError(
                    "Generated segmentation prompts must not be frozen in the "
                    f"manifest; record={record_id!r}"
                )
            if text_path is not None:
                raise ManifestError(
                    f"Segmentation record {record_id!r} cannot carry text_path"
                )

        if task.requires_segmentation_target != (mask_path is not None) and task not in {
            TaskName.POSITIONING
        }:
            raise ManifestError(
                f"Task/mask contract mismatch for record {record_id!r}"
            )
        if self.mask_label_id is not None:
            value = float(self.mask_label_id)
            if not math.isfinite(value):
                raise ManifestError(
                    f"mask_label_id must be finite for {record_id!r}"
                )

        object.__setattr__(self, "task", task)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "prompt_variant", variant)
        object.__setattr__(self, "record_id", record_id)
        object.__setattr__(self, "source_name", source_name)
        object.__setattr__(self, "image_path", image_path)
        object.__setattr__(self, "question", question)
        object.__setattr__(self, "answer", answer)
        object.__setattr__(self, "text_path", text_path)
        object.__setattr__(self, "mask_path", mask_path)
        object.__setattr__(self, "metadata", metadata)

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "task": self.task.value,
            "split": self.split.value,
            "source_name": self.source_name,
            "source_index": self.source_index,
            "image_path": self.image_path,
            "prompt_variant": self.prompt_variant.value,
            "question": self.question,
            "answer": self.answer,
            "text_path": self.text_path,
            "mask_path": self.mask_path,
            "mask_label_id": self.mask_label_id,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ManifestRecord":
        required = {
            "record_id",
            "task",
            "split",
            "source_name",
            "source_index",
            "image_path",
            "prompt_variant",
        }
        missing = sorted(required - set(value))
        if missing:
            raise ManifestError(
                "Manifest record is missing fields: " + ", ".join(missing)
            )
        return cls(
            record_id=str(value["record_id"]),
            task=TaskName.parse(value["task"]),
            split=DataSplit.parse(value["split"]),
            source_name=str(value["source_name"]),
            source_index=int(value["source_index"]),
            image_path=str(value["image_path"]),
            prompt_variant=PromptVariant.parse(value["prompt_variant"]),
            question=value.get("question"),
            answer=value.get("answer"),
            text_path=value.get("text_path"),
            mask_path=value.get("mask_path"),
            mask_label_id=value.get("mask_label_id"),
            metadata=value.get("metadata", {}),
        )


@dataclass(frozen=True, slots=True)
class M3DManifest:
    """Immutable, validated records for exactly one data split."""

    split: DataSplit
    records: tuple[ManifestRecord, ...]

    def __post_init__(self) -> None:
        split = DataSplit.parse(self.split)
        records = tuple(self.records)
        if not records:
            raise ManifestError(f"Manifest for {split.value!r} is empty")

        ids: set[str] = set()
        duplicates: list[str] = []
        for record in records:
            if record.split is not split:
                raise ManifestError(
                    f"Record {record.record_id!r} belongs to {record.split.value}, "
                    f"not {split.value}"
                )
            if record.record_id in ids:
                duplicates.append(record.record_id)
            ids.add(record.record_id)
        if duplicates:
            raise ManifestError(
                "Duplicate manifest record IDs: "
                + ", ".join(sorted(set(duplicates))[:20])
            )

        object.__setattr__(self, "split", split)
        object.__setattr__(self, "records", records)

    def __len__(self) -> int:
        return len(self.records)

    @property
    def counts_by_task(self) -> dict[str, int]:
        counts = Counter(record.task.value for record in self.records)
        return dict(sorted(counts.items()))

    @property
    def counts_by_variant(self) -> dict[str, int]:
        counts = Counter(record.prompt_variant.value for record in self.records)
        return dict(sorted(counts.items()))

    @property
    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        for record in self.records:
            payload = json.dumps(
                record.to_dict(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            digest.update(payload.encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    def records_for_task(self, task: str | TaskName) -> tuple[ManifestRecord, ...]:
        canonical = TaskName.parse(task)
        return tuple(record for record in self.records if record.task is canonical)

    def summary(self) -> dict[str, Any]:
        unique_images = {record.image_path for record in self.records}
        unique_masks = {
            record.mask_path
            for record in self.records
            if record.mask_path is not None
        }
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "split": self.split.value,
            "record_count": len(self.records),
            "unique_image_count": len(unique_images),
            "unique_mask_count": len(unique_masks),
            "counts_by_task": self.counts_by_task,
            "counts_by_variant": self.counts_by_variant,
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class ManifestBuildOptions:
    """Which original M3D logical datasets are expanded into records."""

    verify_files: bool = True
    include_caption: bool = True
    include_vqa_closed: bool = True
    include_vqa_open: bool = True
    include_vqa_yes_no: bool = True
    include_positioning: bool = True
    include_generated_segmentation: bool = True
    include_referring_segmentation: bool = True
    fail_on_split_overlap: bool = True


class ManifestBuilder:
    """Parse original M3D metadata into deterministic portable manifests."""

    def __init__(
        self,
        config: ExperimentConfig,
        options: ManifestBuildOptions | None = None,
    ) -> None:
        self.config = config
        self.options = options or ManifestBuildOptions(
            verify_files=config.data.verify_files_at_startup
        )
        self.data_root = Path(config.data.paths.data_root).expanduser().resolve()
        # /scratch/jiacheng/M3D-modernized/Data/data

        if not self.data_root.is_dir():
            raise ManifestError(f"data_root does not exist: {self.data_root}")
        self._verified_files: set[Path] = set()

    def build_split(self, split: str | DataSplit) -> M3DManifest:
        canonical_split = DataSplit.parse(split)
        records: list[ManifestRecord] = []

        if self.options.include_caption:
            records.extend(self._caption_records(canonical_split))
        if self.options.include_vqa_closed or self.options.include_vqa_open:
            records.extend(self._vqa_records(canonical_split))
        if self.options.include_vqa_yes_no:
            records.extend(self._yes_no_records(canonical_split))
        if self.options.include_referring_segmentation:
            records.extend(self._refseg_records(canonical_split))
        if self.options.include_positioning or self.options.include_generated_segmentation:
            records.extend(self._decathlon_records(canonical_split))

        records.sort(
            key=lambda item: (
                int(item.task.task_id),
                item.source_name,
                item.source_index,
                item.prompt_variant.value,
                item.record_id,
            )
        )
        return M3DManifest(split=canonical_split, records=tuple(records))

    def build_many(
        self,
        splits: Sequence[str | DataSplit],
    ) -> dict[DataSplit, M3DManifest]:
        """
        Build manifests for multiple splits and check for overlap between them.
        build_split("train")
        build_split("validation")
        build_split("test")
        """
        manifests = {
            DataSplit.parse(split): self.build_split(split)
            for split in splits
        }
        """
        manifest的结构是一个字典, 键是DataSplit类型的split, 值是对应的M3DManifest对象。这个方法会遍历传入的splits序列,对每个split调用build_split方法生成对应的manifest, 并将结果存储在manifests字典中。
        manifests = {
            DataSplit.TRAIN: M3DManifest(
                split=DataSplit.TRAIN,
                records=(
                    ManifestRecord(...),
                    ManifestRecord(...),
                    ...
                ),
            ),

            DataSplit.VALIDATION: M3DManifest(
                split=DataSplit.VALIDATION,
                records=(
                    ManifestRecord(...),
                    ManifestRecord(...),
                    ...
                ),
            ),

            DataSplit.TEST: M3DManifest(
                split=DataSplit.TEST,
                records=(
                    ManifestRecord(...),
                    ManifestRecord(...),
                    ...
                ),
            ),
        }
        """

        # 三个 split 之间的 overlap 检查
        self._check_split_overlap(manifests)
        return manifests

    def _caption_records(self, split: DataSplit) -> list[ManifestRecord]:
        metadata_path = self.config.dataset_path(
            self.config.data.paths.caption_json
        )
        # metadata_path = /scratch/users/nus/e1129906/M3D-modernized/Data/data/M3D_Cap_npy/M3D_Cap.json

        document = _read_json_object(metadata_path)
        # 提取出指定 split 的行数据 （train/validation/test）
        rows = _select_json_split(document, split, source=metadata_path)
        records: list[ManifestRecord] = []

        # 遍历每一行数据，提取 image 和 text 的路径，并进行验证和记录创建
        for index, raw in enumerate(rows):
            """
            row = raw
            image = row["image"] or row["Image"] or row["Image Path"]
            text = row["text"] or row["Text"] or row["Text Path"]
            text_path = "M3D_Cap_npy/images/case_0001.txt"
            image_path = "M3D_Cap_npy/images/case_0001.npy"
            """
            row = _require_mapping(raw, metadata_path, index)
            image = _required_value(row, ("image", "Image", "Image Path"))
            text = _required_value(row, ("text", "Text", "Text Path"))
            image_path = self._portable_path(image, base=self.data_root)
            text_path = self._portable_path(text, base=self.data_root)
            self._verify_data_file(image_path, kind="image")
            self._verify_data_file(text_path, kind="text")
            records.append(
                self._record(
                    task=TaskName.CAPTION,
                    split=split,
                    source_name="M3D_Cap",
                    source_index=index,
                    image_path=image_path,
                    text_path=text_path,
                    prompt_variant=PromptVariant.CAPTION,
                )
            )
        return records

    def _vqa_records(self, split: DataSplit) -> list[ManifestRecord]:
        relative_path = {
            DataSplit.TRAIN: self.config.data.paths.vqa_train_csv,
            DataSplit.VALIDATION: self.config.data.paths.vqa_val_csv,
            DataSplit.TEST: self.config.data.paths.vqa_test_csv,
        }[split]
        csv_path = self.config.dataset_path(relative_path)
        # csv_path = /scratch/users/nus/e1129906/M3D-modernized/Data/data/M3D_VQA_npy/M3D_VQA_train.csv

        if not csv_path.is_file():
            LOGGER.warning("Skipping missing VQA %s file: %s", split.value, csv_path)
            return []

        records: list[ManifestRecord] = []
        for index, row in enumerate(_read_csv_rows(csv_path)): # read_csv_rows(csv_path) 读取 CSV 文件的每一行，返回一个迭代器，每次迭代返回一行数据（字典形式）
            image_path = self._portable_path(
                _csv_required(row, "Image Path"), base=self.data_root
            )
            # image_path = M3D_VQA_npy/images/case_0001.npy

            question = _csv_required(row, "Question")
            answer = _csv_required(row, "Answer")
            self._verify_data_file(image_path, kind="image")

            if self.options.include_vqa_closed:
                choices = {
                    letter: _csv_required(row, f"Choice {letter}")
                    for letter in ("A", "B", "C", "D")
                }
                answer_choice = _csv_required(row, "Answer Choice")
                closed_question = (
                    f"{question} Choices: "
                    + " ".join(
                        f"{letter}. {choices[letter]}"
                        for letter in ("A", "B", "C", "D")
                    )
                )
                closed_answer = f"{answer_choice}. {answer}"
                """
                choices = {
                    "A": "Liver",
                    "B": "Lung",
                    "C": "Kidney",
                    "D": "Heart",
                }
                answer_choice = "A"
                closed_question = "What organ is shown in the image? Choices: A. Liver B. Lung C. Kidney D. Heart"
                closed_answer = "A. Liver"
                """
                records.append(
                    self._record(
                        task=TaskName.VQA_CLOSED,
                        split=split,
                        source_name="M3D_VQA",
                        source_index=index,
                        image_path=image_path,
                        question=closed_question,
                        answer=closed_answer,
                        prompt_variant=PromptVariant.VQA_CLOSED,
                        metadata={
                            "answer_choice": answer_choice,
                            "choices": choices,
                            "question_type": _csv_optional(row, "Question Type"),
                        },
                    )
                )

            if self.options.include_vqa_open:
                records.append(
                    self._record(
                        task=TaskName.VQA_OPEN,
                        split=split,
                        source_name="M3D_VQA",
                        source_index=index,
                        image_path=image_path,
                        question=question,
                        answer=answer,
                        prompt_variant=PromptVariant.VQA_OPEN,
                        metadata={
                            "question_type": _csv_optional(row, "Question Type")
                        },
                    )
                )
        return records

    def _yes_no_records(self, split: DataSplit) -> list[ManifestRecord]:
        # The original public configuration exposes only a training CSV for
        # this auxiliary task.  Validation/test are intentionally absent rather
        # than borrowing rows from another split.

        # 只处理训练集(train)
        if split is not DataSplit.TRAIN:
            return []
        csv_path = self.config.dataset_path(
            self.config.data.paths.vqa_yes_no_train_csv
        )
        if not csv_path.is_file():
            LOGGER.warning("Skipping missing yes/no VQA file: %s", csv_path)
            return []

        records: list[ManifestRecord] = []
        for index, row in enumerate(_read_csv_rows(csv_path)):
            image_path = self._portable_path(
                _csv_required(row, "Image Path"), base=self.data_root
            )
            self._verify_data_file(image_path, kind="image")
            records.append(
                self._record(
                    task=TaskName.VQA_YES_NO,
                    split=split,
                    source_name="M3D_VQA_YN",
                    source_index=index,
                    image_path=image_path,
                    question=_csv_required(row, "Question"),
                    answer=_csv_required(row, "Answer"),
                    prompt_variant=PromptVariant.VQA_YES_NO,
                    metadata={
                        "answer_choice": _csv_optional(row, "Answer Choice"),
                        "question_type": _csv_optional(row, "Question Type"),
                    },
                )
            )
        return records

    def _refseg_records(self, split: DataSplit) -> list[ManifestRecord]:
        if split is DataSplit.TRAIN:
            relative_path = self.config.data.paths.referring_segmentation_train_csv
        elif split is DataSplit.TEST:
            relative_path = self.config.data.paths.referring_segmentation_test_csv
        else:
            return []
        
        csv_path = self.config.dataset_path(relative_path)
        if not csv_path.is_file():
            LOGGER.warning("Skipping missing RefSeg %s file: %s", split.value, csv_path)
            return []

        records: list[ManifestRecord] = []
        for index, row in enumerate(_read_csv_rows(csv_path)):
            image_path = self._portable_path(
                _csv_required(row, "Image"), base=self.data_root
            )
            mask_path = self._portable_path(
                _csv_required(row, "Mask"), base=self.data_root
            )
            mask_id = _parse_numeric_label(_csv_required(row, "Mask_ID"))
            self._verify_data_file(image_path, kind="image")
            self._verify_data_file(mask_path, kind="mask")
            records.append(
                self._record(
                    task=TaskName.SEGMENTATION,
                    split=split,
                    source_name="M3D_RefSeg",
                    source_index=index,
                    image_path=image_path,
                    mask_path=mask_path,
                    mask_label_id=mask_id,
                    question=_csv_required(row, "Question"),
                    answer=_csv_required(row, "Answer"),
                    prompt_variant=PromptVariant.REFERRING_SEGMENTATION,
                )
            )
        return records

    def _decathlon_records(self, split: DataSplit) -> list[ManifestRecord]:
        segmentation_root = self.config.dataset_path(
            self.config.data.paths.segmentation_root
        )
        # segmentation_root = /scratch/users/nus/e1129906/M3D-modernized/Data/data/M3D_Seg_npy

        if not segmentation_root.is_dir():
            LOGGER.warning("Skipping missing segmentation root: %s", segmentation_root)
            return []

        records: list[ManifestRecord] = []
        if split is DataSplit.VALIDATION:
            return []
        split_key = "train" if split is DataSplit.TRAIN else "test"

        # 遍历 segmentation_root 下的每个子目录，查找符合模式的 JSON 文件
        for metadata_path in sorted(segmentation_root.glob("*/[0-9][0-9][0-9][0-9].json")):
            dataset_tag = metadata_path.parent.name
            # dataset_tag = 0001
            if metadata_path.stem != dataset_tag:
                continue
            document = _read_json_object(metadata_path)
            raw_rows = document.get(split_key, [])
            if not isinstance(raw_rows, list):
                raise ManifestError(
                    f"{metadata_path}: key {split_key!r} must contain a list"
                )

            for index, raw in enumerate(raw_rows):
                # 确保每一行数据是一个映射（字典），并提取 image 和 mask 的路径
                row = _require_mapping(raw, metadata_path, index)
                image_raw = _single_decathlon_path(
                    _required_value(row, ("image",)), metadata_path, index, "image"
                )
                # image_raw = "0001/images/case_0001.npy"
                mask_raw = _single_decathlon_path(
                    _required_value(row, ("label", "mask")),
                    metadata_path,
                    index,
                    "label",
                )
                image_path = self._portable_path(image_raw, base=segmentation_root)
                mask_path = self._portable_path(mask_raw, base=segmentation_root)
                self._verify_data_file(image_path, kind="image")
                self._verify_data_file(mask_path, kind="mask")
                class_id = _class_id_from_mask_name(mask_path)
                # if mask_path.name == " "M3D_Seg_npy/0001/labels/case_0001_3.npy"":
                #     class_id = 3
                common_metadata = {
                    "dataset_tag": dataset_tag,
                    "class_id": class_id,
                    "decathlon_json": self._portable_path(
                        metadata_path, base=self.data_root
                    ),
                    # decalthon_json = "M3D_Seg_npy/0001/0001.json"
                }
                source_name = f"M3D_Seg_{dataset_tag}"

                if self.options.include_positioning:
                    for variant in (
                        PromptVariant.REC_CLASS,
                        PromptVariant.REC_DESCRIPTION,
                        PromptVariant.REG_CLASS,
                        PromptVariant.REG_DESCRIPTION,
                    ):
                        records.append(
                            self._record(
                                task=TaskName.POSITIONING,
                                split=split,
                                source_name=source_name,
                                source_index=index,
                                image_path=image_path,
                                mask_path=mask_path,
                                prompt_variant=variant,
                                metadata=common_metadata,
                            )
                        )

                if self.options.include_generated_segmentation:
                    for variant in (
                        PromptVariant.SEG_CLASS,
                        PromptVariant.SEG_DESCRIPTION,
                    ):
                        records.append(
                            self._record(
                                task=TaskName.SEGMENTATION,
                                split=split,
                                source_name=source_name,
                                source_index=index,
                                image_path=image_path,
                                mask_path=mask_path,
                                prompt_variant=variant,
                                metadata=common_metadata,
                            )
                        )
        return records

    def _record(self, **values: Any) -> ManifestRecord:
        identity = {
            key: value.value if isinstance(value, Enum) else value
            for key, value in values.items()
            if key not in {"record_id"}
        }
        payload = json.dumps(
            _json_safe_mapping(identity),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        record_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]
        return ManifestRecord(record_id=record_id, **values)

    def _portable_path(self, value: Any, *, base: Path) -> str:
        raw = _clean_scalar(value)
        if raw is None:
            raise ManifestError("Required path value is empty")
        candidate = Path(raw).expanduser()
        if not candidate.is_absolute():
            candidate = base / candidate
        candidate = candidate.resolve()
        # candidate = /scratch/users/nus/e1129906/M3D-modernized/Data/data/M3D_Cap_npy/images/case_0001.npy
        try:
            relative = candidate.relative_to(self.data_root)
        except ValueError as exc:
            raise ManifestError(
                f"Data path escapes data_root: {candidate} (root={self.data_root})"
            ) from exc
        return relative.as_posix() # 

    def _verify_data_file(self, relative_path: str, *, kind: str) -> None:
        if not self.options.verify_files:
            return
        path = (self.data_root / relative_path).resolve()
        if path in self._verified_files:
            return
        if not path.is_file():
            raise ManifestError(f"Missing {kind} file: {path}")
        if path.stat().st_size <= 0:
            raise ManifestError(f"Empty {kind} file: {path}")
        if kind in {"image", "mask"} and not _looks_like_volume(path.name):
            raise ManifestError(f"Unsupported {kind} suffix: {path}")
        self._verified_files.add(path)

    def _check_split_overlap(
        self,
        manifests: Mapping[DataSplit, M3DManifest],
    ) -> None:
        ownership: dict[tuple[str, str], set[DataSplit]] = defaultdict(set)
        for split, manifest in manifests.items():
            for record in manifest.records:
                ownership[(record.task.value, record.image_path)].add(split)
        overlaps = {
            key: splits
            for key, splits in ownership.items()
            if len(splits) > 1
        }
        if overlaps and self.options.fail_on_split_overlap:
            examples = []
            for (task, image), splits in sorted(overlaps.items())[:20]:
                names = ",".join(sorted(split.value for split in splits))
                examples.append(f"{task}:{image} [{names}]")
            raise ManifestError(
                "The same task/image appears in multiple splits. Examples: "
                + "; ".join(examples)
            )
        if overlaps:
            LOGGER.warning(
                "Detected %d task/image overlaps across splits", len(overlaps)
            )


def write_manifest(
    manifest: M3DManifest,
    path: str | os.PathLike[str],
) -> tuple[Path, Path]:
    """Atomically write JSONL records and a sibling summary JSON."""

    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True) # 创建目录：.../manifests\

    """
    生成 JSONL header:
    {
        "type": "m3d_manifest_header",
        "schema_version": 1,
        "split": "train",
        "record_count": 120000,
        "fingerprint": "7fd7d9d5a9..."
    }
    """
    header = {
        "type": "m3d_manifest_header",
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "split": manifest.split.value,
        "record_count": len(manifest),
        "fingerprint": manifest.fingerprint,
    }

    _atomic_write_lines(
        destination,
        [
            json.dumps(header, sort_keys=True, ensure_ascii=False),
            *(
                json.dumps(
                    record.to_dict(),
                    sort_keys=True,
                    ensure_ascii=False,
                )
                for record in manifest.records
            ),
        ],
    )
    summary_path = destination.with_suffix(destination.suffix + ".summary.json")
    _atomic_write_json(summary_path, manifest.summary())
    return destination, summary_path


def read_manifest(path: str | os.PathLike[str]) -> M3DManifest:
    """Read a manifest, validate its header, records, count, and fingerprint."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {source}")
    with source.open("r", encoding="utf-8") as handle:
        try:
            header = json.loads(next(handle))
        except StopIteration as exc:
            raise ManifestError(f"Manifest is empty: {source}") from exc
        except json.JSONDecodeError as exc:
            raise ManifestError(f"Invalid manifest header: {source}") from exc

        if header.get("type") != "m3d_manifest_header":
            raise ManifestError(f"Invalid manifest header type: {source}")
        if int(header.get("schema_version", -1)) != MANIFEST_SCHEMA_VERSION:
            raise ManifestError(
                f"Unsupported manifest schema in {source}: "
                f"{header.get('schema_version')!r}"
            )

        records: list[ManifestRecord] = []
        for line_number, line in enumerate(handle, start=2):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
                records.append(ManifestRecord.from_dict(value))
            except (json.JSONDecodeError, TypeError, ValueError, ManifestError) as exc:
                raise ManifestError(
                    f"Invalid record at {source}:{line_number}: {exc}"
                ) from exc

    manifest = M3DManifest(
        split=DataSplit.parse(header["split"]),
        records=tuple(records),
    )
    if int(header.get("record_count", -1)) != len(manifest):
        raise ManifestError(
            f"Manifest count mismatch in {source}: header={header.get('record_count')}, "
            f"actual={len(manifest)}"
        )
    if header.get("fingerprint") != manifest.fingerprint:
        raise ManifestError(f"Manifest fingerprint mismatch: {source}")
    return manifest


def _select_json_split(
    document: Mapping[str, Any],
    split: DataSplit,
    *,
    source: Path,
) -> list[Any]:
    aliases = {
        DataSplit.TRAIN: ("train", "training"),
        DataSplit.VALIDATION: ("validation", "val", "valid"),
        DataSplit.TEST: ("test", "testing"),
    }[split]
    for key in aliases:
        if key in document:
            value = document[key]
            if not isinstance(value, list):
                raise ManifestError(f"{source}: {key!r} must contain a list")
            return value
    raise ManifestError(
        f"{source}: missing {split.value!r} split; tried keys {aliases}"
    )


def _read_json_object(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise ManifestError(f"Metadata JSON does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8-sig") as handle:
            value = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ManifestError(f"Invalid JSON: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ManifestError(f"Top-level JSON must be an object: {path}")
    return value


def _read_csv_rows(path: Path) -> Iterator[dict[str, str]]:
    if not path.is_file():
        raise ManifestError(f"CSV does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ManifestError(f"CSV has no header: {path}")
        for row_index, raw_row in enumerate(reader):
            row: dict[str, str] = {}
            for key, value in raw_row.items():
                if key is None:
                    continue
                row[_normalise_column_name(key)] = "" if value is None else value.strip()
            if not any(row.values()):
                LOGGER.warning("Skipping blank CSV row %d in %s", row_index + 2, path)
                continue
            yield row


def _normalise_column_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.strip().lower())


def _csv_required(row: Mapping[str, str], column: str) -> str:
    # Normalize column names to be case-insensitive and ignore non-alphanumeric characters
    key = _normalise_column_name(column)
    value = _clean_scalar(row.get(key))
    if value is None:
        raise ManifestError(f"CSV row is missing required column/value {column!r}")
    return value


def _csv_optional(row: Mapping[str, str], column: str) -> str | None:
    return _clean_scalar(row.get(_normalise_column_name(column)))


def _required_value(row: Mapping[str, Any], aliases: Sequence[str]) -> Any:
    for key in aliases:
        if key in row and _clean_scalar(row[key]) is not None:
            return row[key]
    raise ManifestError(
        "Metadata row is missing required key; tried " + ", ".join(aliases)
    )


def _require_mapping(value: Any, source: Path, index: int) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ManifestError(f"{source}: row {index} must be an object")
    return value


def _single_decathlon_path(
    value: Any,
    source: Path,
    index: int,
    field_name: str,
) -> str:
    if isinstance(value, (list, tuple)):
        if len(value) != 1:
            raise ManifestError(
                f"{source}: row {index} field {field_name!r} must contain one path"
            )
        value = value[0]
    cleaned = _clean_scalar(value)
    if cleaned is None:
        raise ManifestError(
            f"{source}: row {index} field {field_name!r} is empty"
        )
    return cleaned


def _class_id_from_mask_name(path: str) -> int:
    name = Path(path).name
    match = re.search(r"_(\d+)(?:\.[^.]+)*(?:\.gz)?$", name)
    if match is None:
        raise ManifestError(
            "Cannot infer class ID from segmentation filename; expected a suffix "
            f"such as '_3.npy': {path}"
        )
    return int(match.group(1))


def _parse_numeric_label(value: str) -> int | float:
    try:
        numeric = float(value)
    except ValueError as exc:
        raise ManifestError(f"Mask_ID must be numeric, got {value!r}") from exc
    if not math.isfinite(numeric):
        raise ManifestError(f"Mask_ID must be finite, got {value!r}")
    if numeric.is_integer():
        return int(numeric)
    return numeric


def _clean_scalar(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    if "\x00" in text:
        raise ManifestError("Metadata value contains a NUL byte")
    return text


def _optional_text(value: Any) -> str | None:
    return _clean_scalar(value)


def _normalise_manifest_path(value: str) -> str:
    path = Path(str(value).strip())
    if path.is_absolute():
        raise ManifestError(
            f"Manifest paths must be relative to data_root, got {value!r}"
        )
    if not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise ManifestError(f"Unsafe manifest path: {value!r}")
    return path.as_posix()


def _looks_like_volume(value: str) -> bool:
    lowered = value.lower()
    return any(lowered.endswith(suffix) for suffix in _SUPPORTED_VOLUME_SUFFIXES)


def _json_safe_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): _json_safe(item)
        for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return _json_safe_mapping(value)
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ManifestError("Manifest metadata cannot contain NaN or Inf")
        return value
    raise ManifestError(
        f"Manifest metadata is not JSON serialisable: {type(value).__name__}"
    )


def _atomic_write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for line in lines:
                handle.write(line)
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_write_lines(
        path,
        [json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False)],
    )


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build deterministic M3D JSONL manifests from original metadata."
    )
    parser.add_argument("--config", required=True, help="Path to experiment YAML")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Manifest directory; defaults to <checkpoint.output_dir>/manifests",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "validation", "test"],
        choices=[item.value for item in DataSplit],
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Configuration override such as data.paths.data_root=/scratch/...",
    )
    parser.add_argument("--no-verify-files", action="store_true")
    parser.add_argument("--allow-split-overlap", action="store_true")
    parser.add_argument(
        "--active-tasks-only",
        action="store_true",
        help=(
            "Build only tasks whose data.task_sampling.task_weights value is "
            "positive. This is required for caption-only projector pretraining."
        ),
    )
    parser.add_argument("--skip-caption", action="store_true")
    parser.add_argument("--skip-vqa-closed", action="store_true")
    parser.add_argument("--skip-vqa-open", action="store_true")
    parser.add_argument("--skip-vqa-yes-no", action="store_true")
    parser.add_argument("--skip-positioning", action="store_true")
    parser.add_argument("--skip-generated-segmentation", action="store_true")
    parser.add_argument("--skip-refseg", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_argument_parser().parse_args(argv)
    # PBS 默认得到
    # args.config = configs/m3d_joint_finetune.yaml
    # args.splits = ["train", "validation", "test"]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    config = load_config(
        args.config,
        args.overrides,
        verify_paths=False,
    )

    active_tasks = {
        TaskName.parse(name)
        for name, weight in config.data.task_sampling.task_weights.items()
        if float(weight) > 0.0
    }

    def include(task: TaskName, explicitly_skipped: bool) -> bool:
        return (
            not explicitly_skipped
            and (not args.active_tasks_only or task in active_tasks)
        )

    options = ManifestBuildOptions(
        verify_files=not args.no_verify_files,
        include_caption=include(TaskName.CAPTION, args.skip_caption),
        include_vqa_closed=include(TaskName.VQA_CLOSED, args.skip_vqa_closed),
        include_vqa_open=include(TaskName.VQA_OPEN, args.skip_vqa_open),
        include_vqa_yes_no=include(TaskName.VQA_YES_NO, args.skip_vqa_yes_no),
        include_positioning=include(TaskName.POSITIONING, args.skip_positioning),
        include_generated_segmentation=include(
            TaskName.SEGMENTATION,
            args.skip_generated_segmentation,
        ),
        include_referring_segmentation=include(
            TaskName.SEGMENTATION,
            args.skip_refseg,
        ),
        fail_on_split_overlap=not args.allow_split_overlap,
    )

    builder = ManifestBuilder(config, options)
    manifests = builder.build_many(args.splits)

    # 确定输出目录，如果用户没有指定，则使用配置文件中的 checkpoint.output_dir 下的 manifests 子目录
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir is not None
        else Path(config.checkpoint.output_dir).resolve() / "manifests"
    )

    for split, manifest in manifests.items():
        # write_manifest 函数将 manifest 写入指定的输出目录，并返回写入的路径和摘要路径
        path, summary_path = write_manifest(
            manifest,
            output_dir / f"{split.value}.jsonl",
        )
        LOGGER.info(
            "Wrote %s records for %s: %s (summary: %s)",
            len(manifest),
            split.value,
            path,
            summary_path,
        )
        LOGGER.info("Counts by task: %s", manifest.counts_by_task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
