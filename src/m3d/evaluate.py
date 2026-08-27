"""Distributed evaluation for M3D-Modernized.

This module evaluates the portable M3D export or an exact distributed training
checkpoint on the validation/test manifests created by :mod:`m3d.data.manifest`.
It preserves the project's explicit task routing:

* caption, VQA and positioning use Main 3-D ViT -> MM projector -> Phi-3;
* segmentation additionally uses the independent SegVol 3-D ViT and decoder;
* an all-zero target remains a real segmentation example.

The evaluation sampler never duplicates examples.  Replicated/DDP-style model
execution therefore permits different local loader lengths.  FSDP2 evaluation
uses an equal number of model calls on every rank and enables Hugging Face's
``synced_gpus`` generation path so that parameter collectives remain aligned.
Per-example rows are written rank-locally and merged on rank 0, avoiding a
large ``all_gather_object`` of generated text and dense metric metadata.

Examples
--------
Portable export (one full replica per GPU)::

    torchrun --standalone --nproc_per_node=2 -m m3d.evaluate \
        --source export \
        --export-dir outputs/m3d-phi3-export \
        --config configs/m3d_joint_finetune.yaml \
        --split test \
        --output-dir outputs/eval-test

FSDP2 checkpoint evaluation::

    torchrun --standalone --nproc_per_node=2 -m m3d.evaluate \
        --source checkpoint \
        --checkpoint outputs/m3d-phi3-joint-modernized-a100 \
        --config configs/m3d_joint_finetune.yaml \
        --strategy fsdp2 \
        --split validation \
        --output-dir outputs/eval-validation-fsdp2

Image-text retrieval can also be scored from aligned precomputed M3D-CLIP
feature matrices using ``--retrieval-image-features`` and
``--retrieval-text-features``.  ``topk`` is always clamped to dataset size, so
small test sets cannot fail with ``selected index k out of range``.
"""

from __future__ import annotations

import argparse
import ast
import contextlib
import copy
import dataclasses
import hashlib
import json
import math
import os
import random
import re
import shutil
import string
import tempfile
import time
import uuid
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping, MutableMapping, Sequence, cast

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.utils.data import DataLoader, Sampler

from .config import ExperimentConfig, load_config
from .data.schema import DataSplit, M3DBatch, M3DSample, TaskName
from .runtime import RuntimeContext, initialize_runtime, make_dataloader_generator

if TYPE_CHECKING:
    from .data.collator import M3DCollator
    from .data.loader import EvaluationDataPipeline
    from .distributed import DistributedM3DModel
    from .inference import BatchInferenceResult, M3DInferenceEngine
    from .tokenization import TokenizerBundle


_EVALUATION_STATE_VERSION = 1
_BOX_PATTERN = re.compile(
    r"<bx_start>\s*(\[[^\]]+\])\s*<bx_end>",
    flags=re.IGNORECASE | re.DOTALL,
)
_FLOAT_PATTERN = re.compile(
    r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
)


class EvaluationError(RuntimeError):
    """Raised when evaluation cannot preserve the requested metric contract."""


class EvaluationInputError(EvaluationError, ValueError):
    """Raised for malformed CLI inputs, predictions or references."""


class EvaluationCompatibilityError(EvaluationError):
    """Raised when a model/export/checkpoint cannot be evaluated strictly."""


class EvaluationSource(str, Enum):
    EXPORT = "export"
    CHECKPOINT = "checkpoint"

    @classmethod
    def parse(cls, value: str | "EvaluationSource") -> "EvaluationSource":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            raise EvaluationInputError(
                f"Unknown evaluation source {value!r}; expected export or checkpoint."
            ) from exc


@dataclass(frozen=True, slots=True)
class EvaluationGenerationSettings:
    """Generation settings with optional cross-rank synchronisation."""

    base: Any
    synced_gpus: bool = False

    def generation_kwargs(self, *, pad_token_id: int, eos_token_id: int) -> dict[str, Any]:
        kwargs = self.base.generation_kwargs(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
        )
        if self.synced_gpus:
            kwargs["synced_gpus"] = True
        return kwargs

    @property
    def max_new_tokens(self) -> int:
        return self.base.max_new_tokens


@dataclass(frozen=True, slots=True)
class EvaluationEnvelope:
    """A collated M3D batch plus references omitted from ``M3DBatch``."""

    batch: M3DBatch
    questions: tuple[str, ...]
    answers: tuple[str, ...]
    metadata: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        size = self.batch.batch_size
        if len(self.questions) != size or len(self.answers) != size:
            raise EvaluationInputError(
                "EvaluationEnvelope question/answer counts must match batch size."
            )
        if len(self.metadata) != size:
            raise EvaluationInputError(
                "EvaluationEnvelope metadata count must match batch size."
            )


class EvaluationEnvelopeCollator:
    """Preserve natural-language references while reusing the production collator."""

    def __init__(self, collator: "M3DCollator") -> None:
        self.collator = collator

    def __call__(self, samples: Sequence[M3DSample]) -> EvaluationEnvelope:
        items = tuple(samples)
        if not items:
            raise EvaluationInputError("Cannot collate an empty evaluation batch.")
        batch = self.collator(items)
        return EvaluationEnvelope(
            batch=batch,
            questions=tuple(item.question for item in items),
            answers=tuple(item.answer for item in items),
            metadata=tuple(dict(item.provenance.metadata) for item in items),
        )


class ExactSubsetSampler(Sampler[int]):
    """Exact rank-strided sampler over the first ``limit`` global records."""

    def __init__(
        self,
        dataset_size: int,
        *,
        rank: int,
        world_size: int,
        limit: int | None,
    ) -> None:
        if dataset_size < 0:
            raise EvaluationInputError("dataset_size cannot be negative.")
        if world_size <= 0 or not 0 <= rank < world_size:
            raise EvaluationInputError("Invalid rank/world_size for evaluation sampler.")
        effective = dataset_size if limit is None else min(dataset_size, int(limit))
        if effective < 0:
            raise EvaluationInputError("max_samples_per_task cannot be negative.")
        self.dataset_size = dataset_size
        self.effective_size = effective
        self.rank = rank
        self.world_size = world_size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.effective_size, self.world_size))

    def __len__(self) -> int:
        if self.rank >= self.effective_size:
            return 0
        return math.ceil((self.effective_size - self.rank) / self.world_size)


@dataclass(frozen=True, slots=True)
class TaskExecutionPlan:
    """Number of real and padded evaluation forwards on one rank."""

    local_real_batches: int
    all_rank_real_batches: tuple[int, ...]
    total_forward_steps: int
    padded_forward_steps: int
    collective_forward: bool

    @classmethod
    def build(
        cls,
        *,
        local_batches: int,
        all_rank_batches: Sequence[int],
        collective_forward: bool,
    ) -> "TaskExecutionPlan":
        counts = tuple(int(value) for value in all_rank_batches)
        if not counts or any(value < 0 for value in counts):
            raise EvaluationInputError("Invalid per-rank evaluation batch counts.")
        if int(local_batches) not in counts:
            # Duplicate counts are fine, but the current count must occur.
            raise EvaluationInputError("Current rank batch count is absent from plan.")
        total = max(counts) if collective_forward else int(local_batches)
        return cls(
            local_real_batches=int(local_batches),
            all_rank_real_batches=counts,
            total_forward_steps=total,
            padded_forward_steps=max(0, total - int(local_batches)),
            collective_forward=bool(collective_forward),
        )


@dataclass(frozen=True, slots=True)
class RetrievalMetrics:
    sample_count: int
    image_to_text: Mapping[str, float]
    text_to_image: Mapping[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_count": self.sample_count,
            "image_to_text": dict(self.image_to_text),
            "text_to_image": dict(self.text_to_image),
        }


@dataclass(frozen=True, slots=True)
class TaskEvaluationSummary:
    task: str
    global_sample_count: int
    metrics: Mapping[str, Any]
    rank_batch_counts: tuple[int, ...]
    padded_forward_steps_by_rank: tuple[int, ...]
    prediction_file: str
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    state_version: int
    source: str
    source_path: str
    split: str
    strategy: str
    world_size: int
    tasks: tuple[TaskEvaluationSummary, ...]
    retrieval: Mapping[str, Any] | None
    model_build: Mapping[str, Any]
    data_pipeline: Mapping[str, Any]
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_version": self.state_version,
            "source": self.source,
            "source_path": self.source_path,
            "split": self.split,
            "strategy": self.strategy,
            "world_size": self.world_size,
            "tasks": [item.to_dict() for item in self.tasks],
            "retrieval": None if self.retrieval is None else dict(self.retrieval),
            "model_build": dict(self.model_build),
            "data_pipeline": dict(self.data_pipeline),
            "elapsed_seconds": self.elapsed_seconds,
        }


# ---------------------------------------------------------------------------
# Small deterministic file helpers
# ---------------------------------------------------------------------------


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{uuid.uuid4().hex}")
    count = 0
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(_canonical_json(row))
                handle.write("\n")
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return count


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            value = json.loads(stripped)
            if not isinstance(value, dict):
                raise EvaluationError(
                    f"{path}:{line_number} must contain a JSON object."
                )
            rows.append(value)
    return rows


# ---------------------------------------------------------------------------
# Text metrics (dependency-light core; optional expensive BERTScore)
# ---------------------------------------------------------------------------


def normalize_answer(text: str) -> str:
    """Lowercase and remove punctuation, English articles and extra spaces."""

    value = str(text).lower()
    value = "".join(character for character in value if character not in string.punctuation)
    value = re.sub(r"\b(a|an|the)\b", " ", value)
    return " ".join(value.split())


def exact_match(prediction: str, reference: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(reference))


def token_f1(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    common = Counter(predicted) & Counter(expected)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2.0 * precision * recall / (precision + recall)


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = [0] * (len(right) + 1)
    for left_token in left:
        current = [0]
        for column, right_token in enumerate(right, start=1):
            if left_token == right_token:
                current.append(previous[column - 1] + 1)
            else:
                current.append(max(previous[column], current[-1]))
        previous = current
    return previous[-1]


def rouge_l_f1(prediction: str, reference: str) -> float:
    predicted = normalize_answer(prediction).split()
    expected = normalize_answer(reference).split()
    if not predicted and not expected:
        return 1.0
    if not predicted or not expected:
        return 0.0
    overlap = _lcs_length(predicted, expected)
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(expected)
    return 2.0 * precision * recall / (precision + recall)


def _corpus_bleu(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    max_order: int,
) -> float | None:
    if not predictions:
        return None
    try:
        import sacrebleu
    except ImportError:
        return None
    # sacrebleu expects references grouped by reference stream.
    metric = sacrebleu.metrics.BLEU(
        smooth_method="exp",
        effective_order=True,
        max_ngram_order=int(max_order),
    ).corpus_score(
        list(predictions),
        [list(references)],
    )
    return float(metric.score / 100.0)


def _meteor_mean(predictions: Sequence[str], references: Sequence[str]) -> float | None:
    if not predictions:
        return None
    try:
        from nltk.translate.meteor_score import meteor_score
    except ImportError:
        return None
    values: list[float] = []
    try:
        for prediction, reference in zip(predictions, references, strict=True):
            values.append(
                float(
                    meteor_score(
                        [normalize_answer(reference).split()],
                        normalize_answer(prediction).split(),
                    )
                )
            )
    except LookupError:
        # NLTK WordNet data is not guaranteed to be available on compute nodes.
        return None
    return float(sum(values) / len(values))


def _bertscore_mean(
    predictions: Sequence[str],
    references: Sequence[str],
    *,
    model_type: str | None,
    device: str,
) -> float | None:
    if not predictions:
        return None
    try:
        from bert_score import score as bert_score
    except ImportError:
        return None
    _, _, f1 = bert_score(
        list(predictions),
        list(references),
        lang="en" if model_type is None else None,
        model_type=model_type,
        device=device,
        verbose=False,
        rescale_with_baseline=False,
    )
    return float(f1.float().mean().item())


def compute_text_metrics(
    rows: Sequence[Mapping[str, Any]],
    *,
    include_caption_metrics: bool,
    include_bertscore: bool,
    bertscore_model: str | None,
    bertscore_device: str,
) -> dict[str, Any]:
    predictions = [str(row["prediction"]) for row in rows]
    references = [str(row["reference"]) for row in rows]
    count = len(rows)
    metrics: dict[str, Any] = {
        "sample_count": count,
        "exact_match": (
            None
            if count == 0
            else float(
                sum(exact_match(pred, ref) for pred, ref in zip(predictions, references, strict=True))
                / count
            )
        ),
        "token_f1": (
            None
            if count == 0
            else float(
                sum(token_f1(pred, ref) for pred, ref in zip(predictions, references, strict=True))
                / count
            )
        ),
        "rouge_l_f1": (
            None
            if count == 0
            else float(
                sum(rouge_l_f1(pred, ref) for pred, ref in zip(predictions, references, strict=True))
                / count
            )
        ),
    }
    if include_caption_metrics:
        metrics.update(
            {
                "bleu_1": _corpus_bleu(predictions, references, max_order=1),
                "bleu_4": _corpus_bleu(predictions, references, max_order=4),
                "meteor": _meteor_mean(predictions, references),
                "bertscore_f1": (
                    _bertscore_mean(
                        predictions,
                        references,
                        model_type=bertscore_model,
                        device=bertscore_device,
                    )
                    if include_bertscore
                    else None
                ),
            }
        )
    return metrics


# ---------------------------------------------------------------------------
# Positioning metrics
# ---------------------------------------------------------------------------


def parse_box(text: str) -> tuple[float, float, float, float, float, float] | None:
    """Parse one normalized six-value M3D box from generated text."""

    match = _BOX_PATTERN.search(str(text))
    candidate = match.group(1) if match is not None else str(text)
    values: Sequence[Any]
    try:
        parsed = ast.literal_eval(candidate)
        values = parsed if isinstance(parsed, (list, tuple)) else ()
    except (SyntaxError, ValueError):
        values = ()
    if len(values) != 6:
        numeric = _FLOAT_PATTERN.findall(candidate)
        if len(numeric) != 6:
            return None
        values = numeric
    try:
        box = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None
    if len(box) != 6 or not all(math.isfinite(value) for value in box):
        return None
    x1, y1, z1, x2, y2, z2 = box
    if not (x1 <= x2 and y1 <= y2 and z1 <= z2):
        return None
    return cast(tuple[float, float, float, float, float, float], box)


def box_iou_3d(
    prediction: Sequence[float],
    reference: Sequence[float],
) -> float:
    if len(prediction) != 6 or len(reference) != 6:
        raise EvaluationInputError("3-D boxes must contain six values.")
    p1 = np.asarray(prediction[:3], dtype=np.float64)
    p2 = np.asarray(prediction[3:], dtype=np.float64)
    r1 = np.asarray(reference[:3], dtype=np.float64)
    r2 = np.asarray(reference[3:], dtype=np.float64)
    intersection_extent = np.maximum(0.0, np.minimum(p2, r2) - np.maximum(p1, r1))
    intersection = float(np.prod(intersection_extent))
    p_volume = float(np.prod(np.maximum(0.0, p2 - p1)))
    r_volume = float(np.prod(np.maximum(0.0, r2 - r1)))
    union = p_volume + r_volume - intersection
    if union <= 0:
        return 1.0 if p_volume == 0 and r_volume == 0 else 0.0
    return intersection / union


def _box_center_distance(prediction: Sequence[float], reference: Sequence[float]) -> float:
    p = (np.asarray(prediction[:3]) + np.asarray(prediction[3:])) / 2.0
    r = (np.asarray(reference[:3]) + np.asarray(reference[3:])) / 2.0
    return float(np.linalg.norm(p - r))


def compute_positioning_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rec_rows = [row for row in rows if row.get("reference_box") is not None]
    reg_rows = [row for row in rows if row.get("reference_box") is None]
    parsed = [row for row in rec_rows if row.get("prediction_box") is not None]
    ious = [float(row["box_iou_3d"]) for row in parsed]
    distances = [float(row["box_center_distance"]) for row in parsed]
    metrics: dict[str, Any] = {
        "sample_count": len(rows),
        "rec_sample_count": len(rec_rows),
        "reg_sample_count": len(reg_rows),
        "box_parse_rate": None if not rec_rows else len(parsed) / len(rec_rows),
        "mean_box_iou_3d": None if not parsed else float(sum(ious) / len(ious)),
        "box_iou_at_0_25": (
            None if not rec_rows else sum(value >= 0.25 for value in ious) / len(rec_rows)
        ),
        "box_iou_at_0_50": (
            None if not rec_rows else sum(value >= 0.50 for value in ious) / len(rec_rows)
        ),
        "mean_normalized_center_distance": (
            None if not parsed else float(sum(distances) / len(distances))
        ),
    }
    if reg_rows:
        metrics["reg_text"] = compute_text_metrics(
            reg_rows,
            include_caption_metrics=False,
            include_bertscore=False,
            bertscore_model=None,
            bertscore_device="cpu",
        )
    return metrics


# ---------------------------------------------------------------------------
# Segmentation metrics
# ---------------------------------------------------------------------------


def segmentation_case_metrics(
    prediction_mask: Tensor,
    target_mask: Tensor,
    *,
    probability: Tensor | None,
) -> dict[str, Any]:
    prediction = prediction_mask.to(dtype=torch.bool).reshape(-1)
    target = target_mask.to(dtype=torch.bool).reshape(-1)
    if prediction.numel() != target.numel():
        raise EvaluationInputError("Prediction and target masks have different sizes.")
    tp = int(torch.logical_and(prediction, target).sum().item())
    fp = int(torch.logical_and(prediction, ~target).sum().item())
    fn = int(torch.logical_and(~prediction, target).sum().item())
    tn = int(torch.logical_and(~prediction, ~target).sum().item())
    target_positive = tp + fn
    prediction_positive = tp + fp
    hard_dice_denom = 2 * tp + fp + fn
    union = tp + fp + fn
    hard_dice = 1.0 if hard_dice_denom == 0 else (2.0 * tp) / hard_dice_denom
    hard_iou = 1.0 if union == 0 else tp / union
    precision = 1.0 if prediction_positive == 0 and target_positive == 0 else (
        0.0 if prediction_positive == 0 else tp / prediction_positive
    )
    recall = 1.0 if target_positive == 0 and prediction_positive == 0 else (
        0.0 if target_positive == 0 else tp / target_positive
    )
    specificity_denom = tn + fp
    specificity = 1.0 if specificity_denom == 0 else tn / specificity_denom

    legacy_soft_dice: float | None = None
    if probability is not None:
        prob = probability.float().reshape(-1)
        target_float = target.float()
        numerator = 2.0 * torch.sum(prob * target_float)
        denominator = torch.sum(prob) + torch.sum(target_float) + 1.0
        legacy_soft_dice = float((numerator / denominator).item())

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "target_foreground_voxels": target_positive,
        "prediction_foreground_voxels": prediction_positive,
        "target_empty": target_positive == 0,
        "prediction_empty": prediction_positive == 0,
        "dice_hard": hard_dice,
        "iou_hard": hard_iou,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "dice_legacy_soft": legacy_soft_dice,
    }


def compute_segmentation_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"sample_count": 0}

    def mean(name: str) -> float | None:
        values = [row.get(name) for row in rows if row.get(name) is not None]
        return None if not values else float(sum(float(value) for value in values) / len(values))

    target_empty = [row for row in rows if bool(row["target_empty"])]
    target_nonempty = [row for row in rows if not bool(row["target_empty"])]
    triggered = [row for row in rows if bool(row["segmentation_triggered"])]
    return {
        "sample_count": len(rows),
        "segmentation_trigger_rate": len(triggered) / len(rows),
        "dice_hard": mean("dice_hard"),
        "iou_hard": mean("iou_hard"),
        "precision": mean("precision"),
        "recall": mean("recall"),
        "specificity": mean("specificity"),
        "dice_legacy_soft": mean("dice_legacy_soft"),
        "empty_target_count": len(target_empty),
        "nonempty_target_count": len(target_nonempty),
        "empty_target_correct_empty_rate": (
            None
            if not target_empty
            else sum(bool(row["prediction_empty"]) for row in target_empty)
            / len(target_empty)
        ),
        "nonempty_target_dice_hard": (
            None
            if not target_nonempty
            else float(
                sum(float(row["dice_hard"]) for row in target_nonempty)
                / len(target_nonempty)
            )
        ),
    }


# ---------------------------------------------------------------------------
# Retrieval metrics from precomputed M3D-CLIP features
# ---------------------------------------------------------------------------


def _load_feature_matrix(path: Path) -> Tensor:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        value = torch.from_numpy(np.load(path, allow_pickle=False))
    elif suffix in {".pt", ".pth"}:
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        if isinstance(loaded, Mapping):
            tensor_values = [item for item in loaded.values() if isinstance(item, Tensor)]
            if len(tensor_values) != 1:
                raise EvaluationInputError(
                    f"Feature checkpoint {path} must contain exactly one tensor."
                )
            value = tensor_values[0]
        elif isinstance(loaded, Tensor):
            value = loaded
        else:
            raise EvaluationInputError(f"Unsupported feature payload in {path}.")
    else:
        raise EvaluationInputError(
            f"Feature matrix {path} must be .npy, .pt or .pth."
        )
    if value.ndim != 2 or not value.is_floating_point():
        raise EvaluationInputError(
            f"Feature matrix must be floating [N,C], got {tuple(value.shape)}/{value.dtype}."
        )
    if not torch.isfinite(value).all():
        raise EvaluationInputError(f"Feature matrix {path} contains NaN or Inf.")
    return value.float().contiguous()


def _recall_at_k(similarity: Tensor, k: int) -> float:
    if similarity.ndim != 2 or similarity.shape[0] != similarity.shape[1]:
        raise EvaluationInputError("Retrieval similarity matrix must be square [N,N].")
    sample_count = int(similarity.shape[0])
    if sample_count == 0:
        return float("nan")
    safe_k = min(max(1, int(k)), sample_count)
    indices = similarity.topk(safe_k, dim=1, largest=True, sorted=False).indices
    targets = torch.arange(sample_count).unsqueeze(1)
    return float(indices.eq(targets).any(dim=1).float().mean().item())


def compute_retrieval_metrics(
    image_features: Tensor,
    text_features: Tensor,
    *,
    ks: Sequence[int] = (1, 5, 10),
    normalise: bool = True,
) -> RetrievalMetrics:
    if image_features.ndim != 2 or text_features.ndim != 2:
        raise EvaluationInputError("Retrieval features must be [N,C].")
    if tuple(image_features.shape) != tuple(text_features.shape):
        raise EvaluationInputError(
            "Image/text retrieval feature shapes differ: "
            f"{tuple(image_features.shape)} vs {tuple(text_features.shape)}."
        )
    if normalise:
        image_features = torch.nn.functional.normalize(image_features.float(), dim=-1)
        text_features = torch.nn.functional.normalize(text_features.float(), dim=-1)
    similarity = image_features @ text_features.transpose(0, 1)
    unique_ks = tuple(sorted(set(int(value) for value in ks if int(value) > 0)))
    if not unique_ks:
        raise EvaluationInputError("At least one positive retrieval K is required.")
    image_to_text = {f"recall_at_{k}": _recall_at_k(similarity, k) for k in unique_ks}
    text_to_image = {
        f"recall_at_{k}": _recall_at_k(similarity.transpose(0, 1), k)
        for k in unique_ks
    }
    return RetrievalMetrics(
        sample_count=int(similarity.shape[0]),
        image_to_text=image_to_text,
        text_to_image=text_to_image,
    )


# ---------------------------------------------------------------------------
# Model/runtime construction
# ---------------------------------------------------------------------------


def _runtime_config(
    config_path: Path,
    *,
    strategy: str,
    overrides: Sequence[str],
    verify_paths: bool,
) -> ExperimentConfig:
    merged = list(overrides)
    merged.extend(
        [
            f"distributed.strategy={strategy}",
            "optimization.checkpoint_language_model=false",
            "optimization.checkpoint_main_vision=false",
            "optimization.checkpoint_seg_vision=false",
            "optimization.checkpoint_segmentation_decoder=false",
            "model.main_vision.activation_checkpoint_every_n_layers=0",
            "model.seg_vision.activation_checkpoint_every_n_layers=0",
        ]
    )
    return load_config(
        config_path,
        overrides=merged,
        resolve_paths=True,
        verify_paths=verify_paths,
    )


def _build_checkpoint_engine(
    *,
    config: ExperimentConfig,
    runtime: RuntimeContext,
    checkpoint_path: Path,
    cache_dir: Path | None,
    local_files_only: bool,
) -> tuple["M3DInferenceEngine", "DistributedM3DModel", Mapping[str, Any]]:
    from .distributed import build_model_synchronously, prepare_distributed_model
    from .export import load_model_only_checkpoint
    from .inference import InferenceBuildReport, M3DInferenceEngine
    from .model.m3d import M3DBuildReport, M3DModel, build_m3d_model
    from .tokenization import build_tokenizer

    with runtime.main_process_first():
        tokenizer_bundle = build_tokenizer(
            config,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    runtime.assert_all_ranks_equal(
        tokenizer_bundle.metadata.to_dict(),
        label="evaluation tokenizer metadata",
    )

    model, model_report = build_model_synchronously(
        runtime,
        lambda: build_m3d_model(
            config,
            tokenizer_bundle,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            torch_dtype=torch.bfloat16,
            load_pretrained_components=False,
            strict_pretrained=True,
        ),
    )
    if not isinstance(model, M3DModel) or not isinstance(model_report, M3DBuildReport):
        raise EvaluationCompatibilityError("M3D builder returned an invalid result.")
    # Keep requires_grad flags identical to training until DDP/FSDP2 has been
    # constructed. DDP rejects a module with no trainable parameters. Evaluation
    # itself runs under torch.inference_mode(), so no gradient graph is created.
    model.eval()
    summary = model.parameter_summary()
    distributed_model, distributed_report = prepare_distributed_model(model, runtime)
    load_model_only_checkpoint(distributed_model, checkpoint_path)
    distributed_model.unwrapped_model.eval()

    build_report = InferenceBuildReport(
        state_version=_EVALUATION_STATE_VERSION,
        export_directory=str(checkpoint_path),
        device=str(runtime.device),
        model_build=model_report.to_dict(),
        state_load={
            "mode": "distributed_checkpoint_model_only",
            "checkpoint": str(checkpoint_path),
            "distributed": distributed_report.to_dict(),
        },
        tokenizer_metadata=tokenizer_bundle.metadata.to_dict(),
        total_parameter_count=int(summary.total),
        main_image_encoder_parameter_count=int(summary.main_vision),
        segmentation_image_encoder_parameter_count=(
            0
            if model.segmentation_image_encoder is None
            else sum(int(parameter.numel()) for parameter in model.segmentation_image_encoder.parameters())
        ),
        shared_image_encoder_parameter_count=int(summary.shared_image_encoder_parameters),
        shared_image_encoder_storage_count=int(summary.shared_image_encoder_storages),
    )
    engine = M3DInferenceEngine(
        config=config,
        tokenizer_bundle=tokenizer_bundle,
        model=distributed_model.unwrapped_model,
        device=runtime.device,
        export_directory=checkpoint_path,
        build_report=build_report,
    )
    return engine, distributed_model, {
        "mode": "checkpoint",
        "model": model_report.to_dict(),
        "distributed": distributed_report.to_dict(),
    }


def _build_export_engine(
    *,
    export_dir: Path,
    runtime: RuntimeContext,
    cache_dir: Path | None,
    local_files_only: bool,
    verify_shard_hashes: bool,
) -> tuple["M3DInferenceEngine", None, Mapping[str, Any]]:
    from .inference import M3DInferenceEngine

    engine = M3DInferenceEngine.from_export(
        export_dir,
        device=runtime.device,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        verify_shard_hashes=verify_shard_hashes,
        allow_cpu=False,
    )
    runtime.assert_all_ranks_equal(
        engine.tokenizer_bundle.metadata.to_dict(),
        label="portable export tokenizer metadata",
    )
    return engine, None, {
        "mode": "portable_export_replicated",
        "inference": engine.build_report.to_dict(),
    }


# ---------------------------------------------------------------------------
# DataLoader construction preserving question/answer strings
# ---------------------------------------------------------------------------


def _build_envelope_loader(
    *,
    pipeline: "EvaluationDataPipeline",
    task: TaskName,
    config: ExperimentConfig,
    tokenizer_bundle: "TokenizerBundle",
    runtime: RuntimeContext,
    batch_size: int,
    max_samples: int | None,
) -> tuple[DataLoader[EvaluationEnvelope], int, int]:
    from .data.collator import build_evaluation_collator

    dataset = pipeline.datasets[task]
    sampler = ExactSubsetSampler(
        len(dataset),
        rank=runtime.rank,
        world_size=runtime.world_size,
        limit=max_samples,
    )
    base_collator = build_evaluation_collator(
        config=config,
        tokenizer_bundle=tokenizer_bundle,
        expected_batch_size=None,
    )
    collator = EvaluationEnvelopeCollator(base_collator)
    kwargs = pipeline.workers.dataloader_kwargs()
    kwargs.update(
        {
            "generator": make_dataloader_generator(runtime, stream=100 + int(task.task_id)),
            "in_order": True,
        }
    )
    loader: DataLoader[EvaluationEnvelope] = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        drop_last=False,
        collate_fn=collator,
        **kwargs,
    )
    return loader, sampler.effective_size, len(sampler)


def _dummy_envelope(
    *,
    pipeline: "EvaluationDataPipeline",
    task: TaskName,
    config: ExperimentConfig,
    tokenizer_bundle: "TokenizerBundle",
) -> EvaluationEnvelope:
    from .data.collator import build_evaluation_collator

    dataset = pipeline.datasets[task]
    if len(dataset) == 0:
        raise EvaluationError(
            f"Cannot create a collective padding batch for empty task {task.value}."
        )
    sample = dataset[0]
    collator = EvaluationEnvelopeCollator(
        build_evaluation_collator(
            config=config,
            tokenizer_bundle=tokenizer_bundle,
            expected_batch_size=None,
        )
    )
    return collator([sample])


# ---------------------------------------------------------------------------
# Prediction row construction
# ---------------------------------------------------------------------------


def _source_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "prompt_variant",
        "rendered_class_name",
        "rendered_description",
        "mask_has_foreground",
        "normalized_box",
        "dataset_tag",
        "class_id",
    )
    return {key: metadata.get(key) for key in keys if key in metadata}


def _rows_from_prediction(
    envelope: EvaluationEnvelope,
    result: "BatchInferenceResult",
    *,
    task: TaskName,
    segmentation_token: str,
) -> list[dict[str, Any]]:
    if len(result.predictions) != envelope.batch.batch_size:
        raise EvaluationError("Inference result batch size differs from input envelope.")
    rows: list[dict[str, Any]] = []
    targets = envelope.batch.segmentation_targets
    for index, prediction in enumerate(result.predictions):
        metadata = envelope.metadata[index]
        raw_reference = envelope.answers[index]
        clean_reference = (
            " ".join(raw_reference.replace(segmentation_token, " ").split())
            if task is TaskName.SEGMENTATION
            else raw_reference
        )
        row: dict[str, Any] = {
            "sample_id": envelope.batch.sample_ids[index],
            "task": task.value,
            "question": envelope.questions[index],
            "reference": clean_reference,
            "raw_reference": raw_reference,
            "prediction": prediction.answer,
            "raw_prediction": prediction.raw_answer,
            "generated_token_ids": list(prediction.generated_token_ids),
            "generated_segmentation_token_count": prediction.generated_segmentation_token_count,
            "stop_reason": prediction.stop_reason,
            "source": _source_metadata(metadata),
        }
        if task is TaskName.POSITIONING:
            reference_box_raw = metadata.get("normalized_box")
            reference_box = (
                None
                if reference_box_raw is None
                else tuple(float(value) for value in cast(Sequence[Any], reference_box_raw))
            )
            prediction_box = parse_box(prediction.raw_answer)
            row["reference_box"] = None if reference_box is None else list(reference_box)
            row["prediction_box"] = None if prediction_box is None else list(prediction_box)
            if reference_box is not None and prediction_box is not None:
                row["box_iou_3d"] = box_iou_3d(prediction_box, reference_box)
                row["box_center_distance"] = _box_center_distance(prediction_box, reference_box)
            else:
                row["box_iou_3d"] = None
                row["box_center_distance"] = None
        elif task is TaskName.SEGMENTATION:
            if targets is None:
                raise EvaluationError("Segmentation envelope lacks target masks.")
            target = targets[index].detach().cpu()
            triggered = prediction.segmentation_mask is not None
            pred_mask = (
                torch.zeros_like(target, dtype=torch.uint8)
                if prediction.segmentation_mask is None
                else prediction.segmentation_mask.detach().cpu().to(torch.uint8)
            )
            probability = (
                torch.zeros_like(target, dtype=torch.float32)
                if prediction.segmentation_probability is None
                else prediction.segmentation_probability.detach().cpu().float()
            )
            row["segmentation_triggered"] = triggered
            row.update(
                segmentation_case_metrics(
                    pred_mask,
                    target,
                    probability=probability,
                )
            )
            row["iou_prediction"] = (
                None
                if prediction.iou_prediction is None
                else [float(value) for value in prediction.iou_prediction.flatten().tolist()]
            )
        rows.append(row)
    return rows


def _metric_for_task(
    task: TaskName,
    rows: Sequence[Mapping[str, Any]],
    *,
    include_bertscore: bool,
    bertscore_model: str | None,
    bertscore_device: str,
) -> dict[str, Any]:
    if task is TaskName.SEGMENTATION:
        text = compute_text_metrics(
            rows,
            include_caption_metrics=False,
            include_bertscore=False,
            bertscore_model=None,
            bertscore_device="cpu",
        )
        return {"text": text, "segmentation": compute_segmentation_metrics(rows)}
    if task is TaskName.POSITIONING:
        return compute_positioning_metrics(rows)
    return compute_text_metrics(
        rows,
        include_caption_metrics=(task is TaskName.CAPTION),
        include_bertscore=(include_bertscore and task is TaskName.CAPTION),
        bertscore_model=bertscore_model,
        bertscore_device=bertscore_device,
    )


# ---------------------------------------------------------------------------
# Task loop and rank-local row merging
# ---------------------------------------------------------------------------


def _task_inference_mode(task: TaskName) -> str:
    # AUTO permits end-to-end measurement of whether [SEG] was generated. A
    # missing trigger is scored as an empty mask instead of aborting the run.
    return "auto" if task is TaskName.SEGMENTATION else "text"


def _merge_rank_rows(
    *,
    output_dir: Path,
    task: TaskName,
    world_size: int,
    expected_count: int,
) -> tuple[list[dict[str, Any]], Path]:
    rows: list[dict[str, Any]] = []
    for rank in range(world_size):
        rank_path = output_dir / "rank_rows" / task.value / f"rank-{rank:05d}.jsonl"
        if not rank_path.is_file():
            raise EvaluationError(f"Missing rank prediction file: {rank_path}")
        rows.extend(_read_jsonl(rank_path))
    sample_ids = [str(row.get("sample_id")) for row in rows]
    if len(sample_ids) != len(set(sample_ids)):
        duplicates = [item for item, count in Counter(sample_ids).items() if count > 1]
        raise EvaluationError(
            f"Evaluation produced duplicate samples for {task.value}: {duplicates[:20]}"
        )
    if len(rows) != expected_count:
        raise EvaluationError(
            f"Evaluation sample count mismatch for {task.value}: "
            f"observed={len(rows)}, expected={expected_count}."
        )
    rows.sort(key=lambda row: str(row["sample_id"]))
    merged_path = output_dir / "predictions" / f"{task.value}.jsonl"
    _write_jsonl(merged_path, rows)
    return rows, merged_path


def evaluate_task(
    *,
    task: TaskName,
    engine: "M3DInferenceEngine",
    pipeline: "EvaluationDataPipeline",
    runtime: RuntimeContext,
    output_dir: Path,
    batch_size: int,
    max_samples: int | None,
    settings: EvaluationGenerationSettings,
    mask_threshold: float,
    collective_forward: bool,
    include_bertscore: bool,
    bertscore_model: str | None,
) -> TaskEvaluationSummary | None:
    started = time.monotonic()
    loader, global_count, local_sample_count = _build_envelope_loader(
        pipeline=pipeline,
        task=task,
        config=engine.config,
        tokenizer_bundle=engine.tokenizer_bundle,
        runtime=runtime,
        batch_size=batch_size,
        max_samples=max_samples,
    )
    local_batches = len(loader)
    all_batches = runtime.all_gather_object(local_batches)
    plan = TaskExecutionPlan.build(
        local_batches=local_batches,
        all_rank_batches=all_batches,
        collective_forward=collective_forward,
    )
    all_padding = tuple(
        max(0, plan.total_forward_steps - int(value)) for value in all_batches
    )
    iterator = iter(loader)
    dummy = (
        _dummy_envelope(
            pipeline=pipeline,
            task=task,
            config=engine.config,
            tokenizer_bundle=engine.tokenizer_bundle,
        )
        if plan.padded_forward_steps > 0
        else None
    )
    local_rows: list[dict[str, Any]] = []
    for step in range(plan.total_forward_steps):
        is_real = step < local_batches
        envelope = next(iterator) if is_real else dummy
        if envelope is None:
            raise EvaluationError("Collective padding plan lacks a dummy envelope.")
        result = engine.predict_tensors(
            envelope.batch.images,
            envelope.questions,
            mode=_task_inference_mode(task),
            generation=settings,
            mask_threshold=mask_threshold,
        )
        if is_real:
            local_rows.extend(
                _rows_from_prediction(
                    envelope,
                    result,
                    task=task,
                    segmentation_token=engine.tokenizer_bundle.metadata.segmentation_token,
                )
            )

    if len(local_rows) != local_sample_count:
        raise EvaluationError(
            f"Rank {runtime.rank} wrote {len(local_rows)} rows for {task.value}, "
            f"expected {local_sample_count}."
        )
    rank_path = output_dir / "rank_rows" / task.value / f"rank-{runtime.rank:05d}.jsonl"
    _write_jsonl(rank_path, local_rows)
    runtime.barrier()

    summary: TaskEvaluationSummary | None = None
    if runtime.is_main_process:
        rows, merged_path = _merge_rank_rows(
            output_dir=output_dir,
            task=task,
            world_size=runtime.world_size,
            expected_count=global_count,
        )
        metrics = _metric_for_task(
            task,
            rows,
            include_bertscore=include_bertscore,
            bertscore_model=bertscore_model,
            bertscore_device=str(runtime.device),
        )
        summary = TaskEvaluationSummary(
            task=task.value,
            global_sample_count=global_count,
            metrics=metrics,
            rank_batch_counts=tuple(int(value) for value in all_batches),
            padded_forward_steps_by_rank=all_padding,
            prediction_file=str(merged_path),
            elapsed_seconds=float(time.monotonic() - started),
        )
        _atomic_write_json(
            output_dir / "metrics" / f"{task.value}.json",
            summary.to_dict(),
        )
    runtime.barrier()
    return summary


# ---------------------------------------------------------------------------
# CLI orchestration
# ---------------------------------------------------------------------------


def _parse_tasks(values: Sequence[str], available: Sequence[TaskName]) -> tuple[TaskName, ...]:
    if not values or values == ["all"]:
        return tuple(available)
    parsed = tuple(TaskName.parse(value) for value in values)
    missing = [task.value for task in parsed if task not in available]
    if missing:
        raise EvaluationInputError(
            f"Requested tasks are absent from this split: {missing}; "
            f"available={[task.value for task in available]}."
        )
    # Preserve canonical task order and remove duplicates.
    selected = set(parsed)
    return tuple(task for task in available if task in selected)


def _prepare_output_directory(path: Path, *, overwrite: bool, runtime: RuntimeContext) -> None:
    if runtime.is_main_process:
        if path.exists():
            if not overwrite:
                raise EvaluationInputError(
                    f"Evaluation output already exists: {path}. Pass --overwrite."
                )
            shutil.rmtree(path)
        path.mkdir(parents=True)
    runtime.barrier()


def _evaluate_retrieval_cli(args: argparse.Namespace) -> Mapping[str, Any] | None:
    if args.retrieval_image_features is None and args.retrieval_text_features is None:
        return None
    if args.retrieval_image_features is None or args.retrieval_text_features is None:
        raise EvaluationInputError(
            "Both --retrieval-image-features and --retrieval-text-features are required."
        )
    image = _load_feature_matrix(args.retrieval_image_features)
    text = _load_feature_matrix(args.retrieval_text_features)
    return compute_retrieval_metrics(
        image,
        text,
        ks=args.retrieval_k,
        normalise=not bool(args.no_retrieval_normalise),
    ).to_dict()


def run_evaluation(args: argparse.Namespace) -> Mapping[str, Any]:
    started = time.monotonic()
    source = EvaluationSource.parse(args.source)
    strategy = str(args.strategy)
    if source is EvaluationSource.EXPORT and strategy == "fsdp2":
        raise EvaluationInputError(
            "Portable export evaluation uses one complete model replica per GPU. "
            "Use --strategy ddp, or evaluate the training checkpoint directly "
            "with --source checkpoint --strategy fsdp2."
        )
    config = _runtime_config(
        args.config,
        strategy=strategy,
        overrides=args.override,
        verify_paths=not bool(args.skip_path_verification),
    )
    runtime = initialize_runtime(config, verbose_all_ranks=bool(args.verbose_all_ranks))
    distributed_model: "DistributedM3DModel | None" = None
    try:
        output_dir = args.output_dir.expanduser().resolve()
        _prepare_output_directory(output_dir, overwrite=bool(args.overwrite), runtime=runtime)

        if source is EvaluationSource.EXPORT:
            if args.export_dir is None:
                raise EvaluationInputError("--export-dir is required for --source export.")
            source_path = args.export_dir.expanduser().resolve()
            engine, distributed_model, model_build = _build_export_engine(
                export_dir=source_path,
                runtime=runtime,
                cache_dir=args.cache_dir,
                local_files_only=bool(args.local_files_only),
                verify_shard_hashes=not bool(args.skip_shard_hash_verification),
            )
            # Use the CLI config for data paths/workers and the export config
            # retained by the engine for model-shape validation. Any incompatible
            # image/token contract fails before inference rather than being hidden.
            eval_config = config
            collective_forward = False
        else:
            if args.checkpoint is None:
                raise EvaluationInputError(
                    "--checkpoint is required for --source checkpoint."
                )
            from .checkpointing import resolve_checkpoint_path

            source_path = resolve_checkpoint_path(args.checkpoint)
            engine, distributed_model, model_build = _build_checkpoint_engine(
                config=config,
                runtime=runtime,
                checkpoint_path=source_path,
                cache_dir=args.cache_dir,
                local_files_only=bool(args.local_files_only),
            )
            eval_config = config
            collective_forward = strategy == "fsdp2"

        from .data.loader import build_evaluation_data_pipeline
        from .tokenization import M3DTextProcessor

        text_processor = M3DTextProcessor(engine.tokenizer_bundle, eval_config)
        pipeline = build_evaluation_data_pipeline(
            config=eval_config,
            runtime=runtime,
            tokenizer_bundle=engine.tokenizer_bundle,
            text_processor=text_processor,
            split=DataSplit.parse(args.split),
            per_device_batch_size=int(args.batch_size),
        )
        selected_tasks = _parse_tasks(args.task, pipeline.tasks)
        from .inference import GenerationSettings

        settings = EvaluationGenerationSettings(
            base=GenerationSettings(
                max_new_tokens=int(args.max_new_tokens),
                do_sample=bool(args.do_sample),
                temperature=float(args.temperature),
                top_p=float(args.top_p),
                num_beams=int(args.num_beams),
                repetition_penalty=float(args.repetition_penalty),
            ),
            synced_gpus=bool(collective_forward),
        )

        task_summaries: list[TaskEvaluationSummary] = []
        for task in selected_tasks:
            summary = evaluate_task(
                task=task,
                engine=engine,
                pipeline=pipeline,
                runtime=runtime,
                output_dir=output_dir,
                batch_size=int(args.batch_size),
                max_samples=args.max_samples_per_task,
                settings=settings,
                mask_threshold=float(args.mask_threshold),
                collective_forward=collective_forward,
                include_bertscore=bool(args.bertscore),
                bertscore_model=args.bertscore_model,
            )
            if runtime.is_main_process and summary is not None:
                task_summaries.append(summary)

        retrieval: Mapping[str, Any] | None = None
        if runtime.is_main_process:
            retrieval = _evaluate_retrieval_cli(args)
            if retrieval is not None:
                _atomic_write_json(output_dir / "metrics" / "retrieval.json", retrieval)
            report = EvaluationReport(
                state_version=_EVALUATION_STATE_VERSION,
                source=source.value,
                source_path=str(source_path),
                split=DataSplit.parse(args.split).value,
                strategy=(
                    "replicated_export"
                    if source is EvaluationSource.EXPORT
                    else strategy
                ),
                world_size=runtime.world_size,
                tasks=tuple(task_summaries),
                retrieval=retrieval,
                model_build=model_build,
                data_pipeline=dict(pipeline.summary()),
                elapsed_seconds=float(time.monotonic() - started),
            )
            payload = report.to_dict()
            _atomic_write_json(output_dir / "evaluation_report.json", payload)
            _atomic_write_json(
                output_dir / "COMPLETED.json",
                {
                    "status": "complete",
                    "state_version": _EVALUATION_STATE_VERSION,
                    "report": "evaluation_report.json",
                    "task_count": len(task_summaries),
                },
            )
        else:
            payload = {"status": "complete", "rank": runtime.rank}
        runtime.barrier()
        return payload
    finally:
        # Keep distributed_model alive until all FSDP2 collectives are complete.
        del distributed_model
        runtime.close()


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m m3d.evaluate",
        description="Distributed M3D caption/VQA/positioning/segmentation evaluation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, required=False)
    parser.add_argument("--source", choices=[item.value for item in EvaluationSource], default="export")
    parser.add_argument("--export-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--strategy", choices=["ddp", "fsdp2"], default="ddp")
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--output-dir", type=Path, required=False)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples-per-task", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--bertscore", action="store_true")
    parser.add_argument("--bertscore-model", default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-shard-hash-verification", action="store_true")
    parser.add_argument("--skip-path-verification", action="store_true")
    parser.add_argument("--verbose-all-ranks", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("-o", "--override", action="append", default=[])
    parser.add_argument("--retrieval-image-features", type=Path, default=None)
    parser.add_argument("--retrieval-text-features", type=Path, default=None)
    parser.add_argument("--retrieval-k", type=int, action="append", default=[1, 5, 10])
    parser.add_argument("--no-retrieval-normalise", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    if args.self_test:
        return args
    missing = [
        name
        for name, value in (
            ("--config", args.config),
            ("--output-dir", args.output_dir),
        )
        if value is None
    ]
    if missing:
        parser.error(f"required arguments missing: {', '.join(missing)}")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    if args.max_samples_per_task is not None and args.max_samples_per_task <= 0:
        parser.error("--max-samples-per-task must be positive")
    if not 0 <= args.mask_threshold <= 1:
        parser.error("--mask-threshold must be in [0,1]")
    return args


# ---------------------------------------------------------------------------
# Dependency-light self-test
# ---------------------------------------------------------------------------


def _self_test() -> dict[str, Any]:
    text_rows = [
        {"prediction": "The left lung.", "reference": "left lung"},
        {"prediction": "no", "reference": "yes"},
    ]
    text = compute_text_metrics(
        text_rows,
        include_caption_metrics=False,
        include_bertscore=False,
        bertscore_model=None,
        bertscore_device="cpu",
    )
    ref_box = (0.1, 0.2, 0.3, 0.5, 0.7, 0.9)
    parsed = parse_box("Answer <bx_start>[0.1, 0.2, 0.3, 0.5, 0.7, 0.9]<bx_end>")
    invalid_box = parse_box("<bx_start>[0.5,0,0,0.1,1,1]<bx_end>")
    target = torch.tensor([[[[0, 1], [0, 1]]]], dtype=torch.float32)
    prediction = torch.tensor([[[[0, 1], [1, 1]]]], dtype=torch.uint8)
    probability = torch.tensor([[[[0.1, 0.8], [0.6, 0.9]]]], dtype=torch.float32)
    seg = segmentation_case_metrics(prediction, target, probability=probability)

    image_features = torch.eye(3, dtype=torch.float32)
    text_features = torch.eye(3, dtype=torch.float32)
    retrieval = compute_retrieval_metrics(
        image_features,
        text_features,
        ks=(1, 5, 10),
    )
    small_topk_safe = all(
        math.isclose(value, 1.0)
        for value in (
            *retrieval.image_to_text.values(),
            *retrieval.text_to_image.values(),
        )
    )
    plan = TaskExecutionPlan.build(
        local_batches=2,
        all_rank_batches=(2, 3),
        collective_forward=True,
    )
    sampler_rank0 = list(ExactSubsetSampler(7, rank=0, world_size=2, limit=5))
    sampler_rank1 = list(ExactSubsetSampler(7, rank=1, world_size=2, limit=5))

    with tempfile.TemporaryDirectory(prefix="m3d-evaluate-self-test-") as temporary:
        root = Path(temporary)
        rank_dir = root / "rank_rows" / "caption"
        _write_jsonl(rank_dir / "rank-00000.jsonl", [{"sample_id": "a"}])
        _write_jsonl(rank_dir / "rank-00001.jsonl", [{"sample_id": "b"}])
        merged, merged_path = _merge_rank_rows(
            output_dir=root,
            task=TaskName.CAPTION,
            world_size=2,
            expected_count=2,
        )
        merge_ok = [row["sample_id"] for row in merged] == ["a", "b"] and merged_path.is_file()

    checks = {
        "normalised_exact_match": math.isclose(float(text["exact_match"]), 0.5),
        "box_parse": parsed == ref_box,
        "invalid_box_rejected": invalid_box is None,
        "box_identity_iou": parsed is not None and math.isclose(box_iou_3d(parsed, ref_box), 1.0),
        "segmentation_hard_dice": math.isclose(float(seg["dice_hard"]), 0.8),
        "retrieval_small_topk_safe": small_topk_safe,
        "collective_padding_plan": plan.padded_forward_steps == 1,
        "exact_subset_no_duplicates": sorted(sampler_rank0 + sampler_rank1) == list(range(5)),
        "rank_jsonl_merge": merge_ok,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "text_metrics": text,
        "segmentation_metrics": seg,
        "retrieval": retrieval.to_dict(),
        **checks,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.self_test:
        print(json.dumps(_self_test(), ensure_ascii=False, indent=2, sort_keys=True))
        return
    payload = run_evaluation(args)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
