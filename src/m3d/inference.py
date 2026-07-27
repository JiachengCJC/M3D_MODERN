"""Portable single-node inference for M3D-Modernized.

This module consumes the portable export produced by :mod:`m3d.export` rather
than a training checkpoint.  It reconstructs the exact adapter-form M3D
architecture, streams the sharded safetensors weights into that architecture,
and exposes a small inference engine for text generation and generated-token
conditioned 3D segmentation.

The inference graph deliberately preserves M3D's two independent image
encoders::

    image
      -> Main 3D ViT -> MM projector -> Phi-3 generation
                                      -> generated [SEG]
                                      -> replay final sequence through Phi-3
                                      -> segmentation prompt projector

    the same image
      -> independent SegVol 3D ViT -> prompt encoder -> mask decoder

The Main 3D ViT is run once per prediction batch.  Its projected visual tokens
are reused for both autoregressive generation and the final hidden-state replay
needed to recover the state that predicted ``[SEG]``.  The SegVol image encoder
is run only for rows whose generated answer contains ``[SEG]`` (or for every row
when segmentation mode is explicitly required).

Input-volume contract
---------------------
* ``.npy`` input must already follow M3D's ``[C,D,H,W]`` or ``[D,H,W]`` layout.
* ``.nii``/``.nii.gz`` input is canonicalised by the existing volume reader and
  converted from nibabel ``[X,Y,Z]`` to M3D ``[D,H,W]``.
* Volumes must already have the configured spatial shape and intensity range.
  This inference path never silently resizes or min-max normalises a scan.
* Returned mask logits remain raw until sigmoid is explicitly applied.

The CLI intentionally handles one image/question pair.  The Python engine also
supports homogeneous batches and only invokes SegVol for the selected rows.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import hashlib
import json
import math
import os
import random
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Iterator, Mapping, MutableMapping, Sequence, cast

import numpy as np
import torch
from torch import Tensor, nn

from m3d.config import ExperimentConfig, load_config
from m3d.data.io import (
    LoadedVolume,
    LocalCacheOptions,
    NodeLocalFileCache,
    VolumeFormat,
    VolumeReader,
    VolumeReaderOptions,
)
from m3d.data.schema import DataSplit
from m3d.data.transforms import build_volume_transform
from m3d.model.language import LanguageModelOutput
from m3d.model.m3d import M3DBuildReport, M3DModel, build_m3d_model
from m3d.model.segvol import SegVolOutput
if TYPE_CHECKING:
    from m3d.tokenization import EncodedPrompt, TokenizerBundle, TokenizerMetadata


_INFERENCE_STATE_VERSION = 1
_EXPORT_COMPLETION_FILE = "COMPLETED.json"
_EXPORT_MANIFEST_FILE = "export_manifest.json"
_EXPORT_CONFIG_FILE = "resolved_config.json"
_TOKENIZER_METADATA_FILE = "m3d_tokenizer_metadata.json"
_FULL_MODEL_DIRECTORY = "m3d_model"
_FULL_MODEL_BASENAME = "m3d_model"


class InferenceError(RuntimeError):
    """Base error for portable M3D inference."""


class InferenceDependencyError(InferenceError, ImportError):
    """Raised when an optional runtime package is unavailable."""


class InferenceCompatibilityError(InferenceError):
    """Raised when an export cannot strictly reconstruct the requested model."""


class InferenceInputError(InferenceError, ValueError):
    """Raised when an image, prompt, or generation option is invalid."""


class InferenceMode(str, Enum):
    """Whether dense segmentation should be skipped, inferred, or required."""

    AUTO = "auto"
    TEXT = "text"
    SEGMENTATION = "segmentation"

    @classmethod
    def parse(cls, value: str | "InferenceMode") -> "InferenceMode":
        if isinstance(value, cls):
            return value
        normalised = str(value).strip().lower().replace("-", "_")
        aliases = {
            "auto": cls.AUTO,
            "text": cls.TEXT,
            "language": cls.TEXT,
            "seg": cls.SEGMENTATION,
            "mask": cls.SEGMENTATION,
            "segmentation": cls.SEGMENTATION,
        }
        try:
            return aliases[normalised]
        except KeyError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise InferenceInputError(
                f"Unknown inference mode {value!r}; expected one of: {allowed}."
            ) from exc


@dataclass(frozen=True, slots=True)
class GenerationSettings:
    """Reviewed generation controls used by :class:`M3DInferenceEngine`."""

    max_new_tokens: int = 128
    do_sample: bool = False
    temperature: float = 1.0
    top_p: float = 1.0
    num_beams: int = 1
    repetition_penalty: float = 1.0

    def __post_init__(self) -> None:
        if self.max_new_tokens <= 0:
            raise InferenceInputError("max_new_tokens must be positive.")
        if not math.isfinite(self.temperature) or self.temperature <= 0:
            raise InferenceInputError("temperature must be finite and positive.")
        if not math.isfinite(self.top_p) or not 0 < self.top_p <= 1:
            raise InferenceInputError("top_p must be in (0, 1].")
        if self.num_beams <= 0:
            raise InferenceInputError("num_beams must be positive.")
        if (
            not math.isfinite(self.repetition_penalty)
            or self.repetition_penalty <= 0
        ):
            raise InferenceInputError(
                "repetition_penalty must be finite and positive."
            )
        if self.do_sample and self.num_beams != 1:
            raise InferenceInputError(
                "This deterministic M3D inference wrapper does not combine sampling "
                "with beam search. Set num_beams=1 when do_sample=True."
            )

    def generation_kwargs(
        self,
        *,
        pad_token_id: int,
        eos_token_id: int,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "max_new_tokens": int(self.max_new_tokens),
            "do_sample": bool(self.do_sample),
            "num_beams": int(self.num_beams),
            "repetition_penalty": float(self.repetition_penalty),
            "pad_token_id": int(pad_token_id),
            "eos_token_id": int(eos_token_id),
            "return_dict_in_generate": True,
            "output_scores": False,
            "output_hidden_states": False,
            "output_attentions": False,
        }
        # Transformers warns when sampling-only controls are supplied while
        # do_sample=False, so only send them for the sampling path.
        if self.do_sample:
            kwargs["temperature"] = float(self.temperature)
            kwargs["top_p"] = float(self.top_p)
        return kwargs


@dataclass(frozen=True, slots=True)
class ExportStateLoadReport:
    """Strict streaming-load report for the portable model state."""

    export_directory: str
    index_file: str
    shard_count: int
    tensor_count: int
    total_tensor_bytes: int
    verified_shard_hashes: bool
    state_sha256: str | None
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class InferenceBuildReport:
    """Serializable description of one loaded inference engine."""

    state_version: int
    export_directory: str
    device: str
    model_build: Mapping[str, Any]
    state_load: Mapping[str, Any]
    tokenizer_metadata: Mapping[str, Any]
    total_parameter_count: int
    main_image_encoder_parameter_count: int
    segmentation_image_encoder_parameter_count: int
    shared_image_encoder_parameter_count: int
    shared_image_encoder_storage_count: int

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(slots=True)
class InferencePrediction:
    """One generated answer and optional dense segmentation result."""

    question: str
    answer: str
    raw_answer: str
    generated_token_ids: tuple[int, ...]
    generated_segmentation_token_count: int
    stop_reason: str
    source_path: str | None
    source_geometry: Mapping[str, Any] | None
    segmentation_probability: Tensor | None = None
    segmentation_mask: Tensor | None = None
    iou_prediction: Tensor | None = None

    @property
    def has_segmentation(self) -> bool:
        return self.segmentation_mask is not None

    def summary(self) -> dict[str, Any]:
        probability = self.segmentation_probability
        mask = self.segmentation_mask
        return {
            "question": self.question,
            "answer": self.answer,
            "raw_answer": self.raw_answer,
            "generated_token_ids": list(self.generated_token_ids),
            "generated_segmentation_token_count": int(
                self.generated_segmentation_token_count
            ),
            "stop_reason": self.stop_reason,
            "source_path": self.source_path,
            "source_geometry": self.source_geometry,
            "has_segmentation": self.has_segmentation,
            "segmentation_shape": None if mask is None else list(mask.shape),
            "foreground_voxel_count": (
                None if mask is None else int(mask.to(torch.int64).sum().item())
            ),
            "probability_min": (
                None if probability is None else float(probability.min().item())
            ),
            "probability_max": (
                None if probability is None else float(probability.max().item())
            ),
            "probability_mean": (
                None if probability is None else float(probability.mean().item())
            ),
            "iou_prediction": (
                None
                if self.iou_prediction is None
                else [float(value) for value in self.iou_prediction.flatten().tolist()]
            ),
        }


@dataclass(frozen=True, slots=True)
class BatchInferenceResult:
    """Predictions and timing information for one homogeneous image batch."""

    predictions: tuple[InferencePrediction, ...]
    prompt_sequence_length: int
    generated_sequence_length: int
    main_vision_seconds: float
    generation_seconds: float
    segmentation_seconds: float
    total_seconds: float

    def summary(self) -> dict[str, Any]:
        return {
            "state_version": _INFERENCE_STATE_VERSION,
            "status": "complete",
            "batch_size": len(self.predictions),
            "prompt_sequence_length": self.prompt_sequence_length,
            "generated_sequence_length": self.generated_sequence_length,
            "main_vision_seconds": self.main_vision_seconds,
            "generation_seconds": self.generation_seconds,
            "segmentation_seconds": self.segmentation_seconds,
            "total_seconds": self.total_seconds,
            "predictions": [item.summary() for item in self.predictions],
        }


# ---------------------------------------------------------------------------
# Portable export validation and strict streaming load
# ---------------------------------------------------------------------------


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise InferenceCompatibilityError(f"Required file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise InferenceCompatibilityError(f"Invalid JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise InferenceCompatibilityError(
            f"Expected a JSON object in {path}, got {type(payload).__name__}."
        )
    return payload


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _validate_export_directory(export_directory: Path) -> dict[str, Any]:
    export_directory = export_directory.expanduser().resolve()
    if not export_directory.is_dir():
        raise InferenceCompatibilityError(
            f"Portable export directory does not exist: {export_directory}"
        )

    completed_path = export_directory / _EXPORT_COMPLETION_FILE
    manifest_path = export_directory / _EXPORT_MANIFEST_FILE
    completed = _read_json(completed_path)
    manifest = _read_json(manifest_path)
    if completed.get("status") != "complete":
        raise InferenceCompatibilityError(
            f"Export is not marked complete in {completed_path}."
        )
    if manifest.get("status") != "complete":
        raise InferenceCompatibilityError(
            f"Export manifest is not complete in {manifest_path}."
        )
    expected_manifest_hash = completed.get("export_manifest_sha256")
    if isinstance(expected_manifest_hash, str):
        observed = _sha256_file(manifest_path)
        if observed != expected_manifest_hash:
            raise InferenceCompatibilityError(
                "Export manifest SHA-256 does not match COMPLETED.json: "
                f"expected={expected_manifest_hash}, observed={observed}."
            )

    for required in (
        export_directory / _EXPORT_CONFIG_FILE,
        export_directory / "tokenizer" / _TOKENIZER_METADATA_FILE,
        export_directory
        / _FULL_MODEL_DIRECTORY
        / f"{_FULL_MODEL_BASENAME}.safetensors.index.json",
    ):
        if not required.is_file():
            raise InferenceCompatibilityError(
                f"Portable export is missing required file: {required}"
            )
    return manifest


def _import_transformers_tokenizer() -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise InferenceDependencyError(
            "Loading an exported tokenizer requires transformers. Install the "
            "reviewed requirements.txt environment."
        ) from exc
    return AutoTokenizer


def _tokenizer_metadata_from_payload(payload: Mapping[str, Any]) -> "TokenizerMetadata":
    from m3d.tokenization import TokenizerMetadata

    required = {
        "tokenizer_name_or_path",
        "original_vocab_size",
        "vocabulary_size",
        "added_token_count",
        "image_token",
        "image_token_id",
        "segmentation_token",
        "segmentation_token_id",
        "box_start_token",
        "box_start_token_id",
        "box_end_token",
        "box_end_token_id",
        "pad_token_id",
        "eos_token_id",
        "visual_token_count",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise InferenceCompatibilityError(
            f"Tokenizer metadata is missing fields: {missing}."
        )
    values = {name: payload[name] for name in required}
    try:
        return TokenizerMetadata(**values)
    except (TypeError, ValueError) as exc:
        raise InferenceCompatibilityError(
            f"Invalid exported tokenizer metadata: {exc}"
        ) from exc


def load_exported_tokenizer(
    export_directory: str | os.PathLike[str],
    *,
    model_max_length: int,
) -> "TokenizerBundle":
    """Load the exact tokenizer and stable M3D token IDs from an export."""

    from m3d.tokenization import TokenizerBundle

    root = Path(export_directory).expanduser().resolve()
    tokenizer_directory = root / "tokenizer"
    metadata = _tokenizer_metadata_from_payload(
        _read_json(tokenizer_directory / _TOKENIZER_METADATA_FILE)
    )
    AutoTokenizer = _import_transformers_tokenizer()
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_directory),
        local_files_only=True,
        trust_remote_code=True,
        use_fast=True,
        model_max_length=int(model_max_length),
        padding_side="right",
        truncation_side="right",
    )
    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    tokenizer.model_max_length = int(model_max_length)

    if len(tokenizer) != metadata.vocabulary_size:
        raise InferenceCompatibilityError(
            "Exported tokenizer vocabulary differs from metadata: "
            f"tokenizer={len(tokenizer)}, metadata={metadata.vocabulary_size}."
        )
    if tokenizer.pad_token_id != metadata.pad_token_id:
        raise InferenceCompatibilityError(
            "Exported tokenizer pad token ID differs from metadata: "
            f"tokenizer={tokenizer.pad_token_id}, metadata={metadata.pad_token_id}."
        )
    if tokenizer.eos_token_id != metadata.eos_token_id:
        raise InferenceCompatibilityError(
            "Exported tokenizer EOS token ID differs from metadata: "
            f"tokenizer={tokenizer.eos_token_id}, metadata={metadata.eos_token_id}."
        )

    token_contract = (
        (metadata.image_token, metadata.image_token_id, "image"),
        (
            metadata.segmentation_token,
            metadata.segmentation_token_id,
            "segmentation",
        ),
        (metadata.box_start_token, metadata.box_start_token_id, "box-start"),
        (metadata.box_end_token, metadata.box_end_token_id, "box-end"),
    )
    for token, expected_id, label in token_contract:
        observed = tokenizer.convert_tokens_to_ids(token)
        if int(observed) != int(expected_id):
            raise InferenceCompatibilityError(
                f"Exported {label} token ID differs from metadata: "
                f"token={token!r}, tokenizer={observed}, metadata={expected_id}."
            )
        encoded = tokenizer.encode(token, add_special_tokens=False)
        if encoded != [int(expected_id)]:
            raise InferenceCompatibilityError(
                f"Exported {label} token no longer encodes as one token: "
                f"token={token!r}, encoded={encoded}."
            )

    repeated_image_prefix = metadata.image_token * metadata.visual_token_count
    prefix_ids = tokenizer.encode(repeated_image_prefix, add_special_tokens=False)
    if len(prefix_ids) != metadata.visual_token_count or any(
        int(token_id) != metadata.image_token_id for token_id in prefix_ids
    ):
        raise InferenceCompatibilityError(
            "Exported tokenizer no longer produces one image placeholder ID per "
            "projected visual token."
        )
    return TokenizerBundle(tokenizer=tokenizer, metadata=metadata)


def _import_safetensors() -> Any:
    try:
        from safetensors import safe_open
    except ImportError as exc:
        raise InferenceDependencyError(
            "Loading the portable M3D state requires safetensors. Install the "
            "reviewed requirements.txt environment."
        ) from exc
    return safe_open


def _state_target_mapping(model: nn.Module) -> dict[str, Tensor]:
    state = model.state_dict(keep_vars=True)
    targets: dict[str, Tensor] = {}
    for name, value in state.items():
        if not isinstance(value, Tensor):
            raise InferenceCompatibilityError(
                f"Model state entry {name!r} is not a tensor."
            )
        targets[str(name)] = value
    if not targets:
        raise InferenceCompatibilityError("Constructed M3D model has an empty state dict.")
    return targets


def _load_sharded_state_into_module(
    module: nn.Module,
    state_directory: Path,
    *,
    basename: str,
    export_root: Path,
    verify_shard_hashes: bool,
) -> ExportStateLoadReport:
    """Stream one indexed safetensors state into an exact target module."""

    started = time.monotonic()
    index_path = state_directory / f"{basename}.safetensors.index.json"
    index = _read_json(index_path)
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise InferenceCompatibilityError(
            f"Safetensors index has no non-empty weight_map: {index_path}"
        )

    targets = _state_target_mapping(module)
    source_names = {str(name) for name in weight_map}
    target_names = set(targets)
    missing = sorted(target_names.difference(source_names))
    unexpected = sorted(source_names.difference(target_names))
    if missing or unexpected:
        raise InferenceCompatibilityError(
            "Portable state keys do not match the reconstructed module "
            f"{type(module).__name__}: "
            f"missing={missing[:20]}{' ...' if len(missing) > 20 else ''}, "
            f"unexpected={unexpected[:20]}{' ...' if len(unexpected) > 20 else ''}."
        )

    shard_to_names: MutableMapping[str, list[str]] = defaultdict(list)
    for name, filename in weight_map.items():
        if not isinstance(name, str) or not isinstance(filename, str):
            raise InferenceCompatibilityError(
                f"Invalid weight_map entry in {index_path}: {name!r} -> {filename!r}."
            )
        shard_to_names[filename].append(name)

    shard_hashes: dict[str, str] = {}
    shard_descriptors = index.get("shards")
    if isinstance(shard_descriptors, list):
        for item in shard_descriptors:
            if isinstance(item, Mapping) and isinstance(item.get("file"), str):
                sha = item.get("sha256")
                if isinstance(sha, str):
                    shard_hashes[str(item["file"])] = sha

    safe_open = _import_safetensors()
    loaded: set[str] = set()
    total_bytes = 0
    resolved_state_directory = state_directory.resolve()
    with torch.no_grad():
        for filename in sorted(shard_to_names):
            shard_path = (state_directory / filename).resolve()
            try:
                shard_path.relative_to(resolved_state_directory)
            except ValueError as exc:
                raise InferenceCompatibilityError(
                    f"Safetensors shard escapes export directory: {filename!r}."
                ) from exc
            if not shard_path.is_file():
                raise InferenceCompatibilityError(
                    f"Safetensors shard does not exist: {shard_path}"
                )
            if verify_shard_hashes:
                expected_hash = shard_hashes.get(filename)
                if expected_hash is None:
                    raise InferenceCompatibilityError(
                        f"No SHA-256 recorded for shard {filename!r}."
                    )
                observed_hash = _sha256_file(shard_path)
                if observed_hash != expected_hash:
                    raise InferenceCompatibilityError(
                        f"Safetensors shard SHA-256 mismatch for {filename}: "
                        f"expected={expected_hash}, observed={observed_hash}."
                    )

            requested_names = sorted(shard_to_names[filename])
            with safe_open(str(shard_path), framework="pt", device="cpu") as handle:
                actual_names = set(handle.keys())
                requested_set = set(requested_names)
                if actual_names != requested_set:
                    raise InferenceCompatibilityError(
                        f"Shard contents differ from index for {filename}: "
                        f"missing={sorted(requested_set - actual_names)}, "
                        f"unexpected={sorted(actual_names - requested_set)}."
                    )
                for name in requested_names:
                    source = handle.get_tensor(name)
                    target = targets[name]
                    if tuple(source.shape) != tuple(target.shape):
                        raise InferenceCompatibilityError(
                            f"State shape mismatch for {name}: export={tuple(source.shape)}, "
                            f"model={tuple(target.shape)}."
                        )
                    if source.dtype != target.dtype:
                        raise InferenceCompatibilityError(
                            f"State dtype mismatch for {name}: export={source.dtype}, "
                            f"model={target.dtype}."
                        )
                    if name in loaded:
                        raise InferenceCompatibilityError(
                            f"State tensor {name!r} appears in multiple shards."
                        )
                    target.copy_(source)
                    total_bytes += int(source.numel() * source.element_size())
                    loaded.add(name)
                    del source

    if loaded != target_names:
        unresolved = sorted(target_names.difference(loaded))
        raise InferenceCompatibilityError(
            f"Streaming state load did not populate all tensors: {unresolved[:20]}."
        )

    metadata = index.get("metadata")
    state_sha256 = (
        str(metadata.get("state_sha256"))
        if isinstance(metadata, Mapping) and metadata.get("state_sha256") is not None
        else None
    )
    return ExportStateLoadReport(
        export_directory=str(export_root),
        index_file=str(index_path),
        shard_count=len(shard_to_names),
        tensor_count=len(loaded),
        total_tensor_bytes=total_bytes,
        verified_shard_hashes=bool(verify_shard_hashes),
        state_sha256=state_sha256,
        elapsed_seconds=float(time.monotonic() - started),
    )


def load_exported_model_state(
    model: M3DModel,
    export_directory: str | os.PathLike[str],
    *,
    verify_shard_hashes: bool = True,
) -> ExportStateLoadReport:
    """Strictly stream the complete adapter-form M3D state into ``model``."""

    root = Path(export_directory).expanduser().resolve()
    return _load_sharded_state_into_module(
        model,
        root / _FULL_MODEL_DIRECTORY,
        basename=_FULL_MODEL_BASENAME,
        export_root=root,
        verify_shard_hashes=verify_shard_hashes,
    )


def load_exported_components(
    model: M3DModel,
    export_directory: str | os.PathLike[str],
    *,
    verify_shard_hashes: bool = True,
) -> dict[str, ExportStateLoadReport]:
    """Load non-language components for a merged-Phi-3 inference model."""

    root = Path(export_directory).expanduser().resolve()
    components_root = root / "components"
    targets: list[tuple[str, nn.Module | None]] = [
        ("main_vision", model.vision_tower),
        ("multimodal_projector", model.mm_projector),
        ("segmentation_projector", model.seg_projector),
        ("segvol", model.seg_module),
    ]
    reports: dict[str, ExportStateLoadReport] = {}
    for name, module in targets:
        directory = components_root / name
        index_path = directory / f"{name}.safetensors.index.json"
        if module is None:
            if index_path.exists():
                raise InferenceCompatibilityError(
                    f"Export contains component {name!r}, but reconstructed model does not."
                )
            continue
        if not index_path.is_file():
            raise InferenceCompatibilityError(
                f"Merged-language inference requires exported component {name}: {index_path}"
            )
        reports[name] = _load_sharded_state_into_module(
            module,
            directory,
            basename=name,
            export_root=root,
            verify_shard_hashes=verify_shard_hashes,
        )
    return reports


# ---------------------------------------------------------------------------
# Prompt/generation sequence handling
# ---------------------------------------------------------------------------


def _extract_generate_sequences(generated: Any) -> Tensor:
    if isinstance(generated, Tensor):
        sequences = generated
    else:
        sequences = getattr(generated, "sequences", None)
    if not isinstance(sequences, Tensor):
        raise InferenceError(
            "Hugging Face generate() did not return a tensor or an object with "
            "a tensor .sequences field."
        )
    if sequences.ndim != 2 or sequences.dtype != torch.long:
        raise InferenceError(
            "Generated sequences must be torch.long [B,S], got "
            f"shape={tuple(sequences.shape)}, dtype={sequences.dtype}."
        )
    return sequences


def _normalise_generated_sequences(
    sequences: Tensor,
    *,
    prompt_input_ids: Tensor,
) -> Tensor:
    """Require generated output to include the exact padded prompt prefix."""

    if int(sequences.shape[0]) != int(prompt_input_ids.shape[0]):
        raise InferenceError(
            "Generation batch size differs from prompt batch size: "
            f"generated={int(sequences.shape[0])}, prompt={int(prompt_input_ids.shape[0])}."
        )
    prompt_width = int(prompt_input_ids.shape[1])
    if int(sequences.shape[1]) <= prompt_width:
        raise InferenceError(
            "Generation returned no new token after the prompt: "
            f"generated_length={int(sequences.shape[1])}, prompt_length={prompt_width}."
        )
    observed_prefix = sequences[:, :prompt_width]
    if not torch.equal(observed_prefix, prompt_input_ids):
        mismatch = observed_prefix.ne(prompt_input_ids).nonzero(as_tuple=False)
        preview = mismatch[:10].detach().cpu().tolist()
        raise InferenceCompatibilityError(
            "Generation output does not preserve the supplied input_ids prefix. "
            "The installed Transformers generation contract differs from the "
            f"reviewed Phi-3 path. First mismatches: {preview}."
        )
    return sequences


def _generated_suffix_valid_mask(
    suffix_ids: Tensor,
    *,
    eos_token_id: int,
    pad_token_id: int,
) -> Tensor:
    """Mark generated positions through the first EOS, excluding later padding."""

    if suffix_ids.ndim != 2 or suffix_ids.dtype != torch.long:
        raise InferenceInputError("suffix_ids must be torch.long [B,T].")
    batch, width = suffix_ids.shape
    positions = torch.arange(width, device=suffix_ids.device).unsqueeze(0).expand(batch, -1)
    eos_positions = torch.where(
        suffix_ids.eq(int(eos_token_id)),
        positions,
        torch.full_like(positions, width),
    )
    first_eos = eos_positions.min(dim=1).values
    valid = positions <= first_eos.unsqueeze(1)
    no_eos = first_eos.eq(width)
    if int(pad_token_id) != int(eos_token_id):
        valid = valid & (
            suffix_ids.ne(int(pad_token_id)) | positions.eq(first_eos.unsqueeze(1))
        )
    # With no EOS, ordinary padding still remains invalid.
    if int(pad_token_id) != int(eos_token_id):
        valid = torch.where(
            no_eos.unsqueeze(1),
            suffix_ids.ne(int(pad_token_id)),
            valid,
        )
    return valid


def _build_full_generation_attention_mask(
    *,
    prompt_attention_mask: Tensor,
    sequences: Tensor,
    eos_token_id: int,
    pad_token_id: int,
) -> Tensor:
    prompt_width = int(prompt_attention_mask.shape[1])
    suffix = sequences[:, prompt_width:]
    suffix_mask = _generated_suffix_valid_mask(
        suffix,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
    )
    return torch.cat(
        [prompt_attention_mask.to(dtype=torch.bool), suffix_mask],
        dim=1,
    )


def _position_ids_from_attention_mask(attention_mask: Tensor) -> Tensor:
    """Build left-padding-safe position IDs for a plain decoder replay."""

    if attention_mask.ndim != 2:
        raise InferenceInputError("attention_mask must be [B,S].")
    valid = attention_mask.to(dtype=torch.bool)
    positions = valid.to(torch.long).cumsum(dim=1) - 1
    return positions.masked_fill(~valid, 0)


def _trim_generated_ids(
    suffix_ids: Tensor,
    suffix_valid_mask: Tensor,
    *,
    eos_token_id: int,
) -> tuple[tuple[int, ...], str]:
    ids = [
        int(value)
        for value in suffix_ids[suffix_valid_mask].detach().cpu().tolist()
    ]
    stop_reason = "length"
    if ids and ids[-1] == int(eos_token_id):
        ids = ids[:-1]
        stop_reason = "eos"
    return tuple(ids), stop_reason


def _decode_answer(
    tokenizer: Any,
    token_ids: Sequence[int],
    *,
    segmentation_token: str,
) -> tuple[str, str]:
    raw = tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=True,
    ).strip()
    # [SEG] is deliberately a normal vocabulary token. Preserve it in the raw
    # answer for auditing and remove it only from the user-facing prose.
    answer = raw.replace(segmentation_token, " ")
    answer = " ".join(answer.split()).strip()
    return answer, raw


# ---------------------------------------------------------------------------
# Input volume and mask output
# ---------------------------------------------------------------------------


def _build_arbitrary_path_volume_reader(
    config: ExperimentConfig,
    image_path: Path,
) -> VolumeReader:
    image_path = image_path.expanduser().resolve()
    main = config.model.main_vision
    seg = config.model.seg_vision
    if seg.enabled and (
        main.image_channels != seg.image_channels or main.image_size != seg.image_size
    ):
        raise InferenceCompatibilityError(
            "The two independent image encoders require the same preprocessed "
            "input channel count and spatial shape."
        )
    options = VolumeReaderOptions(
        expected_image_channels=int(main.image_channels),
        expected_spatial_shape=tuple(int(value) for value in main.image_size),
        image_range=(0.0, 1.0),
        enforce_image_range=True,
    )
    cache = NodeLocalFileCache(
        LocalCacheOptions(
            source_root=image_path.parent,
            cache_root=None,
            enabled=False,
        )
    )
    return VolumeReader(options, cache)


def load_inference_volume(
    config: ExperimentConfig,
    image_path: str | os.PathLike[str],
) -> tuple[Tensor, LoadedVolume]:
    """Read and identity-transform one already preprocessed M3D volume."""

    path = Path(image_path).expanduser().resolve()
    reader = _build_arbitrary_path_volume_reader(config, path)
    loaded = reader.load_image(path)
    transform = build_volume_transform(config, DataSplit.TEST)
    transformed = transform(loaded.tensor, None)
    return transformed.image, loaded


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            np.save(handle, array, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_prediction_volume(
    tensor: Tensor,
    output_path: str | os.PathLike[str],
    *,
    source_volume: LoadedVolume | None,
    probability: bool,
) -> Path:
    """Save ``[D,H,W]`` or ``[1,D,H,W]`` output as NPY or canonical NIfTI."""

    path = Path(output_path).expanduser().resolve()
    value = tensor.detach().cpu()
    if value.ndim == 4 and int(value.shape[0]) == 1:
        value = value[0]
    if value.ndim != 3:
        raise InferenceInputError(
            f"Prediction volume must be [D,H,W] or [1,D,H,W], got {tuple(value.shape)}."
        )
    array_dhw = value.float().numpy() if probability else value.to(torch.uint8).numpy()

    name = path.name.lower()
    if name.endswith(".npy"):
        _atomic_save_npy(path, array_dhw)
        return path
    if not (name.endswith(".nii") or name.endswith(".nii.gz")):
        raise InferenceInputError(
            f"Unsupported prediction output {path}; expected .npy, .nii, or .nii.gz."
        )
    if source_volume is None or source_volume.geometry.affine is None:
        raise InferenceInputError(
            "NIfTI output requires a NIfTI source volume with retained affine. "
            "Use .npy output for a NumPy source."
        )

    try:
        import nibabel as nib
    except ImportError as exc:
        raise InferenceDependencyError(
            "Saving NIfTI output requires nibabel."
        ) from exc

    # M3D [D,H,W] == nibabel [Z,Y,X]. Convert back to [X,Y,Z].
    array_xyz = np.transpose(array_dhw, (2, 1, 0))
    dtype = np.float32 if probability else np.uint8
    nifti = nib.Nifti1Image(
        array_xyz.astype(dtype, copy=False),
        affine=np.asarray(source_volume.geometry.affine, dtype=np.float64),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".nii.gz" if name.endswith(".nii.gz") else ".nii"
    temporary = path.parent / f".{path.name}.tmp-{os.getpid()}{suffix}"
    try:
        nib.save(nifti, str(temporary))
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


# ---------------------------------------------------------------------------
# Inference engine
# ---------------------------------------------------------------------------


def _parameter_storage_identity(parameter: nn.Parameter) -> tuple[Any, ...]:
    tensor = parameter.detach()
    try:
        storage = tensor.untyped_storage()
        return (
            tensor.device.type,
            tensor.device.index,
            int(storage.data_ptr()),
            int(storage.nbytes()),
            int(tensor.storage_offset()),
        )
    except Exception:
        return ("object", id(parameter))


def _shared_image_encoder_counts(model: M3DModel) -> tuple[int, int]:
    segmentation_encoder = model.segmentation_image_encoder
    if segmentation_encoder is None:
        return 0, 0
    main_parameters = {id(parameter) for parameter in model.main_image_encoder.parameters()}
    seg_parameters = {id(parameter) for parameter in segmentation_encoder.parameters()}
    shared_parameters = len(main_parameters.intersection(seg_parameters))

    main_storages = {
        _parameter_storage_identity(parameter)
        for parameter in model.main_image_encoder.parameters()
    }
    seg_storages = {
        _parameter_storage_identity(parameter)
        for parameter in segmentation_encoder.parameters()
    }
    return shared_parameters, len(main_storages.intersection(seg_storages))


def _inference_config_from_export(config: ExperimentConfig, export_root: Path) -> ExperimentConfig:
    result = copy.deepcopy(config)
    result.optimization.checkpoint_language_model = False
    result.optimization.checkpoint_main_vision = False
    result.optimization.checkpoint_seg_vision = False
    result.optimization.checkpoint_segmentation_decoder = False
    result.model.main_vision.activation_checkpoint_every_n_layers = 0
    result.model.seg_vision.activation_checkpoint_every_n_layers = 0
    result.model.lora.adapter_checkpoint_path = None

    # When a merged language export exists, use it directly and disable PEFT.
    # Its base weights already include the final LoRA delta and its embedding/
    # LM-head rows already include the trained M3D tokens.  Non-language M3D
    # components are then loaded from components/ with strict state contracts.
    merged_language = export_root / "language_merged"
    if (merged_language / "config.json").is_file():
        result.model.language_model_name_or_path = str(merged_language)
        result.model.lora.enabled = False
        result.model.lora.adapter_checkpoint_path = None
    result.validate()
    return result


class M3DInferenceEngine:
    """Loaded portable M3D model with text and generated-[SEG] inference."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        tokenizer_bundle: "TokenizerBundle",
        model: M3DModel,
        device: torch.device,
        export_directory: Path,
        build_report: InferenceBuildReport,
    ) -> None:
        self.config = config
        self.tokenizer_bundle = tokenizer_bundle
        from m3d.tokenization import M3DTextProcessor

        self.text_processor = M3DTextProcessor(tokenizer_bundle, config)
        self.model = model
        self.device = device
        self.export_directory = export_directory
        self.build_report = build_report

    @classmethod
    def from_export(
        cls,
        export_directory: str | os.PathLike[str],
        *,
        device: str | torch.device = "cuda:0",
        cache_dir: str | os.PathLike[str] | None = None,
        local_files_only: bool = False,
        verify_shard_hashes: bool = True,
        allow_cpu: bool = False,
    ) -> "M3DInferenceEngine":
        """Reconstruct and strictly load one portable M3D export."""

        root = Path(export_directory).expanduser().resolve()
        _validate_export_directory(root)
        resolved_device = torch.device(device)
        if resolved_device.type == "cuda":
            if not torch.cuda.is_available():
                raise InferenceInputError(
                    f"CUDA device {resolved_device} was requested but CUDA is unavailable."
                )
            if resolved_device.index is None:
                resolved_device = torch.device("cuda", torch.cuda.current_device())
            torch.cuda.set_device(resolved_device)
            if not torch.cuda.is_bf16_supported():
                raise InferenceCompatibilityError(
                    f"Device {resolved_device} does not report BF16 support."
                )
        elif not allow_cpu:
            raise InferenceInputError(
                "Full Phi-3 M3D inference is intended for CUDA. Pass allow_cpu=True "
                "only for controlled diagnostics."
            )

        exported_config = load_config(
            root / _EXPORT_CONFIG_FILE,
            resolve_paths=False,
            verify_paths=False,
        )
        config = _inference_config_from_export(exported_config, root)
        tokenizer_bundle = load_exported_tokenizer(
            root,
            model_max_length=config.model.model_max_length,
        )
        model, model_report = build_m3d_model(
            config,
            tokenizer_bundle,
            cache_dir=cache_dir,
            local_files_only=bool(local_files_only),
            torch_dtype=torch.bfloat16,
            load_pretrained_components=False,
            strict_pretrained=True,
        )
        if not isinstance(model_report, M3DBuildReport):
            raise InferenceCompatibilityError(
                "M3D builder did not return M3DBuildReport."
            )
        merged_language_available = (root / "language_merged" / "config.json").is_file()
        if merged_language_available:
            component_reports = load_exported_components(
                model,
                root,
                verify_shard_hashes=verify_shard_hashes,
            )
            state_report_payload: Mapping[str, Any] = {
                "mode": "merged_language_plus_components",
                "language_directory": str(root / "language_merged"),
                "components": {
                    name: report.to_dict() for name, report in component_reports.items()
                },
            }
        else:
            state_report = load_exported_model_state(
                model,
                root,
                verify_shard_hashes=verify_shard_hashes,
            )
            state_report_payload = {
                "mode": "adapter_form_full_bundle",
                "full_model": state_report.to_dict(),
            }
        model.requires_grad_(False)
        model.eval()
        model.to(resolved_device)

        shared_parameters, shared_storages = _shared_image_encoder_counts(model)
        if shared_parameters or shared_storages:
            raise InferenceCompatibilityError(
                "Portable export reconstructed shared Main/SegVol image encoders: "
                f"parameters={shared_parameters}, storages={shared_storages}."
            )
        summary = model.parameter_summary()
        build_report = InferenceBuildReport(
            state_version=_INFERENCE_STATE_VERSION,
            export_directory=str(root),
            device=str(resolved_device),
            model_build=model_report.to_dict(),
            state_load=state_report_payload,
            tokenizer_metadata=tokenizer_bundle.metadata.to_dict(),
            total_parameter_count=int(summary.total),
            main_image_encoder_parameter_count=int(summary.main_vision),
            segmentation_image_encoder_parameter_count=(
                0
                if model.segmentation_image_encoder is None
                else sum(
                    int(parameter.numel())
                    for parameter in model.segmentation_image_encoder.parameters()
                )
            ),
            shared_image_encoder_parameter_count=shared_parameters,
            shared_image_encoder_storage_count=shared_storages,
        )
        return cls(
            config=config,
            tokenizer_bundle=tokenizer_bundle,
            model=model,
            device=resolved_device,
            export_directory=root,
            build_report=build_report,
        )

    @contextlib.contextmanager
    def _autocast(self) -> Iterator[None]:
        if self.device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                yield
        else:
            with contextlib.nullcontext():
                yield

    def _prepare_prompts(
        self,
        questions: Sequence[str],
    ) -> tuple[Tensor, Tensor, Tensor, tuple["EncodedPrompt", ...]]:
        from m3d.tokenization import pad_generation_prompts

        encoded = tuple(self.text_processor.encode_prompt(question) for question in questions)
        if any(item.was_truncated for item in encoded):
            indices = [index for index, item in enumerate(encoded) if item.was_truncated]
            raise InferenceInputError(
                "One or more generation prompts were truncated before inference: "
                f"rows={indices}. Shorten the question."
            )
        input_ids, attention_mask, lengths = pad_generation_prompts(
            encoded,
            bundle=self.tokenizer_bundle,
            data_config=self.config.data,
            model_max_length=self.config.model.model_max_length,
        )
        return input_ids, attention_mask, lengths, encoded

    def predict_paths(
        self,
        image_paths: Sequence[str | os.PathLike[str]],
        questions: Sequence[str],
        *,
        mode: InferenceMode | str = InferenceMode.AUTO,
        generation: GenerationSettings | None = None,
        mask_threshold: float = 0.5,
    ) -> BatchInferenceResult:
        """Load preprocessed volume paths and run a homogeneous prediction batch."""

        if len(image_paths) != len(questions):
            raise InferenceInputError(
                "image_paths and questions must have the same length."
            )
        images: list[Tensor] = []
        loaded_volumes: list[LoadedVolume] = []
        for path in image_paths:
            image, loaded = load_inference_volume(self.config, path)
            images.append(image)
            loaded_volumes.append(loaded)
        return self.predict_tensors(
            torch.stack(images, dim=0),
            questions,
            mode=mode,
            generation=generation,
            mask_threshold=mask_threshold,
            loaded_volumes=loaded_volumes,
        )

    def predict_tensors(
        self,
        images: Tensor,
        questions: Sequence[str],
        *,
        mode: InferenceMode | str = InferenceMode.AUTO,
        generation: GenerationSettings | None = None,
        mask_threshold: float = 0.5,
        loaded_volumes: Sequence[LoadedVolume | None] | None = None,
    ) -> BatchInferenceResult:
        """Generate text and optionally decode generated ``[SEG]`` masks."""

        started = time.monotonic()
        resolved_mode = InferenceMode.parse(mode)
        settings = generation or GenerationSettings()
        if not 0 <= float(mask_threshold) <= 1:
            raise InferenceInputError("mask_threshold must be in [0, 1].")
        if images.ndim != 5:
            raise InferenceInputError(
                f"images must be [B,C,D,H,W], got {tuple(images.shape)}."
            )
        if not images.is_floating_point():
            raise InferenceInputError("images must use a floating-point dtype.")
        batch_size = int(images.shape[0])
        if batch_size <= 0:
            raise InferenceInputError("Inference batch cannot be empty.")
        if len(questions) != batch_size:
            raise InferenceInputError(
                f"questions length {len(questions)} differs from image batch {batch_size}."
            )
        clean_questions = tuple(str(question).strip() for question in questions)
        if any(not question for question in clean_questions):
            raise InferenceInputError("Every inference question must be non-empty.")

        expected_image_shape = (
            batch_size,
            int(self.config.model.main_vision.image_channels),
            *tuple(int(value) for value in self.config.model.main_vision.image_size),
        )
        if tuple(images.shape) != expected_image_shape:
            raise InferenceInputError(
                "Inference image shape differs from the trained model contract: "
                f"received={tuple(images.shape)}, expected={expected_image_shape}."
            )
        if not torch.isfinite(images).all():
            raise InferenceInputError("Inference images contain NaN or Inf values.")
        image_min = float(images.min().item())
        image_max = float(images.max().item())
        tolerance = 1.0e-4
        if image_min < -tolerance or image_max > 1.0 + tolerance:
            raise InferenceInputError(
                "Inference images must already be normalised to [0,1]: "
                f"observed min={image_min:.6g}, max={image_max:.6g}."
            )

        sources: tuple[LoadedVolume | None, ...]
        if loaded_volumes is None:
            sources = tuple(None for _ in range(batch_size))
        else:
            if len(loaded_volumes) != batch_size:
                raise InferenceInputError(
                    "loaded_volumes length must match image batch size."
                )
            sources = tuple(loaded_volumes)

        prompt_ids_cpu, prompt_mask_cpu, _, _ = self._prepare_prompts(clean_questions)
        prompt_width = int(prompt_ids_cpu.shape[1])
        if prompt_width + settings.max_new_tokens > self.config.model.model_max_length:
            available = self.config.model.model_max_length - prompt_width
            raise InferenceInputError(
                "Generation would exceed model_max_length after dynamic prompt "
                f"padding: prompt_width={prompt_width}, requested_new_tokens="
                f"{settings.max_new_tokens}, available={available}."
            )

        images_device = images.to(
            device=self.device,
            dtype=torch.float32,
            non_blocking=(images.device.type == "cpu" and images.is_pinned()),
        )
        prompt_ids = prompt_ids_cpu.to(self.device, non_blocking=True)
        prompt_mask = prompt_mask_cpu.to(self.device, non_blocking=True)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        main_started = time.monotonic()
        with torch.inference_mode(), self._autocast():
            visual_embeddings = self.model.encode_images(images_device)
        if not isinstance(visual_embeddings, Tensor):
            raise InferenceError("Main image encoder did not return visual token tensor.")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        main_seconds = float(time.monotonic() - main_started)

        generation_started = time.monotonic()
        with torch.inference_mode(), self._autocast():
            generated = self.model.language_model.generate(
                input_ids=prompt_ids,
                attention_mask=prompt_mask,
                visual_embeddings=visual_embeddings,
                **settings.generation_kwargs(
                    pad_token_id=self.tokenizer_bundle.metadata.pad_token_id,
                    eos_token_id=self.tokenizer_bundle.metadata.eos_token_id,
                ),
            )
        sequences = _normalise_generated_sequences(
            _extract_generate_sequences(generated),
            prompt_input_ids=prompt_ids,
        )
        full_attention_mask = _build_full_generation_attention_mask(
            prompt_attention_mask=prompt_mask,
            sequences=sequences,
            eos_token_id=self.tokenizer_bundle.metadata.eos_token_id,
            pad_token_id=self.tokenizer_bundle.metadata.pad_token_id,
        )
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        generation_seconds = float(time.monotonic() - generation_started)

        suffix_ids = sequences[:, prompt_width:]
        suffix_mask = full_attention_mask[:, prompt_width:]
        generated_rows: list[tuple[int, ...]] = []
        answers: list[str] = []
        raw_answers: list[str] = []
        stop_reasons: list[str] = []
        seg_counts: list[int] = []
        segmentation_rows: list[int] = []
        seg_token_id = int(self.tokenizer_bundle.metadata.segmentation_token_id)
        for row in range(batch_size):
            ids, stop_reason = _trim_generated_ids(
                suffix_ids[row],
                suffix_mask[row],
                eos_token_id=self.tokenizer_bundle.metadata.eos_token_id,
            )
            answer, raw_answer = _decode_answer(
                self.tokenizer_bundle.tokenizer,
                ids,
                segmentation_token=self.tokenizer_bundle.metadata.segmentation_token,
            )
            count = sum(token_id == seg_token_id for token_id in ids)
            generated_rows.append(ids)
            answers.append(answer)
            raw_answers.append(raw_answer)
            stop_reasons.append(stop_reason)
            seg_counts.append(count)
            if count > 0:
                segmentation_rows.append(row)

        if resolved_mode is InferenceMode.SEGMENTATION:
            missing_rows = [index for index, count in enumerate(seg_counts) if count == 0]
            if missing_rows:
                raise InferenceError(
                    "Segmentation mode requires the model to generate [SEG] for every "
                    f"row, but it was absent for rows {missing_rows}. Raw answers: "
                    f"{[raw_answers[index] for index in missing_rows]}."
                )
        elif resolved_mode is InferenceMode.TEXT:
            segmentation_rows = []

        probability_by_row: dict[int, Tensor] = {}
        mask_by_row: dict[int, Tensor] = {}
        iou_by_row: dict[int, Tensor] = {}
        segmentation_seconds = 0.0
        if segmentation_rows:
            if self.model.seg_projector is None or self.model.seg_module is None:
                raise InferenceCompatibilityError(
                    "Generated [SEG], but this export does not contain segmentation modules."
                )
            segmentation_started = time.monotonic()
            row_index = torch.tensor(
                segmentation_rows,
                dtype=torch.long,
                device=self.device,
            )
            selected_ids = sequences.index_select(0, row_index)
            selected_mask = full_attention_mask.index_select(0, row_index)
            selected_visual = visual_embeddings.index_select(0, row_index)
            selected_images = images_device.index_select(0, row_index)
            selected_position_ids = _position_ids_from_attention_mask(selected_mask)

            with torch.inference_mode(), self._autocast():
                language_output = self.model.language_model(
                    input_ids=selected_ids,
                    attention_mask=selected_mask,
                    visual_embeddings=selected_visual,
                    labels=None,
                    position_ids=selected_position_ids,
                    use_cache=False,
                    logits_mode="none",
                )
                if not isinstance(language_output, LanguageModelOutput):
                    raise InferenceError(
                        "Language replay did not return LanguageModelOutput."
                    )
                prompt_output = self.model.seg_projector.extract_and_project(
                    last_hidden_state=language_output.last_hidden_state,
                    input_ids=selected_ids,
                    segmentation_token_id=seg_token_id,
                    attention_mask=selected_mask,
                )
                segmentation_output = self.model.seg_module(
                    selected_images,
                    text_embedding=prompt_output.prompt_embeddings,
                    multimask_output=False,
                    return_structured=True,
                )
                if not isinstance(segmentation_output, SegVolOutput):
                    raise InferenceError("SegVol did not return SegVolOutput.")
                probabilities = torch.sigmoid(segmentation_output.logits.float())
                binary_masks = probabilities.gt(float(mask_threshold)).to(torch.uint8)

            probabilities_cpu = probabilities.detach().cpu()
            masks_cpu = binary_masks.detach().cpu()
            iou_cpu = segmentation_output.iou_predictions.float().detach().cpu()
            for local_index, original_row in enumerate(segmentation_rows):
                probability_by_row[original_row] = probabilities_cpu[local_index]
                mask_by_row[original_row] = masks_cpu[local_index]
                iou_by_row[original_row] = iou_cpu[local_index]
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            segmentation_seconds = float(time.monotonic() - segmentation_started)

        predictions: list[InferencePrediction] = []
        for row in range(batch_size):
            source = sources[row]
            predictions.append(
                InferencePrediction(
                    question=clean_questions[row],
                    answer=answers[row],
                    raw_answer=raw_answers[row],
                    generated_token_ids=generated_rows[row],
                    generated_segmentation_token_count=seg_counts[row],
                    stop_reason=stop_reasons[row],
                    source_path=(None if source is None else str(source.source_path)),
                    source_geometry=(
                        None if source is None else source.geometry.to_jsonable()
                    ),
                    segmentation_probability=probability_by_row.get(row),
                    segmentation_mask=mask_by_row.get(row),
                    iou_prediction=iou_by_row.get(row),
                )
            )

        return BatchInferenceResult(
            predictions=tuple(predictions),
            prompt_sequence_length=prompt_width,
            generated_sequence_length=int(sequences.shape[1]),
            main_vision_seconds=main_seconds,
            generation_seconds=generation_seconds,
            segmentation_seconds=segmentation_seconds,
            total_seconds=float(time.monotonic() - started),
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run portable M3D text/segmentation inference on one 3D volume."
    )
    parser.add_argument("--export-dir", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--question", type=str)
    parser.add_argument(
        "--mode",
        choices=[item.value for item in InferenceMode],
        default=InferenceMode.AUTO.value,
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--skip-shard-hash-verification", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--num-beams", type=int, default=1)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--mask-output", type=Path, default=None)
    parser.add_argument("--probability-output", type=Path, default=None)
    parser.add_argument("--build-report", type=Path, default=None)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if not args.self_test:
        missing = [
            flag
            for flag, value in (
                ("--export-dir", args.export_dir),
                ("--image", args.image),
                ("--question", args.question),
            )
            if value is None
        ]
        if missing:
            parser.error(f"required arguments missing: {', '.join(missing)}")
    return args


def _run_cli(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args.seed))

    engine = M3DInferenceEngine.from_export(
        args.export_dir,
        device=args.device,
        cache_dir=args.cache_dir,
        local_files_only=bool(args.local_files_only),
        verify_shard_hashes=not bool(args.skip_shard_hash_verification),
        allow_cpu=bool(args.allow_cpu),
    )
    settings = GenerationSettings(
        max_new_tokens=int(args.max_new_tokens),
        do_sample=bool(args.do_sample),
        temperature=float(args.temperature),
        top_p=float(args.top_p),
        num_beams=int(args.num_beams),
        repetition_penalty=float(args.repetition_penalty),
    )
    result = engine.predict_paths(
        [args.image],
        [args.question],
        mode=args.mode,
        generation=settings,
        mask_threshold=float(args.mask_threshold),
    )
    prediction = result.predictions[0]
    source_image, source_volume = load_inference_volume(engine.config, args.image)
    del source_image

    outputs: dict[str, str] = {}
    if args.mask_output is not None:
        if prediction.segmentation_mask is None:
            raise InferenceError(
                "--mask-output was requested, but this prediction has no segmentation mask."
            )
        outputs["mask"] = str(
            save_prediction_volume(
                prediction.segmentation_mask,
                args.mask_output,
                source_volume=source_volume,
                probability=False,
            )
        )
    if args.probability_output is not None:
        if prediction.segmentation_probability is None:
            raise InferenceError(
                "--probability-output was requested, but this prediction has no "
                "segmentation probability volume."
            )
        outputs["probability"] = str(
            save_prediction_volume(
                prediction.segmentation_probability,
                args.probability_output,
                source_volume=source_volume,
                probability=True,
            )
        )

    payload = result.summary()
    payload["export_directory"] = str(engine.export_directory)
    payload["device"] = str(engine.device)
    payload["saved_outputs"] = outputs
    if args.output_json is not None:
        _atomic_write_json(args.output_json.expanduser().resolve(), payload)
    if args.build_report is not None:
        _atomic_write_json(
            args.build_report.expanduser().resolve(),
            engine.build_report.to_dict(),
        )
    return payload


# ---------------------------------------------------------------------------
# Dependency-light self-test
# ---------------------------------------------------------------------------


class _FakeTokenizer:
    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        del skip_special_tokens, clean_up_tokenization_spaces
        mapping = {10: "yes", 11: "[SEG]", 12: "finding"}
        return " ".join(mapping.get(int(value), f"<{int(value)}>") for value in token_ids)


def _self_test_safetensors(root: Path) -> bool:
    try:
        from safetensors.torch import save_file
    except ImportError:
        return False

    class Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 2)
            self.register_buffer("scale", torch.tensor([2.0]))

    source = Tiny()
    with torch.no_grad():
        source.linear.weight.copy_(torch.arange(6, dtype=torch.float32).reshape(2, 3))
        source.linear.bias.copy_(torch.tensor([3.0, 4.0]))
    state_dir = root / _FULL_MODEL_DIRECTORY
    state_dir.mkdir(parents=True)
    shard_name = "m3d_model.safetensors"
    shard_path = state_dir / shard_name
    state = {name: value.detach().clone() for name, value in source.state_dict().items()}
    save_file(state, str(shard_path))
    descriptor = {
        "metadata": {"state_sha256": "test"},
        "weight_map": {name: shard_name for name in state},
        "shards": [
            {
                "file": shard_name,
                "tensor_count": len(state),
                "byte_count": int(shard_path.stat().st_size),
                "sha256": _sha256_file(shard_path),
            }
        ],
    }
    _atomic_write_json(
        state_dir / "m3d_model.safetensors.index.json",
        descriptor,
    )
    target = Tiny()
    with torch.no_grad():
        for parameter in target.parameters():
            parameter.zero_()
        target.scale.zero_()
    # The loader's type annotation is M3DModel, but it only requires nn.Module
    # state_dict semantics; casting keeps this targeted self-test small.
    report = load_exported_model_state(
        cast(M3DModel, target),
        root,
        verify_shard_hashes=True,
    )
    return (
        report.tensor_count == len(state)
        and all(
            torch.equal(target.state_dict()[name], value)
            for name, value in state.items()
        )
    )


def _self_test() -> dict[str, Any]:
    prompt_ids = torch.tensor(
        [[0, 0, 1, 2], [0, 3, 4, 5]],
        dtype=torch.long,
    )
    prompt_mask = torch.tensor(
        [[False, False, True, True], [False, True, True, True]],
        dtype=torch.bool,
    )
    sequences = torch.tensor(
        [
            [0, 0, 1, 2, 10, 11, 9, 0],
            [0, 3, 4, 5, 12, 9, 0, 0],
        ],
        dtype=torch.long,
    )
    normalised = _normalise_generated_sequences(
        sequences,
        prompt_input_ids=prompt_ids,
    )
    full_mask = _build_full_generation_attention_mask(
        prompt_attention_mask=prompt_mask,
        sequences=normalised,
        eos_token_id=9,
        pad_token_id=0,
    )
    suffix_mask = full_mask[:, prompt_ids.shape[1] :]
    first_ids, first_stop = _trim_generated_ids(
        sequences[0, 4:], suffix_mask[0], eos_token_id=9
    )
    second_ids, second_stop = _trim_generated_ids(
        sequences[1, 4:], suffix_mask[1], eos_token_id=9
    )
    answer, raw = _decode_answer(
        _FakeTokenizer(),
        first_ids,
        segmentation_token="[SEG]",
    )
    positions = _position_ids_from_attention_mask(full_mask)

    with tempfile.TemporaryDirectory(prefix="m3d-inference-test-") as temporary:
        root = Path(temporary)
        state_roundtrip = _self_test_safetensors(root)
        mask_path = root / "mask.npy"
        save_prediction_volume(
            torch.tensor([[[[0, 1], [1, 0]]]], dtype=torch.uint8),
            mask_path,
            source_volume=None,
            probability=False,
        )
        npy_roundtrip = np.array_equal(
            np.load(mask_path, allow_pickle=False),
            np.array([[[0, 1], [1, 0]]], dtype=np.uint8),
        )

    settings_error = False
    try:
        GenerationSettings(do_sample=True, num_beams=2)
    except InferenceInputError:
        settings_error = True

    malformed_prefix = False
    try:
        _normalise_generated_sequences(
            sequences.clone().index_put_((torch.tensor([0]), torch.tensor([2])), torch.tensor([99])),
            prompt_input_ids=prompt_ids,
        )
    except InferenceCompatibilityError:
        malformed_prefix = True

    checks = {
        "segmentation_token_preserved_in_raw": "[SEG]" in raw,
        "segmentation_token_removed_from_answer": "[SEG]" not in answer,
        "state_streaming_roundtrip": state_roundtrip,
        "npy_mask_roundtrip": npy_roundtrip,
        "sampling_beam_conflict_detected": settings_error,
        "malformed_generation_prefix_detected": malformed_prefix,
    }
    return {
        "status": "passed" if all(checks.values()) else "failed",
        "first_generated_ids": list(first_ids),
        "second_generated_ids": list(second_ids),
        "first_stop_reason": first_stop,
        "second_stop_reason": second_stop,
        "raw_answer": raw,
        "clean_answer": answer,
        "left_padding_position_ids": positions.tolist(),
        **checks,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.self_test:
        print(json.dumps(_self_test(), indent=2, sort_keys=True))
        return
    payload = _run_cli(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
