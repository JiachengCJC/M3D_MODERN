"""Tokenizer and dynamic-padding utilities for M3D-Modernized.

This module is constructed after :mod:`m3d.config` and :mod:`m3d.runtime`, but
before datasets and model parameters are built.  It preserves the original M3D
prompt contract while removing its fixed 512-token padding and repeated
``question`` tokenization.

Original M3D text layout::

    <im_patch><im_patch>...<im_patch> QUESTION ANSWER <eos>

The image placeholder is repeated once per output token from the multimodal
projector.  For the default 3D ViT and 2x2x2 spatial pooling this is 256 image
placeholder tokens.

Important compatibility decisions
---------------------------------
* ``<im_patch>``, ``<bx_start>``, and ``<bx_end>`` are additional special
  tokens, matching the original repository.
* ``[SEG]`` is added as a normal vocabulary token, not a special token.  This
  matches the original repository and prevents ``skip_special_tokens=True``
  from silently deleting the segmentation marker during text inspection.
* The tokenizer pads on the right.
* If the base tokenizer has no pad token, its unknown token is reused exactly
  as in the original M3D code; EOS is used only when no unknown token exists.
* Supervised examples are encoded once with character offsets when a fast
  tokenizer is available.  A slow-tokenizer fallback separately encodes the
  prompt and answer without re-tokenizing the prompt a second time.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor
from transformers import AddedToken, AutoTokenizer, PreTrainedTokenizerBase

from .config import DataConfig, ExperimentConfig, ModelConfig


IGNORE_INDEX = -100
DEFAULT_ANSWER_SEPARATOR = " "


class TokenizationError(RuntimeError):
    """Raised when tokenizer structure or a supervised example is invalid."""


@dataclass(frozen=True, slots=True)
class TokenizerMetadata:
    """Stable IDs and vocabulary sizes needed by datasets and the model."""

    tokenizer_name_or_path: str
    original_vocab_size: int
    vocabulary_size: int
    added_token_count: int
    image_token: str
    image_token_id: int
    segmentation_token: str
    segmentation_token_id: int
    box_start_token: str
    box_start_token_id: int
    box_end_token: str
    box_end_token_id: int
    pad_token_id: int
    eos_token_id: int
    visual_token_count: int

    @property
    def requires_embedding_resize(self) -> bool:
        return self.vocabulary_size != self.original_vocab_size

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokenizer_name_or_path": self.tokenizer_name_or_path,
            "original_vocab_size": self.original_vocab_size,
            "vocabulary_size": self.vocabulary_size,
            "added_token_count": self.added_token_count,
            "image_token": self.image_token,
            "image_token_id": self.image_token_id,
            "segmentation_token": self.segmentation_token,
            "segmentation_token_id": self.segmentation_token_id,
            "box_start_token": self.box_start_token,
            "box_start_token_id": self.box_start_token_id,
            "box_end_token": self.box_end_token,
            "box_end_token_id": self.box_end_token_id,
            "pad_token_id": self.pad_token_id,
            "eos_token_id": self.eos_token_id,
            "visual_token_count": self.visual_token_count,
            "requires_embedding_resize": self.requires_embedding_resize,
        }


@dataclass(slots=True)
class TokenizerBundle:
    """The Hugging Face tokenizer together with validated M3D metadata."""

    tokenizer: PreTrainedTokenizerBase
    metadata: TokenizerMetadata

    @property
    def image_prefix(self) -> str:
        return self.metadata.image_token * self.metadata.visual_token_count

    def save_pretrained(self, output_dir: str | os.PathLike[str]) -> None:
        """Save tokenizer files and an atomic M3D metadata manifest."""

        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(destination)

        metadata_path = destination / "m3d_tokenizer_metadata.json"
        temporary_path = metadata_path.with_suffix(".json.tmp")
        with temporary_path.open("w", encoding="utf-8") as handle:
            json.dump(
                self.metadata.to_dict(),
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
        os.replace(temporary_path, metadata_path)


@dataclass(frozen=True, slots=True)
class EncodedText:
    """One variable-length, unpadded training example on CPU."""

    input_ids: Tensor
    labels: Tensor
    attention_mask: Tensor
    prompt_token_count: int
    supervised_token_count: int
    was_truncated: bool

    def __post_init__(self) -> None:
        expected_shape = self.input_ids.shape
        if self.input_ids.ndim != 1:
            raise ValueError(
                f"input_ids must be one-dimensional, got {tuple(expected_shape)}"
            )
        if self.labels.shape != expected_shape:
            raise ValueError("labels must have the same shape as input_ids")
        if self.attention_mask.shape != expected_shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        if self.input_ids.dtype != torch.long:
            raise TypeError("input_ids must use torch.long")
        if self.labels.dtype != torch.long:
            raise TypeError("labels must use torch.long")
        if self.attention_mask.dtype != torch.bool:
            raise TypeError("attention_mask must use torch.bool")
        if not bool(self.attention_mask.all()):
            raise ValueError("EncodedText is unpadded; every attention value must be true")
        if self.prompt_token_count < 0:
            raise ValueError("prompt_token_count cannot be negative")
        if self.supervised_token_count <= 0:
            raise ValueError("a supervised example must contain at least one target token")
        actual_supervised = int(self.labels.ne(IGNORE_INDEX).sum().item())
        if actual_supervised != self.supervised_token_count:
            raise ValueError(
                "supervised_token_count does not match the non-ignored labels: "
                f"metadata={self.supervised_token_count}, labels={actual_supervised}"
            )

    @property
    def length(self) -> int:
        return int(self.input_ids.numel())


@dataclass(frozen=True, slots=True)
class EncodedPrompt:
    """One variable-length prompt used for generation or evaluation."""

    input_ids: Tensor
    attention_mask: Tensor
    was_truncated: bool

    def __post_init__(self) -> None:
        if self.input_ids.ndim != 1:
            raise ValueError("input_ids must be one-dimensional")
        if self.attention_mask.shape != self.input_ids.shape:
            raise ValueError("attention_mask must have the same shape as input_ids")
        if self.input_ids.dtype != torch.long:
            raise TypeError("input_ids must use torch.long")
        if self.attention_mask.dtype != torch.bool:
            raise TypeError("attention_mask must use torch.bool")

    @property
    def length(self) -> int:
        return int(self.input_ids.numel())


@dataclass(frozen=True, slots=True)
class PaddedTextBatch:
    """A right-padded text batch produced at collate time."""

    input_ids: Tensor
    labels: Tensor
    attention_mask: Tensor
    sequence_length: int
    unpadded_lengths: Tensor

    def to_dict(self) -> dict[str, Tensor]:
        return {
            "input_ids": self.input_ids,
            "labels": self.labels,
            "attention_mask": self.attention_mask,
        }


def projected_visual_token_count(model_config: ModelConfig) -> int:
    """Return the exact number of visual tokens emitted by the projector.

    This reproduces ``SpatialPoolingProjector.proj_out_num`` from M3D without
    constructing the model.  The CLS token is deliberately excluded because
    the projector consumes patch features only.
    """

    vision = model_config.main_vision
    projector = model_config.projector

    patch_grid = tuple(
        image_dim // patch_dim
        for image_dim, patch_dim in zip(vision.image_size, vision.patch_size)
    )

    if projector.pooling_type == "spatial":
        if any(dim % projector.pooling_size != 0 for dim in patch_grid):
            raise TokenizationError(
                "Spatial projector pooling must evenly divide every patch-grid "
                f"dimension; grid={patch_grid}, pooling_size={projector.pooling_size}"
            )
        pooled_grid = tuple(dim // projector.pooling_size for dim in patch_grid)
        count = pooled_grid[0] * pooled_grid[1] * pooled_grid[2]
    elif projector.pooling_type == "sequence":
        patch_count = patch_grid[0] * patch_grid[1] * patch_grid[2]
        kernel = projector.pooling_size**3
        if patch_count % kernel != 0:
            raise TokenizationError(
                "Sequence projector pooling must evenly divide the patch count; "
                f"patch_count={patch_count}, kernel={kernel}"
            )
        count = patch_count // kernel
    else:  # Defensive guard; config validation already constrains this value.
        raise TokenizationError(
            f"Unsupported projector pooling type: {projector.pooling_type!r}"
        )

    if count <= 0:
        raise TokenizationError(f"Projected visual-token count must be positive, got {count}")
    return count


def _token_id(tokenizer: PreTrainedTokenizerBase, token: str, *, name: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token)
    if token_id is None:
        raise TokenizationError(f"Tokenizer returned no ID for {name} {token!r}")
    token_id = int(token_id)
    if tokenizer.unk_token_id is not None and token_id == tokenizer.unk_token_id:
        if token != tokenizer.unk_token:
            raise TokenizationError(
                f"{name} {token!r} resolves to the unknown-token ID {token_id}"
            )
    return token_id


def _ensure_single_token(
    tokenizer: PreTrainedTokenizerBase,
    token: str,
    expected_id: int,
    *,
    name: str,
) -> None:
    encoded = tokenizer.encode(token, add_special_tokens=False)
    if encoded != [expected_id]:
        raise TokenizationError(
            f"{name} must encode as exactly one token. "
            f"Expected [{expected_id}], got {encoded} for {token!r}."
        )


def _merge_additional_special_tokens(
    tokenizer: PreTrainedTokenizerBase,
    tokens: Sequence[str],
) -> int:
    """Add M3D special tokens without discarding Phi-3's existing tokens."""

    existing = list(tokenizer.additional_special_tokens or [])
    merged = existing.copy()
    for token in tokens:
        if token not in merged:
            merged.append(token)
    return int(
        tokenizer.add_special_tokens(
            {"additional_special_tokens": merged},
            replace_additional_special_tokens=True,
        )
    )


def build_tokenizer(
    config: ExperimentConfig,
    *,
    cache_dir: str | os.PathLike[str] | None = None,
    local_files_only: bool = False,
) -> TokenizerBundle:
    """Load and validate the Phi-3 tokenizer used by all M3D tasks.

    Model embeddings are *not* resized here because model construction happens
    later.  ``TokenizerMetadata.requires_embedding_resize`` tells the model
    builder whether resizing is required.
    """
    # 取得model配置
    model_config = config.model
    tokenizer_name = (
        model_config.tokenizer_name_or_path
        or model_config.language_model_name_or_path
    )

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name,
        cache_dir=None if cache_dir is None else str(cache_dir),
        model_max_length=model_config.model_max_length,
        padding_side="right",
        truncation_side="right",
        use_fast=True,
        trust_remote_code=model_config.trust_remote_code,
        local_files_only=local_files_only,
    )

    tokenizer.padding_side = "right"
    tokenizer.truncation_side = "right"
    tokenizer.model_max_length = model_config.model_max_length

    original_vocab_size = len(tokenizer)

    added_special = _merge_additional_special_tokens(
        tokenizer,
        (
            model_config.image_token,
            model_config.box_start_token,
            model_config.box_end_token,
        ),
    )

    # Keep [SEG] as a normal added vocabulary token, matching original M3D.
    segmentation_added_token = AddedToken(
        model_config.segmentation_token,
        single_word=False,
        lstrip=False,
        rstrip=False,
        normalized=False,
        special=False,
    )
    added_segmentation = int(
        tokenizer.add_tokens([segmentation_added_token], special_tokens=False)
    )

    if tokenizer.pad_token_id is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            added_pad = tokenizer.add_special_tokens({"pad_token": "<pad>"})
            added_special += int(added_pad)

    if tokenizer.eos_token_id is None:
        raise TokenizationError("Phi-3 tokenizer must define an EOS token")
    if tokenizer.pad_token_id is None:
        raise TokenizationError("Unable to establish a tokenizer pad token")

    image_token_id = _token_id(
        tokenizer,
        model_config.image_token,
        name="image token",
    )
    segmentation_token_id = _token_id(
        tokenizer,
        model_config.segmentation_token,
        name="segmentation token",
    )
    box_start_token_id = _token_id(
        tokenizer,
        model_config.box_start_token,
        name="box-start token",
    )
    box_end_token_id = _token_id(
        tokenizer,
        model_config.box_end_token,
        name="box-end token",
    )

    for name, token, token_id in (
        ("image token", model_config.image_token, image_token_id),
        (
            "segmentation token",
            model_config.segmentation_token,
            segmentation_token_id,
        ),
        ("box-start token", model_config.box_start_token, box_start_token_id),
        ("box-end token", model_config.box_end_token, box_end_token_id),
    ):
        _ensure_single_token(tokenizer, token, token_id, name=name)

    m3d_ids = {
        image_token_id,
        segmentation_token_id,
        box_start_token_id,
        box_end_token_id,
    }
    if len(m3d_ids) != 4:
        raise TokenizationError(
            "M3D image, segmentation, box-start, and box-end tokens must have "
            "four distinct token IDs"
        )

    visual_token_count = projected_visual_token_count(model_config)
    image_prefix = model_config.image_token * visual_token_count
    prefix_ids = tokenizer.encode(image_prefix, add_special_tokens=False)
    if len(prefix_ids) != visual_token_count:
        raise TokenizationError(
            "The repeated image placeholder must produce one token per projected "
            f"visual embedding. Expected {visual_token_count}, got {len(prefix_ids)}."
        )
    if any(token_id != image_token_id for token_id in prefix_ids):
        raise TokenizationError(
            "The repeated image placeholder contains IDs other than image_token_id"
        )

    vocabulary_size = len(tokenizer)
    metadata = TokenizerMetadata(
        tokenizer_name_or_path=str(tokenizer_name),
        original_vocab_size=original_vocab_size,
        vocabulary_size=vocabulary_size,
        added_token_count=vocabulary_size - original_vocab_size,
        image_token=model_config.image_token,
        image_token_id=image_token_id,
        segmentation_token=model_config.segmentation_token,
        segmentation_token_id=segmentation_token_id,
        box_start_token=model_config.box_start_token,
        box_start_token_id=box_start_token_id,
        box_end_token=model_config.box_end_token,
        box_end_token_id=box_end_token_id,
        pad_token_id=int(tokenizer.pad_token_id),
        eos_token_id=int(tokenizer.eos_token_id),
        visual_token_count=visual_token_count,
    )

    expected_added = added_special + added_segmentation
    if metadata.added_token_count < expected_added:
        raise TokenizationError(
            "Tokenizer vocabulary-size accounting is inconsistent: "
            f"vocabulary grew by {metadata.added_token_count}, while add calls "
            f"reported at least {expected_added} new tokens"
        )

    return TokenizerBundle(tokenizer=tokenizer, metadata=metadata)


class M3DTextProcessor:
    """Construct M3D prompts and encode variable-length text examples."""

    def __init__(self, bundle: TokenizerBundle, config: ExperimentConfig) -> None:
        self.bundle = bundle
        self.tokenizer = bundle.tokenizer
        self.config = config
        self.max_length = config.model.model_max_length

        if self.tokenizer.padding_side != "right":
            raise TokenizationError("M3D requires right-side padding")
        if self.max_length <= bundle.metadata.visual_token_count:
            raise TokenizationError(
                "model_max_length must exceed the number of visual placeholder "
                f"tokens ({bundle.metadata.visual_token_count})"
            )

    @property
    def image_prefix(self) -> str:
        return self.bundle.image_prefix

    def add_image_prefix(self, question: str) -> str:
        """Return the original-M3D image-token prefix followed by a question."""

        clean_question = question.strip()
        if not clean_question:
            raise ValueError("question cannot be empty")
        return f"{self.image_prefix} {clean_question}"

    def box_text(self, coordinates: Any) -> str:
        """Wrap an existing coordinate representation in M3D box tokens."""

        return (
            f"{self.bundle.metadata.box_start_token}"
            f"{coordinates}"
            f"{self.bundle.metadata.box_end_token}"
        )

    def encode_supervised(
        self,
        question: str,
        answer: str,
        *,
        prepend_image_tokens: bool = True,
        answer_separator: str = DEFAULT_ANSWER_SEPARATOR,
        required_answer_token_ids: Iterable[int] = (),
    ) -> EncodedText:
        """Encode one question-answer example without fixed-length padding.

        With a fast tokenizer, the concatenated sequence is tokenized once and
        character offsets determine which tokens belong to the answer.  This
        avoids the original implementation's second tokenization of ``question``.
        """

        clean_question = question.strip()
        clean_answer = answer.strip()
        if not clean_question:
            raise ValueError("question cannot be empty")
        if not clean_answer:
            raise ValueError("answer cannot be empty")
        if not answer_separator:
            raise ValueError("answer_separator cannot be empty")

        prompt = (
            self.add_image_prefix(clean_question)
            if prepend_image_tokens
            else clean_question
        )

        if getattr(self.tokenizer, "is_fast", False):
            encoded = self._encode_supervised_fast(
                prompt,
                clean_answer,
                answer_separator=answer_separator,
            )
        else:
            encoded = self._encode_supervised_slow(
                prompt,
                clean_answer,
                answer_separator=answer_separator,
            )

        required_ids = tuple(int(token_id) for token_id in required_answer_token_ids)
        if required_ids:
            supervised_ids = set(
                int(token_id)
                for token_id in encoded.labels[encoded.labels.ne(IGNORE_INDEX)].tolist()
            )
            missing = [token_id for token_id in required_ids if token_id not in supervised_ids]
            if missing:
                raise TokenizationError(
                    "Required answer token IDs are absent from supervised labels, "
                    "usually because the answer was truncated. "
                    f"Missing IDs: {missing}; encoded length={encoded.length}, "
                    f"max_length={self.max_length}."
                )

        return encoded

    def encode_segmentation(
        self,
        question: str,
        answer: str,
        *,
        prepend_image_tokens: bool = True,
    ) -> EncodedText:
        """Encode a segmentation example and require supervised ``[SEG]``."""

        return self.encode_supervised(
            question,
            answer,
            prepend_image_tokens=prepend_image_tokens,
            required_answer_token_ids=(
                self.bundle.metadata.segmentation_token_id,
            ),
        )

    def encode_prompt(
        self,
        question: str,
        *,
        prepend_image_tokens: bool = True,
    ) -> EncodedPrompt:
        """Encode an unpadded prompt for generation without appending EOS."""

        clean_question = question.strip()
        if not clean_question:
            raise ValueError("question cannot be empty")
        prompt = (
            self.add_image_prefix(clean_question)
            if prepend_image_tokens
            else clean_question
        )

        result = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=False,
            return_offsets_mapping=bool(getattr(self.tokenizer, "is_fast", False)),
        )
        input_ids = [int(value) for value in result["input_ids"]]
        if not input_ids:
            raise TokenizationError("Tokenizer produced an empty prompt")

        was_truncated = False
        if "offset_mapping" in result:
            offsets = [tuple(map(int, pair)) for pair in result["offset_mapping"]]
            meaningful_ends = [end for start, end in offsets if end > start]
            was_truncated = bool(meaningful_ends and max(meaningful_ends) < len(prompt))

        ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        return EncodedPrompt(
            input_ids=ids_tensor,
            attention_mask=torch.ones_like(ids_tensor, dtype=torch.bool),
            was_truncated=was_truncated,
        )

    def _encode_supervised_fast(
        self,
        prompt: str,
        answer: str,
        *,
        answer_separator: str,
    ) -> EncodedText:
        full_text = f"{prompt}{answer_separator}{answer}"
        answer_boundary = len(prompt)

        result = self.tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )

        input_ids = [int(value) for value in result["input_ids"]]
        offsets = [tuple(map(int, pair)) for pair in result["offset_mapping"]]
        special_mask = [int(value) for value in result["special_tokens_mask"]]

        if not (len(input_ids) == len(offsets) == len(special_mask)):
            raise TokenizationError("Fast tokenizer returned inconsistent field lengths")

        labels = [IGNORE_INDEX for _ in input_ids]
        prompt_token_count = 0
        for index, ((start, end), is_special) in enumerate(
            zip(offsets, special_mask, strict=True)
        ):
            # Any token overlapping the inserted separator or answer belongs to
            # the target side, matching original ``question_len`` masking.
            belongs_to_answer = end > answer_boundary and not is_special
            if belongs_to_answer:
                labels[index] = input_ids[index]
            else:
                prompt_token_count += 1

        meaningful_ends = [end for start, end in offsets if end > start]
        was_truncated = bool(
            meaningful_ends and max(meaningful_ends) < len(full_text)
        )

        input_ids, labels = self._append_eos_when_space_remains(input_ids, labels)
        return self._finish_supervised(
            input_ids,
            labels,
            prompt_token_count=prompt_token_count,
            was_truncated=was_truncated,
        )

    def _encode_supervised_slow(
        self,
        prompt: str,
        answer: str,
        *,
        answer_separator: str,
    ) -> EncodedText:
        prompt_ids = [
            int(value)
            for value in self.tokenizer.encode(prompt, add_special_tokens=True)
        ]
        answer_ids = [
            int(value)
            for value in self.tokenizer.encode(
                f"{answer_separator}{answer}",
                add_special_tokens=False,
            )
        ]

        combined = prompt_ids + answer_ids
        was_truncated = len(combined) > self.max_length
        combined = combined[: self.max_length]

        prompt_token_count = min(len(prompt_ids), len(combined))
        supervised_count_before_eos = max(0, len(combined) - prompt_token_count)
        labels = [IGNORE_INDEX] * prompt_token_count + combined[prompt_token_count:]

        combined, labels = self._append_eos_when_space_remains(combined, labels)
        if supervised_count_before_eos == 0 and len(combined) >= self.max_length:
            raise TokenizationError(
                "The prompt consumed the complete sequence budget and truncated all "
                "answer tokens. Reduce prompt length or increase model_max_length."
            )

        return self._finish_supervised(
            combined,
            labels,
            prompt_token_count=prompt_token_count,
            was_truncated=was_truncated,
        )

    def _append_eos_when_space_remains(
        self,
        input_ids: list[int],
        labels: list[int],
    ) -> tuple[list[int], list[int]]:
        """Reproduce M3D's insertion of EOS into the first padding position."""

        if len(input_ids) < self.max_length:
            input_ids.append(self.bundle.metadata.eos_token_id)
            labels.append(self.bundle.metadata.eos_token_id)
        return input_ids, labels

    @staticmethod
    def _finish_supervised(
        input_ids: Sequence[int],
        labels: Sequence[int],
        *,
        prompt_token_count: int,
        was_truncated: bool,
    ) -> EncodedText:
        if len(input_ids) != len(labels):
            raise TokenizationError("input_ids and labels have different lengths")
        if not input_ids:
            raise TokenizationError("Tokenizer produced an empty supervised sequence")

        ids_tensor = torch.tensor(input_ids, dtype=torch.long)
        labels_tensor = torch.tensor(labels, dtype=torch.long)
        supervised_count = int(labels_tensor.ne(IGNORE_INDEX).sum().item())
        if supervised_count == 0:
            raise TokenizationError(
                "No supervised answer token remains after truncation"
            )

        return EncodedText(
            input_ids=ids_tensor,
            labels=labels_tensor,
            attention_mask=torch.ones_like(ids_tensor, dtype=torch.bool),
            prompt_token_count=prompt_token_count,
            supervised_token_count=supervised_count,
            was_truncated=was_truncated,
        )


def choose_padded_sequence_length(
    unpadded_lengths: Sequence[int],
    *,
    data_config: DataConfig,
    model_max_length: int,
) -> int:
    """Select the smallest configured bucket that can hold the whole batch."""

    if not unpadded_lengths:
        raise ValueError("unpadded_lengths cannot be empty")
    if any(length <= 0 for length in unpadded_lengths):
        raise ValueError(f"all sequence lengths must be positive: {unpadded_lengths}")

    longest = max(int(length) for length in unpadded_lengths)
    if longest > model_max_length:
        raise TokenizationError(
            f"Sequence length {longest} exceeds model_max_length={model_max_length}"
        )

    if not data_config.dynamic_padding:
        return model_max_length

    for bucket in data_config.sequence_length_buckets:
        if longest <= bucket:
            if bucket % data_config.pad_to_multiple_of != 0:
                raise TokenizationError(
                    f"Sequence bucket {bucket} is not divisible by "
                    f"pad_to_multiple_of={data_config.pad_to_multiple_of}"
                )
            return int(bucket)

    raise TokenizationError(
        f"No sequence-length bucket can hold length {longest}; "
        f"buckets={data_config.sequence_length_buckets}"
    )


def pad_supervised_examples(
    examples: Sequence[EncodedText],
    *,
    bundle: TokenizerBundle,
    data_config: DataConfig,
    model_max_length: int,
) -> PaddedTextBatch:
    """Right-pad variable-length examples at collate time.

    Padding labels use ``IGNORE_INDEX`` and attention masks use ``torch.bool``.
    This function allocates no CUDA tensors; asynchronous host-to-device transfer
    is handled later by the training batch object.
    """

    if not examples:
        raise ValueError("examples cannot be empty")

    lengths = [example.length for example in examples]
    target_length = choose_padded_sequence_length(
        lengths,
        data_config=data_config,
        model_max_length=model_max_length,
    )

    batch_size = len(examples)
    input_ids = torch.full(
        (batch_size, target_length),
        fill_value=bundle.metadata.pad_token_id,
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

    for row, example in enumerate(examples):
        length = example.length
        if length > target_length:
            raise TokenizationError(
                f"Example {row} has length {length}, exceeding selected bucket "
                f"{target_length}"
            )
        input_ids[row, :length].copy_(example.input_ids)
        labels[row, :length].copy_(example.labels)
        attention_mask[row, :length] = True

    return PaddedTextBatch(
        input_ids=input_ids,
        labels=labels,
        attention_mask=attention_mask,
        sequence_length=target_length,
        unpadded_lengths=torch.tensor(lengths, dtype=torch.int32),
    )


def pad_generation_prompts(
    prompts: Sequence[EncodedPrompt],
    *,
    bundle: TokenizerBundle,
    data_config: DataConfig,
    model_max_length: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Left-pad generation prompts and return IDs, mask, and true lengths.

    Training uses right padding to preserve original M3D behaviour.  Batched
    generation for a decoder-only language model must instead left-pad so every
    row's final non-padding token is the point from which generation begins.
    """

    if not prompts:
        raise ValueError("prompts cannot be empty")

    lengths = [prompt.length for prompt in prompts]
    target_length = choose_padded_sequence_length(
        lengths,
        data_config=data_config,
        model_max_length=model_max_length,
    )
    input_ids = torch.full(
        (len(prompts), target_length),
        bundle.metadata.pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros(
        (len(prompts), target_length),
        dtype=torch.bool,
    )
    for row, prompt in enumerate(prompts):
        start = target_length - prompt.length
        input_ids[row, start:].copy_(prompt.input_ids)
        attention_mask[row, start:] = True

    return (
        input_ids,
        attention_mask,
        torch.tensor(lengths, dtype=torch.int32),
    )
