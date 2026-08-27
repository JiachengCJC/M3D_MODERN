"""Phi-3 language-model integration for M3D-Modernized.

This module owns the language side of the multimodal model.  It deliberately
keeps image encoding outside Phi-3: the independent Main 3D ViT and multimodal
projector produce ``[B, N_visual, hidden]`` embeddings, and this wrapper replaces
only the validated ``<im_patch>`` token embeddings before running the decoder.
The independent SegVol image encoder is not referenced here.

Compared with the original M3D implementation, this module makes several
execution details explicit:

* image features are inserted by validated token IDs rather than by assuming a
  fixed ``BOS + 256 placeholders`` slice;
* Phi-3 is loaded with Hugging Face's native SDPA implementation;
* language activation checkpointing is configured independently from both 3-D
  ViTs;
* LoRA is attached only after tokenizer-driven embedding resize;
* newly added input/output token rows are initialised with the old vocabulary
  mean, reproducing M3D's initialisation;
* training can project only supervised causal positions through ``lm_head``.
  This is mathematically identical to full-vocabulary causal cross entropy with
  ``ignore_index=-100`` while avoiding logits for image/prompt/padding tokens;
* the last decoder hidden state is returned directly, without requesting and
  retaining hidden states from every Phi-3 layer.  The segmentation path can
  therefore extract ``[SEG]`` prompts without ``output_hidden_states=True``.

The file imports Transformers and PEFT lazily.  Consequently data-pipeline and
CPU contract tests can import this module on machines where the large language
model dependencies have not yet been installed.  Model construction still
fails immediately with a clear error if the pinned dependencies are absent.
"""

from __future__ import annotations

import argparse
import inspect
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Final, Iterable, Mapping, MutableMapping, Protocol, Sequence, cast

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from m3d.config import ExperimentConfig, LoRAConfig, ModelConfig, OptimizationConfig
if TYPE_CHECKING:
    from m3d.tokenization import TokenizerBundle, TokenizerMetadata


IGNORE_INDEX: Final[int] = -100
SUPPORTED_ATTENTION_IMPLEMENTATION: Final[str] = "sdpa"
DEFAULT_LORA_ADAPTER_NAME: Final[str] = "default"


class LanguageModelError(RuntimeError):
    """Base error for Phi-3 construction and execution."""


class LanguageDependencyError(LanguageModelError, ImportError):
    """Raised when the pinned Transformers or PEFT dependency is unavailable."""


class LanguageConfigurationError(LanguageModelError, ValueError):
    """Raised when model, tokenizer, LoRA, or checkpoint settings disagree."""


class LanguageInputError(LanguageModelError, ValueError):
    """Raised when token tensors or projected image features violate contracts."""


class LogitsMode(str, Enum):
    """Which vocabulary logits should be retained in the returned output."""

    NONE = "none"
    SUPERVISED = "supervised"
    FULL = "full"

    @classmethod
    def parse(cls, value: str | "LogitsMode") -> "LogitsMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            allowed = ", ".join(item.value for item in cls)
            raise LanguageConfigurationError(
                f"Unknown logits mode {value!r}; expected one of: {allowed}."
            ) from exc


@dataclass(frozen=True, slots=True)
class MultimodalEmbeddingOutput:
    """Token embeddings after visual placeholders have been replaced."""

    inputs_embeds: Tensor
    image_token_mask: Tensor
    image_token_counts: Tensor

    @property
    def batch_size(self) -> int:
        return int(self.inputs_embeds.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.inputs_embeds.shape[1])

    @property
    def hidden_size(self) -> int:
        return int(self.inputs_embeds.shape[2])


@dataclass(frozen=True, slots=True)
class CausalLanguageLoss:
    """Exact causal-LM loss and the positions used to compute it."""

    mean: Tensor
    summed: Tensor
    token_count: Tensor
    supervised_logits: Tensor | None
    supervised_labels: Tensor
    supervised_hidden_states: Tensor

    @property
    def count(self) -> int:
        return int(self.supervised_labels.numel())


@dataclass(frozen=True, slots=True)
class LanguageModelOutput:
    """Language output consumed by the complete M3D model."""

    loss: Tensor | None
    loss_sum: Tensor | None
    supervised_token_count: Tensor | None
    last_hidden_state: Tensor
    logits: Tensor | None
    supervised_labels: Tensor | None
    image_token_counts: Tensor
    past_key_values: Any | None = None

    @property
    def batch_size(self) -> int:
        return int(self.last_hidden_state.shape[0])

    @property
    def sequence_length(self) -> int:
        return int(self.last_hidden_state.shape[1])

    @property
    def hidden_size(self) -> int:
        return int(self.last_hidden_state.shape[2])


@dataclass(frozen=True, slots=True)
class TokenResizeReport:
    """Auditable tokenizer/model vocabulary resize result."""

    old_vocabulary_size: int
    new_vocabulary_size: int
    added_token_count: int
    input_rows_initialised_from_mean: tuple[int, ...]
    output_rows_initialised_from_mean: tuple[int, ...]
    tied_input_output_embeddings: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "old_vocabulary_size": self.old_vocabulary_size,
            "new_vocabulary_size": self.new_vocabulary_size,
            "added_token_count": self.added_token_count,
            "input_rows_initialised_from_mean": list(
                self.input_rows_initialised_from_mean
            ),
            "output_rows_initialised_from_mean": list(
                self.output_rows_initialised_from_mean
            ),
            "tied_input_output_embeddings": self.tied_input_output_embeddings,
        }


@dataclass(frozen=True, slots=True)
class LanguageModelBuildReport:
    """Serializable description of the loaded Phi-3 and its trainable subset."""

    model_name_or_path: str
    model_type: str
    hidden_size: int
    vocabulary_size: int
    attention_implementation: str
    gradient_checkpointing: bool
    use_cache: bool
    lora_enabled: bool
    lora_rank: int | None
    lora_alpha: int | None
    lora_target_modules: tuple[str, ...]
    matched_lora_module_count: int
    total_parameter_count: int
    trainable_parameter_count: int
    token_resize: TokenResizeReport

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name_or_path": self.model_name_or_path,
            "model_type": self.model_type,
            "hidden_size": self.hidden_size,
            "vocabulary_size": self.vocabulary_size,
            "attention_implementation": self.attention_implementation,
            "gradient_checkpointing": self.gradient_checkpointing,
            "use_cache": self.use_cache,
            "lora_enabled": self.lora_enabled,
            "lora_rank": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_target_modules": list(self.lora_target_modules),
            "matched_lora_module_count": self.matched_lora_module_count,
            "total_parameter_count": self.total_parameter_count,
            "trainable_parameter_count": self.trainable_parameter_count,
            "token_resize": self.token_resize.to_dict(),
        }


class _TokenizerMetadataProtocol(Protocol):
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


class _TokenizerBundleProtocol(Protocol):
    metadata: _TokenizerMetadataProtocol


class _DecoderOutputProtocol(Protocol):
    last_hidden_state: Tensor
    past_key_values: Any


class _CausalLMProtocol(Protocol):
    config: Any

    def get_input_embeddings(self) -> nn.Module: ...

    def get_output_embeddings(self) -> nn.Module: ...

    def get_decoder(self) -> nn.Module: ...

    def resize_token_embeddings(self, new_num_tokens: int, **kwargs: Any) -> Any: ...


# ---------------------------------------------------------------------------
# Optional dependency loading
# ---------------------------------------------------------------------------


def _import_transformers() -> tuple[Any, Any]:
    try:
        from transformers import AutoConfig, AutoModelForCausalLM
    except ImportError as exc:  # pragma: no cover - exercised on cluster setup
        raise LanguageDependencyError(
            "Transformers is required to build Phi-3. Install the pinned "
            "requirements.txt in the ASPIRE 2A environment first."
        ) from exc
    return AutoConfig, AutoModelForCausalLM


def _import_peft() -> tuple[Any, Any, Any, Any]:
    try:
        from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    except ImportError as exc:  # pragma: no cover - exercised on cluster setup
        raise LanguageDependencyError(
            "PEFT is required because model.lora.enabled=true. Install the "
            "pinned requirements.txt before constructing the model."
        ) from exc
    return LoraConfig, PeftModel, TaskType, get_peft_model


# ---------------------------------------------------------------------------
# Generic Hugging Face model accessors
# ---------------------------------------------------------------------------


def unwrap_peft_model(module: nn.Module) -> nn.Module:
    """Return the underlying causal LM while retaining injected LoRA modules."""

    get_base_model = getattr(module, "get_base_model", None)
    if callable(get_base_model):
        base = get_base_model()
        if isinstance(base, nn.Module):
            return base
    return module


def get_decoder_module(causal_lm: nn.Module) -> nn.Module:
    """Locate Phi-3's decoder without depending on one wrapper layout."""

    base = unwrap_peft_model(causal_lm)
    getter = getattr(base, "get_decoder", None)
    if callable(getter):
        decoder = getter()
        if isinstance(decoder, nn.Module):
            return decoder

    prefix = str(getattr(getattr(base, "config", None), "base_model_prefix", ""))
    if prefix and isinstance(getattr(base, prefix, None), nn.Module):
        return cast(nn.Module, getattr(base, prefix))

    candidate = getattr(base, "model", None)
    if isinstance(candidate, nn.Module):
        return candidate

    raise LanguageConfigurationError(
        f"Cannot locate the decoder module inside {type(causal_lm).__name__}."
    )


def get_input_embedding_module(causal_lm: nn.Module) -> nn.Module:
    getter = getattr(causal_lm, "get_input_embeddings", None)
    if callable(getter):
        module = getter()
        if isinstance(module, nn.Module):
            return module

    base = unwrap_peft_model(causal_lm)
    getter = getattr(base, "get_input_embeddings", None)
    if callable(getter):
        module = getter()
        if isinstance(module, nn.Module):
            return module
    raise LanguageConfigurationError("Language model does not expose input embeddings.")


def get_output_embedding_module(causal_lm: nn.Module) -> nn.Module:
    getter = getattr(causal_lm, "get_output_embeddings", None)
    if callable(getter):
        module = getter()
        if isinstance(module, nn.Module):
            return module

    base = unwrap_peft_model(causal_lm)
    getter = getattr(base, "get_output_embeddings", None)
    if callable(getter):
        module = getter()
        if isinstance(module, nn.Module):
            return module
    raise LanguageConfigurationError("Language model does not expose an LM head.")


def _module_weight(module: nn.Module, *, label: str) -> Tensor:
    weight = getattr(module, "weight", None)
    if not isinstance(weight, Tensor):
        raise LanguageConfigurationError(
            f"{label} ({type(module).__name__}) does not expose a tensor weight."
        )
    if weight.ndim != 2:
        raise LanguageConfigurationError(
            f"{label} weight must be rank 2, got {tuple(weight.shape)}."
        )
    return weight


def _extract_last_hidden_state(outputs: Any) -> Tensor:
    value = getattr(outputs, "last_hidden_state", None)
    if isinstance(value, Tensor):
        return value
    if isinstance(outputs, (tuple, list)) and outputs and isinstance(outputs[0], Tensor):
        return outputs[0]
    raise LanguageModelError(
        "Phi-3 decoder did not return a last_hidden_state tensor."
    )


def _extract_past_key_values(outputs: Any) -> Any | None:
    return getattr(outputs, "past_key_values", None)


# ---------------------------------------------------------------------------
# Vocabulary resize and LoRA configuration
# ---------------------------------------------------------------------------


def _weights_share_storage(first: Tensor, second: Tensor) -> bool:
    if first.device.type == "meta" or second.device.type == "meta":
        return first is second
    try:
        return first.untyped_storage().data_ptr() == second.untyped_storage().data_ptr()
    except RuntimeError:
        return first is second


@torch.no_grad()
def resize_and_initialise_m3d_tokens(
    causal_lm: nn.Module,
    metadata: _TokenizerMetadataProtocol,
) -> TokenResizeReport:
    """Resize embeddings and initialise every added row from old-row means.

    M3D's original tokenizer initialisation averages the pre-existing input and
    output embeddings and writes those averages into the newly added token rows.
    The operation is performed before LoRA wrapping so PEFT sees final shapes.
    """

    input_before = get_input_embedding_module(causal_lm)
    output_before = get_output_embedding_module(causal_lm)
    input_weight_before = _module_weight(input_before, label="input embedding")
    output_weight_before = _module_weight(output_before, label="output embedding")

    old_vocab = int(input_weight_before.shape[0])
    tokenizer_base_vocab = int(metadata.original_vocab_size)
    target_vocab = int(metadata.vocabulary_size)

    if int(output_weight_before.shape[0]) != old_vocab:
        raise LanguageConfigurationError(
            "Input and output vocabulary sizes differ before resize: "
            f"input={tuple(input_weight_before.shape)}, "
            f"output={tuple(output_weight_before.shape)}."
        )

    if tokenizer_base_vocab > old_vocab:
        raise LanguageConfigurationError(
            "The original tokenizer vocabulary exceeds the pretrained model "
            f"capacity: tokenizer={tokenizer_base_vocab}, model={old_vocab}."
        )

    if target_vocab < tokenizer_base_vocab:
        raise LanguageConfigurationError(
            "The tokenizer vocabulary became smaller after adding M3D tokens: "
            f"original={tokenizer_base_vocab}, current={target_vocab}."
        )

    if int(metadata.added_token_count) != target_vocab - tokenizer_base_vocab:
        raise LanguageConfigurationError(
            "Tokenizer added-token accounting is inconsistent: "
            f"original={tokenizer_base_vocab}, current={target_vocab}, "
            f"reported_added={metadata.added_token_count}."
        )

    # Only average real tokenizer rows. Do not include Phi-3's unused padded rows.
    input_mean = (
        input_weight_before[:tokenizer_base_vocab]
        .float()
        .mean(dim=0, keepdim=True)
    )
    output_mean = (
        output_weight_before[:tokenizer_base_vocab]
        .float()
        .mean(dim=0, keepdim=True)
    )

    if target_vocab != old_vocab:
        resize = getattr(causal_lm, "resize_token_embeddings", None)
        if not callable(resize):
            base = unwrap_peft_model(causal_lm)
            resize = getattr(base, "resize_token_embeddings", None)
        if not callable(resize):
            raise LanguageConfigurationError(
                "Language model cannot resize token embeddings."
            )

        # Transformers versions differ on the optional mean_resizing keyword.
        # Explicit post-resize initialisation below is the source of truth.
        signature = inspect.signature(resize)
        kwargs: dict[str, Any] = {}
        if "mean_resizing" in signature.parameters:
            kwargs["mean_resizing"] = False
        resize(target_vocab, **kwargs)

    input_after = get_input_embedding_module(causal_lm)
    output_after = get_output_embedding_module(causal_lm)
    input_weight = _module_weight(input_after, label="resized input embedding")
    output_weight = _module_weight(output_after, label="resized output embedding")

    expected_input_shape = (target_vocab, int(input_weight_before.shape[1]))
    expected_output_shape = (target_vocab, int(output_weight_before.shape[1]))
    if tuple(input_weight.shape) != expected_input_shape:
        raise LanguageConfigurationError(
            "Unexpected resized input embedding shape: "
            f"expected={expected_input_shape}, got={tuple(input_weight.shape)}."
        )
    if tuple(output_weight.shape) != expected_output_shape:
        raise LanguageConfigurationError(
            "Unexpected resized output embedding shape: "
            f"expected={expected_output_shape}, got={tuple(output_weight.shape)}."
        )

    new_rows = tuple(range(tokenizer_base_vocab, target_vocab))
    if new_rows:
        input_weight[tokenizer_base_vocab:target_vocab].copy_(
            input_mean.to(input_weight.dtype)
        )
        output_weight[tokenizer_base_vocab:target_vocab].copy_(
            output_mean.to(output_weight.dtype)
        )

    configuration = getattr(causal_lm, "config", None)
    if configuration is not None:
        configuration.vocab_size = target_vocab
        configuration.pad_token_id = int(metadata.pad_token_id)
        configuration.eos_token_id = int(metadata.eos_token_id)

    return TokenResizeReport(
        old_vocabulary_size=old_vocab,
        new_vocabulary_size=target_vocab,
        added_token_count=int(metadata.added_token_count),
        input_rows_initialised_from_mean=new_rows,
        output_rows_initialised_from_mean=new_rows,
        tied_input_output_embeddings=_weights_share_storage(
            input_weight,
            output_weight,
        ),
    )


def resolve_lora_target_modules(
    causal_lm: nn.Module,
    requested_suffixes: Sequence[str],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Validate that every requested Phi-3 module suffix exists."""

    suffixes = tuple(dict.fromkeys(str(item).strip() for item in requested_suffixes))
    if not suffixes or any(not item for item in suffixes):
        raise LanguageConfigurationError("LoRA target_modules cannot be empty.")

    matched_names = tuple(
        name
        for name, module in causal_lm.named_modules()
        if isinstance(module, nn.Linear)
        and any(name == suffix or name.endswith("." + suffix) for suffix in suffixes)
    )
    missing = [
        suffix
        for suffix in suffixes
        if not any(name == suffix or name.endswith("." + suffix) for name in matched_names)
    ]
    if missing:
        available_suffixes = sorted(
            {
                name.rsplit(".", 1)[-1]
                for name, module in causal_lm.named_modules()
                if isinstance(module, nn.Linear)
            }
        )
        raise LanguageConfigurationError(
            "Requested LoRA target modules are absent from the loaded language "
            f"model: missing={missing}, available_linear_suffixes={available_suffixes}."
        )
    return suffixes, matched_names


def apply_lora_or_load_adapter(
    causal_lm: nn.Module,
    lora_config: LoRAConfig,
    *,
    added_token_count: int,
) -> tuple[nn.Module, tuple[str, ...]]:
    """Attach fresh LoRA layers or load a saved trainable adapter.

    ``embed_tokens`` and ``lm_head`` remain modules-to-save when new M3D tokens
    were added.  This reproduces the original training behaviour in which both
    complete embedding tables are trainable, while ensuring standard PEFT
    adapter checkpoints include them.
    """

    if not lora_config.enabled:
        return causal_lm, ()

    LoraConfig, PeftModel, TaskType, get_peft_model = _import_peft()
    target_suffixes, matched_names = resolve_lora_target_modules(
        causal_lm,
        lora_config.target_modules,
    )

    adapter_path = lora_config.adapter_checkpoint_path
    if adapter_path is not None:
        adapter_dir = Path(adapter_path).expanduser().resolve()
        if not (adapter_dir / "adapter_config.json").is_file():
            raise FileNotFoundError(
                f"LoRA adapter_config.json is missing from {adapter_dir}."
            )
        wrapped = PeftModel.from_pretrained(
            causal_lm,
            str(adapter_dir),
            is_trainable=True,
            adapter_name=DEFAULT_LORA_ADAPTER_NAME,
        )
        return cast(nn.Module, wrapped), matched_names

    modules_to_save = ["embed_tokens", "lm_head"] if added_token_count > 0 else None
    peft_configuration = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=int(lora_config.rank),
        lora_alpha=int(lora_config.alpha),
        lora_dropout=float(lora_config.dropout),
        bias=str(lora_config.bias),
        target_modules=list(target_suffixes),
        modules_to_save=modules_to_save,
        init_lora_weights=True,
    )
    wrapped = get_peft_model(causal_lm, peft_configuration)
    return cast(nn.Module, wrapped), matched_names


def configure_language_gradient_checkpointing(
    causal_lm: nn.Module,
    *,
    enabled: bool,
) -> None:
    """Configure Phi-3 activation checkpointing and cache compatibility."""

    configuration = getattr(causal_lm, "config", None)
    if configuration is not None:
        configuration.use_cache = not enabled

    if enabled:
        method = getattr(causal_lm, "gradient_checkpointing_enable", None)
        if not callable(method):
            base = unwrap_peft_model(causal_lm)
            method = getattr(base, "gradient_checkpointing_enable", None)
        if not callable(method):
            raise LanguageConfigurationError(
                "Loaded language model does not support gradient checkpointing."
            )
        signature = inspect.signature(method)
        if "gradient_checkpointing_kwargs" in signature.parameters:
            method(gradient_checkpointing_kwargs={"use_reentrant": False})
        else:  # pragma: no cover - retained for older compatible Transformers
            method()

        enable_inputs = getattr(causal_lm, "enable_input_require_grads", None)
        if callable(enable_inputs):
            enable_inputs()
    else:
        method = getattr(causal_lm, "gradient_checkpointing_disable", None)
        if callable(method):
            method()


# ---------------------------------------------------------------------------
# Multimodal embedding replacement
# ---------------------------------------------------------------------------


def _validate_token_inputs(
    input_ids: Tensor,
    attention_mask: Tensor,
    labels: Tensor | None,
) -> None:
    if input_ids.ndim != 2:
        raise LanguageInputError(
            f"input_ids must have shape [B,S], got {tuple(input_ids.shape)}."
        )
    if input_ids.dtype == torch.bool or input_ids.is_floating_point():
        raise LanguageInputError(
            f"input_ids must use an integer dtype, got {input_ids.dtype}."
        )
    if attention_mask.shape != input_ids.shape:
        raise LanguageInputError(
            "attention_mask must match input_ids: "
            f"{tuple(attention_mask.shape)} vs {tuple(input_ids.shape)}."
        )
    if attention_mask.device != input_ids.device:
        raise LanguageInputError("attention_mask and input_ids are on different devices.")
    if labels is not None:
        if labels.shape != input_ids.shape:
            raise LanguageInputError(
                f"labels must match input_ids, got {tuple(labels.shape)}."
            )
        if labels.device != input_ids.device:
            raise LanguageInputError("labels and input_ids are on different devices.")
        if labels.dtype == torch.bool or labels.is_floating_point():
            raise LanguageInputError(
                f"labels must use an integer dtype, got {labels.dtype}."
            )


def inject_visual_embeddings(
    *,
    token_embeddings: Tensor,
    input_ids: Tensor,
    attention_mask: Tensor,
    visual_embeddings: Tensor,
    image_token_id: int,
    expected_visual_token_count: int,
    labels: Tensor | None = None,
) -> MultimodalEmbeddingOutput:
    """Replace every valid ``<im_patch>`` embedding with projected 3-D features.

    Boolean selection is row-major, so flattening ``[B,N,H]`` projected tokens
    exactly matches the per-sample token order selected from ``[B,S]``.
    """

    _validate_token_inputs(input_ids, attention_mask, labels)
    if token_embeddings.ndim != 3:
        raise LanguageInputError(
            "token_embeddings must have shape [B,S,H], got "
            f"{tuple(token_embeddings.shape)}."
        )
    if tuple(token_embeddings.shape[:2]) != tuple(input_ids.shape):
        raise LanguageInputError(
            "token_embeddings batch/sequence dimensions must match input_ids."
        )
    if visual_embeddings.ndim != 3:
        raise LanguageInputError(
            "visual_embeddings must have shape [B,N,H], got "
            f"{tuple(visual_embeddings.shape)}."
        )
    if int(visual_embeddings.shape[0]) != int(input_ids.shape[0]):
        raise LanguageInputError(
            "visual_embeddings batch size does not match input_ids: "
            f"{int(visual_embeddings.shape[0])} vs {int(input_ids.shape[0])}."
        )
    if int(visual_embeddings.shape[1]) != int(expected_visual_token_count):
        raise LanguageInputError(
            "Projector visual-token count is incompatible with the tokenizer: "
            f"expected={expected_visual_token_count}, "
            f"got={int(visual_embeddings.shape[1])}."
        )
    if int(visual_embeddings.shape[2]) != int(token_embeddings.shape[2]):
        raise LanguageInputError(
            "Projected visual hidden size differs from Phi-3 hidden size: "
            f"visual={int(visual_embeddings.shape[2])}, "
            f"language={int(token_embeddings.shape[2])}."
        )
    if visual_embeddings.device != token_embeddings.device:
        raise LanguageInputError(
            "visual_embeddings and token_embeddings must be on the same device."
        )
    if not visual_embeddings.is_floating_point():
        raise LanguageInputError("visual_embeddings must use a floating-point dtype.")
    if not torch.isfinite(visual_embeddings).all():
        raise LanguageInputError("visual_embeddings contain NaN or Inf values.")

    valid_tokens = attention_mask.to(dtype=torch.bool)
    image_mask = input_ids.eq(int(image_token_id)) & valid_tokens
    counts = image_mask.sum(dim=1)
    expected = torch.full_like(counts, int(expected_visual_token_count))
    if not torch.equal(counts, expected):
        bad_rows = torch.nonzero(counts.ne(expected), as_tuple=False).flatten()
        details = [
            (int(row), int(counts[row]))
            for row in bad_rows.detach().cpu().tolist()
        ]
        raise LanguageInputError(
            "Every sample must contain exactly the projector's number of valid "
            f"<im_patch> tokens ({expected_visual_token_count}); bad rows/counts={details}."
        )

    if labels is not None and torch.any(labels[image_mask].ne(IGNORE_INDEX)):
        raise LanguageInputError(
            "Image placeholder positions must use label=-100; they are replaced "
            "by visual features and must not contribute to language loss."
        )

    replacement = visual_embeddings.to(dtype=token_embeddings.dtype)
    result = token_embeddings.clone()
    # The boolean mask traverses rows first and positions second, exactly the
    # same order as replacement.reshape(B*N, H).
    result[image_mask] = replacement.reshape(-1, replacement.shape[-1])
    return MultimodalEmbeddingOutput(
        inputs_embeds=result,
        image_token_mask=image_mask,
        image_token_counts=counts,
    )


# ---------------------------------------------------------------------------
# Exact supervised-position causal loss
# ---------------------------------------------------------------------------


def compute_causal_language_loss(
    *,
    last_hidden_state: Tensor,
    labels: Tensor,
    lm_head: nn.Module,
    return_supervised_logits: bool = False,
) -> CausalLanguageLoss:
    """Compute causal cross entropy only where shifted labels are supervised.

    Standard causal-LM loss forms ``logits[:, :-1]`` and ``labels[:, 1:]`` and
    ignores every target equal to ``-100``.  Applying ``lm_head`` only to hidden
    rows whose shifted labels are not ``-100`` is exactly equivalent, but avoids
    allocating ``[B,S,V]`` logits for visual placeholders, question text, and
    dynamic padding.
    """

    if last_hidden_state.ndim != 3:
        raise LanguageInputError(
            "last_hidden_state must have shape [B,S,H], got "
            f"{tuple(last_hidden_state.shape)}."
        )
    if labels.ndim != 2 or tuple(labels.shape) != tuple(last_hidden_state.shape[:2]):
        raise LanguageInputError(
            "labels must have shape [B,S] matching last_hidden_state; got "
            f"labels={tuple(labels.shape)}, hidden={tuple(last_hidden_state.shape)}."
        )
    if labels.device != last_hidden_state.device:
        raise LanguageInputError("labels and last_hidden_state are on different devices.")
    if int(last_hidden_state.shape[1]) < 2:
        raise LanguageInputError("Causal language loss requires sequence length >= 2.")

    shifted_hidden = last_hidden_state[:, :-1, :]
    shifted_labels = labels[:, 1:]
    supervised_mask = shifted_labels.ne(IGNORE_INDEX)
    selected_hidden = shifted_hidden[supervised_mask]
    selected_labels = shifted_labels[supervised_mask]

    if selected_labels.numel() == 0:
        raise LanguageInputError(
            "The batch contains no supervised causal target after shifting labels."
        )

    logits = lm_head(selected_hidden)
    if logits.ndim != 2 or int(logits.shape[0]) != int(selected_labels.numel()):
        raise LanguageModelError(
            "LM head returned an unexpected shape for supervised hidden states: "
            f"hidden={tuple(selected_hidden.shape)}, logits={tuple(logits.shape)}."
        )

    # Cross entropy accumulates in FP32 for stable BF16 training.
    loss_sum = F.cross_entropy(
        logits.float(),
        selected_labels,
        reduction="sum",
    )
    count = torch.tensor(
        selected_labels.numel(),
        dtype=torch.int64,
        device=selected_labels.device,
    )
    mean = loss_sum / count.to(dtype=loss_sum.dtype)
    return CausalLanguageLoss(
        mean=mean,
        summed=loss_sum,
        token_count=count,
        supervised_logits=logits if return_supervised_logits else None,
        supervised_labels=selected_labels,
        supervised_hidden_states=selected_hidden,
    )


def compute_full_causal_language_loss(
    *,
    full_logits: Tensor,
    labels: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Reference full-logit loss used for evaluation and equivalence tests."""

    if full_logits.ndim != 3:
        raise LanguageInputError("full_logits must have shape [B,S,V].")
    if tuple(full_logits.shape[:2]) != tuple(labels.shape):
        raise LanguageInputError("full_logits and labels batch/sequence shapes differ.")
    shifted_logits = full_logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    count = shifted_labels.ne(IGNORE_INDEX).sum()
    if int(count) == 0:
        raise LanguageInputError("No supervised targets remain after causal shift.")
    loss_sum = F.cross_entropy(
        shifted_logits.float().reshape(-1, shifted_logits.shape[-1]),
        shifted_labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="sum",
    )
    return loss_sum / count.to(loss_sum.dtype), loss_sum, count


# ---------------------------------------------------------------------------
# Phi-3 wrapper
# ---------------------------------------------------------------------------


class M3DLanguageModel(nn.Module):
    """Phi-3 wrapper with validated multimodal embedding insertion."""

    def __init__(
        self,
        causal_lm: nn.Module,
        *,
        tokenizer_metadata: _TokenizerMetadataProtocol,
    ) -> None:
        super().__init__()
        self.causal_lm = causal_lm
        self.tokenizer_metadata = tokenizer_metadata

        configuration = getattr(causal_lm, "config", None)
        hidden_size = getattr(configuration, "hidden_size", None)
        vocab_size = getattr(configuration, "vocab_size", None)
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise LanguageConfigurationError(
                "Loaded language model config does not define a positive hidden_size."
            )
        if not isinstance(vocab_size, int) or vocab_size <= 0:
            raise LanguageConfigurationError(
                "Loaded language model config does not define a positive vocab_size."
            )
        if vocab_size != tokenizer_metadata.vocabulary_size:
            raise LanguageConfigurationError(
                "Tokenizer and language-model vocabulary sizes differ after resize: "
                f"tokenizer={tokenizer_metadata.vocabulary_size}, model={vocab_size}."
            )

        self.hidden_size = hidden_size
        self.vocabulary_size = vocab_size
        self.image_token_id = int(tokenizer_metadata.image_token_id)
        self.segmentation_token_id = int(tokenizer_metadata.segmentation_token_id)
        self.visual_token_count = int(tokenizer_metadata.visual_token_count)

    @property
    def config(self) -> Any:
        return getattr(self.causal_lm, "config")

    def get_input_embeddings(self) -> nn.Module:
        return get_input_embedding_module(self.causal_lm)

    def get_output_embeddings(self) -> nn.Module:
        return get_output_embedding_module(self.causal_lm)

    def get_decoder(self) -> nn.Module:
        return get_decoder_module(self.causal_lm)

    def prepare_multimodal_embeddings(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        visual_embeddings: Tensor,
        labels: Tensor | None = None,
    ) -> MultimodalEmbeddingOutput:
        token_embeddings = self.get_input_embeddings()(input_ids)
        if not isinstance(token_embeddings, Tensor):
            raise LanguageModelError("Input embedding module did not return a tensor.")
        return inject_visual_embeddings(
            token_embeddings=token_embeddings,
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeddings=visual_embeddings,
            image_token_id=self.image_token_id,
            expected_visual_token_count=self.visual_token_count,
            labels=labels,
        )

    def forward(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        visual_embeddings: Tensor,
        labels: Tensor | None = None,
        position_ids: Tensor | None = None,
        use_cache: bool = False,
        logits_mode: LogitsMode | str = LogitsMode.NONE,
    ) -> LanguageModelOutput:
        """Run Phi-3 and optionally compute memory-efficient causal loss."""

        mode = LogitsMode.parse(logits_mode)
        if self.training and use_cache:
            raise LanguageConfigurationError(
                "use_cache=True is disabled during training because it conflicts "
                "with activation checkpointing and wastes memory."
            )

        multimodal = self.prepare_multimodal_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeddings=visual_embeddings,
            labels=labels,
        )

        decoder = self.get_decoder()
        decoder_kwargs: dict[str, Any] = {
            "input_ids": None,
            "inputs_embeds": multimodal.inputs_embeds,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "use_cache": use_cache,
            "output_attentions": False,
            "output_hidden_states": False,
            "return_dict": True,
        }
        # Some lightweight test decoders and older compatible models do not
        # expose every keyword.  Real Phi-3 accepts this complete set.
        signature = inspect.signature(decoder.forward)
        if not any(
            parameter.kind is inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ):
            decoder_kwargs = {
                key: value
                for key, value in decoder_kwargs.items()
                if key in signature.parameters
            }

        decoder_outputs = decoder(**decoder_kwargs)
        last_hidden_state = _extract_last_hidden_state(decoder_outputs)
        if tuple(last_hidden_state.shape[:2]) != tuple(input_ids.shape):
            raise LanguageModelError(
                "Phi-3 last hidden state has unexpected batch/sequence shape: "
                f"hidden={tuple(last_hidden_state.shape)}, ids={tuple(input_ids.shape)}."
            )
        if int(last_hidden_state.shape[2]) != self.hidden_size:
            raise LanguageModelError(
                "Phi-3 last hidden size changed unexpectedly: "
                f"expected={self.hidden_size}, got={int(last_hidden_state.shape[2])}."
            )

        lm_head = self.get_output_embeddings()
        loss_output: CausalLanguageLoss | None = None
        if labels is not None:
            loss_output = compute_causal_language_loss(
                last_hidden_state=last_hidden_state,
                labels=labels,
                lm_head=lm_head,
                return_supervised_logits=mode is LogitsMode.SUPERVISED,
            )

        logits: Tensor | None
        if mode is LogitsMode.NONE:
            logits = None
        elif mode is LogitsMode.SUPERVISED:
            if loss_output is None:
                raise LanguageInputError(
                    "logits_mode='supervised' requires labels."
                )
            logits = loss_output.supervised_logits
        else:
            logits = lm_head(last_hidden_state)

        return LanguageModelOutput(
            loss=None if loss_output is None else loss_output.mean,
            loss_sum=None if loss_output is None else loss_output.summed,
            supervised_token_count=(
                None if loss_output is None else loss_output.token_count
            ),
            last_hidden_state=last_hidden_state,
            logits=logits,
            supervised_labels=(
                None if loss_output is None else loss_output.supervised_labels
            ),
            image_token_counts=multimodal.image_token_counts,
            past_key_values=_extract_past_key_values(decoder_outputs),
        )

    @torch.no_grad()
    def generate(
        self,
        *,
        input_ids: Tensor,
        attention_mask: Tensor,
        visual_embeddings: Tensor,
        **generation_kwargs: Any,
    ) -> Any:
        """Generate text while using visual embeddings on the first decode step."""

        multimodal = self.prepare_multimodal_embeddings(
            input_ids=input_ids,
            attention_mask=attention_mask,
            visual_embeddings=visual_embeddings,
            labels=None,
        )
        generate = getattr(self.causal_lm, "generate", None)
        if not callable(generate):
            raise LanguageConfigurationError(
                "Loaded language model does not provide GenerationMixin.generate()."
            )

        kwargs = dict(generation_kwargs)
        kwargs.setdefault("use_cache", True)
        return generate(
            input_ids=input_ids,
            inputs_embeds=multimodal.inputs_embeds,
            attention_mask=attention_mask,
            **kwargs,
        )


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _parameter_counts(module: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(
        parameter.numel() for parameter in module.parameters() if parameter.requires_grad
    )
    return int(total), int(trainable)


def _normalise_model_type(value: Any) -> str:
    return str(value).strip().lower().replace("-", "_")


def build_language_model(
    config: ExperimentConfig,
    tokenizer_bundle: _TokenizerBundleProtocol,
    *,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
) -> tuple[M3DLanguageModel, LanguageModelBuildReport]:
    """Load Phi-3, resize tokens, attach LoRA, and return the M3D wrapper."""

    if config.model.language_model_family != "phi3":
        raise LanguageConfigurationError(
            "This reproducible M3D implementation currently supports only Phi-3."
        )
    if torch_dtype is not torch.bfloat16:
        raise LanguageConfigurationError(
            "The ASPIRE 2A training contract requires torch.bfloat16 for Phi-3."
        )

    AutoConfig, AutoModelForCausalLM = _import_transformers()
    model_name = str(config.model.language_model_name_or_path)
    cache = None if cache_dir is None else str(Path(cache_dir))

    hf_config = AutoConfig.from_pretrained(
        model_name,
        cache_dir=cache,
        trust_remote_code=config.model.trust_remote_code,
        local_files_only=local_files_only,
    )
    model_type = _normalise_model_type(getattr(hf_config, "model_type", ""))
    if model_type not in {"phi3", "lamed_phi3"}:
        raise LanguageConfigurationError(
            f"Expected a Phi-3 config, loaded model_type={model_type!r}."
        )

    hf_config.pad_token_id = tokenizer_bundle.metadata.pad_token_id
    hf_config.eos_token_id = tokenizer_bundle.metadata.eos_token_id
    hf_config.use_cache = not config.optimization.checkpoint_language_model

    causal_lm = AutoModelForCausalLM.from_pretrained(
        model_name,
        config=hf_config,
        cache_dir=cache,
        trust_remote_code=config.model.trust_remote_code,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        attn_implementation=SUPPORTED_ATTENTION_IMPLEMENTATION,
    )
    if not isinstance(causal_lm, nn.Module):
        raise LanguageModelError("AutoModelForCausalLM did not return nn.Module.")

    resize_report = resize_and_initialise_m3d_tokens(
        causal_lm,
        tokenizer_bundle.metadata,
    )
    causal_lm, matched_lora_names = apply_lora_or_load_adapter(
        causal_lm,
        config.model.lora,
        added_token_count=resize_report.added_token_count,
    )
    configure_language_gradient_checkpointing(
        causal_lm,
        enabled=config.optimization.checkpoint_language_model,
    )

    wrapper = M3DLanguageModel(
        causal_lm,
        tokenizer_metadata=tokenizer_bundle.metadata,
    )
    total_parameters, trainable_parameters = _parameter_counts(wrapper)
    report = LanguageModelBuildReport(
        model_name_or_path=model_name,
        model_type=model_type,
        hidden_size=wrapper.hidden_size,
        vocabulary_size=wrapper.vocabulary_size,
        attention_implementation=SUPPORTED_ATTENTION_IMPLEMENTATION,
        gradient_checkpointing=config.optimization.checkpoint_language_model,
        use_cache=bool(getattr(wrapper.config, "use_cache", False)),
        lora_enabled=config.model.lora.enabled,
        lora_rank=(config.model.lora.rank if config.model.lora.enabled else None),
        lora_alpha=(config.model.lora.alpha if config.model.lora.enabled else None),
        lora_target_modules=(
            tuple(config.model.lora.target_modules)
            if config.model.lora.enabled
            else ()
        ),
        matched_lora_module_count=len(matched_lora_names),
        total_parameter_count=total_parameters,
        trainable_parameter_count=trainable_parameters,
        token_resize=resize_report,
    )
    return wrapper, report


# ---------------------------------------------------------------------------
# CPU self-test without Transformers/PEFT
# ---------------------------------------------------------------------------


class _ToyDecoder(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(
        self,
        *,
        inputs_embeds: Tensor,
        attention_mask: Tensor | None = None,
        **_: Any,
    ) -> Any:
        hidden = self.norm(inputs_embeds + torch.tanh(self.proj(inputs_embeds)))
        return SimpleNamespace(last_hidden_state=hidden, past_key_values=None)


class _ToyCausalLM(nn.Module):
    def __init__(self, vocabulary_size: int, hidden_size: int) -> None:
        super().__init__()
        self.model = _ToyDecoder(hidden_size)
        self.embed_tokens = nn.Embedding(vocabulary_size, hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocabulary_size, bias=False)
        self.config = SimpleNamespace(
            hidden_size=hidden_size,
            vocab_size=vocabulary_size,
            model_type="phi3",
            use_cache=False,
            base_model_prefix="model",
        )

    def get_decoder(self) -> nn.Module:
        return self.model

    def get_input_embeddings(self) -> nn.Module:
        return self.embed_tokens

    def get_output_embeddings(self) -> nn.Module:
        return self.lm_head

    def resize_token_embeddings(
        self,
        new_num_tokens: int,
        **_: Any,
    ) -> nn.Module:
        old_input = self.embed_tokens
        old_output = self.lm_head
        hidden_size = int(old_input.embedding_dim)
        copy_rows = min(int(old_input.num_embeddings), new_num_tokens)

        self.embed_tokens = nn.Embedding(new_num_tokens, hidden_size)
        self.lm_head = nn.Linear(hidden_size, new_num_tokens, bias=False)
        with torch.no_grad():
            self.embed_tokens.weight[:copy_rows].copy_(old_input.weight[:copy_rows])
            self.lm_head.weight[:copy_rows].copy_(old_output.weight[:copy_rows])
        self.config.vocab_size = new_num_tokens
        return self.embed_tokens



def _toy_metadata() -> _TokenizerMetadataProtocol:
    return SimpleNamespace(
        tokenizer_name_or_path="toy",
        original_vocab_size=12,
        vocabulary_size=12,
        added_token_count=4,
        image_token="<im_patch>",
        image_token_id=3,
        segmentation_token="[SEG]",
        segmentation_token_id=4,
        box_start_token="<bx_start>",
        box_start_token_id=5,
        box_end_token="<bx_end>",
        box_end_token_id=6,
        pad_token_id=0,
        eos_token_id=2,
        visual_token_count=2,
    )


def _self_test() -> dict[str, Any]:
    torch.manual_seed(41)

    # Phi-3 has padded model embeddings (32,064 rows) beyond its tokenizer
    # vocabulary (32,011 rows). Verify added M3D rows are initialised from the
    # real tokenizer rows even when the model starts with extra padded rows.
    padded_model = _ToyCausalLM(vocabulary_size=16, hidden_size=8)
    padded_input_mean = padded_model.embed_tokens.weight[:12].float().mean(dim=0)
    padded_output_mean = padded_model.lm_head.weight[:12].float().mean(dim=0)
    padded_metadata = SimpleNamespace(
        original_vocab_size=12,
        vocabulary_size=14,
        added_token_count=2,
        pad_token_id=0,
        eos_token_id=2,
    )
    padded_resize = resize_and_initialise_m3d_tokens(
        padded_model,
        padded_metadata,
    )
    assert padded_resize.input_rows_initialised_from_mean == (12, 13)
    assert padded_resize.output_rows_initialised_from_mean == (12, 13)
    assert torch.allclose(
        padded_model.embed_tokens.weight[12:14].float(),
        padded_input_mean.expand(2, -1),
    )
    assert torch.allclose(
        padded_model.lm_head.weight[12:14].float(),
        padded_output_mean.expand(2, -1),
    )

    model = M3DLanguageModel(
        _ToyCausalLM(vocabulary_size=12, hidden_size=8),
        tokenizer_metadata=_toy_metadata(),
    )
    model.train()

    input_ids = torch.tensor(
        [
            [1, 3, 3, 7, 8, 4, 2, 0],
            [1, 3, 3, 9, 7, 4, 2, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = input_ids.ne(0)
    labels = torch.full_like(input_ids, IGNORE_INDEX)
    labels[:, 4:7] = input_ids[:, 4:7]
    visual = torch.randn(2, 2, 8, requires_grad=True)

    embedded = model.prepare_multimodal_embeddings(
        input_ids=input_ids,
        attention_mask=attention_mask,
        visual_embeddings=visual,
        labels=labels,
    )
    assert torch.allclose(embedded.inputs_embeds[embedded.image_token_mask], visual.reshape(-1, 8))
    assert embedded.image_token_counts.tolist() == [2, 2]

    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        visual_embeddings=visual,
        labels=labels,
        logits_mode=LogitsMode.NONE,
    )
    assert output.logits is None
    assert output.loss is not None
    assert output.supervised_token_count is not None

    full_logits = model.get_output_embeddings()(output.last_hidden_state)
    reference_mean, reference_sum, reference_count = compute_full_causal_language_loss(
        full_logits=full_logits,
        labels=labels,
    )
    assert torch.allclose(output.loss, reference_mean, atol=1e-6, rtol=1e-6)
    assert torch.allclose(output.loss_sum, reference_sum, atol=1e-6, rtol=1e-6)
    assert torch.equal(output.supervised_token_count, reference_count)

    output.loss.backward()
    assert visual.grad is not None and torch.isfinite(visual.grad).all()
    assert model.get_output_embeddings().weight.grad is not None

    supervised_output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        visual_embeddings=visual.detach(),
        labels=labels,
        logits_mode=LogitsMode.SUPERVISED,
    )
    assert supervised_output.logits is not None
    assert supervised_output.logits.shape[0] == int(reference_count)
    assert supervised_output.logits.shape[1] == 12

    malformed_detected = False
    try:
        bad_ids = input_ids.clone()
        bad_ids[1, 2] = 7
        model.prepare_multimodal_embeddings(
            input_ids=bad_ids,
            attention_mask=bad_ids.ne(0),
            visual_embeddings=visual.detach(),
            labels=labels,
        )
    except LanguageInputError:
        malformed_detected = True
    assert malformed_detected

    supervised_positions = int(reference_count)
    full_positions = int(input_ids.numel())
    return {
        "status": "passed",
        "multimodal_embedding_shape": list(embedded.inputs_embeds.shape),
        "last_hidden_state_shape": list(output.last_hidden_state.shape),
        "image_token_counts": embedded.image_token_counts.tolist(),
        "supervised_token_count": supervised_positions,
        "full_sequence_position_count": full_positions,
        "lm_head_position_reduction_ratio": 1.0 - supervised_positions / full_positions,
        "selective_loss_matches_full_loss": True,
        "visual_gradient_is_finite": True,
        "malformed_image_prefix_detected": malformed_detected,
        "padded_vocabulary_initialisation_is_correct": True,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run dependency-free CPU contract tests.",
    )
    arguments = parser.parse_args()
    if not arguments.self_test:
        parser.error("Only --self-test is supported when executing this module.")
    print(json.dumps(_self_test(), indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
