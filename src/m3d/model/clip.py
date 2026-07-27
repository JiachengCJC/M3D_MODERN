"""Modern, checkpoint-compatible M3D-CLIP model.

This module reproduces the model architecture from the original M3D-CLIP
implementation while using the modern 3-D ViT defined in :mod:`m3d.model.vit3d`.
The image and text encoders remain independent towers:

* ``vision_encoder`` is a CLS-token 3-D ViT with PyTorch SDPA/Flash-SDPA;
* ``language_encoder`` is a pretrained BERT-family encoder loaded with the
  Hugging Face SDPA implementation when available;
* ``mm_vision_proj`` and ``mm_language_proj`` map both CLS representations to a
  shared contrastive space;
* ``logit_scale`` stores the trainable similarity-temperature parameter.

The important original checkpoint names are preserved exactly:

* ``vision_encoder.*``
* ``language_encoder.*``
* ``mm_vision_proj.weight`` / ``mm_vision_proj.bias``
* ``mm_language_proj.weight`` / ``mm_language_proj.bias``
* ``logit_scale``

Distributed feature gathering and the symmetric contrastive objective are kept
outside this file in ``m3d.model.clip_loss``.  Separating representation
learning from collectives avoids hiding NCCL communication inside model
``forward`` and makes gradient-accumulation and global-loss normalisation
explicit in the trainer.

Published-checkpoint compatibility
----------------------------------
The original M3D code multiplied similarities directly by ``logit_scale`` even
though it initialised the parameter with ``log(1 / 0.07)``.  This module
supports both behaviours:

``legacy_linear``
    Reproduces the published implementation exactly.

``exponential``
    Uses the standard CLIP convention ``exp(logit_scale)`` with a configurable
    upper bound.  This is recommended for a new training run, but should not be
    silently applied to a checkpoint trained with the legacy rule.

Transformers is imported lazily.  CPU contract tests can therefore import and
exercise the model with a tiny injected text encoder even when the pinned
Transformers dependency is unavailable.  Constructing a real BERT encoder
still fails immediately with a clear dependency error.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Final, Mapping, MutableMapping, Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from m3d.config import VisionEncoderConfig
from m3d.model.vit3d import (
    VisionEncoderOutput,
    VisionEncoderRole,
    ViT3DEncoder,
)


# ---------------------------------------------------------------------------
# Optional Hugging Face imports
# ---------------------------------------------------------------------------

try:  # pragma: no cover - exercised on the pinned ASPIRE environment
    from transformers import AutoConfig, AutoModel, PretrainedConfig, PreTrainedModel

    _TRANSFORMERS_AVAILABLE = True
except Exception:  # pragma: no cover - used by dependency-free local tests
    AutoConfig = None  # type: ignore[assignment]
    AutoModel = None  # type: ignore[assignment]
    _TRANSFORMERS_AVAILABLE = False

    class PretrainedConfig:  # type: ignore[no-redef]
        """Small fallback implementing the config methods used in self-tests."""

        model_type = "fallback"

        def __init__(self, **kwargs: Any) -> None:
            for key, value in kwargs.items():
                setattr(self, key, value)

        def to_dict(self) -> dict[str, Any]:
            result = dict(vars(self))
            result["model_type"] = self.model_type
            return result

        @classmethod
        def from_dict(cls, values: Mapping[str, Any]) -> "PretrainedConfig":
            return cls(**dict(values))

        def save_pretrained(self, directory: str | Path) -> None:
            path = Path(directory)
            path.mkdir(parents=True, exist_ok=True)
            (path / "config.json").write_text(
                json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    class PreTrainedModel(nn.Module):  # type: ignore[no-redef]
        config_class: type[PretrainedConfig] | None = None
        base_model_prefix = ""

        def __init__(self, config: PretrainedConfig) -> None:
            super().__init__()
            self.config = config


DEFAULT_LOGIT_SCALE_INIT: Final[float] = math.log(1.0 / 0.07)
DEFAULT_LOGIT_SCALE_MAX: Final[float] = 100.0
DEFAULT_NORMALIZE_EPS: Final[float] = 1.0e-6
SUPPORTED_TEXT_ATTENTION_IMPLEMENTATION: Final[str] = "sdpa"


class M3DCLIPError(RuntimeError):
    """Base exception for M3D-CLIP construction and execution."""


class M3DCLIPDependencyError(M3DCLIPError, ImportError):
    """Raised when a real text encoder is requested without Transformers."""


class M3DCLIPConfigurationError(M3DCLIPError, ValueError):
    """Raised when an architecture or compatibility option is inconsistent."""


class M3DCLIPInputError(M3DCLIPError, ValueError):
    """Raised when image or text tensors violate the model contract."""


class LogitScaleMode(str, Enum):
    """How the trainable temperature parameter scales cosine similarities."""

    LEGACY_LINEAR = "legacy_linear"
    EXPONENTIAL = "exponential"

    @classmethod
    def parse(cls, value: str | "LogitScaleMode") -> "LogitScaleMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as error:
            allowed = ", ".join(item.value for item in cls)
            raise M3DCLIPConfigurationError(
                f"Unknown logit-scale mode {value!r}; expected one of: {allowed}."
            ) from error


class TextPoolingMode(str, Enum):
    """Text-sequence pooling used before the contrastive projection."""

    CLS = "cls"
    MASKED_MEAN = "masked_mean"

    @classmethod
    def parse(cls, value: str | "TextPoolingMode") -> "TextPoolingMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as error:
            allowed = ", ".join(item.value for item in cls)
            raise M3DCLIPConfigurationError(
                f"Unknown text pooling mode {value!r}; expected one of: {allowed}."
            ) from error


class M3DCLIPConfig(PretrainedConfig):
    """Hugging Face compatible configuration for M3D-CLIP.

    The first group of arguments keeps the names and defaults from the original
    repository.  Additional fields make modern execution choices explicit
    without changing the published state-dict layout.
    """

    model_type = "m3d_clip"

    def __init__(
        self,
        language_model_name_or_path: str = "bert-base-uncased",
        local_loss: bool = False,
        gather_loss: bool = True,
        in_channels: int = 1,
        img_size: Sequence[int] = (32, 256, 256),
        patch_size: Sequence[int] = (4, 16, 16),
        hidden_size: int = 768,
        mlp_dim: int = 3072,
        num_layers: int = 12,
        num_heads: int = 12,
        pos_embed: str = "perceptron",
        dropout_rate: float = 0.0,
        spatial_dims: int = 3,
        max_text_len: int = 128,
        vocab_size: int = 30522,
        projection_dim: int | None = None,
        text_hidden_size: int | None = None,
        qkv_bias: bool = False,
        vision_attention_backend: str = "sdpa",
        require_flash_sdpa: bool = True,
        vision_activation_checkpoint_every_n_layers: int = 0,
        text_gradient_checkpointing: bool = False,
        text_attention_implementation: str = SUPPORTED_TEXT_ATTENTION_IMPLEMENTATION,
        text_pooling: str = TextPoolingMode.CLS.value,
        logit_scale_mode: str = LogitScaleMode.LEGACY_LINEAR.value,
        logit_scale_init: float = DEFAULT_LOGIT_SCALE_INIT,
        logit_scale_max: float = DEFAULT_LOGIT_SCALE_MAX,
        normalize_eps: float = DEFAULT_NORMALIZE_EPS,
        freeze_vision_encoder: bool = False,
        freeze_language_encoder: bool = False,
        freeze_projection_layers: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)

        self.language_model_name_or_path = str(language_model_name_or_path)
        self.local_loss = bool(local_loss)
        self.gather_loss = bool(gather_loss)
        self.in_channels = int(in_channels)
        self.img_size = tuple(int(value) for value in img_size)
        self.patch_size = tuple(int(value) for value in patch_size)
        self.hidden_size = int(hidden_size)
        self.mlp_dim = int(mlp_dim)
        self.num_layers = int(num_layers)
        self.num_heads = int(num_heads)
        self.pos_embed = str(pos_embed)
        self.dropout_rate = float(dropout_rate)
        self.spatial_dims = int(spatial_dims)
        self.max_text_len = int(max_text_len)
        self.vocab_size = int(vocab_size)
        self.projection_dim = int(projection_dim or hidden_size)
        self.text_hidden_size = (
            None if text_hidden_size is None else int(text_hidden_size)
        )
        self.qkv_bias = bool(qkv_bias)
        self.vision_attention_backend = str(vision_attention_backend)
        self.require_flash_sdpa = bool(require_flash_sdpa)
        self.vision_activation_checkpoint_every_n_layers = int(
            vision_activation_checkpoint_every_n_layers
        )
        self.text_gradient_checkpointing = bool(text_gradient_checkpointing)
        self.text_attention_implementation = str(text_attention_implementation)
        self.text_pooling = TextPoolingMode.parse(text_pooling).value
        self.logit_scale_mode = LogitScaleMode.parse(logit_scale_mode).value
        self.logit_scale_init = float(logit_scale_init)
        self.logit_scale_max = float(logit_scale_max)
        self.normalize_eps = float(normalize_eps)
        self.freeze_vision_encoder = bool(freeze_vision_encoder)
        self.freeze_language_encoder = bool(freeze_language_encoder)
        self.freeze_projection_layers = bool(freeze_projection_layers)

        self.validate()

    def validate(self) -> None:
        errors: list[str] = []

        if not self.language_model_name_or_path.strip():
            errors.append("language_model_name_or_path cannot be empty")
        if self.spatial_dims != 3:
            errors.append("M3D-CLIP requires spatial_dims=3")
        if self.pos_embed != "perceptron":
            errors.append(
                "Published M3D-CLIP compatibility requires pos_embed='perceptron'"
            )
        if len(self.img_size) != 3 or len(self.patch_size) != 3:
            errors.append("img_size and patch_size must each contain three values")
        elif any(value <= 0 for value in (*self.img_size, *self.patch_size)):
            errors.append("image and patch dimensions must be positive")
        elif any(
            image_dim % patch_dim != 0
            for image_dim, patch_dim in zip(self.img_size, self.patch_size)
        ):
            errors.append("every img_size dimension must be divisible by patch_size")
        if self.in_channels <= 0:
            errors.append("in_channels must be positive")
        if self.hidden_size <= 0 or self.projection_dim <= 0:
            errors.append("hidden_size and projection_dim must be positive")
        if self.num_layers <= 0 or self.num_heads <= 0 or self.mlp_dim <= 0:
            errors.append("num_layers, num_heads and mlp_dim must be positive")
        if self.hidden_size % self.num_heads != 0:
            errors.append("hidden_size must be divisible by num_heads")
        if not 0.0 <= self.dropout_rate < 1.0:
            errors.append("dropout_rate must be in [0, 1)")
        if self.max_text_len <= 0:
            errors.append("max_text_len must be positive")
        if self.vocab_size <= 0:
            errors.append("vocab_size must be positive")
        if self.vision_activation_checkpoint_every_n_layers < 0:
            errors.append(
                "vision_activation_checkpoint_every_n_layers cannot be negative"
            )
        if self.require_flash_sdpa and self.vision_attention_backend != "sdpa":
            errors.append(
                "require_flash_sdpa=true requires vision_attention_backend='sdpa'"
            )
        if self.text_attention_implementation != "sdpa":
            errors.append(
                "The modern M3D-CLIP profile requires text_attention_implementation='sdpa'"
            )
        if not math.isfinite(self.logit_scale_init):
            errors.append("logit_scale_init must be finite")
        if not math.isfinite(self.logit_scale_max) or self.logit_scale_max <= 0:
            errors.append("logit_scale_max must be finite and positive")
        if not math.isfinite(self.normalize_eps) or self.normalize_eps <= 0:
            errors.append("normalize_eps must be finite and positive")

        if errors:
            raise M3DCLIPConfigurationError(
                "Invalid M3D-CLIP configuration:\n- " + "\n- ".join(errors)
            )

    def vision_config(self) -> VisionEncoderConfig:
        """Build the compatible CLS-token 3-D ViT configuration."""

        return VisionEncoderConfig(
            enabled=True,
            architecture="vit3d",
            checkpoint_path=None,
            image_channels=self.in_channels,
            image_size=tuple(self.img_size),
            patch_size=tuple(self.patch_size),
            hidden_size=self.hidden_size,
            depth=self.num_layers,
            num_heads=self.num_heads,
            mlp_dim=self.mlp_dim,
            dropout=self.dropout_rate,
            qkv_bias=self.qkv_bias,
            use_cls_token=True,
            attention_backend=self.vision_attention_backend,  # type: ignore[arg-type]
            require_flash_sdpa=self.require_flash_sdpa,
            activation_checkpoint_every_n_layers=(
                self.vision_activation_checkpoint_every_n_layers
            ),
            freeze=self.freeze_vision_encoder,
            unfreeze_last_n_layers=0,
        )

    def architecture_fingerprint(self) -> str:
        """Stable fingerprint for checkpoint and trainer compatibility checks."""

        keys = (
            "language_model_name_or_path",
            "in_channels",
            "img_size",
            "patch_size",
            "hidden_size",
            "mlp_dim",
            "num_layers",
            "num_heads",
            "projection_dim",
            "text_hidden_size",
            "qkv_bias",
            "text_pooling",
            "logit_scale_mode",
        )
        payload = {key: getattr(self, key) for key in keys}
        serialised = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialised.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class M3DCLIPOutput:
    """Local representations produced by M3D-CLIP.

    ``logits_per_image`` and ``logits_per_text`` are optional local in-batch
    matrices intended for diagnostics.  Distributed training should use the
    dedicated contrastive-loss module so global labels and gradient-preserving
    gathers are handled correctly.
    """

    image_features: Tensor
    text_features: Tensor
    logit_scale: Tensor
    logits_per_image: Tensor | None = None
    logits_per_text: Tensor | None = None

    @property
    def batch_size(self) -> int:
        return int(self.image_features.shape[0])

    @property
    def embedding_dim(self) -> int:
        return int(self.image_features.shape[1])

    def to_dict(self) -> dict[str, Tensor | None]:
        return {
            "image_features": self.image_features,
            "text_features": self.text_features,
            "logit_scale": self.logit_scale,
            "logits_per_image": self.logits_per_image,
            "logits_per_text": self.logits_per_text,
        }


@dataclass(frozen=True, slots=True)
class M3DCLIPParameterSummary:
    total_parameters: int
    trainable_parameters: int
    vision_parameters: int
    text_parameters: int
    projection_parameters: int
    logit_scale_parameters: int

    def to_dict(self) -> dict[str, int]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class M3DCLIPBuildReport:
    architecture_fingerprint: str
    language_model_name_or_path: str
    vision_token_count: int
    vision_hidden_size: int
    text_hidden_size: int
    projection_dim: int
    logit_scale_mode: str
    text_attention_implementation: str
    text_gradient_checkpointing: bool
    parameters: M3DCLIPParameterSummary

    def to_dict(self) -> dict[str, Any]:
        result = dataclasses.asdict(self)
        result["parameters"] = self.parameters.to_dict()
        return result


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _module_device(module: nn.Module) -> torch.device:
    try:
        return next(module.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _module_dtype(module: nn.Module) -> torch.dtype:
    for parameter in module.parameters():
        if parameter.is_floating_point():
            return parameter.dtype
    return torch.float32


def _text_hidden_size(language_encoder: nn.Module) -> int:
    config = getattr(language_encoder, "config", None)
    hidden_size = getattr(config, "hidden_size", None)
    if hidden_size is None:
        raise M3DCLIPConfigurationError(
            "The text encoder must expose config.hidden_size."
        )
    hidden_size = int(hidden_size)
    if hidden_size <= 0:
        raise M3DCLIPConfigurationError(
            f"Text encoder reported invalid hidden size {hidden_size}."
        )
    return hidden_size


def _extract_last_hidden_state(outputs: Any) -> Tensor:
    if isinstance(outputs, Mapping):
        hidden = outputs.get("last_hidden_state")
    else:
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is None and isinstance(outputs, (tuple, list)) and outputs:
            hidden = outputs[0]
    if not isinstance(hidden, Tensor):
        raise M3DCLIPError(
            "Text encoder did not return a tensor named last_hidden_state."
        )
    return hidden


def _validate_text_inputs(input_ids: Tensor, attention_mask: Tensor) -> None:
    if input_ids.ndim != 2:
        raise M3DCLIPInputError(
            f"input_ids must have shape [B, S], got {tuple(input_ids.shape)}."
        )
    if attention_mask.shape != input_ids.shape:
        raise M3DCLIPInputError(
            "attention_mask must have the same [B, S] shape as input_ids: "
            f"input_ids={tuple(input_ids.shape)}, "
            f"attention_mask={tuple(attention_mask.shape)}."
        )
    if input_ids.dtype not in (torch.int32, torch.int64):
        raise M3DCLIPInputError(
            f"input_ids must be integer, got dtype={input_ids.dtype}."
        )
    if attention_mask.dtype not in (
        torch.bool,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    ):
        raise M3DCLIPInputError(
            f"attention_mask must be boolean/integer, got {attention_mask.dtype}."
        )
    valid_counts = attention_mask.to(torch.bool).sum(dim=1)
    if bool(torch.any(valid_counts <= 0)):
        raise M3DCLIPInputError(
            "Every text sample must contain at least one valid token."
        )


def _pool_text_hidden_state(
    hidden_states: Tensor,
    attention_mask: Tensor,
    mode: TextPoolingMode,
) -> Tensor:
    if hidden_states.ndim != 3:
        raise M3DCLIPInputError(
            "Text hidden states must have shape [B, S, C], got "
            f"{tuple(hidden_states.shape)}."
        )
    if hidden_states.shape[:2] != attention_mask.shape:
        raise M3DCLIPInputError(
            "Text hidden-state sequence geometry disagrees with attention_mask: "
            f"hidden={tuple(hidden_states.shape)}, mask={tuple(attention_mask.shape)}."
        )

    if mode is TextPoolingMode.CLS:
        # This reproduces the original M3D-CLIP implementation, which uses the
        # first BERT token after projection and normalisation.
        return hidden_states[:, 0, :]

    weights = attention_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
    numerator = torch.sum(hidden_states * weights, dim=1)
    denominator = torch.sum(weights, dim=1).clamp_min(1.0)
    return numerator / denominator


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class M3DCLIP(PreTrainedModel):
    """M3D image-text dual encoder with independent 3-D ViT and BERT towers."""

    config_class = M3DCLIPConfig
    base_model_prefix = "m3d_clip"
    main_input_name = "images"
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: M3DCLIPConfig,
        *,
        language_encoder: nn.Module | None = None,
        cache_dir: str | Path | None = None,
        local_files_only: bool = False,
        torch_dtype: torch.dtype | None = None,
    ) -> None:
        config.validate()
        super().__init__(config)

        self.vision_encoder = ViT3DEncoder(
            config.vision_config(),
            role=VisionEncoderRole.MAIN,
        )

        if language_encoder is None:
            language_encoder = self._load_language_encoder(
                config,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                torch_dtype=torch_dtype,
            )
        self.language_encoder = language_encoder

        inferred_text_hidden_size = _text_hidden_size(self.language_encoder)
        if (
            config.text_hidden_size is not None
            and int(config.text_hidden_size) != inferred_text_hidden_size
        ):
            raise M3DCLIPConfigurationError(
                "Configured text_hidden_size does not match the loaded encoder: "
                f"configured={config.text_hidden_size}, "
                f"loaded={inferred_text_hidden_size}."
            )
        config.text_hidden_size = inferred_text_hidden_size

        self.mm_vision_proj = nn.Linear(
            config.hidden_size,
            config.projection_dim,
            bias=True,
        )
        self.mm_language_proj = nn.Linear(
            inferred_text_hidden_size,
            config.projection_dim,
            bias=True,
        )
        self.logit_scale = nn.Parameter(
            torch.tensor(float(config.logit_scale_init), dtype=torch.float32)
        )

        self.text_pooling = TextPoolingMode.parse(config.text_pooling)
        self.logit_scale_mode = LogitScaleMode.parse(config.logit_scale_mode)
        self.normalize_eps = float(config.normalize_eps)
        self.logit_scale_max = float(config.logit_scale_max)

        self._configure_text_gradient_checkpointing(
            enabled=config.text_gradient_checkpointing
        )
        self._apply_freeze_policy(config)

    @staticmethod
    def _load_language_encoder(
        config: M3DCLIPConfig,
        *,
        cache_dir: str | Path | None,
        local_files_only: bool,
        torch_dtype: torch.dtype | None,
    ) -> nn.Module:
        if not _TRANSFORMERS_AVAILABLE or AutoModel is None:
            raise M3DCLIPDependencyError(
                "Transformers is required to construct the real M3D-CLIP text "
                "encoder. Install the pinned requirements or inject a compatible "
                "language_encoder for a unit test."
            )

        kwargs: dict[str, Any] = {
            "cache_dir": None if cache_dir is None else str(cache_dir),
            "local_files_only": bool(local_files_only),
            "attn_implementation": config.text_attention_implementation,
        }
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype

        try:
            return AutoModel.from_pretrained(
                config.language_model_name_or_path,
                **kwargs,
            )
        except TypeError as error:
            # A custom BERT-compatible model may not expose the generic
            # attn_implementation keyword.  The pinned BERT path does, so this
            # fallback is explicit and verified after loading.
            kwargs.pop("attn_implementation", None)
            model = AutoModel.from_pretrained(
                config.language_model_name_or_path,
                **kwargs,
            )
            loaded_implementation = getattr(
                getattr(model, "config", None),
                "_attn_implementation",
                None,
            )
            if loaded_implementation not in (None, "sdpa"):
                raise M3DCLIPConfigurationError(
                    "The loaded text encoder did not select SDPA; reported "
                    f"{loaded_implementation!r}."
                ) from error
            return model

    def _configure_text_gradient_checkpointing(self, *, enabled: bool) -> None:
        if not enabled:
            return
        function = getattr(self.language_encoder, "gradient_checkpointing_enable", None)
        if not callable(function):
            raise M3DCLIPConfigurationError(
                "text_gradient_checkpointing=true, but the loaded text encoder "
                "does not expose gradient_checkpointing_enable()."
            )
        try:
            function(gradient_checkpointing_kwargs={"use_reentrant": False})
        except TypeError:
            function()

    def _apply_freeze_policy(self, config: M3DCLIPConfig) -> None:
        if config.freeze_vision_encoder:
            self.vision_encoder.requires_grad_(False)
        if config.freeze_language_encoder:
            self.language_encoder.requires_grad_(False)
        if config.freeze_projection_layers:
            self.mm_vision_proj.requires_grad_(False)
            self.mm_language_proj.requires_grad_(False)

    @property
    def projection_dim(self) -> int:
        return int(self.mm_vision_proj.out_features)

    @property
    def text_hidden_size(self) -> int:
        return int(self.mm_language_proj.in_features)

    @property
    def device(self) -> torch.device:
        return _module_device(self)

    @property
    def dtype(self) -> torch.dtype:
        return _module_dtype(self)

    def effective_logit_scale(self) -> Tensor:
        """Return the scalar applied to cosine similarities."""

        if self.logit_scale_mode is LogitScaleMode.LEGACY_LINEAR:
            return torch.clamp(
                self.logit_scale,
                min=-self.logit_scale_max,
                max=self.logit_scale_max,
            )

        maximum_log = math.log(self.logit_scale_max)
        return torch.exp(torch.clamp(self.logit_scale, max=maximum_log))

    @torch.no_grad()
    def clamp_logit_scale_(self) -> None:
        """Keep the temperature parameter in the numerically valid range."""

        if self.logit_scale_mode is LogitScaleMode.LEGACY_LINEAR:
            self.logit_scale.clamp_(
                min=-self.logit_scale_max,
                max=self.logit_scale_max,
            )
        else:
            self.logit_scale.clamp_(max=math.log(self.logit_scale_max))

    def encode_image(
        self,
        images: Tensor,
        *,
        return_encoder_output: bool = False,
    ) -> Tensor | tuple[Tensor, VisionEncoderOutput]:
        """Encode a 3-D image batch into L2-normalised CLS features."""

        output = self.vision_encoder(
            images,
            output_hidden_states=False,
            return_dict=True,
        )
        if not isinstance(output, VisionEncoderOutput):
            raise M3DCLIPError("Vision encoder returned an unexpected output type.")
        cls_embedding = output.cls_embedding
        if cls_embedding is None:
            raise M3DCLIPConfigurationError(
                "M3D-CLIP vision encoder must contain a CLS token."
            )
        projected = self.mm_vision_proj(cls_embedding)
        features = F.normalize(projected, dim=-1, eps=self.normalize_eps)
        if return_encoder_output:
            return features, output
        return features

    def encode_text(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        *,
        token_type_ids: Tensor | None = None,
        return_hidden_states: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Encode text into L2-normalised projected features."""

        _validate_text_inputs(input_ids, attention_mask)
        if token_type_ids is not None and token_type_ids.shape != input_ids.shape:
            raise M3DCLIPInputError(
                "token_type_ids must match input_ids shape: "
                f"token_type_ids={tuple(token_type_ids.shape)}, "
                f"input_ids={tuple(input_ids.shape)}."
            )

        kwargs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": False,
            "return_dict": True,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids

        outputs = self.language_encoder(**kwargs)
        hidden_states = _extract_last_hidden_state(outputs)
        pooled = _pool_text_hidden_state(
            hidden_states,
            attention_mask,
            self.text_pooling,
        )
        projected = self.mm_language_proj(pooled)
        features = F.normalize(projected, dim=-1, eps=self.normalize_eps)
        if return_hidden_states:
            return features, hidden_states
        return features

    def forward(
        self,
        images: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor | None = None,
        labels: Tensor | None = None,
        *,
        compute_local_similarity: bool = False,
        return_dict: bool = True,
        **_: Any,
    ) -> M3DCLIPOutput | tuple[Tensor, Tensor, Tensor]:
        """Return local image/text features and the effective temperature.

        ``labels`` is accepted for source-level compatibility with the original
        Hugging Face Trainer collator, but loss calculation intentionally lives
        in :mod:`m3d.model.clip_loss`.  Supplying labels here therefore has no
        effect and cannot accidentally create rank-local contrastive targets.
        """

        del labels
        image_features = self.encode_image(images)
        text_features = self.encode_text(
            input_ids,
            attention_mask,
            token_type_ids=token_type_ids,
        )
        if not isinstance(image_features, Tensor) or not isinstance(text_features, Tensor):
            raise M3DCLIPError("Encoder output contract was violated.")
        if image_features.shape != text_features.shape:
            raise M3DCLIPInputError(
                "Image and text projected features must have identical shape: "
                f"image={tuple(image_features.shape)}, "
                f"text={tuple(text_features.shape)}."
            )

        scale = self.effective_logit_scale().to(
            device=image_features.device,
            dtype=image_features.dtype,
        )
        logits_per_image: Tensor | None = None
        logits_per_text: Tensor | None = None
        if compute_local_similarity:
            logits_per_image = scale * image_features @ text_features.transpose(0, 1)
            logits_per_text = logits_per_image.transpose(0, 1)

        if not return_dict:
            return image_features, text_features, scale
        return M3DCLIPOutput(
            image_features=image_features,
            text_features=text_features,
            logit_scale=scale,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
        )

    def parameter_summary(self) -> M3DCLIPParameterSummary:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        vision = sum(parameter.numel() for parameter in self.vision_encoder.parameters())
        text = sum(parameter.numel() for parameter in self.language_encoder.parameters())
        projection = sum(
            parameter.numel()
            for module in (self.mm_vision_proj, self.mm_language_proj)
            for parameter in module.parameters()
        )
        return M3DCLIPParameterSummary(
            total_parameters=total,
            trainable_parameters=trainable,
            vision_parameters=vision,
            text_parameters=text,
            projection_parameters=projection,
            logit_scale_parameters=self.logit_scale.numel(),
        )

    def build_report(self) -> M3DCLIPBuildReport:
        return M3DCLIPBuildReport(
            architecture_fingerprint=self.config.architecture_fingerprint(),
            language_model_name_or_path=self.config.language_model_name_or_path,
            vision_token_count=self.vision_encoder.output_token_count,
            vision_hidden_size=self.vision_encoder.hidden_size,
            text_hidden_size=self.text_hidden_size,
            projection_dim=self.projection_dim,
            logit_scale_mode=self.logit_scale_mode.value,
            text_attention_implementation=self.config.text_attention_implementation,
            text_gradient_checkpointing=self.config.text_gradient_checkpointing,
            parameters=self.parameter_summary(),
        )

    def no_weight_decay_parameter_names(self) -> frozenset[str]:
        names = {
            "logit_scale",
            *(f"vision_encoder.{name}" for name in self.vision_encoder.no_weight_decay_parameter_names()),
        }
        return frozenset(names)

    def freeze_vision_encoder(self) -> None:
        self.vision_encoder.requires_grad_(False)

    def unfreeze_vision_encoder(self) -> None:
        self.vision_encoder.requires_grad_(True)

    def freeze_language_encoder(self) -> None:
        self.language_encoder.requires_grad_(False)

    def unfreeze_language_encoder(self) -> None:
        self.language_encoder.requires_grad_(True)

    def freeze_projection_layers(self) -> None:
        self.mm_vision_proj.requires_grad_(False)
        self.mm_language_proj.requires_grad_(False)

    def unfreeze_projection_layers(self) -> None:
        self.mm_vision_proj.requires_grad_(True)
        self.mm_language_proj.requires_grad_(True)


# ---------------------------------------------------------------------------
# Builders and checkpoint helpers
# ---------------------------------------------------------------------------


def build_m3d_clip(
    config: M3DCLIPConfig,
    *,
    language_encoder: nn.Module | None = None,
    cache_dir: str | Path | None = None,
    local_files_only: bool = False,
    torch_dtype: torch.dtype | None = None,
) -> tuple[M3DCLIP, M3DCLIPBuildReport]:
    model = M3DCLIP(
        config,
        language_encoder=language_encoder,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        torch_dtype=torch_dtype,
    )
    return model, model.build_report()


def _unwrap_state_dict(payload: Any) -> MutableMapping[str, Tensor]:
    if isinstance(payload, Mapping):
        for key in ("state_dict", "model", "module"):
            nested = payload.get(key)
            if isinstance(nested, Mapping) and all(
                isinstance(value, Tensor) for value in nested.values()
            ):
                payload = nested
                break
    if not isinstance(payload, Mapping) or not all(
        isinstance(key, str) and isinstance(value, Tensor)
        for key, value in payload.items()
    ):
        raise M3DCLIPConfigurationError(
            "Checkpoint does not contain a string-to-tensor state dict."
        )
    return {str(key): value for key, value in payload.items()}


def _strip_uniform_prefix(
    state: Mapping[str, Tensor],
    prefixes: Sequence[str] = ("module.", "_orig_mod."),
) -> dict[str, Tensor]:
    current = dict(state)
    changed = True
    while changed and current:
        changed = False
        for prefix in prefixes:
            if all(key.startswith(prefix) for key in current):
                current = {key[len(prefix):]: value for key, value in current.items()}
                changed = True
    return current


def load_m3d_clip_checkpoint(
    model: M3DCLIP,
    checkpoint_path: str | Path,
    *,
    strict: bool = True,
) -> dict[str, Any]:
    """Load an original or modern M3D-CLIP state dict on CPU."""

    path = Path(checkpoint_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"M3D-CLIP checkpoint does not exist: {path}")

    if path.suffix == ".safetensors":
        try:
            from safetensors.torch import load_file
        except Exception as error:  # pragma: no cover - dependency environment
            raise M3DCLIPDependencyError(
                "safetensors is required to load this checkpoint."
            ) from error
        payload = load_file(str(path), device="cpu")
    else:
        payload = torch.load(path, map_location="cpu", weights_only=True)

    state = _strip_uniform_prefix(_unwrap_state_dict(payload))
    incompatible = model.load_state_dict(state, strict=strict)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return {
        "path": str(path),
        "sha256": digest,
        "tensor_count": len(state),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
        "strict": bool(strict),
    }


def convert_legacy_logit_scale_to_exponential_(model: M3DCLIP) -> None:
    """Convert a positive legacy linear scale to the exponential convention.

    This is an explicit migration operation.  It is never performed
    automatically because the source checkpoint's training convention must be
    known by the caller.
    """

    if model.logit_scale_mode is not LogitScaleMode.LEGACY_LINEAR:
        raise M3DCLIPConfigurationError(
            "The model is not currently configured for legacy_linear scaling."
        )
    with torch.no_grad():
        value = float(model.logit_scale.detach().cpu())
        if not math.isfinite(value) or value <= 0:
            raise M3DCLIPConfigurationError(
                "A legacy linear scale must be finite and positive before it can "
                f"be converted; got {value}."
            )
        model.logit_scale.copy_(
            torch.tensor(
                math.log(min(value, model.logit_scale_max)),
                device=model.logit_scale.device,
                dtype=model.logit_scale.dtype,
            )
        )
    model.logit_scale_mode = LogitScaleMode.EXPONENTIAL
    model.config.logit_scale_mode = LogitScaleMode.EXPONENTIAL.value


# Register only when the real Transformers registry is available.  Repeated
# imports can encounter an already-registered config in notebook environments;
# registration errors are therefore ignored only when the existing class is
# exactly this implementation.
if _TRANSFORMERS_AVAILABLE and AutoConfig is not None and AutoModel is not None:  # pragma: no cover
    try:
        AutoConfig.register(M3DCLIPConfig.model_type, M3DCLIPConfig)
    except ValueError:
        pass
    try:
        AutoModel.register(M3DCLIPConfig, M3DCLIP)
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Dependency-free self-test
# ---------------------------------------------------------------------------


class _TinyTextEncoder(nn.Module):
    def __init__(self, *, vocab_size: int, hidden_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)
        self.gradient_checkpointing = False

    def gradient_checkpointing_enable(self, **_: Any) -> None:
        self.gradient_checkpointing = True

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        token_type_ids: Tensor | None = None,
    ) -> Any:
        del output_hidden_states, token_type_ids
        hidden = self.embedding(input_ids)
        hidden = self.projection(self.norm(hidden))
        hidden = hidden * attention_mask.to(hidden.dtype).unsqueeze(-1)
        if return_dict:
            return SimpleNamespace(last_hidden_state=hidden)
        return (hidden,)


def _tiny_config(*, scale_mode: str = "legacy_linear") -> M3DCLIPConfig:
    return M3DCLIPConfig(
        language_model_name_or_path="tiny-text",
        in_channels=1,
        img_size=(4, 8, 8),
        patch_size=(2, 4, 4),
        hidden_size=24,
        mlp_dim=48,
        num_layers=2,
        num_heads=4,
        dropout_rate=0.0,
        max_text_len=8,
        vocab_size=31,
        projection_dim=16,
        text_hidden_size=20,
        vision_attention_backend="math",
        require_flash_sdpa=False,
        vision_activation_checkpoint_every_n_layers=1,
        text_gradient_checkpointing=True,
        logit_scale_mode=scale_mode,
    )


def _run_self_test() -> dict[str, Any]:
    torch.manual_seed(17)
    config = _tiny_config()
    text_encoder = _TinyTextEncoder(vocab_size=31, hidden_size=20)
    model, report = build_m3d_clip(config, language_encoder=text_encoder)
    model.train()

    images = torch.randn(3, 1, 4, 8, 8, requires_grad=True)
    input_ids = torch.tensor(
        [
            [1, 2, 3, 0, 0, 0],
            [1, 4, 5, 6, 0, 0],
            [1, 7, 8, 9, 10, 0],
        ],
        dtype=torch.long,
    )
    attention_mask = input_ids.ne(0)

    output = model(
        images,
        input_ids,
        attention_mask,
        compute_local_similarity=True,
    )
    assert isinstance(output, M3DCLIPOutput)
    assert output.image_features.shape == (3, 16)
    assert output.text_features.shape == (3, 16)
    assert output.logits_per_image is not None
    assert output.logits_per_image.shape == (3, 3)
    torch.testing.assert_close(
        output.image_features.norm(dim=-1),
        torch.ones(3),
        atol=1.0e-5,
        rtol=1.0e-5,
    )
    torch.testing.assert_close(
        output.text_features.norm(dim=-1),
        torch.ones(3),
        atol=1.0e-5,
        rtol=1.0e-5,
    )

    diagnostic_loss = (
        output.logits_per_image.diagonal().mean()
        + output.image_features.square().mean()
        + output.text_features.square().mean()
    )
    diagnostic_loss.backward()
    assert images.grad is not None and torch.isfinite(images.grad).all()
    assert model.vision_encoder.cls_token.grad is not None
    assert model.language_encoder.embedding.weight.grad is not None
    assert model.mm_vision_proj.weight.grad is not None
    assert model.mm_language_proj.weight.grad is not None
    assert model.logit_scale.grad is not None

    state_keys = set(model.state_dict())
    required_keys = {
        "vision_encoder.cls_token",
        "vision_encoder.patch_embedding.position_embeddings",
        "vision_encoder.blocks.0.attn.qkv.weight",
        "language_encoder.embedding.weight",
        "mm_vision_proj.weight",
        "mm_language_proj.weight",
        "logit_scale",
    }
    assert required_keys.issubset(state_keys)

    with torch.no_grad():
        model.logit_scale.fill_(1000.0)
    model.clamp_logit_scale_()
    assert float(model.logit_scale.detach()) == model.logit_scale_max

    exponential_model, _ = build_m3d_clip(
        _tiny_config(scale_mode="exponential"),
        language_encoder=_TinyTextEncoder(vocab_size=31, hidden_size=20),
    )
    expected_scale = 1.0 / 0.07
    torch.testing.assert_close(
        exponential_model.effective_logit_scale(),
        torch.tensor(expected_scale),
        atol=1.0e-5,
        rtol=1.0e-5,
    )

    legacy_model, _ = build_m3d_clip(
        _tiny_config(scale_mode="legacy_linear"),
        language_encoder=_TinyTextEncoder(vocab_size=31, hidden_size=20),
    )
    with torch.no_grad():
        legacy_model.logit_scale.fill_(2.5)
    convert_legacy_logit_scale_to_exponential_(legacy_model)
    torch.testing.assert_close(
        legacy_model.effective_logit_scale(),
        torch.tensor(2.5),
    )

    malformed_mask_detected = False
    try:
        model.encode_text(input_ids, attention_mask[:, :-1])
    except M3DCLIPInputError:
        malformed_mask_detected = True
    assert malformed_mask_detected

    with torch.no_grad():
        original = {
            key: value.detach().clone()
            for key, value in model.state_dict().items()
        }
    restored, _ = build_m3d_clip(
        config,
        language_encoder=_TinyTextEncoder(vocab_size=31, hidden_size=20),
    )
    restored.load_state_dict(original, strict=True)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)

    no_decay = model.no_weight_decay_parameter_names()
    assert "logit_scale" in no_decay
    assert "vision_encoder.cls_token" in no_decay
    assert "vision_encoder.patch_embedding.position_embeddings" in no_decay

    return {
        "status": "passed",
        "image_feature_shape": list(output.image_features.shape),
        "text_feature_shape": list(output.text_features.shape),
        "local_similarity_shape": list(output.logits_per_image.shape),
        "image_features_normalised": True,
        "text_features_normalised": True,
        "vision_gradient_is_finite": True,
        "text_gradient_is_finite": True,
        "original_checkpoint_keys_present": True,
        "strict_state_roundtrip": True,
        "malformed_attention_mask_detected": malformed_mask_detected,
        "legacy_and_exponential_modes_supported": True,
        "text_gradient_checkpointing_enabled": bool(
            text_encoder.gradient_checkpointing
        ),
        "vision_output_token_count": report.vision_token_count,
        "architecture_fingerprint": report.architecture_fingerprint,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the dependency-free CPU contract test.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.self_test:
        raise SystemExit("This module is a library. Use --self-test for validation.")
    print(json.dumps(_run_self_test(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
