"""Modern, checkpoint-compatible 3D Vision Transformer for M3D.

This module defines the 3D ViT implementation used by both image encoders in
M3D-Modernized:

* the **Main 3D ViT**, which produces visual tokens for the multimodal
  projector and Phi-3;
* the **SegVol 3D ViT**, which produces dense spatial tokens for the
  segmentation pathway.

The two encoders share this Python implementation only.  They are instantiated
as separate ``nn.Module`` objects with separate parameters, gradients,
optimizer states and checkpoint entries.

Compatibility goals
-------------------
The original M3D repository builds both encoders from MONAI's ViT using
``pos_embed='perceptron'``.  This implementation deliberately preserves the
important state-dict names and tensor shapes:

* ``patch_embedding.position_embeddings``
* ``patch_embedding.patch_embeddings.1.weight``
* ``patch_embedding.patch_embeddings.1.bias``
* ``blocks.<i>.norm1`` / ``blocks.<i>.norm2``
* ``blocks.<i>.attn.qkv`` / ``blocks.<i>.attn.out_proj``
* ``blocks.<i>.mlp.linear1`` / ``blocks.<i>.mlp.linear2``
* ``norm``
* ``cls_token`` for the Main ViT only

The perceptron patch projection is evaluated with ``torch.nn.functional.conv3d``
for an efficient fused implementation, while retaining the original linear
weight layout ``[hidden_size, patch_volume * in_channels]``.  The conversion to
a Conv3d kernel is a view-and-permute operation, so it is mathematically
identical to MONAI's explicit patch rearrangement followed by ``nn.Linear``.

Performance features
--------------------
* PyTorch SDPA / Flash-SDPA attention through :mod:`m3d.model.attention`.
* Optional non-reentrant activation checkpointing at a configurable layer
  interval.
* Hidden states are not retained unless explicitly requested.
* Fixed, validated 3D shapes avoid silent interpolation or axis changes.
* Patch tokens can be converted to ``[B, C, D, H, W]`` without copying the
  underlying semantic ordering.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, TypeAlias, overload

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from m3d.config import ModelConfig, VisionEncoderConfig
from m3d.model.attention import AttentionBackendName, FusedSelfAttention


Shape3D: TypeAlias = tuple[int, int, int]
StateDict: TypeAlias = Mapping[str, Tensor]

_MONAI_PATCH_PROJECTION_INDEX: Final[int] = 1
_DEFAULT_LAYER_NORM_EPS: Final[float] = 1.0e-5


class VisionEncoderConfigurationError(ValueError):
    """Raised when a 3D ViT configuration is internally inconsistent."""


class VisionEncoderExecutionError(RuntimeError):
    """Raised when an input cannot be processed by the configured encoder."""


class VisionEncoderRole(str, Enum):
    """Semantic role of one independent 3D image encoder."""

    MAIN = "main"
    SEGMENTATION = "segmentation"


@dataclass(frozen=True, slots=True)
class VisionEncoderShape:
    """Static spatial/token geometry of one 3D ViT."""

    image_channels: int
    image_size: Shape3D
    patch_size: Shape3D
    patch_grid: Shape3D
    patch_count: int
    hidden_size: int
    has_cls_token: bool

    @property
    def output_token_count(self) -> int:
        return self.patch_count + int(self.has_cls_token)

    @property
    def patch_volume(self) -> int:
        return math.prod(self.patch_size)

    @property
    def patch_vector_size(self) -> int:
        return self.image_channels * self.patch_volume


@dataclass(frozen=True, slots=True)
class VisionEncoderOutput:
    """Structured output from :class:`ViT3DEncoder`.

    ``last_hidden_state`` contains the final LayerNorm output.  For the Main
    ViT it has one leading CLS token; for the SegVol ViT it contains patch
    tokens only.  ``hidden_states`` mirrors the original MONAI ViT behaviour
    when requested: each entry is the output of a transformer block before the
    final LayerNorm.
    """

    last_hidden_state: Tensor
    patch_grid: Shape3D
    has_cls_token: bool
    hidden_states: tuple[Tensor, ...] | None = None

    @property
    def patch_tokens(self) -> Tensor:
        """Return patch tokens without the optional CLS token."""

        if self.has_cls_token:
            return self.last_hidden_state[:, 1:, :]
        return self.last_hidden_state

    @property
    def cls_embedding(self) -> Tensor | None:
        """Return ``[B, C]`` CLS embeddings for the Main ViT, otherwise None."""

        if not self.has_cls_token:
            return None
        return self.last_hidden_state[:, 0, :]

    def spatial_features(self) -> Tensor:
        """Return patch tokens as ``[B, C, D_grid, H_grid, W_grid]``."""

        tokens = self.patch_tokens
        expected_tokens = math.prod(self.patch_grid)
        if tokens.ndim != 3 or tokens.shape[1] != expected_tokens:
            raise VisionEncoderExecutionError(
                "Cannot reshape patch tokens into the configured spatial grid: "
                f"tokens={tuple(tokens.shape)}, patch_grid={self.patch_grid}."
            )
        batch_size, _, channels = tokens.shape
        return (
            tokens.transpose(1, 2)
            .reshape(batch_size, channels, *self.patch_grid)
            .contiguous()
        )

    def selected_hidden_state(self, layer_index: int, *, remove_cls: bool = False) -> Tensor:
        """Select one retained block output with normal Python indexing."""

        if self.hidden_states is None:
            raise VisionEncoderExecutionError(
                "hidden_states were not retained; call the encoder with "
                "output_hidden_states=True."
            )
        try:
            selected = self.hidden_states[layer_index]
        except IndexError as error:
            raise VisionEncoderExecutionError(
                f"Layer index {layer_index} is outside the retained hidden-state "
                f"range of {len(self.hidden_states)} layers."
            ) from error
        if remove_cls and self.has_cls_token:
            selected = selected[:, 1:, :]
        return selected


@dataclass(frozen=True, slots=True)
class TrainabilitySummary:
    """Parameter counts after applying a freeze/unfreeze policy."""

    role: str
    total_parameters: int
    trainable_parameters: int
    frozen_parameters: int
    trainable_fraction: float


@dataclass(frozen=True, slots=True)
class DualVisionEncoderBundle:
    """The two independent M3D image encoders.

    This is intentionally not an ``nn.Module`` container.  The final M3D model
    attaches ``main_tower`` and ``segmentation_encoder`` at their original
    semantic locations so checkpoint prefixes remain clear.
    """

    main_tower: "ViT3DTower"
    segmentation_encoder: "ViT3DEncoder"

    def __post_init__(self) -> None:
        assert_independent_encoders(
            self.main_tower.vision_tower,
            self.segmentation_encoder,
        )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _as_shape3d(value: Sequence[int] | int, *, name: str) -> Shape3D:
    if isinstance(value, int):
        values = (int(value),) * 3
    else:
        values = tuple(int(item) for item in value)
    if len(values) != 3:
        raise VisionEncoderConfigurationError(
            f"{name} must contain exactly three dimensions; received {values}."
        )
    if any(item <= 0 for item in values):
        raise VisionEncoderConfigurationError(
            f"{name} dimensions must be positive; received {values}."
        )
    return values


def _validate_dropout(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise VisionEncoderConfigurationError(
            f"dropout must be finite and in [0, 1); received {value!r}."
        )
    return value


def _register_ddp_gradient_layout_hook(parameter: nn.Parameter) -> None:
    """Keep singleton-broadcast gradients compatible with DDP bucket views."""

    gradient_shape = tuple(parameter.shape)
    gradient_stride = tuple(parameter.stride())

    def _restore_layout(gradient: Tensor) -> Tensor:
        # FSDP2 supplies a DTensor subclass and owns its gradient layout. DDP
        # supplies an ordinary Tensor whose singleton dimensions may inherit a
        # later full-sequence stride through cat/broadcast backward.
        if (
            type(gradient) is not Tensor
            or tuple(gradient.stride()) == gradient_stride
        ):
            return gradient
        aligned = torch.empty_strided(
            gradient_shape,
            gradient_stride,
            dtype=gradient.dtype,
            device=gradient.device,
        )
        return aligned.copy_(gradient)

    parameter.register_hook(_restore_layout)


def _build_shape(config: VisionEncoderConfig) -> VisionEncoderShape:
    image_size = _as_shape3d(config.image_size, name="image_size")
    patch_size = _as_shape3d(config.patch_size, name="patch_size")
    image_channels = int(config.image_channels)
    hidden_size = int(config.hidden_size)

    if image_channels <= 0:
        raise VisionEncoderConfigurationError("image_channels must be positive.")
    if hidden_size <= 0:
        raise VisionEncoderConfigurationError("hidden_size must be positive.")

    for axis, (image_dim, patch_dim) in enumerate(zip(image_size, patch_size)):
        if patch_dim > image_dim:
            raise VisionEncoderConfigurationError(
                f"patch_size[{axis}]={patch_dim} exceeds image_size[{axis}]={image_dim}."
            )
        if image_dim % patch_dim != 0:
            raise VisionEncoderConfigurationError(
                "Perceptron patch embedding requires exact divisibility: "
                f"image_size={image_size}, patch_size={patch_size}."
            )

    patch_grid = tuple(
        image_dim // patch_dim
        for image_dim, patch_dim in zip(image_size, patch_size)
    )
    return VisionEncoderShape(
        image_channels=image_channels,
        image_size=image_size,
        patch_size=patch_size,
        patch_grid=patch_grid,  # type: ignore[arg-type]
        patch_count=math.prod(patch_grid),
        hidden_size=hidden_size,
        has_cls_token=bool(config.use_cls_token),
    )


def _validate_encoder_config(config: VisionEncoderConfig, role: VisionEncoderRole) -> None:
    shape = _build_shape(config)
    depth = int(config.depth)
    num_heads = int(config.num_heads)
    mlp_dim = int(config.mlp_dim)
    checkpoint_interval = int(config.activation_checkpoint_every_n_layers)

    if config.architecture != "vit3d":
        raise VisionEncoderConfigurationError(
            f"Unsupported vision architecture {config.architecture!r}; expected 'vit3d'."
        )
    if depth <= 0:
        raise VisionEncoderConfigurationError("depth must be positive.")
    if num_heads <= 0:
        raise VisionEncoderConfigurationError("num_heads must be positive.")
    if shape.hidden_size % num_heads != 0:
        raise VisionEncoderConfigurationError(
            "hidden_size must be divisible by num_heads: "
            f"hidden_size={shape.hidden_size}, num_heads={num_heads}."
        )
    if mlp_dim <= 0:
        raise VisionEncoderConfigurationError("mlp_dim must be positive.")
    _validate_dropout(config.dropout)
    if checkpoint_interval < 0:
        raise VisionEncoderConfigurationError(
            "activation_checkpoint_every_n_layers cannot be negative."
        )
    if not 0 <= int(config.unfreeze_last_n_layers) <= depth:
        raise VisionEncoderConfigurationError(
            "unfreeze_last_n_layers must be between zero and depth."
        )
    if config.freeze and int(config.unfreeze_last_n_layers) > 0:
        raise VisionEncoderConfigurationError(
            "freeze=True cannot be combined with unfreeze_last_n_layers>0."
        )
    if role is VisionEncoderRole.MAIN and not config.use_cls_token:
        raise VisionEncoderConfigurationError(
            "The Main M3D vision encoder must use a CLS token for checkpoint "
            "compatibility, even though the multimodal projector consumes patch tokens."
        )
    if role is VisionEncoderRole.SEGMENTATION and config.use_cls_token:
        raise VisionEncoderConfigurationError(
            "The SegVol vision encoder must not use a CLS token."
        )


# ---------------------------------------------------------------------------
# Patch embedding
# ---------------------------------------------------------------------------


class _LinearPatchProjection3D(nn.Module):
    """MONAI-compatible linear patch projection evaluated as Conv3d.

    The parameter layout remains ``[out_features, pD * pH * pW * C]`` with the
    patch vector ordered as ``(pD, pH, pW, C)``.  Before ``F.conv3d`` the weight
    is viewed as ``[out, pD, pH, pW, C]`` and permuted to PyTorch's kernel
    layout ``[out, C, pD, pH, pW]``.
    """

    def __init__(
        self,
        *,
        in_channels: int,
        hidden_size: int,
        patch_size: Shape3D,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.hidden_size = int(hidden_size)
        self.patch_size = patch_size
        self.patch_dim = self.in_channels * math.prod(self.patch_size)

        # Shapes deliberately match nn.Linear(self.patch_dim, hidden_size).
        self.weight = nn.Parameter(torch.empty(self.hidden_size, self.patch_dim))
        self.bias = nn.Parameter(torch.empty(self.hidden_size))

    def reset_parameters_monai(self) -> None:
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=0.02,
            a=-2.0,
            b=2.0,
        )
        nn.init.zeros_(self.bias)

    def _conv_kernel(self) -> Tensor:
        patch_d, patch_h, patch_w = self.patch_size
        return (
            self.weight.view(
                self.hidden_size,
                patch_d,
                patch_h,
                patch_w,
                self.in_channels,
            )
            .permute(0, 4, 1, 2, 3)
            .contiguous()
        )

    def forward(self, images: Tensor) -> Tensor:
        if images.ndim != 5:
            raise VisionEncoderExecutionError(
                "Patch projection expects [B, C, D, H, W]; received "
                f"{tuple(images.shape)}."
            )
        if images.shape[1] != self.in_channels:
            raise VisionEncoderExecutionError(
                f"Patch projection expected {self.in_channels} channels, "
                f"received {images.shape[1]}."
            )
        if not images.dtype.is_floating_point:
            raise VisionEncoderExecutionError(
                f"Vision input must use a floating dtype; received {images.dtype}."
            )

        patches = F.conv3d(
            images,
            self._conv_kernel(),
            self.bias,
            stride=self.patch_size,
            padding=0,
            dilation=1,
            groups=1,
        )
        return patches.flatten(2).transpose(1, 2).contiguous()


class PatchEmbedding3D(nn.Module):
    """Learnable 3D patch and position embedding compatible with MONAI ViT."""

    def __init__(
        self,
        *,
        in_channels: int,
        image_size: Shape3D,
        patch_size: Shape3D,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise VisionEncoderConfigurationError(
                "hidden_size must be divisible by num_heads."
            )

        self.in_channels = int(in_channels)
        self.image_size = image_size
        self.patch_size = patch_size
        self.hidden_size = int(hidden_size)
        self.patch_grid = tuple(
            image_dim // patch_dim
            for image_dim, patch_dim in zip(image_size, patch_size)
        )
        self.n_patches = math.prod(self.patch_grid)
        self.patch_dim = self.in_channels * math.prod(self.patch_size)
        self.proj_type = "perceptron"
        self.pos_embed_type = "learnable"

        # Index 0 and 1 mirror MONAI's Sequential(Rearrange, Linear).  The
        # parameter-bearing module remains at index 1, preserving checkpoint
        # keys such as patch_embeddings.1.weight.
        self.patch_embeddings = nn.Sequential(
            nn.Identity(),
            _LinearPatchProjection3D(
                in_channels=self.in_channels,
                hidden_size=self.hidden_size,
                patch_size=self.patch_size,
            ),
        )
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, self.n_patches, self.hidden_size)
        )
        _register_ddp_gradient_layout_hook(self.position_embeddings)
        self.dropout = nn.Dropout(_validate_dropout(dropout_rate))
        self.reset_parameters_monai()

    @property
    def projection(self) -> _LinearPatchProjection3D:
        module = self.patch_embeddings[_MONAI_PATCH_PROJECTION_INDEX]
        if not isinstance(module, _LinearPatchProjection3D):
            raise RuntimeError("Internal patch projection module was replaced unexpectedly.")
        return module

    def reset_parameters_monai(self) -> None:
        nn.init.trunc_normal_(
            self.position_embeddings,
            mean=0.0,
            std=0.02,
            a=-2.0,
            b=2.0,
        )
        self.projection.reset_parameters_monai()

    def _validate_input(self, images: Tensor) -> None:
        expected = (self.in_channels, *self.image_size)
        if images.ndim != 5:
            raise VisionEncoderExecutionError(
                "Vision input must have shape [B, C, D, H, W]; received "
                f"{tuple(images.shape)}."
            )
        if tuple(images.shape[1:]) != expected:
            raise VisionEncoderExecutionError(
                "Vision input shape does not match the fixed M3D encoder shape: "
                f"received {tuple(images.shape[1:])}, expected {expected}. "
                "Resize/resample during offline preprocessing, not silently inside "
                "the encoder, so image/mask alignment remains explicit."
            )
        if images.device.type not in ("cpu", "cuda"):
            raise VisionEncoderExecutionError(
                f"Vision input must be on CPU or CUDA; received {images.device}."
            )
        if not images.dtype.is_floating_point:
            raise VisionEncoderExecutionError(
                f"Vision input must use a floating dtype; received {images.dtype}."
            )

    def forward(self, images: Tensor) -> Tensor:
        self._validate_input(images)
        tokens = self.patch_embeddings(images)
        if tokens.shape[1:] != (self.n_patches, self.hidden_size):
            raise VisionEncoderExecutionError(
                "Patch embedding produced an unexpected token shape: "
                f"received {tuple(tokens.shape)}, expected "
                f"[B, {self.n_patches}, {self.hidden_size}]."
            )
        return self.dropout(tokens + self.position_embeddings)


# ---------------------------------------------------------------------------
# Transformer blocks
# ---------------------------------------------------------------------------


class MLPBlock(nn.Module):
    """MONAI-compatible ViT feed-forward block."""

    def __init__(
        self,
        hidden_size: int,
        mlp_dim: int,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        dropout_rate = _validate_dropout(dropout_rate)
        self.linear1 = nn.Linear(hidden_size, mlp_dim)
        self.linear2 = nn.Linear(mlp_dim, hidden_size)
        self.fn = nn.GELU(approximate="none")
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fn(self.linear1(x))
        x = self.drop1(x)
        x = self.linear2(x)
        return self.drop2(x)


class ViT3DTransformerBlock(nn.Module):
    """Pre-norm ViT block with fused SDPA attention."""

    def __init__(
        self,
        *,
        hidden_size: int,
        mlp_dim: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        attention_backend: AttentionBackendName = "sdpa",
        require_flash_sdpa: bool = True,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise VisionEncoderConfigurationError(
                "hidden_size must be divisible by num_heads."
            )
        dropout_rate = _validate_dropout(dropout_rate)

        # Attribute names and ordering match MONAI TransformerBlock.
        self.mlp = MLPBlock(hidden_size, mlp_dim, dropout_rate)
        self.norm1 = nn.LayerNorm(hidden_size, eps=_DEFAULT_LAYER_NORM_EPS)
        self.attn = FusedSelfAttention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            qkv_bias=qkv_bias,
            backend=attention_backend,
            require_flash_sdpa=require_flash_sdpa,
        )
        self.norm2 = nn.LayerNorm(hidden_size, eps=_DEFAULT_LAYER_NORM_EPS)

    def forward(
        self,
        x: Tensor,
        *,
        attention_mask: Tensor | None = None,
        valid_token_mask: Tensor | None = None,
    ) -> Tensor:
        x = x + self.attn(
            self.norm1(x),
            attention_mask=attention_mask,
            valid_token_mask=valid_token_mask,
            is_causal=False,
        )
        return x + self.mlp(self.norm2(x))

    def has_trainable_parameters(self) -> bool:
        return any(parameter.requires_grad for parameter in self.parameters())


# ---------------------------------------------------------------------------
# Full encoder
# ---------------------------------------------------------------------------


class ViT3DEncoder(nn.Module):
    """Fixed-resolution 3D ViT with SDPA and optional layer checkpointing."""

    def __init__(
        self,
        config: VisionEncoderConfig,
        *,
        role: VisionEncoderRole,
    ) -> None:
        super().__init__()
        _validate_encoder_config(config, role)

        self.role = role
        self.shape = _build_shape(config)
        self.hidden_size = self.shape.hidden_size
        self.depth = int(config.depth)
        self.num_heads = int(config.num_heads)
        self.mlp_dim = int(config.mlp_dim)
        self.dropout_rate = _validate_dropout(config.dropout)
        self.qkv_bias = bool(config.qkv_bias)
        self.attention_backend: AttentionBackendName = config.attention_backend
        self.require_flash_sdpa = bool(config.require_flash_sdpa)
        self.activation_checkpoint_every_n_layers = int(
            config.activation_checkpoint_every_n_layers
        )
        self.classification = self.shape.has_cls_token

        self.patch_embedding = PatchEmbedding3D(
            in_channels=self.shape.image_channels,
            image_size=self.shape.image_size,
            patch_size=self.shape.patch_size,
            hidden_size=self.shape.hidden_size,
            num_heads=self.num_heads,
            dropout_rate=self.dropout_rate,
        )
        self.blocks = nn.ModuleList(
            [
                ViT3DTransformerBlock(
                    hidden_size=self.shape.hidden_size,
                    mlp_dim=self.mlp_dim,
                    num_heads=self.num_heads,
                    dropout_rate=self.dropout_rate,
                    qkv_bias=self.qkv_bias,
                    attention_backend=self.attention_backend,
                    require_flash_sdpa=self.require_flash_sdpa,
                )
                for _ in range(self.depth)
            ]
        )
        self.norm = nn.LayerNorm(
            self.shape.hidden_size,
            eps=_DEFAULT_LAYER_NORM_EPS,
        )
        if self.shape.has_cls_token:
            self.cls_token = nn.Parameter(
                torch.zeros(1, 1, self.shape.hidden_size)
            )
            _register_ddp_gradient_layout_hook(self.cls_token)

        self.apply_trainability_policy(config)

    @property
    def dtype(self) -> torch.dtype:
        return self.patch_embedding.position_embeddings.dtype

    @property
    def device(self) -> torch.device:
        return self.patch_embedding.position_embeddings.device

    @property
    def patch_grid(self) -> Shape3D:
        return self.shape.patch_grid

    @property
    def num_patches(self) -> int:
        return self.shape.patch_count

    @property
    def output_token_count(self) -> int:
        return self.shape.output_token_count

    def set_activation_checkpointing(self, every_n_layers: int) -> None:
        every_n_layers = int(every_n_layers)
        if every_n_layers < 0:
            raise VisionEncoderConfigurationError(
                "Activation-checkpoint interval cannot be negative."
            )
        self.activation_checkpoint_every_n_layers = every_n_layers

    def _checkpoint_layer(self, layer_index: int) -> bool:
        interval = self.activation_checkpoint_every_n_layers
        return interval > 0 and (layer_index + 1) % interval == 0

    @staticmethod
    def _run_checkpointed_block(
        block: ViT3DTransformerBlock,
        x: Tensor,
        attention_mask: Tensor | None,
        valid_token_mask: Tensor | None,
    ) -> Tensor:
        def block_forward(hidden_states: Tensor) -> Tensor:
            return block(
                hidden_states,
                attention_mask=attention_mask,
                valid_token_mask=valid_token_mask,
            )

        return checkpoint(
            block_forward,
            x,
            use_reentrant=False,
            preserve_rng_state=True,
            determinism_check="default",
        )

    @overload
    def forward(
        self,
        images: Tensor,
        *,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        attention_mask: Tensor | None = None,
        valid_token_mask: Tensor | None = None,
    ) -> VisionEncoderOutput: ...

    @overload
    def forward(
        self,
        images: Tensor,
        *,
        output_hidden_states: bool,
        return_dict: bool,
        attention_mask: Tensor | None = None,
        valid_token_mask: Tensor | None = None,
    ) -> tuple[Tensor, list[Tensor]]: ...

    def forward(
        self,
        images: Tensor,
        *,
        output_hidden_states: bool = False,
        return_dict: bool = True,
        attention_mask: Tensor | None = None,
        valid_token_mask: Tensor | None = None,
    ) -> VisionEncoderOutput | tuple[Tensor, list[Tensor]]:
        x = self.patch_embedding(images)
        if self.shape.has_cls_token:
            # Materialize the tiny per-batch CLS activation. With batch size 1,
            # ``expand`` lets the sliced upstream gradient retain the token
            # sequence stride (for example 288 instead of 32), which violates
            # DDP's gradient-as-bucket-view layout contract and forces a copy
            # on every backward.
            cls_token = self.cls_token.repeat(x.shape[0], 1, 1)
            x = torch.cat((cls_token, x), dim=1)

        expected = (images.shape[0], self.output_token_count, self.hidden_size)
        if tuple(x.shape) != expected:
            raise VisionEncoderExecutionError(
                f"Encoder token shape {tuple(x.shape)} does not match expected {expected}."
            )

        retained: list[Tensor] | None = [] if output_hidden_states else None
        for layer_index, block in enumerate(self.blocks):
            should_checkpoint = (
                self.training
                and torch.is_grad_enabled()
                and self._checkpoint_layer(layer_index)
                and (x.requires_grad or block.has_trainable_parameters())
            )
            if should_checkpoint:
                x = self._run_checkpointed_block(
                    block,
                    x,
                    attention_mask,
                    valid_token_mask,
                )
            else:
                x = block(
                    x,
                    attention_mask=attention_mask,
                    valid_token_mask=valid_token_mask,
                )
            if retained is not None:
                retained.append(x)

        x = self.norm(x)
        if not return_dict:
            return x, retained if retained is not None else []
        return VisionEncoderOutput(
            last_hidden_state=x,
            patch_grid=self.shape.patch_grid,
            has_cls_token=self.shape.has_cls_token,
            hidden_states=tuple(retained) if retained is not None else None,
        )

    def forward_legacy(self, images: Tensor) -> tuple[Tensor, list[Tensor]]:
        """Return the original MONAI ViT tuple ``(last, hidden_states)``."""

        output = self(
            images,
            output_hidden_states=True,
            return_dict=False,
        )
        if not isinstance(output, tuple):  # defensive for type checkers/runtime edits
            raise RuntimeError("Legacy ViT output contract was violated.")
        return output

    def apply_trainability_policy(
        self,
        config: VisionEncoderConfig,
    ) -> TrainabilitySummary:
        """Apply full-train, full-freeze, or last-N-layer unfreezing."""

        for parameter in self.parameters():
            parameter.requires_grad_(True)

        if config.freeze:
            for parameter in self.parameters():
                parameter.requires_grad_(False)
        elif int(config.unfreeze_last_n_layers) > 0:
            for parameter in self.parameters():
                parameter.requires_grad_(False)
            last_n = int(config.unfreeze_last_n_layers)
            for block in self.blocks[-last_n:]:
                for parameter in block.parameters():
                    parameter.requires_grad_(True)
            # The final norm is part of the representation produced by the
            # selected trainable blocks and should adapt with them.
            for parameter in self.norm.parameters():
                parameter.requires_grad_(True)

        return self.trainability_summary()

    def trainability_summary(self) -> TrainabilitySummary:
        total = sum(parameter.numel() for parameter in self.parameters())
        trainable = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return TrainabilitySummary(
            role=self.role.value,
            total_parameters=total,
            trainable_parameters=trainable,
            frozen_parameters=total - trainable,
            trainable_fraction=(trainable / total) if total else 0.0,
        )

    def no_weight_decay_parameter_names(self) -> frozenset[str]:
        names = {"patch_embedding.position_embeddings"}
        if self.shape.has_cls_token:
            names.add("cls_token")
        return frozenset(names)

    def extra_repr(self) -> str:
        return (
            f"role={self.role.value}, image_size={self.shape.image_size}, "
            f"patch_size={self.shape.patch_size}, patch_grid={self.shape.patch_grid}, "
            f"hidden_size={self.hidden_size}, depth={self.depth}, "
            f"num_heads={self.num_heads}, cls_token={self.shape.has_cls_token}, "
            f"checkpoint_every={self.activation_checkpoint_every_n_layers}, "
            f"attention={self.attention_backend}, "
            f"require_flash={self.require_flash_sdpa}"
        )


class ViT3DTower(nn.Module):
    """Main-vision wrapper compatible with the original M3D tower structure."""

    def __init__(self, config: VisionEncoderConfig) -> None:
        super().__init__()
        self.vision_tower = ViT3DEncoder(
            config,
            role=VisionEncoderRole.MAIN,
        )

    def forward(self, images: Tensor) -> Tensor:
        output = self.vision_tower(images)
        if not isinstance(output, VisionEncoderOutput):
            raise RuntimeError("Main vision tower expected a structured ViT output.")
        return output.patch_tokens

    def forward_with_output(
        self,
        images: Tensor,
        *,
        output_hidden_states: bool = False,
    ) -> VisionEncoderOutput:
        output = self.vision_tower(
            images,
            output_hidden_states=output_hidden_states,
        )
        if not isinstance(output, VisionEncoderOutput):
            raise RuntimeError("Main vision tower expected a structured ViT output.")
        return output

    @property
    def dtype(self) -> torch.dtype:
        return self.vision_tower.dtype

    @property
    def device(self) -> torch.device:
        return self.vision_tower.device

    @property
    def hidden_size(self) -> int:
        return self.vision_tower.hidden_size

    @property
    def num_patches(self) -> int:
        return self.vision_tower.num_patches


# ---------------------------------------------------------------------------
# Builders, independence checks, and checkpoint diagnostics
# ---------------------------------------------------------------------------


def build_main_vision_tower(config: VisionEncoderConfig) -> ViT3DTower:
    if not config.enabled:
        raise VisionEncoderConfigurationError("Main vision encoder is disabled.")
    return ViT3DTower(config)


def build_segmentation_vision_encoder(
    config: VisionEncoderConfig,
) -> ViT3DEncoder:
    if not config.enabled:
        raise VisionEncoderConfigurationError(
            "Segmentation vision encoder is disabled."
        )
    return ViT3DEncoder(config, role=VisionEncoderRole.SEGMENTATION)


def build_dual_vision_encoders(model_config: ModelConfig) -> DualVisionEncoderBundle:
    """Instantiate the two M3D encoders as independent modules."""

    main_tower = build_main_vision_tower(model_config.main_vision)
    segmentation_encoder = build_segmentation_vision_encoder(
        model_config.seg_vision
    )
    return DualVisionEncoderBundle(
        main_tower=main_tower,
        segmentation_encoder=segmentation_encoder,
    )


def parameter_identity_set(module: nn.Module) -> frozenset[int]:
    return frozenset(id(parameter) for parameter in module.parameters())


def assert_independent_encoders(
    main_encoder: ViT3DEncoder,
    segmentation_encoder: ViT3DEncoder,
) -> None:
    """Prove that the two encoders do not share module or parameter objects."""

    if main_encoder is segmentation_encoder:
        raise VisionEncoderConfigurationError(
            "Main and SegVol encoders must be different module instances."
        )
    overlap = parameter_identity_set(main_encoder) & parameter_identity_set(
        segmentation_encoder
    )
    if overlap:
        raise VisionEncoderConfigurationError(
            f"Main and SegVol encoders unexpectedly share {len(overlap)} parameters."
        )
    if main_encoder.role is not VisionEncoderRole.MAIN:
        raise VisionEncoderConfigurationError(
            f"Expected Main encoder role, received {main_encoder.role.value!r}."
        )
    if segmentation_encoder.role is not VisionEncoderRole.SEGMENTATION:
        raise VisionEncoderConfigurationError(
            "Expected SegVol encoder role, received "
            f"{segmentation_encoder.role.value!r}."
        )


def monai_compatible_parameter_shapes(
    encoder: ViT3DEncoder,
) -> dict[str, tuple[int, ...]]:
    """Return checkpoint parameter names and shapes for compatibility tests."""

    return {
        name: tuple(parameter.shape)
        for name, parameter in encoder.named_parameters()
    }


def validate_state_dict_shapes(
    encoder: ViT3DEncoder,
    state_dict: StateDict,
    *,
    allow_missing: Iterable[str] = (),
    allow_unexpected: Iterable[str] = (),
) -> None:
    """Validate exact parameter shapes without mutating the encoder.

    Prefix stripping and extraction from complete M3D/SegVol checkpoints are
    intentionally handled by the next checkpoint-loader module.  This function
    validates a state dict that has already been normalised to encoder-local
    names.
    """

    expected = encoder.state_dict()
    allowed_missing = set(allow_missing)
    allowed_unexpected = set(allow_unexpected)

    missing = sorted(set(expected) - set(state_dict) - allowed_missing)
    unexpected = sorted(set(state_dict) - set(expected) - allowed_unexpected)
    mismatched = sorted(
        (
            key,
            tuple(state_dict[key].shape),
            tuple(expected[key].shape),
        )
        for key in set(expected) & set(state_dict)
        if tuple(state_dict[key].shape) != tuple(expected[key].shape)
    )

    if missing or unexpected or mismatched:
        details: list[str] = []
        if missing:
            details.append("missing keys:\n  " + "\n  ".join(missing))
        if unexpected:
            details.append("unexpected keys:\n  " + "\n  ".join(unexpected))
        if mismatched:
            details.append(
                "shape mismatches:\n  "
                + "\n  ".join(
                    f"{key}: checkpoint={actual}, model={wanted}"
                    for key, actual, wanted in mismatched
                )
            )
        raise VisionEncoderConfigurationError(
            "Vision checkpoint is incompatible with the configured encoder:\n"
            + "\n".join(details)
        )


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _reference_perceptron_projection(
    images: Tensor,
    projection: _LinearPatchProjection3D,
) -> Tensor:
    """Explicit MONAI/einops-equivalent patchify + linear reference."""

    batch, channels, depth, height, width = images.shape
    patch_d, patch_h, patch_w = projection.patch_size
    grid_d = depth // patch_d
    grid_h = height // patch_h
    grid_w = width // patch_w

    patches = (
        images.reshape(
            batch,
            channels,
            grid_d,
            patch_d,
            grid_h,
            patch_h,
            grid_w,
            patch_w,
        )
        .permute(0, 2, 4, 6, 3, 5, 7, 1)
        .reshape(batch, grid_d * grid_h * grid_w, -1)
    )
    return F.linear(patches, projection.weight, projection.bias)


def _small_config(*, use_cls_token: bool, checkpoint_every: int = 0) -> VisionEncoderConfig:
    return VisionEncoderConfig(
        image_channels=1,
        image_size=(4, 8, 8),
        patch_size=(2, 4, 4),
        hidden_size=32,
        depth=2,
        num_heads=4,
        mlp_dim=64,
        dropout=0.0,
        qkv_bias=True,
        use_cls_token=use_cls_token,
        attention_backend="math",
        require_flash_sdpa=False,
        activation_checkpoint_every_n_layers=checkpoint_every,
        freeze=False,
        unfreeze_last_n_layers=0,
    )


def run_cpu_self_test() -> dict[str, Any]:
    torch.manual_seed(2026)

    main_config = _small_config(use_cls_token=True, checkpoint_every=1)
    seg_config = _small_config(use_cls_token=False, checkpoint_every=1)
    model_config = ModelConfig(
        main_vision=main_config,
        seg_vision=seg_config,
    )
    bundle = build_dual_vision_encoders(model_config)
    main = bundle.main_tower.vision_tower
    segmentation = bundle.segmentation_encoder
    assert_independent_encoders(main, segmentation)

    images = torch.randn(2, 1, 4, 8, 8, requires_grad=True)

    # Prove that the Conv3d execution matches MONAI's patchify + linear order.
    projected = main.patch_embedding.patch_embeddings(images)
    reference = _reference_perceptron_projection(
        images,
        main.patch_embedding.projection,
    )
    torch.testing.assert_close(projected, reference, rtol=1.0e-5, atol=1.0e-6)

    main.train()
    segmentation.train()
    main_output = main(images, output_hidden_states=True)
    seg_output = segmentation(images)
    if not isinstance(main_output, VisionEncoderOutput):
        raise AssertionError("Main encoder did not return VisionEncoderOutput.")
    if not isinstance(seg_output, VisionEncoderOutput):
        raise AssertionError("Segmentation encoder did not return VisionEncoderOutput.")

    assert tuple(main_output.last_hidden_state.shape) == (2, 9, 32)
    assert tuple(main_output.patch_tokens.shape) == (2, 8, 32)
    assert tuple(main_output.cls_embedding.shape) == (2, 32)
    assert tuple(main_output.spatial_features().shape) == (2, 32, 2, 2, 2)
    assert main_output.hidden_states is not None
    assert len(main_output.hidden_states) == 2

    assert tuple(seg_output.last_hidden_state.shape) == (2, 8, 32)
    assert tuple(seg_output.patch_tokens.shape) == (2, 8, 32)
    assert seg_output.cls_embedding is None
    assert tuple(seg_output.spatial_features().shape) == (2, 32, 2, 2, 2)
    assert seg_output.hidden_states is None

    loss = (
        main_output.last_hidden_state.square().mean()
        + seg_output.last_hidden_state.square().mean()
    )
    loss.backward()
    if images.grad is None or not torch.isfinite(images.grad).all():
        raise AssertionError("3D ViT backward test produced invalid image gradients.")

    main_keys = set(main.state_dict())
    required_keys = {
        "cls_token",
        "patch_embedding.position_embeddings",
        "patch_embedding.patch_embeddings.1.weight",
        "patch_embedding.patch_embeddings.1.bias",
        "blocks.0.mlp.linear1.weight",
        "blocks.0.mlp.linear2.weight",
        "blocks.0.norm1.weight",
        "blocks.0.attn.qkv.weight",
        "blocks.0.attn.out_proj.weight",
        "blocks.0.norm2.weight",
        "norm.weight",
    }
    missing = sorted(required_keys - main_keys)
    if missing:
        raise AssertionError(f"MONAI-compatible state keys are missing: {missing}")
    if "cls_token" in segmentation.state_dict():
        raise AssertionError("SegVol encoder unexpectedly contains a CLS token.")

    # Verify last-N unfreezing policy independently from the training configs.
    partial_config = dataclasses.replace(
        main_config,
        activation_checkpoint_every_n_layers=0,
        unfreeze_last_n_layers=1,
    )
    partial = ViT3DEncoder(partial_config, role=VisionEncoderRole.MAIN)
    summary = partial.trainability_summary()
    assert 0 < summary.trainable_parameters < summary.total_parameters
    assert all(not parameter.requires_grad for parameter in partial.blocks[0].parameters())
    assert all(parameter.requires_grad for parameter in partial.blocks[-1].parameters())

    return {
        "status": "passed",
        "main_output_shape": list(main_output.last_hidden_state.shape),
        "main_patch_shape": list(main_output.patch_tokens.shape),
        "segmentation_output_shape": list(seg_output.last_hidden_state.shape),
        "spatial_feature_shape": list(seg_output.spatial_features().shape),
        "main_parameter_count": sum(p.numel() for p in main.parameters()),
        "segmentation_parameter_count": sum(
            p.numel() for p in segmentation.parameters()
        ),
        "shared_parameter_count": len(
            parameter_identity_set(main) & parameter_identity_set(segmentation)
        ),
        "checkpoint_interval": main.activation_checkpoint_every_n_layers,
        "state_key_count": len(main_keys),
        "partial_trainable_fraction": summary.trainable_fraction,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic CPU tests with Math-SDPA.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.self_test:
        raise SystemExit("Pass --self-test to run the standalone checks.")
    print(json.dumps(run_cpu_self_test(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
