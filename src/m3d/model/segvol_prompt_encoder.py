"""Checkpoint-compatible 3D SegVol prompt encoder for M3D-Modernized.

This module reproduces the prompt encoder used by the original M3D SegVol
path.  In the M3D training graph the prompt encoder receives the projected
Phi-3 ``[SEG]`` representation as a text prompt and returns:

* one sparse prompt token per text prompt, and
* a broadcast no-mask dense embedding over the 3D image-feature grid.

The implementation intentionally preserves the original SegVol state-dict
layout, including:

* ``pe_layer.positional_encoding_gaussian_matrix``
* ``point_embeddings.<0..3>.weight``
* ``not_a_point_embed.weight``
* ``mask_downscaling.<index>``
* ``no_mask_embed.weight``

The legacy ``mask_downscaling`` branch is retained as a 2D module because that
is the architecture stored in the original checkpoint.  M3D itself always
calls the prompt encoder with ``masks=None``; 3D segmentation targets are model
outputs and are never used as input-mask prompts.

Performance improvements
------------------------
* Dense random positional encodings are cached after first construction.
* The no-mask dense embedding uses ``expand`` and does not allocate a repeated
  ``[B, C, D, H, W]`` parameter tensor.
* Text prompts are appended without per-sample Python loops.
* All shape/device contracts are checked before the SegVol decoder runs.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from dataclasses import dataclass
from typing import Any, Final, TypeAlias

import torch
from torch import Tensor, nn

from m3d.config import SegmentationConfig, VisionEncoderConfig


Shape3D: TypeAlias = tuple[int, int, int]
PointPrompt: TypeAlias = tuple[Tensor, Tensor]

_DEFAULT_MASK_INPUT_CHANNELS: Final[int] = 16
_DEFAULT_LAYER_NORM_EPS: Final[float] = 1.0e-6


class SegVolPromptConfigurationError(ValueError):
    """Raised when the prompt encoder configuration is inconsistent."""


class SegVolPromptExecutionError(RuntimeError):
    """Raised when prompt tensors violate the encoder contract."""


@dataclass(frozen=True, slots=True)
class SegVolPromptOutput:
    """Structured output from :class:`SegVolPromptEncoder`."""

    sparse_embeddings: Tensor
    dense_embeddings: Tensor
    dense_positional_encoding: Tensor

    def __post_init__(self) -> None:
        if self.sparse_embeddings.ndim != 3:
            raise SegVolPromptExecutionError(
                "sparse_embeddings must have shape [B, N, C], got "
                f"{tuple(self.sparse_embeddings.shape)}"
            )
        if self.dense_embeddings.ndim != 5:
            raise SegVolPromptExecutionError(
                "dense_embeddings must have shape [B, C, D, H, W], got "
                f"{tuple(self.dense_embeddings.shape)}"
            )
        if self.dense_positional_encoding.ndim != 5:
            raise SegVolPromptExecutionError(
                "dense_positional_encoding must have shape [1, C, D, H, W], "
                f"got {tuple(self.dense_positional_encoding.shape)}"
            )
        if self.sparse_embeddings.shape[0] != self.dense_embeddings.shape[0]:
            raise SegVolPromptExecutionError(
                "Sparse and dense prompt batch sizes differ: "
                f"{self.sparse_embeddings.shape[0]} vs "
                f"{self.dense_embeddings.shape[0]}"
            )
        if self.sparse_embeddings.shape[-1] != self.dense_embeddings.shape[1]:
            raise SegVolPromptExecutionError(
                "Sparse and dense prompt embedding dimensions differ: "
                f"{self.sparse_embeddings.shape[-1]} vs "
                f"{self.dense_embeddings.shape[1]}"
            )
        if self.dense_embeddings.shape[1:] != self.dense_positional_encoding.shape[1:]:
            raise SegVolPromptExecutionError(
                "Dense prompt and positional-encoding shapes differ: "
                f"{tuple(self.dense_embeddings.shape[1:])} vs "
                f"{tuple(self.dense_positional_encoding.shape[1:])}"
            )

    @property
    def batch_size(self) -> int:
        return int(self.sparse_embeddings.shape[0])

    @property
    def prompt_count(self) -> int:
        return int(self.sparse_embeddings.shape[1])

    @property
    def embed_dim(self) -> int:
        return int(self.sparse_embeddings.shape[2])


@dataclass(frozen=True, slots=True)
class SegVolPromptEncoderReport:
    """Serializable description of a built prompt encoder."""

    embed_dim: int
    image_embedding_size: Shape3D
    input_image_size: Shape3D
    mask_input_channels: int
    parameter_count: int
    trainable_parameter_count: int
    frozen: bool
    state_dict_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SegVolPromptConfigurationError(
            f"{name} must be a positive integer, got {value!r}"
        )
    return value


def _shape3d(value: tuple[int, ...] | list[int], *, name: str) -> Shape3D:
    if len(value) != 3:
        raise SegVolPromptConfigurationError(
            f"{name} must contain exactly three values, got {value!r}"
        )
    shape = tuple(_positive_int(int(item), name=f"{name}[{index}]") for index, item in enumerate(value))
    return shape  # type: ignore[return-value]


class LayerNorm2d(nn.Module):
    """Channel-first LayerNorm used by the original SegVol checkpoint.

    This preserves the original module parameter names ``weight`` and ``bias``.
    It is used only by the legacy 2D mask-prompt downscaling branch.
    """

    def __init__(self, num_channels: int, eps: float = _DEFAULT_LAYER_NORM_EPS) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(_positive_int(num_channels, name="num_channels")))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = float(eps)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise SegVolPromptExecutionError(
                f"LayerNorm2d expects [B, C, H, W], got {tuple(x.shape)}"
            )
        mean = x.mean(dim=1, keepdim=True)
        variance = (x - mean).pow(2).mean(dim=1, keepdim=True)
        normalized = (x - mean) * torch.rsqrt(variance + self.eps)
        return (
            normalized * self.weight[:, None, None]
            + self.bias[:, None, None]
        )


class PositionEmbeddingRandom(nn.Module):
    """Random Fourier positional encoding for three-dimensional coordinates."""

    def __init__(
        self,
        num_pos_feats: int = 64,
        scale: float | None = None,
    ) -> None:
        super().__init__()
        num_pos_feats = _positive_int(num_pos_feats, name="num_pos_feats")
        resolved_scale = 1.0 if scale is None or scale <= 0.0 else float(scale)
        self.register_buffer(
            "positional_encoding_gaussian_matrix",
            resolved_scale * torch.randn(3, num_pos_feats),
        )

    @property
    def output_dim(self) -> int:
        return int(self.positional_encoding_gaussian_matrix.shape[1] * 2)

    def _pe_encoding(self, coords: Tensor) -> Tensor:
        if coords.shape[-1] != 3:
            raise SegVolPromptExecutionError(
                "3D coordinates must have final dimension 3, got "
                f"{tuple(coords.shape)}"
            )
        coords = coords.to(
            device=self.positional_encoding_gaussian_matrix.device,
            dtype=self.positional_encoding_gaussian_matrix.dtype,
        )
        coords = 2.0 * coords - 1.0
        coords = coords @ self.positional_encoding_gaussian_matrix
        coords = 2.0 * torch.pi * coords
        return torch.cat((torch.sin(coords), torch.cos(coords)), dim=-1)

    def forward(self, size: Shape3D) -> Tensor:
        """Return ``[C, D, H, W]`` positional encoding.

        The axis arithmetic deliberately matches the original implementation.
        The original variables were named ``h, w, d`` even though M3D supplies
        ``(D, H, W)``.  Preserving the operations and stack order is necessary
        for checkpoint/output compatibility.
        """

        depth, height, width = _shape3d(size, name="size")
        matrix = self.positional_encoding_gaussian_matrix
        device = matrix.device
        dtype = matrix.dtype

        # arange is cheaper and clearer than materialising ones followed by
        # cumsum, while producing exactly the same centred coordinates.
        depth_coords = (
            torch.arange(depth, device=device, dtype=dtype) + 0.5
        ) / depth
        height_coords = (
            torch.arange(height, device=device, dtype=dtype) + 0.5
        ) / height
        width_coords = (
            torch.arange(width, device=device, dtype=dtype) + 0.5
        ) / width

        d_grid, h_grid, w_grid = torch.meshgrid(
            depth_coords,
            height_coords,
            width_coords,
            indexing="ij",
        )

        # Original stack order was [x_embed, y_embed, z_embed] where x varied
        # along the second dimension, y along the first, and z along the third.
        encoded = self._pe_encoding(torch.stack((h_grid, d_grid, w_grid), dim=-1))
        return encoded.permute(3, 0, 1, 2).contiguous()

    def forward_with_coords(
        self,
        coords_input: Tensor,
        image_size: Shape3D,
    ) -> Tensor:
        """Encode absolute ``[x, y, z]`` point coordinates."""

        if coords_input.ndim != 3 or coords_input.shape[-1] != 3:
            raise SegVolPromptExecutionError(
                "coords_input must have shape [B, N, 3], got "
                f"{tuple(coords_input.shape)}"
            )
        depth, height, width = _shape3d(image_size, name="image_size")
        coords = coords_input.to(
            device=self.positional_encoding_gaussian_matrix.device,
            dtype=torch.float32,
        ).clone()

        # Preserve the original SegVol coordinate normalisation exactly:
        # x / H, y / D, z / W for an input size stored as (D, H, W).
        coords[..., 0] = coords[..., 0] / height
        coords[..., 1] = coords[..., 1] / depth
        coords[..., 2] = coords[..., 2] / width
        return self._pe_encoding(coords)


class SegVolPromptEncoder(nn.Module):
    """Encode sparse text/point/box prompts and dense no-mask prompts."""

    def __init__(
        self,
        *,
        embed_dim: int,
        image_embedding_size: Shape3D,
        input_image_size: Shape3D,
        mask_in_chans: int = _DEFAULT_MASK_INPUT_CHANNELS,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        self.embed_dim = _positive_int(embed_dim, name="embed_dim")
        if self.embed_dim % 2 != 0:
            raise SegVolPromptConfigurationError(
                f"embed_dim must be even for sin/cos positional encoding, got {embed_dim}"
            )
        self.image_embedding_size = _shape3d(
            image_embedding_size,
            name="image_embedding_size",
        )
        self.input_image_size = _shape3d(
            input_image_size,
            name="input_image_size",
        )
        self.mask_in_chans = _positive_int(mask_in_chans, name="mask_in_chans")
        if self.mask_in_chans % 4 != 0:
            raise SegVolPromptConfigurationError(
                "mask_in_chans must be divisible by 4 to match the legacy "
                f"SegVol prompt encoder, got {self.mask_in_chans}"
            )

        self.pe_layer = PositionEmbeddingRandom(self.embed_dim // 2)

        self.num_point_embeddings = 4
        self.point_embeddings = nn.ModuleList(
            nn.Embedding(1, self.embed_dim)
            for _ in range(self.num_point_embeddings)
        )
        self.not_a_point_embed = nn.Embedding(1, self.embed_dim)

        # Kept exactly as Conv2d/LayerNorm2d for compatibility with the
        # original SegVol checkpoint. M3D's text-prompt path never executes it.
        self.mask_input_size = tuple(
            4 * item for item in self.image_embedding_size
        )
        self.mask_downscaling = nn.Sequential(
            nn.Conv2d(1, self.mask_in_chans // 4, kernel_size=2, stride=2),
            LayerNorm2d(self.mask_in_chans // 4),
            activation(),
            nn.Conv2d(
                self.mask_in_chans // 4,
                self.mask_in_chans,
                kernel_size=2,
                stride=2,
            ),
            LayerNorm2d(self.mask_in_chans),
            activation(),
            nn.Conv2d(self.mask_in_chans, self.embed_dim, kernel_size=1),
        )
        self.no_mask_embed = nn.Embedding(1, self.embed_dim)

        # Cache is deliberately non-persistent so the checkpoint key set stays
        # identical to the original prompt encoder.
        self.register_buffer(
            "_dense_pe_cache",
            torch.empty(0),
            persistent=False,
        )

    @property
    def device(self) -> torch.device:
        return self.no_mask_embed.weight.device

    @property
    def dtype(self) -> torch.dtype:
        return self.no_mask_embed.weight.dtype

    def _invalidate_dense_pe_cache(self) -> None:
        self._dense_pe_cache = torch.empty(
            0,
            device=self.pe_layer.positional_encoding_gaussian_matrix.device,
            dtype=self.pe_layer.positional_encoding_gaussian_matrix.dtype,
        )

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Tensor],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        self._invalidate_dense_pe_cache()
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def get_dense_pe(self) -> Tensor:
        """Return cached positional encoding with shape ``[1, C, D, H, W]``."""

        expected_shape = (1, self.embed_dim, *self.image_embedding_size)
        target_device = self.device
        target_dtype = self.dtype
        cache = self._dense_pe_cache

        if (
            tuple(cache.shape) != expected_shape
            or cache.device != target_device
            or cache.dtype != target_dtype
        ):
            # Positional encoding is a cached module-state tensor. Build it in the
            # prompt encoder's native dtype instead of allowing the surrounding
            # BF16 autocast context to change its dtype.
            with torch.autocast(
                device_type=target_device.type,
                enabled=False,
            ):
                cache = self.pe_layer(self.image_embedding_size).unsqueeze(0)

            cache = cache.to(
                device=target_device,
                dtype=target_dtype,
            )
            self._dense_pe_cache = cache

        return self._dense_pe_cache

    def _validate_prompt_device(self, tensor: Tensor, *, name: str) -> None:
        if tensor.device != self.device:
            raise SegVolPromptExecutionError(
                f"{name} is on {tensor.device}, but prompt encoder is on "
                f"{self.device}. Move the complete batch/module through the "
                "runtime instead of transferring tensors inside forward()."
            )

    def _embed_points(
        self,
        points: Tensor,
        labels: Tensor,
        *,
        pad: bool,
    ) -> Tensor:
        if points.ndim != 3 or points.shape[-1] != 3:
            raise SegVolPromptExecutionError(
                f"points must have shape [B, N, 3], got {tuple(points.shape)}"
            )
        if labels.shape != points.shape[:2]:
            raise SegVolPromptExecutionError(
                "point labels must have shape [B, N] matching points; got "
                f"points={tuple(points.shape)}, labels={tuple(labels.shape)}"
            )
        self._validate_prompt_device(points, name="points")
        self._validate_prompt_device(labels, name="point labels")

        shifted_points = points.to(dtype=torch.float32) + 0.5
        effective_labels = labels.to(dtype=torch.long)
        if pad:
            padding_point = shifted_points.new_zeros(
                shifted_points.shape[0],
                1,
                3,
            )
            padding_label = effective_labels.new_full(
                (effective_labels.shape[0], 1),
                -1,
            )
            shifted_points = torch.cat((shifted_points, padding_point), dim=1)
            effective_labels = torch.cat(
                (effective_labels, padding_label),
                dim=1,
            )

        valid_labels = torch.logical_or(
            torch.logical_or(effective_labels == -1, effective_labels == 0),
            effective_labels == 1,
        )
        if not bool(torch.all(valid_labels)):
            invalid = torch.unique(effective_labels[~valid_labels]).tolist()
            raise SegVolPromptExecutionError(
                f"point labels must be -1, 0 or 1; found {invalid}"
            )

        embeddings = self.pe_layer.forward_with_coords(
            shifted_points,
            self.input_image_size,
        ).to(dtype=self.dtype)

        # Vectorised equivalent of the original indexed in-place updates.
        is_padding = effective_labels == -1
        is_negative = effective_labels == 0
        is_positive = effective_labels == 1
        embeddings = embeddings.masked_fill(is_padding.unsqueeze(-1), 0.0)
        embeddings = embeddings + is_padding.unsqueeze(-1) * self.not_a_point_embed.weight
        embeddings = embeddings + is_negative.unsqueeze(-1) * self.point_embeddings[0].weight
        embeddings = embeddings + is_positive.unsqueeze(-1) * self.point_embeddings[1].weight
        return embeddings

    def _embed_boxes(self, boxes: Tensor) -> Tensor:
        if boxes.ndim == 2 and boxes.shape[-1] == 6:
            coords = boxes.reshape(-1, 2, 3)
        elif boxes.ndim == 3 and boxes.shape[1:] == (2, 3):
            coords = boxes
        else:
            raise SegVolPromptExecutionError(
                "boxes must have shape [B, 6] or [B, 2, 3], got "
                f"{tuple(boxes.shape)}"
            )
        self._validate_prompt_device(boxes, name="boxes")
        coords = coords.to(dtype=torch.float32) + 0.5
        embeddings = self.pe_layer.forward_with_coords(
            coords,
            self.input_image_size,
        ).to(dtype=self.dtype)
        corner_offsets = torch.stack(
            (
                self.point_embeddings[2].weight[0],
                self.point_embeddings[3].weight[0],
            ),
            dim=0,
        )
        return embeddings + corner_offsets.unsqueeze(0)

    def _embed_masks(self, masks: Tensor) -> Tensor:
        """Run the original checkpoint-compatible 2D mask-prompt branch.

        The M3D text-to-segmentation path never supplies input mask prompts.
        A 5D 3D mask is rejected explicitly rather than being accidentally fed
        to the original Conv2d architecture.
        """

        if masks.ndim != 4:
            raise SegVolPromptExecutionError(
                "The original SegVol checkpoint stores a 2D Conv2d mask-prompt "
                "branch, which accepts [B, 1, H, W]. M3D uses masks=None for "
                "prompt encoding. Received shape "
                f"{tuple(masks.shape)}."
            )
        if masks.shape[1] != 1:
            raise SegVolPromptExecutionError(
                f"mask prompts must have one channel, got {masks.shape[1]}"
            )
        self._validate_prompt_device(masks, name="masks")
        return self.mask_downscaling(masks.to(dtype=self.dtype))

    def _resolve_batch_size(
        self,
        points: PointPrompt | None,
        boxes: Tensor | None,
        masks: Tensor | None,
        text_embedding: Tensor | None,
    ) -> int:
        candidates: list[tuple[str, int]] = []
        if points is not None:
            candidates.append(("points", int(points[0].shape[0])))
        if boxes is not None:
            candidates.append(("boxes", int(boxes.shape[0])))
        if masks is not None:
            candidates.append(("masks", int(masks.shape[0])))
        if text_embedding is not None:
            candidates.append(("text_embedding", int(text_embedding.shape[0])))
        if not candidates:
            return 1
        unique_sizes = {size for _, size in candidates}
        if len(unique_sizes) != 1:
            description = ", ".join(f"{name}={size}" for name, size in candidates)
            raise SegVolPromptExecutionError(
                f"Prompt batch sizes disagree: {description}"
            )
        batch_size = unique_sizes.pop()
        if batch_size <= 0:
            raise SegVolPromptExecutionError(
                f"Prompt batch size must be positive, got {batch_size}"
            )
        return batch_size

    def _prepare_text_embedding(self, text_embedding: Tensor) -> Tensor:
        if text_embedding.ndim == 2:
            text_embedding = text_embedding.unsqueeze(1)
        elif text_embedding.ndim != 3:
            raise SegVolPromptExecutionError(
                "text_embedding must have shape [B, C] or [B, N, C], got "
                f"{tuple(text_embedding.shape)}"
            )
        if text_embedding.shape[-1] != self.embed_dim:
            raise SegVolPromptExecutionError(
                f"text prompt dimension must be {self.embed_dim}, got "
                f"{text_embedding.shape[-1]}"
            )
        self._validate_prompt_device(text_embedding, name="text_embedding")
        if not torch.is_floating_point(text_embedding):
            raise SegVolPromptExecutionError(
                f"text_embedding must be floating point, got {text_embedding.dtype}"
            )
        return text_embedding.to(dtype=self.dtype)

    def forward(
        self,
        points: PointPrompt | None = None,
        boxes: Tensor | None = None,
        masks: Tensor | None = None,
        text_embedding: Tensor | None = None,
        *,
        return_structured: bool = False,
    ) -> tuple[Tensor, Tensor] | SegVolPromptOutput:
        batch_size = self._resolve_batch_size(
            points,
            boxes,
            masks,
            text_embedding,
        )

        sparse_parts: list[Tensor] = []
        if points is not None:
            coordinates, labels = points
            sparse_parts.append(
                self._embed_points(
                    coordinates,
                    labels,
                    pad=boxes is None,
                )
            )
        if boxes is not None:
            sparse_parts.append(self._embed_boxes(boxes))
        if text_embedding is not None:
            sparse_parts.append(self._prepare_text_embedding(text_embedding))

        if sparse_parts:
            sparse_embeddings = torch.cat(sparse_parts, dim=1)
        else:
            sparse_embeddings = self.no_mask_embed.weight.new_empty(
                batch_size,
                0,
                self.embed_dim,
            )

        if masks is not None:
            dense_embeddings = self._embed_masks(masks)
            if dense_embeddings.ndim != 5:
                # The legacy 2D branch is retained only for checkpoint
                # compatibility and is not accepted by the 3D decoder.
                raise SegVolPromptExecutionError(
                    "Legacy 2D mask prompt produced a 4D embedding and cannot "
                    "be consumed by the 3D SegVol decoder. Use masks=None, as "
                    "the original M3D training path does."
                )
        else:
            dense_embeddings = self.no_mask_embed.weight.reshape(
                1,
                self.embed_dim,
                1,
                1,
                1,
            ).expand(
                batch_size,
                self.embed_dim,
                *self.image_embedding_size,
            ).clone()
            # This value crosses the PromptEncoder FSDP2 forward boundary.
            # Returning the zero-stride expanded parameter view directly would
            # leave MaskDecoder holding storage that FSDP frees when it
            # reshards ``no_mask_embed.weight`` after this forward. ``clone``
            # materializes an activation while preserving autograd to the
            # embedding parameter.

        dense_pe = self.get_dense_pe()
        if return_structured:
            return SegVolPromptOutput(
                sparse_embeddings=sparse_embeddings,
                dense_embeddings=dense_embeddings,
                dense_positional_encoding=dense_pe,
            )
        return sparse_embeddings, dense_embeddings

    def freeze(self) -> None:
        self.requires_grad_(False)

    def unfreeze(self) -> None:
        self.requires_grad_(True)

    def report(self) -> SegVolPromptEncoderReport:
        parameter_count = sum(parameter.numel() for parameter in self.parameters())
        trainable_count = sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )
        return SegVolPromptEncoderReport(
            embed_dim=self.embed_dim,
            image_embedding_size=self.image_embedding_size,
            input_image_size=self.input_image_size,
            mask_input_channels=self.mask_in_chans,
            parameter_count=parameter_count,
            trainable_parameter_count=trainable_count,
            frozen=trainable_count == 0,
            state_dict_keys=tuple(sorted(self.state_dict().keys())),
        )


def build_segvol_prompt_encoder(
    *,
    segmentation_config: SegmentationConfig,
    segmentation_vision_config: VisionEncoderConfig,
    mask_input_channels: int = _DEFAULT_MASK_INPUT_CHANNELS,
) -> SegVolPromptEncoder:
    """Build the checkpoint-compatible prompt encoder from project config."""

    image_size = _shape3d(
        segmentation_vision_config.image_size,
        name="seg_vision.image_size",
    )
    patch_size = _shape3d(
        segmentation_vision_config.patch_size,
        name="seg_vision.patch_size",
    )
    for axis, (image_axis, patch_axis) in enumerate(zip(image_size, patch_size, strict=True)):
        if image_axis % patch_axis != 0:
            raise SegVolPromptConfigurationError(
                "seg_vision.image_size must be divisible by patch_size; "
                f"axis {axis}: {image_axis} % {patch_axis} != 0"
            )
    image_embedding_size: Shape3D = tuple(
        image_axis // patch_axis
        for image_axis, patch_axis in zip(image_size, patch_size, strict=True)
    )  # type: ignore[assignment]

    if segmentation_config.prompt_embed_dim != segmentation_vision_config.hidden_size:
        raise SegVolPromptConfigurationError(
            "SegVol prompt_embed_dim must equal the segmentation image encoder "
            "hidden size so sparse/dense prompts can enter the mask decoder; "
            f"got {segmentation_config.prompt_embed_dim} and "
            f"{segmentation_vision_config.hidden_size}."
        )

    encoder = SegVolPromptEncoder(
        embed_dim=segmentation_config.prompt_embed_dim,
        image_embedding_size=image_embedding_size,
        input_image_size=image_size,
        mask_in_chans=mask_input_channels,
    )
    if segmentation_config.freeze_prompt_encoder:
        encoder.freeze()
    return encoder


def validate_segvol_prompt_contract(
    *,
    encoder: SegVolPromptEncoder,
    segmentation_prompt_dim: int,
    segmentation_image_embedding_size: Shape3D,
) -> None:
    if encoder.embed_dim != int(segmentation_prompt_dim):
        raise SegVolPromptConfigurationError(
            "Language-to-SegVol projector output dimension does not match the "
            f"prompt encoder: {segmentation_prompt_dim} vs {encoder.embed_dim}."
        )
    expected = _shape3d(
        segmentation_image_embedding_size,
        name="segmentation_image_embedding_size",
    )
    if encoder.image_embedding_size != expected:
        raise SegVolPromptConfigurationError(
            "Prompt encoder image embedding grid does not match SegVol image "
            f"features: {encoder.image_embedding_size} vs {expected}."
        )


def _legacy_dense_pe_reference(
    pe_layer: PositionEmbeddingRandom,
    size: Shape3D,
) -> Tensor:
    depth, height, width = size
    matrix = pe_layer.positional_encoding_gaussian_matrix
    grid = torch.ones((depth, height, width), device=matrix.device, dtype=matrix.dtype)
    y_embed = grid.cumsum(dim=0) - 0.5
    x_embed = grid.cumsum(dim=1) - 0.5
    z_embed = grid.cumsum(dim=2) - 0.5
    y_embed = y_embed / depth
    x_embed = x_embed / height
    z_embed = z_embed / width
    encoded = pe_layer._pe_encoding(torch.stack((x_embed, y_embed, z_embed), dim=-1))
    return encoded.permute(3, 0, 1, 2).contiguous()


def _self_test() -> dict[str, Any]:
    torch.manual_seed(7)
    encoder = SegVolPromptEncoder(
        embed_dim=32,
        image_embedding_size=(2, 3, 4),
        input_image_size=(8, 12, 16),
        mask_in_chans=16,
    )

    text = torch.randn(2, 32, requires_grad=True)
    output = encoder(
        text_embedding=text,
        return_structured=True,
    )
    assert isinstance(output, SegVolPromptOutput)
    assert output.sparse_embeddings.shape == (2, 1, 32)
    assert output.dense_embeddings.shape == (2, 32, 2, 3, 4)
    assert output.dense_positional_encoding.shape == (1, 32, 2, 3, 4)
    assert output.dense_embeddings.stride(0) != 0

    loss = (
        output.sparse_embeddings.square().mean()
        + output.dense_embeddings.square().mean()
    )
    loss.backward()
    assert text.grad is not None and torch.isfinite(text.grad).all()
    assert encoder.no_mask_embed.weight.grad is not None
    assert torch.isfinite(encoder.no_mask_embed.weight.grad).all()

    reference = _legacy_dense_pe_reference(
        encoder.pe_layer,
        encoder.image_embedding_size,
    )
    modern = encoder.get_dense_pe().squeeze(0)
    torch.testing.assert_close(modern, reference, rtol=0.0, atol=0.0)
    first_cache_ptr = encoder.get_dense_pe().data_ptr()
    second_cache_ptr = encoder.get_dense_pe().data_ptr()
    assert first_cache_ptr == second_cache_ptr

    points = torch.tensor(
        [
            [[1.0, 2.0, 3.0]],
            [[2.0, 3.0, 4.0]],
        ]
    )
    labels = torch.tensor([[1], [0]])
    point_sparse, _ = encoder(points=(points, labels))
    assert point_sparse.shape == (2, 2, 32)  # one point + padding point

    boxes = torch.tensor(
        [
            [0.0, 0.0, 0.0, 3.0, 4.0, 5.0],
            [1.0, 1.0, 1.0, 4.0, 5.0, 6.0],
        ]
    )
    box_sparse, _ = encoder(boxes=boxes)
    assert box_sparse.shape == (2, 2, 32)

    expected_keys = {
        "pe_layer.positional_encoding_gaussian_matrix",
        "point_embeddings.0.weight",
        "point_embeddings.1.weight",
        "point_embeddings.2.weight",
        "point_embeddings.3.weight",
        "not_a_point_embed.weight",
        "mask_downscaling.0.weight",
        "mask_downscaling.0.bias",
        "mask_downscaling.1.weight",
        "mask_downscaling.1.bias",
        "mask_downscaling.3.weight",
        "mask_downscaling.3.bias",
        "mask_downscaling.4.weight",
        "mask_downscaling.4.bias",
        "mask_downscaling.6.weight",
        "mask_downscaling.6.bias",
        "no_mask_embed.weight",
    }
    assert set(encoder.state_dict().keys()) == expected_keys

    caught_3d_mask_error = False
    try:
        encoder(masks=torch.zeros(2, 1, 8, 12, 16))
    except SegVolPromptExecutionError:
        caught_3d_mask_error = True
    assert caught_3d_mask_error

    return {
        "status": "passed",
        "sparse_shape": list(output.sparse_embeddings.shape),
        "dense_shape": list(output.dense_embeddings.shape),
        "dense_pe_shape": list(output.dense_positional_encoding.shape),
        "dense_pe_legacy_equivalence": True,
        "dense_pe_cache_reused": first_cache_ptr == second_cache_ptr,
        "no_mask_dense_storage_materialized": (
            output.dense_embeddings.stride(0) != 0
        ),
        "no_mask_dense_gradient_preserved": True,
        "point_prompt_shape": list(point_sparse.shape),
        "box_prompt_shape": list(box_sparse.shape),
        "checkpoint_key_count": len(expected_keys),
        "legacy_3d_mask_prompt_rejected": caught_3d_mask_error,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run CPU-only checkpoint/shape/numerical tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.self_test:
        raise SystemExit("Nothing to do. Pass --self-test.")
    print(json.dumps(_self_test(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
