"""Checkpoint-compatible 3D SegVol mask decoder for M3D-Modernized.

This module reproduces the mask decoder used by the original M3D SegVol path
while connecting it to the SDPA-based :mod:`m3d.model.segvol_transformer`.
The decoder keeps the original state-dict layout, including:

* ``iou_token.weight`` and ``mask_tokens.weight``;
* ``output_upscaling.<index>``;
* ``output_hypernetworks_mlps.<mask>.layers.<layer>``;
* ``iou_prediction_head.layers.<layer>``; and
* ``txt_align_upscaled_embedding``.

The mathematical model is unchanged.  The implementation removes a redundant
``repeat`` of the text-similarity map, validates all tensor contracts before
large operations, and uses batched matrix multiplication for mask generation.

For the production M3D geometry, the independent SegVol image encoder produces
``[B, 768, 8, 16, 16]`` features.  Two transposed convolutions upscale these to
``[B, 96, 32, 64, 64]`` before the final logits are resized to the original CT
shape by the outer SegVol module.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from dataclasses import dataclass
from typing import Any, Final, Iterable, TypeAlias

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from m3d.config import OptimizationConfig, SegmentationConfig, VisionEncoderConfig
from m3d.model.segvol_transformer import (
    TwoWayTransformer,
    build_segvol_two_way_transformer,
)


Shape3D: TypeAlias = tuple[int, int, int]

_DEFAULT_NUM_MULTIMASK_OUTPUTS: Final[int] = 3
_DEFAULT_IOU_HEAD_DEPTH: Final[int] = 3
_DEFAULT_IOU_HEAD_HIDDEN_DIM: Final[int] = 256
_DEFAULT_LAYER_NORM_EPS: Final[float] = 1.0e-5
_DEFAULT_TRANSFORMER_MLP_DIM: Final[int] = 2048
_DEFAULT_ATTENTION_DOWNSAMPLE_RATE: Final[int] = 2


class SegVolMaskDecoderConfigurationError(ValueError):
    """Raised when the decoder architecture is internally inconsistent."""


class SegVolMaskDecoderExecutionError(RuntimeError):
    """Raised when runtime tensors violate the mask-decoder contract."""


@dataclass(frozen=True, slots=True)
class SegVolMaskDecoderOutput:
    """Structured mask-decoder result.

    ``masks`` contains either one mask token or the three disambiguation mask
    tokens, depending on ``multimask_output``. ``all_masks`` always contains
    every mask token and is useful for diagnostics without recomputing the
    decoder.
    """

    masks: Tensor
    iou_predictions: Tensor
    all_masks: Tensor
    all_iou_predictions: Tensor
    upscaled_embedding_shape: tuple[int, int, int, int, int]
    multimask_output: bool

    def __post_init__(self) -> None:
        if self.masks.ndim != 5:
            raise SegVolMaskDecoderExecutionError(
                f"masks must be [B,M,D,H,W], got {tuple(self.masks.shape)}."
            )
        if self.iou_predictions.ndim != 2:
            raise SegVolMaskDecoderExecutionError(
                "iou_predictions must be [B,M], got "
                f"{tuple(self.iou_predictions.shape)}."
            )
        if self.all_masks.ndim != 5 or self.all_iou_predictions.ndim != 2:
            raise SegVolMaskDecoderExecutionError(
                "all_masks/all_iou_predictions have invalid rank."
            )
        if self.masks.shape[:2] != self.iou_predictions.shape:
            raise SegVolMaskDecoderExecutionError(
                "Selected mask and IoU shapes disagree: "
                f"{tuple(self.masks.shape[:2])} vs "
                f"{tuple(self.iou_predictions.shape)}."
            )
        if self.all_masks.shape[:2] != self.all_iou_predictions.shape:
            raise SegVolMaskDecoderExecutionError(
                "All-mask and all-IoU shapes disagree: "
                f"{tuple(self.all_masks.shape[:2])} vs "
                f"{tuple(self.all_iou_predictions.shape)}."
            )

    @property
    def batch_size(self) -> int:
        return int(self.masks.shape[0])

    @property
    def selected_mask_count(self) -> int:
        return int(self.masks.shape[1])


@dataclass(frozen=True, slots=True)
class SegVolMaskDecoderReport:
    """Serializable architecture and parameter summary."""

    transformer_dim: int
    image_embedding_size: Shape3D
    first_upscaled_size: Shape3D
    low_resolution_mask_size: Shape3D
    upscaled_channels: int
    num_multimask_outputs: int
    num_mask_tokens: int
    iou_head_depth: int
    iou_head_hidden_dim: int
    parameter_count: int
    trainable_parameter_count: int
    frozen: bool
    state_dict_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class MLP(nn.Module):
    """SegVol-compatible MLP used by mask hypernetworks and the IoU head.

    The ``layers`` ModuleList and its exact nesting intentionally match the
    original checkpoint keys.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int,
        sigmoid_output: bool = False,
    ) -> None:
        super().__init__()
        input_dim = _positive_int(input_dim, name="input_dim")
        hidden_dim = _positive_int(hidden_dim, name="hidden_dim")
        output_dim = _positive_int(output_dim, name="output_dim")
        num_layers = _positive_int(num_layers, name="num_layers")

        hidden_dims = [hidden_dim] * (num_layers - 1)
        in_dims = [input_dim, *hidden_dims]
        out_dims = [*hidden_dims, output_dim]
        self.layers = nn.ModuleList(
            nn.Linear(in_features, out_features)
            for in_features, out_features in zip(in_dims, out_dims, strict=True)
        )
        self.num_layers = num_layers
        self.sigmoid_output = bool(sigmoid_output)

    def forward(self, x: Tensor) -> Tensor:
        if not x.is_floating_point():
            raise SegVolMaskDecoderExecutionError(
                f"MLP input must be floating point, got {x.dtype}."
            )
        for layer_index, layer in enumerate(self.layers):
            x = layer(x)
            if layer_index < self.num_layers - 1:
                x = F.relu(x)
        if self.sigmoid_output:
            x = torch.sigmoid(x)
        return x


class SegVolMaskDecoder(nn.Module):
    """Three-dimensional SegVol mask decoder with SDPA two-way attention."""

    def __init__(
        self,
        *,
        transformer_dim: int,
        transformer: TwoWayTransformer,
        image_size: Shape3D,
        patch_size: Shape3D,
        num_multimask_outputs: int = _DEFAULT_NUM_MULTIMASK_OUTPUTS,
        activation: type[nn.Module] = nn.GELU,
        iou_head_depth: int = _DEFAULT_IOU_HEAD_DEPTH,
        iou_head_hidden_dim: int = _DEFAULT_IOU_HEAD_HIDDEN_DIM,
        frozen: bool = False,
    ) -> None:
        super().__init__()

        transformer_dim = _positive_int(transformer_dim, name="transformer_dim")
        image_size = _shape3d(image_size, name="image_size")
        patch_size = _shape3d(patch_size, name="patch_size")
        num_multimask_outputs = _positive_int(
            num_multimask_outputs,
            name="num_multimask_outputs",
        )
        iou_head_depth = _positive_int(iou_head_depth, name="iou_head_depth")
        iou_head_hidden_dim = _positive_int(
            iou_head_hidden_dim,
            name="iou_head_hidden_dim",
        )
        if not isinstance(transformer, TwoWayTransformer):
            raise SegVolMaskDecoderConfigurationError(
                "transformer must be a TwoWayTransformer instance."
            )
        if transformer.embedding_dim != transformer_dim:
            raise SegVolMaskDecoderConfigurationError(
                "transformer.embedding_dim must equal transformer_dim: "
                f"{transformer.embedding_dim} != {transformer_dim}."
            )
        if transformer_dim % 8 != 0:
            raise SegVolMaskDecoderConfigurationError(
                "transformer_dim must be divisible by 8 for the original "
                f"SegVol upscaling/hypernetwork layout; got {transformer_dim}."
            )
        if not isinstance(activation, type) or not issubclass(activation, nn.Module):
            raise SegVolMaskDecoderConfigurationError(
                "activation must be an nn.Module class, for example nn.GELU."
            )
        for axis, (image_axis, patch_axis) in enumerate(
            zip(image_size, patch_size, strict=True)
        ):
            if image_axis % patch_axis != 0:
                raise SegVolMaskDecoderConfigurationError(
                    "image_size must be divisible by patch_size on every axis: "
                    f"axis={axis}, image={image_axis}, patch={patch_axis}."
                )

        self.transformer_dim = transformer_dim
        self.transformer = transformer
        self.num_multimask_outputs = num_multimask_outputs
        self.num_mask_tokens = num_multimask_outputs + 1
        self.image_size = image_size
        self.patch_size = patch_size
        self.image_embedding_size: Shape3D = tuple(
            image_axis // patch_axis
            for image_axis, patch_axis in zip(image_size, patch_size, strict=True)
        )  # type: ignore[assignment]
        self.first_upscaled_size: Shape3D = tuple(
            axis * 2 for axis in self.image_embedding_size
        )  # type: ignore[assignment]
        self.low_resolution_mask_size: Shape3D = tuple(
            axis * 4 for axis in self.image_embedding_size
        )  # type: ignore[assignment]
        self.upscaled_channels = transformer_dim // 8
        self.iou_head_depth = iou_head_depth
        self.iou_head_hidden_dim = iou_head_hidden_dim

        # Names and layouts intentionally reproduce the original checkpoint.
        self.iou_token = nn.Embedding(1, transformer_dim)
        self.mask_tokens = nn.Embedding(self.num_mask_tokens, transformer_dim)

        self.output_upscaling = nn.Sequential(
            nn.ConvTranspose3d(
                transformer_dim,
                transformer_dim // 4,
                kernel_size=2,
                stride=2,
            ),
            nn.LayerNorm(
                (
                    transformer_dim // 4,
                    *self.first_upscaled_size,
                ),
                eps=_DEFAULT_LAYER_NORM_EPS,
            ),
            activation(),
            nn.ConvTranspose3d(
                transformer_dim // 4,
                transformer_dim // 8,
                kernel_size=2,
                stride=2,
            ),
            activation(),
        )

        self.output_hypernetworks_mlps = nn.ModuleList(
            MLP(
                transformer_dim,
                transformer_dim,
                transformer_dim // 8,
                3,
            )
            for _ in range(self.num_mask_tokens)
        )
        self.iou_prediction_head = MLP(
            transformer_dim,
            iou_head_hidden_dim,
            self.num_mask_tokens,
            iou_head_depth,
        )
        self.txt_align_upscaled_embedding = nn.Linear(
            transformer_dim,
            transformer_dim // 8,
        )

        if frozen:
            self.requires_grad_(False)

    def forward(
        self,
        image_embeddings: Tensor,
        text_embedding: Tensor | None,
        image_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        dense_prompt_embeddings: Tensor,
        multimask_output: bool,
        *,
        return_structured: bool = False,
    ) -> tuple[Tensor, Tensor] | SegVolMaskDecoderOutput:
        """Predict one mask or the three disambiguation masks.

        This public positional argument order matches the original SegVol
        ``MaskDecoder`` so existing call sites remain straightforward.
        """

        all_masks, all_iou_predictions, upscaled_shape = self.predict_masks(
            image_embeddings=image_embeddings,
            text_embedding=text_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        mask_slice = slice(1, None) if bool(multimask_output) else slice(0, 1)
        selected_masks = all_masks[:, mask_slice, ...]
        selected_iou = all_iou_predictions[:, mask_slice]

        if return_structured:
            return SegVolMaskDecoderOutput(
                masks=selected_masks,
                iou_predictions=selected_iou,
                all_masks=all_masks,
                all_iou_predictions=all_iou_predictions,
                upscaled_embedding_shape=upscaled_shape,
                multimask_output=bool(multimask_output),
            )
        return selected_masks, selected_iou

    def predict_masks(
        self,
        image_embeddings: Tensor,
        text_embedding: Tensor | None,
        image_pe: Tensor,
        sparse_prompt_embeddings: Tensor,
        dense_prompt_embeddings: Tensor,
    ) -> tuple[Tensor, Tensor, tuple[int, int, int, int, int]]:
        """Predict all mask tokens and their quality scores."""

        batch_size = _validate_decoder_inputs(
            module=self,
            image_embeddings=image_embeddings,
            text_embedding=text_embedding,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        output_tokens = torch.cat(
            (self.iou_token.weight, self.mask_tokens.weight),
            dim=0,
        )
        output_tokens = output_tokens.unsqueeze(0).expand(batch_size, -1, -1)
        tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        image_embeddings = _expand_batch_view(
            image_embeddings,
            batch_size=batch_size,
            name="image_embeddings",
        )
        image_pe = _expand_batch_view(
            image_pe,
            batch_size=batch_size,
            name="image_pe",
        )

        source = image_embeddings + dense_prompt_embeddings
        hidden_states, source_tokens = self.transformer(
            source,
            image_pe,
            tokens,
        )

        iou_token_out = hidden_states[:, 0, :]
        mask_tokens_out = hidden_states[:, 1 : 1 + self.num_mask_tokens, :]

        source = (
            source_tokens.transpose(1, 2)
            .reshape(
                batch_size,
                self.transformer_dim,
                *self.image_embedding_size,
            )
            .contiguous()
        )
        upscaled_embedding = self.output_upscaling(source)
        expected_upscaled_shape = (
            batch_size,
            self.upscaled_channels,
            *self.low_resolution_mask_size,
        )
        if tuple(upscaled_embedding.shape) != expected_upscaled_shape:
            raise SegVolMaskDecoderExecutionError(
                "output_upscaling produced an unexpected shape: "
                f"received={tuple(upscaled_embedding.shape)}, "
                f"expected={expected_upscaled_shape}."
            )

        # There are only four mask tokens in the production architecture. This
        # static ModuleList loop preserves checkpoint keys and is unrolled by
        # torch.compile; it does not loop over voxels or batch elements.
        hypernetwork_inputs = torch.stack(
            tuple(
                hypernetwork(mask_tokens_out[:, mask_index, :])
                for mask_index, hypernetwork in enumerate(
                    self.output_hypernetworks_mlps
                )
            ),
            dim=1,
        )

        flattened_upscaled = upscaled_embedding.flatten(start_dim=2)
        masks = torch.bmm(hypernetwork_inputs, flattened_upscaled).reshape(
            batch_size,
            self.num_mask_tokens,
            *self.low_resolution_mask_size,
        )

        if text_embedding is not None:
            text_query = self.txt_align_upscaled_embedding(text_embedding).unsqueeze(1)
            text_similarity = torch.bmm(text_query, flattened_upscaled).reshape(
                batch_size,
                1,
                *self.low_resolution_mask_size,
            )
            # Broadcasting replaces the original materialising repeat over all
            # mask tokens and produces the same result.
            masks = masks + text_similarity

        iou_predictions = self.iou_prediction_head(iou_token_out)
        _validate_decoder_outputs(
            masks=masks,
            iou_predictions=iou_predictions,
            batch_size=batch_size,
            num_mask_tokens=self.num_mask_tokens,
            spatial_size=self.low_resolution_mask_size,
        )
        return masks, iou_predictions, tuple(upscaled_embedding.shape)  # type: ignore[return-value]

    def freeze(self) -> None:
        self.requires_grad_(False)

    def unfreeze(self) -> None:
        self.requires_grad_(True)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def architecture_report(self) -> SegVolMaskDecoderReport:
        return SegVolMaskDecoderReport(
            transformer_dim=self.transformer_dim,
            image_embedding_size=self.image_embedding_size,
            first_upscaled_size=self.first_upscaled_size,
            low_resolution_mask_size=self.low_resolution_mask_size,
            upscaled_channels=self.upscaled_channels,
            num_multimask_outputs=self.num_multimask_outputs,
            num_mask_tokens=self.num_mask_tokens,
            iou_head_depth=self.iou_head_depth,
            iou_head_hidden_dim=self.iou_head_hidden_dim,
            parameter_count=self.parameter_count,
            trainable_parameter_count=self.trainable_parameter_count,
            frozen=self.trainable_parameter_count == 0,
            state_dict_keys=tuple(sorted(self.state_dict().keys())),
        )

    def extra_repr(self) -> str:
        return (
            f"transformer_dim={self.transformer_dim}, "
            f"image_embedding_size={self.image_embedding_size}, "
            f"low_resolution_mask_size={self.low_resolution_mask_size}, "
            f"num_mask_tokens={self.num_mask_tokens}"
        )


def build_segvol_mask_decoder(
    segmentation_config: SegmentationConfig,
    segmentation_vision_config: VisionEncoderConfig,
    optimization_config: OptimizationConfig,
) -> SegVolMaskDecoder:
    """Build the production checkpoint-compatible SegVol mask decoder."""

    if not segmentation_config.enabled:
        raise SegVolMaskDecoderConfigurationError(
            "Cannot build SegVolMaskDecoder when segmentation is disabled."
        )
    if segmentation_config.architecture != "segvol":
        raise SegVolMaskDecoderConfigurationError(
            "Only architecture='segvol' is supported, got "
            f"{segmentation_config.architecture!r}."
        )
    if segmentation_config.prompt_embed_dim != segmentation_vision_config.hidden_size:
        raise SegVolMaskDecoderConfigurationError(
            "SegVol decoder dimension must equal the independent SegVol image "
            "encoder hidden size: "
            f"{segmentation_config.prompt_embed_dim} != "
            f"{segmentation_vision_config.hidden_size}."
        )

    transformer = build_segvol_two_way_transformer(
        segmentation_config=segmentation_config,
        segmentation_vision_config=segmentation_vision_config,
        optimization_config=optimization_config,
        mlp_dim=_DEFAULT_TRANSFORMER_MLP_DIM,
        attention_downsample_rate=_DEFAULT_ATTENTION_DOWNSAMPLE_RATE,
    )
    return SegVolMaskDecoder(
        transformer_dim=segmentation_config.prompt_embed_dim,
        transformer=transformer,
        image_size=_shape3d(
            segmentation_vision_config.image_size,
            name="seg_vision.image_size",
        ),
        patch_size=_shape3d(
            segmentation_vision_config.patch_size,
            name="seg_vision.patch_size",
        ),
        num_multimask_outputs=_DEFAULT_NUM_MULTIMASK_OUTPUTS,
        activation=nn.GELU,
        iou_head_depth=_DEFAULT_IOU_HEAD_DEPTH,
        iou_head_hidden_dim=_DEFAULT_IOU_HEAD_HIDDEN_DIM,
        frozen=segmentation_config.freeze_mask_decoder,
    )


def segvol_mask_decoder_parameter_names(module: nn.Module) -> tuple[str, ...]:
    """Return sorted keys for checkpoint compatibility reports."""

    return tuple(sorted(module.state_dict().keys()))


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SegVolMaskDecoderConfigurationError(
            f"{name} must be a positive integer, got {value!r}."
        )
    return value


def _shape3d(value: Iterable[int], *, name: str) -> Shape3D:
    values = tuple(int(item) for item in value)
    if len(values) != 3:
        raise SegVolMaskDecoderConfigurationError(
            f"{name} must contain exactly three values, got {values}."
        )
    for axis, item in enumerate(values):
        _positive_int(item, name=f"{name}[{axis}]")
    return values  # type: ignore[return-value]


def _expand_batch_view(tensor: Tensor, *, batch_size: int, name: str) -> Tensor:
    if tensor.shape[0] == batch_size:
        return tensor
    if tensor.shape[0] == 1:
        return tensor.expand(batch_size, *tensor.shape[1:])
    raise SegVolMaskDecoderExecutionError(
        f"{name} batch must be 1 or {batch_size}, got {tensor.shape[0]}."
    )


def _validate_decoder_inputs(
    *,
    module: SegVolMaskDecoder,
    image_embeddings: Tensor,
    text_embedding: Tensor | None,
    image_pe: Tensor,
    sparse_prompt_embeddings: Tensor,
    dense_prompt_embeddings: Tensor,
) -> int:
    if image_embeddings.ndim != 5:
        raise SegVolMaskDecoderExecutionError(
            "image_embeddings must be [B,C,D,H,W], got "
            f"{tuple(image_embeddings.shape)}."
        )
    if image_pe.ndim != 5:
        raise SegVolMaskDecoderExecutionError(
            f"image_pe must be [1 or B,C,D,H,W], got {tuple(image_pe.shape)}."
        )
    if sparse_prompt_embeddings.ndim != 3:
        raise SegVolMaskDecoderExecutionError(
            "sparse_prompt_embeddings must be [B,N,C], got "
            f"{tuple(sparse_prompt_embeddings.shape)}."
        )
    if dense_prompt_embeddings.ndim != 5:
        raise SegVolMaskDecoderExecutionError(
            "dense_prompt_embeddings must be [B,C,D,H,W], got "
            f"{tuple(dense_prompt_embeddings.shape)}."
        )

    batch_size = int(sparse_prompt_embeddings.shape[0])
    if batch_size <= 0 or sparse_prompt_embeddings.shape[1] <= 0:
        raise SegVolMaskDecoderExecutionError(
            "sparse_prompt_embeddings must contain a non-empty batch and at "
            "least one prompt token."
        )
    expected_dense_shape = (
        batch_size,
        module.transformer_dim,
        *module.image_embedding_size,
    )
    if tuple(dense_prompt_embeddings.shape) != expected_dense_shape:
        raise SegVolMaskDecoderExecutionError(
            "dense_prompt_embeddings shape mismatch: "
            f"received={tuple(dense_prompt_embeddings.shape)}, "
            f"expected={expected_dense_shape}."
        )
    if sparse_prompt_embeddings.shape[-1] != module.transformer_dim:
        raise SegVolMaskDecoderExecutionError(
            "sparse prompt hidden size mismatch: "
            f"{sparse_prompt_embeddings.shape[-1]} != {module.transformer_dim}."
        )

    expected_image_tail = (
        module.transformer_dim,
        *module.image_embedding_size,
    )
    if tuple(image_embeddings.shape[1:]) != expected_image_tail:
        raise SegVolMaskDecoderExecutionError(
            "image_embeddings channel/spatial shape mismatch: "
            f"received={tuple(image_embeddings.shape[1:])}, "
            f"expected={expected_image_tail}."
        )
    if image_embeddings.shape[0] not in (1, batch_size):
        raise SegVolMaskDecoderExecutionError(
            "image_embeddings batch must be 1 or match prompt batch: "
            f"{image_embeddings.shape[0]} vs {batch_size}."
        )
    if tuple(image_pe.shape[1:]) != expected_image_tail:
        raise SegVolMaskDecoderExecutionError(
            "image_pe channel/spatial shape mismatch: "
            f"received={tuple(image_pe.shape[1:])}, expected={expected_image_tail}."
        )
    if image_pe.shape[0] not in (1, batch_size):
        raise SegVolMaskDecoderExecutionError(
            f"image_pe batch must be 1 or {batch_size}, got {image_pe.shape[0]}."
        )

    if text_embedding is not None:
        if text_embedding.ndim != 2:
            raise SegVolMaskDecoderExecutionError(
                "text_embedding must be [B,C], got "
                f"{tuple(text_embedding.shape)}."
            )
        if tuple(text_embedding.shape) != (batch_size, module.transformer_dim):
            raise SegVolMaskDecoderExecutionError(
                "text_embedding shape mismatch: "
                f"received={tuple(text_embedding.shape)}, "
                f"expected={(batch_size, module.transformer_dim)}."
            )

    tensors = {
        "image_embeddings": image_embeddings,
        "image_pe": image_pe,
        "sparse_prompt_embeddings": sparse_prompt_embeddings,
        "dense_prompt_embeddings": dense_prompt_embeddings,
    }
    if text_embedding is not None:
        tensors["text_embedding"] = text_embedding

    reference_device = sparse_prompt_embeddings.device
    reference_dtype = sparse_prompt_embeddings.dtype
    for name, tensor in tensors.items():
        # Model-boundary activations must be ordinary local tensors. In
        # particular, allowing an FSDP DTensor parameter/view to reach these
        # numerical validations can dispatch a distributed reduction (and
        # older PyTorch releases may crash instead of raising cleanly).
        if type(tensor) is not Tensor:
            raise SegVolMaskDecoderExecutionError(
                f"{name} must be a local Tensor, got {type(tensor).__name__}."
            )
        if not tensor.is_floating_point():
            raise SegVolMaskDecoderExecutionError(
                f"{name} must be floating point, got {tensor.dtype}."
            )
        if tensor.device != reference_device:
            raise SegVolMaskDecoderExecutionError(
                f"{name} is on {tensor.device}, expected {reference_device}."
            )
        if tensor.dtype != reference_dtype:
            raise SegVolMaskDecoderExecutionError(
                f"{name} uses {tensor.dtype}, expected {reference_dtype}."
            )
        # PyTorch 2.6 on macOS/ARM can crash in the native CPU BF16 isfinite
        # kernel after composable-FSDP collectives. Upcast only this validation
        # view; CUDA and the actual model computation remain unchanged.
        try:
            finite_view = (
                tensor.float()
                if tensor.device.type == "cpu" and tensor.dtype is torch.bfloat16
                else tensor
            )
        except RuntimeError as exc:
            raise SegVolMaskDecoderExecutionError(
                f"{name} does not own accessible local tensor storage."
            ) from exc
        if not torch.isfinite(finite_view).all():
            raise SegVolMaskDecoderExecutionError(
                f"{name} contains NaN or Inf values."
            )
    return batch_size


def _validate_decoder_outputs(
    *,
    masks: Tensor,
    iou_predictions: Tensor,
    batch_size: int,
    num_mask_tokens: int,
    spatial_size: Shape3D,
) -> None:
    expected_masks = (batch_size, num_mask_tokens, *spatial_size)
    expected_iou = (batch_size, num_mask_tokens)
    if tuple(masks.shape) != expected_masks:
        raise SegVolMaskDecoderExecutionError(
            f"Mask output shape is {tuple(masks.shape)}, expected {expected_masks}."
        )
    if tuple(iou_predictions.shape) != expected_iou:
        raise SegVolMaskDecoderExecutionError(
            "IoU output shape is "
            f"{tuple(iou_predictions.shape)}, expected {expected_iou}."
        )
    finite_masks = (
        masks.float()
        if masks.device.type == "cpu" and masks.dtype is torch.bfloat16
        else masks
    )
    finite_iou = (
        iou_predictions.float()
        if iou_predictions.device.type == "cpu"
        and iou_predictions.dtype is torch.bfloat16
        else iou_predictions
    )
    if not torch.isfinite(finite_masks).all() or not torch.isfinite(finite_iou).all():
        raise SegVolMaskDecoderExecutionError(
            "Mask decoder produced NaN or Inf values."
        )


# ---------------------------------------------------------------------------
# Legacy-reference helpers used only by the CPU self-test.
# ---------------------------------------------------------------------------


def _legacy_predict_masks(
    decoder: SegVolMaskDecoder,
    image_embeddings: Tensor,
    text_embedding: Tensor | None,
    image_pe: Tensor,
    sparse_prompt_embeddings: Tensor,
    dense_prompt_embeddings: Tensor,
) -> tuple[Tensor, Tensor]:
    output_tokens = torch.cat(
        (decoder.iou_token.weight, decoder.mask_tokens.weight),
        dim=0,
    )
    output_tokens = output_tokens.unsqueeze(0).expand(
        sparse_prompt_embeddings.size(0),
        -1,
        -1,
    )
    tokens = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

    if image_embeddings.shape[0] != tokens.shape[0]:
        source = torch.repeat_interleave(
            image_embeddings,
            tokens.shape[0],
            dim=0,
        )
    else:
        source = image_embeddings
    source = source + dense_prompt_embeddings
    positional_source = torch.repeat_interleave(
        image_pe,
        tokens.shape[0],
        dim=0,
    )

    batch_size, channels, depth, height, width = source.shape
    hidden_states, source_tokens = decoder.transformer(
        source,
        positional_source,
        tokens,
    )
    iou_token_out = hidden_states[:, 0, :]
    mask_tokens_out = hidden_states[:, 1 : 1 + decoder.num_mask_tokens, :]

    source = source_tokens.transpose(1, 2).view(
        batch_size,
        channels,
        depth,
        height,
        width,
    )
    upscaled_embedding = decoder.output_upscaling(source)
    hypernetwork_inputs = []
    for mask_index in range(decoder.num_mask_tokens):
        hypernetwork_inputs.append(
            decoder.output_hypernetworks_mlps[mask_index](
                mask_tokens_out[:, mask_index, :]
            )
        )
    hypernetwork_inputs_tensor = torch.stack(hypernetwork_inputs, dim=1)

    batch_size, channels, depth, height, width = upscaled_embedding.shape
    masks = (
        hypernetwork_inputs_tensor
        @ upscaled_embedding.view(
            batch_size,
            channels,
            depth * height * width,
        )
    ).view(batch_size, -1, depth, height, width)

    if text_embedding is not None:
        text_embedding_down = decoder.txt_align_upscaled_embedding(
            text_embedding
        ).unsqueeze(1)
        upscaled_flat = upscaled_embedding.view(
            batch_size,
            channels,
            depth * height * width,
        )
        similarity = (text_embedding_down @ upscaled_flat).view(
            batch_size,
            -1,
            depth,
            height,
            width,
        )
        similarity = similarity.repeat(1, masks.shape[1], 1, 1, 1)
        masks = masks + similarity

    iou_predictions = decoder.iou_prediction_head(iou_token_out)
    return masks, iou_predictions


def run_cpu_self_test() -> dict[str, object]:
    torch.manual_seed(29)

    transformer = TwoWayTransformer(
        depth=2,
        embedding_dim=32,
        num_heads=4,
        mlp_dim=64,
        attention_downsample_rate=2,
        attention_backend="math",
        require_flash_sdpa=False,
        activation_checkpointing=False,
    )
    decoder = SegVolMaskDecoder(
        transformer_dim=32,
        transformer=transformer,
        image_size=(8, 16, 16),
        patch_size=(4, 8, 8),
        num_multimask_outputs=3,
        activation=nn.GELU,
        iou_head_depth=3,
        iou_head_hidden_dim=16,
    )
    decoder.train()

    batch_size = 2
    image = torch.randn(batch_size, 32, 2, 2, 2, requires_grad=True)
    image_pe = torch.randn(1, 32, 2, 2, 2)
    sparse = torch.randn(batch_size, 1, 32, requires_grad=True)
    dense = torch.randn(batch_size, 32, 2, 2, 2, requires_grad=True)
    text = torch.randn(batch_size, 32, requires_grad=True)

    modern_masks, modern_iou, _ = decoder.predict_masks(
        image,
        text,
        image_pe,
        sparse,
        dense,
    )
    legacy_masks, legacy_iou = _legacy_predict_masks(
        decoder,
        image,
        text,
        image_pe,
        sparse,
        dense,
    )
    torch.testing.assert_close(modern_masks, legacy_masks, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(modern_iou, legacy_iou, rtol=1e-5, atol=1e-6)

    structured = decoder(
        image,
        text,
        image_pe,
        sparse,
        dense,
        False,
        return_structured=True,
    )
    assert isinstance(structured, SegVolMaskDecoderOutput)
    if structured.masks.shape != (2, 1, 8, 8, 8):
        raise AssertionError(
            f"Unexpected single-mask shape: {tuple(structured.masks.shape)}"
        )
    multimask, multi_iou = decoder(
        image,
        text,
        image_pe,
        sparse,
        dense,
        True,
    )
    if multimask.shape != (2, 3, 8, 8, 8) or multi_iou.shape != (2, 3):
        raise AssertionError("Multimask selection produced incorrect shapes.")

    loss = (
        structured.masks.square().mean()
        + structured.iou_predictions.square().mean()
    )
    loss.backward()
    for name, tensor in {
        "image": image,
        "sparse": sparse,
        "dense": dense,
        "text": text,
    }.items():
        if tensor.grad is None or not torch.isfinite(tensor.grad).all():
            raise AssertionError(f"{name} did not receive a finite gradient.")

    no_text_masks, no_text_iou, _ = decoder.predict_masks(
        image.detach(),
        None,
        image_pe,
        sparse.detach(),
        dense.detach(),
    )
    if no_text_masks.shape != modern_masks.shape or no_text_iou.shape != modern_iou.shape:
        raise AssertionError("The no-text branch returned an invalid shape.")

    state_keys = segvol_mask_decoder_parameter_names(decoder)
    required_keys = {
        "iou_token.weight",
        "mask_tokens.weight",
        "output_upscaling.0.weight",
        "output_upscaling.0.bias",
        "output_upscaling.1.weight",
        "output_upscaling.1.bias",
        "output_upscaling.3.weight",
        "output_upscaling.3.bias",
        "output_hypernetworks_mlps.0.layers.0.weight",
        "output_hypernetworks_mlps.3.layers.2.bias",
        "iou_prediction_head.layers.0.weight",
        "iou_prediction_head.layers.2.bias",
        "txt_align_upscaled_embedding.weight",
        "txt_align_upscaled_embedding.bias",
        "transformer.layers.0.self_attn.q_proj.weight",
    }
    missing = sorted(required_keys - set(state_keys))
    if missing:
        raise AssertionError(f"Missing checkpoint-compatible keys: {missing}")

    invalid_shape_detected = False
    try:
        decoder(
            image[:, :, :1],
            text,
            image_pe,
            sparse,
            dense,
            False,
        )
    except SegVolMaskDecoderExecutionError:
        invalid_shape_detected = True
    if not invalid_shape_detected:
        raise AssertionError("Invalid image embedding shape was not rejected.")

    report = decoder.architecture_report()
    return {
        "status": "passed",
        "legacy_numerical_equivalence": True,
        "single_mask_shape": list(structured.masks.shape),
        "multimask_shape": list(multimask.shape),
        "all_mask_shape": list(structured.all_masks.shape),
        "iou_shape": list(structured.all_iou_predictions.shape),
        "upscaled_embedding_shape": list(structured.upscaled_embedding_shape),
        "text_similarity_broadcast_equivalence": True,
        "no_text_branch": True,
        "checkpoint_key_count": len(state_keys),
        "required_checkpoint_keys_present": True,
        "invalid_shape_detected": invalid_shape_detected,
        "report": report.to_dict(),
    }


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the CPU numerical and checkpoint-compatibility test.",
    )
    return parser


def main() -> None:
    args = _build_argument_parser().parse_args()
    if not args.self_test:
        raise SystemExit("Pass --self-test to run this module directly.")
    print(json.dumps(run_cpu_self_test(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
