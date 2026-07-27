"""Complete checkpoint-compatible SegVol module for M3D-Modernized.

This file connects the three SegVol components implemented in the preceding
modules:

* an **independent** 3D ViT image encoder;
* the checkpoint-compatible prompt encoder; and
* the SDPA-based 3D mask decoder.

The public module names deliberately remain ``image_encoder``,
``prompt_encoder`` and ``mask_decoder`` so the published SegVol checkpoint can
be loaded strictly by :func:`m3d.model.checkpoint.load_segmentation_module_checkpoint`.

M3D's normal segmentation path is::

    image [B, 1, D, H, W]
      -> SegVol image_encoder (the second, independent 3D ViT)
      -> image features [B, C, Dp, Hp, Wp]
      -> projected Phi-3 [SEG] prompt [B, C]
      -> prompt_encoder
      -> mask_decoder
      -> low-resolution logits
      -> trilinear resize to [B, 1, D, H, W]

No segmentation target is inspected here.  A genuine all-zero target remains a
normal segmentation sample; task routing is performed explicitly by the outer
multimodal model and data batch metadata.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from dataclasses import dataclass
from typing import Any, Final, TypeAlias, overload

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from m3d.config import OptimizationConfig, SegmentationConfig, VisionEncoderConfig
from m3d.model.segvol_mask_decoder import (
    SegVolMaskDecoder,
    SegVolMaskDecoderOutput,
    build_segvol_mask_decoder,
)
from m3d.model.segvol_prompt_encoder import (
    PointPrompt,
    SegVolPromptEncoder,
    SegVolPromptOutput,
    build_segvol_prompt_encoder,
)
from m3d.model.vit3d import (
    ViT3DEncoder,
    VisionEncoderOutput,
    build_segmentation_vision_encoder,
)


Shape3D: TypeAlias = tuple[int, int, int]

_INTERPOLATION_MODE: Final[str] = "trilinear"
_INTERPOLATION_ALIGN_CORNERS: Final[bool] = False


class SegVolConfigurationError(ValueError):
    """Raised when SegVol components have incompatible static geometry."""


class SegVolExecutionError(RuntimeError):
    """Raised when runtime tensors violate the SegVol execution contract."""


@dataclass(frozen=True, slots=True)
class SegVolOutput:
    """Structured output from :class:`SegVol`.

    ``logits`` are resized to the requested output volume.  They are raw logits
    and must *not* be thresholded before BCE-with-logits or Dice loss.
    ``low_resolution_logits`` are the selected mask-token outputs before the
    final trilinear resize.
    """

    logits: Tensor
    low_resolution_logits: Tensor
    iou_predictions: Tensor
    output_spatial_size: Shape3D
    image_embedding_shape: tuple[int, int, int, int, int]
    sparse_prompt_count: int
    multimask_output: bool

    def __post_init__(self) -> None:
        if self.logits.ndim != 5:
            raise SegVolExecutionError(
                f"logits must be [B,M,D,H,W], got {tuple(self.logits.shape)}."
            )
        if self.low_resolution_logits.ndim != 5:
            raise SegVolExecutionError(
                "low_resolution_logits must be [B,M,D,H,W], got "
                f"{tuple(self.low_resolution_logits.shape)}."
            )
        if self.iou_predictions.ndim != 2:
            raise SegVolExecutionError(
                "iou_predictions must be [B,M], got "
                f"{tuple(self.iou_predictions.shape)}."
            )
        if self.logits.shape[:2] != self.iou_predictions.shape:
            raise SegVolExecutionError(
                "High-resolution mask and IoU shapes disagree: "
                f"{tuple(self.logits.shape[:2])} vs "
                f"{tuple(self.iou_predictions.shape)}."
            )
        if self.low_resolution_logits.shape[:2] != self.iou_predictions.shape:
            raise SegVolExecutionError(
                "Low-resolution mask and IoU shapes disagree: "
                f"{tuple(self.low_resolution_logits.shape[:2])} vs "
                f"{tuple(self.iou_predictions.shape)}."
            )
        if tuple(self.logits.shape[-3:]) != self.output_spatial_size:
            raise SegVolExecutionError(
                "Output logits do not match output_spatial_size: "
                f"{tuple(self.logits.shape[-3:])} vs {self.output_spatial_size}."
            )
        if len(self.image_embedding_shape) != 5:
            raise SegVolExecutionError(
                "image_embedding_shape must describe [B,C,D,H,W]."
            )
        if self.sparse_prompt_count <= 0:
            raise SegVolExecutionError(
                "SegVol requires at least one sparse text/point/box prompt."
            )

    @property
    def batch_size(self) -> int:
        return int(self.logits.shape[0])

    @property
    def mask_count(self) -> int:
        return int(self.logits.shape[1])

    def probabilities(self) -> Tensor:
        """Return sigmoid probabilities without changing stored logits."""

        return torch.sigmoid(self.logits)

    def binary_masks(self, threshold: float = 0.5) -> Tensor:
        """Return float binary masks using a probability threshold."""

        if not 0.0 <= float(threshold) <= 1.0:
            raise ValueError(f"threshold must be in [0,1], got {threshold}.")
        return (self.probabilities() > float(threshold)).to(self.logits.dtype)


@dataclass(frozen=True, slots=True)
class SegVolArchitectureReport:
    """Static architecture and trainability summary."""

    image_size: Shape3D
    patch_size: Shape3D
    feature_shape: Shape3D
    low_resolution_mask_size: Shape3D
    prompt_embed_dim: int
    image_encoder_parameters: int
    prompt_encoder_parameters: int
    mask_decoder_parameters: int
    total_parameters: int
    trainable_parameters: int
    state_dict_key_count: int
    required_checkpoint_prefixes_present: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class SegVol(nn.Module):
    """Complete text-prompted 3D SegVol module used by M3D.

    The second 3D image encoder is owned by this module as ``image_encoder``.
    It is not the Main 3D ViT and shares no parameters with it.
    """

    def __init__(
        self,
        *,
        image_encoder: ViT3DEncoder,
        mask_decoder: SegVolMaskDecoder,
        prompt_encoder: SegVolPromptEncoder,
        roi_size: Shape3D,
        patch_size: Shape3D,
    ) -> None:
        super().__init__()
        if not isinstance(image_encoder, ViT3DEncoder):
            raise SegVolConfigurationError(
                "image_encoder must be a ViT3DEncoder instance."
            )
        if not isinstance(mask_decoder, SegVolMaskDecoder):
            raise SegVolConfigurationError(
                "mask_decoder must be a SegVolMaskDecoder instance."
            )
        if not isinstance(prompt_encoder, SegVolPromptEncoder):
            raise SegVolConfigurationError(
                "prompt_encoder must be a SegVolPromptEncoder instance."
            )

        self.roi_size = _shape3d(roi_size, name="roi_size")
        self.patch_size = _shape3d(patch_size, name="patch_size")
        for axis, (roi_axis, patch_axis) in enumerate(
            zip(self.roi_size, self.patch_size, strict=True)
        ):
            if roi_axis % patch_axis != 0:
                raise SegVolConfigurationError(
                    "roi_size must be divisible by patch_size on every axis: "
                    f"axis={axis}, roi={roi_axis}, patch={patch_axis}."
                )

        self.feat_shape: Shape3D = tuple(
            roi_axis // patch_axis
            for roi_axis, patch_axis in zip(
                self.roi_size,
                self.patch_size,
                strict=True,
            )
        )  # type: ignore[assignment]

        # These names intentionally reproduce the original SegVol state dict.
        self.image_encoder = image_encoder
        self.mask_decoder = mask_decoder
        self.prompt_encoder = prompt_encoder

        self._validate_component_contracts()

    @property
    def device(self) -> torch.device:
        return self.image_encoder.device

    @property
    def dtype(self) -> torch.dtype:
        return self.image_encoder.dtype

    @property
    def low_resolution_mask_size(self) -> Shape3D:
        return self.mask_decoder.low_resolution_mask_size

    def _validate_component_contracts(self) -> None:
        errors: list[str] = []
        encoder_shape = self.image_encoder.shape

        if encoder_shape.has_cls_token:
            errors.append(
                "SegVol image_encoder must not use a CLS token; "
                "set seg_vision.use_cls_token=false"
            )
        if encoder_shape.image_size != self.roi_size:
            errors.append(
                "image_encoder image_size differs from SegVol roi_size: "
                f"{encoder_shape.image_size} != {self.roi_size}"
            )
        if encoder_shape.patch_size != self.patch_size:
            errors.append(
                "image_encoder patch_size differs from SegVol patch_size: "
                f"{encoder_shape.patch_size} != {self.patch_size}"
            )
        if encoder_shape.patch_grid != self.feat_shape:
            errors.append(
                "image_encoder patch_grid differs from SegVol feature shape: "
                f"{encoder_shape.patch_grid} != {self.feat_shape}"
            )
        if self.prompt_encoder.embed_dim != encoder_shape.hidden_size:
            errors.append(
                "prompt embedding dimension differs from image encoder hidden size: "
                f"{self.prompt_encoder.embed_dim} != {encoder_shape.hidden_size}"
            )
        if self.prompt_encoder.image_embedding_size != self.feat_shape:
            errors.append(
                "prompt encoder feature shape differs from SegVol feature shape: "
                f"{self.prompt_encoder.image_embedding_size} != {self.feat_shape}"
            )
        if self.prompt_encoder.input_image_size != self.roi_size:
            errors.append(
                "prompt encoder input image size differs from roi_size: "
                f"{self.prompt_encoder.input_image_size} != {self.roi_size}"
            )
        if self.mask_decoder.transformer_dim != encoder_shape.hidden_size:
            errors.append(
                "mask decoder dimension differs from image encoder hidden size: "
                f"{self.mask_decoder.transformer_dim} != "
                f"{encoder_shape.hidden_size}"
            )
        if self.mask_decoder.image_embedding_size != self.feat_shape:
            errors.append(
                "mask decoder feature shape differs from SegVol feature shape: "
                f"{self.mask_decoder.image_embedding_size} != {self.feat_shape}"
            )
        if errors:
            raise SegVolConfigurationError(
                "Incompatible SegVol components:\n  - " + "\n  - ".join(errors)
            )

    def _validate_image(self, image: Tensor) -> None:
        if image.ndim != 5:
            raise SegVolExecutionError(
                f"image must be [B,C,D,H,W], got {tuple(image.shape)}."
            )
        if image.shape[0] <= 0:
            raise SegVolExecutionError("image batch size must be positive.")
        expected = (
            self.image_encoder.shape.image_channels,
            *self.roi_size,
        )
        if tuple(image.shape[1:]) != expected:
            raise SegVolExecutionError(
                "SegVol image shape differs from its fixed pretrained geometry: "
                f"received={tuple(image.shape[1:])}, expected={expected}."
            )
        if not torch.is_floating_point(image):
            raise SegVolExecutionError(
                f"image must be floating point, got {image.dtype}."
            )
        if image.device != self.device:
            raise SegVolExecutionError(
                f"image is on {image.device}, but SegVol is on {self.device}."
            )

    def _validate_image_embeddings(self, image_embeddings: Tensor) -> None:
        expected_tail = (
            self.image_encoder.hidden_size,
            *self.feat_shape,
        )
        if image_embeddings.ndim != 5:
            raise SegVolExecutionError(
                "image_embeddings must be [B,C,D,H,W], got "
                f"{tuple(image_embeddings.shape)}."
            )
        if image_embeddings.shape[0] <= 0:
            raise SegVolExecutionError(
                "image_embeddings batch size must be positive."
            )
        if tuple(image_embeddings.shape[1:]) != expected_tail:
            raise SegVolExecutionError(
                "image_embeddings do not match SegVol geometry: "
                f"received={tuple(image_embeddings.shape[1:])}, "
                f"expected={expected_tail}."
            )
        if not torch.is_floating_point(image_embeddings):
            raise SegVolExecutionError(
                "image_embeddings must be floating point, got "
                f"{image_embeddings.dtype}."
            )
        if image_embeddings.device != self.device:
            raise SegVolExecutionError(
                f"image_embeddings are on {image_embeddings.device}, but "
                f"SegVol is on {self.device}."
            )

    def encode_image(self, image: Tensor) -> Tensor:
        """Run only the independent SegVol image encoder.

        The returned dense feature map can be reused for several prompt
        decodes during inference without re-running the 3D ViT.
        """

        self._validate_image(image)
        output = self.image_encoder(
            image,
            output_hidden_states=False,
            return_dict=True,
        )
        if not isinstance(output, VisionEncoderOutput):
            raise SegVolExecutionError(
                "SegVol image encoder returned an unexpected output type."
            )
        image_embeddings = output.spatial_features()
        self._validate_image_embeddings(image_embeddings)
        return image_embeddings

    @overload
    def decode_embeddings(
        self,
        image_embeddings: Tensor,
        *,
        output_spatial_size: Shape3D,
        text_embedding: Tensor | None,
        boxes: Tensor | None = None,
        points: PointPrompt | None = None,
        multimask_output: bool = False,
        return_structured: bool = False,
    ) -> Tensor: ...

    @overload
    def decode_embeddings(
        self,
        image_embeddings: Tensor,
        *,
        output_spatial_size: Shape3D,
        text_embedding: Tensor | None,
        boxes: Tensor | None = None,
        points: PointPrompt | None = None,
        multimask_output: bool = False,
        return_structured: bool,
    ) -> Tensor | SegVolOutput: ...

    def decode_embeddings(
        self,
        image_embeddings: Tensor,
        *,
        output_spatial_size: Shape3D,
        text_embedding: Tensor | None,
        boxes: Tensor | None = None,
        points: PointPrompt | None = None,
        multimask_output: bool = False,
        return_structured: bool = False,
    ) -> Tensor | SegVolOutput:
        """Decode cached image features with text/point/box prompts."""

        self._validate_image_embeddings(image_embeddings)
        output_spatial_size = _shape3d(
            output_spatial_size,
            name="output_spatial_size",
        )
        if text_embedding is None and boxes is None and points is None:
            raise SegVolExecutionError(
                "At least one text, box or point prompt is required. "
                "M3D normally supplies the projected Phi-3 [SEG] embedding."
            )

        prepared_text = self._prepare_text_embedding(text_embedding)
        prompt_output = self.prompt_encoder(
            points=points,
            boxes=boxes,
            masks=None,
            text_embedding=prepared_text,
            return_structured=True,
        )
        if not isinstance(prompt_output, SegVolPromptOutput):
            raise SegVolExecutionError(
                "SegVol prompt encoder returned an unexpected output type."
            )

        image_batch = int(image_embeddings.shape[0])
        prompt_batch = prompt_output.batch_size
        if image_batch not in {1, prompt_batch}:
            raise SegVolExecutionError(
                "Image/prompt batch mismatch. The image batch must either equal "
                "the prompt batch or be one for multi-prompt inference: "
                f"image_batch={image_batch}, prompt_batch={prompt_batch}."
            )

        decoder_output = self.mask_decoder(
            image_embeddings=image_embeddings,
            text_embedding=prepared_text,
            image_pe=prompt_output.dense_positional_encoding,
            sparse_prompt_embeddings=prompt_output.sparse_embeddings,
            dense_prompt_embeddings=prompt_output.dense_embeddings,
            multimask_output=bool(multimask_output),
            return_structured=True,
        )
        if not isinstance(decoder_output, SegVolMaskDecoderOutput):
            raise SegVolExecutionError(
                "SegVol mask decoder returned an unexpected output type."
            )

        logits = F.interpolate(
            decoder_output.masks,
            size=output_spatial_size,
            mode=_INTERPOLATION_MODE,
            align_corners=_INTERPOLATION_ALIGN_CORNERS,
        )
        expected_masks = 3 if bool(multimask_output) else 1
        expected_shape = (
            prompt_batch,
            expected_masks,
            *output_spatial_size,
        )
        if tuple(logits.shape) != expected_shape:
            raise SegVolExecutionError(
                "Final interpolation produced an unexpected mask shape: "
                f"received={tuple(logits.shape)}, expected={expected_shape}."
            )

        if not return_structured:
            return logits
        return SegVolOutput(
            logits=logits,
            low_resolution_logits=decoder_output.masks,
            iou_predictions=decoder_output.iou_predictions,
            output_spatial_size=output_spatial_size,
            image_embedding_shape=tuple(image_embeddings.shape),  # type: ignore[arg-type]
            sparse_prompt_count=prompt_output.prompt_count,
            multimask_output=bool(multimask_output),
        )

    def _prepare_text_embedding(
        self,
        text_embedding: Tensor | None,
    ) -> Tensor | None:
        if text_embedding is None:
            return None
        if text_embedding.ndim not in {2, 3}:
            raise SegVolExecutionError(
                "text_embedding must be [B,C] or [B,N,C], got "
                f"{tuple(text_embedding.shape)}."
            )
        if text_embedding.shape[-1] != self.prompt_encoder.embed_dim:
            raise SegVolExecutionError(
                "text_embedding dimension does not match SegVol prompt dim: "
                f"{text_embedding.shape[-1]} != {self.prompt_encoder.embed_dim}."
            )
        if not torch.is_floating_point(text_embedding):
            raise SegVolExecutionError(
                "text_embedding must be floating point, got "
                f"{text_embedding.dtype}."
            )
        if text_embedding.device != self.device:
            raise SegVolExecutionError(
                f"text_embedding is on {text_embedding.device}, but SegVol is "
                f"on {self.device}."
            )
        # The language model and SegVol normally run under the same BF16
        # autocast. This explicit dtype alignment also makes eager inference
        # safe when the caller supplies FP32 language features to a BF16 module.
        return text_embedding.to(dtype=self.prompt_encoder.dtype)

    @overload
    def forward(
        self,
        image: Tensor,
        text_embedding: Tensor | None = None,
        *,
        text_emb: Tensor | None = None,
        text: Any | None = None,
        boxes: Tensor | None = None,
        points: PointPrompt | None = None,
        multimask_output: bool = False,
        return_structured: bool = False,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        image: Tensor,
        text_embedding: Tensor | None = None,
        *,
        text_emb: Tensor | None = None,
        text: Any | None = None,
        boxes: Tensor | None = None,
        points: PointPrompt | None = None,
        multimask_output: bool = False,
        return_structured: bool,
    ) -> Tensor | SegVolOutput: ...

    def forward(
        self,
        image: Tensor,
        text_embedding: Tensor | None = None,
        *,
        text_emb: Tensor | None = None,
        text: Any | None = None,
        boxes: Tensor | None = None,
        points: PointPrompt | None = None,
        multimask_output: bool = False,
        return_structured: bool = False,
    ) -> Tensor | SegVolOutput:
        """Encode one volume and predict raw segmentation logits.

        ``text_emb`` is retained as a legacy alias for the original M3D call
        site.  Raw strings are intentionally not encoded here because the
        modern M3D graph obtains prompts from Phi-3's ``[SEG]`` hidden state.
        """

        resolved_text = _resolve_text_embedding_alias(
            text_embedding=text_embedding,
            text_emb=text_emb,
        )
        if text is not None and resolved_text is None:
            raise SegVolExecutionError(
                "SegVol does not own a text encoder. Supply the projected Phi-3 "
                "[SEG] embedding through text_embedding/text_emb."
            )

        image_embeddings = self.encode_image(image)
        return self.decode_embeddings(
            image_embeddings,
            output_spatial_size=tuple(image.shape[-3:]),  # type: ignore[arg-type]
            text_embedding=resolved_text,
            boxes=boxes,
            points=points,
            multimask_output=multimask_output,
            return_structured=return_structured,
        )

    def forward_decoder(
        self,
        image_embedding: Tensor,
        img_shape: Shape3D,
        text_emb: Tensor | None = None,
        text: Any | None = None,
        boxes: Tensor | None = None,
        points: PointPrompt | None = None,
        *,
        multimask_output: bool = False,
        return_structured: bool = False,
    ) -> Tensor | SegVolOutput:
        """Legacy-compatible decoder entry point from the original SegVol."""

        if text is not None and text_emb is None:
            raise SegVolExecutionError(
                "Raw text cannot be encoded by SegVol; supply text_emb."
            )
        return self.decode_embeddings(
            image_embedding,
            output_spatial_size=img_shape,
            text_embedding=text_emb,
            boxes=boxes,
            points=points,
            multimask_output=multimask_output,
            return_structured=return_structured,
        )

    def set_image_encoder_checkpointing(self, every_n_layers: int) -> None:
        self.image_encoder.set_activation_checkpointing(every_n_layers)

    def freeze_image_encoder(self) -> None:
        self.image_encoder.requires_grad_(False)

    def unfreeze_image_encoder(self) -> None:
        self.image_encoder.requires_grad_(True)

    def freeze_prompt_encoder(self) -> None:
        self.prompt_encoder.freeze()

    def unfreeze_prompt_encoder(self) -> None:
        self.prompt_encoder.unfreeze()

    def freeze_mask_decoder(self) -> None:
        self.mask_decoder.freeze()

    def unfreeze_mask_decoder(self) -> None:
        self.mask_decoder.unfreeze()

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

    def architecture_report(self) -> SegVolArchitectureReport:
        keys = tuple(sorted(self.state_dict().keys()))
        required_prefixes = (
            "image_encoder.",
            "prompt_encoder.",
            "mask_decoder.",
        )
        return SegVolArchitectureReport(
            image_size=self.roi_size,
            patch_size=self.patch_size,
            feature_shape=self.feat_shape,
            low_resolution_mask_size=self.low_resolution_mask_size,
            prompt_embed_dim=self.prompt_encoder.embed_dim,
            image_encoder_parameters=sum(
                parameter.numel()
                for parameter in self.image_encoder.parameters()
            ),
            prompt_encoder_parameters=sum(
                parameter.numel()
                for parameter in self.prompt_encoder.parameters()
            ),
            mask_decoder_parameters=sum(
                parameter.numel()
                for parameter in self.mask_decoder.parameters()
            ),
            total_parameters=self.parameter_count,
            trainable_parameters=self.trainable_parameter_count,
            state_dict_key_count=len(keys),
            required_checkpoint_prefixes_present=all(
                any(key.startswith(prefix) for key in keys)
                for prefix in required_prefixes
            ),
        )

    def extra_repr(self) -> str:
        return (
            f"roi_size={self.roi_size}, patch_size={self.patch_size}, "
            f"feat_shape={self.feat_shape}, "
            f"low_res_mask={self.low_resolution_mask_size}"
        )


def build_segvol_module(
    *,
    segmentation_config: SegmentationConfig,
    segmentation_vision_config: VisionEncoderConfig,
    optimization_config: OptimizationConfig,
) -> SegVol:
    """Build the complete production SegVol branch from project config."""

    if not segmentation_config.enabled:
        raise SegVolConfigurationError(
            "Cannot build SegVol when segmentation.enabled=false."
        )
    if segmentation_config.architecture != "segvol":
        raise SegVolConfigurationError(
            "Only segmentation architecture='segvol' is supported, got "
            f"{segmentation_config.architecture!r}."
        )
    if not segmentation_vision_config.enabled:
        raise SegVolConfigurationError(
            "SegVol requires the independent seg_vision encoder."
        )
    if segmentation_vision_config.use_cls_token:
        raise SegVolConfigurationError(
            "seg_vision.use_cls_token must be false for dense SegVol features."
        )

    image_encoder = build_segmentation_vision_encoder(
        segmentation_vision_config
    )
    if optimization_config.checkpoint_seg_vision:
        interval = int(
            segmentation_vision_config.activation_checkpoint_every_n_layers
        )
        image_encoder.set_activation_checkpointing(interval or 1)
    else:
        image_encoder.set_activation_checkpointing(0)

    prompt_encoder = build_segvol_prompt_encoder(
        segmentation_config=segmentation_config,
        segmentation_vision_config=segmentation_vision_config,
    )
    mask_decoder = build_segvol_mask_decoder(
        segmentation_config=segmentation_config,
        segmentation_vision_config=segmentation_vision_config,
        optimization_config=optimization_config,
    )

    return SegVol(
        image_encoder=image_encoder,
        mask_decoder=mask_decoder,
        prompt_encoder=prompt_encoder,
        roi_size=_shape3d(
            segmentation_vision_config.image_size,
            name="seg_vision.image_size",
        ),
        patch_size=_shape3d(
            segmentation_vision_config.patch_size,
            name="seg_vision.patch_size",
        ),
    )


def _resolve_text_embedding_alias(
    *,
    text_embedding: Tensor | None,
    text_emb: Tensor | None,
) -> Tensor | None:
    if text_embedding is not None and text_emb is not None:
        if text_embedding is not text_emb:
            raise SegVolExecutionError(
                "Provide only one of text_embedding or legacy text_emb."
            )
    return text_embedding if text_embedding is not None else text_emb


def _shape3d(value: Any, *, name: str) -> Shape3D:
    try:
        items = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise SegVolConfigurationError(
            f"{name} must contain exactly three positive integers."
        ) from exc
    if len(items) != 3 or any(item <= 0 for item in items):
        raise SegVolConfigurationError(
            f"{name} must contain exactly three positive integers, got {items}."
        )
    return items  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


def _tiny_configs() -> tuple[
    SegmentationConfig,
    VisionEncoderConfig,
    OptimizationConfig,
]:
    segmentation = SegmentationConfig(
        enabled=True,
        architecture="segvol",
        prompt_embed_dim=32,
        decoder_depth=2,
        decoder_heads=4,
        freeze_prompt_encoder=False,
        freeze_mask_decoder=False,
    )
    vision = VisionEncoderConfig(
        enabled=True,
        architecture="vit3d",
        image_channels=1,
        image_size=(8, 16, 16),
        patch_size=(4, 8, 8),
        hidden_size=32,
        depth=2,
        num_heads=4,
        mlp_dim=64,
        dropout=0.0,
        qkv_bias=False,
        use_cls_token=False,
        attention_backend="math",
        require_flash_sdpa=False,
        activation_checkpoint_every_n_layers=1,
        freeze=False,
        unfreeze_last_n_layers=0,
    )
    optimization = OptimizationConfig(
        checkpoint_seg_vision=True,
        checkpoint_segmentation_decoder=True,
    )
    return segmentation, vision, optimization


def _legacy_reference(
    module: SegVol,
    image: Tensor,
    text_embedding: Tensor,
) -> Tensor:
    """Reference the original segvol.py execution order."""

    encoder_output = module.image_encoder(
        image,
        output_hidden_states=True,
        return_dict=False,
    )
    if not isinstance(encoder_output, tuple):
        raise AssertionError("Expected legacy ViT tuple output.")
    image_tokens, _ = encoder_output
    batch_size = image.shape[0]
    image_embedding = image_tokens.transpose(1, 2).reshape(
        batch_size,
        -1,
        *module.feat_shape,
    )
    sparse, dense = module.prompt_encoder(
        points=None,
        boxes=None,
        masks=None,
        text_embedding=text_embedding,
    )
    dense_pe = module.prompt_encoder.get_dense_pe()
    low_res, _ = module.mask_decoder(
        image_embeddings=image_embedding,
        text_embedding=text_embedding,
        image_pe=dense_pe,
        sparse_prompt_embeddings=sparse,
        dense_prompt_embeddings=dense,
        multimask_output=False,
    )
    return F.interpolate(
        low_res,
        size=tuple(image.shape[-3:]),
        mode="trilinear",
        align_corners=False,
    )


def run_self_test() -> dict[str, Any]:
    torch.manual_seed(17)
    segmentation, vision, optimization = _tiny_configs()
    module = build_segvol_module(
        segmentation_config=segmentation,
        segmentation_vision_config=vision,
        optimization_config=optimization,
    )
    module.train()

    images = torch.randn(2, 1, 8, 16, 16, requires_grad=True)
    text_embeddings = torch.randn(2, 32, requires_grad=True)

    structured = module(
        images,
        text_embedding=text_embeddings,
        return_structured=True,
    )
    if not isinstance(structured, SegVolOutput):
        raise AssertionError("Structured SegVol output was not returned.")
    if tuple(structured.logits.shape) != (2, 1, 8, 16, 16):
        raise AssertionError(tuple(structured.logits.shape))
    if tuple(structured.low_resolution_logits.shape) != (2, 1, 8, 8, 8):
        raise AssertionError(tuple(structured.low_resolution_logits.shape))

    loss = structured.logits.square().mean() + structured.iou_predictions.square().mean()
    loss.backward()
    if images.grad is None or not bool(torch.isfinite(images.grad).all()):
        raise AssertionError("Image gradient is missing/non-finite.")
    if text_embeddings.grad is None or not bool(
        torch.isfinite(text_embeddings.grad).all()
    ):
        raise AssertionError("Text gradient is missing/non-finite.")

    module.eval()
    with torch.no_grad():
        modern = module(images.detach(), text_embedding=text_embeddings.detach())
        legacy = _legacy_reference(
            module,
            images.detach(),
            text_embeddings.detach(),
        )
        torch.testing.assert_close(modern, legacy, rtol=0.0, atol=0.0)

        cached_embeddings = module.encode_image(images.detach()[:1])
        repeated_text = torch.randn(3, 32)
        multi_prompt = module.decode_embeddings(
            cached_embeddings,
            output_spatial_size=(8, 16, 16),
            text_embedding=repeated_text,
            return_structured=True,
        )
        if not isinstance(multi_prompt, SegVolOutput):
            raise AssertionError("Expected structured cached decode output.")
        if tuple(multi_prompt.logits.shape) != (3, 1, 8, 16, 16):
            raise AssertionError(tuple(multi_prompt.logits.shape))

        multimask = module(
            images.detach(),
            text_embedding=text_embeddings.detach(),
            multimask_output=True,
            return_structured=True,
        )
        if not isinstance(multimask, SegVolOutput):
            raise AssertionError("Expected structured multimask output.")
        if tuple(multimask.logits.shape) != (2, 3, 8, 16, 16):
            raise AssertionError(tuple(multimask.logits.shape))

    alias_error = False
    try:
        module(
            images.detach(),
            text_embedding=text_embeddings.detach(),
            text_emb=text_embeddings.detach().clone(),
        )
    except SegVolExecutionError:
        alias_error = True
    if not alias_error:
        raise AssertionError("Conflicting text aliases were not rejected.")

    dimension_error = False
    try:
        module(images.detach(), text_embedding=torch.randn(2, 31))
    except SegVolExecutionError:
        dimension_error = True
    if not dimension_error:
        raise AssertionError("Wrong text embedding dimension was not rejected.")

    report = module.architecture_report()
    state_keys = tuple(module.state_dict().keys())
    required_examples = (
        "image_encoder.patch_embedding.position_embeddings",
        "prompt_encoder.no_mask_embed.weight",
        "mask_decoder.iou_token.weight",
    )
    if not all(key in state_keys for key in required_examples):
        raise AssertionError("Required SegVol checkpoint keys are missing.")

    return {
        "status": "passed",
        "legacy_numerical_equivalence": True,
        "single_mask_shape": list(structured.logits.shape),
        "low_resolution_shape": list(structured.low_resolution_logits.shape),
        "multimask_shape": list(multimask.logits.shape),
        "cached_multi_prompt_shape": list(multi_prompt.logits.shape),
        "feature_shape": list(module.feat_shape),
        "checkpoint_key_count": report.state_dict_key_count,
        "required_checkpoint_prefixes_present": (
            report.required_checkpoint_prefixes_present
        ),
        "conflicting_alias_detected": alias_error,
        "wrong_prompt_dimension_detected": dimension_error,
        "trainable_parameter_count": report.trainable_parameters,
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a small CPU forward/backward and compatibility test.",
    )
    args = parser.parse_args()
    if not args.self_test:
        parser.print_help()
        return
    print(json.dumps(run_self_test(), indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
