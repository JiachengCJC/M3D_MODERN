"""Multimodal projector for M3D.

This module reproduces the original M3D spatial-pooling projector while using
plain PyTorch operations and explicit shape contracts.  It connects the **Main
3D ViT** to the language model.  It does not consume, merge, or share parameters
with the independent SegVol image encoder.

The legacy M3D projector performs these operations::

    [B, 2048, 768]
        -> reshape [B, 768, 8, 16, 16]
        -> AvgPool3d(kernel=2, stride=2)
        -> reshape [B, 256, 768]
        -> MLP(768 -> language_hidden_size -> ...)

Important compatibility details
-------------------------------
* The trainable submodule is named ``projector`` so legacy checkpoint keys such
  as ``model.mm_projector.projector.0.weight`` can be mapped to
  ``projector.0.weight`` by :mod:`m3d.model.checkpoint`.
* The Main ViT CLS token is removed before pooling.
* The default MLP layout exactly matches the original implementation:
  ``Linear`` followed by repeated ``GELU, Linear`` pairs.  No extra LayerNorm,
  bias removal, residual branch, or dropout is inserted.
* Both legacy pooling modes are supported.  ``spatial`` is the recommended M3D
  mode; ``sequence`` is retained for checkpoint/configuration compatibility.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Literal, Mapping, Sequence, TypeAlias, cast

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from m3d.config import ProjectorConfig, VisionEncoderConfig
from m3d.model.vit3d import VisionEncoderOutput


Shape3D: TypeAlias = tuple[int, int, int]
PoolingType: TypeAlias = Literal["spatial", "sequence"]
LayerType: TypeAlias = Literal["linear", "mlp"]

_DEFAULT_POOLING_SIZE: Final[int] = 2


class ProjectorError(RuntimeError):
    """Base exception for projector construction or execution failures."""


class ProjectorConfigurationError(ProjectorError, ValueError):
    """Raised when projector geometry or architecture is invalid."""


class ProjectorInputError(ProjectorError, ValueError):
    """Raised when input visual tokens violate the projector contract."""


@dataclass(frozen=True, slots=True)
class ProjectorGeometry:
    """Static geometry before and after pooling."""

    input_grid: Shape3D
    output_grid: Shape3D
    input_token_count: int
    output_token_count: int
    pooling_size: int
    pooling_type: PoolingType

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_grid": list(self.input_grid),
            "output_grid": list(self.output_grid),
            "input_token_count": self.input_token_count,
            "output_token_count": self.output_token_count,
            "pooling_size": self.pooling_size,
            "pooling_type": self.pooling_type,
        }


@dataclass(frozen=True, slots=True)
class ProjectorOutput:
    """Structured visual-token output passed to the language model."""

    projected_tokens: Tensor
    input_patch_grid: Shape3D
    pooled_patch_grid: Shape3D

    @property
    def batch_size(self) -> int:
        return int(self.projected_tokens.shape[0])

    @property
    def token_count(self) -> int:
        return int(self.projected_tokens.shape[1])

    @property
    def hidden_size(self) -> int:
        return int(self.projected_tokens.shape[2])


@dataclass(frozen=True, slots=True)
class ProjectorBuildReport:
    """Serializable description of a constructed projector."""

    layer_type: LayerType
    num_layers: int
    input_hidden_size: int
    output_hidden_size: int
    geometry: ProjectorGeometry
    parameter_count: int
    trainable_parameter_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "layer_type": self.layer_type,
            "num_layers": self.num_layers,
            "input_hidden_size": self.input_hidden_size,
            "output_hidden_size": self.output_hidden_size,
            "geometry": self.geometry.as_dict(),
            "parameter_count": self.parameter_count,
            "trainable_parameter_count": self.trainable_parameter_count,
        }


def _shape3(value: Sequence[int], *, label: str) -> Shape3D:
    if len(value) != 3:
        raise ProjectorConfigurationError(
            f"{label} must contain exactly three integers; received {value!r}."
        )
    result = cast(Shape3D, tuple(int(item) for item in value))
    if any(item <= 0 for item in result):
        raise ProjectorConfigurationError(
            f"{label} values must all be positive; received {result}."
        )
    return result


def compute_patch_grid(
    image_size: Sequence[int],
    patch_size: Sequence[int],
) -> Shape3D:
    """Compute ``(D, H, W)`` patch grid with exact divisibility checks."""

    image = _shape3(image_size, label="image_size")
    patch = _shape3(patch_size, label="patch_size")
    if any(image_dim % patch_dim != 0 for image_dim, patch_dim in zip(image, patch)):
        raise ProjectorConfigurationError(
            "image_size must be exactly divisible by patch_size: "
            f"image_size={image}, patch_size={patch}."
        )
    return cast(Shape3D, tuple(i // p for i, p in zip(image, patch)))


def compute_projector_geometry(
    *,
    image_size: Sequence[int],
    patch_size: Sequence[int],
    pooling_size: int = _DEFAULT_POOLING_SIZE,
    pooling_type: PoolingType = "spatial",
) -> ProjectorGeometry:
    """Build and validate visual-token pooling geometry."""

    grid = compute_patch_grid(image_size, patch_size)
    pool = int(pooling_size)
    if pool <= 0:
        raise ProjectorConfigurationError(
            f"pooling_size must be positive; received {pool}."
        )
    if pooling_type not in ("spatial", "sequence"):
        raise ProjectorConfigurationError(
            "pooling_type must be 'spatial' or 'sequence'; "
            f"received {pooling_type!r}."
        )

    input_tokens = math.prod(grid)
    if pooling_type == "spatial":
        if any(axis % pool != 0 for axis in grid):
            raise ProjectorConfigurationError(
                "Every patch-grid axis must be divisible by pooling_size for "
                f"spatial pooling: grid={grid}, pooling_size={pool}."
            )
        output_grid = cast(Shape3D, tuple(axis // pool for axis in grid))
        output_tokens = math.prod(output_grid)
    else:
        sequence_kernel = pool**3
        if input_tokens % sequence_kernel != 0:
            raise ProjectorConfigurationError(
                "The patch-token count must be divisible by pooling_size**3 "
                "for legacy sequence pooling: "
                f"tokens={input_tokens}, pooling_size={pool}, "
                f"kernel={sequence_kernel}."
            )
        # Sequence pooling has no physical 3-D output geometry.  The legacy
        # projector nevertheless produces the same token count when the input
        # grid is divisible on every axis, so expose the natural reduced grid
        # when possible; otherwise use a one-dimensional sentinel geometry.
        if all(axis % pool == 0 for axis in grid):
            output_grid = cast(Shape3D, tuple(axis // pool for axis in grid))
        else:
            output_grid = (input_tokens // sequence_kernel, 1, 1)
        output_tokens = input_tokens // sequence_kernel

    return ProjectorGeometry(
        input_grid=grid,
        output_grid=output_grid,
        input_token_count=input_tokens,
        output_token_count=output_tokens,
        pooling_size=pool,
        pooling_type=pooling_type,
    )


def _build_projection_stack(
    *,
    in_dim: int,
    out_dim: int,
    layer_type: LayerType,
    num_layers: int,
) -> nn.Sequential:
    """Reproduce the original M3D projector layer ordering exactly."""

    in_dim = int(in_dim)
    out_dim = int(out_dim)
    depth = int(num_layers)
    if in_dim <= 0 or out_dim <= 0:
        raise ProjectorConfigurationError(
            f"Projection dimensions must be positive: in={in_dim}, out={out_dim}."
        )
    if depth <= 0:
        raise ProjectorConfigurationError(
            f"num_layers must be positive; received {depth}."
        )
    if layer_type not in ("linear", "mlp"):
        raise ProjectorConfigurationError(
            f"layer_type must be 'linear' or 'mlp'; received {layer_type!r}."
        )

    modules: list[nn.Module] = [nn.Linear(in_dim, out_dim)]
    for _ in range(1, depth):
        if layer_type == "mlp":
            modules.append(nn.GELU())
        modules.append(nn.Linear(out_dim, out_dim))
    return nn.Sequential(*modules)


class SpatialPoolingProjector(nn.Module):
    """Original M3D multimodal projector with stricter contracts.

    Parameters are intentionally named exactly as in the legacy module.  The
    output can be returned as a raw tensor (the default, matching original M3D)
    or as :class:`ProjectorOutput` for code that wants geometry metadata.
    """

    def __init__(
        self,
        image_size: Sequence[int],
        patch_size: Sequence[int],
        in_dim: int,
        out_dim: int,
        layer_type: LayerType,
        layer_num: int,
        pooling_type: PoolingType = "spatial",
        pooling_size: int = _DEFAULT_POOLING_SIZE,
        *,
        freeze: bool = False,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.layer_type = layer_type
        self.layer_num = int(layer_num)
        self.pooling_type = pooling_type
        self.pooling_size = int(pooling_size)
        self.geometry = compute_projector_geometry(
            image_size=image_size,
            patch_size=patch_size,
            pooling_size=self.pooling_size,
            pooling_type=self.pooling_type,
        )

        # The name ``projector`` is a checkpoint compatibility requirement.
        self.projector = _build_projection_stack(
            in_dim=self.in_dim,
            out_dim=self.out_dim,
            layer_type=self.layer_type,
            num_layers=self.layer_num,
        )
        if freeze:
            self.requires_grad_(False)

    @property
    def num_patches_pre(self) -> list[int]:
        """Legacy-compatible pre-pooling patch-grid property."""

        return list(self.geometry.input_grid)

    @property
    def num_patches_post(self) -> list[int]:
        """Legacy-compatible post-pooling patch-grid property."""

        return list(self.geometry.output_grid)

    @property
    def proj_out_num(self) -> int:
        """Number of visual tokens inserted into the language sequence."""

        return self.geometry.output_token_count

    @property
    def input_token_count(self) -> int:
        return self.geometry.input_token_count

    def _extract_patch_tokens(
        self,
        visual_features: Tensor | VisionEncoderOutput,
    ) -> Tensor:
        if isinstance(visual_features, VisionEncoderOutput):
            if tuple(visual_features.patch_grid) != self.geometry.input_grid:
                raise ProjectorInputError(
                    "Vision output patch grid does not match projector geometry: "
                    f"vision={visual_features.patch_grid}, "
                    f"projector={self.geometry.input_grid}."
                )
            tokens = visual_features.patch_tokens
        elif isinstance(visual_features, Tensor):
            tokens = visual_features
            if tokens.ndim != 3:
                raise ProjectorInputError(
                    "Visual token tensor must have shape [B, N, C]; "
                    f"received {tuple(tokens.shape)}."
                )
            # Accept either patch-only tokens or Main ViT output with one CLS.
            if tokens.shape[1] == self.geometry.input_token_count + 1:
                tokens = tokens[:, 1:, :]
        else:
            raise ProjectorInputError(
                "visual_features must be a Tensor or VisionEncoderOutput; "
                f"received {type(visual_features).__name__}."
            )

        expected = (
            self.geometry.input_token_count,
            self.in_dim,
        )
        if tokens.ndim != 3 or tuple(tokens.shape[1:]) != expected:
            raise ProjectorInputError(
                "Patch token shape does not match projector configuration: "
                f"received={tuple(tokens.shape)}, expected=[B, {expected[0]}, "
                f"{expected[1]}]."
            )
        if not tokens.is_floating_point():
            raise ProjectorInputError(
                f"Visual tokens must be floating point; received {tokens.dtype}."
            )
        return tokens

    def pool_tokens(self, patch_tokens: Tensor) -> Tensor:
        """Pool patch tokens without applying the trainable projection MLP."""

        if self.pooling_type == "spatial":
            batch_size, _, channels = patch_tokens.shape
            d_grid, h_grid, w_grid = self.geometry.input_grid
            spatial = (
                patch_tokens.transpose(1, 2)
                .reshape(batch_size, channels, d_grid, h_grid, w_grid)
                .contiguous()
            )
            pooled = F.avg_pool3d(
                spatial,
                kernel_size=self.pooling_size,
                stride=self.pooling_size,
            )
            pooled_tokens = pooled.flatten(start_dim=2).transpose(1, 2).contiguous()
        else:
            # Exact legacy sequence-pooling semantics: contiguous groups of
            # pooling_size**3 flattened tokens are averaged.
            pooled_tokens = F.avg_pool1d(
                patch_tokens.transpose(1, 2),
                kernel_size=self.pooling_size**3,
                stride=self.pooling_size**3,
            ).transpose(1, 2).contiguous()

        expected = (
            patch_tokens.shape[0],
            self.geometry.output_token_count,
            self.in_dim,
        )
        if tuple(pooled_tokens.shape) != expected:
            raise ProjectorInputError(
                "Pooling produced an unexpected shape: "
                f"received={tuple(pooled_tokens.shape)}, expected={expected}."
            )
        return pooled_tokens

    def forward(
        self,
        visual_features: Tensor | VisionEncoderOutput,
        *,
        return_output: bool = False,
    ) -> Tensor | ProjectorOutput:
        patch_tokens = self._extract_patch_tokens(visual_features)
        pooled_tokens = self.pool_tokens(patch_tokens)

        # nn.Linear applies over the last dimension directly.  This is
        # mathematically identical to the original flatten -> Sequential ->
        # unflatten implementation, while avoiding two explicit reshape calls.
        projected = self.projector(pooled_tokens)
        expected = (
            patch_tokens.shape[0],
            self.geometry.output_token_count,
            self.out_dim,
        )
        if tuple(projected.shape) != expected:
            raise ProjectorInputError(
                "Projection produced an unexpected shape: "
                f"received={tuple(projected.shape)}, expected={expected}."
            )

        if return_output:
            return ProjectorOutput(
                projected_tokens=projected,
                input_patch_grid=self.geometry.input_grid,
                pooled_patch_grid=self.geometry.output_grid,
            )
        return projected

    def build_report(self) -> ProjectorBuildReport:
        parameters = tuple(self.parameters())
        return ProjectorBuildReport(
            layer_type=self.layer_type,
            num_layers=self.layer_num,
            input_hidden_size=self.in_dim,
            output_hidden_size=self.out_dim,
            geometry=self.geometry,
            parameter_count=sum(parameter.numel() for parameter in parameters),
            trainable_parameter_count=sum(
                parameter.numel() for parameter in parameters if parameter.requires_grad
            ),
        )

    def extra_repr(self) -> str:
        return (
            f"in_dim={self.in_dim}, out_dim={self.out_dim}, "
            f"layer_type={self.layer_type!r}, layer_num={self.layer_num}, "
            f"pooling_type={self.pooling_type!r}, "
            f"pooling_size={self.pooling_size}, "
            f"tokens={self.input_token_count}->{self.proj_out_num}"
        )


# More explicit modern name while preserving the original class name above.
M3DMultimodalProjector = SpatialPoolingProjector


def build_multimodal_projector(
    *,
    projector_config: ProjectorConfig,
    main_vision_config: VisionEncoderConfig,
    language_hidden_size: int,
) -> SpatialPoolingProjector:
    """Construct the Main-ViT-to-language projector from typed config."""

    if projector_config.projector_type != "spatial_pooling":
        raise ProjectorConfigurationError(
            "This full M3D reproduction supports projector_type='spatial_pooling'; "
            f"received {projector_config.projector_type!r}."
        )
    if not main_vision_config.enabled:
        raise ProjectorConfigurationError(
            "The multimodal projector requires the Main 3D ViT to be enabled."
        )
    projector = SpatialPoolingProjector(
        image_size=main_vision_config.image_size,
        patch_size=main_vision_config.patch_size,
        in_dim=main_vision_config.hidden_size,
        out_dim=int(language_hidden_size),
        layer_type=projector_config.layer_type,
        layer_num=projector_config.num_layers,
        pooling_type=projector_config.pooling_type,
        pooling_size=projector_config.pooling_size,
        freeze=projector_config.freeze,
    )
    return projector


def validate_visual_token_contract(
    *,
    projector: SpatialPoolingProjector,
    configured_visual_token_count: int,
) -> None:
    """Ensure tokenizer image placeholders equal projector output tokens."""

    configured = int(configured_visual_token_count)
    if configured != projector.proj_out_num:
        raise ProjectorConfigurationError(
            "Tokenizer/projector visual-token mismatch: "
            f"tokenizer={configured}, projector={projector.proj_out_num}. "
            "The number of <im_patch> placeholders must exactly equal the "
            "number of projected visual tokens."
        )


def projector_state_key_summary(projector: nn.Module) -> tuple[str, ...]:
    """Return sorted state-dict keys for checkpoint compatibility reports."""

    return tuple(sorted(projector.state_dict().keys()))


def _legacy_reference_forward(
    module: SpatialPoolingProjector,
    patch_tokens: Tensor,
) -> Tensor:
    """Literal reference implementation of original M3D projector forward."""

    batch_size = patch_tokens.shape[0]
    if module.pooling_type == "spatial":
        d_grid, h_grid, w_grid = module.geometry.input_grid
        x = patch_tokens.reshape(
            batch_size,
            d_grid,
            h_grid,
            w_grid,
            module.in_dim,
        ).permute(0, 4, 1, 2, 3)
        x = F.avg_pool3d(
            x,
            kernel_size=module.pooling_size,
            stride=module.pooling_size,
        )
        x = x.permute(0, 2, 3, 4, 1).reshape(
            batch_size,
            module.proj_out_num,
            module.in_dim,
        )
    else:
        x = patch_tokens.permute(0, 2, 1)
        x = F.avg_pool1d(
            x,
            kernel_size=module.pooling_size**3,
            stride=module.pooling_size**3,
        )
        x = x.permute(0, 2, 1)

    flattened = x.reshape(batch_size * module.proj_out_num, module.in_dim)
    flattened = module.projector(flattened)
    return flattened.reshape(batch_size, module.proj_out_num, module.out_dim)


def _run_self_test() -> Mapping[str, Any]:
    torch.manual_seed(17)

    projector = SpatialPoolingProjector(
        image_size=(8, 16, 16),
        patch_size=(4, 8, 8),
        in_dim=32,
        out_dim=48,
        layer_type="mlp",
        layer_num=2,
        pooling_type="spatial",
        pooling_size=2,
    )
    # Grid is 2x2x2 -> one pooled token.
    patch_tokens = torch.randn(2, 8, 32, requires_grad=True)
    modern = cast(Tensor, projector(patch_tokens))
    reference = _legacy_reference_forward(projector, patch_tokens)
    torch.testing.assert_close(modern, reference, rtol=1e-6, atol=1e-6)
    modern.square().mean().backward()
    if patch_tokens.grad is None or not torch.isfinite(patch_tokens.grad).all():
        raise AssertionError("Projector backward did not produce finite gradients.")

    with_cls = torch.cat((torch.randn(2, 1, 32), patch_tokens.detach()), dim=1)
    from_cls = cast(Tensor, projector(with_cls))
    without_cls = cast(Tensor, projector(patch_tokens.detach()))
    torch.testing.assert_close(from_cls, without_cls, rtol=0.0, atol=0.0)

    state_keys = projector_state_key_summary(projector)
    expected_keys = (
        "projector.0.bias",
        "projector.0.weight",
        "projector.2.bias",
        "projector.2.weight",
    )
    if state_keys != expected_keys:
        raise AssertionError(
            f"Unexpected checkpoint keys: {state_keys}; expected {expected_keys}."
        )

    sequence_projector = SpatialPoolingProjector(
        image_size=(8, 16, 16),
        patch_size=(4, 8, 8),
        in_dim=16,
        out_dim=24,
        layer_type="linear",
        layer_num=2,
        pooling_type="sequence",
        pooling_size=2,
    )
    sequence_tokens = torch.randn(2, 8, 16)
    sequence_modern = cast(Tensor, sequence_projector(sequence_tokens))
    sequence_reference = _legacy_reference_forward(
        sequence_projector,
        sequence_tokens,
    )
    torch.testing.assert_close(
        sequence_modern,
        sequence_reference,
        rtol=1e-6,
        atol=1e-6,
    )

    frozen = SpatialPoolingProjector(
        image_size=(8, 16, 16),
        patch_size=(4, 8, 8),
        in_dim=8,
        out_dim=8,
        layer_type="linear",
        layer_num=1,
        freeze=True,
    )
    if any(parameter.requires_grad for parameter in frozen.parameters()):
        raise AssertionError("freeze=True did not freeze all projector parameters.")

    production_geometry = compute_projector_geometry(
        image_size=(32, 256, 256),
        patch_size=(4, 16, 16),
        pooling_size=2,
        pooling_type="spatial",
    )
    if production_geometry.input_token_count != 2048:
        raise AssertionError("Expected 2048 pre-pooling visual tokens.")
    if production_geometry.output_token_count != 256:
        raise AssertionError("Expected 256 post-pooling visual tokens.")

    validate_visual_token_contract(
        projector=projector,
        configured_visual_token_count=projector.proj_out_num,
    )
    mismatch_detected = False
    try:
        validate_visual_token_contract(
            projector=projector,
            configured_visual_token_count=999,
        )
    except ProjectorConfigurationError:
        mismatch_detected = True
    if not mismatch_detected:
        raise AssertionError("Visual-token contract mismatch was not detected.")

    return {
        "status": "passed",
        "spatial_output_shape": list(modern.shape),
        "sequence_output_shape": list(sequence_modern.shape),
        "checkpoint_keys": list(state_keys),
        "production_input_tokens": production_geometry.input_token_count,
        "production_output_tokens": production_geometry.output_token_count,
        "legacy_numerical_equivalence": True,
        "cls_token_removed": True,
        "visual_token_mismatch_detected": mismatch_detected,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run CPU numerical and checkpoint-compatibility tests.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path for the self-test JSON report.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.self_test:
        raise SystemExit("Nothing to do. Use --self-test.")
    report = dict(_run_self_test())
    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
