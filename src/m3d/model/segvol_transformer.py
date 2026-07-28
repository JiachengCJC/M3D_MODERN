"""Modernized SegVol two-way transformer with PyTorch SDPA.

This module reproduces the original SegVol/SAM two-way transformer topology:

1. sparse-token self-attention,
2. sparse tokens attending to dense image tokens,
3. an MLP on sparse tokens,
4. dense image tokens attending back to sparse tokens,
5. a final sparse-token-to-image attention layer.

Only the attention *kernel* changes.  The projection modules, residual order,
normalization order, MLP structure, and state-dict names remain compatible with
the original SegVol checkpoint.  ``ProjectedSDPAAttention`` delegates to
``torch.nn.functional.scaled_dot_product_attention`` and can require the Flash
SDPA backend on ASPIRE 2A A100 GPUs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Callable, TypeAlias, TypeVar

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from m3d.config import OptimizationConfig, SegmentationConfig, VisionEncoderConfig
from m3d.model.attention import (
    AttentionBackendName,
    ProjectedSDPAAttention,
)


__all__ = [
    "MLPBlock",
    "SegVolTransformerConfigurationError",
    "SegVolTransformerExecutionError",
    "SegVolTransformerReport",
    "TwoWayAttentionBlock",
    "TwoWayTransformer",
    "build_segvol_two_way_transformer",
    "segvol_transformer_parameter_names",
]


_DEFAULT_MLP_DIM = 2048
_DEFAULT_ATTENTION_DOWNSAMPLE_RATE = 2
_DEFAULT_ACTIVATION_CHECKPOINT_INTERVAL = 1

ActivationFactory: TypeAlias = Callable[[], nn.Module]
TModule = TypeVar("TModule", bound=nn.Module)


class SegVolTransformerConfigurationError(ValueError):
    """Raised when the two-way transformer configuration is invalid."""


class SegVolTransformerExecutionError(RuntimeError):
    """Raised when runtime tensor contracts are violated."""


@dataclass(frozen=True, slots=True)
class SegVolTransformerReport:
    """Static description of a constructed two-way transformer."""

    depth: int
    embedding_dim: int
    num_heads: int
    mlp_dim: int
    attention_downsample_rate: int
    attention_backend: str
    activation_checkpointing: bool
    activation_checkpoint_interval: int
    parameter_count: int
    trainable_parameter_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "depth": self.depth,
            "embedding_dim": self.embedding_dim,
            "num_heads": self.num_heads,
            "mlp_dim": self.mlp_dim,
            "attention_downsample_rate": self.attention_downsample_rate,
            "attention_backend": self.attention_backend,
            "activation_checkpointing": self.activation_checkpointing,
            "activation_checkpoint_interval": self.activation_checkpoint_interval,
            "parameter_count": self.parameter_count,
            "trainable_parameter_count": self.trainable_parameter_count,
        }


class MLPBlock(nn.Module):
    """SegVol-compatible two-layer MLP.

    The names ``lin1``, ``lin2`` and ``act`` intentionally match the original
    implementation so that keys such as ``layers.0.mlp.lin1.weight`` load
    without conversion.
    """

    def __init__(
        self,
        embedding_dim: int,
        mlp_dim: int,
        activation: type[nn.Module] = nn.ReLU,
    ) -> None:
        super().__init__()
        embedding_dim = _positive_int(embedding_dim, name="embedding_dim")
        mlp_dim = _positive_int(mlp_dim, name="mlp_dim")
        if not isinstance(activation, type) or not issubclass(activation, nn.Module):
            raise SegVolTransformerConfigurationError(
                "activation must be an nn.Module class, for example nn.ReLU."
            )

        self.embedding_dim = embedding_dim
        self.mlp_dim = mlp_dim
        self.lin1 = nn.Linear(embedding_dim, mlp_dim)
        self.lin2 = nn.Linear(mlp_dim, embedding_dim)
        self.act = activation()

    def forward(self, x: Tensor) -> Tensor:
        _validate_rank3_embedding(
            x,
            name="mlp input",
            embedding_dim=self.embedding_dim,
        )
        return self.lin2(self.act(self.lin1(x)))


class TwoWayAttentionBlock(nn.Module):
    """One SegVol two-way attention block using projected SDPA kernels."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int = _DEFAULT_MLP_DIM,
        activation: type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = _DEFAULT_ATTENTION_DOWNSAMPLE_RATE,
        skip_first_layer_pe: bool = False,
        *,
        attention_backend: AttentionBackendName = "sdpa",
        require_flash_sdpa: bool = True,
    ) -> None:
        super().__init__()

        embedding_dim, num_heads, mlp_dim, attention_downsample_rate = (
            _validate_transformer_dimensions(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                attention_downsample_rate=attention_downsample_rate,
            )
        )

        # Module names intentionally reproduce the original checkpoint layout.
        self.self_attn = ProjectedSDPAAttention(
            embedding_dim,
            num_heads,
            downsample_rate=1,
            backend=attention_backend,
            require_flash_sdpa=require_flash_sdpa,
        )
        self.norm1 = nn.LayerNorm(embedding_dim)

        self.cross_attn_token_to_image = ProjectedSDPAAttention(
            embedding_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
            backend=attention_backend,
            require_flash_sdpa=require_flash_sdpa,
        )
        self.norm2 = nn.LayerNorm(embedding_dim)

        self.mlp = MLPBlock(embedding_dim, mlp_dim, activation)
        self.norm3 = nn.LayerNorm(embedding_dim)

        self.norm4 = nn.LayerNorm(embedding_dim)
        self.cross_attn_image_to_token = ProjectedSDPAAttention(
            embedding_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
            backend=attention_backend,
            require_flash_sdpa=require_flash_sdpa,
        )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.attention_downsample_rate = attention_downsample_rate
        self.skip_first_layer_pe = bool(skip_first_layer_pe)

    def forward(
        self,
        queries: Tensor,
        keys: Tensor,
        query_pe: Tensor,
        key_pe: Tensor,
    ) -> tuple[Tensor, Tensor]:
        _validate_token_inputs(
            queries=queries,
            keys=keys,
            query_pe=query_pe,
            key_pe=key_pe,
            embedding_dim=self.embedding_dim,
        )
        input_dtype = queries.dtype

        # (1) Sparse-token self-attention.  The first block deliberately omits
        # both positional encoding and the residual around this first
        # self-attention, matching the original SAM/SegVol implementation.
        if self.skip_first_layer_pe:
            queries = self.self_attn(q=queries, k=queries, v=queries)
        else:
            q = queries + query_pe
            attention_output = self.self_attn(q=q, k=q, v=queries)
            queries = queries + attention_output
        queries = self.norm1(queries)

        # (2) Sparse tokens attend to dense image tokens.
        q = queries + query_pe
        k = keys + key_pe
        attention_output = self.cross_attn_token_to_image(q=q, k=k, v=keys)
        queries = self.norm2(queries + attention_output)

        # (3) MLP on sparse tokens.
        queries = self.norm3(queries + self.mlp(queries))

        # (4) Dense image tokens attend back to sparse tokens.
        q = queries + query_pe
        k = keys + key_pe
        attention_output = self.cross_attn_image_to_token(q=k, k=q, v=queries)
        keys = self.norm4(keys + attention_output)

        # CPU and CUDA autocast have different allowlists for LayerNorm and
        # residual additions. Preserve the block's input activation dtype at
        # layer boundaries so a multi-layer decoder cannot leave sparse and
        # dense streams in different dtypes. The Linear projections still run
        # in the autocast compute dtype.
        return (
            queries.to(dtype=input_dtype),
            keys.to(dtype=input_dtype),
        )

    def extra_repr(self) -> str:
        return (
            f"embedding_dim={self.embedding_dim}, num_heads={self.num_heads}, "
            f"mlp_dim={self.mlp_dim}, "
            f"attention_downsample_rate={self.attention_downsample_rate}, "
            f"skip_first_layer_pe={self.skip_first_layer_pe}"
        )


class TwoWayTransformer(nn.Module):
    """SegVol-compatible two-way transformer with optional layer checkpointing.

    The public ``forward`` signature and return type intentionally match the
    original module used by ``MaskDecoder``:

    ``processed_sparse_tokens, processed_dense_image_tokens = transformer(...)``
    """

    def __init__(
        self,
        depth: int,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        activation: type[nn.Module] = nn.ReLU,
        attention_downsample_rate: int = _DEFAULT_ATTENTION_DOWNSAMPLE_RATE,
        *,
        attention_backend: AttentionBackendName = "sdpa",
        require_flash_sdpa: bool = True,
        activation_checkpointing: bool = False,
        activation_checkpoint_interval: int = _DEFAULT_ACTIVATION_CHECKPOINT_INTERVAL,
    ) -> None:
        super().__init__()

        depth = _positive_int(depth, name="depth")
        embedding_dim, num_heads, mlp_dim, attention_downsample_rate = (
            _validate_transformer_dimensions(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                attention_downsample_rate=attention_downsample_rate,
            )
        )
        activation_checkpoint_interval = _positive_int(
            activation_checkpoint_interval,
            name="activation_checkpoint_interval",
        )

        self.depth = depth
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.mlp_dim = mlp_dim
        self.attention_downsample_rate = attention_downsample_rate
        self.attention_backend = str(attention_backend)
        self.require_flash_sdpa = bool(require_flash_sdpa)
        self.activation_checkpointing = bool(activation_checkpointing)
        self.activation_checkpoint_interval = activation_checkpoint_interval

        self.layers = nn.ModuleList(
            TwoWayAttentionBlock(
                embedding_dim=embedding_dim,
                num_heads=num_heads,
                mlp_dim=mlp_dim,
                activation=activation,
                attention_downsample_rate=attention_downsample_rate,
                skip_first_layer_pe=(layer_index == 0),
                attention_backend=attention_backend,
                require_flash_sdpa=require_flash_sdpa,
            )
            for layer_index in range(depth)
        )

        # Names intentionally match the original SegVol checkpoint.
        self.final_attn_token_to_image = ProjectedSDPAAttention(
            embedding_dim,
            num_heads,
            downsample_rate=attention_downsample_rate,
            backend=attention_backend,
            require_flash_sdpa=require_flash_sdpa,
        )
        self.norm_final_attn = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        image_embedding: Tensor,
        image_pe: Tensor,
        point_embedding: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Run sparse↔dense two-way attention.

        Args:
            image_embedding: Dense image features ``[B, C, D, H, W]``.
            image_pe: Dense positional encoding ``[1 or B, C, D, H, W]``.
            point_embedding: Sparse prompt/output tokens ``[B, N, C]``.

        Returns:
            ``(processed_sparse_tokens, processed_image_tokens)`` where the
            second tensor is flattened to ``[B, D*H*W, C]`` for checkpoint
            compatibility with the original mask decoder.
        """

        batch_size, channels, spatial_shape = _validate_dense_inputs(
            image_embedding=image_embedding,
            image_pe=image_pe,
            point_embedding=point_embedding,
            embedding_dim=self.embedding_dim,
        )

        # [B,C,D,H,W] -> [B,D*H*W,C]
        keys = image_embedding.flatten(start_dim=2).transpose(1, 2).contiguous()
        key_pe = image_pe.flatten(start_dim=2).transpose(1, 2)
        if image_pe.shape[0] == 1 and batch_size > 1:
            # ``expand`` avoids materialising B copies of the fixed positional
            # encoding.  It is mathematically identical to repeat_interleave.
            key_pe = key_pe.expand(batch_size, -1, -1)
        key_pe = key_pe.contiguous()

        queries = point_embedding
        query_pe = point_embedding

        for layer_index, layer in enumerate(self.layers):
            if self._should_checkpoint_layer(layer_index, queries, keys):
                queries, keys = checkpoint(
                    layer,
                    queries,
                    keys,
                    query_pe,
                    key_pe,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )
            else:
                queries, keys = layer(
                    queries=queries,
                    keys=keys,
                    query_pe=query_pe,
                    key_pe=key_pe,
                )

        # Final sparse-token-to-image attention.
        q = queries + query_pe
        k = keys + key_pe
        attention_output = self.final_attn_token_to_image(q=q, k=k, v=keys)
        queries = self.norm_final_attn(queries + attention_output)

        expected_image_tokens = spatial_shape[0] * spatial_shape[1] * spatial_shape[2]
        if keys.shape != (batch_size, expected_image_tokens, channels):
            raise SegVolTransformerExecutionError(
                "Two-way transformer changed the dense token shape unexpectedly: "
                f"received {tuple(keys.shape)}, expected "
                f"{(batch_size, expected_image_tokens, channels)}."
            )

        return queries, keys

    def _should_checkpoint_layer(
        self,
        layer_index: int,
        queries: Tensor,
        keys: Tensor,
    ) -> bool:
        if not self.activation_checkpointing:
            return False
        if not self.training or not torch.is_grad_enabled():
            return False
        if (layer_index + 1) % self.activation_checkpoint_interval != 0:
            return False

        # Checkpointing is useful whenever either input activations or layer
        # parameters participate in autograd.
        if queries.requires_grad or keys.requires_grad:
            return True
        return any(parameter.requires_grad for parameter in self.layers[layer_index].parameters())

    def report(self) -> SegVolTransformerReport:
        parameters = tuple(self.parameters())
        return SegVolTransformerReport(
            depth=self.depth,
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            mlp_dim=self.mlp_dim,
            attention_downsample_rate=self.attention_downsample_rate,
            attention_backend=(
                "flash-sdpa"
                if self.require_flash_sdpa and self.attention_backend == "sdpa"
                else self.attention_backend
            ),
            activation_checkpointing=self.activation_checkpointing,
            activation_checkpoint_interval=self.activation_checkpoint_interval,
            parameter_count=sum(parameter.numel() for parameter in parameters),
            trainable_parameter_count=sum(
                parameter.numel() for parameter in parameters if parameter.requires_grad
            ),
        )

    def extra_repr(self) -> str:
        return (
            f"depth={self.depth}, embedding_dim={self.embedding_dim}, "
            f"num_heads={self.num_heads}, mlp_dim={self.mlp_dim}, "
            f"attention_downsample_rate={self.attention_downsample_rate}, "
            f"backend={self.attention_backend}, "
            f"require_flash_sdpa={self.require_flash_sdpa}, "
            f"activation_checkpointing={self.activation_checkpointing}"
        )


def build_segvol_two_way_transformer(
    segmentation_config: SegmentationConfig,
    segmentation_vision_config: VisionEncoderConfig,
    optimization_config: OptimizationConfig,
    *,
    mlp_dim: int = _DEFAULT_MLP_DIM,
    attention_downsample_rate: int = _DEFAULT_ATTENTION_DOWNSAMPLE_RATE,
) -> TwoWayTransformer:
    """Build the production SegVol decoder transformer from project config.

    ``mlp_dim=2048`` and ``attention_downsample_rate=2`` are intentionally
    fixed to the original SegVol architecture because the already-approved
    configuration schema does not expose architecture-changing alternatives.
    """

    if segmentation_config.prompt_embed_dim != segmentation_vision_config.hidden_size:
        raise SegVolTransformerConfigurationError(
            "SegVol decoder embedding dimension must equal the independent "
            "SegVol image encoder hidden size: "
            f"prompt_embed_dim={segmentation_config.prompt_embed_dim}, "
            f"seg_vision.hidden_size={segmentation_vision_config.hidden_size}."
        )

    transformer = TwoWayTransformer(
        depth=segmentation_config.decoder_depth,
        embedding_dim=segmentation_config.prompt_embed_dim,
        num_heads=segmentation_config.decoder_heads,
        mlp_dim=mlp_dim,
        activation=nn.ReLU,
        attention_downsample_rate=attention_downsample_rate,
        attention_backend=segmentation_vision_config.attention_backend,
        require_flash_sdpa=segmentation_vision_config.require_flash_sdpa,
        activation_checkpointing=optimization_config.checkpoint_segmentation_decoder,
        activation_checkpoint_interval=1,
    )
    return transformer


def segvol_transformer_parameter_names(module: nn.Module) -> tuple[str, ...]:
    """Return sorted state-dict keys for checkpoint compatibility reports."""

    return tuple(sorted(module.state_dict().keys()))


def _positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool):
        raise SegVolTransformerConfigurationError(f"{name} must be an integer, not bool.")
    value = int(value)
    if value <= 0:
        raise SegVolTransformerConfigurationError(f"{name} must be positive; got {value}.")
    return value


def _validate_transformer_dimensions(
    *,
    embedding_dim: int,
    num_heads: int,
    mlp_dim: int,
    attention_downsample_rate: int,
) -> tuple[int, int, int, int]:
    embedding_dim = _positive_int(embedding_dim, name="embedding_dim")
    num_heads = _positive_int(num_heads, name="num_heads")
    mlp_dim = _positive_int(mlp_dim, name="mlp_dim")
    attention_downsample_rate = _positive_int(
        attention_downsample_rate,
        name="attention_downsample_rate",
    )

    if embedding_dim % attention_downsample_rate != 0:
        raise SegVolTransformerConfigurationError(
            "embedding_dim must be divisible by attention_downsample_rate: "
            f"embedding_dim={embedding_dim}, "
            f"attention_downsample_rate={attention_downsample_rate}."
        )
    internal_dim = embedding_dim // attention_downsample_rate
    if embedding_dim % num_heads != 0:
        raise SegVolTransformerConfigurationError(
            "num_heads must divide embedding_dim for sparse self-attention: "
            f"embedding_dim={embedding_dim}, num_heads={num_heads}."
        )
    if internal_dim % num_heads != 0:
        raise SegVolTransformerConfigurationError(
            "num_heads must divide the downsampled attention dimension: "
            f"internal_dim={internal_dim}, num_heads={num_heads}."
        )
    return embedding_dim, num_heads, mlp_dim, attention_downsample_rate


def _validate_rank3_embedding(
    tensor: Tensor,
    *,
    name: str,
    embedding_dim: int,
) -> None:
    if tensor.ndim != 3:
        raise SegVolTransformerExecutionError(
            f"{name} must be [B,N,C]; received shape {tuple(tensor.shape)}."
        )
    if tensor.shape[-1] != embedding_dim:
        raise SegVolTransformerExecutionError(
            f"{name} hidden size must be {embedding_dim}; received {tensor.shape[-1]}."
        )
    if not tensor.is_floating_point():
        raise SegVolTransformerExecutionError(
            f"{name} must be floating point; received {tensor.dtype}."
        )


def _validate_token_inputs(
    *,
    queries: Tensor,
    keys: Tensor,
    query_pe: Tensor,
    key_pe: Tensor,
    embedding_dim: int,
) -> None:
    for name, tensor in (
        ("queries", queries),
        ("keys", keys),
        ("query_pe", query_pe),
        ("key_pe", key_pe),
    ):
        _validate_rank3_embedding(
            tensor,
            name=name,
            embedding_dim=embedding_dim,
        )

    if queries.shape != query_pe.shape:
        raise SegVolTransformerExecutionError(
            "queries and query_pe must have identical shape; received "
            f"{tuple(queries.shape)} and {tuple(query_pe.shape)}."
        )
    if keys.shape != key_pe.shape:
        raise SegVolTransformerExecutionError(
            "keys and key_pe must have identical shape; received "
            f"{tuple(keys.shape)} and {tuple(key_pe.shape)}."
        )
    if queries.shape[0] != keys.shape[0]:
        raise SegVolTransformerExecutionError(
            "Sparse and dense token batches must match; received "
            f"{queries.shape[0]} and {keys.shape[0]}."
        )

    reference_device = queries.device
    reference_dtype = queries.dtype
    for name, tensor in (
        ("keys", keys),
        ("query_pe", query_pe),
        ("key_pe", key_pe),
    ):
        if tensor.device != reference_device:
            raise SegVolTransformerExecutionError(
                f"{name} is on {tensor.device}, expected {reference_device}."
            )
        if tensor.dtype != reference_dtype:
            raise SegVolTransformerExecutionError(
                f"{name} uses {tensor.dtype}, expected {reference_dtype}."
            )


def _validate_dense_inputs(
    *,
    image_embedding: Tensor,
    image_pe: Tensor,
    point_embedding: Tensor,
    embedding_dim: int,
) -> tuple[int, int, tuple[int, int, int]]:
    if image_embedding.ndim != 5:
        raise SegVolTransformerExecutionError(
            "image_embedding must be [B,C,D,H,W]; received "
            f"{tuple(image_embedding.shape)}."
        )
    if image_pe.ndim != 5:
        raise SegVolTransformerExecutionError(
            f"image_pe must be [1 or B,C,D,H,W]; received {tuple(image_pe.shape)}."
        )
    _validate_rank3_embedding(
        point_embedding,
        name="point_embedding",
        embedding_dim=embedding_dim,
    )

    batch_size, channels, depth, height, width = image_embedding.shape
    if batch_size <= 0 or min(depth, height, width) <= 0:
        raise SegVolTransformerExecutionError(
            f"image_embedding has an empty dimension: {tuple(image_embedding.shape)}."
        )
    if channels != embedding_dim:
        raise SegVolTransformerExecutionError(
            "image_embedding channel count must equal embedding_dim: "
            f"channels={channels}, embedding_dim={embedding_dim}."
        )
    if point_embedding.shape[0] != batch_size:
        raise SegVolTransformerExecutionError(
            "point_embedding batch must match image_embedding batch: "
            f"{point_embedding.shape[0]} != {batch_size}."
        )
    if point_embedding.shape[1] <= 0:
        raise SegVolTransformerExecutionError(
            "point_embedding must contain at least one sparse token."
        )

    if image_pe.shape[0] not in (1, batch_size):
        raise SegVolTransformerExecutionError(
            "image_pe batch must be 1 or equal image batch; received "
            f"image_pe={image_pe.shape[0]}, image={batch_size}."
        )
    if image_pe.shape[1:] != image_embedding.shape[1:]:
        raise SegVolTransformerExecutionError(
            "image_pe spatial/channel shape must match image_embedding; received "
            f"{tuple(image_pe.shape[1:])} and {tuple(image_embedding.shape[1:])}."
        )

    tensors = {
        "image_embedding": image_embedding,
        "image_pe": image_pe,
        "point_embedding": point_embedding,
    }
    reference_device = image_embedding.device
    reference_dtype = image_embedding.dtype
    for name, tensor in tensors.items():
        if not tensor.is_floating_point():
            raise SegVolTransformerExecutionError(
                f"{name} must be floating point; received {tensor.dtype}."
            )
        if tensor.device != reference_device:
            raise SegVolTransformerExecutionError(
                f"{name} is on {tensor.device}, expected {reference_device}."
            )
        if tensor.dtype != reference_dtype:
            raise SegVolTransformerExecutionError(
                f"{name} uses {tensor.dtype}, expected {reference_dtype}."
            )
        if not torch.isfinite(tensor).all():
            raise SegVolTransformerExecutionError(f"{name} contains NaN or Inf values.")

    return batch_size, channels, (depth, height, width)


# ---------------------------------------------------------------------------
# Numerical-reference helpers used only by the CPU self-test.
# ---------------------------------------------------------------------------


def _legacy_projected_attention(
    module: ProjectedSDPAAttention,
    q: Tensor,
    k: Tensor,
    v: Tensor,
) -> Tensor:
    projected_q = module._separate_heads(module.q_proj(q))
    projected_k = module._separate_heads(module.k_proj(k))
    projected_v = module._separate_heads(module.v_proj(v))
    scale = projected_q.shape[-1] ** -0.5
    attention = torch.softmax(
        torch.matmul(projected_q, projected_k.transpose(-2, -1)) * scale,
        dim=-1,
    )
    output = torch.matmul(attention, projected_v)
    output = module._recombine_heads(output)
    return module.out_proj(output)


def _legacy_block_forward(
    block: TwoWayAttentionBlock,
    queries: Tensor,
    keys: Tensor,
    query_pe: Tensor,
    key_pe: Tensor,
) -> tuple[Tensor, Tensor]:
    if block.skip_first_layer_pe:
        queries = _legacy_projected_attention(
            block.self_attn,
            queries,
            queries,
            queries,
        )
    else:
        q = queries + query_pe
        queries = queries + _legacy_projected_attention(
            block.self_attn,
            q,
            q,
            queries,
        )
    queries = block.norm1(queries)

    q = queries + query_pe
    k = keys + key_pe
    queries = block.norm2(
        queries
        + _legacy_projected_attention(
            block.cross_attn_token_to_image,
            q,
            k,
            keys,
        )
    )
    queries = block.norm3(queries + block.mlp(queries))

    q = queries + query_pe
    k = keys + key_pe
    keys = block.norm4(
        keys
        + _legacy_projected_attention(
            block.cross_attn_image_to_token,
            k,
            q,
            queries,
        )
    )
    return queries, keys


def _legacy_transformer_forward(
    transformer: TwoWayTransformer,
    image_embedding: Tensor,
    image_pe: Tensor,
    point_embedding: Tensor,
) -> tuple[Tensor, Tensor]:
    batch_size = image_embedding.shape[0]
    keys = image_embedding.flatten(2).transpose(1, 2).contiguous()
    key_pe = image_pe.flatten(2).transpose(1, 2)
    if image_pe.shape[0] == 1 and batch_size > 1:
        key_pe = key_pe.expand(batch_size, -1, -1)
    key_pe = key_pe.contiguous()
    queries = point_embedding
    query_pe = point_embedding

    for layer in transformer.layers:
        queries, keys = _legacy_block_forward(
            layer,
            queries,
            keys,
            query_pe,
            key_pe,
        )

    q = queries + query_pe
    k = keys + key_pe
    queries = transformer.norm_final_attn(
        queries
        + _legacy_projected_attention(
            transformer.final_attn_token_to_image,
            q,
            k,
            keys,
        )
    )
    return queries, keys


def run_cpu_self_test() -> dict[str, object]:
    torch.manual_seed(17)

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
    transformer.train()

    image = torch.randn(2, 32, 2, 2, 2, requires_grad=True)
    image_pe = torch.randn(1, 32, 2, 2, 2)
    sparse = torch.randn(2, 3, 32, requires_grad=True)

    modern_queries, modern_keys = transformer(image, image_pe, sparse)
    legacy_queries, legacy_keys = _legacy_transformer_forward(
        transformer,
        image,
        image_pe,
        sparse,
    )
    torch.testing.assert_close(modern_queries, legacy_queries, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(modern_keys, legacy_keys, rtol=1e-5, atol=1e-6)

    loss = modern_queries.square().mean() + modern_keys.square().mean()
    loss.backward()
    if image.grad is None or sparse.grad is None:
        raise AssertionError("Backward did not produce input gradients.")
    if not torch.isfinite(image.grad).all() or not torch.isfinite(sparse.grad).all():
        raise AssertionError("Backward produced non-finite gradients.")

    autocast_image = image.detach().clone().requires_grad_(True)
    autocast_sparse = sparse.detach().clone().requires_grad_(True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        autocast_queries, autocast_keys = transformer(
            autocast_image,
            image_pe,
            autocast_sparse,
        )
        autocast_loss = (
            autocast_queries.float().square().mean()
            + autocast_keys.float().square().mean()
        )
    autocast_loss.backward()
    if autocast_image.grad is None or autocast_sparse.grad is None:
        raise AssertionError("Autocast backward did not produce input gradients.")
    if (
        not torch.isfinite(autocast_image.grad).all()
        or not torch.isfinite(autocast_sparse.grad).all()
    ):
        raise AssertionError("Autocast backward produced non-finite gradients.")

    checkpointed = TwoWayTransformer(
        depth=2,
        embedding_dim=32,
        num_heads=4,
        mlp_dim=64,
        attention_downsample_rate=2,
        attention_backend="math",
        require_flash_sdpa=False,
        activation_checkpointing=True,
    )
    checkpointed.load_state_dict(transformer.state_dict(), strict=True)
    checkpointed.train()

    checkpoint_image = image.detach().clone().requires_grad_(True)
    checkpoint_sparse = sparse.detach().clone().requires_grad_(True)
    checkpoint_queries, checkpoint_keys = checkpointed(
        checkpoint_image,
        image_pe,
        checkpoint_sparse,
    )
    torch.testing.assert_close(checkpoint_queries, modern_queries.detach(), rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(checkpoint_keys, modern_keys.detach(), rtol=1e-5, atol=1e-6)
    (checkpoint_queries.mean() + checkpoint_keys.mean()).backward()
    if checkpoint_image.grad is None or checkpoint_sparse.grad is None:
        raise AssertionError("Checkpointed backward did not produce gradients.")

    state_keys = segvol_transformer_parameter_names(transformer)
    required_keys = {
        "layers.0.self_attn.q_proj.weight",
        "layers.0.self_attn.k_proj.weight",
        "layers.0.self_attn.v_proj.weight",
        "layers.0.self_attn.out_proj.weight",
        "layers.0.cross_attn_token_to_image.q_proj.weight",
        "layers.0.mlp.lin1.weight",
        "layers.0.mlp.lin2.weight",
        "layers.0.cross_attn_image_to_token.out_proj.weight",
        "final_attn_token_to_image.q_proj.weight",
        "norm_final_attn.weight",
    }
    missing_required = sorted(required_keys.difference(state_keys))
    if missing_required:
        raise AssertionError(f"Missing checkpoint-compatible keys: {missing_required}")

    # Invalid positional-encoding batches must fail before attention execution.
    invalid_batch_detected = False
    try:
        transformer(
            image.detach(),
            torch.randn(3, 32, 2, 2, 2),
            sparse.detach(),
        )
    except SegVolTransformerExecutionError:
        invalid_batch_detected = True
    if not invalid_batch_detected:
        raise AssertionError("Invalid image_pe batch was not rejected.")

    report = transformer.report()
    return {
        "status": "passed",
        "legacy_numerical_equivalence": True,
        "activation_checkpoint_equivalence": True,
        "query_output_shape": list(modern_queries.shape),
        "image_token_output_shape": list(modern_keys.shape),
        "checkpoint_key_count": len(state_keys),
        "required_checkpoint_keys_present": True,
        "invalid_image_pe_batch_detected": invalid_batch_detected,
        "report": report.as_dict(),
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
