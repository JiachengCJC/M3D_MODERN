"""Efficient attention primitives used by M3D-Modernized.

This module is the single attention implementation shared at the *code* level
by the two independent 3D image encoders and the SegVol two-way transformer.
Sharing this Python implementation does not share model parameters:

* ``main_vision`` creates its own :class:`FusedSelfAttention` modules.
* ``seg_vision`` creates a second, independent set of modules.
* the SegVol decoder creates :class:`ProjectedSDPAAttention` modules whose
  parameter names remain compatible with the original implementation.

The default execution path uses ``torch.nn.functional.scaled_dot_product_attention``
(SDPA).  On ASPIRE 2A A100 GPUs, ``require_flash_sdpa=True`` forces PyTorch's
Flash-Attention backend instead of silently falling back to the memory-hungry
mathematical implementation.

Checkpoint compatibility
------------------------
``FusedSelfAttention`` deliberately exposes the same parameter names used by
MONAI's self-attention block:

* ``qkv.weight`` / ``qkv.bias``
* ``out_proj.weight`` / ``out_proj.bias``

``ProjectedSDPAAttention`` preserves the original SegVol names:

* ``q_proj``
* ``k_proj``
* ``v_proj``
* ``out_proj``

Consequently, changing the attention kernel does not by itself require
retraining or rewriting those checkpoint keys.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import math
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel


AttentionBackendName: TypeAlias = Literal["sdpa", "math"]
AttentionMask: TypeAlias = Tensor | None

_BOOL_MASK_DESCRIPTION: Final[str] = (
    "Boolean SDPA masks use True for positions that are allowed to attend and "
    "False for positions that are masked."
)


class AttentionConfigurationError(ValueError):
    """Raised when an attention module is configured inconsistently."""


class AttentionExecutionError(RuntimeError):
    """Raised when the selected SDPA backend cannot execute the request."""


@dataclass(frozen=True, slots=True)
class AttentionPolicy:
    """Runtime policy for one attention module.

    Parameters
    ----------
    backend:
        ``"sdpa"`` lets PyTorch use SDPA.  If ``require_flash`` is true, the
        backend is restricted to Flash Attention.  ``"math"`` forces the
        reference mathematical SDPA backend and is intended for CPU tests and
        numerical debugging.
    require_flash:
        Require the Flash SDPA backend.  This is the production setting for
        both 3D ViTs on ASPIRE 2A A100 GPUs.
    """

    backend: AttentionBackendName = "sdpa"
    require_flash: bool = True

    def __post_init__(self) -> None:
        if self.backend not in ("sdpa", "math"):
            raise AttentionConfigurationError(
                f"Unsupported attention backend {self.backend!r}; expected "
                "'sdpa' or 'math'."
            )
        if self.require_flash and self.backend != "sdpa":
            raise AttentionConfigurationError(
                "require_flash=True requires backend='sdpa'."
            )

    @property
    def label(self) -> str:
        if self.backend == "math":
            return "math-sdpa"
        if self.require_flash:
            return "flash-sdpa"
        return "automatic-sdpa"


@dataclass(frozen=True, slots=True)
class AttentionShape:
    """Validated logical shape of an attention operation."""

    batch_size: int
    num_heads: int
    query_length: int
    key_value_length: int
    head_dim: int

    @property
    def score_elements(self) -> int:
        """Elements in the score matrix materialised by naive attention."""

        return (
            self.batch_size
            * self.num_heads
            * self.query_length
            * self.key_value_length
        )

    def naive_score_bytes(self, dtype: torch.dtype) -> int:
        """Approximate bytes of a naive attention score matrix."""

        return self.score_elements * torch.empty((), dtype=dtype).element_size()


@dataclass(frozen=True, slots=True)
class AttentionRuntimeInfo:
    """Serializable diagnostics for one configured attention module."""

    policy: str
    hidden_size: int
    num_heads: int
    head_dim: int
    dropout: float
    parameter_count: int


def _validate_probability(value: float, *, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value < 1.0:
        raise AttentionConfigurationError(
            f"{name} must be finite and in [0, 1); received {value!r}."
        )
    return value


def _validate_projection_dimensions(
    *,
    embedding_dim: int,
    num_heads: int,
    internal_dim: int | None = None,
) -> tuple[int, int]:
    embedding_dim = int(embedding_dim)
    num_heads = int(num_heads)
    internal_dim = embedding_dim if internal_dim is None else int(internal_dim)

    if embedding_dim <= 0:
        raise AttentionConfigurationError("embedding_dim must be positive.")
    if internal_dim <= 0:
        raise AttentionConfigurationError("internal_dim must be positive.")
    if num_heads <= 0:
        raise AttentionConfigurationError("num_heads must be positive.")
    if internal_dim % num_heads != 0:
        raise AttentionConfigurationError(
            "The projected attention dimension must be divisible by num_heads: "
            f"internal_dim={internal_dim}, num_heads={num_heads}."
        )

    return internal_dim, internal_dim // num_heads


def _ensure_rank3(name: str, tensor: Tensor, *, last_dim: int) -> None:
    if tensor.ndim != 3:
        raise AttentionExecutionError(
            f"{name} must have shape [batch, tokens, channels]; "
            f"received {tuple(tensor.shape)}."
        )
    if tensor.shape[-1] != last_dim:
        raise AttentionExecutionError(
            f"{name} has channel size {tensor.shape[-1]}, expected {last_dim}."
        )
    if tensor.device.type not in ("cpu", "cuda"):
        raise AttentionExecutionError(
            f"{name} must be on CPU or CUDA; received {tensor.device}."
        )
    if not tensor.dtype.is_floating_point:
        raise AttentionExecutionError(
            f"{name} must use a floating dtype; received {tensor.dtype}."
        )


def _ensure_qkv_compatible(q: Tensor, k: Tensor, v: Tensor) -> AttentionShape:
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4:
        raise AttentionExecutionError(
            "Projected q, k and v must have shape [B, H, N, D]; received "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}."
        )

    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise AttentionExecutionError("q, k and v batch sizes must match.")
    if q.shape[1] != k.shape[1] or q.shape[1] != v.shape[1]:
        raise AttentionExecutionError("q, k and v head counts must match.")
    if k.shape[-2] != v.shape[-2]:
        raise AttentionExecutionError("k and v token counts must match.")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise AttentionExecutionError("q, k and v head dimensions must match.")
    if q.device != k.device or q.device != v.device:
        raise AttentionExecutionError("q, k and v must be on the same device.")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise AttentionExecutionError("q, k and v must use the same dtype.")

    return AttentionShape(
        batch_size=int(q.shape[0]),
        num_heads=int(q.shape[1]),
        query_length=int(q.shape[-2]),
        key_value_length=int(k.shape[-2]),
        head_dim=int(q.shape[-1]),
    )


@contextlib.contextmanager
def _backend_context(
    policy: AttentionPolicy,
    *,
    device: torch.device,
) -> Iterator[None]:
    """Select the requested PyTorch SDPA backend."""

    if policy.backend == "math":
        with sdpa_kernel(SDPBackend.MATH):
            yield
        return

    if policy.require_flash:
        if device.type != "cuda":
            raise AttentionExecutionError(
                "Flash SDPA was required, but the input is not on CUDA. "
                "Use AttentionPolicy(backend='math', require_flash=False) for "
                "CPU tests."
            )
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            yield
        return

    # Automatic SDPA lets PyTorch select a supported fused implementation and
    # fall back only when the requested shape/mask cannot use it.
    yield


def _normalise_attention_mask(
    mask: Tensor,
    *,
    batch_size: int,
    num_heads: int,
    query_length: int,
    key_value_length: int,
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> Tensor:
    """Normalise an SDPA mask to a broadcastable 4-D representation.

    Accepted shapes are ``[Q, K]``, ``[B, Q, K]``, ``[B, 1, Q, K]`` and
    ``[B, H, Q, K]``.  Boolean masks follow PyTorch SDPA semantics: True means
    allowed, False means masked.
    """

    if mask.device != device:
        mask = mask.to(device=device, non_blocking=True)

    if mask.ndim == 2:
        if tuple(mask.shape) != (query_length, key_value_length):
            raise AttentionExecutionError(
                f"{name} shape {tuple(mask.shape)} does not match "
                f"[Q, K]=[{query_length}, {key_value_length}]."
            )
        mask = mask.unsqueeze(0).unsqueeze(0)
    elif mask.ndim == 3:
        if tuple(mask.shape) != (
            batch_size,
            query_length,
            key_value_length,
        ):
            raise AttentionExecutionError(
                f"{name} shape {tuple(mask.shape)} does not match "
                f"[B, Q, K]=[{batch_size}, {query_length}, "
                f"{key_value_length}]."
            )
        mask = mask.unsqueeze(1)
    elif mask.ndim == 4:
        expected_tail = (query_length, key_value_length)
        if tuple(mask.shape[-2:]) != expected_tail:
            raise AttentionExecutionError(
                f"{name} last dimensions {tuple(mask.shape[-2:])} do not "
                f"match [Q, K]={expected_tail}."
            )
        if mask.shape[0] not in (1, batch_size):
            raise AttentionExecutionError(
                f"{name} batch dimension must be 1 or {batch_size}; "
                f"received {mask.shape[0]}."
            )
        if mask.shape[1] not in (1, num_heads):
            raise AttentionExecutionError(
                f"{name} head dimension must be 1 or {num_heads}; "
                f"received {mask.shape[1]}."
            )
    else:
        raise AttentionExecutionError(
            f"{name} must have rank 2, 3 or 4; received rank {mask.ndim}. "
            + _BOOL_MASK_DESCRIPTION
        )

    if mask.dtype == torch.bool:
        return mask
    if not mask.dtype.is_floating_point:
        raise AttentionExecutionError(
            f"{name} must be bool or floating point; received {mask.dtype}."
        )
    return mask.to(dtype=dtype)


def valid_token_mask_to_sdpa(
    valid_token_mask: Tensor,
    *,
    query_length: int | None = None,
) -> Tensor:
    """Convert a Hugging Face-style valid-token mask to an SDPA mask.

    ``valid_token_mask`` has shape ``[B, K]`` and uses True/1 for valid keys.
    The returned boolean tensor has shape ``[B, 1, Q, K]`` and follows SDPA's
    True=allowed convention.
    """

    if valid_token_mask.ndim != 2:
        raise AttentionExecutionError(
            "valid_token_mask must have shape [B, K]; received "
            f"{tuple(valid_token_mask.shape)}."
        )
    if valid_token_mask.dtype != torch.bool:
        if not valid_token_mask.dtype.is_floating_point and valid_token_mask.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise AttentionExecutionError(
                "valid_token_mask must be boolean or numeric 0/1 values."
            )
        valid_token_mask = valid_token_mask != 0

    batch_size, key_length = map(int, valid_token_mask.shape)
    query_length = key_length if query_length is None else int(query_length)
    if query_length <= 0:
        raise AttentionExecutionError("query_length must be positive.")

    return valid_token_mask.view(batch_size, 1, 1, key_length).expand(
        batch_size,
        1,
        query_length,
        key_length,
    )


def _bool_mask_to_additive(mask: Tensor, *, dtype: torch.dtype) -> Tensor:
    zero = torch.zeros((), dtype=dtype, device=mask.device)
    negative_infinity = torch.full(
        (),
        float("-inf"),
        dtype=dtype,
        device=mask.device,
    )
    return torch.where(mask, zero, negative_infinity)


def _merge_attention_masks(
    first: Tensor | None,
    second: Tensor | None,
    *,
    dtype: torch.dtype,
) -> Tensor | None:
    if first is None:
        return second
    if second is None:
        return first

    if first.dtype == torch.bool and second.dtype == torch.bool:
        return first & second

    if first.dtype == torch.bool:
        first = _bool_mask_to_additive(first, dtype=dtype)
    else:
        first = first.to(dtype=dtype)

    if second.dtype == torch.bool:
        second = _bool_mask_to_additive(second, dtype=dtype)
    else:
        second = second.to(dtype=dtype)

    return first + second


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    policy: AttentionPolicy,
    attention_mask: AttentionMask = None,
    valid_key_mask: Tensor | None = None,
    dropout_p: float = 0.0,
    training: bool = False,
    is_causal: bool = False,
) -> Tensor:
    """Execute validated SDPA using the configured backend.

    This wrapper intentionally does not expose attention probabilities.  Fused
    Flash SDPA avoids materialising that matrix, which is the source of its
    memory advantage for the 2048-token 3D ViTs.
    """

    shape = _ensure_qkv_compatible(q, k, v)
    dropout_p = _validate_probability(dropout_p, name="dropout_p")

    if attention_mask is not None:
        attention_mask = _normalise_attention_mask(
            attention_mask,
            batch_size=shape.batch_size,
            num_heads=shape.num_heads,
            query_length=shape.query_length,
            key_value_length=shape.key_value_length,
            device=q.device,
            dtype=q.dtype,
            name="attention_mask",
        )

    key_mask: Tensor | None = None
    if valid_key_mask is not None:
        if valid_key_mask.shape != (shape.batch_size, shape.key_value_length):
            raise AttentionExecutionError(
                "valid_key_mask must have shape [B, K] equal to "
                f"[{shape.batch_size}, {shape.key_value_length}]; received "
                f"{tuple(valid_key_mask.shape)}."
            )
        key_mask = valid_token_mask_to_sdpa(
            valid_key_mask.to(device=q.device, non_blocking=True),
            query_length=shape.query_length,
        )

    merged_mask = _merge_attention_masks(
        attention_mask,
        key_mask,
        dtype=q.dtype,
    )

    if is_causal and merged_mask is not None:
        raise AttentionExecutionError(
            "PyTorch SDPA cannot receive both is_causal=True and an explicit "
            "attention mask in this wrapper. Build one combined explicit mask "
            "and call with is_causal=False."
        )

    effective_dropout = dropout_p if training else 0.0

    try:
        with _backend_context(policy, device=q.device):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=merged_mask,
                dropout_p=effective_dropout,
                is_causal=is_causal,
            )
    except (RuntimeError, NotImplementedError) as error:
        details = (
            f"policy={policy.label}, device={q.device}, dtype={q.dtype}, "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}, v={tuple(v.shape)}, "
            f"mask={None if merged_mask is None else tuple(merged_mask.shape)}, "
            f"causal={is_causal}, dropout={effective_dropout}"
        )
        raise AttentionExecutionError(
            "The configured SDPA backend could not execute this attention "
            f"operation ({details}). Original error: {error}"
        ) from error


class FusedSelfAttention(nn.Module):
    """Checkpoint-compatible multi-head self-attention using PyTorch SDPA."""

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        dropout_rate: float = 0.0,
        qkv_bias: bool = False,
        *,
        backend: AttentionBackendName = "sdpa",
        require_flash_sdpa: bool = True,
    ) -> None:
        super().__init__()

        hidden_size = int(hidden_size)
        num_heads = int(num_heads)
        _, head_dim = _validate_projection_dimensions(
            embedding_dim=hidden_size,
            num_heads=num_heads,
        )

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.dropout_rate = _validate_probability(
            dropout_rate,
            name="dropout_rate",
        )
        self.policy = AttentionPolicy(
            backend=backend,
            require_flash=require_flash_sdpa,
        )

        # Names intentionally match MONAI SABlock checkpoints.
        self.qkv = nn.Linear(
            hidden_size,
            hidden_size * 3,
            bias=bool(qkv_bias),
        )
        self.out_proj = nn.Linear(hidden_size, hidden_size)

    def _project_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, token_count, _ = x.shape
        qkv = self.qkv(x)
        qkv = qkv.reshape(
            batch_size,
            token_count,
            3,
            self.num_heads,
            self.head_dim,
        )
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)
        return q, k, v

    def forward(
        self,
        x: Tensor,
        *,
        attention_mask: AttentionMask = None,
        valid_token_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        _ensure_rank3("x", x, last_dim=self.hidden_size)

        q, k, v = self._project_qkv(x)
        output = scaled_dot_product_attention(
            q,
            k,
            v,
            policy=self.policy,
            attention_mask=attention_mask,
            valid_key_mask=valid_token_mask,
            dropout_p=self.dropout_rate,
            training=self.training,
            is_causal=is_causal,
        )

        output = output.transpose(1, 2).reshape(
            x.shape[0],
            x.shape[1],
            self.hidden_size,
        )
        return self.out_proj(output)

    def extra_repr(self) -> str:
        return (
            f"hidden_size={self.hidden_size}, num_heads={self.num_heads}, "
            f"head_dim={self.head_dim}, dropout={self.dropout_rate}, "
            f"backend={self.policy.label}"
        )

    def runtime_info(self) -> AttentionRuntimeInfo:
        return AttentionRuntimeInfo(
            policy=self.policy.label,
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout=self.dropout_rate,
            parameter_count=sum(parameter.numel() for parameter in self.parameters()),
        )


class ProjectedSDPAAttention(nn.Module):
    """SegVol-compatible projected self/cross attention implemented with SDPA.

    The projection names and shapes match the original SegVol ``Attention``
    class, including its optional embedding downsampling.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        downsample_rate: int = 1,
        *,
        dropout_rate: float = 0.0,
        backend: AttentionBackendName = "sdpa",
        require_flash_sdpa: bool = True,
    ) -> None:
        super().__init__()

        embedding_dim = int(embedding_dim)
        num_heads = int(num_heads)
        downsample_rate = int(downsample_rate)
        if downsample_rate <= 0:
            raise AttentionConfigurationError("downsample_rate must be positive.")
        if embedding_dim % downsample_rate != 0:
            raise AttentionConfigurationError(
                "embedding_dim must be divisible by downsample_rate: "
                f"embedding_dim={embedding_dim}, "
                f"downsample_rate={downsample_rate}."
            )

        internal_dim = embedding_dim // downsample_rate
        internal_dim, head_dim = _validate_projection_dimensions(
            embedding_dim=embedding_dim,
            internal_dim=internal_dim,
            num_heads=num_heads,
        )

        self.embedding_dim = embedding_dim
        self.internal_dim = internal_dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.downsample_rate = downsample_rate
        self.dropout_rate = _validate_probability(
            dropout_rate,
            name="dropout_rate",
        )
        self.policy = AttentionPolicy(
            backend=backend,
            require_flash=require_flash_sdpa,
        )

        # Names intentionally match the original SegVol decoder checkpoint.
        self.q_proj = nn.Linear(embedding_dim, internal_dim)
        self.k_proj = nn.Linear(embedding_dim, internal_dim)
        self.v_proj = nn.Linear(embedding_dim, internal_dim)
        self.out_proj = nn.Linear(internal_dim, embedding_dim)

    def _separate_heads(self, x: Tensor) -> Tensor:
        batch_size, token_count, _ = x.shape
        x = x.reshape(
            batch_size,
            token_count,
            self.num_heads,
            self.head_dim,
        )
        return x.transpose(1, 2)

    def _recombine_heads(self, x: Tensor) -> Tensor:
        batch_size, _, token_count, _ = x.shape
        return x.transpose(1, 2).reshape(
            batch_size,
            token_count,
            self.internal_dim,
        )

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        *,
        attention_mask: AttentionMask = None,
        valid_key_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        _ensure_rank3("q", q, last_dim=self.embedding_dim)
        _ensure_rank3("k", k, last_dim=self.embedding_dim)
        _ensure_rank3("v", v, last_dim=self.embedding_dim)

        if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
            raise AttentionExecutionError("q, k and v batch sizes must match.")
        if k.shape[1] != v.shape[1]:
            raise AttentionExecutionError("k and v token counts must match.")
        if q.device != k.device or q.device != v.device:
            raise AttentionExecutionError("q, k and v must be on the same device.")
        raw_dtypes_match = q.dtype == k.dtype and q.dtype == v.dtype
        if not raw_dtypes_match and not torch.is_autocast_enabled(q.device.type):
            raise AttentionExecutionError("q, k and v must use the same dtype.")

        projected_q = self._separate_heads(self.q_proj(q))
        projected_k = self._separate_heads(self.k_proj(k))
        projected_v = self._separate_heads(self.v_proj(v))
        # Residual connections and positional encodings in the SegVol
        # two-way transformer can promote only one of q/k/v before entering
        # this module. Under autocast, each Linear projection deliberately
        # establishes the common compute dtype. Validate the projected tensors
        # through scaled_dot_product_attention instead of rejecting the safe
        # pre-projection mixture. Outside autocast the strict raw-dtype contract
        # above remains unchanged.

        output = scaled_dot_product_attention(
            projected_q,
            projected_k,
            projected_v,
            policy=self.policy,
            attention_mask=attention_mask,
            valid_key_mask=valid_key_mask,
            dropout_p=self.dropout_rate,
            training=self.training,
            is_causal=is_causal,
        )
        output = self._recombine_heads(output)
        return self.out_proj(output)

    def extra_repr(self) -> str:
        return (
            f"embedding_dim={self.embedding_dim}, "
            f"internal_dim={self.internal_dim}, "
            f"num_heads={self.num_heads}, head_dim={self.head_dim}, "
            f"downsample_rate={self.downsample_rate}, "
            f"dropout={self.dropout_rate}, backend={self.policy.label}"
        )

    def runtime_info(self) -> AttentionRuntimeInfo:
        return AttentionRuntimeInfo(
            policy=self.policy.label,
            hidden_size=self.embedding_dim,
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            dropout=self.dropout_rate,
            parameter_count=sum(parameter.numel() for parameter in self.parameters()),
        )


# Alias kept intentionally close to the original SegVol class name.  New code
# should prefer ProjectedSDPAAttention for clarity.
SegVolAttention = ProjectedSDPAAttention


def attention_parameter_names(module: nn.Module) -> tuple[str, ...]:
    """Return sorted parameter names for checkpoint compatibility tests."""

    return tuple(sorted(name for name, _ in module.named_parameters()))


def estimate_naive_attention_score_memory(
    *,
    batch_size: int,
    num_heads: int,
    query_length: int,
    key_value_length: int,
    dtype: torch.dtype,
) -> int:
    """Estimate bytes used only by a naive materialised score matrix."""

    shape = AttentionShape(
        batch_size=int(batch_size),
        num_heads=int(num_heads),
        query_length=int(query_length),
        key_value_length=int(key_value_length),
        head_dim=1,
    )
    if min(
        shape.batch_size,
        shape.num_heads,
        shape.query_length,
        shape.key_value_length,
    ) <= 0:
        raise AttentionConfigurationError(
            "All attention memory dimensions must be positive."
        )
    return shape.naive_score_bytes(dtype)


def _manual_reference_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    *,
    mask: Tensor | None = None,
) -> Tensor:
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if mask is not None:
        if mask.dtype == torch.bool:
            scores = scores.masked_fill(~mask, float("-inf"))
        else:
            scores = scores + mask
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, v)


def run_cpu_self_test() -> dict[str, object]:
    """Run deterministic CPU checks without requiring an ASPIRE 2A GPU."""

    torch.manual_seed(7)

    self_attention = FusedSelfAttention(
        hidden_size=32,
        num_heads=4,
        dropout_rate=0.0,
        qkv_bias=True,
        backend="math",
        require_flash_sdpa=False,
    )
    self_attention.eval()

    x = torch.randn(2, 9, 32, requires_grad=True)
    valid = torch.tensor(
        [
            [1, 1, 1, 1, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )
    output = self_attention(x, valid_token_mask=valid)
    if output.shape != x.shape:
        raise AssertionError(f"Unexpected self-attention shape {output.shape}.")
    output.square().mean().backward()
    if x.grad is None or not torch.isfinite(x.grad).all():
        raise AssertionError("Self-attention backward produced invalid gradients.")

    projected = ProjectedSDPAAttention(
        embedding_dim=32,
        num_heads=4,
        downsample_rate=2,
        backend="math",
        require_flash_sdpa=False,
    )
    projected.eval()
    q = torch.randn(2, 3, 32, requires_grad=True)
    k = torch.randn(2, 11, 32, requires_grad=True)
    v = torch.randn(2, 11, 32, requires_grad=True)
    cross_output = projected(q, k, v)
    if cross_output.shape != q.shape:
        raise AssertionError(
            f"Unexpected projected-attention shape {cross_output.shape}."
        )
    cross_output.sum().backward()
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.grad is None or not torch.isfinite(tensor.grad).all():
            raise AssertionError(f"{name} gradient is missing or non-finite.")

    # SegVol residual/position additions can produce FP32 q/k alongside BF16 v
    # inside a BF16 autocast region. The three projections must establish one
    # common attention dtype without weakening the non-autocast input contract.
    mixed_q = torch.randn(1, 3, 32, dtype=torch.float32, requires_grad=True)
    mixed_k = torch.randn(1, 5, 32, dtype=torch.float32, requires_grad=True)
    mixed_v = torch.randn(1, 5, 32, dtype=torch.bfloat16, requires_grad=True)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        mixed_output = projected(mixed_q, mixed_k, mixed_v)
    if mixed_output.dtype != torch.bfloat16:
        raise AssertionError(
            f"Autocast projected attention returned {mixed_output.dtype}."
        )
    mixed_output.float().sum().backward()
    for name, tensor in (("mixed_q", mixed_q), ("mixed_k", mixed_k), ("mixed_v", mixed_v)):
        if tensor.grad is None or not torch.isfinite(tensor.grad).all():
            raise AssertionError(
                f"{name} autocast gradient is missing or non-finite."
            )

    # Low-level SDPA parity with explicit attention on a small tensor.
    raw_q = torch.randn(1, 2, 5, 8)
    raw_k = torch.randn(1, 2, 7, 8)
    raw_v = torch.randn(1, 2, 7, 8)
    mask = torch.ones(1, 1, 5, 7, dtype=torch.bool)
    mask[..., -1] = False
    actual = scaled_dot_product_attention(
        raw_q,
        raw_k,
        raw_v,
        policy=AttentionPolicy(backend="math", require_flash=False),
        attention_mask=mask,
    )
    expected = _manual_reference_attention(raw_q, raw_k, raw_v, mask=mask)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)

    expected_self_names = (
        "out_proj.bias",
        "out_proj.weight",
        "qkv.bias",
        "qkv.weight",
    )
    actual_self_names = attention_parameter_names(self_attention)
    if actual_self_names != expected_self_names:
        raise AssertionError(
            "FusedSelfAttention checkpoint keys changed: "
            f"{actual_self_names}."
        )

    expected_projected_names = (
        "k_proj.bias",
        "k_proj.weight",
        "out_proj.bias",
        "out_proj.weight",
        "q_proj.bias",
        "q_proj.weight",
        "v_proj.bias",
        "v_proj.weight",
    )
    actual_projected_names = attention_parameter_names(projected)
    if actual_projected_names != expected_projected_names:
        raise AssertionError(
            "ProjectedSDPAAttention checkpoint keys changed: "
            f"{actual_projected_names}."
        )

    m3d_score_bytes = estimate_naive_attention_score_memory(
        batch_size=1,
        num_heads=12,
        query_length=2049,
        key_value_length=2049,
        dtype=torch.bfloat16,
    )

    return {
        "status": "passed",
        "self_attention_output_shape": list(output.shape),
        "projected_attention_output_shape": list(cross_output.shape),
        "self_attention_parameter_names": list(actual_self_names),
        "projected_attention_parameter_names": list(actual_projected_names),
        "m3d_naive_score_bytes_per_layer_bf16": m3d_score_bytes,
        "m3d_naive_score_mib_per_layer_bf16": m3d_score_bytes / (1024**2),
    }


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic CPU attention and checkpoint-key tests.",
    )
    return parser


def main() -> None:
    arguments = _build_argument_parser().parse_args()
    if not arguments.self_test:
        raise SystemExit("Pass --self-test to run the module validation.")

    result = run_cpu_self_test()
    for field in sorted(result):
        print(f"{field}: {result[field]}")


if __name__ == "__main__":
    main()
