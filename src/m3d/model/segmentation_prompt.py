"""Language-to-segmentation prompt projection for M3D-Modernized.

The original M3D segmentation path uses the hidden state that predicts each
``[SEG]`` token, averages those states when more than one segmentation token is
present, and projects the result into SegVol's prompt embedding dimension::

    Phi-3 last hidden state
        -> select hidden state immediately before every [SEG] token
        -> average per sample
        -> Linear -> ReLU -> Linear -> Dropout
        -> SegVol text prompt embedding

This module preserves that mathematical behaviour while removing the original
per-sample Python loop, ``Tensor.tolist()``, hard-coded ``.cuda()`` allocation,
and the need to request hidden states from every language-model layer.

The two image encoders remain independent.  This module only connects Phi-3 to
SegVol's prompt encoder; it neither combines nor shares image-encoder weights.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Mapping

import torch
from torch import Tensor, nn

from m3d.config import SegmentationConfig


DEFAULT_SEGMENTATION_PROJECTOR_DROPOUT: Final[float] = 0.1


class SegmentationPromptError(RuntimeError):
    """Base error for language-to-segmentation prompt handling."""


class SegmentationPromptConfigurationError(SegmentationPromptError, ValueError):
    """Raised when projector dimensions or options are invalid."""


class SegmentationPromptInputError(SegmentationPromptError, ValueError):
    """Raised when language hidden states and token tensors are incompatible."""


class SegmentationTokenAlignment(str, Enum):
    """How a ``[SEG]`` token is aligned to a causal-LM hidden state.

    ``NEXT_TOKEN`` reproduces the published M3D implementation.  For a
    segmentation token at token position ``j``, it selects hidden state
    ``j - 1`` because that hidden state predicts the token at ``j``.

    ``TOKEN_POSITION`` selects hidden state ``j`` itself.  It is provided for
    controlled experiments but is not the default M3D behaviour.
    """

    NEXT_TOKEN = "next_token"
    TOKEN_POSITION = "token_position"


@dataclass(frozen=True, slots=True)
class SegmentationPromptOutput:
    """Projected SegVol prompts and their auditable extraction metadata."""

    prompt_embeddings: Tensor
    pooled_language_states: Tensor
    aligned_token_mask: Tensor
    segmentation_token_counts: Tensor
    alignment: SegmentationTokenAlignment

    def __post_init__(self) -> None:
        if self.prompt_embeddings.ndim != 2:
            raise SegmentationPromptInputError(
                "prompt_embeddings must have shape [batch, prompt_dim], got "
                f"{tuple(self.prompt_embeddings.shape)}"
            )
        if self.pooled_language_states.ndim != 2:
            raise SegmentationPromptInputError(
                "pooled_language_states must have shape [batch, hidden], got "
                f"{tuple(self.pooled_language_states.shape)}"
            )
        if self.aligned_token_mask.ndim != 2:
            raise SegmentationPromptInputError(
                "aligned_token_mask must have shape [batch, sequence], got "
                f"{tuple(self.aligned_token_mask.shape)}"
            )
        if self.segmentation_token_counts.ndim != 1:
            raise SegmentationPromptInputError(
                "segmentation_token_counts must have shape [batch], got "
                f"{tuple(self.segmentation_token_counts.shape)}"
            )

        batch_size = int(self.prompt_embeddings.shape[0])
        if int(self.pooled_language_states.shape[0]) != batch_size:
            raise SegmentationPromptInputError(
                "prompt and pooled-state batch dimensions do not match"
            )
        if int(self.aligned_token_mask.shape[0]) != batch_size:
            raise SegmentationPromptInputError(
                "prompt and token-mask batch dimensions do not match"
            )
        if int(self.segmentation_token_counts.shape[0]) != batch_size:
            raise SegmentationPromptInputError(
                "prompt and token-count batch dimensions do not match"
            )

    @property
    def batch_size(self) -> int:
        return int(self.prompt_embeddings.shape[0])

    @property
    def prompt_dim(self) -> int:
        return int(self.prompt_embeddings.shape[1])

    @property
    def language_hidden_size(self) -> int:
        return int(self.pooled_language_states.shape[1])


@dataclass(frozen=True, slots=True)
class SegmentationPromptBuildReport:
    """Serializable description of the language-to-SegVol projector."""

    language_hidden_size: int
    prompt_embed_dim: int
    dropout: float
    alignment: str
    require_exactly_one_token: bool
    parameter_count: int
    trainable_parameter_count: int
    state_dict_keys: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "language_hidden_size": self.language_hidden_size,
            "prompt_embed_dim": self.prompt_embed_dim,
            "dropout": self.dropout,
            "alignment": self.alignment,
            "require_exactly_one_token": self.require_exactly_one_token,
            "parameter_count": self.parameter_count,
            "trainable_parameter_count": self.trainable_parameter_count,
            "state_dict_keys": list(self.state_dict_keys),
        }


def _validate_positive_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SegmentationPromptConfigurationError(
            f"{name} must be a positive integer, got {value!r}"
        )
    return value


def _normalise_alignment(
    value: SegmentationTokenAlignment | str,
) -> SegmentationTokenAlignment:
    if isinstance(value, SegmentationTokenAlignment):
        return value
    try:
        return SegmentationTokenAlignment(str(value))
    except ValueError as exc:
        allowed = ", ".join(item.value for item in SegmentationTokenAlignment)
        raise SegmentationPromptConfigurationError(
            f"Unknown segmentation-token alignment {value!r}; expected one of: "
            f"{allowed}"
        ) from exc


def build_aligned_segmentation_token_mask(
    input_ids: Tensor,
    *,
    segmentation_token_id: int,
    attention_mask: Tensor | None = None,
    alignment: SegmentationTokenAlignment | str = SegmentationTokenAlignment.NEXT_TOKEN,
) -> Tensor:
    """Build a boolean mask over hidden-state positions used as SegVol prompts.

    Args:
        input_ids: Token IDs with shape ``[batch, sequence]``.
        segmentation_token_id: Vocabulary ID assigned to ``[SEG]``.
        attention_mask: Optional valid-token mask with the same shape.  ``True``
            or non-zero means the token is valid.
        alignment: ``next_token`` reproduces M3D's one-position left shift.

    Returns:
        Boolean tensor with shape ``[batch, sequence]`` whose ``True`` entries
        select the language hidden states to pool.
    """

    if input_ids.ndim != 2:
        raise SegmentationPromptInputError(
            f"input_ids must have shape [batch, sequence], got {tuple(input_ids.shape)}"
        )
    if input_ids.dtype == torch.bool or input_ids.is_floating_point():
        raise SegmentationPromptInputError(
            f"input_ids must use an integer dtype, got {input_ids.dtype}"
        )
    if isinstance(segmentation_token_id, bool) or not isinstance(
        segmentation_token_id, int
    ):
        raise SegmentationPromptInputError(
            "segmentation_token_id must be an integer vocabulary ID"
        )
    if segmentation_token_id < 0:
        raise SegmentationPromptInputError(
            f"segmentation_token_id cannot be negative, got {segmentation_token_id}"
        )

    valid_tokens: Tensor | None = None
    if attention_mask is not None:
        if attention_mask.shape != input_ids.shape:
            raise SegmentationPromptInputError(
                "attention_mask must have the same shape as input_ids; got "
                f"{tuple(attention_mask.shape)} and {tuple(input_ids.shape)}"
            )
        if attention_mask.device != input_ids.device:
            raise SegmentationPromptInputError(
                "attention_mask and input_ids must be on the same device"
            )
        valid_tokens = attention_mask.to(dtype=torch.bool)

    token_mask = input_ids.eq(segmentation_token_id)
    if valid_tokens is not None:
        token_mask = token_mask & valid_tokens

    resolved_alignment = _normalise_alignment(alignment)
    if resolved_alignment is SegmentationTokenAlignment.TOKEN_POSITION:
        return token_mask

    # Reproduce the original M3D code without creating a device-specific tensor:
    #
    #   seg_token_mask = input_ids[:, 1:] == seg_token_id
    #   seg_token_mask = cat([seg_token_mask, zeros(..., 1)], dim=1)
    #
    # The hidden state immediately before [SEG] is the causal state that predicts
    # [SEG].  ``new_zeros`` preserves device and dtype without hard-coded .cuda().
    if int(input_ids.shape[1]) == 0:
        return token_mask
    return torch.cat(
        (
            token_mask[:, 1:],
            token_mask.new_zeros((token_mask.shape[0], 1)),
        ),
        dim=1,
    )


def pool_segmentation_language_states(
    last_hidden_state: Tensor,
    aligned_token_mask: Tensor,
    *,
    require_exactly_one_token: bool = False,
) -> tuple[Tensor, Tensor]:
    """Vectorially average selected language hidden states per batch item.

    This function has no per-sample Python loop and does not call ``.tolist()``
    or ``.item()`` in its successful hot path.
    """

    if last_hidden_state.ndim != 3:
        raise SegmentationPromptInputError(
            "last_hidden_state must have shape [batch, sequence, hidden], got "
            f"{tuple(last_hidden_state.shape)}"
        )
    if not last_hidden_state.is_floating_point():
        raise SegmentationPromptInputError(
            "last_hidden_state must be floating point, got "
            f"{last_hidden_state.dtype}"
        )
    if aligned_token_mask.ndim != 2:
        raise SegmentationPromptInputError(
            "aligned_token_mask must have shape [batch, sequence], got "
            f"{tuple(aligned_token_mask.shape)}"
        )
    if aligned_token_mask.shape != last_hidden_state.shape[:2]:
        raise SegmentationPromptInputError(
            "aligned_token_mask must match the first two hidden-state dimensions; "
            f"got mask={tuple(aligned_token_mask.shape)}, hidden="
            f"{tuple(last_hidden_state.shape)}"
        )
    if aligned_token_mask.device != last_hidden_state.device:
        raise SegmentationPromptInputError(
            "aligned_token_mask and last_hidden_state must be on the same device"
        )

    mask = aligned_token_mask.to(dtype=torch.bool)
    counts = mask.sum(dim=1)

    # torch._assert is compatible with graph capture and avoids a Python-side
    # Tensor.tolist() branch in the normal forward path.
    torch._assert(
        torch.all(counts > 0),
        "Every segmentation sample must contain at least one valid [SEG] token.",
    )
    if require_exactly_one_token:
        torch._assert(
            torch.all(counts == 1),
            "Every segmentation sample must contain exactly one valid [SEG] token.",
        )

    weights = mask.to(dtype=last_hidden_state.dtype)
    pooled = torch.bmm(
        weights.unsqueeze(1),
        last_hidden_state,
    ).squeeze(1)
    pooled = pooled / counts.to(dtype=last_hidden_state.dtype).unsqueeze(1)
    return pooled, counts


class SegmentationPromptProjector(nn.Sequential):
    """Project Phi-3 segmentation states into SegVol prompt embeddings.

    Subclassing ``nn.Sequential`` is intentional.  When an instance is assigned
    to ``model.seg_projector``, the resulting legacy-compatible state keys are::

        seg_projector.0.weight
        seg_projector.0.bias
        seg_projector.2.weight
        seg_projector.2.bias

    This matches the original M3D ``nn.Sequential`` layout.
    """

    language_hidden_size: int
    prompt_embed_dim: int
    dropout_probability: float
    alignment: SegmentationTokenAlignment
    require_exactly_one_token: bool

    def __init__(
        self,
        language_hidden_size: int,
        prompt_embed_dim: int,
        *,
        dropout: float = DEFAULT_SEGMENTATION_PROJECTOR_DROPOUT,
        alignment: SegmentationTokenAlignment | str = (
            SegmentationTokenAlignment.NEXT_TOKEN
        ),
        require_exactly_one_token: bool = False,
    ) -> None:
        language_hidden_size = _validate_positive_int(
            language_hidden_size,
            name="language_hidden_size",
        )
        prompt_embed_dim = _validate_positive_int(
            prompt_embed_dim,
            name="prompt_embed_dim",
        )
        if not 0.0 <= float(dropout) < 1.0:
            raise SegmentationPromptConfigurationError(
                f"dropout must be in [0, 1), got {dropout!r}"
            )

        super().__init__(
            nn.Linear(language_hidden_size, language_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(language_hidden_size, prompt_embed_dim),
            nn.Dropout(float(dropout)),
        )

        self.language_hidden_size = language_hidden_size
        self.prompt_embed_dim = prompt_embed_dim
        self.dropout_probability = float(dropout)
        self.alignment = _normalise_alignment(alignment)
        self.require_exactly_one_token = bool(require_exactly_one_token)

    def forward(self, hidden_states: Tensor) -> Tensor:
        if hidden_states.ndim not in (2, 3):
            raise SegmentationPromptInputError(
                "SegmentationPromptProjector expects [batch, hidden] or "
                f"[batch, tokens, hidden], got {tuple(hidden_states.shape)}"
            )
        if int(hidden_states.shape[-1]) != self.language_hidden_size:
            raise SegmentationPromptInputError(
                "Language hidden size does not match the projector: expected "
                f"{self.language_hidden_size}, got {int(hidden_states.shape[-1])}"
            )
        if not hidden_states.is_floating_point():
            raise SegmentationPromptInputError(
                f"hidden_states must be floating point, got {hidden_states.dtype}"
            )
        return super().forward(hidden_states)

    def extract_and_project(
        self,
        *,
        last_hidden_state: Tensor,
        input_ids: Tensor,
        segmentation_token_id: int,
        attention_mask: Tensor | None = None,
    ) -> SegmentationPromptOutput:
        """Extract, pool and project all ``[SEG]`` prompts in a homogeneous batch."""

        if input_ids.device != last_hidden_state.device:
            raise SegmentationPromptInputError(
                "input_ids and last_hidden_state must be on the same device"
            )
        if input_ids.shape != last_hidden_state.shape[:2]:
            raise SegmentationPromptInputError(
                "input_ids must match the hidden-state batch and sequence "
                f"dimensions; got input_ids={tuple(input_ids.shape)}, hidden="
                f"{tuple(last_hidden_state.shape)}"
            )

        aligned_mask = build_aligned_segmentation_token_mask(
            input_ids,
            segmentation_token_id=segmentation_token_id,
            attention_mask=attention_mask,
            alignment=self.alignment,
        )
        pooled_states, counts = pool_segmentation_language_states(
            last_hidden_state,
            aligned_mask,
            require_exactly_one_token=self.require_exactly_one_token,
        )
        prompts = self(pooled_states)

        return SegmentationPromptOutput(
            prompt_embeddings=prompts,
            pooled_language_states=pooled_states,
            aligned_token_mask=aligned_mask,
            segmentation_token_counts=counts,
            alignment=self.alignment,
        )

    def freeze(self) -> None:
        self.requires_grad_(False)

    def unfreeze(self) -> None:
        self.requires_grad_(True)

    def build_report(self) -> SegmentationPromptBuildReport:
        parameters = tuple(self.parameters())
        return SegmentationPromptBuildReport(
            language_hidden_size=self.language_hidden_size,
            prompt_embed_dim=self.prompt_embed_dim,
            dropout=self.dropout_probability,
            alignment=self.alignment.value,
            require_exactly_one_token=self.require_exactly_one_token,
            parameter_count=sum(int(parameter.numel()) for parameter in parameters),
            trainable_parameter_count=sum(
                int(parameter.numel())
                for parameter in parameters
                if parameter.requires_grad
            ),
            state_dict_keys=tuple(sorted(self.state_dict())),
        )


def build_segmentation_prompt_projector(
    *,
    language_hidden_size: int,
    segmentation_config: SegmentationConfig,
    dropout: float = DEFAULT_SEGMENTATION_PROJECTOR_DROPOUT,
    alignment: SegmentationTokenAlignment | str = SegmentationTokenAlignment.NEXT_TOKEN,
    require_exactly_one_token: bool = False,
) -> SegmentationPromptProjector:
    """Build the legacy-compatible Phi-3 -> SegVol prompt projector."""

    if not segmentation_config.enabled:
        raise SegmentationPromptConfigurationError(
            "Cannot build a segmentation prompt projector when segmentation is disabled"
        )

    projector = SegmentationPromptProjector(
        language_hidden_size=language_hidden_size,
        prompt_embed_dim=segmentation_config.prompt_embed_dim,
        dropout=dropout,
        alignment=alignment,
        require_exactly_one_token=require_exactly_one_token,
    )
    return projector


def validate_segmentation_prompt_contract(
    *,
    projector: SegmentationPromptProjector,
    language_hidden_size: int,
    segvol_prompt_embed_dim: int,
) -> None:
    """Ensure Phi-3, projector and SegVol dimensions agree before allocation."""

    errors: list[str] = []
    if projector.language_hidden_size != int(language_hidden_size):
        errors.append(
            "projector language_hidden_size="
            f"{projector.language_hidden_size} but Phi-3 hidden_size="
            f"{language_hidden_size}"
        )
    if projector.prompt_embed_dim != int(segvol_prompt_embed_dim):
        errors.append(
            "projector prompt_embed_dim="
            f"{projector.prompt_embed_dim} but SegVol embed_dim="
            f"{segvol_prompt_embed_dim}"
        )
    if errors:
        raise SegmentationPromptConfigurationError(
            "Invalid language/segmentation prompt contract:\n- "
            + "\n- ".join(errors)
        )


def _legacy_loop_reference(
    *,
    last_hidden_state: Tensor,
    input_ids: Tensor,
    segmentation_token_id: int,
) -> tuple[Tensor, Tensor]:
    """Small reference used only by the self-test to mirror original M3D."""

    mask = input_ids[:, 1:].eq(segmentation_token_id)
    mask = torch.cat((mask, mask.new_zeros((mask.shape[0], 1))), dim=1)

    pooled: list[Tensor] = []
    counts: list[Tensor] = []
    for index in range(int(last_hidden_state.shape[0])):
        selected = last_hidden_state[index][mask[index]]
        counts.append(mask[index].sum())
        pooled.append(selected.mean(dim=0))
    return torch.stack(pooled, dim=0), torch.stack(counts, dim=0)


def _run_self_test() -> Mapping[str, Any]:
    torch.manual_seed(7)

    batch_size = 3
    sequence_length = 8
    hidden_size = 12
    prompt_dim = 6
    segmentation_token_id = 99

    input_ids = torch.tensor(
        [
            [1, 4, 5, 99, 6, 7, 0, 0],
            [1, 99, 4, 5, 99, 7, 8, 0],
            [1, 3, 4, 5, 6, 99, 7, 8],
        ],
        dtype=torch.long,
    )
    attention_mask = input_ids.ne(0)
    hidden = torch.randn(
        batch_size,
        sequence_length,
        hidden_size,
        requires_grad=True,
    )

    projector = SegmentationPromptProjector(
        hidden_size,
        prompt_dim,
        dropout=0.0,
        alignment=SegmentationTokenAlignment.NEXT_TOKEN,
    )
    projector.train()

    output = projector.extract_and_project(
        last_hidden_state=hidden,
        input_ids=input_ids,
        segmentation_token_id=segmentation_token_id,
        attention_mask=attention_mask,
    )
    legacy_pooled, legacy_counts = _legacy_loop_reference(
        last_hidden_state=hidden,
        input_ids=input_ids,
        segmentation_token_id=segmentation_token_id,
    )

    if not torch.allclose(output.pooled_language_states, legacy_pooled):
        raise AssertionError("Vectorised segmentation-state pooling differs from legacy M3D")
    if not torch.equal(output.segmentation_token_counts, legacy_counts):
        raise AssertionError("Vectorised segmentation-token counts differ from legacy M3D")

    loss = output.prompt_embeddings.square().mean()
    loss.backward()
    if hidden.grad is None or not torch.isfinite(hidden.grad).all():
        raise AssertionError("Backward did not produce finite language-state gradients")
    if not all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in projector.parameters()
    ):
        raise AssertionError("Backward did not produce finite projector gradients")

    expected_keys = (
        "0.bias",
        "0.weight",
        "2.bias",
        "2.weight",
    )
    actual_keys = tuple(sorted(projector.state_dict()))
    if actual_keys != expected_keys:
        raise AssertionError(
            f"Legacy projector keys changed: expected {expected_keys}, got {actual_keys}"
        )

    token_position_mask = build_aligned_segmentation_token_mask(
        input_ids,
        segmentation_token_id=segmentation_token_id,
        attention_mask=attention_mask,
        alignment=SegmentationTokenAlignment.TOKEN_POSITION,
    )
    if not torch.equal(
        token_position_mask,
        input_ids.eq(segmentation_token_id) & attention_mask,
    ):
        raise AssertionError("TOKEN_POSITION alignment is incorrect")

    missing_token_detected = False
    try:
        pool_segmentation_language_states(
            hidden.detach(),
            torch.zeros(
                batch_size,
                sequence_length,
                dtype=torch.bool,
            ),
        )
    except (AssertionError, RuntimeError):
        missing_token_detected = True
    if not missing_token_detected:
        raise AssertionError("Missing [SEG] tokens were not rejected")

    report = projector.build_report()
    return {
        "status": "passed",
        "legacy_numerical_equivalence": True,
        "vectorised_multiple_seg_tokens": True,
        "prompt_shape": list(output.prompt_embeddings.shape),
        "pooled_state_shape": list(output.pooled_language_states.shape),
        "segmentation_token_counts": output.segmentation_token_counts.tolist(),
        "checkpoint_keys": list(actual_keys),
        "missing_token_detected": missing_token_detected,
        "build_report": report.to_dict(),
    }


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run a CPU numerical and checkpoint-contract self-test.",
    )
    arguments = parser.parse_args()
    if not arguments.self_test:
        parser.error("No action requested. Use --self-test.")
    print(json.dumps(_run_self_test(), indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
