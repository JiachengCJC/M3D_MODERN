"""Safe, auditable pretrained-weight loading for M3D-Modernized.

This module loads *initial model weights*.  It is intentionally separate from
training-state checkpoints, which later save optimizer, scheduler, sampler and
RNG state through PyTorch Distributed Checkpoint.

Supported M3D sources
---------------------
* Main M3D-CLIP ViT files such as ``pretrained_ViT.bin``.
* M3D projector files containing keys such as
  ``model.mm_projector.projector.0.weight`` and
  ``model.embed_tokens.weight``.
* Full SegVol files containing keys such as
  ``model.image_encoder.*``, ``model.prompt_encoder.*`` and
  ``model.mask_decoder.*``; ``model.text_encoder.*`` is ignored.
* Plain PyTorch state dictionaries, common ``state_dict`` wrappers and
  ``safetensors`` files.

The two image encoders remain separate modules.  Loading identical source
weights into both encoders would still create independent parameters, but the
normal M3D configuration loads the Main ViT from M3D-CLIP and the SegVol ViT
from the full SegVol checkpoint.

Security and correctness
------------------------
* PyTorch files are opened with ``weights_only=True``.
* Pickled arbitrary Python objects are never intentionally loaded.
* Every matched tensor is checked for exact shape compatibility.
* Ambiguous suffix matches are rejected.
* Strict mode requires every target tensor to be present.
* Reports include source fingerprints, renamed keys, missing keys, ignored
  source tensors and shape mismatches.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Final, TypeAlias

import torch
from torch import Tensor, nn

from m3d.model.vit3d import (
    ViT3DEncoder,
    ViT3DTower,
    assert_independent_encoders,
)


TensorMap: TypeAlias = dict[str, Tensor]
PathLike: TypeAlias = str | os.PathLike[str]

_SUPPORTED_SUFFIXES: Final[tuple[str, ...]] = (
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".safetensors",
)

# Wrappers added by DDP, PEFT, Transformers or a full model container.  They
# are removed recursively while constructing possible key variants.
_GENERIC_WRAPPER_PREFIXES: Final[tuple[str, ...]] = (
    "module.",
    "_orig_mod.",
    "base_model.model.",
    "base_model.",
)

# Well-known outer mapping keys used by training frameworks.
_OUTER_STATE_KEYS: Final[tuple[str, ...]] = (
    "state_dict",
    "model_state_dict",
    "module_state_dict",
    "weights",
    "params",
    "model",
    "module",
)


class CheckpointError(RuntimeError):
    """Base error for pretrained checkpoint handling."""


class CheckpointFormatError(CheckpointError):
    """Raised when a file is not a supported tensor checkpoint."""


class CheckpointCompatibilityError(CheckpointError):
    """Raised when checkpoint tensors cannot safely populate a target module."""


class ComponentKind(str, Enum):
    """Known M3D component layouts in legacy checkpoints."""

    GENERIC = "generic"
    MAIN_VISION = "main_vision"
    SEGMENTATION_VISION = "segmentation_vision"
    PROJECTOR = "projector"
    SEGMENTATION_MODULE = "segmentation_module"


_COMPONENT_PREFIXES: Final[Mapping[ComponentKind, tuple[str, ...]]] = {
    ComponentKind.GENERIC: ("",),
    ComponentKind.MAIN_VISION: (
        "",
        "vision_tower.vision_tower.",
        "vision_tower.",
        "model.vision_tower.vision_tower.",
        "model.vision_tower.",
        "model.model.vision_tower.vision_tower.",
        "model.model.vision_tower.",
    ),
    ComponentKind.SEGMENTATION_VISION: (
        "",
        "image_encoder.",
        "seg_module.image_encoder.",
        "model.image_encoder.",
        "model.seg_module.image_encoder.",
        "model.model.seg_module.image_encoder.",
        "segmentation_module.image_encoder.",
        "model.segmentation_module.image_encoder.",
    ),
    ComponentKind.PROJECTOR: (
        "",
        "mm_projector.",
        "model.mm_projector.",
        "model.model.mm_projector.",
        "projector.",
    ),
    ComponentKind.SEGMENTATION_MODULE: (
        "",
        "seg_module.",
        "model.seg_module.",
        "model.model.seg_module.",
        "segmentation_module.",
        "model.segmentation_module.",
        # The published SegVol checkpoint stores the module directly beneath
        # ``model.*`` after excluding ``model.text_encoder.*``.
        "model.",
    ),
}


@dataclass(frozen=True, slots=True)
class CheckpointSource:
    """Metadata and tensor content read from one checkpoint file."""

    path: Path
    sha256: str
    size_bytes: int
    tensors: Mapping[str, Tensor]
    outer_container: str | None

    @property
    def tensor_count(self) -> int:
        return len(self.tensors)

    @property
    def total_elements(self) -> int:
        return sum(int(tensor.numel()) for tensor in self.tensors.values())

    @property
    def total_tensor_bytes(self) -> int:
        return sum(
            int(tensor.numel()) * int(tensor.element_size())
            for tensor in self.tensors.values()
        )


@dataclass(frozen=True, slots=True)
class ShapeMismatch:
    """One source tensor that matched by name but not by shape."""

    target_key: str
    source_key: str
    target_shape: tuple[int, ...]
    source_shape: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class KeyMatch:
    """One deterministic source-to-target key mapping."""

    target_key: str
    source_key: str
    priority: int


@dataclass(frozen=True, slots=True)
class ExtractionResult:
    """Result of mapping source tensors to one target module."""

    state_dict: Mapping[str, Tensor]
    matches: tuple[KeyMatch, ...]
    missing_target_keys: tuple[str, ...]
    ignored_source_keys: tuple[str, ...]
    shape_mismatches: tuple[ShapeMismatch, ...]
    ambiguous_target_keys: Mapping[str, tuple[str, ...]]

    @property
    def matched_tensor_count(self) -> int:
        return len(self.state_dict)


@dataclass(frozen=True, slots=True)
class ModuleLoadReport:
    """Serializable report returned after loading one module."""

    module_name: str
    component: str
    checkpoint_path: str
    checkpoint_sha256: str
    checkpoint_size_bytes: int
    source_tensor_count: int
    matched_tensor_count: int
    matched_parameter_elements: int
    missing_target_keys: tuple[str, ...]
    ignored_source_keys: tuple[str, ...]
    shape_mismatches: tuple[ShapeMismatch, ...]
    ambiguous_target_keys: Mapping[str, tuple[str, ...]]
    renamed_keys: tuple[tuple[str, str], ...]
    assign: bool
    strict: bool

    @property
    def successful(self) -> bool:
        return not (
            self.missing_target_keys
            or self.shape_mismatches
            or self.ambiguous_target_keys
        )

    def to_dict(self, *, max_key_examples: int | None = 50) -> dict[str, Any]:
        """Convert the report to JSON-compatible data.

        ``max_key_examples`` limits potentially very long key lists in files
        written for humans.  Use ``None`` for every key.
        """

        def limit(values: Sequence[Any]) -> list[Any]:
            if max_key_examples is None:
                return list(values)
            return list(values[: max(0, int(max_key_examples))])

        ignored = limit(self.ignored_source_keys)
        missing = limit(self.missing_target_keys)
        renamed = limit(self.renamed_keys)
        mismatches = limit(self.shape_mismatches)
        ambiguous_items = list(self.ambiguous_target_keys.items())
        if max_key_examples is not None:
            ambiguous_items = ambiguous_items[: max(0, int(max_key_examples))]

        return {
            "module_name": self.module_name,
            "component": self.component,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_sha256": self.checkpoint_sha256,
            "checkpoint_size_bytes": self.checkpoint_size_bytes,
            "source_tensor_count": self.source_tensor_count,
            "matched_tensor_count": self.matched_tensor_count,
            "matched_parameter_elements": self.matched_parameter_elements,
            "successful": self.successful,
            "strict": self.strict,
            "assign": self.assign,
            "missing_target_key_count": len(self.missing_target_keys),
            "missing_target_keys": missing,
            "ignored_source_key_count": len(self.ignored_source_keys),
            "ignored_source_keys": ignored,
            "shape_mismatch_count": len(self.shape_mismatches),
            "shape_mismatches": [dataclasses.asdict(item) for item in mismatches],
            "ambiguous_target_key_count": len(self.ambiguous_target_keys),
            "ambiguous_target_keys": {
                key: list(values) for key, values in ambiguous_items
            },
            "renamed_key_count": len(self.renamed_keys),
            "renamed_keys": [
                {"source": source, "target": target}
                for source, target in renamed
            ],
        }


@dataclass(frozen=True, slots=True)
class EmbeddingLoadReport:
    """Result of restoring token embeddings from an M3D projector file."""

    checkpoint_path: str
    checkpoint_sha256: str
    source_key: str
    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    copied_rows: int
    copied_mode: str


@dataclass(frozen=True, slots=True)
class DualVisionLoadReport:
    """Reports for the two independent image encoders."""

    main_vision: ModuleLoadReport | None
    segmentation_vision: ModuleLoadReport | None
    shared_parameter_count_after_load: int


# ---------------------------------------------------------------------------
# File reading and fingerprinting
# ---------------------------------------------------------------------------


def sha256_file(path: PathLike, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    """Return a streaming SHA-256 fingerprint for ``path``."""

    resolved = Path(path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _is_tensor_mapping(value: Any) -> bool:
    return isinstance(value, Mapping) and any(
        isinstance(item, Tensor) for item in value.values()
    )


def _flatten_tensor_leaves(
    value: Mapping[Any, Any],
    *,
    prefix: str = "",
) -> TensorMap:
    """Flatten nested string-keyed mappings, retaining only tensor leaves."""

    flattened: TensorMap = {}
    for raw_key, child in value.items():
        if not isinstance(raw_key, str):
            continue
        key = f"{prefix}.{raw_key}" if prefix else raw_key
        if isinstance(child, Tensor):
            flattened[key] = child
        elif isinstance(child, Mapping):
            flattened.update(_flatten_tensor_leaves(child, prefix=key))
    return flattened


def _unwrap_state_container(value: Any) -> tuple[Mapping[Any, Any], str | None]:
    """Unwrap common checkpoint containers without touching arbitrary objects."""

    if not isinstance(value, Mapping):
        raise CheckpointFormatError(
            "Checkpoint root must be a mapping containing tensors; received "
            f"{type(value).__name__}."
        )

    current: Mapping[Any, Any] = value
    path: list[str] = []

    # Prefer explicit state_dict containers even when metadata is present.
    for outer_key in _OUTER_STATE_KEYS:
        child = current.get(outer_key)
        if isinstance(child, Mapping) and _flatten_tensor_leaves(child):
            current = child
            path.append(outer_key)
            break

    # A nested wrapper may itself contain another standard wrapper.
    while True:
        next_item: tuple[str, Mapping[Any, Any]] | None = None
        for outer_key in _OUTER_STATE_KEYS:
            child = current.get(outer_key)
            if isinstance(child, Mapping) and _flatten_tensor_leaves(child):
                next_item = (outer_key, child)
                break
        if next_item is None:
            break
        outer_key, child = next_item
        current = child
        path.append(outer_key)

    return current, ".".join(path) if path else None


def _load_safetensors(path: Path) -> TensorMap:
    try:
        from safetensors.torch import load_file
    except ImportError as error:  # pragma: no cover - dependency is in requirements
        raise CheckpointFormatError(
            "A .safetensors file was provided, but the safetensors package is "
            "not installed. Install the reviewed requirements.txt."
        ) from error
    return dict(load_file(str(path), device="cpu"))


def _load_torch_weights(path: Path) -> Any:
    """Open a PyTorch checkpoint in weights-only mode."""

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:  # pragma: no cover - only for older PyTorch
        raise CheckpointFormatError(
            "This project requires a PyTorch version whose torch.load supports "
            "weights_only=True. The ASPIRE 2A profile uses PyTorch 2.6.0."
        ) from error
    except Exception as error:
        raise CheckpointFormatError(
            f"Failed to read weights-only checkpoint {path}: {error}"
        ) from error


def read_checkpoint(path: PathLike) -> CheckpointSource:
    """Read one tensor checkpoint safely onto CPU."""

    resolved = Path(path).expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise CheckpointFormatError(f"Checkpoint is not a regular file: {resolved}")
    if resolved.stat().st_size <= 0:
        raise CheckpointFormatError(f"Checkpoint is empty: {resolved}")

    lower_name = resolved.name.lower()
    if not any(lower_name.endswith(suffix) for suffix in _SUPPORTED_SUFFIXES):
        raise CheckpointFormatError(
            f"Unsupported checkpoint extension for {resolved}. Supported: "
            f"{', '.join(_SUPPORTED_SUFFIXES)}."
        )

    if lower_name.endswith(".safetensors"):
        tensors = _load_safetensors(resolved)
        outer_container = None
    else:
        loaded = _load_torch_weights(resolved)
        unwrapped, outer_container = _unwrap_state_container(loaded)
        tensors = _flatten_tensor_leaves(unwrapped)

    if not tensors:
        raise CheckpointFormatError(
            f"Checkpoint {resolved} contains no tensor state dictionary."
        )

    invalid_keys = [key for key in tensors if not key or "\x00" in key]
    if invalid_keys:
        raise CheckpointFormatError(
            f"Checkpoint contains invalid state keys: {invalid_keys[:5]!r}."
        )

    for key, tensor in tensors.items():
        if tensor.layout != torch.strided:
            raise CheckpointFormatError(
                f"Checkpoint tensor {key!r} uses unsupported layout {tensor.layout}."
            )

    return CheckpointSource(
        path=resolved,
        sha256=sha256_file(resolved),
        size_bytes=int(resolved.stat().st_size),
        tensors=tensors,
        outer_container=outer_container,
    )


# ---------------------------------------------------------------------------
# Deterministic key mapping
# ---------------------------------------------------------------------------


def _strip_repeated_prefix(key: str, prefix: str) -> str:
    while prefix and key.startswith(prefix):
        key = key[len(prefix) :]
    return key


def _generic_variants(source_key: str) -> set[str]:
    variants = {source_key}
    changed = True
    while changed:
        changed = False
        for candidate in tuple(variants):
            for prefix in _GENERIC_WRAPPER_PREFIXES:
                if candidate.startswith(prefix):
                    stripped = candidate[len(prefix) :]
                    if stripped not in variants:
                        variants.add(stripped)
                        changed = True
    return variants


def _candidate_target_keys(
    source_key: str,
    *,
    component: ComponentKind,
    target_keys: set[str],
) -> dict[str, int]:
    """Return possible target keys and deterministic priorities.

    Lower priority values are preferred:
      0: exact key after generic wrappers;
      10+: explicit component-prefix removal;
      100: unambiguous dotted suffix match.
    """

    candidates: dict[str, int] = {}
    generic = _generic_variants(source_key)

    for variant in generic:
        if variant in target_keys:
            candidates[variant] = min(candidates.get(variant, 10_000), 0)

        for prefix_index, component_prefix in enumerate(
            _COMPONENT_PREFIXES[component]
        ):
            if not component_prefix:
                continue
            if variant.startswith(component_prefix):
                stripped = variant[len(component_prefix) :]
                stripped = _strip_repeated_prefix(stripped, "module.")
                if stripped in target_keys:
                    priority = 10 + prefix_index
                    candidates[stripped] = min(
                        candidates.get(stripped, 10_000), priority
                    )

    # Final compatibility path for unusual full-model prefixes.  Requiring a
    # dotted suffix avoids matching a target key against a partial token.
    suffix_matches = [
        target_key
        for target_key in target_keys
        if source_key.endswith("." + target_key)
    ]
    if len(suffix_matches) == 1:
        target_key = suffix_matches[0]
        candidates[target_key] = min(candidates.get(target_key, 10_000), 100)

    return candidates


def _source_key_allowed(source_key: str, component: ComponentKind) -> bool:
    """Filter known non-model branches before suffix matching."""

    generic_variants = _generic_variants(source_key)
    if any(
        variant.startswith("model.text_encoder.")
        or variant.startswith("text_encoder.")
        or ".text_encoder." in variant
        for variant in generic_variants
    ):
        return False

    if component is ComponentKind.SEGMENTATION_MODULE:
        allowed_roots = (
            "image_encoder.",
            "prompt_encoder.",
            "mask_decoder.",
            "model.image_encoder.",
            "model.prompt_encoder.",
            "model.mask_decoder.",
            "seg_module.",
            "model.seg_module.",
            "segmentation_module.",
        )
        return any(
            variant.startswith(allowed_roots) or variant in {"pixel_mean", "pixel_std"}
            for variant in generic_variants
        )

    return True


def extract_component_state_dict(
    source: CheckpointSource,
    module: nn.Module,
    *,
    component: ComponentKind,
    allowed_missing: Iterable[str] = (),
) -> ExtractionResult:
    """Map a full checkpoint onto one module without loading it yet."""

    target_state = module.state_dict()
    target_keys = set(target_state)
    allowed_missing_set = set(allowed_missing)

    proposals: MutableMapping[str, list[tuple[int, str]]] = defaultdict(list)
    source_candidates: dict[str, dict[str, int]] = {}

    for source_key in sorted(source.tensors):
        if not _source_key_allowed(source_key, component):
            continue
        candidates = _candidate_target_keys(
            source_key,
            component=component,
            target_keys=target_keys,
        )
        source_candidates[source_key] = candidates
        for target_key, priority in candidates.items():
            proposals[target_key].append((priority, source_key))

    selected: TensorMap = {}
    matches: list[KeyMatch] = []
    mismatches: list[ShapeMismatch] = []
    ambiguous: dict[str, tuple[str, ...]] = {}
    used_source_keys: set[str] = set()

    for target_key in sorted(target_keys):
        options = proposals.get(target_key, [])
        if not options:
            continue
        best_priority = min(priority for priority, _ in options)
        best_sources = sorted(
            source_key
            for priority, source_key in options
            if priority == best_priority
        )
        if len(best_sources) != 1:
            ambiguous[target_key] = tuple(best_sources)
            continue

        source_key = best_sources[0]
        source_tensor = source.tensors[source_key]
        target_tensor = target_state[target_key]
        if tuple(source_tensor.shape) != tuple(target_tensor.shape):
            mismatches.append(
                ShapeMismatch(
                    target_key=target_key,
                    source_key=source_key,
                    target_shape=tuple(int(item) for item in target_tensor.shape),
                    source_shape=tuple(int(item) for item in source_tensor.shape),
                )
            )
            used_source_keys.add(source_key)
            continue

        selected[target_key] = source_tensor
        matches.append(
            KeyMatch(
                target_key=target_key,
                source_key=source_key,
                priority=best_priority,
            )
        )
        used_source_keys.add(source_key)

    missing = tuple(
        sorted(
            target_keys
            - set(selected)
            - set(ambiguous)
            - {item.target_key for item in mismatches}
            - allowed_missing_set
        )
    )
    ignored = tuple(sorted(set(source.tensors) - used_source_keys))

    return ExtractionResult(
        state_dict=selected,
        matches=tuple(matches),
        missing_target_keys=missing,
        ignored_source_keys=ignored,
        shape_mismatches=tuple(mismatches),
        ambiguous_target_keys=ambiguous,
    )


def _format_compatibility_error(
    *,
    module_name: str,
    component: ComponentKind,
    source: CheckpointSource,
    extraction: ExtractionResult,
) -> str:
    lines = [
        "Checkpoint is incompatible with the target module.",
        f"  module: {module_name}",
        f"  component: {component.value}",
        f"  checkpoint: {source.path}",
        f"  sha256: {source.sha256}",
        f"  matched tensors: {extraction.matched_tensor_count}",
    ]
    if extraction.missing_target_keys:
        lines.append(
            "  missing target keys:\n    "
            + "\n    ".join(extraction.missing_target_keys[:30])
        )
    if extraction.shape_mismatches:
        lines.append("  shape mismatches:")
        for item in extraction.shape_mismatches[:30]:
            lines.append(
                "    "
                f"{item.source_key} -> {item.target_key}: "
                f"source={item.source_shape}, target={item.target_shape}"
            )
    if extraction.ambiguous_target_keys:
        lines.append("  ambiguous target keys:")
        for target_key, source_keys in list(
            extraction.ambiguous_target_keys.items()
        )[:30]:
            lines.append(f"    {target_key}: {list(source_keys)!r}")
    return "\n".join(lines)


def _auto_assign(module: nn.Module) -> bool:
    tensors = list(module.parameters(recurse=True)) + list(module.buffers(recurse=True))
    return bool(tensors) and any(tensor.device.type == "meta" for tensor in tensors)


def load_module_from_checkpoint(
    module: nn.Module,
    checkpoint: PathLike | CheckpointSource,
    *,
    component: ComponentKind = ComponentKind.GENERIC,
    module_name: str | None = None,
    strict: bool = True,
    assign: bool | None = None,
    allowed_missing: Iterable[str] = (),
) -> ModuleLoadReport:
    """Load one module from a direct or full-model checkpoint.

    Strictness applies to target completeness.  Source tensors belonging to
    other components are reported as ignored, not treated as an error.
    """

    source = checkpoint if isinstance(checkpoint, CheckpointSource) else read_checkpoint(checkpoint)
    display_name = module_name or module.__class__.__name__
    extraction = extract_component_state_dict(
        source,
        module,
        component=component,
        allowed_missing=allowed_missing,
    )

    incompatible = bool(
        extraction.shape_mismatches or extraction.ambiguous_target_keys
    )
    if strict:
        incompatible = incompatible or bool(extraction.missing_target_keys)
    if incompatible:
        raise CheckpointCompatibilityError(
            _format_compatibility_error(
                module_name=display_name,
                component=component,
                source=source,
                extraction=extraction,
            )
        )

    use_assign = _auto_assign(module) if assign is None else bool(assign)
    load_result = module.load_state_dict(
        dict(extraction.state_dict),
        strict=False,
        assign=use_assign,
    )

    # ``load_state_dict`` is a second line of defence.  Missing keys explicitly
    # allowed by the caller remain acceptable.
    unexpected_after_load = tuple(sorted(load_result.unexpected_keys))
    allowed_missing_set = set(allowed_missing)
    missing_after_load = tuple(
        sorted(set(load_result.missing_keys) - allowed_missing_set)
    )
    if unexpected_after_load or (strict and missing_after_load):
        raise CheckpointCompatibilityError(
            "PyTorch load_state_dict reported an unexpected incompatibility "
            f"for {display_name}: missing={missing_after_load}, "
            f"unexpected={unexpected_after_load}."
        )

    matched_elements = sum(
        int(tensor.numel()) for tensor in extraction.state_dict.values()
    )
    renamed = tuple(
        (match.source_key, match.target_key)
        for match in extraction.matches
        if match.source_key != match.target_key
    )

    return ModuleLoadReport(
        module_name=display_name,
        component=component.value,
        checkpoint_path=str(source.path),
        checkpoint_sha256=source.sha256,
        checkpoint_size_bytes=source.size_bytes,
        source_tensor_count=source.tensor_count,
        matched_tensor_count=extraction.matched_tensor_count,
        matched_parameter_elements=matched_elements,
        missing_target_keys=extraction.missing_target_keys,
        ignored_source_keys=extraction.ignored_source_keys,
        shape_mismatches=extraction.shape_mismatches,
        ambiguous_target_keys=extraction.ambiguous_target_keys,
        renamed_keys=renamed,
        assign=use_assign,
        strict=bool(strict),
    )


# ---------------------------------------------------------------------------
# M3D-specific loading helpers
# ---------------------------------------------------------------------------


def load_main_vision_checkpoint(
    encoder: ViT3DEncoder | ViT3DTower,
    checkpoint: PathLike | CheckpointSource,
    *,
    strict: bool = True,
) -> ModuleLoadReport:
    """Load the independent Main M3D-CLIP vision encoder."""

    target = encoder.vision_tower if isinstance(encoder, ViT3DTower) else encoder
    return load_module_from_checkpoint(
        target,
        checkpoint,
        component=ComponentKind.MAIN_VISION,
        module_name="main_vision_encoder",
        strict=strict,
    )


def load_segmentation_vision_checkpoint(
    encoder: ViT3DEncoder,
    checkpoint: PathLike | CheckpointSource,
    *,
    strict: bool = True,
) -> ModuleLoadReport:
    """Extract and load only SegVol's independent image encoder."""

    return load_module_from_checkpoint(
        encoder,
        checkpoint,
        component=ComponentKind.SEGMENTATION_VISION,
        module_name="segmentation_vision_encoder",
        strict=strict,
    )


def load_projector_checkpoint(
    projector: nn.Module,
    checkpoint: PathLike | CheckpointSource,
    *,
    strict: bool = True,
) -> ModuleLoadReport:
    """Load ``mm_projector`` weights while ignoring saved token embeddings."""

    return load_module_from_checkpoint(
        projector,
        checkpoint,
        component=ComponentKind.PROJECTOR,
        module_name="multimodal_projector",
        strict=strict,
    )


def load_segmentation_module_checkpoint(
    segmentation_module: nn.Module,
    checkpoint: PathLike | CheckpointSource,
    *,
    strict: bool = True,
) -> ModuleLoadReport:
    """Load image encoder, prompt encoder and mask decoder from SegVol.

    Legacy ``model.text_encoder.*`` tensors are ignored because M3D supplies
    the text prompt from the language model's ``[SEG]`` hidden state.
    """

    return load_module_from_checkpoint(
        segmentation_module,
        checkpoint,
        component=ComponentKind.SEGMENTATION_MODULE,
        module_name="segmentation_module",
        strict=strict,
    )


def load_dual_vision_checkpoints(
    *,
    main_encoder: ViT3DEncoder | ViT3DTower,
    segmentation_encoder: ViT3DEncoder,
    main_checkpoint: PathLike | CheckpointSource | None,
    segmentation_checkpoint: PathLike | CheckpointSource | None,
    strict: bool = True,
) -> DualVisionLoadReport:
    """Load both encoders and verify that they still share no parameters."""

    main_module = (
        main_encoder.vision_tower
        if isinstance(main_encoder, ViT3DTower)
        else main_encoder
    )
    assert_independent_encoders(main_module, segmentation_encoder)

    main_report = (
        load_main_vision_checkpoint(main_encoder, main_checkpoint, strict=strict)
        if main_checkpoint is not None
        else None
    )
    segmentation_report = (
        load_segmentation_vision_checkpoint(
            segmentation_encoder,
            segmentation_checkpoint,
            strict=strict,
        )
        if segmentation_checkpoint is not None
        else None
    )

    # Loading with ``assign=True`` on meta modules replaces Parameters, so the
    # independence assertion must be repeated after loading.
    assert_independent_encoders(main_module, segmentation_encoder)
    main_parameter_ids = {id(parameter) for parameter in main_module.parameters()}
    seg_parameter_ids = {
        id(parameter) for parameter in segmentation_encoder.parameters()
    }
    shared_count = len(main_parameter_ids & seg_parameter_ids)

    return DualVisionLoadReport(
        main_vision=main_report,
        segmentation_vision=segmentation_report,
        shared_parameter_count_after_load=shared_count,
    )


# ---------------------------------------------------------------------------
# Token-embedding and LoRA helpers
# ---------------------------------------------------------------------------


def _find_unique_tensor_by_suffix(
    source: CheckpointSource,
    suffixes: Sequence[str],
) -> tuple[str, Tensor]:
    matches: list[tuple[str, Tensor]] = []
    for key, tensor in source.tensors.items():
        if any(key == suffix or key.endswith("." + suffix) for suffix in suffixes):
            matches.append((key, tensor))
    if not matches:
        raise CheckpointCompatibilityError(
            f"Checkpoint {source.path} does not contain any of: {list(suffixes)!r}."
        )
    if len(matches) > 1:
        exact = [item for item in matches if item[0] in suffixes]
        if len(exact) == 1:
            return exact[0]
        raise CheckpointCompatibilityError(
            "Checkpoint contains multiple possible token-embedding tensors: "
            f"{[key for key, _ in matches]!r}."
        )
    return matches[0]


def load_input_embeddings_from_projector_checkpoint(
    embedding: nn.Embedding,
    checkpoint: PathLike | CheckpointSource,
    *,
    num_new_tokens: int,
) -> EmbeddingLoadReport:
    """Restore input embeddings saved with the original M3D projector.

    This reproduces the original M3D behaviour:
    * equal shapes -> copy the complete embedding table;
    * source rows equal ``num_new_tokens`` -> copy only the final new-token rows.
    """

    source = checkpoint if isinstance(checkpoint, CheckpointSource) else read_checkpoint(checkpoint)
    source_key, source_weight = _find_unique_tensor_by_suffix(
        source,
        ("model.embed_tokens.weight", "embed_tokens.weight"),
    )
    target_weight = embedding.weight

    if source_weight.ndim != 2 or target_weight.ndim != 2:
        raise CheckpointCompatibilityError(
            "Input embedding weights must be rank-2 matrices: "
            f"source={tuple(source_weight.shape)}, target={tuple(target_weight.shape)}."
        )
    if source_weight.shape[1] != target_weight.shape[1]:
        raise CheckpointCompatibilityError(
            "Embedding dimensions differ: "
            f"source={source_weight.shape[1]}, target={target_weight.shape[1]}."
        )
    if num_new_tokens <= 0:
        raise CheckpointCompatibilityError("num_new_tokens must be positive.")
    if num_new_tokens > target_weight.shape[0]:
        raise CheckpointCompatibilityError(
            f"num_new_tokens={num_new_tokens} exceeds target vocabulary size "
            f"{target_weight.shape[0]}."
        )

    with torch.no_grad():
        source_cast = source_weight.to(
            device=target_weight.device,
            dtype=target_weight.dtype,
        )
        if tuple(source_weight.shape) == tuple(target_weight.shape):
            target_weight.copy_(source_cast)
            copied_rows = int(target_weight.shape[0])
            mode = "full_table"
        elif int(source_weight.shape[0]) == int(num_new_tokens):
            target_weight[-num_new_tokens:].copy_(source_cast)
            copied_rows = int(num_new_tokens)
            mode = "new_token_rows"
        else:
            raise CheckpointCompatibilityError(
                "Unexpected saved embedding shape. Expected either the complete "
                "current table or exactly num_new_tokens rows: "
                f"source={tuple(source_weight.shape)}, "
                f"target={tuple(target_weight.shape)}, "
                f"num_new_tokens={num_new_tokens}."
            )

    return EmbeddingLoadReport(
        checkpoint_path=str(source.path),
        checkpoint_sha256=source.sha256,
        source_key=source_key,
        source_shape=tuple(int(item) for item in source_weight.shape),
        target_shape=tuple(int(item) for item in target_weight.shape),
        copied_rows=copied_rows,
        copied_mode=mode,
    )


def load_peft_adapter(
    model: nn.Module,
    adapter_path: PathLike,
    *,
    is_trainable: bool = True,
    adapter_name: str = "default",
) -> nn.Module:
    """Attach a saved PEFT LoRA adapter after the base model is constructed."""

    resolved = Path(adapter_path).expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise CheckpointFormatError(
            f"PEFT adapter path must be a directory: {resolved}"
        )
    required = resolved / "adapter_config.json"
    if not required.is_file():
        raise CheckpointFormatError(
            f"PEFT adapter directory lacks adapter_config.json: {resolved}"
        )

    try:
        from peft import PeftModel
    except ImportError as error:  # pragma: no cover - dependency is reviewed
        raise CheckpointError(
            "PEFT is required to load a LoRA adapter. Install requirements.txt."
        ) from error

    return PeftModel.from_pretrained(
        model,
        str(resolved),
        adapter_name=adapter_name,
        is_trainable=is_trainable,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def write_load_report(
    report: ModuleLoadReport | DualVisionLoadReport | EmbeddingLoadReport,
    output_path: PathLike,
) -> Path:
    """Atomically write a JSON checkpoint-load report."""

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(report, ModuleLoadReport):
        payload: Any = report.to_dict(max_key_examples=None)
    elif isinstance(report, DualVisionLoadReport):
        payload = {
            "main_vision": (
                report.main_vision.to_dict(max_key_examples=None)
                if report.main_vision is not None
                else None
            ),
            "segmentation_vision": (
                report.segmentation_vision.to_dict(max_key_examples=None)
                if report.segmentation_vision is not None
                else None
            ),
            "shared_parameter_count_after_load": (
                report.shared_parameter_count_after_load
            ),
        }
    else:
        payload = dataclasses.asdict(report)

    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
        text=True,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    return destination


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


class _TinyVision(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.patch_embedding = nn.Linear(4, 8)
        self.norm = nn.LayerNorm(8)


class _TinyProjector(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(8, 6),
            nn.GELU(),
            nn.Linear(6, 6),
        )


class _TinySegmentation(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.image_encoder = _TinyVision()
        self.prompt_encoder = nn.Linear(8, 8)
        self.mask_decoder = nn.Linear(8, 1)


def _clone_state(module: nn.Module) -> TensorMap:
    return {
        key: tensor.detach().clone()
        for key, tensor in module.state_dict().items()
    }


def run_cpu_self_test() -> dict[str, Any]:
    """Exercise legacy key extraction, strict shape checks and embeddings."""

    torch.manual_seed(7)
    with tempfile.TemporaryDirectory(prefix="m3d-checkpoint-test-") as directory:
        root = Path(directory)

        source_vision = _TinyVision()
        direct_path = root / "pretrained_ViT.bin"
        torch.save(_clone_state(source_vision), direct_path)

        target_vision = _TinyVision()
        for parameter in target_vision.parameters():
            nn.init.zeros_(parameter)
        direct_report = load_module_from_checkpoint(
            target_vision,
            direct_path,
            component=ComponentKind.MAIN_VISION,
            strict=True,
        )
        for key, expected in source_vision.state_dict().items():
            if not torch.equal(target_vision.state_dict()[key], expected):
                raise AssertionError(f"Direct vision load failed for {key}.")

        source_projector = _TinyProjector()
        embedding_source = torch.randn(4, 6)
        projector_state = {
            f"model.mm_projector.{key}": value.detach().clone()
            for key, value in source_projector.state_dict().items()
        }
        projector_state["model.embed_tokens.weight"] = embedding_source
        projector_path = root / "mm_projector.bin"
        torch.save(projector_state, projector_path)

        target_projector = _TinyProjector()
        projector_report = load_projector_checkpoint(
            target_projector,
            projector_path,
            strict=True,
        )
        if not projector_report.renamed_keys:
            raise AssertionError("Projector prefix stripping was not exercised.")

        target_embedding = nn.Embedding(10, 6)
        old_rows = target_embedding.weight[:-4].detach().clone()
        embedding_report = load_input_embeddings_from_projector_checkpoint(
            target_embedding,
            projector_path,
            num_new_tokens=4,
        )
        if not torch.equal(target_embedding.weight[-4:], embedding_source):
            raise AssertionError("New token rows were not restored.")
        if not torch.equal(target_embedding.weight[:-4], old_rows):
            raise AssertionError("Existing token rows were unexpectedly changed.")

        source_segmentation = _TinySegmentation()
        seg_state = {
            f"model.{key}": value.detach().clone()
            for key, value in source_segmentation.state_dict().items()
        }
        seg_state["model.text_encoder.weight"] = torch.randn(3, 3)
        seg_path = root / "pytorch_model.bin"
        torch.save({"state_dict": seg_state, "epoch": 2}, seg_path)

        target_segmentation = _TinySegmentation()
        seg_report = load_segmentation_module_checkpoint(
            target_segmentation,
            seg_path,
            strict=True,
        )
        if "model.text_encoder.weight" not in seg_report.ignored_source_keys:
            raise AssertionError("Legacy text encoder was not ignored.")

        target_seg_encoder = _TinyVision()
        seg_encoder_report = load_module_from_checkpoint(
            target_seg_encoder,
            seg_path,
            component=ComponentKind.SEGMENTATION_VISION,
            strict=True,
        )

        bad_state = _clone_state(source_vision)
        bad_state["norm.weight"] = torch.randn(9)
        bad_path = root / "bad.bin"
        torch.save(bad_state, bad_path)
        shape_error_detected = False
        try:
            load_module_from_checkpoint(
                _TinyVision(),
                bad_path,
                component=ComponentKind.MAIN_VISION,
                strict=True,
            )
        except CheckpointCompatibilityError:
            shape_error_detected = True
        if not shape_error_detected:
            raise AssertionError("Shape mismatch was not rejected.")

        report_path = write_load_report(
            projector_report,
            root / "load_report.json",
        )
        if not report_path.is_file():
            raise AssertionError("Load report was not written.")

        return {
            "status": "passed",
            "direct_vision_matched": direct_report.matched_tensor_count,
            "projector_matched": projector_report.matched_tensor_count,
            "segmentation_module_matched": seg_report.matched_tensor_count,
            "segmentation_encoder_matched": (
                seg_encoder_report.matched_tensor_count
            ),
            "embedding_mode": embedding_report.copied_mode,
            "shape_error_detected": shape_error_detected,
            "weights_only_read": True,
        }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run deterministic CPU checkpoint-loading tests.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if not args.self_test:
        raise SystemExit("No action selected. Use --self-test.")
    print(json.dumps(run_cpu_self_test(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
