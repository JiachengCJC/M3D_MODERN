"""Export a completed distributed M3D checkpoint into portable artifacts.

Training checkpoints in :mod:`m3d.checkpointing` are deliberately optimised for
exact distributed resume.  They contain sharded/replicated model and optimizer
state under ``torch.distributed.checkpoint`` (DCP), plus scheduler, sampler and
rank-local RNG sidecars.  Those files are the correct training source of truth,
but they are not the most convenient format for inference or publication.

This module performs a separate, explicit export step:

1. build the exact M3D architecture on every rank;
2. wrap it with DDP or composable FSDP2;
3. load *model state only* from a completed DCP checkpoint;
4. gather a full CPU model state on rank 0;
5. write a sharded safetensors M3D bundle;
6. optionally save the PEFT adapter and/or a LoRA-merged Phi-3 model.

The two 3D image encoders remain separate throughout the process.  The exported
state contains independent ``vision_tower.vision_tower.*`` and
``seg_module.image_encoder.*`` parameter namespaces.

The command is intended to run under the same two-rank ``torchrun`` layout used
for training.  FSDP2 export never materialises the complete 4B model on GPU;
``get_model_state_dict(full_state_dict=True, cpu_offload=True)`` gathers the
portable state directly to rank-0 CPU memory.

Example::

    torchrun --standalone --nproc_per_node=2 -m m3d.export \
        --config configs/m3d_joint_finetune.yaml \
        --checkpoint outputs/m3d-phi3-finetune \
        --output-dir outputs/m3d-phi3-export \
        --format all

The final directory is written through an ``.incomplete-*`` staging directory
and atomically renamed only after all requested artifacts and hashes have been
created successfully.
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import inspect
import json
import os
import re
import shutil
import tempfile
import time
import traceback
import uuid
from collections.abc import Callable, Mapping, MutableMapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

import torch
import torch.distributed.checkpoint as dcp
from safetensors import safe_open
from safetensors.torch import save_file
from torch import Tensor, nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from .checkpointing import resolve_checkpoint_path
from .config import ExperimentConfig, load_config
from .distributed import (
    DistributedM3DModel,
    build_model_synchronously,
    prepare_distributed_model,
)
from .model.language import M3DLanguageModel, build_language_model
from .model.m3d import build_m3d_model
from .runtime import RuntimeContext, atomic_write_json, distributed_runtime

if TYPE_CHECKING:
    from .tokenization import TokenizerBundle


_EXPORT_STATE_VERSION = 1
_COMPLETION_FILE = "COMPLETED.json"
_TRAINER_STATE_FILE = "trainer_state.json"
_CHECKPOINT_CONFIG_FILE = "resolved_config.json"
_DCP_DIR = "dcp"

ExportFormat = Literal["bundle", "adapter", "merged", "all"]
DistributedStrategy = Literal["ddp", "fsdp2"]


class ExportError(RuntimeError):
    """Base class for portable-export failures."""


class ExportCompatibilityError(ExportError):
    """Raised when the requested architecture differs from the checkpoint."""


class ExportDependencyError(ExportError):
    """Raised when a required export dependency is unavailable."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _canonical_json(payload: Any) -> str:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _sha256_file(path: Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except FileNotFoundError as exc:
        raise ExportCompatibilityError(f"Required file is missing: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ExportCompatibilityError(f"Malformed JSON file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExportCompatibilityError(
            f"JSON root must be an object, got {type(payload).__name__}: {path}"
        )
    return payload


def _parse_byte_size(value: str | int) -> int:
    """Parse values such as ``4GB`` or ``4096MiB`` into bytes."""

    if isinstance(value, int) and not isinstance(value, bool):
        if value <= 0:
            raise ValueError("Byte size must be positive.")
        return value
    text = str(value).strip().upper().replace(" ", "")
    match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)(B|KB|MB|GB|TB|KIB|MIB|GIB|TIB)?", text)
    if match is None:
        raise ValueError(
            f"Invalid byte-size value {value!r}; examples: 4GB, 4096MiB."
        )
    amount = float(match.group(1))
    unit = match.group(2) or "B"
    multipliers = {
        "B": 1,
        "KB": 1000,
        "MB": 1000**2,
        "GB": 1000**3,
        "TB": 1000**4,
        "KIB": 1024,
        "MIB": 1024**2,
        "GIB": 1024**3,
        "TIB": 1024**4,
    }
    result = int(amount * multipliers[unit])
    if result <= 0:
        raise ValueError("Byte size must be positive.")
    return result


def _tensor_nbytes(tensor: Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _normalise_state_tensor(name: str, value: Any) -> Tensor:
    if not isinstance(value, Tensor):
        raise ExportError(
            f"Portable model state must contain tensors only; {name!r} is "
            f"{type(value).__name__}."
        )
    if value.layout is not torch.strided:
        raise ExportError(
            f"Tensor {name!r} has unsupported layout {value.layout}; full CPU "
            "state gathering should have produced strided tensors."
        )
    if value.is_meta:
        raise ExportError(f"Tensor {name!r} is still on the meta device.")
    return value.detach().to(device="cpu").contiguous()


def _architecture_contract_from_mapping(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Return the state-layout portion of an ExperimentConfig mapping.

    File locations, freezing choices, attention kernels and activation
    checkpointing do not alter parameter names/shapes, so they are deliberately
    excluded.  LoRA rank/targets, encoder dimensions and segmentation structure
    remain part of the contract.
    """

    model_raw = payload.get("model")
    if not isinstance(model_raw, Mapping):
        raise ExportCompatibilityError("Resolved config does not contain model mapping.")
    model = copy.deepcopy(dict(model_raw))

    for encoder_name in ("main_vision", "seg_vision"):
        encoder = model.get(encoder_name)
        if isinstance(encoder, MutableMapping):
            for key in (
                "checkpoint_path",
                "freeze",
                "unfreeze_last_n_layers",
                "attention_backend",
                "require_flash_sdpa",
                "activation_checkpoint_every_n_layers",
            ):
                encoder.pop(key, None)

    projector = model.get("projector")
    if isinstance(projector, MutableMapping):
        projector.pop("checkpoint_path", None)
        projector.pop("freeze", None)

    segmentation = model.get("segmentation")
    if isinstance(segmentation, MutableMapping):
        segmentation.pop("checkpoint_path", None)
        segmentation.pop("freeze_prompt_encoder", None)
        segmentation.pop("freeze_mask_decoder", None)
        segmentation.pop("dice_loss_weight", None)
        segmentation.pop("bce_loss_weight", None)

    lora = model.get("lora")
    if isinstance(lora, MutableMapping):
        lora.pop("adapter_checkpoint_path", None)

    return {
        "schema_version": payload.get("schema_version"),
        "model": model,
    }


def _validate_checkpoint_configuration(
    config: ExperimentConfig,
    checkpoint_path: Path,
) -> tuple[str, str]:
    saved = _read_json(checkpoint_path / _CHECKPOINT_CONFIG_FILE)
    saved_contract = _architecture_contract_from_mapping(saved)
    current_contract = _architecture_contract_from_mapping(config.to_dict())
    saved_hash = _sha256_payload(saved_contract)
    current_hash = _sha256_payload(current_contract)
    if saved_hash != current_hash:
        raise ExportCompatibilityError(
            "Current config does not reproduce the checkpoint model layout. "
            f"checkpoint_contract_sha256={saved_hash}, "
            f"current_contract_sha256={current_hash}. Use the same model and "
            "LoRA architecture that produced the checkpoint."
        )
    return saved_hash, current_hash


def _checkpoint_fingerprint(path: Path) -> str:
    entries: list[dict[str, Any]] = []
    for relative in (
        Path(_COMPLETION_FILE),
        Path(_TRAINER_STATE_FILE),
        Path(_CHECKPOINT_CONFIG_FILE),
        Path(_DCP_DIR) / ".metadata",
    ):
        candidate = path / relative
        if candidate.is_file():
            entries.append(
                {
                    "path": relative.as_posix(),
                    "size": candidate.stat().st_size,
                    "sha256": _sha256_file(candidate),
                }
            )
    dcp_dir = path / _DCP_DIR
    if not dcp_dir.is_dir():
        raise ExportCompatibilityError(f"Checkpoint has no DCP directory: {path}")
    for candidate in sorted(dcp_dir.iterdir(), key=lambda item: item.name):
        if candidate.name == ".metadata" or not candidate.is_file():
            continue
        entries.append(
            {
                "path": f"{_DCP_DIR}/{candidate.name}",
                "size": candidate.stat().st_size,
            }
        )
    return _sha256_payload(entries)


class _ModelOnlyDCPState(Stateful):
    """Load only the model part of a training DCP application state."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.options = StateDictOptions(
            full_state_dict=False,
            cpu_offload=False,
            strict=True,
            keep_submodule_prefixes=True,
            flatten_optimizer_state_dict=False,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": get_model_state_dict(
                self.model,
                options=self.options,
            )
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        model_state = state_dict.get("model")
        if not isinstance(model_state, dict):
            raise ExportCompatibilityError(
                "DCP application state does not contain a model dictionary."
            )
        incompatible = set_model_state_dict(
            self.model,
            model_state,
            options=self.options,
        )
        missing = list(getattr(incompatible, "missing_keys", ()))
        unexpected = list(getattr(incompatible, "unexpected_keys", ()))
        if missing or unexpected:
            raise ExportCompatibilityError(
                "DCP model state is not strict-compatible: "
                f"missing={missing}, unexpected={unexpected}."
            )


def load_model_only_checkpoint(
    distributed_model: DistributedM3DModel,
    checkpoint_path: Path,
) -> None:
    """Restore model tensors without constructing/restoring an optimizer."""

    completion = _read_json(checkpoint_path / _COMPLETION_FILE)
    if completion.get("status") != "complete":
        raise ExportCompatibilityError(
            f"Checkpoint is not marked complete: {checkpoint_path}"
        )
    state = _ModelOnlyDCPState(distributed_model.wrapped_model)
    try:
        dcp.load(
            {"application": state},
            checkpoint_id=checkpoint_path / _DCP_DIR,
        )
    except Exception as exc:
        raise ExportCompatibilityError(
            f"Could not load model state from {checkpoint_path / _DCP_DIR}."
        ) from exc


def gather_full_cpu_state(
    distributed_model: DistributedM3DModel,
    runtime: RuntimeContext,
) -> dict[str, Tensor]:
    """Gather a canonical full state on rank 0 and return empty state elsewhere."""

    options = StateDictOptions(
        full_state_dict=True,
        cpu_offload=True,
        strict=True,
        keep_submodule_prefixes=True,
        flatten_optimizer_state_dict=False,
    )
    try:
        gathered = get_model_state_dict(
            distributed_model.wrapped_model,
            options=options,
        )
    except Exception as exc:
        raise ExportError("Full CPU model-state gathering failed.") from exc

    if runtime.is_main_process:
        if not gathered:
            raise ExportError("Rank 0 received an empty full model state.")
        state: dict[str, Tensor] = {}
        for name, value in gathered.items():
            state[str(name)] = _normalise_state_tensor(str(name), value)
        return state
    if gathered:
        raise ExportError(
            "Non-zero rank unexpectedly received a full state despite CPU offload."
        )
    return {}


@dataclass(frozen=True, slots=True)
class SafetensorsShard:
    file: str
    tensor_count: int
    byte_count: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class SafetensorsExportReport:
    directory: str
    index_file: str
    tensor_count: int
    total_byte_count: int
    max_shard_byte_count: int
    shards: tuple[SafetensorsShard, ...]
    state_sha256: str

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["shards"] = [item.to_dict() for item in self.shards]
        return payload


def _plan_shards(
    state: Mapping[str, Tensor],
    *,
    max_shard_bytes: int,
) -> list[list[str]]:
    if max_shard_bytes <= 0:
        raise ValueError("max_shard_bytes must be positive.")
    names = sorted(state)
    if not names:
        raise ExportError("Cannot export an empty state dictionary.")

    shards: list[list[str]] = []
    current: list[str] = []
    current_bytes = 0
    for name in names:
        size = _tensor_nbytes(state[name])
        if current and current_bytes + size > max_shard_bytes:
            shards.append(current)
            current = []
            current_bytes = 0
        current.append(name)
        current_bytes += size
        # A single tensor larger than the requested limit must remain intact.
        if size > max_shard_bytes:
            shards.append(current)
            current = []
            current_bytes = 0
    if current:
        shards.append(current)
    return shards


def save_sharded_safetensors(
    state: Mapping[str, Tensor],
    destination: Path,
    *,
    basename: str,
    max_shard_bytes: int,
    metadata: Mapping[str, str] | None = None,
) -> SafetensorsExportReport:
    """Write a complete strict-loadable tensor mapping in bounded shards.

    Each tensor is cloned while constructing its shard.  This deliberately
    breaks tied-storage aliases because safetensors rejects duplicate storages.
    Keeping both state-dict names makes later strict ``load_state_dict`` simple;
    tied modules are re-established by the architecture itself.
    """

    destination.mkdir(parents=True, exist_ok=False)
    normalised = {
        str(name): _normalise_state_tensor(str(name), tensor)
        for name, tensor in state.items()
    }
    shard_plan = _plan_shards(normalised, max_shard_bytes=max_shard_bytes)
    shard_count = len(shard_plan)
    weight_map: dict[str, str] = {}
    shard_reports: list[SafetensorsShard] = []
    state_descriptor: list[dict[str, Any]] = []

    for shard_index, names in enumerate(shard_plan, start=1):
        filename = (
            f"{basename}.safetensors"
            if shard_count == 1
            else f"{basename}-{shard_index:05d}-of-{shard_count:05d}.safetensors"
        )
        path = destination / filename
        # Clone per shard rather than cloning the complete model at once.
        payload = {
            name: normalised[name].clone(memory_format=torch.contiguous_format)
            for name in names
        }
        save_metadata = {
            "format": "pt",
            "m3d_export_state_version": str(_EXPORT_STATE_VERSION),
        }
        if metadata is not None:
            save_metadata.update({str(key): str(value) for key, value in metadata.items()})
        save_file(payload, str(path), metadata=save_metadata)

        byte_count = int(path.stat().st_size)
        shard_reports.append(
            SafetensorsShard(
                file=filename,
                tensor_count=len(names),
                byte_count=byte_count,
                sha256=_sha256_file(path),
            )
        )
        for name in names:
            weight_map[name] = filename
            tensor = normalised[name]
            state_descriptor.append(
                {
                    "name": name,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "nbytes": _tensor_nbytes(tensor),
                }
            )
        del payload

    total_tensor_bytes = sum(_tensor_nbytes(tensor) for tensor in normalised.values())
    index_payload = {
        "metadata": {
            "format": "pt",
            "total_size": total_tensor_bytes,
            "tensor_count": len(normalised),
            "state_sha256": _sha256_payload(state_descriptor),
        },
        "weight_map": weight_map,
        "shards": [item.to_dict() for item in shard_reports],
    }
    index_path = destination / f"{basename}.safetensors.index.json"
    atomic_write_json(index_path, index_payload)

    return SafetensorsExportReport(
        directory=str(destination),
        index_file=index_path.name,
        tensor_count=len(normalised),
        total_byte_count=total_tensor_bytes,
        max_shard_byte_count=max_shard_bytes,
        shards=tuple(shard_reports),
        state_sha256=_sha256_payload(state_descriptor),
    )


def load_sharded_safetensors_for_test(directory: Path, *, basename: str) -> dict[str, Tensor]:
    """Small strict loader used by the self-test and later inference loader."""

    index = _read_json(directory / f"{basename}.safetensors.index.json")
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, Mapping):
        raise ExportError("Safetensors index has no weight_map.")
    files = sorted({str(value) for value in weight_map.values()})
    state: dict[str, Tensor] = {}
    for filename in files:
        path = directory / filename
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                if name in state:
                    raise ExportError(f"Duplicate tensor {name!r} across shards.")
                state[name] = handle.get_tensor(name)
    missing = sorted(set(weight_map) - set(state))
    unexpected = sorted(set(state) - set(weight_map))
    if missing or unexpected:
        raise ExportError(
            f"Safetensors index mismatch: missing={missing}, unexpected={unexpected}."
        )
    return state


def _extract_prefix_state(
    state: Mapping[str, Tensor],
    prefix: str,
    *,
    strip_prefix: bool,
    required: bool,
) -> dict[str, Tensor]:
    selected = {
        (name[len(prefix) :] if strip_prefix else name): tensor
        for name, tensor in state.items()
        if name.startswith(prefix)
    }
    if required and not selected:
        raise ExportError(f"No tensors found under required prefix {prefix!r}.")
    return selected


def _component_states(full_state: Mapping[str, Tensor]) -> dict[str, dict[str, Tensor]]:
    components = {
        "main_vision": _extract_prefix_state(
            full_state, "vision_tower.", strip_prefix=True, required=True
        ),
        "multimodal_projector": _extract_prefix_state(
            full_state, "mm_projector.", strip_prefix=True, required=True
        ),
    }
    segmentation_projector = _extract_prefix_state(
        full_state, "seg_projector.", strip_prefix=True, required=False
    )
    segvol = _extract_prefix_state(
        full_state, "seg_module.", strip_prefix=True, required=False
    )
    if bool(segmentation_projector) != bool(segvol):
        raise ExportError(
            "Exported model contains only one of seg_projector or seg_module."
        )
    if segmentation_projector:
        components["segmentation_projector"] = segmentation_projector
        components["segvol"] = segvol
    return components


def _assert_independent_exported_encoders(full_state: Mapping[str, Tensor]) -> None:
    main = _extract_prefix_state(
        full_state,
        "vision_tower.vision_tower.",
        strip_prefix=True,
        required=True,
    )
    seg = _extract_prefix_state(
        full_state,
        "seg_module.image_encoder.",
        strip_prefix=True,
        required=True,
    )
    if set(main) != set(seg):
        # CLS-token differences are expected; compare the common architecture
        # keys while requiring both namespaces to exist independently.
        common = set(main) & set(seg)
        if not common:
            raise ExportError("Main and SegVol encoder namespaces have no common keys.")
    for name in set(main) & set(seg):
        if main[name] is seg[name]:
            raise ExportError(
                f"Export gather aliased Main and SegVol tensor object {name!r}."
            )


def _build_cpu_language_from_full_state(
    config: ExperimentConfig,
    tokenizer_bundle: TokenizerBundle,
    full_state: Mapping[str, Tensor],
    *,
    cache_dir: Path | None,
    local_files_only: bool,
) -> tuple[M3DLanguageModel, dict[str, Any]]:
    """Build only Phi-3+PEFT on CPU and load its exported subtree strictly."""

    export_config = copy.deepcopy(config)
    export_config.optimization.checkpoint_language_model = False
    export_config.model.lora.adapter_checkpoint_path = None
    language, report = build_language_model(
        export_config,
        tokenizer_bundle,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        torch_dtype=torch.bfloat16,
    )
    language_state = _extract_prefix_state(
        full_state,
        "language_model.",
        strip_prefix=True,
        required=True,
    )
    incompatible = language.load_state_dict(language_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise ExportCompatibilityError(
            "Language subtree did not load strictly: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}."
        )
    language.eval()
    return language, report.to_dict()


def _save_peft_adapter(
    language: M3DLanguageModel,
    tokenizer_bundle: TokenizerBundle,
    destination: Path,
) -> dict[str, Any]:
    causal_lm = language.causal_lm
    peft_config = getattr(causal_lm, "peft_config", None)
    if not isinstance(peft_config, Mapping) or not peft_config:
        raise ExportCompatibilityError(
            "Adapter export was requested, but the loaded language model is not a "
            "PEFT model with active adapter configuration."
        )
    save = getattr(causal_lm, "save_pretrained", None)
    if not callable(save):
        raise ExportDependencyError("PEFT model does not provide save_pretrained().")
    destination.mkdir(parents=True, exist_ok=False)
    save(
        str(destination),
        safe_serialization=True,
    )
    tokenizer_bundle.save_pretrained(destination / "tokenizer")
    files = _directory_file_manifest(destination)
    return {
        "directory": str(destination),
        "adapter_names": sorted(str(name) for name in peft_config),
        "files": files,
    }


def _merge_and_save_language(
    language: M3DLanguageModel,
    tokenizer_bundle: TokenizerBundle,
    destination: Path,
    *,
    max_shard_size: str,
) -> dict[str, Any]:
    causal_lm = language.causal_lm
    merge = getattr(causal_lm, "merge_and_unload", None)
    if callable(merge):
        kwargs: dict[str, Any] = {}
        try:
            signature = inspect.signature(merge)
            if "safe_merge" in signature.parameters:
                kwargs["safe_merge"] = True
        except (TypeError, ValueError):
            pass
        merged = merge(**kwargs)
    else:
        # LoRA may have been disabled in the training architecture. In that case
        # the causal LM is already a normal Hugging Face model.
        merged = causal_lm

    if not isinstance(merged, nn.Module):
        raise ExportError("merge_and_unload() did not return an nn.Module.")
    configuration = getattr(merged, "config", None)
    if configuration is not None:
        setattr(configuration, "use_cache", True)
        setattr(configuration, "m3d_image_token_id", language.image_token_id)
        setattr(
            configuration,
            "m3d_segmentation_token_id",
            language.segmentation_token_id,
        )
        setattr(configuration, "m3d_visual_token_count", language.visual_token_count)

    save = getattr(merged, "save_pretrained", None)
    if not callable(save):
        raise ExportDependencyError(
            "Merged Phi-3 model does not provide save_pretrained()."
        )
    destination.mkdir(parents=True, exist_ok=False)
    save(
        str(destination),
        safe_serialization=True,
        max_shard_size=max_shard_size,
    )
    tokenizer_bundle.save_pretrained(destination / "tokenizer")
    atomic_write_json(
        destination / "m3d_language_metadata.json",
        {
            "state_version": _EXPORT_STATE_VERSION,
            "image_token_id": language.image_token_id,
            "segmentation_token_id": language.segmentation_token_id,
            "visual_token_count": language.visual_token_count,
            "vocabulary_size": language.vocabulary_size,
            "lora_merged": callable(merge),
        },
    )
    return {
        "directory": str(destination),
        "lora_merged": callable(merge),
        "files": _directory_file_manifest(destination),
    }


def _directory_file_manifest(root: Path) -> list[dict[str, Any]]:
    files: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size": path.stat().st_size,
                "sha256": _sha256_file(path),
            }
        )
    return files


@dataclass(frozen=True, slots=True)
class ExportRequest:
    checkpoint_path: str
    output_dir: str
    export_format: ExportFormat
    strategy: DistributedStrategy
    max_shard_size: str
    max_shard_bytes: int
    cache_dir: str | None
    local_files_only: bool
    overwrite: bool

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class ExportReport:
    state_version: int
    status: str
    created_at_utc: str
    checkpoint_path: str
    checkpoint_fingerprint: str
    checkpoint_optimizer_step: int
    checkpoint_epoch: int
    architecture_contract_sha256: str
    strategy: str
    world_size: int
    export_format: str
    output_dir: str
    full_bundle: Mapping[str, Any]
    components: Mapping[str, Any]
    adapter: Mapping[str, Any] | None
    merged_language: Mapping[str, Any] | None
    tokenizer_metadata: Mapping[str, Any]
    model_build_report: Mapping[str, Any]
    language_export_build_report: Mapping[str, Any] | None
    elapsed_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _prepare_staging_directory(output_dir: Path, *, overwrite: bool) -> Path:
    output_dir = output_dir.expanduser().resolve()
    if output_dir.exists():
        if not overwrite:
            raise ExportError(
                f"Output directory already exists: {output_dir}. Use --overwrite "
                "only after confirming it is safe to replace."
            )
        if output_dir.is_symlink():
            raise ExportError(f"Refusing to replace symlink {output_dir}.")
        if not output_dir.is_dir():
            raise ExportError(f"Export output exists and is not a directory: {output_dir}.")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = output_dir.parent / f".{output_dir.name}.incomplete-{uuid.uuid4().hex}"
    staging.mkdir(parents=False, exist_ok=False)
    return staging


def _rebase_report_paths(payload: Any, *, old_root: Path, new_root: Path) -> Any:
    """Replace staging-directory prefixes in a JSON-compatible report."""

    old_text = str(old_root)
    new_text = str(new_root)
    if isinstance(payload, str):
        if payload == old_text:
            return new_text
        prefix = old_text + os.sep
        if payload.startswith(prefix):
            return new_text + payload[len(old_text) :]
        return payload
    if isinstance(payload, Mapping):
        return {
            str(key): _rebase_report_paths(value, old_root=old_root, new_root=new_root)
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [
            _rebase_report_paths(value, old_root=old_root, new_root=new_root)
            for value in payload
        ]
    if isinstance(payload, tuple):
        return tuple(
            _rebase_report_paths(value, old_root=old_root, new_root=new_root)
            for value in payload
        )
    return payload


def _commit_staging_directory(
    staging: Path,
    output_dir: Path,
    *,
    overwrite: bool,
) -> None:
    """Commit staging while preserving an old export until the new one is ready."""

    backup: Path | None = None
    if output_dir.exists():
        if not overwrite:
            raise ExportError(f"Output directory appeared during export: {output_dir}.")
        backup = output_dir.parent / f".{output_dir.name}.backup-{uuid.uuid4().hex}"
        os.replace(output_dir, backup)
    try:
        os.replace(staging, output_dir)
    except Exception:
        if backup is not None and backup.exists() and not output_dir.exists():
            os.replace(backup, output_dir)
        raise
    if backup is not None and backup.exists():
        shutil.rmtree(backup)


def _export_rank_zero(
    *,
    full_state: Mapping[str, Tensor],
    config: ExperimentConfig,
    tokenizer_bundle: TokenizerBundle,
    model_build_report: Mapping[str, Any],
    checkpoint_path: Path,
    output_dir: Path,
    request: ExportRequest,
) -> ExportReport:
    started = time.monotonic()
    trainer_state = _read_json(checkpoint_path / _TRAINER_STATE_FILE)
    contract_hash, _ = _validate_checkpoint_configuration(config, checkpoint_path)
    _assert_independent_exported_encoders(full_state)

    if output_dir.resolve() == checkpoint_path.resolve():
        raise ExportError("Export directory must differ from the source checkpoint.")
    if checkpoint_path.resolve().is_relative_to(output_dir.resolve()):
        raise ExportError("Export directory cannot be an ancestor of the checkpoint.")
    if output_dir.resolve().is_relative_to(checkpoint_path.resolve()):
        raise ExportError("Export directory cannot be inside the source checkpoint.")

    staging = _prepare_staging_directory(output_dir, overwrite=request.overwrite)
    try:
        config.save_resolved(staging / "resolved_config.json")
        tokenizer_bundle.save_pretrained(staging / "tokenizer")

        full_report = save_sharded_safetensors(
            full_state,
            staging / "m3d_model",
            basename="m3d_model",
            max_shard_bytes=request.max_shard_bytes,
            metadata={
                "artifact": "complete_m3d_adapter_form",
                "source_checkpoint": checkpoint_path.name,
            },
        )

        component_reports: dict[str, Any] = {}
        for component_name, component_state in _component_states(full_state).items():
            component_report = save_sharded_safetensors(
                component_state,
                staging / "components" / component_name,
                basename=component_name,
                max_shard_bytes=request.max_shard_bytes,
                metadata={"artifact": component_name},
            )
            component_reports[component_name] = component_report.to_dict()

        adapter_report: Mapping[str, Any] | None = None
        merged_report: Mapping[str, Any] | None = None
        language_build_report: Mapping[str, Any] | None = None
        needs_language_instance = request.export_format in {"adapter", "merged", "all"}
        if needs_language_instance:
            language, language_build_report = _build_cpu_language_from_full_state(
                config,
                tokenizer_bundle,
                full_state,
                cache_dir=(None if request.cache_dir is None else Path(request.cache_dir)),
                local_files_only=request.local_files_only,
            )
            if request.export_format in {"adapter", "all"}:
                adapter_report = _save_peft_adapter(
                    language,
                    tokenizer_bundle,
                    staging / "language_adapter",
                )
            if request.export_format in {"merged", "all"}:
                merged_report = _merge_and_save_language(
                    language,
                    tokenizer_bundle,
                    staging / "language_merged",
                    max_shard_size=request.max_shard_size,
                )
            del language

        full_report_payload = _rebase_report_paths(
            full_report.to_dict(), old_root=staging, new_root=output_dir
        )
        component_reports = cast(
            dict[str, Any],
            _rebase_report_paths(component_reports, old_root=staging, new_root=output_dir),
        )
        adapter_report = cast(
            Mapping[str, Any] | None,
            _rebase_report_paths(adapter_report, old_root=staging, new_root=output_dir),
        )
        merged_report = cast(
            Mapping[str, Any] | None,
            _rebase_report_paths(merged_report, old_root=staging, new_root=output_dir),
        )

        report = ExportReport(
            state_version=_EXPORT_STATE_VERSION,
            status="complete",
            created_at_utc=_utc_now(),
            checkpoint_path=str(checkpoint_path),
            checkpoint_fingerprint=_checkpoint_fingerprint(checkpoint_path),
            checkpoint_optimizer_step=int(trainer_state.get("optimizer_step", -1)),
            checkpoint_epoch=int(trainer_state.get("epoch", -1)),
            architecture_contract_sha256=contract_hash,
            strategy=request.strategy,
            world_size=int(os.environ.get("WORLD_SIZE", "1")),
            export_format=request.export_format,
            output_dir=str(output_dir),
            full_bundle=full_report_payload,
            components=component_reports,
            adapter=adapter_report,
            merged_language=merged_report,
            tokenizer_metadata=tokenizer_bundle.metadata.to_dict(),
            model_build_report=model_build_report,
            language_export_build_report=language_build_report,
            elapsed_seconds=float(time.monotonic() - started),
        )
        atomic_write_json(staging / "export_manifest.json", report.to_dict())
        files = _directory_file_manifest(staging)
        atomic_write_json(
            staging / "COMPLETED.json",
            {
                "state_version": _EXPORT_STATE_VERSION,
                "status": "complete",
                "created_at_utc": report.created_at_utc,
                "source_checkpoint": str(checkpoint_path),
                "export_manifest_sha256": _sha256_file(
                    staging / "export_manifest.json"
                ),
                "files": files,
            },
        )
        _commit_staging_directory(
            staging,
            output_dir,
            overwrite=request.overwrite,
        )
        return report
    except Exception:
        failure = staging / "FAILED.json"
        try:
            atomic_write_json(
                failure,
                {
                    "state_version": _EXPORT_STATE_VERSION,
                    "status": "failed",
                    "failed_at_utc": _utc_now(),
                    "traceback": traceback.format_exc(),
                },
            )
        except Exception:
            pass
        raise


def _coordinate_rank_zero_export(
    runtime: RuntimeContext,
    *,
    status_path: Path,
    export_fn: Callable[[], ExportReport] | None,
    poll_seconds: float = 5.0,
) -> dict[str, Any]:
    """Coordinate long rank-0 CPU I/O without holding a collective open."""

    if runtime.is_main_process:
        atomic_write_json(
            status_path,
            {
                "status": "running",
                "started_at_utc": _utc_now(),
                "rank": runtime.rank,
            },
        )
        try:
            if export_fn is None:
                raise ExportError("Rank 0 export callback is missing.")
            report = export_fn()
            payload = {
                "status": "complete",
                "completed_at_utc": _utc_now(),
                "report": report.to_dict(),
            }
        except Exception:
            payload = {
                "status": "failed",
                "failed_at_utc": _utc_now(),
                "traceback": traceback.format_exc(),
            }
        atomic_write_json(status_path, payload)
    else:
        while True:
            if status_path.is_file():
                try:
                    payload = _read_json(status_path)
                except ExportCompatibilityError:
                    time.sleep(poll_seconds)
                    continue
                if payload.get("status") in {"complete", "failed"}:
                    break
            time.sleep(poll_seconds)

    payload = _read_json(status_path)
    runtime.barrier()
    if payload.get("status") != "complete":
        raise ExportError(
            "Rank-0 portable export failed:\n" + str(payload.get("traceback", payload))
        )
    return payload


def _yaml_override(key: str, value: Any) -> str:
    return f"{key}={json.dumps(value)}"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a completed M3D distributed checkpoint."
    )
    parser.add_argument("--config", type=Path, required=False)
    parser.add_argument("--checkpoint", type=Path, required=False)
    parser.add_argument("--output-dir", type=Path, required=False)
    parser.add_argument(
        "--format",
        choices=("bundle", "adapter", "merged", "all"),
        default="all",
        dest="export_format",
    )
    parser.add_argument("--strategy", choices=("ddp", "fsdp2"), default=None)
    parser.add_argument("--max-shard-size", default="4GB")
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose-all-ranks", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args(argv)
    if not args.self_test:
        missing = [
            name
            for name in ("config", "checkpoint", "output_dir")
            if getattr(args, name) is None
        ]
        if missing:
            parser.error("Missing required arguments: " + ", ".join(missing))
    return args


def run_export(args: argparse.Namespace) -> dict[str, Any]:
    checkpoint_path = resolve_checkpoint_path(cast(Path, args.checkpoint))
    output_dir = cast(Path, args.output_dir).expanduser().resolve()
    max_shard_bytes = _parse_byte_size(str(args.max_shard_size))

    # Runtime logs must not pre-create the atomically committed export directory.
    job_tag = os.environ.get("PBS_JOBID", str(os.getpid())).replace("/", "_")
    runtime_dir = output_dir.parent / f".{output_dir.name}.runtime-{job_tag}"
    base_config = load_config(
        cast(Path, args.config),
        verify_paths=False,
    )
    strategy = cast(
        DistributedStrategy,
        base_config.distributed.strategy if args.strategy is None else args.strategy,
    )
    overrides = (
        _yaml_override("distributed.strategy", strategy),
        _yaml_override("checkpoint.output_dir", str(runtime_dir)),
        _yaml_override("checkpoint.resume_from", None),
    )
    config = load_config(
        cast(Path, args.config),
        overrides=overrides,
        verify_paths=False,
    )
    contract_hash, _ = _validate_checkpoint_configuration(config, checkpoint_path)
    # DCP supplies the trained adapter tensors. Recreate the PEFT structure from
    # rank/targets instead of depending on the original warm-start directory.
    config.model.lora.adapter_checkpoint_path = None

    request = ExportRequest(
        checkpoint_path=str(checkpoint_path),
        output_dir=str(output_dir),
        export_format=cast(ExportFormat, args.export_format),
        strategy=strategy,
        max_shard_size=str(args.max_shard_size),
        max_shard_bytes=max_shard_bytes,
        cache_dir=None if args.cache_dir is None else str(args.cache_dir.resolve()),
        local_files_only=bool(args.local_files_only),
        overwrite=bool(args.overwrite),
    )

    with distributed_runtime(
        config,
        verbose_all_ranks=bool(args.verbose_all_ranks),
    ) as runtime:
        runtime.assert_all_ranks_equal(
            request.to_dict(),
            label="portable export request",
        )
        if runtime.is_main_process:
            runtime_dir.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                runtime_dir / "export_request.json",
                {
                    **request.to_dict(),
                    "architecture_contract_sha256": contract_hash,
                },
            )
        runtime.barrier()

        from .tokenization import build_tokenizer

        with runtime.main_process_first():
            tokenizer_bundle = build_tokenizer(
                config,
                cache_dir=args.cache_dir,
                local_files_only=bool(args.local_files_only),
            )
        runtime.assert_all_ranks_equal(
            tokenizer_bundle.metadata.to_dict(),
            label="export tokenizer metadata",
        )

        model, model_build_report = build_model_synchronously(
            runtime,
            lambda: build_m3d_model(
                config,
                tokenizer_bundle,
                cache_dir=args.cache_dir,
                local_files_only=bool(args.local_files_only),
                torch_dtype=torch.bfloat16,
                load_pretrained_components=False,
                strict_pretrained=True,
            ),
        )
        distributed_model, distributed_report = prepare_distributed_model(
            model,
            runtime,
        )
        distributed_model.eval()

        load_model_only_checkpoint(distributed_model, checkpoint_path)
        runtime.barrier()
        full_state = gather_full_cpu_state(distributed_model, runtime)

        build_payload = {
            "model": model_build_report.to_dict(),
            "distributed": distributed_report.to_dict(),
        }
        status_path = runtime_dir / "rank0_export_status.json"
        export_fn: Callable[[], ExportReport] | None = None
        if runtime.is_main_process:
            export_fn = lambda: _export_rank_zero(
                full_state=full_state,
                config=config,
                tokenizer_bundle=tokenizer_bundle,
                model_build_report=build_payload,
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                request=request,
            )
        payload = _coordinate_rank_zero_export(
            runtime,
            status_path=status_path,
            export_fn=export_fn,
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return cast(dict[str, Any], payload["report"])


def _self_test() -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="m3d-export-test-") as temporary:
        root = Path(temporary)
        shared = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        state = {
            "encoder.weight": shared,
            "decoder.weight": shared,
            "small.bias": torch.tensor([1.0, 2.0]),
        }
        report = save_sharded_safetensors(
            state,
            root / "bundle",
            basename="toy",
            max_shard_bytes=40,
            metadata={"test": "true"},
        )
        restored = load_sharded_safetensors_for_test(
            root / "bundle",
            basename="toy",
        )
        roundtrip = set(restored) == set(state) and all(
            torch.equal(restored[name], tensor)
            for name, tensor in state.items()
        )

        config_a = {
            "schema_version": 1,
            "model": {
                "main_vision": {
                    "hidden_size": 32,
                    "checkpoint_path": "/a.bin",
                    "freeze": False,
                    "attention_backend": "sdpa",
                },
                "seg_vision": {
                    "hidden_size": 32,
                    "checkpoint_path": "/b.bin",
                    "freeze": False,
                    "attention_backend": "sdpa",
                },
                "projector": {
                    "num_layers": 2,
                    "checkpoint_path": "/p.bin",
                    "freeze": False,
                },
                "segmentation": {
                    "enabled": True,
                    "prompt_embed_dim": 32,
                    "checkpoint_path": "/s.bin",
                    "dice_loss_weight": 1.0,
                    "bce_loss_weight": 1.0,
                },
                "lora": {
                    "enabled": True,
                    "rank": 8,
                    "target_modules": ["q_proj"],
                    "adapter_checkpoint_path": "/adapter",
                },
            },
        }
        config_b = copy.deepcopy(config_a)
        config_b["model"]["main_vision"]["checkpoint_path"] = "/different.bin"
        config_b["model"]["main_vision"]["freeze"] = True
        config_b["model"]["segmentation"]["dice_loss_weight"] = 2.0
        contract_stable = _sha256_payload(
            _architecture_contract_from_mapping(config_a)
        ) == _sha256_payload(_architecture_contract_from_mapping(config_b))
        config_b["model"]["lora"]["rank"] = 16
        contract_detects_layout_change = _sha256_payload(
            _architecture_contract_from_mapping(config_a)
        ) != _sha256_payload(_architecture_contract_from_mapping(config_b))

        source = nn.Linear(4, 3)
        optimizer = torch.optim.AdamW(source.parameters(), lr=1e-3)

        class _ModelOptimizerForTest(Stateful):
            def __init__(self, model: nn.Module, optim: torch.optim.Optimizer) -> None:
                self.model = model
                self.optim = optim
                self.options = StateDictOptions(strict=True)

            def state_dict(self) -> dict[str, Any]:
                from torch.distributed.checkpoint.state_dict import get_state_dict

                model_state, optim_state = get_state_dict(
                    self.model,
                    self.optim,
                    options=self.options,
                )
                return {"model": model_state, "optimizer": optim_state}

            def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
                raise NotImplementedError

        dcp_path = root / "dcp"
        dcp.save(
            {"application": _ModelOptimizerForTest(source, optimizer)},
            checkpoint_id=dcp_path,
        )
        target = nn.Linear(4, 3)
        dcp.load(
            {"application": _ModelOnlyDCPState(target)},
            checkpoint_id=dcp_path,
        )
        model_only_dcp_roundtrip = all(
            torch.equal(left, right)
            for left, right in zip(source.parameters(), target.parameters(), strict=True)
        )

        result = {
            "status": "passed",
            "byte_size_4gb": _parse_byte_size("4GB"),
            "byte_size_4gib": _parse_byte_size("4GiB"),
            "safetensors_shard_count": len(report.shards),
            "shared_storage_roundtrip": roundtrip,
            "architecture_contract_ignores_runtime_fields": contract_stable,
            "architecture_contract_detects_lora_rank": contract_detects_layout_change,
            "model_only_dcp_roundtrip": model_only_dcp_roundtrip,
        }
        if not all(
            (
                roundtrip,
                contract_stable,
                contract_detects_layout_change,
                model_only_dcp_roundtrip,
                len(report.shards) >= 2,
            )
        ):
            raise AssertionError(result)
        return result


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.self_test:
        print(json.dumps(_self_test(), indent=2, sort_keys=True))
        return
    report = run_export(args)
    rank = int(os.environ.get("RANK", "0"))
    if rank == 0:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
