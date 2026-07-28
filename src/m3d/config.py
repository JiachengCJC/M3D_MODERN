"""Typed configuration system for M3D-Modernized.

This module is imported before datasets and models are constructed.  It keeps
all training decisions in one YAML file and validates combinations that would
otherwise fail much later inside distributed training.

The project intentionally keeps two independent image encoders:

* ``model.main_vision`` feeds the multimodal projector and language model.
* ``model.seg_vision`` feeds the SegVol-style segmentation path.

They may share the same Python implementation, but they never share parameters.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, MutableMapping, Sequence, TypeVar, get_args, get_origin, get_type_hints

import numpy as np
import torch
import yaml


CONFIG_SCHEMA_VERSION = 1
LATEST_CHECKPOINT_SENTINEL = "latest"

T = TypeVar("T")

AttentionBackend = Literal["sdpa", "math"]
DistributedStrategy = Literal["ddp", "fsdp2"]
TrainingStage = Literal[
    "projector_pretrain",
    "lora_finetune",
    "segmentation_finetune",
    "joint_finetune",
]


@dataclass(slots=True)
class VisionEncoderConfig:
    """Configuration shared by the *definition* of each independent 3D ViT."""

    enabled: bool = True
    architecture: str = "vit3d"
    checkpoint_path: str | None = None

    image_channels: int = 1
    image_size: tuple[int, int, int] = (32, 256, 256)
    patch_size: tuple[int, int, int] = (4, 16, 16)

    hidden_size: int = 768
    depth: int = 12
    num_heads: int = 12
    mlp_dim: int = 3072
    dropout: float = 0.0
    qkv_bias: bool = False
    use_cls_token: bool = True

    attention_backend: AttentionBackend = "sdpa"
    require_flash_sdpa: bool = True
    activation_checkpoint_every_n_layers: int = 0

    freeze: bool = False
    unfreeze_last_n_layers: int = 0


@dataclass(slots=True)
class ProjectorConfig:
    projector_type: Literal["spatial_pooling"] = "spatial_pooling"
    layer_type: Literal["linear", "mlp"] = "mlp"
    num_layers: int = 2
    pooling_type: Literal["spatial", "sequence"] = "spatial"
    pooling_size: int = 2
    checkpoint_path: str | None = None
    freeze: bool = False


@dataclass(slots=True)
class SegmentationConfig:
    enabled: bool = True
    architecture: Literal["segvol"] = "segvol"
    checkpoint_path: str | None = None
    prompt_embed_dim: int = 768
    decoder_depth: int = 2
    decoder_heads: int = 8
    dice_loss_weight: float = 1.0
    bce_loss_weight: float = 1.0
    freeze_prompt_encoder: bool = False
    freeze_mask_decoder: bool = False


@dataclass(slots=True)
class LoRAConfig:
    enabled: bool = True
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    bias: Literal["none", "all", "lora_only"] = "none"
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    )
    adapter_checkpoint_path: str | None = None


@dataclass(slots=True)
class ModelConfig:
    language_model_name_or_path: str = "microsoft/Phi-3-mini-4k-instruct"
    tokenizer_name_or_path: str | None = None
    language_model_family: Literal["phi3"] = "phi3"
    trust_remote_code: bool = True
    model_max_length: int = 512

    image_token: str = "<im_patch>"
    segmentation_token: str = "[SEG]"
    box_start_token: str = "<bx_start>"
    box_end_token: str = "<bx_end>"

    # Two different objects are created by default.  They share no parameters.
    main_vision: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    seg_vision: VisionEncoderConfig = field(
        default_factory=lambda: VisionEncoderConfig(use_cls_token=False)
    )
    projector: ProjectorConfig = field(default_factory=ProjectorConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    lora: LoRAConfig = field(default_factory=LoRAConfig)


@dataclass(slots=True)
class DatasetPathsConfig:
    data_root: str = "./Data/data"
    caption_json: str = "M3D_Cap_npy/M3D_Cap.json"
    vqa_train_csv: str = "M3D-VQA/M3D_VQA_train.csv"
    vqa_val_csv: str = "M3D-VQA/M3D_VQA_val.csv"
    vqa_test_csv: str = "M3D-VQA/M3D_VQA_test.csv"
    vqa_yes_no_train_csv: str = "M3D-VQA/M3D_VQA_yn_train.csv"
    segmentation_root: str = "M3D_Seg_npy"
    referring_segmentation_train_csv: str = "M3D_RefSeg_npy/M3D_RefSeg.csv"
    referring_segmentation_test_csv: str = "M3D_RefSeg_npy/M3D_RefSeg_test.csv"


@dataclass(slots=True)
class TaskSamplingConfig:
    """Sampling policy for task-homogeneous distributed batches."""

    enabled: bool = True
    homogeneous_batches: bool = True
    temperature_alpha: float = 0.5
    steps_per_epoch: int | None = None
    task_weights: dict[str, float] = field(
        default_factory=lambda: {
            "caption": 1.0,
            "vqa_closed": 1.0,
            "vqa_open": 1.0,
            "vqa_yes_no": 0.5,
            "positioning": 1.0,
            "segmentation": 2.0,
        }
    )


@dataclass(slots=True)
class DataConfig:
    paths: DatasetPathsConfig = field(default_factory=DatasetPathsConfig)
    task_sampling: TaskSamplingConfig = field(default_factory=TaskSamplingConfig)

    dynamic_padding: bool = True
    pad_to_multiple_of: int = 8
    sequence_length_buckets: tuple[int, ...] = (128, 256, 384, 512)
    pretokenize_text: bool = True

    num_workers: int = 8
    persistent_workers: bool = True
    pin_memory: bool = True
    prefetch_factor: int = 4
    non_blocking_transfer: bool = True

    # PBS can set this to $TMPDIR after staging data locally.
    local_cache_root: str | None = None
    verify_files_at_startup: bool = True


@dataclass(slots=True)
class LearningRateConfig:
    language_model: float = 1.0e-5
    main_vision: float = 5.0e-6
    seg_vision: float = 5.0e-6
    projector: float = 5.0e-5
    segmentation_projector: float = 5.0e-5
    segmentation_decoder: float = 1.0e-5
    token_embeddings: float = 5.0e-5


@dataclass(slots=True)
class OptimizationConfig:
    stage: TrainingStage = "lora_finetune"
    epochs: float = 5.0
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 4

    precision: Literal["bf16"] = "bf16"
    allow_tf32: bool = True
    matmul_precision: Literal["highest", "high", "medium"] = "high"

    optimizer: Literal["adamw_fused"] = "adamw_fused"
    learning_rates: LearningRateConfig = field(default_factory=LearningRateConfig)
    betas: tuple[float, float] = (0.9, 0.95)
    epsilon: float = 1.0e-8
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0

    scheduler: Literal["cosine"] = "cosine"
    warmup_ratio: float = 0.03

    # The model file will apply checkpointing independently to LLM and both ViTs.
    checkpoint_language_model: bool = True
    checkpoint_main_vision: bool = True
    checkpoint_seg_vision: bool = True
    checkpoint_segmentation_decoder: bool = False

    # torch.compile is enabled only after eager-mode correctness tests pass.
    compile_model: bool = False
    compile_mode: Literal["default", "reduce-overhead", "max-autotune"] = "default"


@dataclass(slots=True)
class DDPConfig:
    find_unused_parameters: bool = True
    gradient_as_bucket_view: bool = True
    static_graph: bool = False
    bucket_cap_mb: int = 25


@dataclass(slots=True)
class FSDP2Config:
    reshard_after_forward: bool = True
    cpu_offload: bool = False
    mixed_precision: Literal["bf16"] = "bf16"
    wrap_language_layers: bool = True
    wrap_main_vision_layers: bool = True
    wrap_seg_vision_layers: bool = True
    wrap_segmentation_decoder_layers: bool = True


@dataclass(slots=True)
class DistributedConfig:
    strategy: DistributedStrategy = "ddp"
    # ``gloo`` is accepted only by the explicitly gated local CPU integration
    # harness.  The production runtime still rejects it unless
    # M3D_CPU_DISTRIBUTED_SMOKE=1 is present.
    backend: Literal["nccl", "gloo"] = "nccl"
    timeout_seconds: int = 1800
    ddp: DDPConfig = field(default_factory=DDPConfig)
    fsdp2: FSDP2Config = field(default_factory=FSDP2Config)


@dataclass(slots=True)
class CheckpointConfig:
    output_dir: str = "./outputs/m3d-phi3-finetune"
    resume_from: str | None = None
    save_every_steps: int = 1000
    keep_last_n: int = 2
    save_optimizer: bool = True
    save_scheduler: bool = True
    save_rng_state: bool = True
    asynchronous: bool = True
    format: Literal["distributed_checkpoint"] = "distributed_checkpoint"
    export_safetensors_at_end: bool = True


@dataclass(slots=True)
class LoggingConfig:
    log_every_steps: int = 10
    report_to: tuple[Literal["tensorboard"], ...] = ("tensorboard",)
    tensorboard_dir: str = "./logs/tensorboard"
    profile_steps: tuple[int, ...] = ()
    log_gpu_memory: bool = True


@dataclass(slots=True)
class RuntimeConfig:
    seed: int = 42
    deterministic: bool = False
    fail_on_nondeterministic_ops: bool = False
    detect_anomaly: bool = False


@dataclass(slots=True)
class ExperimentConfig:
    schema_version: int = CONFIG_SCHEMA_VERSION
    experiment_name: str = "m3d-phi3-modernized"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optimization: OptimizationConfig = field(default_factory=OptimizationConfig)
    distributed: DistributedConfig = field(default_factory=DistributedConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)

    def validate(self) -> None:
        """Reject incompatible settings before any GPU memory is allocated."""

        errors: list[str] = []

        if self.schema_version != CONFIG_SCHEMA_VERSION:
            errors.append(
                f"schema_version must be {CONFIG_SCHEMA_VERSION}, got {self.schema_version}"
            )

        if not self.experiment_name.strip():
            errors.append("experiment_name cannot be empty")

        if self.model.model_max_length <= 0:
            errors.append("model.model_max_length must be positive")

        for name, vision in (
            ("model.main_vision", self.model.main_vision),
            ("model.seg_vision", self.model.seg_vision),
        ):
            if len(vision.image_size) != 3 or len(vision.patch_size) != 3:
                errors.append(f"{name}.image_size and patch_size must have three values")
                continue

            if any(value <= 0 for value in (*vision.image_size, *vision.patch_size)):
                errors.append(f"{name} image and patch dimensions must be positive")

            if any(
                image_dim % patch_dim != 0
                for image_dim, patch_dim in zip(vision.image_size, vision.patch_size)
            ):
                errors.append(f"{name}.image_size must be divisible by patch_size")

            if vision.hidden_size % vision.num_heads != 0:
                errors.append(f"{name}.hidden_size must be divisible by num_heads")

            if vision.depth <= 0 or vision.mlp_dim <= 0:
                errors.append(f"{name}.depth and mlp_dim must be positive")

            if not 0.0 <= vision.dropout < 1.0:
                errors.append(f"{name}.dropout must be in [0, 1)")

            if vision.require_flash_sdpa and vision.attention_backend != "sdpa":
                errors.append(
                    f"{name}.require_flash_sdpa requires attention_backend='sdpa'"
                )

            if vision.activation_checkpoint_every_n_layers < 0:
                errors.append(
                    f"{name}.activation_checkpoint_every_n_layers cannot be negative"
                )

            if not 0 <= vision.unfreeze_last_n_layers <= vision.depth:
                errors.append(
                    f"{name}.unfreeze_last_n_layers must be between 0 and depth"
                )

            if vision.freeze and vision.unfreeze_last_n_layers > 0:
                errors.append(
                    f"{name} cannot set freeze=true and unfreeze_last_n_layers>0 together"
                )

        if self.model.main_vision is self.model.seg_vision:
            errors.append("main_vision and seg_vision must be independent config objects")

        if not self.model.segmentation.enabled and self.model.seg_vision.enabled:
            errors.append(
                "model.seg_vision.enabled should be false when segmentation is disabled"
            )

        if self.model.segmentation.enabled and not self.model.seg_vision.enabled:
            errors.append("segmentation requires model.seg_vision.enabled=true")

        if self.model.projector.pooling_size <= 0:
            errors.append("model.projector.pooling_size must be positive")

        lora = self.model.lora
        if lora.enabled:
            if lora.rank <= 0 or lora.alpha <= 0:
                errors.append("LoRA rank and alpha must be positive")
            if not 0.0 <= lora.dropout < 1.0:
                errors.append("LoRA dropout must be in [0, 1)")
            if not lora.target_modules:
                errors.append("LoRA target_modules cannot be empty when LoRA is enabled")

        stage = self.optimization.stage
        if stage == "projector_pretrain" and lora.enabled:
            errors.append("projector_pretrain must set model.lora.enabled=false")
        if stage == "lora_finetune" and not lora.enabled:
            errors.append("lora_finetune requires model.lora.enabled=true")
        if stage in {"segmentation_finetune", "joint_finetune"}:
            if not self.model.segmentation.enabled:
                errors.append(f"{stage} requires segmentation to be enabled")

        task_sampling = self.data.task_sampling
        if not 0.0 <= task_sampling.temperature_alpha <= 1.0:
            errors.append("data.task_sampling.temperature_alpha must be in [0, 1]")
        if task_sampling.enabled and not task_sampling.homogeneous_batches:
            errors.append(
                "optimized task sampling requires homogeneous_batches=true so all ranks "
                "execute the same conditional model path"
            )
        if not task_sampling.task_weights:
            errors.append("data.task_sampling.task_weights cannot be empty")
        for task_name, weight in task_sampling.task_weights.items():
            if not task_name.strip() or weight < 0.0:
                errors.append("task names must be non-empty and weights must be non-negative")
        if not any(weight > 0.0 for weight in task_sampling.task_weights.values()):
            errors.append("at least one task weight must be positive")
        if self.model.segmentation.enabled and task_sampling.task_weights.get(
            "segmentation", 0.0
        ) <= 0.0:
            errors.append(
                "segmentation is enabled but the segmentation task weight is not positive"
            )

        if self.data.dynamic_padding:
            if self.data.pad_to_multiple_of <= 0:
                errors.append("data.pad_to_multiple_of must be positive")
            if not self.data.sequence_length_buckets:
                errors.append(
                    "dynamic padding requires at least one sequence_length_bucket"
                )
            elif tuple(sorted(set(self.data.sequence_length_buckets))) != tuple(
                self.data.sequence_length_buckets
            ):
                errors.append(
                    "data.sequence_length_buckets must be unique and increasing"
                )
            elif self.data.sequence_length_buckets[-1] != self.model.model_max_length:
                errors.append(
                    "the largest sequence_length_bucket must equal model.model_max_length"
                )

        if self.data.num_workers < 0:
            errors.append("data.num_workers cannot be negative")
        if self.data.num_workers == 0 and self.data.persistent_workers:
            errors.append("persistent_workers requires data.num_workers > 0")
        if self.data.num_workers > 0 and self.data.prefetch_factor <= 0:
            errors.append("data.prefetch_factor must be positive")

        optim = self.optimization
        if optim.epochs <= 0:
            errors.append("optimization.epochs must be positive")
        if optim.per_device_batch_size <= 0:
            errors.append("optimization.per_device_batch_size must be positive")
        if optim.gradient_accumulation_steps <= 0:
            errors.append("optimization.gradient_accumulation_steps must be positive")
        if not 0.0 <= optim.warmup_ratio < 1.0:
            errors.append("optimization.warmup_ratio must be in [0, 1)")
        if optim.max_grad_norm <= 0:
            errors.append("optimization.max_grad_norm must be positive")
        for group_name, learning_rate in dataclasses.asdict(
            optim.learning_rates
        ).items():
            if learning_rate < 0.0:
                errors.append(f"learning rate {group_name} cannot be negative")

        if self.distributed.strategy == "ddp":
            if task_sampling.enabled and not self.distributed.ddp.find_unused_parameters:
                errors.append(
                    "DDP task-homogeneous batches conditionally skip segmentation modules; "
                    "set distributed.ddp.find_unused_parameters=true for the first correct "
                    "implementation"
                )
            if (
                self.distributed.ddp.static_graph
                and self.distributed.ddp.find_unused_parameters
            ):
                errors.append(
                    "DDP static_graph and find_unused_parameters=true are incompatible with "
                    "the changing text/segmentation execution paths"
                )

        if self.checkpoint.save_every_steps <= 0:
            errors.append("checkpoint.save_every_steps must be positive")
        if self.checkpoint.keep_last_n <= 0:
            errors.append("checkpoint.keep_last_n must be positive")
        if self.logging.log_every_steps <= 0:
            errors.append("logging.log_every_steps must be positive")

        if errors:
            formatted = "\n".join(f"  - {error}" for error in errors)
            raise ValueError(f"Invalid M3D configuration:\n{formatted}")

    def resolve_paths(self, base_dir: str | os.PathLike[str]) -> None:
        """Resolve project-local paths relative to the YAML file directory."""

        base = Path(base_dir).expanduser().resolve()

        def resolve(value: str | None) -> str | None:
            if value is None or not value.strip():
                return value
            path = Path(os.path.expandvars(value)).expanduser()
            if path.is_absolute():
                return str(path.resolve())
            return str((base / path).resolve())

        self.data.paths.data_root = resolve(self.data.paths.data_root) or ""
        self.data.local_cache_root = resolve(self.data.local_cache_root)
        self.checkpoint.output_dir = resolve(self.checkpoint.output_dir) or ""
        resume_from = self.checkpoint.resume_from
        self.checkpoint.resume_from = (
            LATEST_CHECKPOINT_SENTINEL
            if resume_from is not None
            and resume_from.strip() == LATEST_CHECKPOINT_SENTINEL
            else resolve(resume_from)
        )
        self.logging.tensorboard_dir = resolve(self.logging.tensorboard_dir) or ""

        self.model.main_vision.checkpoint_path = resolve(
            self.model.main_vision.checkpoint_path
        )
        self.model.seg_vision.checkpoint_path = resolve(
            self.model.seg_vision.checkpoint_path
        )
        self.model.projector.checkpoint_path = resolve(
            self.model.projector.checkpoint_path
        )
        self.model.segmentation.checkpoint_path = resolve(
            self.model.segmentation.checkpoint_path
        )
        self.model.lora.adapter_checkpoint_path = resolve(
            self.model.lora.adapter_checkpoint_path
        )

    def dataset_path(self, relative_path: str) -> Path:
        """Return a dataset path under ``data_root`` with traversal protection."""

        root = Path(self.data.paths.data_root).resolve()
        candidate = (root / os.path.expandvars(relative_path)).resolve()
        try:
            candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"Dataset path escapes data_root: {relative_path!r}"
            ) from exc
        return candidate

    def verify_required_paths(self) -> None:
        """Check local files required by the selected training stage."""

        missing: list[str] = []
        data_paths = self.data.paths

        required_dataset_files = {
            "caption_json": data_paths.caption_json,
            "vqa_train_csv": data_paths.vqa_train_csv,
            "vqa_val_csv": data_paths.vqa_val_csv,
            "vqa_yes_no_train_csv": data_paths.vqa_yes_no_train_csv,
            "referring_segmentation_train_csv": (
                data_paths.referring_segmentation_train_csv
            ),
        }

        for name, relative_path in required_dataset_files.items():
            path = self.dataset_path(relative_path)
            if not path.is_file():
                missing.append(f"data.paths.{name}: {path}")

        for name, relative_path in (
            ("segmentation_root", data_paths.segmentation_root),
        ):
            path = self.dataset_path(relative_path)
            if not path.is_dir():
                missing.append(f"data.paths.{name}: {path}")

        for name, path_value in (
            ("model.main_vision.checkpoint_path", self.model.main_vision.checkpoint_path),
            ("model.seg_vision.checkpoint_path", self.model.seg_vision.checkpoint_path),
            ("model.projector.checkpoint_path", self.model.projector.checkpoint_path),
            (
                "model.segmentation.checkpoint_path",
                self.model.segmentation.checkpoint_path,
            ),
            ("model.lora.adapter_checkpoint_path", self.model.lora.adapter_checkpoint_path),
            ("checkpoint.resume_from", self.checkpoint.resume_from),
        ):
            if (
                path_value is not None
                and path_value != LATEST_CHECKPOINT_SENTINEL
                and not Path(path_value).exists()
            ):
                missing.append(f"{name}: {path_value}")

        if missing:
            formatted = "\n".join(f"  - {item}" for item in missing)
            raise FileNotFoundError(f"Missing required M3D paths:\n{formatted}")

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def save_resolved(self, output_path: str | os.PathLike[str]) -> None:
        """Atomically save the exact resolved configuration used by a run."""

        destination = Path(output_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(
                self.to_dict(),
                handle,
                sort_keys=False,
                allow_unicode=True,
            )
        os.replace(temporary, destination)


def _convert_value(expected_type: Any, value: Any, field_path: str) -> Any:
    """Convert YAML values into the annotated dataclass field type."""

    origin = get_origin(expected_type)
    args = get_args(expected_type)

    if dataclasses.is_dataclass(expected_type):
        if not isinstance(value, Mapping):
            raise TypeError(f"{field_path} must be a mapping")
        return _dataclass_from_mapping(expected_type, value, field_path)

    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise TypeError(f"{field_path} must be a list or tuple")
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_convert_value(args[0], item, field_path) for item in value)
        if args and len(value) != len(args):
            raise TypeError(f"{field_path} must contain exactly {len(args)} values")
        return tuple(
            _convert_value(item_type, item, field_path)
            for item_type, item in zip(args, value)
        )

    if origin is dict:
        if not isinstance(value, Mapping):
            raise TypeError(f"{field_path} must be a mapping")
        key_type, value_type = args
        return {
            _convert_value(key_type, key, field_path): _convert_value(
                value_type, item, field_path
            )
            for key, item in value.items()
        }

    # PEP 604 union, including Optional[T].
    if origin is not None and str(origin) in {"<class 'types.UnionType'>", "typing.Union"}:
        if value is None and type(None) in args:
            return None
        non_none = [arg for arg in args if arg is not type(None)]
        last_error: Exception | None = None
        for candidate in non_none:
            try:
                return _convert_value(candidate, value, field_path)
            except (TypeError, ValueError) as exc:
                last_error = exc
        raise TypeError(f"{field_path} does not match its allowed types") from last_error

    if origin is Literal:
        if value not in args:
            allowed = ", ".join(repr(item) for item in args)
            raise ValueError(f"{field_path} must be one of: {allowed}")
        return value

    if expected_type is Any:
        return value
    if expected_type is bool:
        if not isinstance(value, bool):
            raise TypeError(f"{field_path} must be true or false")
        return value
    if expected_type is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{field_path} must be an integer")
        return value
    if expected_type is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{field_path} must be numeric")
        return float(value)
    if expected_type is str:
        if not isinstance(value, str):
            raise TypeError(f"{field_path} must be a string")
        return value

    return value


def _dataclass_from_mapping(
    cls: type[T], values: Mapping[str, Any], field_path: str = "config"
) -> T:
    """Strictly construct a nested dataclass and reject unknown YAML keys."""

    field_definitions = {item.name: item for item in dataclasses.fields(cls)}
    unknown = sorted(set(values) - set(field_definitions))
    if unknown:
        raise KeyError(
            f"Unknown keys under {field_path}: {', '.join(unknown)}"
        )

    type_hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in values.items():
        expected_type = type_hints[name]
        kwargs[name] = _convert_value(
            expected_type,
            value,
            f"{field_path}.{name}",
        )
    return cls(**kwargs)


def _set_override(config: MutableMapping[str, Any], expression: str) -> None:
    """Apply one ``section.key=value`` command-line override."""

    if "=" not in expression:
        raise ValueError(
            f"Invalid override {expression!r}; expected dotted.path=value"
        )

    dotted_key, raw_value = expression.split("=", 1)
    keys = [part.strip() for part in dotted_key.split(".") if part.strip()]
    if not keys:
        raise ValueError(f"Invalid override key in {expression!r}")

    try:
        parsed_value = yaml.safe_load(raw_value)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML override value in {expression!r}") from exc

    current: MutableMapping[str, Any] = config
    for key in keys[:-1]:
        existing = current.get(key)
        if existing is None:
            current[key] = {}
            existing = current[key]
        if not isinstance(existing, MutableMapping):
            raise ValueError(
                f"Cannot override {dotted_key!r}: {key!r} is not a mapping"
            )
        current = existing
    current[keys[-1]] = parsed_value


def load_config(
    path: str | os.PathLike[str],
    overrides: Sequence[str] = (),
    *,
    resolve_paths: bool = True,
    verify_paths: bool | None = None,
) -> ExperimentConfig:
    """Load, override, type-check, validate, and optionally verify a YAML config."""

    # Load the YAML file and apply command-line overrides.
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {source}")

    # 读取 YAML 文件， 并将其内容加载为 Python 对象
    with source.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)

    if loaded is None:
        loaded = {}
    if not isinstance(loaded, MutableMapping):
        raise TypeError("The top level of the configuration must be a mapping")

    raw_config: MutableMapping[str, Any] = copy.deepcopy(loaded)
    for expression in overrides:
        _set_override(raw_config, expression)

    config = _dataclass_from_mapping(ExperimentConfig, raw_config)
    if resolve_paths:
        config.resolve_paths(source.parent)
    config.validate()

    should_verify = (
        config.data.verify_files_at_startup if verify_paths is None else verify_paths
    )
    if should_verify:
        config.verify_required_paths()

    return config


def seed_everything(config: RuntimeConfig, rank: int = 0) -> int:
    """Seed Python, NumPy, CPU CUDA generators, with a deterministic rank offset."""

    seed = int(config.seed) + int(rank)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = not config.deterministic
    torch.backends.cudnn.deterministic = config.deterministic
    torch.use_deterministic_algorithms(
        config.deterministic,
        warn_only=not config.fail_on_nondeterministic_ops,
    )
    torch.autograd.set_detect_anomaly(config.detect_anomaly)
    return seed


def configure_torch_runtime(config: OptimizationConfig) -> None:
    """Apply process-local numerical runtime settings before model creation."""

    if config.precision != "bf16":
        raise ValueError("ASPIRE 2A optimized training currently requires BF16")

    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.allow_tf32
    torch.set_float32_matmul_precision(config.matmul_precision)


def config_fingerprint(config: ExperimentConfig) -> str:
    """Return a stable JSON representation useful for logs and checkpoints."""

    return json.dumps(
        config.to_dict(),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
