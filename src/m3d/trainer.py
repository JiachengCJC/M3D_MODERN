"""Task-aware distributed training loop for M3D-Modernized.

This module is the first place where the complete training stack executes as
one system:

* task-homogeneous distributed ``DataLoader`` batches;
* explicit text versus segmentation model graphs;
* BF16 autocast without ``GradScaler``;
* exact per-microbatch global loss normalisation;
* communication-free gradient accumulation on non-update microbatches;
* gradient clipping, AdamW, and optimizer-step-based cosine scheduling;
* checkpoint-exact sampler/scheduler progress;
* low-synchronisation metric and CUDA-memory logging;
* optional targeted PyTorch profiler traces.

The central numerical detail is the backward scalar.  Every rank first
all-reduces only the small count tensors needed by
:func:`m3d.model.loss.compose_data_parallel_backward_loss`.  The local loss
sums are then scaled by ``world_size / global_count``.  DDP/FSDP2's gradient
average consequently produces the exact global token/sample/voxel mean for
that microbatch.  The scalar is finally divided by the current accumulation
window size, including a shorter final window at an epoch boundary.

A batch is committed to the durable data cursor only after its backward pass
succeeds.  On an optimizer-update microbatch, commit happens after clipping,
``optimizer.step()``, ``scheduler.step()``, and ``zero_grad`` all succeed.
Checkpoints are therefore always written at a recoverable accumulation
boundary and never contain a data cursor ahead of the model update.
"""

from __future__ import annotations

import contextlib
import dataclasses
import json
import math
import time
from collections.abc import Generator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import torch
from torch import Tensor
from torch.optim import Optimizer

from .checkpointing import (
    CheckpointManager,
    CheckpointResumeReport,
    CheckpointSaveReport,
    should_save_checkpoint,
)
from .config import ExperimentConfig
from .data.sampler import sampler_position_from_batch
from .data.schema import M3DBatch, TASK_ORDER, TaskName
from .distributed import DistributedM3DModel
from .model.loss import M3DLossOutput, compose_data_parallel_backward_loss
from .optim import optimizer_groups_by_name
from .runtime import RuntimeContext, atomic_write_json
from .scheduler import CosineWarmupScheduler, validate_resume_progress

if TYPE_CHECKING:
    from .data.loader import TrainingDataPipeline


class TrainerError(RuntimeError):
    """Base class for training-loop failures."""


class TrainerConfigurationError(TrainerError):
    """Raised when connected components disagree before training starts."""


class NonFiniteTrainingError(TrainerError):
    """Raised synchronously on every rank when one rank observes NaN/Inf loss."""


class TrainingProgressError(TrainerError):
    """Raised when sampler, epoch, and scheduler cursors diverge."""


@dataclass(frozen=True, slots=True)
class GlobalLossNormalisers:
    """All-reduced counts needed to build one exact backward scalar."""

    language_token_count: Tensor
    segmentation_sample_count: Tensor | None
    segmentation_voxel_count: Tensor | None

    def __post_init__(self) -> None:
        fields = (
            ("language_token_count", self.language_token_count),
            ("segmentation_sample_count", self.segmentation_sample_count),
            ("segmentation_voxel_count", self.segmentation_voxel_count),
        )
        for name, value in fields:
            if value is None:
                continue
            if not isinstance(value, Tensor) or value.ndim != 0:
                raise TrainerConfigurationError(f"{name} must be a scalar Tensor.")
            if value.dtype not in (torch.int32, torch.int64):
                raise TrainerConfigurationError(
                    f"{name} must be an integer Tensor, got {value.dtype}."
                )


@dataclass(frozen=True, slots=True)
class TrainingResult:
    """Durable summary returned after the complete training plan finishes."""

    completed_optimizer_steps: int
    total_optimizer_steps: int
    final_epoch: int
    final_committed_microbatches: int
    elapsed_seconds: float
    final_checkpoint: str | None
    resumed_from: str | None
    finished_at_unix: float

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True, slots=True)
class TrainerBuildReport:
    """Static trainer wiring report written before the first batch."""

    strategy: str
    world_size: int
    per_device_batch_size: int
    gradient_accumulation_steps: int
    epoch_microbatch_limits: tuple[int, ...]
    optimizer_steps_per_epoch: tuple[int, ...]
    total_optimizer_steps: int
    log_every_optimizer_steps: int
    checkpoint_every_optimizer_steps: int
    profile_optimizer_steps: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = dataclasses.asdict(self)
        payload["epoch_microbatch_limits"] = list(self.epoch_microbatch_limits)
        payload["optimizer_steps_per_epoch"] = list(
            self.optimizer_steps_per_epoch
        )
        payload["profile_optimizer_steps"] = list(self.profile_optimizer_steps)
        return payload


class _MetricRuntime(Protocol):
    world_size: int

    def reduce_scalar_dict(
        self,
        values: Mapping[str, float | int | Tensor],
        *,
        average: bool = True,
    ) -> dict[str, float]: ...


@dataclass(slots=True)
class MetricAccumulator:
    """Accumulate detached scalar sums on-device until a logging boundary.

    No ``Tensor.item()`` or device-to-host transfer occurs in ``update``.  A
    single packed distributed reduction is performed by ``flush``.
    """

    device: torch.device
    _values: dict[str, Tensor] = field(init=False, default_factory=dict)
    _started_at: float = field(init=False, default_factory=time.monotonic)
    _microbatch_updates: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self.reset()

    def _zero(self) -> Tensor:
        return torch.zeros((), device=self.device, dtype=torch.float64)

    def reset(self) -> None:
        names = [
            "objective_contribution_sum",
            "microbatch_count",
            "sample_count",
            "language_sum",
            "language_token_count",
            "dice_sum",
            "segmentation_sample_count",
            "bce_sum",
            "segmentation_voxel_count",
            "foreground_voxel_count",
            "empty_target_count",
            "legacy_minus_one_voxel_count",
            "gradient_norm_sum",
            "gradient_norm_count",
        ]
        names.extend(f"task/{task.value}" for task in TASK_ORDER)
        self._values = {name: self._zero() for name in names}
        self._started_at = time.monotonic()
        self._microbatch_updates = 0

    @property
    def is_empty(self) -> bool:
        return self._microbatch_updates == 0

    def update_microbatch(
        self,
        *,
        batch: M3DBatch,
        loss_output: M3DLossOutput,
        backward_loss: Tensor,
    ) -> None:
        if not isinstance(batch, M3DBatch):
            raise TypeError("batch must be M3DBatch")
        if not isinstance(loss_output, M3DLossOutput):
            raise TypeError("loss_output must be M3DLossOutput")
        if not isinstance(backward_loss, Tensor) or backward_loss.ndim != 0:
            raise TypeError("backward_loss must be a scalar Tensor")

        values = self._values
        values["objective_contribution_sum"].add_(
            backward_loss.detach().to(device=self.device, dtype=torch.float64)
        )
        values["microbatch_count"].add_(1.0)
        self._microbatch_updates += 1
        values["sample_count"].add_(float(batch.batch_size))
        values["language_sum"].add_(
            loss_output.language_sum.detach().to(
                device=self.device, dtype=torch.float64
            )
        )
        values["language_token_count"].add_(
            loss_output.language_token_count.detach().to(
                device=self.device, dtype=torch.float64
            )
        )
        values[f"task/{batch.task.value}"].add_(1.0)

        segmentation = loss_output.segmentation
        if segmentation is not None:
            values["dice_sum"].add_(
                segmentation.dice_sum.detach().to(
                    device=self.device, dtype=torch.float64
                )
            )
            values["segmentation_sample_count"].add_(
                segmentation.sample_count.detach().to(
                    device=self.device, dtype=torch.float64
                )
            )
            values["bce_sum"].add_(
                segmentation.bce_sum.detach().to(
                    device=self.device, dtype=torch.float64
                )
            )
            values["segmentation_voxel_count"].add_(
                segmentation.voxel_count.detach().to(
                    device=self.device, dtype=torch.float64
                )
            )
            values["foreground_voxel_count"].add_(
                segmentation.foreground_voxel_count.detach().to(
                    device=self.device, dtype=torch.float64
                )
            )
            values["empty_target_count"].add_(
                segmentation.empty_target_count.detach().to(
                    device=self.device, dtype=torch.float64
                )
            )
            values["legacy_minus_one_voxel_count"].add_(
                segmentation.legacy_minus_one_voxel_count.detach().to(
                    device=self.device, dtype=torch.float64
                )
            )

    def update_optimizer(self, gradient_norm: Tensor) -> None:
        if not isinstance(gradient_norm, Tensor) or gradient_norm.numel() != 1:
            raise TypeError("gradient_norm must be a scalar Tensor")
        self._values["gradient_norm_sum"].add_(
            gradient_norm.detach().to(device=self.device, dtype=torch.float64)
        )
        self._values["gradient_norm_count"].add_(1.0)

    def flush(
        self,
        runtime: _MetricRuntime,
        *,
        learning_rates: Mapping[str, float],
        cuda_memory: Mapping[str, float] | None = None,
    ) -> dict[str, float]:
        """Reduce accumulated sums and return globally meaningful metrics."""

        elapsed = max(time.monotonic() - self._started_at, 1.0e-12)
        reduced = runtime.reduce_scalar_dict(self._values, average=False)
        world_size = int(runtime.world_size)
        if world_size <= 0:
            raise TrainerConfigurationError("runtime.world_size must be positive")

        microbatch_replicas = reduced["microbatch_count"]
        logical_microbatches = microbatch_replicas / float(world_size)
        if logical_microbatches <= 0.0:
            self.reset()
            return {}

        def ratio(numerator: str, denominator: str) -> float:
            denominator_value = reduced[denominator]
            if denominator_value <= 0.0:
                return 0.0
            return reduced[numerator] / denominator_value

        metrics: dict[str, float] = {
            "loss/objective_microbatch_mean": ratio(
                "objective_contribution_sum", "microbatch_count"
            ),
            "loss/language_token_mean": ratio(
                "language_sum", "language_token_count"
            ),
            "loss/dice_sample_mean": ratio(
                "dice_sum", "segmentation_sample_count"
            ),
            "loss/bce_voxel_mean": ratio(
                "bce_sum", "segmentation_voxel_count"
            ),
            "count/language_tokens": reduced["language_token_count"],
            "count/segmentation_samples": reduced[
                "segmentation_sample_count"
            ],
            "count/segmentation_voxels": reduced[
                "segmentation_voxel_count"
            ],
            "count/foreground_voxels": reduced["foreground_voxel_count"],
            "count/empty_targets": reduced["empty_target_count"],
            "count/legacy_minus_one_voxels": reduced[
                "legacy_minus_one_voxel_count"
            ],
            "throughput/global_samples_per_second": (
                reduced["sample_count"] / elapsed
            ),
            "throughput/microbatches_per_second": logical_microbatches / elapsed,
            "time/log_window_seconds": elapsed,
        }
        if reduced["gradient_norm_count"] > 0.0:
            metrics["optim/gradient_norm"] = ratio(
                "gradient_norm_sum", "gradient_norm_count"
            )

        for task in TASK_ORDER:
            logical_count = reduced[f"task/{task.value}"] / float(world_size)
            metrics[f"task_microbatches/{task.value}"] = logical_count
            metrics[f"task_fraction/{task.value}"] = (
                logical_count / logical_microbatches
            )

        for group_name, learning_rate in sorted(learning_rates.items()):
            metrics[f"lr/{group_name}"] = float(learning_rate)
        if cuda_memory is not None:
            metrics.update(
                {
                    f"cuda/{name}": float(value)
                    for name, value in sorted(cuda_memory.items())
                }
            )

        self.reset()
        return metrics


class TrainingEventLogger:
    """Rank-zero JSON and optional TensorBoard logging."""

    def __init__(self, config: ExperimentConfig, runtime: RuntimeContext) -> None:
        self.config = config
        self.runtime = runtime
        self._writer: Any | None = None
        if runtime.is_main_process and "tensorboard" in config.logging.report_to:
            try:
                from torch.utils.tensorboard import SummaryWriter
            except Exception as exc:  # pragma: no cover - environment dependent
                raise TrainerConfigurationError(
                    "logging.report_to requests TensorBoard, but "
                    "torch.utils.tensorboard could not be imported."
                ) from exc
            directory = Path(config.logging.tensorboard_dir).expanduser().resolve()
            directory.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(directory))

    def log(
        self,
        metrics: Mapping[str, float],
        *,
        optimizer_step: int,
        epoch: int,
        committed_microbatches: int,
    ) -> None:
        if not self.runtime.is_main_process or not metrics:
            return
        payload = {
            "optimizer_step": int(optimizer_step),
            "epoch": int(epoch),
            "committed_microbatches": int(committed_microbatches),
            **{name: float(value) for name, value in sorted(metrics.items())},
        }
        self.runtime.logger.info("train_metrics %s", json.dumps(payload, sort_keys=True))
        if self._writer is not None:
            for name, value in metrics.items():
                self._writer.add_scalar(name, float(value), optimizer_step)
            self._writer.add_scalar("progress/epoch", float(epoch), optimizer_step)
            self._writer.add_scalar(
                "progress/committed_microbatches",
                float(committed_microbatches),
                optimizer_step,
            )

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
            self._writer = None


def _loss_finite_flag(loss_output: M3DLossOutput) -> Tensor:
    tensors = [loss_output.language_sum]
    if loss_output.segmentation is not None:
        tensors.extend(
            [
                loss_output.segmentation.dice_sum,
                loss_output.segmentation.bce_sum,
            ]
        )
    flags = torch.stack(
        [torch.isfinite(value.detach()).to(dtype=torch.int64) for value in tensors]
    )
    return flags.prod()


def _global_loss_normalisers(
    *,
    runtime: RuntimeContext,
    batch: M3DBatch,
    loss_output: M3DLossOutput,
) -> GlobalLossNormalisers:
    """All-reduce task identity, finite status, and normalising counts once."""

    segmentation = loss_output.segmentation
    zero = torch.zeros((), device=loss_output.device, dtype=torch.int64)
    segmentation_samples = zero if segmentation is None else segmentation.sample_count
    segmentation_voxels = zero if segmentation is None else segmentation.voxel_count

    local_task_id = torch.tensor(
        int(batch.task_id),
        device=loss_output.device,
        dtype=torch.int64,
    )
    packed = torch.stack(
        [
            local_task_id,
            local_task_id.square(),
            _loss_finite_flag(loss_output).to(
                device=loss_output.device, dtype=torch.int64
            ),
            loss_output.language_token_count.detach().to(
                device=loss_output.device, dtype=torch.int64
            ),
            segmentation_samples.detach().to(
                device=loss_output.device, dtype=torch.int64
            ),
            segmentation_voxels.detach().to(
                device=loss_output.device, dtype=torch.int64
            ),
        ]
    )
    reduced = runtime.all_reduce_sum(packed)
    world_size = int(runtime.world_size)
    # Zero integer variance proves every rank supplied the same task ID. Unlike
    # comparing against the local ID, this produces the same pass/fail decision
    # on every rank even when rank tasks have diverged.
    task_ok = reduced[1] * world_size == reduced[0].square()
    finite_ok = reduced[2].eq(world_size)
    counts_ok = reduced[3].gt(0)
    if batch.task.requires_segmentation_target:
        counts_ok = counts_ok & reduced[4].gt(0) & reduced[5].gt(0)
    else:
        counts_ok = counts_ok & reduced[4].eq(0) & reduced[5].eq(0)

    # One host synchronisation checks all invariants after the already-required
    # count collective.  This makes every rank raise before backward instead of
    # allowing one rank to enter DDP/FSDP communication with invalid loss.
    if not bool((task_ok & finite_ok & counts_ok).item()):
        diagnostic = {
            "local_task": batch.task.value,
            "reduced_task_id_sum": int(reduced[0].item()),
            "reduced_task_id_square_sum": int(reduced[1].item()),
            "finite_rank_count": int(reduced[2].item()),
            "world_size": runtime.world_size,
            "global_language_tokens": int(reduced[3].item()),
            "global_segmentation_samples": int(reduced[4].item()),
            "global_segmentation_voxels": int(reduced[5].item()),
        }
        if not bool(finite_ok.item()):
            raise NonFiniteTrainingError(
                "At least one rank produced a non-finite loss component: "
                f"{diagnostic}."
            )
        raise TrainingProgressError(
            "Distributed task/count contract failed before backward: "
            f"{diagnostic}."
        )

    return GlobalLossNormalisers(
        language_token_count=reduced[3],
        segmentation_sample_count=(reduced[4] if segmentation is not None else None),
        segmentation_voxel_count=(reduced[5] if segmentation is not None else None),
    )


def _compose_backward_loss(
    *,
    runtime: RuntimeContext,
    batch: M3DBatch,
    loss_output: M3DLossOutput,
) -> Tensor:
    normalisers = _global_loss_normalisers(
        runtime=runtime,
        batch=batch,
        loss_output=loss_output,
    )
    backward_loss = compose_data_parallel_backward_loss(
        loss_output,
        world_size=runtime.world_size,
        global_language_token_count=normalisers.language_token_count,
        global_segmentation_sample_count=normalisers.segmentation_sample_count,
        global_segmentation_voxel_count=normalisers.segmentation_voxel_count,
    )
    finite_rank_count = runtime.all_reduce_sum(
        torch.isfinite(backward_loss.detach()).to(dtype=torch.int64)
    )
    if int(finite_rank_count.item()) != int(runtime.world_size):
        raise NonFiniteTrainingError(
            "At least one rank produced a non-finite global-normalised "
            "backward loss."
        )
    return backward_loss


def _max_cuda_memory(runtime: RuntimeContext) -> dict[str, float]:
    """Return maximum allocator counters across ranks in GiB."""

    if runtime.device.type != "cuda" or not torch.cuda.is_available():
        return {}
    gib = float(1024**3)
    local = torch.tensor(
        [
            torch.cuda.memory_allocated(runtime.device) / gib,
            torch.cuda.memory_reserved(runtime.device) / gib,
            torch.cuda.max_memory_allocated(runtime.device) / gib,
            torch.cuda.max_memory_reserved(runtime.device) / gib,
        ],
        device=runtime.device,
        dtype=torch.float64,
    )
    if runtime.process_group_initialized:
        torch.distributed.all_reduce(local, op=torch.distributed.ReduceOp.MAX)
    values = local.cpu().tolist()
    return {
        "allocated_gib_max_rank": float(values[0]),
        "reserved_gib_max_rank": float(values[1]),
        "peak_allocated_gib_max_rank": float(values[2]),
        "peak_reserved_gib_max_rank": float(values[3]),
    }


@contextlib.contextmanager
def _maybe_profile_update(
    *,
    config: ExperimentConfig,
    runtime: RuntimeContext,
    upcoming_optimizer_step: int,
    is_update_microbatch: bool,
) -> Generator[None, None, None]:
    targets = set(int(step) for step in config.logging.profile_steps)
    enabled = is_update_microbatch and upcoming_optimizer_step in targets
    if not enabled:
        yield
        return

    activities = [torch.profiler.ProfilerActivity.CPU]
    if runtime.device.type == "cuda" and torch.cuda.is_available():
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    output_root = (
        Path(config.checkpoint.output_dir).expanduser().resolve()
        / "profiles"
    )
    output_root.mkdir(parents=True, exist_ok=True)
    trace_path = output_root / (
        f"rank-{runtime.rank:05d}-optimizer-step-"
        f"{upcoming_optimizer_step:08d}.json"
    )
    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
    ) as profiler:
        yield
    profiler.export_chrome_trace(str(trace_path))
    runtime.logger.info("Profiler trace written to %s", trace_path)


class M3DTrainer:
    """Complete task-aware training engine for one distributed rank."""

    def __init__(
        self,
        *,
        config: ExperimentConfig,
        runtime: RuntimeContext,
        distributed_model: DistributedM3DModel,
        optimizer: Optimizer,
        scheduler: CosineWarmupScheduler,
        data_pipeline: "TrainingDataPipeline",
        checkpoint_manager: CheckpointManager | None = None,
        event_logger: TrainingEventLogger | None = None,
    ) -> None:
        if not isinstance(config, ExperimentConfig):
            raise TypeError("config must be ExperimentConfig")
        if not isinstance(runtime, RuntimeContext):
            raise TypeError("runtime must be RuntimeContext")
        if not isinstance(distributed_model, DistributedM3DModel):
            raise TypeError("distributed_model must be DistributedM3DModel")
        if not isinstance(optimizer, Optimizer):
            raise TypeError("optimizer must be torch.optim.Optimizer")
        if not isinstance(scheduler, CosineWarmupScheduler):
            raise TypeError("scheduler must be CosineWarmupScheduler")
        required_pipeline_attributes = (
            "runtime",
            "batch_sampler",
            "loader",
            "epoch",
            "committed_step",
            "steps_per_epoch",
            "non_blocking_transfer",
            "set_epoch",
            "commit_batch",
        )
        missing_pipeline_attributes = [
            name for name in required_pipeline_attributes
            if not hasattr(data_pipeline, name)
        ]
        if missing_pipeline_attributes:
            raise TypeError(
                "data_pipeline does not satisfy TrainingDataPipeline contract; "
                f"missing={missing_pipeline_attributes}."
            )

        self.config = config
        self.runtime = runtime
        self.distributed_model = distributed_model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.data_pipeline = data_pipeline
        self.checkpoint_manager = checkpoint_manager or CheckpointManager(
            config=config,
            runtime=runtime,
            distributed_model=distributed_model,
            optimizer=optimizer,
            scheduler=scheduler,
            data_pipeline=data_pipeline,
        )
        self.event_logger = event_logger or TrainingEventLogger(config, runtime)
        self.metrics = MetricAccumulator(runtime.device)
        self._last_checkpoint_report: CheckpointSaveReport | None = None
        self._resumed_report: CheckpointResumeReport | None = None
        self._closed = False
        self._validate_wiring()

    def _validate_wiring(self) -> None:
        self.config.validate()
        if self.data_pipeline.runtime is not self.runtime:
            raise TrainerConfigurationError(
                "TrainingDataPipeline and trainer must share one RuntimeContext."
            )
        if self.distributed_model.runtime is not self.runtime:
            raise TrainerConfigurationError(
                "Distributed model and trainer must share one RuntimeContext."
            )
        if self.scheduler.optimizer is not self.optimizer:
            raise TrainerConfigurationError(
                "Scheduler does not reference the trainer optimizer."
            )
        if self.checkpoint_manager.optimizer is not self.optimizer:
            raise TrainerConfigurationError(
                "Checkpoint manager does not reference the trainer optimizer."
            )
        if self.checkpoint_manager.scheduler is not self.scheduler:
            raise TrainerConfigurationError(
                "Checkpoint manager does not reference the trainer scheduler."
            )
        if self.checkpoint_manager.data_pipeline is not self.data_pipeline:
            raise TrainerConfigurationError(
                "Checkpoint manager does not reference the trainer data pipeline."
            )
        if self.scheduler.plan.steps_per_epoch != self.data_pipeline.steps_per_epoch:
            raise TrainerConfigurationError(
                "Scheduler/data steps_per_epoch mismatch: "
                f"scheduler={self.scheduler.plan.steps_per_epoch}, "
                f"data={self.data_pipeline.steps_per_epoch}."
            )
        if self.runtime.world_size <= 1:
            raise TrainerConfigurationError(
                "M3DTrainer expects the distributed DDP/FSDP2 runtime path."
            )
        optimizer_groups_by_name(self.optimizer)
        profile_steps = tuple(int(step) for step in self.config.logging.profile_steps)
        invalid_profile_steps = [
            step
            for step in profile_steps
            if step <= 0 or step > self.scheduler.total_optimizer_steps
        ]
        if invalid_profile_steps:
            raise TrainerConfigurationError(
                "logging.profile_steps contains optimizer steps outside the "
                f"training plan: {invalid_profile_steps}."
            )

    def build_report(self) -> TrainerBuildReport:
        return TrainerBuildReport(
            strategy=self.distributed_model.strategy,
            world_size=self.runtime.world_size,
            per_device_batch_size=self.config.optimization.per_device_batch_size,
            gradient_accumulation_steps=(
                self.config.optimization.gradient_accumulation_steps
            ),
            epoch_microbatch_limits=self.scheduler.plan.epoch_microbatch_limits,
            optimizer_steps_per_epoch=(
                self.scheduler.plan.optimizer_steps_per_epoch
            ),
            total_optimizer_steps=self.scheduler.total_optimizer_steps,
            log_every_optimizer_steps=self.config.logging.log_every_steps,
            checkpoint_every_optimizer_steps=(
                self.config.checkpoint.save_every_steps
            ),
            profile_optimizer_steps=tuple(
                int(step) for step in self.config.logging.profile_steps
            ),
        )

    def write_build_report(self) -> None:
        if not self.runtime.is_main_process:
            return
        path = (
            Path(self.config.checkpoint.output_dir).expanduser().resolve()
            / "trainer_build_report.json"
        )
        atomic_write_json(path, self.build_report().to_dict())

    def resume_if_configured(self) -> CheckpointResumeReport | None:
        resume_from = self.config.checkpoint.resume_from
        if resume_from is None:
            validate_resume_progress(
                self.scheduler,
                epoch=self.data_pipeline.epoch,
                committed_microbatches=self.data_pipeline.committed_step,
            )
            return None
        report = self.checkpoint_manager.load(resume_from, exact=True)
        self._resumed_report = report
        self.runtime.logger.info(
            "Exact checkpoint resume complete: %s",
            json.dumps(report.to_dict(), sort_keys=True),
        )
        return report

    def _validate_batch_position(
        self,
        batch: M3DBatch,
        *,
        epoch: int,
        epoch_limit: int,
    ) -> int:
        observed_epoch, observed_step, observed_task = sampler_position_from_batch(batch)
        if observed_epoch != epoch:
            raise TrainingProgressError(
                f"DataLoader yielded epoch {observed_epoch}, expected {epoch}."
            )
        if observed_step != self.data_pipeline.committed_step:
            raise TrainingProgressError(
                "DataLoader/sampler cursor mismatch: "
                f"yielded_step={observed_step}, "
                f"committed_step={self.data_pipeline.committed_step}."
            )
        if observed_task is not batch.task:
            raise TrainingProgressError(
                "Batch task disagrees with sampler provenance: "
                f"batch={batch.task.value}, sampler={observed_task.value}."
            )
        if observed_step >= epoch_limit:
            raise TrainingProgressError(
                f"Trainer attempted to consume step {observed_step} beyond the "
                f"planned epoch limit {epoch_limit}."
            )
        expected_task = self.data_pipeline.batch_sampler.schedule.task_at(observed_step)
        if expected_task is not batch.task:
            raise TrainingProgressError(
                "Batch task disagrees with deterministic schedule: "
                f"step={observed_step}, expected={expected_task.value}, "
                f"observed={batch.task.value}."
            )
        return observed_step

    def _log_if_due(self, *, force: bool = False) -> None:
        if self.metrics.is_empty:
            return
        step = self.scheduler.completed_optimizer_steps
        due = step > 0 and step % int(self.config.logging.log_every_steps) == 0
        if not force and not due:
            return
        learning_rates = self.scheduler.learning_rates_by_group()
        memory = (
            _max_cuda_memory(self.runtime)
            if self.config.logging.log_gpu_memory
            else None
        )
        metrics = self.metrics.flush(
            self.runtime,
            learning_rates=learning_rates,
            cuda_memory=memory,
        )
        self.event_logger.log(
            metrics,
            optimizer_step=step,
            epoch=self.data_pipeline.epoch,
            committed_microbatches=self.data_pipeline.committed_step,
        )

    def _save_if_due(self, *, final_update: bool = False) -> None:
        if not should_save_checkpoint(
            self.scheduler,
            self.config.checkpoint,
            final_update=final_update,
        ):
            return
        pending = self.checkpoint_manager.save(
            force_synchronous=bool(final_update)
        )
        if not pending.asynchronous:
            self._last_checkpoint_report = pending.wait()
            self.runtime.logger.info(
                "Checkpoint saved: %s",
                json.dumps(
                    self._last_checkpoint_report.to_dict(), sort_keys=True
                ),
            )

    def _run_microbatch(
        self,
        cpu_batch: M3DBatch,
        *,
        epoch: int,
        epoch_limit: int,
    ) -> None:
        microbatch_index = self._validate_batch_position(
            cpu_batch,
            epoch=epoch,
            epoch_limit=epoch_limit,
        )
        should_update = self.scheduler.plan.is_optimizer_update_microbatch(
            epoch,
            microbatch_index,
        )
        accumulation_window = self.scheduler.plan.accumulation_window_size(
            epoch,
            microbatch_index,
        )
        if accumulation_window <= 0:
            raise TrainingProgressError("Accumulation window cannot be empty.")

        gpu_batch = cpu_batch.to(
            self.runtime.device,
            non_blocking=self.data_pipeline.non_blocking_transfer,
        )
        upcoming_optimizer_step = self.scheduler.completed_optimizer_steps + 1
        gradient_norm: Tensor | None = None
        backward_loss: Tensor | None = None
        output_loss: M3DLossOutput | None = None

        with _maybe_profile_update(
            config=self.config,
            runtime=self.runtime,
            upcoming_optimizer_step=upcoming_optimizer_step,
            is_update_microbatch=should_update,
        ):
            with self.distributed_model.gradient_sync(enabled=should_update):
                with self.runtime.autocast():
                    output = self.distributed_model.forward_batch(
                        gpu_batch,
                        logits_mode="none",
                        return_intermediates=False,
                    )
                    if output.loss_output is None:
                        raise TrainerError(
                            "Training forward returned no loss_output."
                        )
                    output_loss = output.loss_output
                    backward_loss = _compose_backward_loss(
                        runtime=self.runtime,
                        batch=gpu_batch,
                        loss_output=output_loss,
                    )
                    scaled_backward_loss = (
                        backward_loss / float(accumulation_window)
                    )
                scaled_backward_loss.backward()

            if should_update:
                gradient_norm = self.distributed_model.clip_grad_norm_(
                    self.config.optimization.max_grad_norm
                )
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)
                # The durable cursor advances only after the model update and
                # scheduler counter have both succeeded.
                self.data_pipeline.commit_batch(cpu_batch)
            else:
                # Mid-window gradients are intentionally not checkpointable, but
                # the in-process cursor still records successful consumption.
                self.data_pipeline.commit_batch(cpu_batch)

        if output_loss is None or backward_loss is None:
            raise TrainerError("Microbatch completed without a loss record.")
        self.metrics.update_microbatch(
            batch=gpu_batch,
            loss_output=output_loss,
            backward_loss=backward_loss,
        )
        if gradient_norm is not None:
            self.metrics.update_optimizer(gradient_norm)

        if should_update:
            final_update = self.scheduler.is_finished
            self._log_if_due(force=final_update)
            self._save_if_due(final_update=final_update)

        del gpu_batch, output_loss, backward_loss

    def train(self, *, resume: bool = True) -> TrainingResult:
        if self._closed:
            raise TrainerError("Trainer has already been closed.")
        started = time.monotonic()
        self.write_build_report()
        self.distributed_model.train()
        self.optimizer.zero_grad(set_to_none=True)
        if self.runtime.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.runtime.device)

        resumed = self.resume_if_configured() if resume else None
        start_epoch = self.data_pipeline.epoch
        if not 0 <= start_epoch < self.scheduler.plan.epoch_count:
            if self.scheduler.is_finished and start_epoch == (
                self.scheduler.plan.epoch_count - 1
            ):
                pass
            else:
                raise TrainingProgressError(
                    f"Resume epoch {start_epoch} is outside the training plan."
                )

        normal_completion = False
        try:
            for epoch in range(start_epoch, self.scheduler.plan.epoch_count):
                epoch_limit = self.scheduler.plan.microbatches_for_epoch(epoch)
                committed = (
                    self.data_pipeline.committed_step
                    if self.data_pipeline.epoch == epoch
                    else 0
                )
                self.data_pipeline.set_epoch(epoch, committed_step=committed)
                if committed > epoch_limit:
                    raise TrainingProgressError(
                        f"Committed microbatches {committed} exceed epoch {epoch} "
                        f"limit {epoch_limit}."
                    )
                if committed == epoch_limit:
                    continue

                for cpu_batch in self.data_pipeline.loader:
                    _, observed_step, _ = sampler_position_from_batch(cpu_batch)
                    if observed_step >= epoch_limit:
                        # A fractional final epoch consumes only a prefix of the
                        # sampler's deterministic full-epoch schedule.
                        break
                    self._run_microbatch(
                        cpu_batch,
                        epoch=epoch,
                        epoch_limit=epoch_limit,
                    )

                if self.data_pipeline.committed_step != epoch_limit:
                    raise TrainingProgressError(
                        "Epoch ended before the planned microbatch limit: "
                        f"epoch={epoch}, committed={self.data_pipeline.committed_step}, "
                        f"expected={epoch_limit}."
                    )
                self._log_if_due(force=True)

            if not self.scheduler.is_finished:
                raise TrainingProgressError(
                    "All epoch loops completed before the scheduler reached its "
                    f"planned optimizer steps: completed="
                    f"{self.scheduler.completed_optimizer_steps}, total="
                    f"{self.scheduler.total_optimizer_steps}."
                )
            # The final update path already requests a synchronous checkpoint.
            # This call is idempotent with respect to the save cadence only when
            # no final checkpoint was started (for example, a zero-length resume).
            if self.checkpoint_manager.pending is None:
                latest = None
                try:
                    latest = self.checkpoint_manager.resolve_resume_path()
                except Exception:
                    latest = None
                if latest is None:
                    self._save_if_due(final_update=True)

            final_report = self.checkpoint_manager.close()
            if final_report is not None:
                self._last_checkpoint_report = final_report
            normal_completion = True
        except BaseException:
            # Never create an emergency checkpoint from an unknown accumulation
            # position.  The last completed checkpoint remains the recovery
            # point.  Clearing gradients prevents accidental reuse if a caller
            # catches the exception in-process.
            self.optimizer.zero_grad(set_to_none=True)
            self.runtime.logger.exception(
                "M3D training aborted at optimizer_step=%d epoch=%d "
                "committed_microbatches=%d",
                self.scheduler.completed_optimizer_steps,
                self.data_pipeline.epoch,
                self.data_pipeline.committed_step,
            )
            raise
        finally:
            self.event_logger.close()
            self._closed = True
            if normal_completion:
                self.runtime.logger.info("M3D trainer closed cleanly.")

        elapsed = time.monotonic() - started
        result = TrainingResult(
            completed_optimizer_steps=self.scheduler.completed_optimizer_steps,
            total_optimizer_steps=self.scheduler.total_optimizer_steps,
            final_epoch=self.data_pipeline.epoch,
            final_committed_microbatches=self.data_pipeline.committed_step,
            elapsed_seconds=elapsed,
            final_checkpoint=(
                self._last_checkpoint_report.checkpoint_path
                if self._last_checkpoint_report is not None
                else (None if resumed is None else resumed.checkpoint_path)
            ),
            resumed_from=(
                None if resumed is None else resumed.checkpoint_path
            ),
            finished_at_unix=time.time(),
        )
        if self.runtime.is_main_process:
            output = (
                Path(self.config.checkpoint.output_dir).expanduser().resolve()
                / "training_result.json"
            )
            atomic_write_json(output, result.to_dict())
        return result


# ---------------------------------------------------------------------------
# Dependency-light self-test
# ---------------------------------------------------------------------------


class _LocalMetricRuntime:
    world_size = 1

    def reduce_scalar_dict(
        self,
        values: Mapping[str, float | int | Tensor],
        *,
        average: bool = True,
    ) -> dict[str, float]:
        result: dict[str, float] = {}
        for name, value in values.items():
            if isinstance(value, Tensor):
                result[name] = float(value.detach().cpu())
            else:
                result[name] = float(value)
        return result


def _self_test() -> dict[str, Any]:
    from .model.loss import SegmentationLossOutput

    device = torch.device("cpu")
    accumulator = MetricAccumulator(device)
    language_sum = torch.tensor(6.0)
    language_count = torch.tensor(3, dtype=torch.int64)
    segmentation = SegmentationLossOutput(
        total=torch.tensor(1.5),
        dice=torch.tensor(0.5),
        bce=torch.tensor(1.0),
        dice_sum=torch.tensor(1.0),
        bce_sum=torch.tensor(8.0),
        sample_count=torch.tensor(2, dtype=torch.int64),
        voxel_count=torch.tensor(8, dtype=torch.int64),
        foreground_voxel_count=torch.tensor(3, dtype=torch.int64),
        empty_target_count=torch.tensor(1, dtype=torch.int64),
        legacy_minus_one_voxel_count=torch.tensor(0, dtype=torch.int64),
        dice_weight=1.0,
        bce_weight=1.0,
    )
    loss_output = M3DLossOutput(
        task=TaskName.SEGMENTATION,
        total=torch.tensor(3.5),
        language=torch.tensor(2.0),
        language_sum=language_sum,
        language_token_count=language_count,
        segmentation=segmentation,
    )

    # A minimal M3DBatch is not required to test tensor collation internals;
    # construct one through object.__new__ and set only fields consumed by the
    # accumulator. This keeps the trainer self-test independent of tokenizers.
    batch = cast(M3DBatch, object.__new__(M3DBatch))
    object.__setattr__(batch, "task", TaskName.SEGMENTATION)
    object.__setattr__(batch, "sample_ids", ("a", "b"))
    object.__setattr__(batch, "images", torch.empty(2, 1, 1, 1, 1))
    object.__setattr__(batch, "text", None)
    object.__setattr__(batch, "segmentation_targets", None)
    object.__setattr__(batch, "provenance", ())

    accumulator.update_microbatch(
        batch=batch,
        loss_output=loss_output,
        backward_loss=torch.tensor(3.5),
    )
    accumulator.update_optimizer(torch.tensor(2.25))
    metrics = accumulator.flush(
        _LocalMetricRuntime(),
        learning_rates={"main_vision/decay": 5.0e-6},
    )

    finite = int(_loss_finite_flag(loss_output)) == 1
    nonfinite_output = dataclasses.replace(
        loss_output,
        language_sum=torch.tensor(float("nan")),
    )
    nonfinite_detected = int(_loss_finite_flag(nonfinite_output)) == 0
    passed = (
        math.isclose(metrics["loss/language_token_mean"], 2.0)
        and math.isclose(metrics["loss/dice_sample_mean"], 0.5)
        and math.isclose(metrics["loss/bce_voxel_mean"], 1.0)
        and math.isclose(metrics["optim/gradient_norm"], 2.25)
        and math.isclose(metrics["task_fraction/segmentation"], 1.0)
        and finite
        and nonfinite_detected
    )
    if not passed:
        raise AssertionError(f"Trainer self-test failed: {metrics}")
    return {
        "status": "passed",
        "language_token_mean": metrics["loss/language_token_mean"],
        "dice_sample_mean": metrics["loss/dice_sample_mean"],
        "bce_voxel_mean": metrics["loss/bce_voxel_mean"],
        "gradient_norm": metrics["optim/gradient_norm"],
        "segmentation_task_fraction": metrics["task_fraction/segmentation"],
        "finite_loss_detected": finite,
        "nonfinite_loss_detected": nonfinite_detected,
    }


__all__ = [
    "GlobalLossNormalisers",
    "M3DTrainer",
    "MetricAccumulator",
    "NonFiniteTrainingError",
    "TrainerBuildReport",
    "TrainerConfigurationError",
    "TrainerError",
    "TrainingEventLogger",
    "TrainingProgressError",
    "TrainingResult",
]


if __name__ == "__main__":
    print(json.dumps(_self_test(), indent=2, sort_keys=True))
