"""Build reproducible task-aware DataLoaders for modernized M3D training.

This module is the final assembly point of the data pipeline.  It connects:

* the validated JSONL manifest;
* the task-indexed dataset catalogue;
* one map-style dataset per M3D task;
* the deterministic distributed task sampler;
* task-homogeneous collation with dynamic text padding; and
* PyTorch DataLoader worker, prefetch, pinned-memory, and checkpoint state.

The training path deliberately uses ``batch_sampler=`` rather than combining
``batch_size``, ``shuffle``, and a normal sampler.  The batch sampler owns the
entire global schedule, ensuring every data-parallel rank executes the same
model branch on a given microbatch:

* text/positioning step: Main 3D ViT + language model;
* segmentation step: Main 3D ViT + SegVol 3D ViT + language model.

Evaluation uses one DataLoader per task.  This keeps every evaluation batch
homogeneous without task weighting.  The exact distributed evaluation sampler
never pads with duplicate examples; consequently different ranks can receive a
different number of evaluation batches.  Evaluation code must accumulate local
metrics first and perform collective reduction only after each rank finishes
its local task loader.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any, Generic, TypeVar

import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from m3d.config import ExperimentConfig
from m3d.data.collator import (
    M3DCollator,
    build_evaluation_collator,
    build_training_collator,
)
from m3d.data.dataset_catalog import (
    DatasetCatalog,
    load_dataset_catalog,
    validate_catalog_for_config,
)
from m3d.data.datasets import (
    M3DDatasetCollection,
    TextProcessorProtocol,
    build_task_datasets,
)
from m3d.data.sampler import (
    DistributedTaskBatchSampler,
    TaskMultiplexDataset,
    build_task_batch_sampler,
    validate_distributed_schedule,
)
from m3d.data.schema import DataSplit, M3DBatch, TaskName
from m3d.runtime import (
    RuntimeContext,
    dataloader_worker_init_fn,
    make_dataloader_generator,
)
from m3d.tokenization import TokenizerBundle


class DataLoaderBuildError(RuntimeError):
    """Raised when a configured M3D DataLoader cannot be constructed safely."""


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class WorkerSettings:
    """Resolved PyTorch DataLoader worker settings.

    ``prefetch_factor`` and ``persistent_workers`` are meaningful only when
    workers exist.  Keeping the resolved values explicit avoids accidentally
    passing unsupported combinations into :class:`torch.utils.data.DataLoader`.
    """

    num_workers: int
    pin_memory: bool
    persistent_workers: bool
    prefetch_factor: int | None
    non_blocking_transfer: bool

    def __post_init__(self) -> None:
        if not isinstance(self.num_workers, int) or isinstance(self.num_workers, bool):
            raise TypeError("num_workers must be an integer")
        if self.num_workers < 0:
            raise DataLoaderBuildError("num_workers cannot be negative")
        if self.num_workers == 0:
            if self.persistent_workers:
                raise DataLoaderBuildError(
                    "persistent_workers requires num_workers > 0"
                )
            if self.prefetch_factor is not None:
                raise DataLoaderBuildError(
                    "prefetch_factor must be None when num_workers == 0"
                )
        elif self.prefetch_factor is None or self.prefetch_factor <= 0:
            raise DataLoaderBuildError(
                "A positive prefetch_factor is required when num_workers > 0"
            )

    @classmethod
    def from_config(cls, config: ExperimentConfig) -> "WorkerSettings":
        data = config.data
        workers = int(data.num_workers)
        return cls(
            num_workers=workers,
            pin_memory=bool(data.pin_memory),
            persistent_workers=bool(data.persistent_workers) if workers > 0 else False,
            prefetch_factor=int(data.prefetch_factor) if workers > 0 else None,
            non_blocking_transfer=bool(data.non_blocking_transfer),
        )

    def dataloader_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "worker_init_fn": dataloader_worker_init_fn,
        }
        if self.prefetch_factor is not None:
            kwargs["prefetch_factor"] = self.prefetch_factor
        return kwargs

    def as_dict(self) -> dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "persistent_workers": self.persistent_workers,
            "prefetch_factor": self.prefetch_factor,
            "non_blocking_transfer": self.non_blocking_transfer,
        }


class ExactDistributedEvaluationSampler(Sampler[int]):
    """Shard a map-style evaluation dataset without duplicate padding.

    PyTorch's standard :class:`DistributedSampler` pads indices when a dataset
    size is not divisible by ``world_size``.  Duplicate evaluation examples can
    bias exact metrics, so this sampler instead assigns ``rank, rank+world, ...``.

    The trade-off is that ranks can have different local lengths.  Callers must
    not issue one collective per local evaluation batch.  They should finish
    local accumulation and then reduce aggregate counts/sums once per task.
    """

    def __init__(self, dataset: Dataset[Any], *, rank: int, world_size: int) -> None:
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise TypeError("rank must be an integer")
        if not isinstance(world_size, int) or isinstance(world_size, bool):
            raise TypeError("world_size must be an integer")
        if world_size <= 0:
            raise DataLoaderBuildError("world_size must be positive")
        if not 0 <= rank < world_size:
            raise DataLoaderBuildError(
                f"rank must be in [0, {world_size}), got {rank}"
            )
        size = len(dataset)
        if size < 0:
            raise DataLoaderBuildError("Dataset length cannot be negative")
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.dataset_size = size

    def __iter__(self) -> Iterator[int]:
        return iter(range(self.rank, self.dataset_size, self.world_size))

    def __len__(self) -> int:
        if self.rank >= self.dataset_size:
            return 0
        return math.ceil((self.dataset_size - self.rank) / self.world_size)


@dataclass(slots=True)
class TrainingDataPipeline:
    """Stateful training DataLoader and its durable task-sampler cursor."""

    catalog: DatasetCatalog
    datasets: M3DDatasetCollection
    multiplexed_dataset: TaskMultiplexDataset
    batch_sampler: DistributedTaskBatchSampler
    collator: M3DCollator
    loader: DataLoader[M3DBatch]
    runtime: RuntimeContext
    workers: WorkerSettings

    def __post_init__(self) -> None:
        if self.catalog.split is not DataSplit.TRAIN:
            raise DataLoaderBuildError(
                f"Training pipeline requires train split, got {self.catalog.split.value!r}"
            )
        if self.datasets.split is not DataSplit.TRAIN:
            raise DataLoaderBuildError("Training datasets do not belong to train split")
        if self.runtime.rank != self.batch_sampler.rank:
            raise DataLoaderBuildError("Runtime rank and task-sampler rank differ")
        if self.runtime.world_size != self.batch_sampler.world_size:
            raise DataLoaderBuildError("Runtime world size and sampler world size differ")

    def __len__(self) -> int:
        """Return remaining microbatches, not the full epoch length after resume."""

        return len(self.batch_sampler)

    @property
    def epoch(self) -> int:
        return self.batch_sampler.epoch

    @property
    def committed_step(self) -> int:
        return self.batch_sampler.committed_step

    @property
    def steps_per_epoch(self) -> int:
        return self.batch_sampler.steps_per_epoch

    @property
    def non_blocking_transfer(self) -> bool:
        return self.workers.non_blocking_transfer

    def set_epoch(self, epoch: int, *, committed_step: int = 0) -> None:
        """Build the deterministic schedule for an epoch and update workers.

        The dataset collection uses shared memory for its epoch, so persistent
        workers observe this update without being recreated.
        """

        self.batch_sampler.set_epoch(epoch, committed_step=committed_step)
        validate_distributed_schedule(self.batch_sampler, self.runtime)

    def commit_batch(self, batch: M3DBatch) -> None:
        """Advance the checkpoint-safe cursor after successful model consumption."""

        self.batch_sampler.commit_batch(batch)

    def state_dict(self) -> dict[str, Any]:
        return {
            "state_version": 1,
            "catalog_manifest_fingerprint": self.catalog.manifest_fingerprint,
            "workers": self.workers.as_dict(),
            "sampler": self.batch_sampler.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if not isinstance(state, Mapping):
            raise TypeError("Training data state must be a mapping")
        if state.get("state_version") != 1:
            raise DataLoaderBuildError(
                f"Unsupported training data state version {state.get('state_version')!r}"
            )
        checkpoint_manifest = state.get("catalog_manifest_fingerprint")
        if checkpoint_manifest != self.catalog.manifest_fingerprint:
            raise DataLoaderBuildError(
                "Manifest changed since checkpoint: "
                f"checkpoint={checkpoint_manifest!r}, "
                f"current={self.catalog.manifest_fingerprint!r}"
            )
        self.batch_sampler.load_state_dict(state.get("sampler", {}))
        validate_distributed_schedule(self.batch_sampler, self.runtime)

    def summary(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "split": self.catalog.split.value,
                "manifest_fingerprint": self.catalog.manifest_fingerprint,
                "tasks": [task.value for task in self.datasets.tasks],
                "worker_settings": self.workers.as_dict(),
                "sampler": self.batch_sampler.summary(),
            }
        )


@dataclass(frozen=True, slots=True)
class EvaluationTaskLoader:
    """One exact distributed loader for a single evaluation task."""

    task: TaskName
    dataset_size: int
    local_sample_count: int
    sampler: ExactDistributedEvaluationSampler
    loader: DataLoader[M3DBatch]

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", TaskName.parse(self.task))
        if self.dataset_size < 0 or self.local_sample_count < 0:
            raise DataLoaderBuildError("Evaluation sample counts cannot be negative")
        if self.local_sample_count != len(self.sampler):
            raise DataLoaderBuildError(
                "Evaluation local_sample_count does not match sampler length"
            )


@dataclass(frozen=True, slots=True)
class EvaluationDataPipeline:
    """Task-indexed evaluation loaders for one validation or test split."""

    catalog: DatasetCatalog
    datasets: M3DDatasetCollection
    loaders: Mapping[TaskName, EvaluationTaskLoader]
    workers: WorkerSettings

    def __post_init__(self) -> None:
        if self.catalog.split is DataSplit.TRAIN:
            raise DataLoaderBuildError("Evaluation pipeline cannot use train split")
        parsed: dict[TaskName, EvaluationTaskLoader] = {}
        for task, item in self.loaders.items():
            canonical = TaskName.parse(task)
            if canonical is not item.task:
                raise DataLoaderBuildError(
                    f"Evaluation mapping key {canonical.value!r} differs from loader task"
                )
            parsed[canonical] = item
        object.__setattr__(self, "loaders", MappingProxyType(parsed))

    @property
    def split(self) -> DataSplit:
        return self.catalog.split

    @property
    def tasks(self) -> tuple[TaskName, ...]:
        return tuple(sorted(self.loaders, key=lambda task: int(task.task_id)))

    def __contains__(self, task: object) -> bool:
        try:
            return TaskName.parse(task) in self.loaders  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return False

    def __getitem__(self, task: str | TaskName) -> EvaluationTaskLoader:
        canonical = TaskName.parse(task)
        try:
            return self.loaders[canonical]
        except KeyError as exc:
            available = ", ".join(item.value for item in self.tasks)
            raise KeyError(
                f"Task {canonical.value!r} is absent from {self.split.value}; "
                f"available={available}"
            ) from exc

    def set_epoch(self, epoch: int) -> None:
        """Update deterministic prompt/transform epoch for evaluation datasets."""

        self.datasets.set_epoch(epoch)

    def summary(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "split": self.split.value,
                "manifest_fingerprint": self.catalog.manifest_fingerprint,
                "worker_settings": self.workers.as_dict(),
                "tasks": {
                    task.value: {
                        "global_samples": item.dataset_size,
                        "local_samples": item.local_sample_count,
                        "local_batches": len(item.loader),
                    }
                    for task, item in self.loaders.items()
                },
            }
        )


@dataclass(frozen=True, slots=True)
class DataPipelines:
    """Container returned to the future trainer entry point."""

    train: TrainingDataPipeline
    validation: EvaluationDataPipeline | None = None
    test: EvaluationDataPipeline | None = None

    def summary(self) -> dict[str, Any]:
        return {
            "train": dict(self.train.summary()),
            "validation": (
                None if self.validation is None else dict(self.validation.summary())
            ),
            "test": None if self.test is None else dict(self.test.summary()),
        }


def _common_dataloader_kwargs(
    *,
    workers: WorkerSettings,
    runtime: RuntimeContext,
) -> dict[str, Any]:
    kwargs = workers.dataloader_kwargs()
    kwargs.update(
        {
            "generator": make_dataloader_generator(runtime),
            # Preserve sampler order even if worker completion is out of order.
            "in_order": True,
        }
    )
    return kwargs


def build_training_data_pipeline(
    *,
    config: ExperimentConfig,
    runtime: RuntimeContext,
    tokenizer_bundle: TokenizerBundle,
    text_processor: TextProcessorProtocol,
    catalog: DatasetCatalog | None = None,
    text_cache: Mapping[tuple[str, int], Any] | None = None,
) -> TrainingDataPipeline:
    """Build the exact distributed training input pipeline."""

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be ExperimentConfig")
    if not isinstance(runtime, RuntimeContext):
        raise TypeError("runtime must be RuntimeContext")
    if not isinstance(tokenizer_bundle, TokenizerBundle):
        raise TypeError("tokenizer_bundle must be TokenizerBundle")
    config.validate()

    train_catalog = catalog or load_dataset_catalog(config, DataSplit.TRAIN)
    if train_catalog.split is not DataSplit.TRAIN:
        raise DataLoaderBuildError(
            f"Expected train catalogue, got {train_catalog.split.value!r}"
        )
    validate_catalog_for_config(train_catalog, config)

    datasets = build_task_datasets(
        train_catalog,
        config,
        text_processor,
        text_cache=text_cache,
    )
    multiplexed, batch_sampler = build_task_batch_sampler(
        datasets=datasets,
        config=config,
        runtime=runtime,
    )
    collator = build_training_collator(
        config=config,
        tokenizer_bundle=tokenizer_bundle,
        rank=runtime.rank,
    )
    workers = WorkerSettings.from_config(config)

    kwargs = _common_dataloader_kwargs(workers=workers, runtime=runtime)
    loader: DataLoader[M3DBatch] = DataLoader(
        multiplexed,
        batch_sampler=batch_sampler,
        collate_fn=collator,
        **kwargs,
    )

    pipeline = TrainingDataPipeline(
        catalog=train_catalog,
        datasets=datasets,
        multiplexed_dataset=multiplexed,
        batch_sampler=batch_sampler,
        collator=collator,
        loader=loader,
        runtime=runtime,
        workers=workers,
    )
    if runtime.is_main_process:
        runtime.logger.info(
            "Training DataLoader initialized: %s",
            json.dumps(dict(pipeline.summary()), sort_keys=True, default=str),
        )
    return pipeline


def build_evaluation_data_pipeline(
    *,
    config: ExperimentConfig,
    runtime: RuntimeContext,
    tokenizer_bundle: TokenizerBundle,
    text_processor: TextProcessorProtocol,
    split: str | DataSplit,
    catalog: DatasetCatalog | None = None,
    per_device_batch_size: int | None = None,
    text_cache: Mapping[tuple[str, int], Any] | None = None,
) -> EvaluationDataPipeline:
    """Build one exact distributed DataLoader per evaluation task."""

    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be ExperimentConfig")
    if not isinstance(runtime, RuntimeContext):
        raise TypeError("runtime must be RuntimeContext")
    if not isinstance(tokenizer_bundle, TokenizerBundle):
        raise TypeError("tokenizer_bundle must be TokenizerBundle")
    parsed_split = DataSplit.parse(split)
    if parsed_split is DataSplit.TRAIN:
        raise DataLoaderBuildError("Use build_training_data_pipeline for train split")

    batch_size = (
        config.optimization.per_device_batch_size
        if per_device_batch_size is None
        else per_device_batch_size
    )
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise DataLoaderBuildError("Evaluation per-device batch size must be positive")

    eval_catalog = catalog or load_dataset_catalog(config, parsed_split)
    if eval_catalog.split is not parsed_split:
        raise DataLoaderBuildError(
            f"Expected {parsed_split.value!r} catalogue, got {eval_catalog.split.value!r}"
        )

    datasets = build_task_datasets(
        eval_catalog,
        config,
        text_processor,
        text_cache=text_cache,
        record_augmentation_metadata=False,
    )
    collator = build_evaluation_collator(
        config=config,
        tokenizer_bundle=tokenizer_bundle,
        expected_batch_size=None,
    )
    workers = WorkerSettings.from_config(config)
    common = _common_dataloader_kwargs(workers=workers, runtime=runtime)

    task_loaders: dict[TaskName, EvaluationTaskLoader] = {}
    for task in datasets.tasks:
        dataset = datasets[task]
        sampler = ExactDistributedEvaluationSampler(
            dataset,
            rank=runtime.rank,
            world_size=runtime.world_size,
        )
        loader: DataLoader[M3DBatch] = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=False,
            drop_last=False,
            collate_fn=collator,
            **common,
        )
        task_loaders[task] = EvaluationTaskLoader(
            task=task,
            dataset_size=len(dataset),
            local_sample_count=len(sampler),
            sampler=sampler,
            loader=loader,
        )

    pipeline = EvaluationDataPipeline(
        catalog=eval_catalog,
        datasets=datasets,
        loaders=task_loaders,
        workers=workers,
    )
    if runtime.is_main_process:
        runtime.logger.info(
            "%s DataLoaders initialized: %s",
            parsed_split.value.capitalize(),
            json.dumps(dict(pipeline.summary()), sort_keys=True, default=str),
        )
    return pipeline


def build_data_pipelines(
    *,
    config: ExperimentConfig,
    runtime: RuntimeContext,
    tokenizer_bundle: TokenizerBundle,
    text_processor: TextProcessorProtocol,
    include_validation: bool = True,
    include_test: bool = False,
) -> DataPipelines:
    """Build the training loader and optional evaluation loaders.

    Rank 0 should create shared manifests before this function is called.  Every
    rank then reads and fingerprints the same files, which the runtime and task
    sampler verify collectively.
    """

    train = build_training_data_pipeline(
        config=config,
        runtime=runtime,
        tokenizer_bundle=tokenizer_bundle,
        text_processor=text_processor,
    )
    validation = (
        build_evaluation_data_pipeline(
            config=config,
            runtime=runtime,
            tokenizer_bundle=tokenizer_bundle,
            text_processor=text_processor,
            split=DataSplit.VALIDATION,
        )
        if include_validation
        else None
    )
    test = (
        build_evaluation_data_pipeline(
            config=config,
            runtime=runtime,
            tokenizer_bundle=tokenizer_bundle,
            text_processor=text_processor,
            split=DataSplit.TEST,
        )
        if include_test
        else None
    )
    return DataPipelines(train=train, validation=validation, test=test)


def move_batch_to_runtime(
    batch: M3DBatch,
    *,
    runtime: RuntimeContext,
    pipeline: TrainingDataPipeline | EvaluationDataPipeline,
) -> M3DBatch:
    """Move a pinned CPU batch to the rank's GPU with configured async transfer."""

    if not isinstance(batch, M3DBatch):
        raise TypeError("batch must be M3DBatch")
    non_blocking = pipeline.workers.non_blocking_transfer
    return batch.to(runtime.device, non_blocking=non_blocking)


__all__ = [
    "DataLoaderBuildError",
    "DataPipelines",
    "EvaluationDataPipeline",
    "EvaluationTaskLoader",
    "ExactDistributedEvaluationSampler",
    "TrainingDataPipeline",
    "WorkerSettings",
    "build_data_pipelines",
    "build_evaluation_data_pipeline",
    "build_training_data_pipeline",
    "move_batch_to_runtime",
]
