"""Distributed task-balanced batch sampling for M3D.

This module is responsible for one of the central training-system changes in
M3D-Modernized: every global microbatch contains exactly one task, and every
rank executes that same task on the same step.

The sampler operates over one map-style dataset per task.  It first builds a
reproducible epoch-level task schedule using

    score(task) = configured_weight(task) * dataset_size(task) ** alpha

and then draws a non-overlapping global batch from that task.  Each rank receives
one contiguous slice of the global batch.  Consequently, with two ranks and a
per-rank batch size of one, a segmentation step is always

    rank 0 -> segmentation sample A
    rank 1 -> segmentation sample B

rather than one rank entering the text-only graph while another enters the
SegVol graph.

The implementation is intentionally random-access and checkpoint-aware.  A
batch is a pure function of epoch, step, task occurrence, dataset fingerprint,
world size, and seed.  DataLoader prefetching can therefore issue future batches
without moving the checkpoint cursor.  The trainer advances the durable cursor
only after it has actually consumed a microbatch by calling ``commit_batch``.

DataLoader integration
----------------------

    multiplexed, batch_sampler = build_task_batch_sampler(
        datasets=dataset_collection,
        config=config,
        runtime=runtime,
    )

    loader = torch.utils.data.DataLoader(
        multiplexed,
        batch_sampler=batch_sampler,
        collate_fn=collator,
        num_workers=config.data.num_workers,
        pin_memory=config.data.pin_memory,
        persistent_workers=config.data.persistent_workers,
        prefetch_factor=config.data.prefetch_factor,
    )

Do not also pass ``batch_size``, ``shuffle``, ``sampler``, or ``drop_last`` to
DataLoader when a batch sampler is supplied.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch
from torch.utils.data import Dataset, Sampler

from m3d.config import ExperimentConfig
from m3d.data.datasets import M3DDatasetCollection
from m3d.data.schema import (
    DataSplit,
    M3DBatch,
    M3DSample,
    TASK_ORDER,
    TaskDatasetInfo,
    TaskName,
    canonical_task_mapping,
)
from m3d.runtime import RuntimeContext


SAMPLER_STATE_VERSION = 1

SAMPLER_EPOCH_METADATA_KEY = "m3d_sampler_epoch"
SAMPLER_STEP_METADATA_KEY = "m3d_sampler_step"
SAMPLER_TASK_OCCURRENCE_METADATA_KEY = "m3d_sampler_task_occurrence"
SAMPLER_RANK_METADATA_KEY = "m3d_sampler_rank"
SAMPLER_LOCAL_SLOT_METADATA_KEY = "m3d_sampler_local_slot"
SAMPLER_GLOBAL_SLOT_METADATA_KEY = "m3d_sampler_global_slot"


class TaskSamplerError(RuntimeError):
    """Raised when task scheduling or distributed sample partitioning fails."""


def _stable_seed(*parts: object) -> int:
    """Create a process-independent positive seed accepted by torch.Generator."""

    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    # torch.Generator.manual_seed accepts signed 64-bit values.  Keeping the
    # high bit clear also makes generated state portable across platforms.
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_finite_nonnegative(value: float) -> bool:
    return math.isfinite(value) and value >= 0.0


@dataclass(frozen=True, slots=True)
class TaskSamplingProbability:
    """One task's size, configured weight, score, and normalized probability."""

    task: TaskName
    dataset_name: str
    dataset_size: int
    configured_weight: float
    temperature_alpha: float
    score: float
    probability: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", TaskName.parse(self.task))
        if not self.dataset_name.strip():
            raise TaskSamplerError("dataset_name cannot be empty")
        if self.dataset_size <= 0:
            raise TaskSamplerError("dataset_size must be positive")
        for name, value in (
            ("configured_weight", self.configured_weight),
            ("temperature_alpha", self.temperature_alpha),
            ("score", self.score),
            ("probability", self.probability),
        ):
            if not math.isfinite(float(value)):
                raise TaskSamplerError(f"{name} must be finite, got {value!r}")
        if self.configured_weight <= 0.0:
            raise TaskSamplerError("active task configured_weight must be positive")
        if self.score <= 0.0:
            raise TaskSamplerError("active task score must be positive")
        if not 0.0 < self.probability <= 1.0:
            raise TaskSamplerError("probability must be in (0, 1]")

    def as_dict(self) -> dict[str, Any]:
        return {
            "task": self.task.value,
            "dataset_name": self.dataset_name,
            "dataset_size": self.dataset_size,
            "configured_weight": self.configured_weight,
            "temperature_alpha": self.temperature_alpha,
            "score": self.score,
            "probability": self.probability,
        }


@dataclass(frozen=True, slots=True)
class EpochTaskSchedule:
    """Immutable task sequence and per-step task occurrence for one epoch."""

    epoch: int
    tasks: tuple[TaskName, ...]
    occurrence_by_step: tuple[int, ...]
    counts: Mapping[TaskName, int]
    fingerprint: str

    def __post_init__(self) -> None:
        if self.epoch < 0:
            raise TaskSamplerError("epoch cannot be negative")
        tasks = tuple(TaskName.parse(task) for task in self.tasks)
        occurrences = tuple(int(value) for value in self.occurrence_by_step)
        if not tasks:
            raise TaskSamplerError("epoch task schedule cannot be empty")
        if len(tasks) != len(occurrences):
            raise TaskSamplerError(
                "tasks and occurrence_by_step must have identical lengths"
            )
        if any(value < 0 for value in occurrences):
            raise TaskSamplerError("task occurrence indices cannot be negative")

        canonical_counts = {
            TaskName.parse(task): int(count) for task, count in self.counts.items()
        }
        if any(count <= 0 for count in canonical_counts.values()):
            raise TaskSamplerError("scheduled task counts must be positive")
        observed = {task: tasks.count(task) for task in set(tasks)}
        if observed != canonical_counts:
            raise TaskSamplerError(
                f"schedule count mismatch: observed={observed}, declared={canonical_counts}"
            )

        running: dict[TaskName, int] = {task: 0 for task in canonical_counts}
        for step, (task, occurrence) in enumerate(zip(tasks, occurrences)):
            expected = running[task]
            if occurrence != expected:
                raise TaskSamplerError(
                    f"Step {step} occurrence for {task.value!r} is {occurrence}, "
                    f"expected {expected}"
                )
            running[task] += 1

        object.__setattr__(self, "tasks", tasks)
        object.__setattr__(self, "occurrence_by_step", occurrences)
        object.__setattr__(
            self,
            "counts",
            MappingProxyType(dict(sorted(canonical_counts.items(), key=lambda item: int(item[0].task_id)))),
        )

    def __len__(self) -> int:
        return len(self.tasks)

    def task_at(self, step: int) -> TaskName:
        try:
            return self.tasks[step]
        except IndexError as exc:
            raise TaskSamplerError(
                f"step {step} is outside epoch schedule [0, {len(self)})"
            ) from exc

    def occurrence_at(self, step: int) -> int:
        try:
            return self.occurrence_by_step[step]
        except IndexError as exc:
            raise TaskSamplerError(
                f"step {step} is outside epoch schedule [0, {len(self)})"
            ) from exc

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "steps": len(self),
            "counts": {
                task.value: count for task, count in self.counts.items()
            },
            "fingerprint": self.fingerprint,
        }


@dataclass(frozen=True, slots=True)
class TaskSampleIndex:
    """Picklable index sent by the batch sampler to a DataLoader worker."""

    task: TaskName
    sample_index: int
    epoch: int
    step: int
    task_occurrence: int
    rank: int
    local_slot: int
    global_slot: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", TaskName.parse(self.task))
        integer_fields = {
            "sample_index": self.sample_index,
            "epoch": self.epoch,
            "step": self.step,
            "task_occurrence": self.task_occurrence,
            "rank": self.rank,
            "local_slot": self.local_slot,
            "global_slot": self.global_slot,
        }
        for name, value in integer_fields.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be int, got {type(value).__name__}")
            if value < 0:
                raise TaskSamplerError(f"{name} cannot be negative")


@dataclass(frozen=True, slots=True)
class TaskBatchPlan:
    """Debuggable global and rank-local sample assignment for one microbatch."""

    epoch: int
    step: int
    task: TaskName
    task_occurrence: int
    cycle: int
    batch_in_cycle: int
    global_indices: tuple[int, ...]
    local_indices: tuple[TaskSampleIndex, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "task", TaskName.parse(self.task))
        if self.epoch < 0 or self.step < 0 or self.task_occurrence < 0:
            raise TaskSamplerError("epoch, step, and occurrence cannot be negative")
        if self.cycle < 0 or self.batch_in_cycle < 0:
            raise TaskSamplerError("cycle and batch_in_cycle cannot be negative")
        if not self.global_indices or not self.local_indices:
            raise TaskSamplerError("batch plans cannot be empty")
        if len(set(self.global_indices)) != len(self.global_indices):
            raise TaskSamplerError(
                "A global task batch contains duplicate sample indices"
            )
        for index in self.local_indices:
            if index.task is not self.task:
                raise TaskSamplerError("local index task does not match batch task")
            if index.epoch != self.epoch or index.step != self.step:
                raise TaskSamplerError("local index epoch/step does not match batch")

    def as_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "step": self.step,
            "task": self.task.value,
            "task_occurrence": self.task_occurrence,
            "cycle": self.cycle,
            "batch_in_cycle": self.batch_in_cycle,
            "global_indices": list(self.global_indices),
            "local_sample_indices": [item.sample_index for item in self.local_indices],
        }


class TaskMultiplexDataset(Dataset[M3DSample]):
    """Route structured sampler indices to one of the task-specific datasets."""

    def __init__(self, collection: M3DDatasetCollection) -> None:
        if not isinstance(collection, M3DDatasetCollection):
            raise TypeError("collection must be an M3DDatasetCollection")
        self.collection = collection

    def __len__(self) -> int:
        # DataLoader does not use this value when batch_sampler is supplied, but
        # returning the physical record count remains useful for diagnostics.
        return sum(len(self.collection[task]) for task in self.collection.tasks)

    def __getitem__(self, index: TaskSampleIndex) -> M3DSample:
        if not isinstance(index, TaskSampleIndex):
            raise TypeError(
                "TaskMultiplexDataset expects TaskSampleIndex objects from "
                f"DistributedTaskBatchSampler, got {type(index).__name__}"
            )
        dataset = self.collection[index.task]
        if index.sample_index >= len(dataset):
            raise IndexError(
                f"Sample index {index.sample_index} is outside task "
                f"{index.task.value!r} dataset of size {len(dataset)}"
            )

        sample = dataset[index.sample_index]
        if sample.task is not index.task:
            raise TaskSamplerError(
                f"Task dataset returned {sample.task.value!r} for requested "
                f"task {index.task.value!r}"
            )

        metadata = dict(sample.provenance.metadata)
        reserved = {
            SAMPLER_EPOCH_METADATA_KEY,
            SAMPLER_STEP_METADATA_KEY,
            SAMPLER_TASK_OCCURRENCE_METADATA_KEY,
            SAMPLER_RANK_METADATA_KEY,
            SAMPLER_LOCAL_SLOT_METADATA_KEY,
            SAMPLER_GLOBAL_SLOT_METADATA_KEY,
        }
        collisions = sorted(reserved.intersection(metadata))
        if collisions:
            raise TaskSamplerError(
                "Sample provenance already uses sampler-reserved metadata keys: "
                + ", ".join(collisions)
            )
        metadata.update(
            {
                SAMPLER_EPOCH_METADATA_KEY: index.epoch,
                SAMPLER_STEP_METADATA_KEY: index.step,
                SAMPLER_TASK_OCCURRENCE_METADATA_KEY: index.task_occurrence,
                SAMPLER_RANK_METADATA_KEY: index.rank,
                SAMPLER_LOCAL_SLOT_METADATA_KEY: index.local_slot,
                SAMPLER_GLOBAL_SLOT_METADATA_KEY: index.global_slot,
            }
        )
        provenance = replace(sample.provenance, metadata=metadata)
        return replace(sample, provenance=provenance)


class DistributedTaskBatchSampler(Sampler[list[TaskSampleIndex]]):
    """Generate deterministic task-homogeneous local batches for one rank.

    The sampler drops the per-task remainder at the end of each shuffled cycle.
    A new permutation is generated for every cycle, so the same records are not
    permanently discarded.  Requiring each active task to contain at least one
    *global* batch guarantees no duplicate record can appear inside a batch.
    """

    def __init__(
        self,
        *,
        dataset: TaskMultiplexDataset,
        task_weights: Mapping[str | TaskName, float],
        temperature_alpha: float,
        local_batch_size: int,
        world_size: int,
        rank: int,
        base_seed: int,
        steps_per_epoch: int | None = None,
        guarantee_each_task: bool = True,
    ) -> None:
        if not isinstance(dataset, TaskMultiplexDataset):
            raise TypeError("dataset must be a TaskMultiplexDataset")
        if dataset.collection.split is not DataSplit.TRAIN:
            raise TaskSamplerError(
                "DistributedTaskBatchSampler is the training sampler; "
                f"received split={dataset.collection.split.value!r}"
            )
        for name, value in (
            ("local_batch_size", local_batch_size),
            ("world_size", world_size),
            ("base_seed", base_seed),
        ):
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be int")
        if local_batch_size <= 0:
            raise TaskSamplerError("local_batch_size must be positive")
        if world_size <= 0:
            raise TaskSamplerError("world_size must be positive")
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise TypeError("rank must be int")
        if not 0 <= rank < world_size:
            raise TaskSamplerError(
                f"rank must be in [0, {world_size}), got {rank}"
            )
        if not math.isfinite(float(temperature_alpha)) or not 0.0 <= temperature_alpha <= 1.0:
            raise TaskSamplerError("temperature_alpha must be finite and in [0, 1]")
        if steps_per_epoch is not None:
            if not isinstance(steps_per_epoch, int) or isinstance(steps_per_epoch, bool):
                raise TypeError("steps_per_epoch must be int or None")
            if steps_per_epoch <= 0:
                raise TaskSamplerError("steps_per_epoch must be positive")

        self.dataset = dataset
        self.local_batch_size = local_batch_size
        self.world_size = world_size
        self.rank = rank
        self.global_batch_size = local_batch_size * world_size
        self.base_seed = base_seed
        self.temperature_alpha = float(temperature_alpha)
        self.guarantee_each_task = bool(guarantee_each_task)

        infos = {info.task: info for info in dataset.collection.task_infos()}
        canonical_weights = canonical_task_mapping(task_weights)
        unknown_positive = [
            task.value
            for task, weight in canonical_weights.items()
            if float(weight) > 0.0 and task not in infos
        ]
        if unknown_positive:
            raise TaskSamplerError(
                "Positive task weights reference tasks absent from the training "
                "dataset collection: " + ", ".join(sorted(unknown_positive))
            )

        active_infos: dict[TaskName, TaskDatasetInfo] = {}
        active_weights: dict[TaskName, float] = {}
        for task in TASK_ORDER:
            if task not in infos:
                continue
            weight = float(canonical_weights.get(task, 0.0))
            if not _is_finite_nonnegative(weight):
                raise TaskSamplerError(
                    f"Task weight for {task.value!r} must be finite and non-negative"
                )
            if weight == 0.0:
                continue
            info = infos[task]
            if info.size < self.global_batch_size:
                raise TaskSamplerError(
                    f"Task {task.value!r} contains {info.size} samples, smaller than "
                    f"global batch size {self.global_batch_size}. Reduce per-device "
                    "batch size/world size or add more samples; this sampler does not "
                    "duplicate records inside a global batch."
                )
            active_infos[task] = info
            active_weights[task] = weight

        if not active_infos:
            raise TaskSamplerError("No training task has a positive usable weight")

        self._infos = MappingProxyType(active_infos)
        self._weights = MappingProxyType(active_weights)
        self._probabilities = self._build_probabilities()

        inferred_steps = math.ceil(
            sum(info.size for info in self._infos.values()) / self.global_batch_size
        )
        self.steps_per_epoch = steps_per_epoch or max(1, inferred_steps)
        if self.guarantee_each_task and self.steps_per_epoch < len(self._infos):
            raise TaskSamplerError(
                f"steps_per_epoch={self.steps_per_epoch} is smaller than the number "
                f"of active tasks ({len(self._infos)}), so each task cannot appear."
            )

        self._sampler_fingerprint = self._compute_sampler_fingerprint()
        self._epoch = 0
        self._committed_step = 0
        self._schedule = self._build_epoch_schedule(epoch=0)
        self._permutation_cache: dict[tuple[TaskName, int], torch.Tensor] = {}

        # Keep only the newest two cycles per task.  This bounds memory while
        # allowing DataLoader prefetch to briefly span a cycle boundary.
        self._max_cached_cycles_per_task = 2
        self.dataset.collection.set_epoch(0)

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def committed_step(self) -> int:
        return self._committed_step

    @property
    def remaining_steps(self) -> int:
        return self.steps_per_epoch - self._committed_step

    @property
    def sampler_fingerprint(self) -> str:
        return self._sampler_fingerprint

    @property
    def schedule(self) -> EpochTaskSchedule:
        return self._schedule

    @property
    def schedule_fingerprint(self) -> str:
        return self._schedule.fingerprint

    @property
    def active_tasks(self) -> tuple[TaskName, ...]:
        return tuple(self._infos)

    @property
    def probabilities(self) -> tuple[TaskSamplingProbability, ...]:
        return self._probabilities

    def __len__(self) -> int:
        """Return batches remaining in the current epoch."""

        return self.remaining_steps

    def __iter__(self) -> Iterator[list[TaskSampleIndex]]:
        # Use the durable cursor as the iterator start.  Prefetch can request
        # future batches, but it does not mutate committed_step.
        start_step = self._committed_step
        for step in range(start_step, self.steps_per_epoch):
            yield list(self.plan_for_step(step).local_indices)

    def _build_probabilities(self) -> tuple[TaskSamplingProbability, ...]:
        scores: dict[TaskName, float] = {}
        for task, info in self._infos.items():
            weight = self._weights[task]
            score = weight * math.pow(float(info.size), self.temperature_alpha)
            if not math.isfinite(score) or score <= 0.0:
                raise TaskSamplerError(
                    f"Computed invalid sampling score {score!r} for {task.value!r}"
                )
            scores[task] = score

        total = math.fsum(scores.values())
        if not math.isfinite(total) or total <= 0.0:
            raise TaskSamplerError("Total task sampling score must be positive")

        probabilities = tuple(
            TaskSamplingProbability(
                task=task,
                dataset_name=self._infos[task].dataset_name,
                dataset_size=self._infos[task].size,
                configured_weight=self._weights[task],
                temperature_alpha=self.temperature_alpha,
                score=scores[task],
                probability=scores[task] / total,
            )
            for task in self._infos
        )
        probability_sum = math.fsum(item.probability for item in probabilities)
        if not math.isclose(probability_sum, 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise TaskSamplerError(
                f"Task probabilities sum to {probability_sum}, expected 1"
            )
        return probabilities

    def _compute_sampler_fingerprint(self) -> str:
        return _sha256_json(
            {
                "state_version": SAMPLER_STATE_VERSION,
                "manifest_fingerprint": self.dataset.collection.manifest_fingerprint,
                "dataset_fingerprints": {
                    task.value: self.dataset.collection[task].fingerprint
                    for task in self._infos
                },
                "task_probabilities": [item.as_dict() for item in self._probabilities],
                "local_batch_size": self.local_batch_size,
                "world_size": self.world_size,
                "global_batch_size": self.global_batch_size,
                "base_seed": self.base_seed,
                "steps_per_epoch": self.steps_per_epoch,
                "guarantee_each_task": self.guarantee_each_task,
                "remainder_policy": "drop_per_task_cycle",
            }
        )

    def _allocate_task_counts(self) -> dict[TaskName, int]:
        active = [item.task for item in self._probabilities]
        counts = {task: 0 for task in active}
        remaining = self.steps_per_epoch

        if self.guarantee_each_task:
            for task in active:
                counts[task] = 1
            remaining -= len(active)

        if remaining == 0:
            return counts

        probability_by_task = {
            item.task: item.probability for item in self._probabilities
        }
        raw = {
            task: probability_by_task[task] * remaining for task in active
        }
        floors = {task: math.floor(value) for task, value in raw.items()}
        for task, value in floors.items():
            counts[task] += value

        unallocated = remaining - sum(floors.values())
        # Largest-remainder allocation makes the per-epoch quota closely match
        # the requested distribution.  Stable task_id ordering resolves ties.
        priority = sorted(
            active,
            key=lambda task: (
                -(raw[task] - floors[task]),
                int(task.task_id),
            ),
        )
        for task in priority[:unallocated]:
            counts[task] += 1

        if sum(counts.values()) != self.steps_per_epoch:
            raise TaskSamplerError(
                "Internal task-count allocation did not fill the epoch"
            )
        if self.guarantee_each_task and any(value <= 0 for value in counts.values()):
            raise TaskSamplerError("A guaranteed task received zero scheduled steps")
        return counts

    def _build_epoch_schedule(self, *, epoch: int) -> EpochTaskSchedule:
        if epoch < 0:
            raise TaskSamplerError("epoch cannot be negative")
        counts = self._allocate_task_counts()
        unshuffled: list[TaskName] = []
        for task in self._infos:
            unshuffled.extend([task] * counts[task])
        if len(unshuffled) != self.steps_per_epoch:
            raise TaskSamplerError("Task schedule has the wrong number of steps")

        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            _stable_seed(
                "m3d-task-schedule",
                self.base_seed,
                epoch,
                self._sampler_fingerprint,
            )
        )
        order = torch.randperm(len(unshuffled), generator=generator).tolist()
        tasks = tuple(unshuffled[index] for index in order)

        running = {task: 0 for task in counts}
        occurrences: list[int] = []
        for task in tasks:
            occurrences.append(running[task])
            running[task] += 1

        schedule_payload = {
            "sampler_fingerprint": self._sampler_fingerprint,
            "epoch": epoch,
            "tasks": [task.value for task in tasks],
            "occurrences": occurrences,
            "counts": {task.value: counts[task] for task in self._infos},
        }
        return EpochTaskSchedule(
            epoch=epoch,
            tasks=tasks,
            occurrence_by_step=tuple(occurrences),
            counts=counts,
            fingerprint=_sha256_json(schedule_payload),
        )

    def set_epoch(self, epoch: int, *, committed_step: int = 0) -> None:
        """Select a deterministic epoch schedule and update persistent workers."""

        if not isinstance(epoch, int) or isinstance(epoch, bool):
            raise TypeError("epoch must be int")
        if not isinstance(committed_step, int) or isinstance(committed_step, bool):
            raise TypeError("committed_step must be int")
        if epoch < 0:
            raise TaskSamplerError("epoch cannot be negative")
        if not 0 <= committed_step <= self.steps_per_epoch:
            raise TaskSamplerError(
                f"committed_step must be in [0, {self.steps_per_epoch}]"
            )

        self._epoch = epoch
        self._schedule = self._build_epoch_schedule(epoch=epoch)
        self._committed_step = committed_step
        self._permutation_cache.clear()
        self.dataset.collection.set_epoch(epoch)

    def _full_batches_per_cycle(self, task: TaskName) -> int:
        result = self._infos[task].size // self.global_batch_size
        if result <= 0:
            raise TaskSamplerError(
                f"Task {task.value!r} cannot form one global batch"
            )
        return result

    def _permutation(self, task: TaskName, cycle: int) -> torch.Tensor:
        key = (task, cycle)
        cached = self._permutation_cache.get(key)
        if cached is not None:
            return cached

        info = self._infos[task]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(
            _stable_seed(
                "m3d-task-permutation",
                self.base_seed,
                self._epoch,
                task.value,
                cycle,
                self.dataset.collection[task].fingerprint,
                self._sampler_fingerprint,
            )
        )
        permutation = torch.randperm(
            info.size,
            generator=generator,
            dtype=torch.int64,
        )
        self._permutation_cache[key] = permutation
        self._prune_permutation_cache(task)
        return permutation

    def _prune_permutation_cache(self, task: TaskName) -> None:
        cycles = sorted(
            cycle
            for cached_task, cycle in self._permutation_cache
            if cached_task is task
        )
        while len(cycles) > self._max_cached_cycles_per_task:
            cycle = cycles.pop(0)
            self._permutation_cache.pop((task, cycle), None)

    def plan_for_step(self, step: int) -> TaskBatchPlan:
        """Return the exact global and local assignment for a microbatch."""

        if not isinstance(step, int) or isinstance(step, bool):
            raise TypeError("step must be int")
        if not 0 <= step < self.steps_per_epoch:
            raise TaskSamplerError(
                f"step must be in [0, {self.steps_per_epoch}), got {step}"
            )

        task = self._schedule.task_at(step)
        occurrence = self._schedule.occurrence_at(step)
        batches_per_cycle = self._full_batches_per_cycle(task)
        cycle, batch_in_cycle = divmod(occurrence, batches_per_cycle)
        permutation = self._permutation(task, cycle)
        start = batch_in_cycle * self.global_batch_size
        stop = start + self.global_batch_size
        global_indices = tuple(int(value) for value in permutation[start:stop].tolist())
        if len(global_indices) != self.global_batch_size:
            raise TaskSamplerError(
                f"Task {task.value!r} produced {len(global_indices)} global indices, "
                f"expected {self.global_batch_size}"
            )

        rank_start = self.rank * self.local_batch_size
        rank_stop = rank_start + self.local_batch_size
        rank_indices = global_indices[rank_start:rank_stop]
        local_indices = tuple(
            TaskSampleIndex(
                task=task,
                sample_index=sample_index,
                epoch=self._epoch,
                step=step,
                task_occurrence=occurrence,
                rank=self.rank,
                local_slot=local_slot,
                global_slot=rank_start + local_slot,
            )
            for local_slot, sample_index in enumerate(rank_indices)
        )
        return TaskBatchPlan(
            epoch=self._epoch,
            step=step,
            task=task,
            task_occurrence=occurrence,
            cycle=cycle,
            batch_in_cycle=batch_in_cycle,
            global_indices=global_indices,
            local_indices=local_indices,
        )

    def commit_step(self, *, epoch: int, step: int) -> None:
        """Advance the durable cursor after a microbatch was actually consumed."""

        if epoch != self._epoch:
            raise TaskSamplerError(
                f"Cannot commit epoch {epoch}; sampler is at epoch {self._epoch}"
            )
        if step != self._committed_step:
            raise TaskSamplerError(
                "Microbatches must be committed exactly once and in order: "
                f"received step={step}, expected step={self._committed_step}"
            )
        if step >= self.steps_per_epoch:
            raise TaskSamplerError("Cannot commit beyond the end of the epoch")
        self._committed_step = step + 1

    def commit_batch(self, batch: M3DBatch) -> None:
        """Commit a collated batch using sampler metadata in its provenance."""

        epoch, step, task = sampler_position_from_batch(batch)
        expected_task = self._schedule.task_at(step)
        if task is not expected_task:
            raise TaskSamplerError(
                f"Batch task {task.value!r} does not match scheduled task "
                f"{expected_task.value!r} at step {step}"
            )
        self.commit_step(epoch=epoch, step=step)

    def state_dict(self) -> dict[str, Any]:
        """Return prefetch-safe state representing only consumed microbatches."""

        return {
            "state_version": SAMPLER_STATE_VERSION,
            "sampler_fingerprint": self._sampler_fingerprint,
            "schedule_fingerprint": self._schedule.fingerprint,
            "epoch": self._epoch,
            "committed_step": self._committed_step,
            "steps_per_epoch": self.steps_per_epoch,
            "local_batch_size": self.local_batch_size,
            "world_size": self.world_size,
            "global_batch_size": self.global_batch_size,
            "base_seed": self.base_seed,
            "temperature_alpha": self.temperature_alpha,
            "active_tasks": [task.value for task in self._infos],
            "dataset_state": self.dataset.collection.state_dict(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore exact position and reject data/topology/config mismatches."""

        if not isinstance(state, Mapping):
            raise TypeError("sampler state must be a mapping")
        version = state.get("state_version")
        if version != SAMPLER_STATE_VERSION:
            raise TaskSamplerError(
                f"Unsupported sampler state version {version!r}; "
                f"expected {SAMPLER_STATE_VERSION}"
            )
        expected_scalars = {
            "sampler_fingerprint": self._sampler_fingerprint,
            "steps_per_epoch": self.steps_per_epoch,
            "local_batch_size": self.local_batch_size,
            "world_size": self.world_size,
            "global_batch_size": self.global_batch_size,
            "base_seed": self.base_seed,
        }
        mismatches = {
            key: (state.get(key), expected)
            for key, expected in expected_scalars.items()
            if state.get(key) != expected
        }
        if mismatches:
            details = ", ".join(
                f"{key}: checkpoint={actual!r}, current={expected!r}"
                for key, (actual, expected) in mismatches.items()
            )
            raise TaskSamplerError(
                "Sampler checkpoint is incompatible with the current run: " + details
            )

        epoch = state.get("epoch")
        committed_step = state.get("committed_step")
        if not isinstance(epoch, int) or isinstance(epoch, bool):
            raise TaskSamplerError("Checkpoint epoch must be an integer")
        if not isinstance(committed_step, int) or isinstance(committed_step, bool):
            raise TaskSamplerError("Checkpoint committed_step must be an integer")

        self.set_epoch(epoch, committed_step=committed_step)
        checkpoint_schedule = state.get("schedule_fingerprint")
        if checkpoint_schedule != self._schedule.fingerprint:
            raise TaskSamplerError(
                "Rebuilt task schedule does not match the checkpoint: "
                f"checkpoint={checkpoint_schedule!r}, "
                f"current={self._schedule.fingerprint!r}"
            )
        self._validate_dataset_state(state.get("dataset_state"))

    def _validate_dataset_state(self, state: Any) -> None:
        if not isinstance(state, Mapping):
            raise TaskSamplerError("Checkpoint dataset_state must be a mapping")
        current = self.dataset.collection.state_dict()
        for key in ("split", "manifest_fingerprint", "dataset_fingerprints"):
            if state.get(key) != current.get(key):
                raise TaskSamplerError(
                    f"Dataset state mismatch for {key}: "
                    f"checkpoint={state.get(key)!r}, current={current.get(key)!r}"
                )

    def summary(self) -> dict[str, Any]:
        return {
            "sampler_fingerprint": self._sampler_fingerprint,
            "epoch": self._epoch,
            "committed_step": self._committed_step,
            "steps_per_epoch": self.steps_per_epoch,
            "remaining_steps": self.remaining_steps,
            "local_batch_size": self.local_batch_size,
            "world_size": self.world_size,
            "global_batch_size": self.global_batch_size,
            "temperature_alpha": self.temperature_alpha,
            "probabilities": [item.as_dict() for item in self._probabilities],
            "schedule": self._schedule.as_dict(),
        }


def sampler_position_from_samples(
    samples: Sequence[M3DSample],
) -> tuple[int, int, TaskName]:
    """Validate one local batch's sampler metadata and return epoch/step/task."""

    if not samples:
        raise TaskSamplerError("Cannot read sampler position from an empty batch")
    first = samples[0]
    values: list[tuple[int, int, TaskName, int, int]] = []
    for local_position, sample in enumerate(samples):
        metadata = sample.provenance.metadata
        missing = [
            key
            for key in (
                SAMPLER_EPOCH_METADATA_KEY,
                SAMPLER_STEP_METADATA_KEY,
                SAMPLER_LOCAL_SLOT_METADATA_KEY,
                SAMPLER_RANK_METADATA_KEY,
            )
            if key not in metadata
        ]
        if missing:
            raise TaskSamplerError(
                f"Sample {sample.provenance.sample_id!r} lacks sampler metadata: "
                + ", ".join(missing)
            )
        epoch = metadata[SAMPLER_EPOCH_METADATA_KEY]
        step = metadata[SAMPLER_STEP_METADATA_KEY]
        local_slot = metadata[SAMPLER_LOCAL_SLOT_METADATA_KEY]
        rank = metadata[SAMPLER_RANK_METADATA_KEY]
        if not all(isinstance(item, int) and not isinstance(item, bool) for item in (epoch, step, local_slot, rank)):
            raise TaskSamplerError("Sampler metadata values must be integers")
        if local_slot != local_position:
            raise TaskSamplerError(
                f"Local sample order mismatch: position={local_position}, "
                f"metadata local_slot={local_slot}"
            )
        values.append((epoch, step, sample.task, rank, local_slot))

    reference = values[0][:4]
    for value in values[1:]:
        if value[:4] != reference:
            raise TaskSamplerError(
                "Samples in one local batch disagree on epoch, step, task, or rank: "
                f"first={reference!r}, other={value[:4]!r}"
            )
    return reference[0], reference[1], reference[2]


def sampler_position_from_batch(batch: M3DBatch) -> tuple[int, int, TaskName]:
    """Read sampler epoch/step from a collated batch's provenance records."""

    if not isinstance(batch, M3DBatch):
        raise TypeError("batch must be an M3DBatch")
    if not batch.provenance:
        raise TaskSamplerError("Batch provenance cannot be empty")
    values: list[tuple[int, int, TaskName, int]] = []
    for provenance in batch.provenance:
        metadata = provenance.metadata
        try:
            epoch = metadata[SAMPLER_EPOCH_METADATA_KEY]
            step = metadata[SAMPLER_STEP_METADATA_KEY]
            rank = metadata[SAMPLER_RANK_METADATA_KEY]
        except KeyError as exc:
            raise TaskSamplerError(
                f"Batch provenance {provenance.sample_id!r} lacks {exc.args[0]!r}"
            ) from exc
        if not all(isinstance(item, int) and not isinstance(item, bool) for item in (epoch, step, rank)):
            raise TaskSamplerError("Batch sampler metadata must contain integers")
        values.append((epoch, step, batch.task, rank))

    reference = values[0]
    if any(value != reference for value in values[1:]):
        raise TaskSamplerError(
            f"Batch provenance disagrees on sampler position: {values!r}"
        )
    return reference[0], reference[1], reference[2]


def validate_distributed_schedule(
    sampler: DistributedTaskBatchSampler,
    runtime: RuntimeContext,
) -> None:
    """Perform one startup collective proving all ranks built the same schedule."""

    if not isinstance(sampler, DistributedTaskBatchSampler):
        raise TypeError("sampler must be a DistributedTaskBatchSampler")
    if not isinstance(runtime, RuntimeContext):
        raise TypeError("runtime must be a RuntimeContext")
    if runtime.rank != sampler.rank or runtime.world_size != sampler.world_size:
        raise TaskSamplerError(
            "Runtime topology and sampler topology differ: "
            f"runtime(rank={runtime.rank}, world={runtime.world_size}), "
            f"sampler(rank={sampler.rank}, world={sampler.world_size})"
        )
    runtime.assert_all_ranks_equal(
        sampler.sampler_fingerprint,
        label="task sampler fingerprint",
    )
    runtime.assert_all_ranks_equal(
        sampler.schedule_fingerprint,
        label=f"epoch {sampler.epoch} task schedule fingerprint",
    )


def assert_distributed_task_for_step(
    *,
    sampler: DistributedTaskBatchSampler,
    runtime: RuntimeContext,
    step: int,
) -> None:
    """Optional debugging collective verifying the selected task for one step."""

    task = sampler.schedule.task_at(step)
    runtime.assert_all_ranks_equal(
        task.value,
        label=f"training task at epoch={sampler.epoch} step={step}",
    )


def build_task_batch_sampler(
    *,
    datasets: M3DDatasetCollection,
    config: ExperimentConfig,
    runtime: RuntimeContext,
) -> tuple[TaskMultiplexDataset, DistributedTaskBatchSampler]:
    """Build and cross-rank validate the modern M3D training sampler."""

    if not isinstance(datasets, M3DDatasetCollection):
        raise TypeError("datasets must be an M3DDatasetCollection")
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig")
    if not isinstance(runtime, RuntimeContext):
        raise TypeError("runtime must be a RuntimeContext")
    config.validate()

    policy = config.data.task_sampling
    if not policy.enabled:
        raise TaskSamplerError(
            "The modern M3D training path requires data.task_sampling.enabled=true"
        )
    if not policy.homogeneous_batches:
        raise TaskSamplerError(
            "The modern M3D training path requires homogeneous_batches=true"
        )

    multiplexed = TaskMultiplexDataset(datasets)
    sampler = DistributedTaskBatchSampler(
        dataset=multiplexed,
        task_weights=policy.task_weights,
        temperature_alpha=policy.temperature_alpha,
        local_batch_size=config.optimization.per_device_batch_size,
        world_size=runtime.world_size,
        rank=runtime.rank,
        base_seed=config.runtime.seed,
        steps_per_epoch=policy.steps_per_epoch,
        guarantee_each_task=True,
    )
    validate_distributed_schedule(sampler, runtime)
    if runtime.is_main_process:
        runtime.logger.info(
            "Task sampler initialized: %s",
            json.dumps(sampler.summary(), sort_keys=True),
        )
    return multiplexed, sampler


__all__ = [
    "SAMPLER_EPOCH_METADATA_KEY",
    "SAMPLER_GLOBAL_SLOT_METADATA_KEY",
    "SAMPLER_LOCAL_SLOT_METADATA_KEY",
    "SAMPLER_RANK_METADATA_KEY",
    "SAMPLER_STATE_VERSION",
    "SAMPLER_STEP_METADATA_KEY",
    "SAMPLER_TASK_OCCURRENCE_METADATA_KEY",
    "DistributedTaskBatchSampler",
    "EpochTaskSchedule",
    "TaskBatchPlan",
    "TaskMultiplexDataset",
    "TaskSampleIndex",
    "TaskSamplerError",
    "TaskSamplingProbability",
    "assert_distributed_task_for_step",
    "build_task_batch_sampler",
    "sampler_position_from_batch",
    "sampler_position_from_samples",
    "validate_distributed_schedule",
]
