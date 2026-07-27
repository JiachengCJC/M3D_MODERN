"""Distributed and numerical runtime utilities for M3D-Modernized.

This module runs after :mod:`m3d.config` has loaded and validated the YAML
configuration, but before tokenizer, datasets, or model parameters are created.
It provides one process per GPU under ``torchrun`` and supports both of the
project's data-parallel execution modes:

* DDP for highest throughput when the complete model fits on each A100;
* FSDP2 for parameter/gradient/optimizer sharding when DDP does not fit.

The module does not build or wrap the model.  It only establishes the process
group, CUDA device, DeviceMesh (for FSDP2), random state, numerical flags, and
rank-aware logging used by later files.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import json
import logging
import os
import random
import socket
import sys
import time
from collections.abc import Generator, Iterable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor

from .config import (
    ExperimentConfig,
    config_fingerprint,
    configure_torch_runtime,
    seed_everything,
)


LOGGER_NAME = "m3d"
T = TypeVar("T")


class DistributedRuntimeError(RuntimeError):
    """Raised when launcher, rank, CUDA, or collective state is inconsistent."""


@dataclass(frozen=True, slots=True)
class LauncherEnvironment:
    """Values supplied by ``torchrun`` through environment variables."""

    rank: int
    local_rank: int
    world_size: int
    local_world_size: int
    node_rank: int
    master_addr: str | None
    master_port: int | None

    @property
    def launched_with_torchrun(self) -> bool:
        return self.world_size > 1 or "RANK" in os.environ


@dataclass(slots=True)
class RuntimeContext:
    """Process-local runtime state shared by the training components."""

    config: ExperimentConfig
    launcher: LauncherEnvironment
    device: torch.device
    distributed: bool
    process_group_initialized: bool
    seed: int
    logger: logging.Logger
    device_mesh: Any | None = None

    @property
    def rank(self) -> int:
        return self.launcher.rank

    @property
    def local_rank(self) -> int:
        return self.launcher.local_rank

    @property
    def world_size(self) -> int:
        return self.launcher.world_size

    @property
    def local_world_size(self) -> int:
        return self.launcher.local_world_size

    @property
    def node_rank(self) -> int:
        return self.launcher.node_rank

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def is_local_main_process(self) -> bool:
        return self.local_rank == 0

    def barrier(self) -> None:
        """Synchronize all ranks using the rank's already-bound CUDA device."""

        if self.process_group_initialized:
            dist.barrier(device_ids=[self.local_rank])

    def all_reduce_sum(self, value: Tensor) -> Tensor:
        """Return a sum-reduced clone without mutating the caller's tensor."""

        result = value.clone()
        if self.process_group_initialized:
            dist.all_reduce(result, op=dist.ReduceOp.SUM)
        return result

    def all_reduce_mean(self, value: Tensor) -> Tensor:
        """Return the mean of a tensor across data-parallel ranks."""

        result = self.all_reduce_sum(value)
        if self.world_size > 1:
            result.div_(self.world_size)
        return result

    def reduce_scalar_dict(
        self,
        values: Mapping[str, float | int | Tensor],
        *,
        average: bool = True,
    ) -> dict[str, float]:
        """Reduce scalar metrics with one packed collective.

        Packing all values into one tensor is substantially cheaper than issuing
        one NCCL collective per loss or metric.
        """

        names = sorted(values)
        packed_values: list[Tensor] = []
        for name in names:
            value = values[name]
            if isinstance(value, Tensor):
                if value.numel() != 1:
                    raise ValueError(
                        f"Metric {name!r} must be scalar, got shape {tuple(value.shape)}"
                    )
                packed_values.append(value.detach().to(self.device, dtype=torch.float64))
            else:
                packed_values.append(
                    torch.tensor(float(value), device=self.device, dtype=torch.float64)
                )

        if not packed_values:
            return {}

        packed = torch.stack(packed_values)
        if self.process_group_initialized:
            dist.all_reduce(packed, op=dist.ReduceOp.SUM)
            if average:
                packed.div_(self.world_size)

        return {
            name: float(value)
            for name, value in zip(names, packed.cpu().tolist(), strict=True)
        }

    def broadcast_object(self, value: T | None, *, source_rank: int = 0) -> T:
        """Broadcast a small picklable Python object from ``source_rank``."""

        if not 0 <= source_rank < self.world_size:
            raise ValueError(
                f"source_rank must be in [0, {self.world_size}), got {source_rank}"
            )

        if not self.process_group_initialized:
            if value is None:
                raise ValueError("Single-process broadcast cannot broadcast None")
            return value

        payload: list[Any] = [value if self.rank == source_rank else None]
        dist.broadcast_object_list(payload, src=source_rank, device=self.device)
        if payload[0] is None:
            raise DistributedRuntimeError("Object broadcast unexpectedly returned None")
        return payload[0]

    def all_gather_object(self, value: T) -> list[T]:
        """Gather a small diagnostic object from every rank."""

        if not self.process_group_initialized:
            return [value]

        gathered: list[Any] = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered, value)
        return gathered

    def assert_all_ranks_equal(self, value: Any, *, label: str) -> None:
        """Fail fast when ranks disagree about a control-flow decision.

        This will later be used for task-homogeneous batching, ensuring every
        rank enters either the text path or the SegVol path on the same step.
        """

        gathered = self.all_gather_object(value)
        reference = gathered[0]
        mismatches = [
            (rank, item)
            for rank, item in enumerate(gathered)
            if item != reference
        ]
        if mismatches:
            details = ", ".join(
                f"rank {rank}={item!r}" for rank, item in mismatches
            )
            raise DistributedRuntimeError(
                f"Ranks disagree on {label}: rank 0={reference!r}; {details}"
            )

    @contextlib.contextmanager
    def main_process_first(self) -> Generator[None, None, None]:
        """Let rank 0 build shared metadata/cache before the remaining ranks."""

        if not self.process_group_initialized:
            yield
            return

        if not self.is_main_process:
            self.barrier()

        try:
            yield
        finally:
            if self.is_main_process:
                self.barrier()

    def autocast(self) -> torch.autocast:
        """Return the BF16 autocast context used for forward and loss."""

        return torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
            enabled=True,
            cache_enabled=True,
        )

    def cuda_memory_snapshot(self) -> dict[str, float]:
        """Return process-local CUDA allocator counters in GiB."""

        gib = float(1024**3)
        return {
            "allocated_gib": torch.cuda.memory_allocated(self.device) / gib,
            "reserved_gib": torch.cuda.memory_reserved(self.device) / gib,
            "peak_allocated_gib": torch.cuda.max_memory_allocated(self.device) / gib,
            "peak_reserved_gib": torch.cuda.max_memory_reserved(self.device) / gib,
        }

    def close(self) -> None:
        """Synchronize and destroy the default process group exactly once."""

        if dist.is_available() and dist.is_initialized():
            try:
                dist.barrier(device_ids=[self.local_rank])
            except Exception:  # Best effort during exception unwinding.
                self.logger.exception("Final distributed barrier failed")
            finally:
                dist.destroy_process_group()
        self.process_group_initialized = False


class _RankFilter(logging.Filter):
    """Inject rank fields into each log record."""

    def __init__(self, rank: int, local_rank: int) -> None:
        super().__init__()
        self.rank = rank
        self.local_rank = local_rank

    def filter(self, record: logging.LogRecord) -> bool:
        record.rank = self.rank
        record.local_rank = self.local_rank
        return True


def _environment_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise DistributedRuntimeError(
            f"Environment variable {name} must be an integer, got {raw!r}"
        ) from exc
    return value


def read_launcher_environment() -> LauncherEnvironment:
    """Read and validate rank metadata exported by ``torchrun``."""

    rank = _environment_int("RANK", 0)
    local_rank = _environment_int("LOCAL_RANK", 0)
    world_size = _environment_int("WORLD_SIZE", 1)
    local_world_size = _environment_int("LOCAL_WORLD_SIZE", 1)

    if rank < 0:
        raise DistributedRuntimeError(f"RANK cannot be negative, got {rank}")
    if local_rank < 0:
        raise DistributedRuntimeError(
            f"LOCAL_RANK cannot be negative, got {local_rank}"
        )
    if world_size <= 0:
        raise DistributedRuntimeError(
            f"WORLD_SIZE must be positive, got {world_size}"
        )
    if local_world_size <= 0:
        raise DistributedRuntimeError(
            f"LOCAL_WORLD_SIZE must be positive, got {local_world_size}"
        )
    if rank >= world_size:
        raise DistributedRuntimeError(
            f"RANK={rank} must be smaller than WORLD_SIZE={world_size}"
        )
    if local_rank >= local_world_size:
        raise DistributedRuntimeError(
            "LOCAL_RANK must be smaller than LOCAL_WORLD_SIZE: "
            f"{local_rank} >= {local_world_size}"
        )

    master_port_raw = os.environ.get("MASTER_PORT")
    master_port = None
    if master_port_raw is not None:
        try:
            master_port = int(master_port_raw)
        except ValueError as exc:
            raise DistributedRuntimeError(
                f"MASTER_PORT must be an integer, got {master_port_raw!r}"
            ) from exc
        if not 1 <= master_port <= 65535:
            raise DistributedRuntimeError(
                f"MASTER_PORT must be in [1, 65535], got {master_port}"
            )

    return LauncherEnvironment(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        local_world_size=local_world_size,
        node_rank=rank // local_world_size,
        master_addr=os.environ.get("MASTER_ADDR"),
        master_port=master_port,
    )


def _configure_nccl_environment() -> None:
    """Set conservative diagnostics without overriding cluster networking."""

    # Do not set NCCL_SOCKET_IFNAME, NCCL_IB_DISABLE, NCCL_P2P_DISABLE, or HPE
    # fabric variables here.  ASPIRE 2A should retain the site-provided network
    # configuration.  These options only improve failure visibility.
    os.environ.setdefault("NCCL_DEBUG", "WARN")
    os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")


def _validate_cuda_environment(launcher: LauncherEnvironment) -> torch.device:
    if not torch.cuda.is_available():
        raise DistributedRuntimeError(
            "CUDA is unavailable. M3D training must run inside an ASPIRE 2A "
            "GPU PBS allocation, not on a login node."
        )

    visible_devices = torch.cuda.device_count()
    if visible_devices <= 0:
        raise DistributedRuntimeError("PyTorch reports zero visible CUDA devices")
    if launcher.local_rank >= visible_devices:
        raise DistributedRuntimeError(
            f"LOCAL_RANK={launcher.local_rank}, but only {visible_devices} CUDA "
            "device(s) are visible to this process"
        )
    if launcher.world_size > 1 and launcher.local_world_size != visible_devices:
        raise DistributedRuntimeError(
            "LOCAL_WORLD_SIZE and visible GPU count differ: "
            f"{launcher.local_world_size} != {visible_devices}. Start one torchrun "
            "process per allocated GPU."
        )

    device = torch.device("cuda", launcher.local_rank)
    torch.cuda.set_device(device)

    capability = torch.cuda.get_device_capability(device)
    if capability < (8, 0):
        raise DistributedRuntimeError(
            "The optimized BF16/Flash-SDPA profile requires NVIDIA compute "
            f"capability 8.0 or newer, got {capability[0]}.{capability[1]}"
        )
    if not torch.cuda.is_bf16_supported():
        raise DistributedRuntimeError(
            f"GPU {launcher.local_rank} does not report BF16 support"
        )

    return device


def _configure_logger(
    *,
    rank: int,
    local_rank: int,
    output_dir: Path,
    verbose_all_ranks: bool,
) -> logging.Logger:
    """Create one console logger and one file per rank without duplicate handlers."""

    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(LOGGER_NAME)
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.DEBUG)

    rank_filter = _RankFilter(rank, local_rank)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | rank=%(rank)d local=%(local_rank)d | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO if rank == 0 or verbose_all_ranks else logging.WARNING)
    console.setFormatter(formatter)
    console.addFilter(rank_filter)
    logger.addHandler(console)

    file_handler = logging.FileHandler(
        output_dir / f"rank_{rank:05d}.log",
        mode="a",
        encoding="utf-8",
        delay=False,
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(rank_filter)
    logger.addHandler(file_handler)

    return logger


def _config_digest(config: ExperimentConfig) -> str:
    encoded = config_fingerprint(config).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _verify_identical_config(context: RuntimeContext) -> None:
    digest = _config_digest(context.config)
    all_digests = context.all_gather_object(digest)
    if len(set(all_digests)) != 1:
        details = ", ".join(
            f"rank {rank}={item}" for rank, item in enumerate(all_digests)
        )
        raise DistributedRuntimeError(
            f"Ranks loaded different experiment configurations: {details}"
        )


def _distributed_health_check(context: RuntimeContext) -> None:
    """Run a tiny NCCL collective after process-group initialization."""

    value = torch.tensor(
        float(context.rank + 1),
        device=context.device,
        dtype=torch.float32,
    )
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    expected = context.world_size * (context.world_size + 1) / 2
    if float(value.item()) != float(expected):
        raise DistributedRuntimeError(
            f"NCCL health check returned {value.item()}, expected {expected}"
        )


def _build_device_mesh(context: RuntimeContext) -> Any | None:
    if context.config.distributed.strategy != "fsdp2":
        return None
    if not context.process_group_initialized:
        raise DistributedRuntimeError("FSDP2 requires an initialized process group")

    try:
        from torch.distributed.device_mesh import init_device_mesh
    except ImportError as exc:
        raise DistributedRuntimeError(
            "PyTorch installation does not provide DeviceMesh required by FSDP2"
        ) from exc

    return init_device_mesh(
        "cuda",
        mesh_shape=(context.world_size,),
        mesh_dim_names=("data_parallel",),
    )


def initialize_runtime(
    config: ExperimentConfig,
    *,
    verbose_all_ranks: bool = False,
) -> RuntimeContext:
    """Initialize CUDA, logging, NCCL, random state, and optional DeviceMesh.

    Call this once per process immediately after loading the YAML configuration.
    The function deliberately performs no model or dataset allocation.
    """

    launcher = read_launcher_environment()

    if config.distributed.strategy in {"ddp", "fsdp2"} and launcher.world_size == 1:
        raise DistributedRuntimeError(
            f"distributed.strategy={config.distributed.strategy!r} requires torchrun "
            "with more than one process. For ASPIRE 2A use "
            "--nproc_per_node=2."
        )

    _configure_nccl_environment()
    device = _validate_cuda_environment(launcher)

    # Apply numerical settings before any model parameters or CUDA kernels are
    # created. The seed has a rank offset so stochastic data augmentation differs
    # between ranks while remaining reproducible.
    configure_torch_runtime(config.optimization)
    seed = seed_everything(config.runtime, rank=launcher.rank)

    output_dir = Path(config.checkpoint.output_dir).expanduser().resolve()
    logs_dir = output_dir / "logs"

    # Each process can safely create the same directory tree with exist_ok=True.
    logger = _configure_logger(
        rank=launcher.rank,
        local_rank=launcher.local_rank,
        output_dir=logs_dir,
        verbose_all_ranks=verbose_all_ranks,
    )

    initialized = False
    context = RuntimeContext(
        config=config,
        launcher=launcher,
        device=device,
        distributed=launcher.world_size > 1,
        process_group_initialized=False,
        seed=seed,
        logger=logger,
    )

    try:
        if launcher.world_size > 1:
            if not launcher.launched_with_torchrun:
                raise DistributedRuntimeError(
                    "Distributed training must be launched with torchrun"
                )
            if launcher.master_addr is None or launcher.master_port is None:
                raise DistributedRuntimeError(
                    "MASTER_ADDR and MASTER_PORT are required for env:// rendezvous"
                )
            if dist.is_initialized():
                raise DistributedRuntimeError(
                    "The default process group was already initialized before "
                    "m3d.runtime.initialize_runtime()"
                )

            dist.init_process_group(
                backend=config.distributed.backend,
                init_method="env://",
                timeout=timedelta(seconds=config.distributed.timeout_seconds),
                world_size=launcher.world_size,
                rank=launcher.rank,
                device_id=device,
            )
            initialized = True
            context.process_group_initialized = True

            if dist.get_rank() != launcher.rank:
                raise DistributedRuntimeError(
                    f"Process-group rank {dist.get_rank()} differs from RANK={launcher.rank}"
                )
            if dist.get_world_size() != launcher.world_size:
                raise DistributedRuntimeError(
                    "Process-group world size differs from WORLD_SIZE: "
                    f"{dist.get_world_size()} != {launcher.world_size}"
                )

            _distributed_health_check(context)
            _verify_identical_config(context)
            context.device_mesh = _build_device_mesh(context)
            context.barrier()

        properties = torch.cuda.get_device_properties(device)
        logger.info(
            "Runtime initialized: host=%s strategy=%s world_size=%d device=%s "
            "gpu=%s memory=%.2f GiB capability=%d.%d seed=%d",
            socket.gethostname(),
            config.distributed.strategy,
            launcher.world_size,
            device,
            properties.name,
            properties.total_memory / (1024**3),
            properties.major,
            properties.minor,
            seed,
        )
        logger.info(
            "Numerics: precision=%s TF32=%s matmul_precision=%s deterministic=%s",
            config.optimization.precision,
            config.optimization.allow_tf32,
            config.optimization.matmul_precision,
            config.runtime.deterministic,
        )

        # Register a best-effort cleanup for unhandled exceptions or early exit.
        atexit.register(context.close)
        return context

    except Exception:
        logger.exception("Runtime initialization failed")
        if initialized and dist.is_initialized():
            dist.destroy_process_group()
        raise


@contextlib.contextmanager
def distributed_runtime(
    config: ExperimentConfig,
    *,
    verbose_all_ranks: bool = False,
) -> Generator[RuntimeContext, None, None]:
    """Context-manager wrapper that guarantees process-group cleanup."""

    context = initialize_runtime(
        config,
        verbose_all_ranks=verbose_all_ranks,
    )
    try:
        yield context
    finally:
        context.close()


def dataloader_worker_init_fn(worker_id: int) -> None:
    """Seed one DataLoader worker from PyTorch's per-worker initial seed."""

    # torch.initial_seed() already incorporates the rank-local DataLoader
    # generator state. Reducing modulo 2**32 satisfies NumPy's seed range.
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def make_dataloader_generator(context: RuntimeContext, *, stream: int = 0) -> torch.Generator:
    """Create a reproducible CPU generator for a DataLoader/sampler stream."""

    if stream < 0:
        raise ValueError(f"stream must be non-negative, got {stream}")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(context.seed + stream * 1_000_003)
    return generator


def atomic_write_json(path: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    """Write JSON through a temporary file and atomic rename."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
    os.replace(temporary, destination)


def log_rank_zero(context: RuntimeContext, level: int, message: str, *args: Any) -> None:
    """Log a message only from global rank 0."""

    if context.is_main_process:
        context.logger.log(level, message, *args)


def timed_barrier(context: RuntimeContext, *, label: str) -> float:
    """Synchronize ranks and return local wait time in seconds."""

    start = time.perf_counter()
    context.barrier()
    elapsed = time.perf_counter() - start
    context.logger.debug("Barrier %s completed in %.6f seconds", label, elapsed)
    return elapsed


def assert_finite_tensor(tensor: Tensor, *, name: str) -> None:
    """Raise with a useful name before NaN/Inf silently reaches collectives."""

    if not bool(torch.isfinite(tensor).all().item()):
        raise FloatingPointError(f"Tensor {name!r} contains NaN or Inf")
