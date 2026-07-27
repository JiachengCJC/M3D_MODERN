#!/usr/bin/env python3
"""ASPIRE 2A GPU and distributed-training preflight for M3D-Modernized.

Run this file on an ASPIRE 2A GPU compute node *after* running
``scripts/00_setup_environment.sh``.

Recommended two-GPU invocation::

    source /scratch/$USER/envs/m3d-modern/bin/activate
    torchrun --standalone --nnodes=1 --nproc_per_node=2 \
        scripts/01_preflight.py --expected-gpus 2

The check intentionally exercises the same hardware paths that the rewritten
M3D training stack will depend on:

* PyTorch 2.6.0 built for CUDA 11.8;
* NVIDIA A100 / Ampere BF16 execution;
* non-causal Flash-SDPA with the M3D vision-token shape;
* causal Flash-SDPA with a language-model shape;
* a backward pass through the 3D patch embedding;
* NCCL initialization and an all-reduce across all local ranks.

This file does not load the M3D model or any dataset. Its sole purpose is to
fail early when the compute-node environment is unsuitable for training.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.attention import SDPBackend, sdpa_kernel


EXPECTED_PYTHON = (3, 10)
EXPECTED_TORCH_PREFIX = "2.6.0"
EXPECTED_TORCH_CUDA = "11.8"
EXPECTED_MODULE_MARKERS = (
    "PrgEnv-gnu/8.3.3",
    "gcc/11.4.0-nscc",
    "python/3.10.9",
    "cuda/11.8.0",
)


class PreflightError(RuntimeError):
    """Raised when a required ASPIRE 2A training capability is unavailable."""


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Validate the ASPIRE 2A environment before M3D training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--expected-gpus",
        type=int,
        default=int(os.environ.get("M3D_EXPECTED_GPUS", "2")),
        help="Number of GPUs/ranks expected for this training job.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=project_root / "preflight_report.json",
        help="Rank-zero JSON report output path.",
    )
    parser.add_argument(
        "--allow-non-a100",
        action="store_true",
        help="Allow Ampere-or-newer GPUs whose model name is not A100.",
    )
    parser.add_argument(
        "--allow-single-process",
        action="store_true",
        help=(
            "Permit a non-torchrun invocation. NCCL collectives will not be "
            "validated when WORLD_SIZE is one."
        ),
    )
    parser.add_argument(
        "--attention-seq-len",
        type=int,
        default=2049,
        help="Vision Flash-SDPA sequence length (2048 patches plus CLS token).",
    )
    args = parser.parse_args()

    if args.expected_gpus < 1:
        parser.error("--expected-gpus must be at least 1")
    if args.attention_seq_len < 1:
        parser.error("--attention-seq-len must be at least 1")

    return args


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise PreflightError(f"{name} must be an integer, received {raw!r}.") from exc


def run_command(command: list[str], *, timeout: int = 20) -> str:
    """Run a system command and return stripped stdout."""
    try:
        completed = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise PreflightError(f"Required command is unavailable: {command[0]}") from exc
    except subprocess.CalledProcessError as exc:
        output = exc.stdout.strip() if exc.stdout else "<no output>"
        raise PreflightError(
            f"Command failed ({' '.join(command)}):\n{output}"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise PreflightError(f"Command timed out: {' '.join(command)}") from exc

    return completed.stdout.strip()


def human_bytes(number: int) -> str:
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    value = float(number)
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")


def nccl_version_string() -> str:
    version = torch.cuda.nccl.version()
    if version is None:
        return "unknown"
    if isinstance(version, tuple):
        return ".".join(str(component) for component in version)

    # PyTorch may expose NCCL as an integer such as 22602 -> 2.26.2.
    major = version // 10_000
    minor = (version % 10_000) // 100
    patch = version % 100
    return f"{major}.{minor}.{patch}"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def check_software_contract() -> dict[str, Any]:
    require(
        sys.version_info[:2] == EXPECTED_PYTHON,
        "Expected Python 3.10 from python/3.10.9, detected "
        f"{sys.version.split()[0]}.",
    )
    require(
        torch.__version__.startswith(EXPECTED_TORCH_PREFIX),
        f"Expected PyTorch {EXPECTED_TORCH_PREFIX}, detected {torch.__version__}.",
    )
    require(
        torch.version.cuda == EXPECTED_TORCH_CUDA,
        f"Expected the PyTorch CUDA 11.8 build, detected {torch.version.cuda}.",
    )

    loaded_modules = os.environ.get("LOADEDMODULES", "")
    missing_modules = [
        marker for marker in EXPECTED_MODULE_MARKERS if marker not in loaded_modules
    ]
    require(
        not missing_modules,
        "The ASPIRE 2A module stack is incomplete. Missing markers: "
        + ", ".join(missing_modules)
        + ". Run scripts/00_setup_environment.sh or load the same modules in "
        "the PBS job before activating the virtual environment.",
    )

    nvcc_output = run_command(["nvcc", "--version"])
    require(
        "release 11.8" in nvcc_output,
        "nvcc is not CUDA 11.8:\n" + nvcc_output,
    )

    gcc_output = run_command(["gcc", "--version"]).splitlines()[0]
    require("11.4.0" in gcc_output, f"Expected GCC 11.4.0, detected: {gcc_output}")

    return {
        "python": sys.version.split()[0],
        "pytorch": torch.__version__,
        "pytorch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "nccl": nccl_version_string(),
        "gcc": gcc_output,
        "nvcc": next(
            (line.strip() for line in nvcc_output.splitlines() if "release" in line),
            nvcc_output,
        ),
        "loaded_modules": loaded_modules.split(":"),
    }


def initialize_distributed(args: argparse.Namespace) -> tuple[int, int, int]:
    world_size = env_int("WORLD_SIZE", 1)
    rank = env_int("RANK", 0)
    local_rank = env_int("LOCAL_RANK", 0)

    if world_size == 1:
        require(
            args.allow_single_process or args.expected_gpus == 1,
            "This preflight was started as one process, but the target job uses "
            f"{args.expected_gpus} GPUs. Run it with:\n"
            f"  torchrun --standalone --nnodes=1 --nproc_per_node={args.expected_gpus} "
            "scripts/01_preflight.py "
            f"--expected-gpus {args.expected_gpus}",
        )
        return rank, local_rank, world_size

    require(
        world_size == args.expected_gpus,
        f"WORLD_SIZE={world_size}, but --expected-gpus={args.expected_gpus}.",
    )

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        timeout=timedelta(minutes=5),
    )
    return rank, local_rank, world_size


def configure_device(
    args: argparse.Namespace,
    *,
    local_rank: int,
    world_size: int,
) -> tuple[torch.device, dict[str, Any]]:
    require(torch.cuda.is_available(), "CUDA is not available on this process.")

    visible_count = torch.cuda.device_count()
    require(
        visible_count >= args.expected_gpus,
        f"Expected at least {args.expected_gpus} visible GPUs, found {visible_count}.",
    )
    require(
        0 <= local_rank < visible_count,
        f"LOCAL_RANK={local_rank} is invalid for {visible_count} visible GPUs.",
    )

    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    properties = torch.cuda.get_device_properties(device)
    capability = torch.cuda.get_device_capability(device)

    require(
        capability >= (8, 0),
        "The optimized M3D path requires Ampere or newer; detected compute "
        f"capability {capability[0]}.{capability[1]} on {properties.name}.",
    )
    require(
        torch.cuda.is_bf16_supported(),
        f"{properties.name} does not report BF16 support.",
    )
    require(
        args.allow_non_a100 or "A100" in properties.name,
        f"Expected an ASPIRE 2A A100, detected {properties.name}. Pass "
        "--allow-non-a100 only when this is deliberate.",
    )

    # These settings are part of the eventual training runtime contract.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    return device, {
        "local_rank": local_rank,
        "world_size": world_size,
        "name": properties.name,
        "compute_capability": f"{capability[0]}.{capability[1]}",
        "total_memory_bytes": properties.total_memory,
        "total_memory": human_bytes(properties.total_memory),
        "multiprocessor_count": properties.multi_processor_count,
        "bf16_supported": torch.cuda.is_bf16_supported(),
    }


def assert_finite(name: str, tensor: Tensor) -> None:
    require(
        bool(torch.isfinite(tensor).all().item()),
        f"{name} contains NaN or Inf values.",
    )


def flash_sdpa_training_test(
    *,
    device: torch.device,
    sequence_length: int,
    causal: bool,
    label: str,
) -> dict[str, Any]:
    """Force PyTorch's Flash-SDPA backend and run forward plus backward."""
    batch_size = 1
    num_heads = 12
    head_dim = 64
    dtype = torch.bfloat16

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    generator = torch.Generator(device=device)
    generator.manual_seed(17 + device.index)

    q = torch.randn(
        batch_size,
        num_heads,
        sequence_length,
        head_dim,
        generator=generator,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)

    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    # Supplying only FLASH_ATTENTION makes unsupported shapes fail instead of
    # silently falling back to the math backend.
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=causal,
        )
        loss = output.float().square().mean()
        loss.backward()

    end.record()
    torch.cuda.synchronize(device)

    expected_shape = (batch_size, num_heads, sequence_length, head_dim)
    require(
        tuple(output.shape) == expected_shape,
        f"{label} SDPA output shape is {tuple(output.shape)}, expected "
        f"{expected_shape}.",
    )
    assert_finite(f"{label} output", output)
    assert_finite(f"{label} q.grad", q.grad)
    assert_finite(f"{label} k.grad", k.grad)
    assert_finite(f"{label} v.grad", v.grad)

    result = {
        "backend": "FLASH_ATTENTION",
        "dtype": str(dtype).removeprefix("torch."),
        "causal": causal,
        "shape": list(expected_shape),
        "forward_backward_ms": start.elapsed_time(end),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        "peak_memory": human_bytes(torch.cuda.max_memory_allocated(device)),
        "loss": float(loss.detach().cpu()),
    }

    del output, loss, q, k, v
    torch.cuda.empty_cache()
    return result


def patch_embedding_3d_test(device: torch.device) -> dict[str, Any]:
    """Exercise the 3D patch embedding used by both independent encoders."""
    dtype = torch.bfloat16
    expected_shape = (1, 768, 8, 16, 16)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    patch_embed = nn.Conv3d(
        in_channels=1,
        out_channels=768,
        kernel_size=(4, 16, 16),
        stride=(4, 16, 16),
        bias=True,
        device=device,
        dtype=dtype,
    )
    image = torch.randn(
        1,
        1,
        32,
        256,
        256,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )

    torch.cuda.synchronize(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()

    tokens_3d = patch_embed(image)
    loss = tokens_3d.float().square().mean()
    loss.backward()

    end.record()
    torch.cuda.synchronize(device)

    require(
        tuple(tokens_3d.shape) == expected_shape,
        f"3D patch embedding returned {tuple(tokens_3d.shape)}, expected "
        f"{expected_shape}.",
    )
    assert_finite("3D patch tokens", tokens_3d)
    assert_finite("3D patch image gradient", image.grad)

    result = {
        "input_shape": [1, 1, 32, 256, 256],
        "output_shape": list(expected_shape),
        "dtype": str(dtype).removeprefix("torch."),
        "forward_backward_ms": start.elapsed_time(end),
        "peak_memory_bytes": torch.cuda.max_memory_allocated(device),
        "peak_memory": human_bytes(torch.cuda.max_memory_allocated(device)),
        "loss": float(loss.detach().cpu()),
    }

    del patch_embed, image, tokens_3d, loss
    torch.cuda.empty_cache()
    return result


def nccl_all_reduce_test(
    *,
    device: torch.device,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    if world_size == 1:
        return {
            "executed": False,
            "reason": "WORLD_SIZE is one; no inter-rank collective was possible.",
        }

    # Four million FP32 values = 16 MiB per rank: large enough to exercise NCCL
    # without wasting a meaningful fraction of A100 memory.
    element_count = 4 * 1024 * 1024
    payload = torch.full(
        (element_count,),
        fill_value=float(rank + 1),
        dtype=torch.float32,
        device=device,
    )
    expected_value = world_size * (world_size + 1) / 2

    dist.barrier(device_ids=[device.index])
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    first_value = float(payload[0].item())
    require(
        math.isclose(first_value, expected_value, rel_tol=0.0, abs_tol=1e-5),
        f"NCCL all-reduce produced {first_value}, expected {expected_value}.",
    )
    require(
        bool(torch.all(payload == expected_value).item()),
        "NCCL all-reduce payload is not uniform after reduction.",
    )

    payload_bytes = payload.numel() * payload.element_size()
    del payload
    return {
        "executed": True,
        "payload_bytes": payload_bytes,
        "payload": human_bytes(payload_bytes),
        "elapsed_ms": elapsed * 1000.0,
        "effective_payload_gib_per_second": (
            payload_bytes / elapsed / (1024**3) if elapsed > 0 else None
        ),
        "expected_reduced_value": expected_value,
    }


def query_nvidia_smi() -> list[dict[str, str]]:
    output = run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,pci.bus_id,driver_version,memory.total",
            "--format=csv,noheader,nounits",
        ]
    )
    rows: list[dict[str, str]] = []
    for line in output.splitlines():
        fields = [field.strip() for field in line.split(",")]
        require(
            len(fields) == 6,
            f"Unexpected nvidia-smi row: {line!r}",
        )
        rows.append(
            {
                "index": fields[0],
                "name": fields[1],
                "uuid": fields[2],
                "pci_bus_id": fields[3],
                "driver_version": fields[4],
                "memory_mib": fields[5],
            }
        )
    return rows


def gather_rank_reports(
    local_report: dict[str, Any],
    *,
    world_size: int,
) -> list[dict[str, Any]]:
    if world_size == 1:
        return [local_report]

    gathered: list[dict[str, Any] | None] = [None for _ in range(world_size)]
    dist.all_gather_object(gathered, local_report)
    require(all(item is not None for item in gathered), "Rank report gathering failed.")
    return [item for item in gathered if item is not None]


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path = path.expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def print_rank(rank: int, message: str) -> None:
    print(f"[M3D preflight][rank {rank}] {message}", flush=True)


def main() -> int:
    args = parse_args()
    distributed_initialized = False
    rank = 0

    try:
        software = check_software_contract()
        rank, local_rank, world_size = initialize_distributed(args)
        distributed_initialized = dist.is_initialized()

        device, gpu = configure_device(
            args,
            local_rank=local_rank,
            world_size=world_size,
        )
        print_rank(rank, f"Using {gpu['name']} on cuda:{local_rank}")

        vision_attention = flash_sdpa_training_test(
            device=device,
            sequence_length=args.attention_seq_len,
            causal=False,
            label="vision",
        )
        print_rank(rank, "Vision Flash-SDPA forward/backward passed")

        language_attention = flash_sdpa_training_test(
            device=device,
            sequence_length=512,
            causal=True,
            label="language",
        )
        print_rank(rank, "Causal language Flash-SDPA forward/backward passed")

        patch_embedding = patch_embedding_3d_test(device)
        print_rank(rank, "3D patch-embedding forward/backward passed")

        nccl = nccl_all_reduce_test(
            device=device,
            rank=rank,
            world_size=world_size,
        )
        if nccl["executed"]:
            print_rank(rank, "NCCL all-reduce passed")

        local_report = {
            "rank": rank,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "gpu": gpu,
            "tests": {
                "vision_flash_sdpa": vision_attention,
                "language_flash_sdpa": language_attention,
                "patch_embedding_3d": patch_embedding,
                "nccl_all_reduce": nccl,
            },
        }
        rank_reports = gather_rank_reports(local_report, world_size=world_size)

        if rank == 0:
            report = {
                "status": "passed",
                "timestamp_unix": time.time(),
                "host": socket.gethostname(),
                "platform": platform.platform(),
                "pbs_job_id": os.environ.get("PBS_JOBID"),
                "world_size": world_size,
                "expected_gpus": args.expected_gpus,
                "software": software,
                "nvidia_smi": query_nvidia_smi(),
                "ranks": rank_reports,
            }
            atomic_write_json(args.report, report)
            print("\n[M3D preflight] ALL CHECKS PASSED", flush=True)
            print(f"[M3D preflight] Report: {args.report.resolve()}", flush=True)

        if distributed_initialized:
            dist.barrier(device_ids=[device.index])
        return 0

    except Exception as exc:
        print_rank(rank, f"FAILED: {type(exc).__name__}: {exc}")
        return 1

    finally:
        if distributed_initialized and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
