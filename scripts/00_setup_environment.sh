#!/usr/bin/env bash
#
# M3D-Modernized — ASPIRE 2A environment bootstrap
# =================================================
# This is the first file to run.
#
# Target platform:
#   NSCC ASPIRE 2A
#   RHEL 8
#   NVIDIA A100 GPUs
#
# ASPIRE 2A software profile used by this project:
#   PrgEnv-gnu : 8.3.3
#   GCC        : 11.4.0-nscc
#   Python     : 3.10.9
#   CUDA       : 11.8.0
#   CMake      : 3.31.3
#   Ninja      : 1.11.1
#   PyTorch    : 2.6.0 + CUDA 11.8
#   TorchVision: 0.21.0 + CUDA 11.8
#
# Why PyTorch is installed inside the virtual environment instead of loading
# pytorch/2.6.0-py3-cu11.8 directly:
#   - the project remains isolated from unrelated packages in the module;
#   - pip freeze gives a complete reproducibility lock;
#   - later package installation cannot silently modify the shared module;
#   - the selected PyTorch/CUDA versions still exactly match ASPIRE 2A's
#     available PyTorch 2.6.0 CUDA 11.8 software profile.
#
# We deliberately do not load separate cuDNN or NCCL modules. The official
# PyTorch CUDA wheel carries the matching CUDA user-space libraries. Loading a
# second cuDNN/NCCL stack can create runtime symbol conflicts.
#
# Usage on an ASPIRE 2A login node:
#   cd /scratch/users/industry/theiahealth/theiahth/M3D-modernized
#   bash scripts/00_setup_environment.sh
#
# Optional custom locations:
#   M3D_VENV_DIR=/scratch/users/industry/theiahealth/theiahth/envs/m3d-modern \
#   M3D_CACHE_ROOT=/scratch/users/industry/theiahealth/theiahth/.cache/m3d-modern \
#   bash scripts/00_setup_environment.sh
#
# 它的职责是：
# 1. 加载 ASPIRE 2A 上固定版本的软件模块。
# 2. 创建独立的 Python 3.10 虚拟环境。
# 3. 安装 PyTorch 2.6.0、TorchVision 0.21.0 和 requirements.txt 中的依赖。
# 4. 验证 Python、CUDA、PyTorch 和关键依赖是否正确。
# 5. 有 GPU 时测试 A100、BF16 和 PyTorch Flash-SDPA。
# 6. 生成环境锁定文件，方便以后复现。
#
# 成功后，项目目录大致会多出：
#
# M3D-modernized/
# ├── .venv/                         # Python 虚拟环境
# ├── .cache/
# │   ├── pip/                       # pip 下载缓存
# │   ├── huggingface/               # Hugging Face 缓存目录
# │   ├── torch/                     # PyTorch 缓存目录
# │   ├── triton/                    # Triton 编译缓存
# │   └── torch_extensions/          # PyTorch C++/CUDA 扩展缓存
# ├── modules.lock.txt               # 本次加载的系统 modules
# ├── environment.lock.txt           # 本次安装的全部 Python 包版本
# ├── requirements.txt
# └── scripts/
#     └── 00_setup_environment.sh
#

set -Eeuo pipefail
IFS=$'\n\t'
umask 022

readonly SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)" # /scratch/users/industry/theiahealth/theiahth/M3D-modernized/scripts
readonly PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd -P)" # /scratch/users/industry/theiahealth/theiahth/M3D-modernized

# -----------------------------------------------------------------------------
# ASPIRE 2A module versions
# -----------------------------------------------------------------------------
readonly MODULE_PRGENV="PrgEnv-gnu/8.3.3"
readonly MODULE_GCC="gcc/11.4.0-nscc"
readonly MODULE_PYTHON="python/3.10.9"
readonly MODULE_CUDA="cuda/11.8.0"
readonly MODULE_CMAKE="cmake/3.31.3"
readonly MODULE_NINJA="ninja/1.11.1"

# PyTorch 2.6.0 is the CUDA 11.8 PyTorch version listed on ASPIRE 2A.
readonly TORCH_VERSION="2.6.0"
readonly TORCHVISION_VERSION="0.21.0"
readonly PYTORCH_INDEX_URL="https://download.pytorch.org/whl/cu118"

# -----------------------------------------------------------------------------
# User-overridable local paths
# -----------------------------------------------------------------------------
readonly VENV_DIR="${M3D_VENV_DIR:-${PROJECT_ROOT}/.venv}"
readonly CACHE_ROOT="${M3D_CACHE_ROOT:-${PROJECT_ROOT}/.cache}"
readonly REQUIREMENTS_FILE="${PROJECT_ROOT}/requirements.txt"
readonly LOCK_FILE="${PROJECT_ROOT}/environment.lock.txt"
readonly MODULE_LOCK_FILE="${PROJECT_ROOT}/modules.lock.txt"

export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${CACHE_ROOT}/pip}"
export HF_HOME="${HF_HOME:-${CACHE_ROOT}/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${CACHE_ROOT}/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${CACHE_ROOT}/triton}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-${CACHE_ROOT}/torch_extensions}"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false

log() {
    printf '[M3D setup] %s\n' "$*"
}

fail() {
    printf '[M3D setup] ERROR: %s\n' "$*" >&2
    exit 1
}

on_error() {
    local exit_code=$?
    local line_number=${1:-unknown}
    printf '[M3D setup] ERROR: command failed at line %s (exit code %s).\n' \
        "${line_number}" "${exit_code}" >&2
    exit "${exit_code}"
}
trap 'on_error ${LINENO}' ERR

# The "module" command is normally a shell function, not an executable.
type module >/dev/null 2>&1 || \
    fail "The environment-modules command is unavailable. Run this on ASPIRE 2A."

# -----------------------------------------------------------------------------
# 1. Load an explicit ASPIRE 2A software stack
# -----------------------------------------------------------------------------
log "Purging loaded modules"
module purge

log "Loading ${MODULE_PRGENV}"
module load "${MODULE_PRGENV}"

# Load all other modules first because one of them may replace GCC.
log "Loading ${MODULE_PYTHON}"
module load "${MODULE_PYTHON}"

log "Loading ${MODULE_CUDA}"
module load "${MODULE_CUDA}"

log "Loading ${MODULE_CMAKE}"
module load "${MODULE_CMAKE}"

log "Loading ${MODULE_NINJA}"
module load "${MODULE_NINJA}"

# Detect the GCC module that remains after all dependency resolution.
LOADED_GCC="$(
    module -t list 2>&1 |
        grep -m1 '^gcc/' || true
)"

if [[ -z "${LOADED_GCC}" ]]; then
    log "No GCC module is loaded; loading ${MODULE_GCC}"
    module load "${MODULE_GCC}"

elif [[ "${LOADED_GCC}" != "${MODULE_GCC}" ]]; then
    log "Swapping ${LOADED_GCC} for ${MODULE_GCC}"
    module swap "${LOADED_GCC}" "${MODULE_GCC}"

else
    log "Required GCC already loaded: ${MODULE_GCC}"
fi

# Clear Bash's cached command paths after changing compiler modules.
hash -r

# Verify the final module state.
FINAL_GCC_MODULE="$(
    module -t list 2>&1 |
        grep -m1 '^gcc/' || true
)"

[[ "${FINAL_GCC_MODULE}" == "${MODULE_GCC}" ]] || \
    fail "Expected final GCC module ${MODULE_GCC}, but resolved ${FINAL_GCC_MODULE:-none}."

FINAL_GCC_VERSION="$(gcc -dumpfullversion -dumpversion)"

[[ "${FINAL_GCC_VERSION}" == 11.4.0* ]] || \
    fail "Expected GCC 11.4.0, but gcc reports ${FINAL_GCC_VERSION}."

# Record exactly what the module system resolved.
module -t list 2> "${MODULE_LOCK_FILE}"

log "Resolved module stack:"
cat "${MODULE_LOCK_FILE}"

# -----------------------------------------------------------------------------
# 2. Validate system tools before creating the environment
# -----------------------------------------------------------------------------
command -v python >/dev/null 2>&1 || fail "python was not provided by ${MODULE_PYTHON}."
command -v gcc >/dev/null 2>&1 || fail "gcc was not provided by ${MODULE_GCC}."
command -v nvcc >/dev/null 2>&1 || fail "nvcc was not provided by ${MODULE_CUDA}."
command -v cmake >/dev/null 2>&1 || fail "cmake was not provided by ${MODULE_CMAKE}."
command -v ninja >/dev/null 2>&1 || fail "ninja was not provided by ${MODULE_NINJA}."

python - <<'PY'
import sys

expected = (3, 10)
actual = sys.version_info[:2]

if actual != expected:
    raise SystemExit(
        f"Expected Python 3.10 from ASPIRE 2A, detected {sys.version.split()[0]}."
    )

print(f"[M3D setup] Python: {sys.version.split()[0]}")
PY

log "GCC: $(gcc --version | head -n 1)"
log "NVCC: $(nvcc --version | grep 'release' | tail -n 1 | sed 's/^ *//')"
log "CMake: $(cmake --version | head -n 1)"
log "Ninja: $(ninja --version)"

# Derive CUDA_HOME from the module-provided nvcc location.
readonly CUDA_BIN_DIR="$(cd -- "$(dirname -- "$(command -v nvcc)")" && pwd -P)"
export CUDA_HOME="$(cd -- "${CUDA_BIN_DIR}/.." && pwd -P)"
export CUDACXX="${CUDA_HOME}/bin/nvcc"

log "CUDA_HOME: ${CUDA_HOME}"

mkdir -p \
    "${PIP_CACHE_DIR}" \
    "${HF_HOME}" \
    "${TORCH_HOME}" \
    "${TRITON_CACHE_DIR}" \
    "${TORCH_EXTENSIONS_DIR}"

# -----------------------------------------------------------------------------
# 3. Create an isolated Python virtual environment
# -----------------------------------------------------------------------------
if [[ ! -d "${VENV_DIR}" ]]; then
    log "Creating virtual environment: ${VENV_DIR}"
    python -m venv "${VENV_DIR}"
else
    log "Reusing virtual environment: ${VENV_DIR}"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# Reject a venv created by a different Python minor version.
python - <<'PY'
import sys

if sys.version_info[:2] != (3, 10):
    raise SystemExit(
        "The existing virtual environment was not created with Python 3.10. "
        "Delete it or choose a new M3D_VENV_DIR."
    )
PY

log "Upgrading packaging tools"
python -m pip install --upgrade \
    "pip==25.2" \
    "setuptools==80.9.0" \
    "wheel==0.45.1"

# -----------------------------------------------------------------------------
# 4. Install the ASPIRE 2A-aligned PyTorch stack
# -----------------------------------------------------------------------------
log "Installing PyTorch ${TORCH_VERSION} with CUDA 11.8"
python -m pip install --upgrade \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    --index-url "${PYTORCH_INDEX_URL}"

# requirements.txt is intentionally reviewed as the next project file.
if [[ ! -f "${REQUIREMENTS_FILE}" ]]; then
    fail "Missing ${REQUIREMENTS_FILE}. Review and add requirements.txt before rerunning setup."
fi

log "Installing remaining M3D dependencies"
python -m pip install --upgrade --requirement "${REQUIREMENTS_FILE}"

log "Checking Python dependency consistency"
python -m pip check

# -----------------------------------------------------------------------------
# 5. CPU-safe package validation
# -----------------------------------------------------------------------------
log "Running package validation"
python - <<'PY'
from __future__ import annotations

import importlib
import platform

import torch

expected_torch = "2.6.0"

if not torch.__version__.startswith(expected_torch):
    raise RuntimeError(
        f"Expected PyTorch {expected_torch}, detected {torch.__version__}."
    )

if torch.version.cuda != "11.8":
    raise RuntimeError(
        f"Expected PyTorch CUDA runtime 11.8, detected {torch.version.cuda}."
    )

required_modules = (
    "accelerate",
    "einops",
    "monai",
    "nibabel",
    "numpy",
    "pandas",
    "peft",
    "safetensors",
    "transformers",
)

print(f"[M3D setup] Platform: {platform.platform()}")
print(f"[M3D setup] PyTorch: {torch.__version__}")
print(f"[M3D setup] PyTorch CUDA runtime: {torch.version.cuda}")
print(f"[M3D setup] CUDA visible on this node: {torch.cuda.is_available()}")

for module_name in required_modules:
    module = importlib.import_module(module_name)
    version = getattr(module, "__version__", "unknown")
    print(f"[M3D setup] {module_name}: {version}")
PY

# -----------------------------------------------------------------------------
# 6. GPU validation when the setup is run inside a GPU allocation
# -----------------------------------------------------------------------------
python - <<'PY'
from __future__ import annotations

import torch
import torch.nn.functional as F

if not torch.cuda.is_available():
    print(
        "[M3D setup] No GPU is visible on this node. The A100/BF16/Flash-SDPA "
        "checks will run again in the PBS training preflight."
    )
    raise SystemExit(0)

properties = torch.cuda.get_device_properties(0)
capability = torch.cuda.get_device_capability(0)

print(f"[M3D setup] GPU: {properties.name}")
print(f"[M3D setup] Compute capability: {capability[0]}.{capability[1]}")
print(f"[M3D setup] GPU memory: {properties.total_memory / 1024**3:.1f} GiB")

# ASPIRE 2A A100 GPUs have compute capability 8.0.
if capability < (8, 0):
    raise RuntimeError(
        "This optimized profile requires an Ampere-class GPU or newer."
    )

if not torch.cuda.is_bf16_supported():
    raise RuntimeError("The allocated GPU does not report BF16 support.")

q = torch.randn(1, 4, 128, 64, device="cuda", dtype=torch.bfloat16)
k = torch.randn_like(q)
v = torch.randn_like(q)

# Force the FlashAttention SDPA backend for this smoke test. This verifies that
# the exact backend required by the rewritten M3D attention layers is usable.
from torch.nn.attention import SDPBackend, sdpa_kernel

with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
    output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        dropout_p=0.0,
        is_causal=False,
    )

if output.shape != q.shape:
    raise RuntimeError(
        f"Flash SDPA returned {tuple(output.shape)}, expected {tuple(q.shape)}."
    )

if not torch.isfinite(output).all():
    raise RuntimeError("Flash SDPA produced non-finite values.")

print("[M3D setup] A100 BF16 Flash-SDPA smoke test: passed")
PY

# -----------------------------------------------------------------------------
# 7. Write an exact reproducibility lock
# -----------------------------------------------------------------------------
log "Writing Python package lock: ${LOCK_FILE}"
python -m pip freeze --all | LC_ALL=C sort > "${LOCK_FILE}"

cat <<EOF_MESSAGE

[M3D setup] ASPIRE 2A environment setup completed.

Activate it in future login/PBS sessions only after loading the same modules:

    module reset
    module load ${MODULE_PRGENV}
    module load ${MODULE_GCC}
    module load ${MODULE_PYTHON}
    module load ${MODULE_CUDA}
    module load ${MODULE_CMAKE}
    module load ${MODULE_NINJA}
    source "${VENV_DIR}/bin/activate"

Module lock:
    ${MODULE_LOCK_FILE}

Python package lock:
    ${LOCK_FILE}
EOF_MESSAGE
