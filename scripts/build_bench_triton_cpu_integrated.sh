#!/usr/bin/env bash
# Stable benchmark build for bench_triton_cpu_integrated (see LOCAL_SETUP.md).
# Builds CPU-only PyTorch from pytorch-stable and Triton CPU from this repo.

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: build_bench_triton_cpu_integrated.sh <command>

Commands:
  pytorch   Build and install PyTorch (pytorch-stable), CPU-only.
  triton    Build and install Triton CPU (triton-cpu-stable).
  all       Run pytorch, triton, then the direct-kernel test (below).
  direct    Run scripts/test_direct_triton_cpu_kernel.py (needs torch + triton in env).
  smoke     Optional Inductor + Triton CPU checks (pytorch-stable).

Environment (optional overrides):
  CONDA_BASE       Miniconda root (default: $HOME/miniconda3)
  CONDA_ENV        Env name (default: bench_triton_cpu_integrated)
  PYTORCH_REPO     PyTorch stable tree (default: sibling ../pytorch-stable next to this repo)
  TRITON_REPO      This Triton tree (default: parent of scripts/)
  LLVM_BUILD       LLVM build prefix (default: $LLM_CPU_COMPILER_ROOT/llvm-project/build)
  LLM_CPU_COMPILER_ROOT  Parent of llvm-project, pytorch-stable, triton-cpu-stable
                         (default: parent directory of TRITON_REPO, e.g. .../llm-cpu-compiler)
  DIRECT_TRITON_TEST     Path to direct kernel smoke script
                         (default: $TRITON_REPO/scripts/test_direct_triton_cpu_kernel.py)

Example:
  ./scripts/build_bench_triton_cpu_integrated.sh all
EOF
  exit "${1:-0}"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRITON_REPO="${TRITON_REPO:-$(cd "$SCRIPT_DIR/.." && pwd)}"
LLM_CPU_COMPILER_ROOT="${LLM_CPU_COMPILER_ROOT:-$(cd "$TRITON_REPO/.." && pwd)}"
PYTORCH_REPO="${PYTORCH_REPO:-$(cd "$TRITON_REPO/../pytorch-stable" && pwd)}"
LLVM_BUILD="${LLVM_BUILD:-$LLM_CPU_COMPILER_ROOT/llvm-project/build}"
CONDA_BASE="${CONDA_BASE:-$HOME/miniconda3}"
CONDA_ENV="${CONDA_ENV:-bench_triton_cpu_integrated}"
DIRECT_TRITON_TEST="${DIRECT_TRITON_TEST:-$TRITON_REPO/scripts/test_direct_triton_cpu_kernel.py}"

activate_conda() {
  if [[ "${CONDA_DEFAULT_ENV:-}" == "$CONDA_ENV" ]]; then
    return 0
  fi
  if [[ ! -f "$CONDA_BASE/etc/profile.d/conda.sh" ]]; then
    echo "error: conda.sh not found at $CONDA_BASE/etc/profile.d/conda.sh" >&2
    echo "Set CONDA_BASE or run: conda activate $CONDA_ENV" >&2
    exit 1
  fi
  # shellcheck source=/dev/null
  source "$CONDA_BASE/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
}

build_pytorch() {
  echo "==> PyTorch (CPU) from $PYTORCH_REPO"
  [[ -d "$PYTORCH_REPO" ]] || { echo "error: PYTORCH_REPO missing: $PYTORCH_REPO" >&2; exit 1; }
  cd "$PYTORCH_REPO"

  conda install -y -c conda-forge "cmake>=3.18,<4" ninja
  python -m pip uninstall -y torch || true
  rm -rf build

  export CMAKE_PREFIX_PATH="${CONDA_PREFIX:-}"
  export USE_CUDA=0
  export USE_ROCM=0
  export USE_XPU=0
  export MAX_JOBS="${MAX_JOBS:-16}"

  python -m pip install -r requirements.txt
  python -m pip install -v --no-build-isolation .
}

build_triton() {
  echo "==> Triton CPU from $TRITON_REPO"
  [[ -d "$TRITON_REPO/python" ]] || { echo "error: TRITON_REPO/python missing: $TRITON_REPO" >&2; exit 1; }
  cd "$TRITON_REPO"

  python -m pip uninstall -y triton || true
  rm -rf python/build python/__pycache__

  export TRITON_CODEGEN_BACKENDS=cpu
  export LLVM_SYSPATH="$LLVM_BUILD"
  export LLVM_INCLUDE_DIRS="$LLVM_BUILD/include"
  export LLVM_LIBRARY_DIR="$LLVM_BUILD/lib"
  export MAX_JOBS="${MAX_JOBS:-16}"

  [[ -d "$LLVM_BUILD" ]] || echo "warning: LLVM_BUILD not a directory: $LLVM_BUILD (build may fail)" >&2

  python -m pip install -v --no-build-isolation ./python
}

run_direct_triton_kernel_test() {
  echo "==> Direct Triton CPU kernel test: $DIRECT_TRITON_TEST"
  [[ -f "$DIRECT_TRITON_TEST" ]] || { echo "error: missing $DIRECT_TRITON_TEST" >&2; exit 1; }
  python "$DIRECT_TRITON_TEST"
}

run_smoke() {
  echo "==> Smoke: Triton CPU + Inductor (pytorch-stable)"
  cd "$PYTORCH_REPO"
  python test/inductor/triton_cpu_compile_integration.py --fast --fresh --allow-extern-gemm
  python test/inductor/test_triton_cpu_compile_integration.py
}

main() {
  local cmd="${1:-}"
  [[ "$cmd" == "-h" || "$cmd" == "--help" ]] && usage 0
  [[ -z "$cmd" ]] && usage 1

  activate_conda

  case "$cmd" in
    pytorch) build_pytorch ;;
    triton) build_triton ;;
    all)
      build_pytorch
      build_triton
      run_direct_triton_kernel_test
      ;;
    direct) run_direct_triton_kernel_test ;;
    smoke) run_smoke ;;
    *) usage 1 ;;
  esac

  echo "==> Done: $cmd"
}

main "$@"
