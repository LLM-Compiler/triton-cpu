# Local setup (this machine)

Living notes: conda envs, worktrees, and builds. Update as things change.

## Conda

| Env | Python | Purpose |
|-----|--------|---------|
| `base` | 3.13 | Default Miniconda env; not used for PyTorch/Triton work. |
| `triton_cpu_integrated` | 3.9 | Dev: editable PyTorch from this repo, Triton CPU backend, frequent rebuilds. |
| `bench_triton_cpu_integrated` | 3.9 | Benchmarks: stable installs (e.g. wheels), separate from dev rebuilds. |

- **Root:** `/home/shavin/miniconda3`
- **Shell:** `conda` init in `~/.bashrc`; Intel oneAPI `setvars.sh` + `TRITON_CPU_BACKEND=1`; **no** auto-`conda activate` — activate the env you need manually.

## Git worktrees

| Path | Branch | Commit (at add) | Role |
|------|--------|-------------------|------|
| `…/llm-cpu-compiler/pytorch` | `feature-branch` | `latest active commit` | Primary clone (this file). |
| `…/llm-cpu-compiler/pytorch-stable` | `triton_cpu_backend` | `41f9c98caf` | Second worktree; use for stable / pinned work vs main tree. |

### triton-cpu (separate repo)

| Path | Branch | Commit (at add) | Role |
|------|--------|-------------------|------|
| `…/llm-cpu-compiler/triton-cpu` | `feature branch` | `last commit` | Primary clone. |
| `…/llm-cpu-compiler/triton-cpu-stable` | `triton_cpu_backend` | `a274cb00` | Second worktree; stable / pinned Triton vs main tree. |

## Builds per env / tree

_TBD._ Record pinned torch commit or wheel version per bench env; note editable dev path for `triton_cpu_integrated`.

## Stable benchmark build command (PyTorch)

Run from `…/llm-cpu-compiler/pytorch-stable` with `bench_triton_cpu_integrated`:

```bash
conda activate bench_triton_cpu_integrated
conda install -y -c conda-forge "cmake>=3.18,<4" ninja
python -m pip uninstall -y torch || true
rm -rf build
export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname "$(which conda)")/../"}
export USE_CUDA=0
export USE_ROCM=0
export USE_XPU=0
export MAX_JOBS=16
python -m pip install -r requirements.txt
python -m pip install -v --no-build-isolation .
```
## Stable benchmark build command (Triton CPU)

Run from `…/llm-cpu-compiler/triton-cpu-stable` with `bench_triton_cpu_integrated`:

```bash
conda activate bench_triton_cpu_integrated
rm -rf python/build python/__pycache__
export TRITON_CODEGEN_BACKENDS=cpu
export LLVM_SYSPATH=/home/shavin/llm-cpu-compiler/llvm-project/build
export LLVM_INCLUDE_DIRS=/home/shavin/llm-cpu-compiler/llvm-project/build/include
export LLVM_LIBRARY_DIR=/home/shavin/llm-cpu-compiler/llvm-project/build/lib
export MAX_JOBS=16
python -m pip install -v --no-build-isolation ./python
```
