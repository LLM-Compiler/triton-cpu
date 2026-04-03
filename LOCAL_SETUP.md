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

## Automated bench build script (PyTorch + Triton + direct kernel check)

From **`triton-cpu-stable`** repo root, this script activates `bench_triton_cpu_integrated` (via `conda.sh` under `CONDA_BASE`), builds CPU-only PyTorch from the sibling **`pytorch-stable`** tree, installs Triton CPU from this repo, then runs the **direct kernel** test below.

```bash
cd …/llm-cpu-compiler/triton-cpu-stable
./scripts/build_bench_triton_cpu_integrated.sh all
```

Commands: `pytorch`, `triton`, `all` (pytorch → triton → direct test), `direct` (only the direct kernel test), `smoke` (Inductor integration under `pytorch-stable`). Override paths with `PYTORCH_REPO`, `TRITON_REPO`, `LLVM_BUILD`, `CONDA_BASE`, `CONDA_ENV` if needed (`--help` in the script lists them).

## Direct Triton CPU kernel test (manual launch + `tl.constexpr`)

**File:** `scripts/test_direct_triton_cpu_kernel.py`

Small vector-add kernel launched like a hand-written example (`BLOCK_SIZE` as `tl.constexpr` and passed at launch). Use this to confirm the CPU launcher accepts **manual** constexpr launches after a Triton install (and to catch `PyArg_ParseTuple` arity bugs vs Inductor-style calls that omit constexpr slots).

```bash
conda activate bench_triton_cpu_integrated
cd …/llm-cpu-compiler/triton-cpu-stable
python scripts/test_direct_triton_cpu_kernel.py
```

Requires **PyTorch** (CPU tensors) and Triton with **`cpu` in `triton.backends.backends`**. It is also run automatically at the end of `./scripts/build_bench_triton_cpu_integrated.sh all`.

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
python -m pip uninstall -y triton
rm -rf python/build python/__pycache__
export TRITON_CODEGEN_BACKENDS=cpu
export LLVM_SYSPATH=/home/shavin/llm-cpu-compiler/llvm-project/build
export LLVM_INCLUDE_DIRS=/home/shavin/llm-cpu-compiler/llvm-project/build/include
export LLVM_LIBRARY_DIR=/home/shavin/llm-cpu-compiler/llvm-project/build/lib
export MAX_JOBS=16
python -m pip install -v --no-build-isolation ./python
```

## Post-build: Triton CPU + Inductor smoke (LLM-style `torch.compile`)

After PyTorch and Triton (CPU backend) are installed in this stable tree (see above), run from **`pytorch-stable` repo root**:

**Manual:**

```bash
conda activate bench_triton_cpu_integrated
cd …/llm-cpu-compiler/pytorch-stable
python test/inductor/triton_cpu_compile_integration.py --fast --fresh --allow-extern-gemm
```

Strict check (no `extern_kernels.*` in generated `output_code.py`) only passes if Inductor routes GEMMs to Triton on CPU; omit `--allow-extern-gemm` when validating that path.

**As a test:**

```bash
cd …/llm-cpu-compiler/pytorch-stable
python test/inductor/test_triton_cpu_compile_integration.py
```

Requires Triton with the **CPU** backend (`"cpu" in triton.backends.backends`). Skipped otherwise.
