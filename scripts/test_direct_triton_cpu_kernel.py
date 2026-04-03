#!/usr/bin/env python3
"""
Direct Triton CPU kernel smoke test (manual launch with tl.constexpr).

Launches a small vector-add kernel with BLOCK_SIZE passed as a keyword (same
pattern as hand-written kernels). This exercises the CPU backend launcher with
constexpr in the launch tuple and catches arity / PyArg_ParseTuple mismatches
between that path and Inductor-style launches that omit constexpr arguments.

Requires: torch (CPU), triton built with CPU backend.
"""
from __future__ import annotations

import sys

import torch
import triton
import triton.language as tl
from triton.backends import backends


@triton.jit
def add_kernel(
    a_ptr,
    b_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, a + b, mask=mask)


def triton_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape
    a = a.contiguous()
    b = b.contiguous()
    out = torch.empty_like(a)
    n = out.numel()
    block = 1024
    grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
    add_kernel[grid](a, b, out, n, BLOCK_SIZE=block)
    return out


def main() -> int:
    if "cpu" not in backends:
        print("error: Triton CPU backend not available (expected 'cpu' in triton.backends.backends)", file=sys.stderr)
        return 2

    device = torch.device("cpu")
    size = 100_000
    torch.manual_seed(0)
    a = torch.rand(size, device=device, dtype=torch.float32)
    b = torch.rand(size, device=device, dtype=torch.float32)
    triton_out = triton_add(a, b)
    ref = a + b
    max_diff = torch.max(torch.abs(triton_out - ref)).item()
    print(f"Maximum difference between Triton and PyTorch: {max_diff}")
    if max_diff >= 1e-5:
        print("error: results do not match", file=sys.stderr)
        return 1
    print("ok: direct Triton CPU kernel (constexpr launch) passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
