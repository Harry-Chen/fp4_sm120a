# Random Hadamard Transform GEMM for SM120

SM120 port of the Random Hadamard Transform (RHT) GEMM kernel used by
Transformer Engine's FP4 training recipe.

## Overview

The FP4 training recipe applies a random Hadamard rotation to activations and
weights before FP4 quantization. The existing Transformer Engine kernel
(`rht_gemm_ntt_w_sfc` in `hadamard_transform_cast_fusion.cu`) uses SM100-only
features (UMMA/tcgen05, TMEM, 232KB shared memory) that are not available on
SM120 family GPUs (RTX 50x0).

This implementation provides a drop-in replacement using:

- **WMMA** (`wmma.mma.sync.aligned.m16n16k16.f32.bf16.bf16.f32`) for the 16x16
  Hadamard matrix multiply — the natural fit since B is always exactly 16x16.
- **PTX FP4/FP8 conversion** (`cvt.rn.satfinite.e2m1x2.f32`,
  `cvt.rn.satfinite.e4m3x2.f32`) for quantization.
- **Software stochastic rounding** (noise injection + round-to-nearest) from
  the `stochastic_rounding/` polyfill since `cvt.rs.satfinite.e2m1x4.f32` is
  not available on SM120.

Shared memory usage is ~13KB per block, well within SM120's 99KB limit.

## Operation

```
Input:  A (m × n, BF16, col-major)  ×  B (16 × 16, BF16, Hadamard matrix)
Output: C (m × n, FP4 E2M1, row-major)  +  SFC (m × n/16, FP8 UE4M3, row-major)
```

For each group of 16 columns in A, the kernel:
1. Multiplies by the 16×16 Hadamard matrix B via WMMA
2. Optionally rounds through BF16 for bitwise compatibility (`!kUseFastMath`)
3. Computes per-16-element amax and FP8 UE4M3 scale factor (SFC)
4. Quantizes to FP4 E2M1 with optional stochastic rounding

## API

```cpp
#include "rht_gemm_sm120.cuh"

rht_gemm_sm120::rht_gemm_ntt_w_sfc<TA, TB, TC, TSFC,
    kEnableStochasticRounding, kUseFastMath>(
    m, n, A, B, C, SFC, global_amax, rng_state, sm_count, stream, k_tile_size);
```

Same signature as the reference `rht_gemm_ntt_w_sfc`. Constraints: `m % 128 == 0`
and `n % 64 == 0`.

## Building

### Correctness test + benchmark (SM120)

Runs on RTX 5090 or any SM120 family GPU. Compares the kernel output against
a naive CPU reference.

```bash
make                        # default CUDA_ARCH=120a
make CUDA_ARCH=120f         # family-compatible
make CUDA_ARCH=121a         # SM121 specific
./test_correctness.exe
```

Requires CUDA 13.1+ and CUTLASS headers at `$CUTLASS_HOME` (default
`/home/harry/cutlass`). Header-only — no CUTLASS runtime dependency.

### Comparison test vs TE reference (SM100)

Compiles both our WMMA kernel and the original Transformer Engine SM100
UMMA kernel into a single binary. Must be **run on an SM100 GPU** (GB200,
B200, etc.) to execute both kernels and compare outputs byte-by-byte.

```bash
make test_compare.exe       # compiles for compute_100a
./test_compare.exe          # run on GB200
```

Additional requirements:

- **CUTLASS** source at `$CUTLASS_HOME` (default `/home/harry/cutlass`)
- **Transformer Engine** source at `$TE_HOME` (default
  `/home/harry/TransformerEngine`) — only the header files are used
- The TE headers with heavy dependencies (cuDNN, etc.) are replaced by
  minimal stubs in `te_ref/stubs/`; the real TE `ptx.cuh` and
  `curanddx.hpp` are symlinked from the TE source tree

The comparison test uses separate compilation (`-rdc=true`) to isolate our
kernel's includes (`sr.sm120.cuh`) from the TE reference's includes
(`ptx.cuh`) which define overlapping symbols.

## Performance

On RTX 5090 (1792 GB/s memory bandwidth):

| Size (m × n) | Time (ms) | Bandwidth (GB/s) | % of Peak |
|---|---|---|---|
| 1024 × 1024 | 0.006 | 431 | 24% |
| 2048 × 2048 | 0.012 | 871 | 49% |
| 4096 × 4096 | 0.035 | 1216 | 68% |
| 8192 × 5120 | 0.085 | 1259 | 70% |
| 8192 × 10240 | 0.169 | 1270 | 71% |

The kernel is memory-bound (arithmetic intensity ~12.7 FLOP/byte). The ~29%
gap to peak is primarily due to write amplification: FP4 output is row-major
with stride N/2 bytes between rows, causing scattered writes that hit many
L2 cache lines.

## Files

- `rht_gemm_sm120.cuh` — Main kernel header (drop-in replacement)
- `sr.sm120.cuh` — Symlink to `../stochastic_rounding/sr.sm120.cuh` (FP4 SR polyfill)
- `test_correctness.cu` — Correctness test vs naive CPU reference + benchmark
- `test_compare.cu` — Main driver for SM100 comparison test
- `test_compare_ours.cu` — Our kernel instantiation (separate TU)
- `test_compare_ref.cu` — TE reference kernel instantiation (separate TU)
- `te_ref/` — Original TE reference implementation (SM100 only)
  - `hadamard_transform_cast_fusion.cu` — Full TE source (for reference)
  - `hadamard_transform_cast_fusion_core.inc` — Extracted core kernel code
  - `stubs/` — Minimal header stubs replacing TE's heavy dependencies
