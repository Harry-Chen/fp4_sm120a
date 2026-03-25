# NVFP4 Support for SM120 Family

Patches and kernels to enable Transformer Engine's NVFP4 (FP4 E2M1) training
on NVIDIA SM120 family GPUs (RTX 50x0, DGX Spark), where several SM100-era
PTX instructions are missing or changed.

## Background

Transformer Engine uses FP4 E2M1 quantization for forward-pass activations and
weights in FP4 training recipes. The kernels rely on SM100-specific PTX
instructions that are not all available on the SM120 family:

| Instruction | SM100 | SM120 | Status |
|-------------|-------|-------|--------|
| `cvt.rs.satfinite.e2m1x4.f32` | Yes | **No** | Polyfilled in `stochastic_rounding/` |
| `cvt.rn.satfinite.e2m1x2.f32` | Yes | Yes | Used by polyfill |
| RHT GEMM (Random Hadamard Transform) | SM100 UMMA/tcgen05 | WMMA polyfill | Done in `rht_gemm/` |

## Components

### [`stochastic_rounding/`](stochastic_rounding/) — Done

Drop-in polyfills for the stochastic rounding FP4 conversion functions removed
in SM120. Two implementations:

- **`sr.sm120.cuh`** — Uses `cvt.rn.satfinite.e2m1x2.f32` + software SR noise
  injection. Requires SM120 family hardware.
- **`sr.software.cuh`** — Pure software quantization. Works on any CUDA
  architecture.

Polyfilled functions:
- `mul_cvt_bf16_to_fp4_4x_with_stochastic_rounding`
- `mul_cvt_fp32_to_fp4_4x_with_stochastic_rounding`
- `cvt_fp32_to_fp4_4x_with_stochastic_rounding`
- `mul_cvt_bf16_to_fp4_8x_stochastic_rounding`

Validated against native SM100 hardware (`compare.cu` on B300) — bit-exact
match for the SM120 polyfill, statistical equivalence for the software polyfill.

See [`stochastic_rounding/README.md`](stochastic_rounding/README.md) for details.

### [`rht_gemm/`](rht_gemm/) — Done

Drop-in replacement for Transformer Engine's `rht_gemm_ntt_w_sfc` (Random
Hadamard Transform GEMM) using WMMA instead of SM100's UMMA/tcgen05.

Validated against native SM100 TE reference on GB300 — **0% mismatch** on both
FP4 output and SFC scale factors across all tested sizes (256×128 to 8192×5120).

## Building

Each component has its own Makefile. See the README in each subdirectory.

```bash
# Build stochastic rounding tests (SM120)
cd stochastic_rounding && make

# Build stochastic rounding comparison test (requires SM100 hardware)
cd stochastic_rounding && make compare.exe CUDA_ARCH=100a

# Build RHT GEMM correctness test (SM120)
cd rht_gemm && make

# Build RHT GEMM comparison test vs TE reference (requires SM100 hardware)
cd rht_gemm && make test_compare.exe
```

## License

Apache 2.0
