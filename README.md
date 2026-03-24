# FP4 E2M1 Stochastic Rounding Polyfill for SM120 Family

Kernel polyfill for `mul_cvt_bf16_to_fp4_4x_with_stochastic_rounding` on devices
that lack the `cvt.rs.satfinite.e2m1x4.f32` PTX instruction — primarily the
NVIDIA RTX 5090 (Blackwell, SM120) and DGX Spark (SM121).

## Background

SM120a removed the single-instruction stochastic-rounding FP4 conversion
(`cvt.rs.satfinite.e2m1x4.f32`) that was available on SM100a. The replacement
instruction `cvt.rn.satfinite.e2m1x2.f32` only supports **round-to-nearest** and
packs **2** E2M1 values at a time instead of 4.

This repo contains two independent polyfill implementations — one generated with
Claude, one with ChatGPT — that restore stochastic rounding semantics on top of
the available hardware.

## File Structure

```
cvt.claude.cuh      # Drop-in header: hardware RN + software SR noise
cvt.claude.cu       # Tests and benchmarks for the Claude implementation
cvt.chatgpt.cuh     # Drop-in header: pure software quantization
cvt.chatgpt.cu      # Tests and benchmarks for the ChatGPT implementation
```

The `.cuh` headers are self-contained drop-in replacements with minimal
dependencies — just include one and call `mul_cvt_bf16_to_fp4_4x_with_stochastic_rounding()`.

## Implementations

### `cvt.claude.cuh` — Hardware RN + Software SR Noise

Applies software stochastic rounding noise *before* the hardware round-to-nearest
conversion, so the deterministic RN instruction produces stochastic rounding
behavior:

1. Convert 4x BF16 to FP32 and multiply by scale (`mul.f32x2`)
2. For each value, inject symmetric noise in [-ULP/2, ULP/2) clamped to the ULP
   floor (`apply_sr_noise_e2m1`), so `P(round_up) = (|x| - |x_lo|) / ULP`
3. Pack with two `cvt.rn.satfinite.e2m1x2.f32` calls, combined into a 16-bit result

It can be built on `sm_120f, sm_120a, sm_121a`.

### `cvt.chatgpt.cuh` — Pure Software Quantization (with optional hardware path)

Performs the full quantization in software using explicit bracket lookup and
threshold comparison:

1. Convert 4x BF16 to FP32, multiply by scale
2. Find the E2M1 floor/ceil bracket for each value
3. Compute `p_up = (|x| - floor) / (ceil - floor)`, convert to a 32-bit threshold
4. Compare against per-lane random bits to decide round-up vs round-down
5. Pack four 4-bit nibbles into a 16-bit result

Also includes a gated hardware path (`ARCH_HAS_STOCHASTIC_ROUNDING=1`) using
the original `cvt.rs.satfinite.e2m1x4.f32` for architectures that support it.

It is pure software implementation and can be built on any CUDA architecture.

## E2M1 Format

4-bit floating point with 1 sign bit, 2 exponent bits, 1 mantissa bit.
Representable values: {0, 0.5, 1, 1.5, 2, 3, 4, 6} (and their negatives).

## Building

Requires CUDA toolkit with SM120 support (CUDA 12.8+).

```bash
make            # builds cvt.claude.exe and cvt.chatgpt.exe
make clean      # removes executables
```

Set `CUDA_ARCH` for a different CUDA architecture, and `CUDA_HOME` if
your CUDA installation is not at `/usr/local/cuda`:

```bash
make CUDA_HOME=/path/to/cuda CUDA_ARCH=121a
```

## Test Suites

### Claude (`cvt.claude.exe`)

| Test | Description |
|------|-------------|
| Test 0 | Bare `cvt_e2m1x4_rn` sanity check — verifies nibble packing order |
| Test 1 | SR probability — measured vs expected round-up probability for 16 test values |
| Test 2 | Unbiasedness — verifies `E[SR(x)] ~ x` over 2.56M trials per value |
| Test 3 | Exact representable values — SR of exact E2M1 values always returns itself |
| Test 4 | Saturation — values beyond \|6.0\| clamp to +/-6.0 |
| Test 5 | All 4 lanes — valid output and correct sign across all nibble positions |
| Test 6 | Throughput benchmark — SR kernel vs read-only baseline at 4/64/256 MB |

### ChatGPT (`cvt.chatgpt.exe`)

| Test | Description |
|------|-------------|
| Fixed examples | Spot-check specific input/scale/rbits combinations |
| Exact representables | All 15 exact E2M1 values round-trip correctly |
| Saturation / specials | Overflow, infinity, and NaN clamp to +/-6.0 |
| Scale semantics | Verifies even/odd lane scale factors (`scale.x` / `scale.y`) |
| CPU/GPU consistency | 20,000 random cases compared between CPU reference and GPU kernel |
| Probability suites | Round-up probability across 7 RNG streams at midpoint and non-midpoint |
| Throughput benchmark | Quantize kernel vs stream baseline with effective bandwidth reporting |

## License

Apache 2.0

