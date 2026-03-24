#pragma once

#include <cstdint>

// ===================================================================
// FP4 E2M1 types — use NVIDIA native types when available
// ===================================================================
#if FP4_TYPE_SUPPORTED
#include <cuda_fp4.h>
using fp4e2m1   = __nv_fp4_e2m1;
using fp4e2m1x2 = __nv_fp4x2_e2m1;
using fp4e2m1x4 = __nv_fp4x4_e2m1;
#else
struct fp4e2m1x4 {
    uint16_t __x;
};
#endif

// ===================================================================
// E2M1 SR noise injection
// ===================================================================
//
// Algorithm: add symmetric noise in [-ULP/2, ULP/2) then clamp to
// x_lo so we stay in the correct ULP bracket. Hardware RN then
// yields P(round_up) = (|x| - |x_lo|) / ULP, i.e. stochastic rounding.

__device__ __forceinline__
float apply_sr_noise_e2m1(float x, unsigned rand_byte) {
    unsigned u = __float_as_uint(x);
    unsigned abs_u = u & 0x7FFFFFFFu;
    unsigned exp = abs_u >> 23;

    // E2M1 ULP: 0.5 for exp<128, 1.0 for exp==128, 2.0 for exp>=129
    unsigned ulp_bexp = min(max(exp, 127u), 129u) - 1u;
    float ulp = __uint_as_float(ulp_bexp << 23);

    // E2M1 truncation of |x| (floor to nearest representable)
    float x_lo_abs;
    if (exp >= 127u) {
        // Normal E2M1: mask off all but 1 mantissa bit
        x_lo_abs = __uint_as_float(abs_u & 0xFFC00000u);
    } else {
        x_lo_abs = (exp >= 126u) ? 0.5f : 0.0f;
    }

    // Symmetric noise in [-ULP/2, ULP/2)
    // (rand_byte - 127.5f) / 256 * ULP
    // rand_byte is {-128, ..., 127}, avg is -0.5
    float noise = (float)((int)rand_byte - 127.5f) * __uint_as_float(0x3B800000u) * ulp;

    // Clamp: prevent noise from pushing below x_lo (ULP boundary crossing)
    float ax_noisy = fmaxf(fabsf(x) + noise, x_lo_abs);

    return copysignf(ax_noisy, x);
}

// ===================================================================
// e2m1x2 wrapper: pack two cvt.rn.satfinite.e2m1x2 into 16 bits
// ===================================================================
//
// Hardware packing of cvt.rn.satfinite.e2m1x2.f32 r8, src1, src2:
//   r8[7:4] = e2m1(src1)    <- high nibble
//   r8[3:0] = e2m1(src2)    <- low nibble
//
// Original e2m1x4 layout: cvt.*.e2m1x4 %0, {a, b, c, d}
//   bits[3:0]=a, [7:4]=b, [11:8]=c, [15:12]=d
//
// To reproduce with two e2m1x2 calls:
//   lo8 = cvt(b, a)  ->  lo8[3:0]=a, lo8[7:4]=b
//   hi8 = cvt(d, c)  ->  hi8[3:0]=c, hi8[7:4]=d
//   result = lo8 | (hi8 << 8)

__device__ __forceinline__
unsigned short cvt_e2m1x4_rn(float a, float b, float c, float d) {
    unsigned short result;
    asm volatile(
        "{\n\t"
        ".reg .b8  lo8, hi8;\n\t"
        ".reg .b32 lo32, hi32, packed32;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 lo8, %2, %1;\n\t"  // (b, a) -> a=low, b=high
        "cvt.rn.satfinite.e2m1x2.f32 hi8, %4, %3;\n\t"  // (d, c) -> c=low, d=high
        "cvt.u32.u8 lo32, lo8;\n\t"
        "cvt.u32.u8 hi32, hi8;\n\t"
        "shl.b32 hi32, hi32, 8;\n\t"
        "or.b32  packed32, lo32, hi32;\n\t"
        "cvt.u16.u32 %0, packed32;\n\t"
        "}"
        : "=h"(result)
        : "f"(a), "f"(b), "f"(c), "f"(d)
    );
    return result;
}

// ===================================================================
// Main conversion: drop-in replacement
// ===================================================================

__device__ __forceinline__
fp4e2m1x4 mul_cvt_bf16_to_fp4_4x_with_stochastic_rounding(
    const uint64_t in_4x, const float2 scale, const uint32_t rbits) {

    float v0, v1, v2, v3;
    asm volatile(
        "{\n\t"
        ".reg .b16  b0, b1, b2, b3;\n\t"
        ".reg .b64  p01, p23;\n\t"
        "mov.b64 {b0, b1, b2, b3}, %4;\n\t"
        "cvt.f32.bf16 %0, b0;\n\t"
        "cvt.f32.bf16 %1, b1;\n\t"
        "cvt.f32.bf16 %2, b2;\n\t"
        "cvt.f32.bf16 %3, b3;\n\t"
        "mov.b64 p01, {%0, %1};\n\t"
        "mov.b64 p23, {%2, %3};\n\t"
        "mul.f32x2 p01, p01, %5;\n\t"
        "mul.f32x2 p23, p23, %5;\n\t"
        "mov.b64 {%1, %0}, p01;\n\t"
        "mov.b64 {%3, %2}, p23;\n\t"
        "}"
        : "=f"(v0), "=f"(v1), "=f"(v2), "=f"(v3)
        : "l"(in_4x), "l"(reinterpret_cast<const uint64_t &>(scale))
    );

    // Original: cvt.rs.satfinite.e2m1x4.f32 %0, {v2, v3, v0, v1}, rbits
    // Layout:   nibble0=v2, nibble1=v3, nibble2=v0, nibble3=v1
    fp4e2m1x4 result;
    result.__x = fp32x4_to_e2m1x4_sr(v2, v3, v0, v1, rbits);
    return result;
}

// ===================================================================
// Standalone FP32x4 -> E2M1x4 SR (convenience wrapper)
// ===================================================================

__device__ __forceinline__
unsigned short fp32x4_to_e2m1x4_sr(float a, float b, float c, float d,
                                     unsigned rbits) {
    a = apply_sr_noise_e2m1(a, (rbits      ) & 0xFFu);
    b = apply_sr_noise_e2m1(b, (rbits >>  8) & 0xFFu);
    c = apply_sr_noise_e2m1(c, (rbits >> 16) & 0xFFu);
    d = apply_sr_noise_e2m1(d, (rbits >> 24) & 0xFFu);
    return cvt_e2m1x4_rn(a, b, c, d);
}

// ===================================================================
// FP32 input variant: drop-in replacement for
// cvt.rs.satfinite.e2m1x4.f32 with float2 inputs and float2 scale
// ===================================================================
//
// Original asm mapping for (in01, in23, scale):
//   mul.f32x2 then swap on readback, then pack {v2, v3, v0, v1}
//   Same shuffle as the BF16 variant, just without BF16->FP32 conversion.

__device__ __forceinline__
fp4e2m1x4 mul_cvt_fp32_to_fp4_4x_with_stochastic_rounding(
    const float2 in01, const float2 in23, const float2 scale, const uint32_t rbits) {

    float v0, v1, v2, v3;
    asm volatile(
        "{\n\t"
        ".reg .b64  p01, p23;\n\t"
        "mov.b64 p01, %4;\n\t"
        "mov.b64 p23, %5;\n\t"
        "mul.f32x2 p01, p01, %6;\n\t"
        "mul.f32x2 p23, p23, %6;\n\t"
        "mov.b64 {%1, %0}, p01;\n\t"
        "mov.b64 {%3, %2}, p23;\n\t"
        "}"
        : "=f"(v0), "=f"(v1), "=f"(v2), "=f"(v3)
        : "l"(reinterpret_cast<const uint64_t &>(in01)),
          "l"(reinterpret_cast<const uint64_t &>(in23)),
          "l"(reinterpret_cast<const uint64_t &>(scale))
    );

    fp4e2m1x4 result;
    result.__x = fp32x4_to_e2m1x4_sr(v2, v3, v0, v1, rbits);
    return result;
}

// ===================================================================
// 8x BF16->FP4 with stochastic rounding and scalar scale
// ===================================================================
//
// Original asm packs {v3, v2, v1, v0} for each group of 4:
//   b03: nibble0=v3, nibble1=v2, nibble2=v1, nibble3=v0
//   b47: nibble0=v7, nibble1=v6, nibble2=v5, nibble3=v4
//   result = b03 | (b47 << 16)

template <typename ScaleT>
__device__ __forceinline__ uint32_t mul_cvt_bf16_to_fp4_8x_stochastic_rounding(
    const uint64_t in03, const uint64_t in47, const ScaleT scaling_coefficient,
    const uint32_t rbits03, const uint32_t rbits47) {

    float scale_f = static_cast<float>(scaling_coefficient);

    float v0, v1, v2, v3, v4, v5, v6, v7;
    asm volatile(
        "{\n\t"
        ".reg .b16 b0, b1, b2, b3, b4, b5, b6, b7;\n\t"
        "mov.b64 {b0, b1, b2, b3}, %8;\n\t"
        "mov.b64 {b4, b5, b6, b7}, %9;\n\t"
        "cvt.f32.bf16 %0, b0;\n\t"
        "cvt.f32.bf16 %1, b1;\n\t"
        "cvt.f32.bf16 %2, b2;\n\t"
        "cvt.f32.bf16 %3, b3;\n\t"
        "cvt.f32.bf16 %4, b4;\n\t"
        "cvt.f32.bf16 %5, b5;\n\t"
        "cvt.f32.bf16 %6, b6;\n\t"
        "cvt.f32.bf16 %7, b7;\n\t"
        "}"
        : "=f"(v0), "=f"(v1), "=f"(v2), "=f"(v3),
          "=f"(v4), "=f"(v5), "=f"(v6), "=f"(v7)
        : "l"(in03), "l"(in47)
    );

    v0 *= scale_f; v1 *= scale_f; v2 *= scale_f; v3 *= scale_f;
    v4 *= scale_f; v5 *= scale_f; v6 *= scale_f; v7 *= scale_f;

    // SR noise + pack matching {v3, v2, v1, v0} element order
    unsigned short b03 = fp32x4_to_e2m1x4_sr(v3, v2, v1, v0, rbits03);
    unsigned short b47 = fp32x4_to_e2m1x4_sr(v7, v6, v5, v4, rbits47);

    return (uint32_t)b03 | ((uint32_t)b47 << 16);
}
