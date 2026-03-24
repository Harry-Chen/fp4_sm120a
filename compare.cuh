#pragma once
//
// Side-by-side comparison of:
//   - native  cvt.rs.satfinite.e2m1x4.f32 (SM100a hardware SR)
//   - polyfill from cvt.claude.cuh (software SR + cvt.rn.satfinite.e2m1x2.f32)
//
// Build with CUDA_ARCH=100a to run on B300.
//

#include <cstdint>

struct fp4e2m1x4 {
    uint16_t __x;
};

// ===================================================================
// NATIVE: uses cvt.rs.satfinite.e2m1x4.f32 (requires SM100a)
// ===================================================================

__device__ __forceinline__
fp4e2m1x4 native_mul_cvt_bf16_to_fp4_4x_sr(
    const uint64_t in_4x, const float2 scale, const uint32_t rbits) {
    uint16_t out_4x = 0;
    asm volatile(
        "{\n"
        ".reg.b64 v01; \n\t"
        ".reg.b64 v23; \n\t"
        ".reg.b16 v0_bf16; \n\t"
        ".reg.b16 v1_bf16; \n\t"
        ".reg.b16 v2_bf16; \n\t"
        ".reg.b16 v3_bf16; \n\t"
        ".reg.b32 v0; \n\t"
        ".reg.b32 v1; \n\t"
        ".reg.b32 v2; \n\t"
        ".reg.b32 v3; \n\t"
        "mov.b64 {v0_bf16, v1_bf16, v2_bf16, v3_bf16} , %1; \n\t"
        "cvt.f32.bf16 v0, v0_bf16; \n\t"
        "cvt.f32.bf16 v1, v1_bf16; \n\t"
        "cvt.f32.bf16 v2, v2_bf16; \n\t"
        "cvt.f32.bf16 v3, v3_bf16; \n\t"
        "mov.b64 v01, {v0, v1}; \n\t"
        "mov.b64 v23, {v2, v3}; \n\t"
        "mul.f32x2 v01, v01, %2; \n\t"
        "mul.f32x2 v23, v23, %2; \n\t"
        "mov.b64 {v1, v0}, v01; \n\t"
        "mov.b64 {v3, v2}, v23; \n\t"
        "cvt.rs.satfinite.e2m1x4.f32 %0, {v2, v3, v0, v1}, %3; \n\t"
        "}"
        : "=h"(out_4x)
        : "l"(in_4x), "l"(reinterpret_cast<const uint64_t &>(scale)), "r"(rbits));
    fp4e2m1x4 result;
    result.__x = out_4x;
    return result;
}

// ===================================================================
// POLYFILL: software SR noise + cvt.rn.satfinite.e2m1x2.f32
// (copied from cvt.claude.cuh to avoid type conflicts)
// ===================================================================

__device__ __forceinline__
float polyfill_apply_sr_noise_e2m1(float x, unsigned rand_byte) {
    unsigned u = __float_as_uint(x);
    unsigned abs_u = u & 0x7FFFFFFFu;
    unsigned exp = abs_u >> 23;

    unsigned ulp_bexp = min(max(exp, 127u), 129u) - 1u;
    float ulp = __uint_as_float(ulp_bexp << 23);

    float x_lo_abs;
    if (exp >= 127u) {
        x_lo_abs = __uint_as_float(abs_u & 0xFFC00000u);
    } else {
        x_lo_abs = (exp >= 126u) ? 0.5f : 0.0f;
    }

    float noise = (float)((int)rand_byte - 127.5f) * __uint_as_float(0x3B800000u) * ulp;
    float ax_noisy = fmaxf(fabsf(x) + noise, x_lo_abs);
    return copysignf(ax_noisy, x);
}

__device__ __forceinline__
unsigned short polyfill_cvt_e2m1x4_rn(float a, float b, float c, float d) {
    unsigned short result;
    asm volatile(
        "{\n\t"
        ".reg .b8  lo8, hi8;\n\t"
        ".reg .b32 lo32, hi32, packed32;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 lo8, %2, %1;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 hi8, %4, %3;\n\t"
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

__device__ __forceinline__
unsigned short polyfill_fp32x4_to_e2m1x4_sr(float a, float b, float c, float d,
                                              unsigned rbits) {
    a = polyfill_apply_sr_noise_e2m1(a, (rbits      ) & 0xFFu);
    b = polyfill_apply_sr_noise_e2m1(b, (rbits >>  8) & 0xFFu);
    c = polyfill_apply_sr_noise_e2m1(c, (rbits >> 16) & 0xFFu);
    d = polyfill_apply_sr_noise_e2m1(d, (rbits >> 24) & 0xFFu);
    return polyfill_cvt_e2m1x4_rn(a, b, c, d);
}

__device__ __forceinline__
fp4e2m1x4 polyfill_mul_cvt_bf16_to_fp4_4x_sr(
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

    fp4e2m1x4 result;
    result.__x = polyfill_fp32x4_to_e2m1x4_sr(v2, v3, v0, v1, rbits);
    return result;
}
