#pragma once
//
// Side-by-side comparison of native hardware vs polyfills.
// Build with CUDA_ARCH=100a to run on B300.
//
// Contains three implementations of each function:
//   native_*    — uses cvt.rs.satfinite.e2m1x4.f32 (SM100a hardware SR)
//   claude_*    — software SR noise + cvt.rn.satfinite.e2m1x2.f32
//   chatgpt_*   — pure software quantization (no special PTX needed)
//

#include <cstdint>
#include <cmath>

struct fp4e2m1x4 {
    uint16_t __x;
};

// ===================================================================
// NATIVE implementations (requires SM100a)
// ===================================================================

__device__ __forceinline__
fp4e2m1x4 native_mul_cvt_bf16_to_fp4_4x_sr(
    const uint64_t in_4x, const float2 scale, const uint32_t rbits) {
    uint16_t out_4x = 0;
    asm volatile(
        "{\n"
        ".reg.b64 v01; \n\t"
        ".reg.b64 v23; \n\t"
        ".reg.b16 v0_bf16, v1_bf16, v2_bf16, v3_bf16; \n\t"
        ".reg.b32 v0, v1, v2, v3; \n\t"
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
    fp4e2m1x4 r; r.__x = out_4x; return r;
}

__device__ __forceinline__
fp4e2m1x4 native_cvt_fp32_to_fp4_4x_sr(
    const float2 in01, const float2 in23, const uint32_t rbits) {
    uint16_t out_4x;
    asm volatile(
        "{\n"
        "cvt.rs.satfinite.e2m1x4.f32 %0, {%3, %4, %1, %2}, %5; \n\t"
        "}"
        : "=h"(out_4x)
        : "f"(in01.y), "f"(in01.x), "f"(in23.y), "f"(in23.x), "r"(rbits));
    fp4e2m1x4 r; r.__x = out_4x; return r;
}

// ===================================================================
// CLAUDE polyfill: software SR noise + cvt.rn.satfinite.e2m1x2.f32
// ===================================================================

__device__ __forceinline__
float claude_apply_sr_noise_e2m1(float x, unsigned rand_byte) {
    unsigned u = __float_as_uint(x);
    unsigned abs_u = u & 0x7FFFFFFFu;
    unsigned exp = abs_u >> 23;
    unsigned ulp_bexp = min(max(exp, 127u), 129u) - 1u;
    float ulp = __uint_as_float(ulp_bexp << 23);
    float x_lo_abs;
    if (exp >= 127u)
        x_lo_abs = __uint_as_float(abs_u & 0xFFC00000u);
    else
        x_lo_abs = (exp >= 126u) ? 0.5f : 0.0f;
    float noise = (float)((int)rand_byte - 127.5f) * __uint_as_float(0x3B800000u) * ulp;
    float ax_noisy = fmaxf(fabsf(x) + noise, x_lo_abs);
    return copysignf(ax_noisy, x);
}

__device__ __forceinline__
unsigned short claude_cvt_e2m1x4_rn(float a, float b, float c, float d) {
    // Hardware e2m1x4 layout: {a,b,c,d} → bits[3:0]=d, [7:4]=c, [11:8]=b, [15:12]=a
    unsigned short result;
    asm volatile(
        "{\n\t"
        ".reg .b8  lo8, hi8;\n\t"
        ".reg .b32 lo32, hi32, packed32;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 lo8, %3, %4;\n\t"  // (c, d) -> d=low, c=high
        "cvt.rn.satfinite.e2m1x2.f32 hi8, %1, %2;\n\t"  // (a, b) -> b=low, a=high
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
unsigned short claude_fp32x4_to_e2m1x4_sr(float a, float b, float c, float d,
                                            unsigned rbits) {
    a = claude_apply_sr_noise_e2m1(a, (rbits      ) & 0xFFu);
    b = claude_apply_sr_noise_e2m1(b, (rbits >>  8) & 0xFFu);
    c = claude_apply_sr_noise_e2m1(c, (rbits >> 16) & 0xFFu);
    d = claude_apply_sr_noise_e2m1(d, (rbits >> 24) & 0xFFu);
    return claude_cvt_e2m1x4_rn(a, b, c, d);
}

__device__ __forceinline__
fp4e2m1x4 claude_mul_cvt_bf16_to_fp4_4x_sr(
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
    fp4e2m1x4 r;
    r.__x = claude_fp32x4_to_e2m1x4_sr(v2, v3, v0, v1, rbits);
    return r;
}

__device__ __forceinline__
fp4e2m1x4 claude_cvt_fp32_to_fp4_4x_sr(
    const float2 in01, const float2 in23, const uint32_t rbits) {
    fp4e2m1x4 r;
    r.__x = claude_fp32x4_to_e2m1x4_sr(in23.y, in23.x, in01.y, in01.x, rbits);
    return r;
}

// ===================================================================
// CHATGPT polyfill: pure software quantization
// ===================================================================

__device__ __forceinline__
uint32_t chatgpt_mix_lane_bits(uint32_t rbits, int lane) {
    uint32_t x = rbits ^ (0x9E3779B9u * (uint32_t)(lane + 1));
    x ^= x >> 16; x *= 0x7FEB352Du;
    x ^= x >> 15; x *= 0x846CA68Bu;
    x ^= x >> 16;
    return x;
}

__device__ __forceinline__
uint8_t chatgpt_pack_fp4_e2m1(bool neg, uint8_t mag_code) {
    return (uint8_t)((neg ? 0x8 : 0x0) | (mag_code & 0x7));
}

__device__ __forceinline__
float chatgpt_fp4_e2m1_code_to_abs_float(uint8_t mag_code) {
    const float t[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    return t[mag_code & 0x7];
}

__device__ __forceinline__
uint64_t chatgpt_prob_to_threshold_u32(float p_up) {
    if (!(p_up > 0.0f)) return 0ull;
    if (p_up >= 1.0f) return (1ull << 32);
    uint64_t thr = (uint64_t)((double)p_up * 4294967296.0);
    if (thr > (1ull << 32)) thr = (1ull << 32);
    return thr;
}

__device__ __forceinline__
uint8_t chatgpt_quantize_fp32_to_fp4_e2m1_sr(float x, uint32_t rnd) {
    if (!isfinite(x))
        return chatgpt_pack_fp4_e2m1(signbit(x), 7);
    bool neg = signbit(x);
    float ax = fabsf(x);
    if (ax >= 6.0f)
        return chatgpt_pack_fp4_e2m1(neg, 7);
    const float kVals[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    for (uint8_t i = 0; i < 8; ++i)
        if (ax == kVals[i]) return chatgpt_pack_fp4_e2m1(neg, i);
    uint8_t hi = 1;
    while (hi < 8 && !(ax < kVals[hi])) ++hi;
    uint8_t lo = hi - 1;
    float p_up = (ax - kVals[lo]) / (kVals[hi] - kVals[lo]);
    uint64_t threshold = chatgpt_prob_to_threshold_u32(p_up);
    uint8_t chosen = ((uint64_t)rnd < threshold) ? hi : lo;
    return chatgpt_pack_fp4_e2m1(neg, chosen);
}

__device__ __forceinline__
uint16_t chatgpt_fp32x4_to_fp4_nibbles_sr(
    float v0, float v1, float v2, float v3, uint32_t rbits) {
    uint8_t q0 = chatgpt_quantize_fp32_to_fp4_e2m1_sr(v0, chatgpt_mix_lane_bits(rbits, 0));
    uint8_t q1 = chatgpt_quantize_fp32_to_fp4_e2m1_sr(v1, chatgpt_mix_lane_bits(rbits, 1));
    uint8_t q2 = chatgpt_quantize_fp32_to_fp4_e2m1_sr(v2, chatgpt_mix_lane_bits(rbits, 2));
    uint8_t q3 = chatgpt_quantize_fp32_to_fp4_e2m1_sr(v3, chatgpt_mix_lane_bits(rbits, 3));
    return (uint16_t)((q0 & 0xF) | ((q1 & 0xF) << 4) | ((q2 & 0xF) << 8) | ((q3 & 0xF) << 12));
}

__device__ __forceinline__
fp4e2m1x4 chatgpt_mul_cvt_bf16_to_fp4_4x_sr(
    const uint64_t in_4x, const float2 scale, const uint32_t rbits) {
    float v0 = __uint_as_float((uint32_t)(uint16_t)(in_4x      ) << 16) * scale.x;
    float v1 = __uint_as_float((uint32_t)(uint16_t)(in_4x >> 16) << 16) * scale.y;
    float v2 = __uint_as_float((uint32_t)(uint16_t)(in_4x >> 32) << 16) * scale.x;
    float v3 = __uint_as_float((uint32_t)(uint16_t)(in_4x >> 48) << 16) * scale.y;
    fp4e2m1x4 r;
    r.__x = chatgpt_fp32x4_to_fp4_nibbles_sr(v0, v1, v2, v3, rbits);
    return r;
}

__device__ __forceinline__
fp4e2m1x4 chatgpt_cvt_fp32_to_fp4_4x_sr(
    const float2 in01, const float2 in23, const uint32_t rbits) {
    fp4e2m1x4 r;
    r.__x = chatgpt_fp32x4_to_fp4_nibbles_sr(in01.x, in01.y, in23.x, in23.y, rbits);
    return r;
}
