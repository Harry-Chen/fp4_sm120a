#pragma once

#include <stdint.h>
#include <math.h>
#include <string.h>

#ifndef ARCH_HAS_STOCHASTIC_ROUNDING
#define ARCH_HAS_STOCHASTIC_ROUNDING 0
#endif

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

static inline __host__ __device__ fp4e2m1x4 make_fp4e2m1x4(uint16_t x) {
  fp4e2m1x4 out;
  out.__x = x;
  return out;
}

static inline __host__ __device__ uint16_t raw_bits(fp4e2m1x4 x) {
  return x.__x;
}

static inline __host__ __device__ float fp4_e2m1_code_to_abs_float(uint8_t mag_code) {
  switch (mag_code & 0x7) {
    case 0: return 0.0f;
    case 1: return 0.5f;
    case 2: return 1.0f;
    case 3: return 1.5f;
    case 4: return 2.0f;
    case 5: return 3.0f;
    case 6: return 4.0f;
    default: return 6.0f;
  }
}

static inline __host__ __device__ uint8_t pack_fp4_e2m1(bool neg, uint8_t mag_code) {
  return static_cast<uint8_t>((neg ? 0x8 : 0x0) | (mag_code & 0x7));
}

static inline __host__ __device__ float bf16_bits_to_float(uint16_t x) {
  uint32_t u = static_cast<uint32_t>(x) << 16;
  float f;
#if defined(__CUDA_ARCH__)
  f = __uint_as_float(u);
#else
  memcpy(&f, &u, sizeof(f));
#endif
  return f;
}

static inline __host__ __device__ uint16_t get_bf16_lane(uint64_t packed, int lane) {
  return static_cast<uint16_t>((packed >> (16 * lane)) & 0xFFFFu);
}

static inline __host__ __device__ uint32_t mix_lane_bits(uint32_t rbits, int lane) {
  uint32_t x = rbits ^ (0x9E3779B9u * static_cast<uint32_t>(lane + 1));
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  x ^= x >> 16;
  return x;
}

static inline __host__ __device__ float decode_one_fp4_e2m1(uint8_t nibble) {
  const bool neg = (nibble & 0x8) != 0;
  const float v = fp4_e2m1_code_to_abs_float(nibble & 0x7);
  return neg ? -v : v;
}

static inline __host__ __device__ uint16_t pack_4_fp4_nibbles(uint8_t x0, uint8_t x1,
                                                               uint8_t x2, uint8_t x3) {
  return static_cast<uint16_t>((x0 & 0xF) |
                               ((x1 & 0xF) << 4) |
                               ((x2 & 0xF) << 8) |
                               ((x3 & 0xF) << 12));
}

static inline __host__ __device__ uint64_t prob_to_threshold_u32(float p_up) {
  if (!(p_up > 0.0f)) return 0ull;
  if (p_up >= 1.0f) return (1ull << 32);
  const double scaled = static_cast<double>(p_up) * 4294967296.0;
  uint64_t thr = static_cast<uint64_t>(scaled);
  if (thr > (1ull << 32)) thr = (1ull << 32);
  return thr;
}

static inline __host__ __device__ uint8_t quantize_fp32_to_fp4_e2m1_sr(float x, uint32_t rnd) {
  if (!isfinite(x)) {
    return pack_fp4_e2m1(signbit(x), 7);
  }

  const bool neg = signbit(x);
  const float ax = fabsf(x);

  if (ax >= 6.0f) {
    return pack_fp4_e2m1(neg, 7);
  }

  constexpr float kVals[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};

  for (uint8_t i = 0; i < 8; ++i) {
    if (ax == kVals[i]) {
      return pack_fp4_e2m1(neg, i);
    }
  }

  uint8_t hi = 1;
  while (hi < 8 && !(ax < kVals[hi])) {
    ++hi;
  }
  const uint8_t lo = static_cast<uint8_t>(hi - 1);

  const float vlo = kVals[lo];
  const float vhi = kVals[hi];
  const float p_up = (ax - vlo) / (vhi - vlo);

  const uint64_t threshold = prob_to_threshold_u32(p_up);
  const uint8_t chosen = (static_cast<uint64_t>(rnd) < threshold) ? hi : lo;
  return pack_fp4_e2m1(neg, chosen);
}

// ===================================================================
// Standalone FP32x4 -> FP4 nibbles with stochastic rounding
// ===================================================================

static inline __host__ __device__ uint16_t fp32x4_to_fp4_nibbles_sr(
    float v0, float v1, float v2, float v3, uint32_t rbits) {
  const uint8_t q0 = quantize_fp32_to_fp4_e2m1_sr(v0, mix_lane_bits(rbits, 0));
  const uint8_t q1 = quantize_fp32_to_fp4_e2m1_sr(v1, mix_lane_bits(rbits, 1));
  const uint8_t q2 = quantize_fp32_to_fp4_e2m1_sr(v2, mix_lane_bits(rbits, 2));
  const uint8_t q3 = quantize_fp32_to_fp4_e2m1_sr(v3, mix_lane_bits(rbits, 3));
  return pack_4_fp4_nibbles(q0, q1, q2, q3);
}

__device__ __forceinline__ fp4e2m1x4 mul_cvt_bf16_to_fp4_4x_with_stochastic_rounding(
    const uint64_t in_4x, const float2 scale, const uint32_t rbits) {
  uint16_t out_4x = 0;

  constexpr bool has_rs = ARCH_HAS_STOCHASTIC_ROUNDING;
  if constexpr (has_rs) {
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
        "mov.b64 {v0_bf16, v1_bf16, v2_bf16, v3_bf16}, %1; \n\t"
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
  } else {
    out_4x = fp32x4_to_fp4_nibbles_sr(
        bf16_bits_to_float(get_bf16_lane(in_4x, 0)) * scale.x,
        bf16_bits_to_float(get_bf16_lane(in_4x, 1)) * scale.y,
        bf16_bits_to_float(get_bf16_lane(in_4x, 2)) * scale.x,
        bf16_bits_to_float(get_bf16_lane(in_4x, 3)) * scale.y,
        rbits);
  }

  return make_fp4e2m1x4(out_4x);
}

// ===================================================================
// FP32 input variant with float2 scale
// ===================================================================

__device__ __forceinline__ fp4e2m1x4 mul_cvt_fp32_to_fp4_4x_with_stochastic_rounding(
    const float2 in01, const float2 in23, const float2 scale, const uint32_t rbits) {
  uint16_t out_4x = 0;

  constexpr bool has_rs = ARCH_HAS_STOCHASTIC_ROUNDING;
  if constexpr (has_rs) {
    asm volatile(
        "{\n"
        ".reg.b64 v01; \n\t"
        ".reg.b64 v23; \n\t"
        ".reg.b32 v0; \n\t"
        ".reg.b32 v1; \n\t"
        ".reg.b32 v2; \n\t"
        ".reg.b32 v3; \n\t"
        "mov.b64 {v0, v1} , %1; \n\t"
        "mov.b64 {v2, v3} , %2; \n\t"
        "mov.b64 v01, {v0, v1}; \n\t"
        "mov.b64 v23, {v2, v3}; \n\t"
        "mul.f32x2 v01, v01, %3; \n\t"
        "mul.f32x2 v23, v23, %3; \n\t"
        "mov.b64 {v1, v0}, v01; \n\t"
        "mov.b64 {v3, v2}, v23; \n\t"
        "cvt.rs.satfinite.e2m1x4.f32 %0, {v2, v3, v0, v1}, %4; \n\t"
        "}"
        : "=h"(out_4x)
        : "l"(reinterpret_cast<const uint64_t &>(in01)),
          "l"(reinterpret_cast<const uint64_t &>(in23)),
          "l"(reinterpret_cast<const uint64_t &>(scale)), "r"(rbits));
  } else {
    out_4x = fp32x4_to_fp4_nibbles_sr(
        in01.x * scale.x, in01.y * scale.y,
        in23.x * scale.x, in23.y * scale.y,
        rbits);
  }

  return make_fp4e2m1x4(out_4x);
}

// ===================================================================
// FP32 input, no scale
// ===================================================================

__device__ __forceinline__ fp4e2m1x4 cvt_fp32_to_fp4_4x_with_stochastic_rounding(
    const float2 in01, const float2 in23, const uint32_t rbits) {
  uint16_t out_4x = 0;

  constexpr bool has_rs = ARCH_HAS_STOCHASTIC_ROUNDING;
  if constexpr (has_rs) {
    asm volatile(
        "{\n"
        "cvt.rs.satfinite.e2m1x4.f32 %0, {%3, %4, %1, %2}, %5; \n\t"
        "}"
        : "=h"(out_4x)
        : "f"(in01.y), "f"(in01.x), "f"(in23.y), "f"(in23.x), "r"(rbits));
  } else {
    out_4x = fp32x4_to_fp4_nibbles_sr(
        in01.x, in01.y, in23.x, in23.y, rbits);
  }

  return make_fp4e2m1x4(out_4x);
}

// ===================================================================
// 8x BF16->FP4 with stochastic rounding and scalar scale
// ===================================================================

template <typename ScaleT>
__device__ __forceinline__ uint32_t mul_cvt_bf16_to_fp4_8x_stochastic_rounding(
    const uint64_t in03, const uint64_t in47, const ScaleT scaling_coefficient,
    const uint32_t rbits03, const uint32_t rbits47) {
  uint32_t out_8x = 0;

  constexpr bool has_rs = ARCH_HAS_STOCHASTIC_ROUNDING;
  if constexpr (has_rs) {
    // Use scalar float path for the asm (works for both float and bf16 scale
    // since we static_cast to float in the else branch; the asm path is only
    // compiled on architectures with cvt.rs.satfinite.e2m1x4)
    const float scale_f = static_cast<float>(scaling_coefficient);
    asm volatile(
        "{\n"
        ".reg.b16 v0_bf16, v1_bf16, v2_bf16, v3_bf16, v4_bf16, v5_bf16, v6_bf16, v7_bf16; \n\t"
        "mov.b64 {v0_bf16, v1_bf16, v2_bf16, v3_bf16}, %1; \n\t"
        "mov.b64 {v4_bf16, v5_bf16, v6_bf16, v7_bf16}, %2; \n\t"
        ".reg.b32 v0, v1, v2, v3, v4, v5, v6, v7; \n\t"
        "cvt.f32.bf16 v0, v0_bf16; \n\t"
        "cvt.f32.bf16 v1, v1_bf16; \n\t"
        "cvt.f32.bf16 v2, v2_bf16; \n\t"
        "cvt.f32.bf16 v3, v3_bf16; \n\t"
        "cvt.f32.bf16 v4, v4_bf16; \n\t"
        "cvt.f32.bf16 v5, v5_bf16; \n\t"
        "cvt.f32.bf16 v6, v6_bf16; \n\t"
        "cvt.f32.bf16 v7, v7_bf16; \n\t"
        "mul.f32 v0, v0, %3; \n\t"
        "mul.f32 v1, v1, %3; \n\t"
        "mul.f32 v2, v2, %3; \n\t"
        "mul.f32 v3, v3, %3; \n\t"
        "mul.f32 v4, v4, %3; \n\t"
        "mul.f32 v5, v5, %3; \n\t"
        "mul.f32 v6, v6, %3; \n\t"
        "mul.f32 v7, v7, %3; \n\t"
        ".reg.b16 b03, b47; \n\t"
        "cvt.rs.satfinite.e2m1x4.f32 b03, {v3, v2, v1, v0}, %4; \n\t"
        "cvt.rs.satfinite.e2m1x4.f32 b47, {v7, v6, v5, v4}, %5; \n\t"
        "mov.b32 %0, {b03, b47};\n"
        "}"
        : "=r"(out_8x)
        : "l"(in03), "l"(in47), "f"(scale_f), "r"(rbits03), "r"(rbits47));
  } else {
    const float scale_f = static_cast<float>(scaling_coefficient);

    const uint16_t lo = fp32x4_to_fp4_nibbles_sr(
        bf16_bits_to_float(get_bf16_lane(in03, 0)) * scale_f,
        bf16_bits_to_float(get_bf16_lane(in03, 1)) * scale_f,
        bf16_bits_to_float(get_bf16_lane(in03, 2)) * scale_f,
        bf16_bits_to_float(get_bf16_lane(in03, 3)) * scale_f,
        rbits03);
    const uint16_t hi = fp32x4_to_fp4_nibbles_sr(
        bf16_bits_to_float(get_bf16_lane(in47, 0)) * scale_f,
        bf16_bits_to_float(get_bf16_lane(in47, 1)) * scale_f,
        bf16_bits_to_float(get_bf16_lane(in47, 2)) * scale_f,
        bf16_bits_to_float(get_bf16_lane(in47, 3)) * scale_f,
        rbits47);
    out_8x = (uint32_t)lo | ((uint32_t)hi << 16);
  }

  return out_8x;
}
