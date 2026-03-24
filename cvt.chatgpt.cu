#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <algorithm>
#include <limits>
#include <random>
#include <vector>

#include "cvt.chatgpt.cuh"

static uint16_t float_to_bf16_rne_bits(float x) {
  uint32_t u;
  memcpy(&u, &x, sizeof(u));
  const uint32_t lsb = (u >> 16) & 1u;
  const uint32_t bias = 0x7FFFu + lsb;
  u += bias;
  return static_cast<uint16_t>(u >> 16);
}

static uint64_t pack_4_bf16(float a, float b, float c, float d) {
  const uint64_t x0 = float_to_bf16_rne_bits(a);
  const uint64_t x1 = float_to_bf16_rne_bits(b);
  const uint64_t x2 = float_to_bf16_rne_bits(c);
  const uint64_t x3 = float_to_bf16_rne_bits(d);
  return x0 | (x1 << 16) | (x2 << 32) | (x3 << 48);
}

static uint16_t cpu_reference(uint64_t in_4x, float2 scale, uint32_t rbits) {
  const float v0 = bf16_bits_to_float(get_bf16_lane(in_4x, 0)) * scale.x;
  const float v1 = bf16_bits_to_float(get_bf16_lane(in_4x, 1)) * scale.y;
  const float v2 = bf16_bits_to_float(get_bf16_lane(in_4x, 2)) * scale.x;
  const float v3 = bf16_bits_to_float(get_bf16_lane(in_4x, 3)) * scale.y;

  const uint8_t q0 = quantize_fp32_to_fp4_e2m1_sr(v0, mix_lane_bits(rbits, 0));
  const uint8_t q1 = quantize_fp32_to_fp4_e2m1_sr(v1, mix_lane_bits(rbits, 1));
  const uint8_t q2 = quantize_fp32_to_fp4_e2m1_sr(v2, mix_lane_bits(rbits, 2));
  const uint8_t q3 = quantize_fp32_to_fp4_e2m1_sr(v3, mix_lane_bits(rbits, 3));

  return pack_4_fp4_nibbles(q0, q1, q2, q3);
}

static uint16_t cpu_reference_fp32(float2 in01, float2 in23, float2 scale, uint32_t rbits) {
  const float v0 = in01.x * scale.x;
  const float v1 = in01.y * scale.y;
  const float v2 = in23.x * scale.x;
  const float v3 = in23.y * scale.y;

  const uint8_t q0 = quantize_fp32_to_fp4_e2m1_sr(v0, mix_lane_bits(rbits, 0));
  const uint8_t q1 = quantize_fp32_to_fp4_e2m1_sr(v1, mix_lane_bits(rbits, 1));
  const uint8_t q2 = quantize_fp32_to_fp4_e2m1_sr(v2, mix_lane_bits(rbits, 2));
  const uint8_t q3 = quantize_fp32_to_fp4_e2m1_sr(v3, mix_lane_bits(rbits, 3));

  return pack_4_fp4_nibbles(q0, q1, q2, q3);
}

static uint32_t cpu_reference_8x(uint64_t in03, uint64_t in47, float scale,
                                  uint32_t rbits03, uint32_t rbits47) {
  const float v0 = bf16_bits_to_float(get_bf16_lane(in03, 0)) * scale;
  const float v1 = bf16_bits_to_float(get_bf16_lane(in03, 1)) * scale;
  const float v2 = bf16_bits_to_float(get_bf16_lane(in03, 2)) * scale;
  const float v3 = bf16_bits_to_float(get_bf16_lane(in03, 3)) * scale;
  const float v4 = bf16_bits_to_float(get_bf16_lane(in47, 0)) * scale;
  const float v5 = bf16_bits_to_float(get_bf16_lane(in47, 1)) * scale;
  const float v6 = bf16_bits_to_float(get_bf16_lane(in47, 2)) * scale;
  const float v7 = bf16_bits_to_float(get_bf16_lane(in47, 3)) * scale;

  const uint8_t q0 = quantize_fp32_to_fp4_e2m1_sr(v0, mix_lane_bits(rbits03, 0));
  const uint8_t q1 = quantize_fp32_to_fp4_e2m1_sr(v1, mix_lane_bits(rbits03, 1));
  const uint8_t q2 = quantize_fp32_to_fp4_e2m1_sr(v2, mix_lane_bits(rbits03, 2));
  const uint8_t q3 = quantize_fp32_to_fp4_e2m1_sr(v3, mix_lane_bits(rbits03, 3));
  const uint8_t q4 = quantize_fp32_to_fp4_e2m1_sr(v4, mix_lane_bits(rbits47, 0));
  const uint8_t q5 = quantize_fp32_to_fp4_e2m1_sr(v5, mix_lane_bits(rbits47, 1));
  const uint8_t q6 = quantize_fp32_to_fp4_e2m1_sr(v6, mix_lane_bits(rbits47, 2));
  const uint8_t q7 = quantize_fp32_to_fp4_e2m1_sr(v7, mix_lane_bits(rbits47, 3));

  const uint16_t lo = pack_4_fp4_nibbles(q0, q1, q2, q3);
  const uint16_t hi = pack_4_fp4_nibbles(q4, q5, q6, q7);
  return (uint32_t)lo | ((uint32_t)hi << 16);
}

static void print_fp4x4(uint16_t bits) {
  for (int i = 0; i < 4; ++i) {
    const uint8_t nib = (bits >> (4 * i)) & 0xF;
    printf("lane%d: nibble=0x%X val=%g\n", i, nib, decode_one_fp4_e2m1(nib));
  }
}

__global__ void test_kernel(const uint64_t* __restrict__ in, const float2 scale,
                            const uint32_t* __restrict__ rbits,
                            uint16_t* __restrict__ out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = raw_bits(mul_cvt_bf16_to_fp4_4x_with_stochastic_rounding(in[idx], scale, rbits[idx]));
  }
}

__global__ void test_fp32_kernel(const float2* __restrict__ in01, const float2* __restrict__ in23,
                                const float2 scale, const uint32_t* __restrict__ rbits,
                                uint16_t* __restrict__ out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = raw_bits(mul_cvt_fp32_to_fp4_4x_with_stochastic_rounding(
        in01[idx], in23[idx], scale, rbits[idx]));
  }
}

__global__ void test_8x_kernel(const uint64_t* __restrict__ in03, const uint64_t* __restrict__ in47,
                               const float scale, const uint32_t* __restrict__ rbits03,
                               const uint32_t* __restrict__ rbits47,
                               uint32_t* __restrict__ out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    out[idx] = mul_cvt_bf16_to_fp4_8x_stochastic_rounding(
        in03[idx], in47[idx], scale, rbits03[idx], rbits47[idx]);
  }
}

__global__ void stream_baseline_kernel(const uint64_t* __restrict__ in,
                                       const uint32_t* __restrict__ rbits,
                                       uint16_t* __restrict__ out, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    const uint64_t x = in[idx];
    const uint32_t r = rbits[idx];
    const uint16_t y = static_cast<uint16_t>(((x >> 11) ^ (x >> 37) ^ r) & 0xFFFFu);
    out[idx] = y;
  }
}

static void check_cuda(cudaError_t err, const char* where) {
  if (err != cudaSuccess) {
    fprintf(stderr, "CUDA error at %s: %s\n", where, cudaGetErrorString(err));
    exit(1);
  }
}

static void test_fixed_examples() {
  printf("=== fixed examples ===\n");
  {
    const uint64_t in = pack_4_bf16(0.25f, 1.25f, -2.7f, 10.0f);
    const float2 scale = make_float2(2.0f, 0.5f);
    const uint32_t rbits = 0x12345678u;
    const uint16_t ref = cpu_reference(in, scale, rbits);
    printf("sample A bits = 0x%04X\n", ref);
    print_fp4x4(ref);
  }
  {
    const uint64_t in = pack_4_bf16(-0.0f, 0.5f, 6.0f, -INFINITY);
    const float2 scale = make_float2(1.0f, 1.0f);
    const uint32_t rbits = 0xCAFEBABEu;
    const uint16_t ref = cpu_reference(in, scale, rbits);
    printf("sample B bits = 0x%04X\n", ref);
    print_fp4x4(ref);
  }
  printf("\n");
}

static void test_exact_representables() {
  printf("=== exact representables ===\n");
  const float reps[] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                        -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};
  for (float x : reps) {
    const uint64_t in = pack_4_bf16(x, x, x, x);
    const uint16_t bits = cpu_reference(in, make_float2(1.0f, 1.0f), 0x13579BDFu);
    const uint8_t nib = bits & 0xF;
    const float y = decode_one_fp4_e2m1(nib);
    printf("x=%6g -> nibble=0x%X decoded=%6g\n", x, nib, y);
    if (y != x) {
      fprintf(stderr, "exact-representable mismatch: x=%g decoded=%g\n", x, y);
      exit(1);
    }
  }
  printf("\n");
}

static void test_saturation_and_specials() {
  printf("=== saturation / specials ===\n");
  struct Case { float x; float expected; } cases[] = {
      {7.0f, 6.0f}, {-7.0f, -6.0f}, {INFINITY, 6.0f}, {-INFINITY, -6.0f},
      {1000.0f, 6.0f}, {-1000.0f, -6.0f}};

  for (const auto& c : cases) {
    const uint64_t in = pack_4_bf16(c.x, c.x, c.x, c.x);
    const uint16_t bits = cpu_reference(in, make_float2(1.0f, 1.0f), 0x2468ACE0u);
    const float y = decode_one_fp4_e2m1(bits & 0xF);
    printf("x=%8g -> decoded=%4g\n", c.x, y);
    if (y != c.expected) {
      fprintf(stderr, "saturation mismatch: x=%g expected=%g got=%g\n", c.x, c.expected, y);
      exit(1);
    }
  }

  const uint64_t in_nan = pack_4_bf16(NAN, NAN, NAN, NAN);
  const uint16_t bits_nan = cpu_reference(in_nan, make_float2(1.0f, 1.0f), 0x10203040u);
  const float y_nan = decode_one_fp4_e2m1(bits_nan & 0xF);
  printf("x=NaN -> decoded=%g (policy: saturate finite max preserving signbit)\n\n", y_nan);
}

static void test_scale_semantics() {
  printf("=== scale semantics ===\n");
  const uint64_t in = pack_4_bf16(1.0f, 1.0f, 1.0f, 1.0f);
  const float2 scale = make_float2(2.0f, 0.5f);
  const uint16_t bits = cpu_reference(in, scale, 0xABCDEF01u);
  print_fp4x4(bits);

  const float y0 = decode_one_fp4_e2m1((bits >> 0) & 0xF);
  const float y1 = decode_one_fp4_e2m1((bits >> 4) & 0xF);
  const float y2 = decode_one_fp4_e2m1((bits >> 8) & 0xF);
  const float y3 = decode_one_fp4_e2m1((bits >> 12) & 0xF);

  if (!(y0 == 2.0f && y1 == 0.5f && y2 == 2.0f && y3 == 0.5f)) {
    fprintf(stderr, "scale semantics mismatch\n");
    exit(1);
  }
  printf("\n");
}

static void test_cpu_gpu_consistency() {
  printf("=== CPU/GPU consistency ===\n");
  constexpr int N = 20000;
  std::mt19937 rng(12345);
  std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
  std::uniform_int_distribution<uint32_t> dist_u32(0u, 0xFFFFFFFFu);

  std::vector<uint64_t> h_in(N);
  std::vector<uint32_t> h_rbits(N);
  std::vector<uint16_t> h_out(N), h_ref(N);

  const float2 scale = make_float2(0.75f, 1.25f);

  for (int i = 0; i < N; ++i) {
    h_in[i] = pack_4_bf16(dist(rng), dist(rng), dist(rng), dist(rng));
    h_rbits[i] = dist_u32(rng);
    h_ref[i] = cpu_reference(h_in[i], scale, h_rbits[i]);
  }

  uint64_t* d_in = nullptr;
  uint32_t* d_rbits = nullptr;
  uint16_t* d_out = nullptr;
  check_cuda(cudaMalloc(&d_in, N * sizeof(uint64_t)), "cudaMalloc d_in");
  check_cuda(cudaMalloc(&d_rbits, N * sizeof(uint32_t)), "cudaMalloc d_rbits");
  check_cuda(cudaMalloc(&d_out, N * sizeof(uint16_t)), "cudaMalloc d_out");

  check_cuda(cudaMemcpy(d_in, h_in.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice), "H2D in");
  check_cuda(cudaMemcpy(d_rbits, h_rbits.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice), "H2D rbits");

  test_kernel<<<(N + 255) / 256, 256>>>(d_in, scale, d_rbits, d_out, N);
  check_cuda(cudaGetLastError(), "kernel launch");
  check_cuda(cudaDeviceSynchronize(), "kernel sync");

  check_cuda(cudaMemcpy(h_out.data(), d_out, N * sizeof(uint16_t), cudaMemcpyDeviceToHost), "D2H out");

  int mismatches = 0;
  for (int i = 0; i < N; ++i) {
    if (h_out[i] != h_ref[i]) {
      if (mismatches < 10) {
        printf("mismatch at %d: gpu=0x%04X cpu=0x%04X\n", i, h_out[i], h_ref[i]);
      }
      ++mismatches;
    }
  }

  printf("N=%d mismatches=%d\n\n", N, mismatches);
  if (mismatches != 0) exit(1);

  cudaFree(d_in);
  cudaFree(d_rbits);
  cudaFree(d_out);
}

static uint32_t splitmix32_step(uint32_t x) {
  x += 0x9E3779B9u;
  x ^= x >> 16;
  x *= 0x85EBCA6Bu;
  x ^= x >> 13;
  x *= 0xC2B2AE35u;
  x ^= x >> 16;
  return x;
}

static void print_probability_line(const char* label, uint64_t N, uint64_t up_count, double expected) {
  const double p = static_cast<double>(up_count) / static_cast<double>(N);
  const double delta = p - expected;
  printf("%-24s N=%8llu up=%8llu p=%.12f delta=%+.9f\n",
         label,
         static_cast<unsigned long long>(N),
         static_cast<unsigned long long>(up_count),
         p,
         delta);
}

template <typename RngFn>
static uint64_t run_probability_count(float x, uint64_t N, RngFn&& rng_fn) {
  const uint64_t in = pack_4_bf16(x, x, x, x);
  const float2 scale = make_float2(1.0f, 1.0f);
  uint64_t up_count = 0;
  for (uint64_t i = 0; i < N; ++i) {
    const uint32_t rbits = rng_fn(i);
    const uint16_t bits = cpu_reference(in, scale, rbits);
    const float q = decode_one_fp4_e2m1(bits & 0xF);
    if (q == 1.5f) ++up_count;
  }
  return up_count;
}

static void test_probability_suites() {
  constexpr uint64_t N = (1ull << 20);

  printf("=== midpoint probability across multiple random streams ===\n");
  printf("theoretical p_up = 0.5, sigma ~= %.9f for N=%llu\n",
         sqrt(0.25 / static_cast<double>(N)),
         static_cast<unsigned long long>(N));
  print_probability_line("identity counter", N,
                         run_probability_count(1.25f, N, [](uint64_t i) {
                           return static_cast<uint32_t>(i);
                         }),
                         0.5);
  print_probability_line("lcg seed A", N,
                         run_probability_count(1.25f, N, [](uint64_t i) {
                           return static_cast<uint32_t>(i * 1664525u + 1013904223u);
                         }),
                         0.5);
  print_probability_line("lcg seed B", N,
                         run_probability_count(1.25f, N, [](uint64_t i) {
                           return static_cast<uint32_t>((i + 0x31415926u) * 22695477u + 1u);
                         }),
                         0.5);
  print_probability_line("splitmix stream A", N,
                         run_probability_count(1.25f, N, [](uint64_t i) {
                           return splitmix32_step(static_cast<uint32_t>(i));
                         }),
                         0.5);
  print_probability_line("splitmix stream B", N,
                         run_probability_count(1.25f, N, [](uint64_t i) {
                           return splitmix32_step(static_cast<uint32_t>(i + 0xDEADBEEFu));
                         }),
                         0.5);
  {
    std::mt19937 gen(12345u);
    auto count = run_probability_count(1.25f, N, [&](uint64_t) {
      return gen();
    });
    print_probability_line("mt19937 seed 12345", N, count, 0.5);
  }
  {
    std::mt19937 gen(987654321u);
    auto count = run_probability_count(1.25f, N, [&](uint64_t) {
      return gen();
    });
    print_probability_line("mt19937 seed 987654321", N, count, 0.5);
  }
  printf("\n");

  printf("=== non-midpoint probability across multiple random streams ===\n");
  printf("input=1.125, theoretical p_up = 0.25, sigma ~= %.9f for N=%llu\n",
         sqrt(0.25 * 0.75 / static_cast<double>(N)),
         static_cast<unsigned long long>(N));
  print_probability_line("identity counter", N,
                         run_probability_count(1.125f, N, [](uint64_t i) {
                           return static_cast<uint32_t>(i);
                         }),
                         0.25);
  print_probability_line("lcg seed A", N,
                         run_probability_count(1.125f, N, [](uint64_t i) {
                           return static_cast<uint32_t>(i * 1664525u + 1013904223u);
                         }),
                         0.25);
  print_probability_line("lcg seed B", N,
                         run_probability_count(1.125f, N, [](uint64_t i) {
                           return static_cast<uint32_t>((i + 0x31415926u) * 22695477u + 1u);
                         }),
                         0.25);
  print_probability_line("splitmix stream A", N,
                         run_probability_count(1.125f, N, [](uint64_t i) {
                           return splitmix32_step(static_cast<uint32_t>(i));
                         }),
                         0.25);
  print_probability_line("splitmix stream B", N,
                         run_probability_count(1.125f, N, [](uint64_t i) {
                           return splitmix32_step(static_cast<uint32_t>(i + 0xDEADBEEFu));
                         }),
                         0.25);
  {
    std::mt19937 gen(12345u);
    auto count = run_probability_count(1.125f, N, [&](uint64_t) {
      return gen();
    });
    print_probability_line("mt19937 seed 12345", N, count, 0.25);
  }
  {
    std::mt19937 gen(987654321u);
    auto count = run_probability_count(1.125f, N, [&](uint64_t) {
      return gen();
    });
    print_probability_line("mt19937 seed 987654321", N, count, 0.25);
  }
  printf("\n");
}

static void test_fp32_cpu_gpu_consistency() {
  printf("=== FP32 CPU/GPU consistency ===\n");
  constexpr int N = 20000;
  std::mt19937 rng(54321);
  std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
  std::uniform_int_distribution<uint32_t> dist_u32(0u, 0xFFFFFFFFu);

  std::vector<float2> h_in01(N), h_in23(N);
  std::vector<uint32_t> h_rbits(N);
  std::vector<uint16_t> h_out(N), h_ref(N);

  const float2 scale = make_float2(0.75f, 1.25f);

  for (int i = 0; i < N; ++i) {
    h_in01[i] = make_float2(dist(rng), dist(rng));
    h_in23[i] = make_float2(dist(rng), dist(rng));
    h_rbits[i] = dist_u32(rng);
    h_ref[i] = cpu_reference_fp32(h_in01[i], h_in23[i], scale, h_rbits[i]);
  }

  float2* d_in01 = nullptr; float2* d_in23 = nullptr;
  uint32_t* d_rbits = nullptr; uint16_t* d_out = nullptr;
  check_cuda(cudaMalloc(&d_in01, N * sizeof(float2)), "malloc d_in01");
  check_cuda(cudaMalloc(&d_in23, N * sizeof(float2)), "malloc d_in23");
  check_cuda(cudaMalloc(&d_rbits, N * sizeof(uint32_t)), "malloc d_rbits");
  check_cuda(cudaMalloc(&d_out, N * sizeof(uint16_t)), "malloc d_out");

  check_cuda(cudaMemcpy(d_in01, h_in01.data(), N * sizeof(float2), cudaMemcpyHostToDevice), "H2D in01");
  check_cuda(cudaMemcpy(d_in23, h_in23.data(), N * sizeof(float2), cudaMemcpyHostToDevice), "H2D in23");
  check_cuda(cudaMemcpy(d_rbits, h_rbits.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice), "H2D rbits");

  test_fp32_kernel<<<(N + 255) / 256, 256>>>(d_in01, d_in23, scale, d_rbits, d_out, N);
  check_cuda(cudaGetLastError(), "fp32 kernel launch");
  check_cuda(cudaDeviceSynchronize(), "fp32 kernel sync");

  check_cuda(cudaMemcpy(h_out.data(), d_out, N * sizeof(uint16_t), cudaMemcpyDeviceToHost), "D2H out");

  int mismatches = 0;
  for (int i = 0; i < N; ++i) {
    if (h_out[i] != h_ref[i]) {
      if (mismatches < 10)
        printf("mismatch at %d: gpu=0x%04X cpu=0x%04X\n", i, h_out[i], h_ref[i]);
      ++mismatches;
    }
  }

  printf("N=%d mismatches=%d\n\n", N, mismatches);
  if (mismatches != 0) exit(1);

  cudaFree(d_in01); cudaFree(d_in23); cudaFree(d_rbits); cudaFree(d_out);
}

static void test_8x_cpu_gpu_consistency() {
  printf("=== 8x BF16 CPU/GPU consistency ===\n");
  constexpr int N = 20000;
  std::mt19937 rng(67890);
  std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
  std::uniform_int_distribution<uint32_t> dist_u32(0u, 0xFFFFFFFFu);

  std::vector<uint64_t> h_in03(N), h_in47(N);
  std::vector<uint32_t> h_rbits03(N), h_rbits47(N);
  std::vector<uint32_t> h_out(N), h_ref(N);

  const float scale = 0.75f;

  for (int i = 0; i < N; ++i) {
    h_in03[i] = pack_4_bf16(dist(rng), dist(rng), dist(rng), dist(rng));
    h_in47[i] = pack_4_bf16(dist(rng), dist(rng), dist(rng), dist(rng));
    h_rbits03[i] = dist_u32(rng);
    h_rbits47[i] = dist_u32(rng);
    h_ref[i] = cpu_reference_8x(h_in03[i], h_in47[i], scale, h_rbits03[i], h_rbits47[i]);
  }

  uint64_t *d_in03, *d_in47; uint32_t *d_rb03, *d_rb47, *d_out;
  check_cuda(cudaMalloc(&d_in03, N * sizeof(uint64_t)), "malloc d_in03");
  check_cuda(cudaMalloc(&d_in47, N * sizeof(uint64_t)), "malloc d_in47");
  check_cuda(cudaMalloc(&d_rb03, N * sizeof(uint32_t)), "malloc d_rb03");
  check_cuda(cudaMalloc(&d_rb47, N * sizeof(uint32_t)), "malloc d_rb47");
  check_cuda(cudaMalloc(&d_out, N * sizeof(uint32_t)), "malloc d_out");

  check_cuda(cudaMemcpy(d_in03, h_in03.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice), "H2D in03");
  check_cuda(cudaMemcpy(d_in47, h_in47.data(), N * sizeof(uint64_t), cudaMemcpyHostToDevice), "H2D in47");
  check_cuda(cudaMemcpy(d_rb03, h_rbits03.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice), "H2D rb03");
  check_cuda(cudaMemcpy(d_rb47, h_rbits47.data(), N * sizeof(uint32_t), cudaMemcpyHostToDevice), "H2D rb47");

  test_8x_kernel<<<(N + 255) / 256, 256>>>(d_in03, d_in47, scale, d_rb03, d_rb47, d_out, N);
  check_cuda(cudaGetLastError(), "8x kernel launch");
  check_cuda(cudaDeviceSynchronize(), "8x kernel sync");

  check_cuda(cudaMemcpy(h_out.data(), d_out, N * sizeof(uint32_t), cudaMemcpyDeviceToHost), "D2H out");

  int mismatches = 0;
  for (int i = 0; i < N; ++i) {
    if (h_out[i] != h_ref[i]) {
      if (mismatches < 10)
        printf("mismatch at %d: gpu=0x%08X cpu=0x%08X\n", i, h_out[i], h_ref[i]);
      ++mismatches;
    }
  }

  printf("N=%d mismatches=%d\n\n", N, mismatches);
  if (mismatches != 0) exit(1);

  cudaFree(d_in03); cudaFree(d_in47); cudaFree(d_rb03); cudaFree(d_rb47); cudaFree(d_out);
}

struct PerfStats {
  float ms = 0.0f;
  double groups_per_sec = 0.0;
  double values_per_sec = 0.0;
  double gib_per_sec = 0.0;
};

template <typename KernelLauncher>
static PerfStats run_benchmark(const char* label, int n, int warmup, int iters,
                               double bytes_per_group, KernelLauncher&& launcher) {
  cudaEvent_t start, stop;
  check_cuda(cudaEventCreate(&start), "event create start");
  check_cuda(cudaEventCreate(&stop), "event create stop");

  for (int i = 0; i < warmup; ++i) {
    launcher();
  }
  check_cuda(cudaDeviceSynchronize(), "warmup sync");

  check_cuda(cudaEventRecord(start), "event record start");
  for (int i = 0; i < iters; ++i) {
    launcher();
  }
  check_cuda(cudaEventRecord(stop), "event record stop");
  check_cuda(cudaEventSynchronize(stop), "event sync stop");

  float total_ms = 0.0f;
  check_cuda(cudaEventElapsedTime(&total_ms, start, stop), "event elapsed time");
  check_cuda(cudaEventDestroy(start), "event destroy start");
  check_cuda(cudaEventDestroy(stop), "event destroy stop");

  const float ms = total_ms / static_cast<float>(iters);
  const double sec = static_cast<double>(ms) * 1.0e-3;
  const double groups_per_sec = static_cast<double>(n) / sec;
  const double values_per_sec = groups_per_sec * 4.0;
  const double gib_per_sec = (groups_per_sec * bytes_per_group) / static_cast<double>(1ull << 30);

  printf("%-24s time=%.3f ms  groups/s=%.3f G  vals/s=%.3f G  eff_bw=%.3f GiB/s\n",
         label,
         ms,
         groups_per_sec / 1.0e9,
         values_per_sec / 1.0e9,
         gib_per_sec);

  PerfStats s;
  s.ms = ms;
  s.groups_per_sec = groups_per_sec;
  s.values_per_sec = values_per_sec;
  s.gib_per_sec = gib_per_sec;
  return s;
}

static double theoretical_peak_bandwidth_gib_per_s(int dev) {
  int mem_clock_khz = 0;
  int mem_bus_width_bits = 0;

  const cudaError_t e1 = cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, dev);
  const cudaError_t e2 = cudaDeviceGetAttribute(&mem_bus_width_bits, cudaDevAttrGlobalMemoryBusWidth, dev);

  if (e1 != cudaSuccess || e2 != cudaSuccess || mem_clock_khz <= 0 || mem_bus_width_bits <= 0) {
    return -1.0;
  }

  const double mem_clock_hz = static_cast<double>(mem_clock_khz) * 1000.0;
  const double bus_bytes = static_cast<double>(mem_bus_width_bits) / 8.0;
  const double bytes_per_s = 2.0 * mem_clock_hz * bus_bytes;
  return bytes_per_s / static_cast<double>(1ull << 30);
}

static void test_throughput() {
  printf("=== throughput benchmark ===\n");

  int dev = 0;
  check_cuda(cudaGetDevice(&dev), "cudaGetDevice");
  cudaDeviceProp prop;
  check_cuda(cudaGetDeviceProperties(&prop, dev), "cudaGetDeviceProperties");

  const int n = 1 << 24;  // groups of 4 values
  const int threads = 256;
  const int blocks = (n + threads - 1) / threads;
  const int warmup = 10;
  const int iters = 50;
  const double bytes_per_group = sizeof(uint64_t) + sizeof(uint32_t) + sizeof(uint16_t);

  std::vector<uint64_t> h_in(n);
  std::vector<uint32_t> h_rbits(n);
  std::mt19937 rng(123456u);
  std::uniform_real_distribution<float> dist(-8.0f, 8.0f);
  std::uniform_int_distribution<uint32_t> dist_u32(0u, 0xFFFFFFFFu);
  for (int i = 0; i < n; ++i) {
    h_in[i] = pack_4_bf16(dist(rng), dist(rng), dist(rng), dist(rng));
    h_rbits[i] = dist_u32(rng);
  }

  uint64_t* d_in = nullptr;
  uint32_t* d_rbits = nullptr;
  uint16_t* d_out = nullptr;
  check_cuda(cudaMalloc(&d_in, static_cast<size_t>(n) * sizeof(uint64_t)), "perf malloc d_in");
  check_cuda(cudaMalloc(&d_rbits, static_cast<size_t>(n) * sizeof(uint32_t)), "perf malloc d_rbits");
  check_cuda(cudaMalloc(&d_out, static_cast<size_t>(n) * sizeof(uint16_t)), "perf malloc d_out");
  check_cuda(cudaMemcpy(d_in, h_in.data(), static_cast<size_t>(n) * sizeof(uint64_t), cudaMemcpyHostToDevice), "perf memcpy in");
  check_cuda(cudaMemcpy(d_rbits, h_rbits.data(), static_cast<size_t>(n) * sizeof(uint32_t), cudaMemcpyHostToDevice), "perf memcpy rbits");

  const float2 scale = make_float2(0.75f, 1.25f);
  volatile uint16_t sink = 0;

  const PerfStats quant = run_benchmark("quantize kernel", n, warmup, iters, bytes_per_group, [&]() {
    test_kernel<<<blocks, threads>>>(d_in, scale, d_rbits, d_out, n);
  });
  check_cuda(cudaGetLastError(), "quant bench launch");
  check_cuda(cudaDeviceSynchronize(), "quant bench sync");

  const PerfStats base = run_benchmark("stream baseline", n, warmup, iters, bytes_per_group, [&]() {
    stream_baseline_kernel<<<blocks, threads>>>(d_in, d_rbits, d_out, n);
  });
  check_cuda(cudaGetLastError(), "baseline bench launch");
  check_cuda(cudaDeviceSynchronize(), "baseline bench sync");

  check_cuda(cudaMemcpy((void*)&sink, d_out, sizeof(uint16_t), cudaMemcpyDeviceToHost), "perf copy sink");
  (void)sink;

  const double peak_gib_per_s = theoretical_peak_bandwidth_gib_per_s(dev);
  const double quant_vs_base = (base.gib_per_sec > 0.0) ? (quant.gib_per_sec / base.gib_per_sec) : 0.0;
  const double quant_vs_peak = (peak_gib_per_s > 0.0) ? (quant.gib_per_sec / peak_gib_per_s) : 0.0;
  const double base_vs_peak = (peak_gib_per_s > 0.0) ? (base.gib_per_sec / peak_gib_per_s) : 0.0;

  printf("device: %s\n", prop.name);
  printf("problem size: %d groups = %.3f M groups = %.3f M values\n",
         n, n / 1.0e6, (4.0 * n) / 1.0e6);
  printf("bytes/group: %.0f (8B input + 4B rng + 2B output)\n", bytes_per_group);
  if (peak_gib_per_s > 0.0) {
    printf("theoretical peak mem bw: %.3f GiB/s\n", peak_gib_per_s);
  } else {
    printf("theoretical peak mem bw: unavailable\n");
  }
  printf("quantize / baseline eff_bw: %.3f\n", quant_vs_base);
  if (peak_gib_per_s > 0.0) {
    printf("quantize / theoretical peak: %.3f\n", quant_vs_peak);
    printf("baseline / theoretical peak: %.3f\n", base_vs_peak);
  } else {
    printf("quantize / theoretical peak: unavailable\n");
    printf("baseline / theoretical peak: unavailable\n");
  }

  if (quant_vs_base >= 0.80) {
    printf("assessment: likely memory-bound or close to memory-bound (quantize tracks streaming baseline).\n");
  } else if (quant_vs_base >= 0.50) {
    printf("assessment: mixed regime; memory traffic is important, but arithmetic/instruction cost is also visible.\n");
  } else {
    printf("assessment: likely not purely memory-bound; arithmetic/instruction overhead dominates over raw streaming cost.\n");
  }
  printf("\n");

  cudaFree(d_in);
  cudaFree(d_rbits);
  cudaFree(d_out);
}

int main() {
  test_fixed_examples();
  test_exact_representables();
  test_saturation_and_specials();
  test_scale_semantics();
  test_cpu_gpu_consistency();
  test_fp32_cpu_gpu_consistency();
  test_8x_cpu_gpu_consistency();
  test_probability_suites();
  test_throughput();
  printf("All tests passed.\n");
  return 0;
}
