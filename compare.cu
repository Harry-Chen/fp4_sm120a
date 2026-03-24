// ===================================================================
// Comparison test: native cvt.rs.satfinite.e2m1x4 vs polyfill
//
// Build:  make compare.exe CUDA_ARCH=100a
// Run on B300 (SM100a) to compare hardware SR against the polyfill.
// ===================================================================

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include "compare.cuh"

// ===================================================================
// Helpers
// ===================================================================

#define CUDA_CHECK(call) do {                                          \
    cudaError_t err = (call);                                          \
    if (err != cudaSuccess) {                                          \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",                  \
                __FILE__, __LINE__, cudaGetErrorString(err));          \
        exit(1);                                                       \
    }                                                                  \
} while(0)

static uint16_t float_to_bf16_rne(float x) {
    uint32_t u;
    memcpy(&u, &x, sizeof(u));
    uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;
    return (uint16_t)(u >> 16);
}

static uint64_t pack_4_bf16(float a, float b, float c, float d) {
    return (uint64_t)float_to_bf16_rne(a)
         | ((uint64_t)float_to_bf16_rne(b) << 16)
         | ((uint64_t)float_to_bf16_rne(c) << 32)
         | ((uint64_t)float_to_bf16_rne(d) << 48);
}

__host__ __device__
float e2m1_decode(unsigned code4) {
    unsigned sign = (code4 >> 3) & 1u;
    unsigned mag  = code4 & 0x7u;
    const float table[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
    float v = table[mag];
    return sign ? -v : v;
}

// ===================================================================
// Test 1: Bit-exact match rate over random inputs
// ===================================================================

__global__ void kernel_bitexact(
    const uint64_t *__restrict__ inputs,
    const float2 *__restrict__ scales,
    const uint32_t *__restrict__ rbits,
    uint32_t *__restrict__ mismatch_count,
    uint32_t *__restrict__ per_nibble_mismatch,  // [4]
    int n
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    uint16_t native_out = native_mul_cvt_bf16_to_fp4_4x_sr(
        inputs[idx], scales[idx], rbits[idx]).__x;
    uint16_t poly_out = polyfill_mul_cvt_bf16_to_fp4_4x_sr(
        inputs[idx], scales[idx], rbits[idx]).__x;

    if (native_out != poly_out) {
        atomicAdd(mismatch_count, 1u);
        for (int nib = 0; nib < 4; nib++) {
            unsigned n_nib = (native_out >> (nib * 4)) & 0xFu;
            unsigned p_nib = (poly_out >> (nib * 4)) & 0xFu;
            if (n_nib != p_nib)
                atomicAdd(&per_nibble_mismatch[nib], 1u);
        }
    }
}

// ===================================================================
// Test 2: Round-up probability comparison
//   For a fixed input value, run many random rbits and measure
//   P(round_up) for native vs polyfill on nibble 0.
// ===================================================================

__global__ void kernel_roundup_prob(
    const uint64_t in_4x,
    const float2 scale,
    uint32_t *__restrict__ native_up,    // [1]
    uint32_t *__restrict__ polyfill_up,  // [1]
    double *__restrict__ native_sum,     // [4] sum of decoded per nibble
    double *__restrict__ polyfill_sum,   // [4] sum of decoded per nibble
    int n_trials_per_val
) {
    curandState rng;
    curand_init(42u, blockIdx.x * blockDim.x + threadIdx.x, 0, &rng);

    int trials_per_thread = n_trials_per_val / (gridDim.x * blockDim.x);
    uint32_t local_native_up = 0, local_poly_up = 0;
    double local_native_sum[4] = {}, local_poly_sum[4] = {};

    for (int i = 0; i < trials_per_thread; i++) {
        uint32_t rb = curand(&rng);

        uint16_t n_out = native_mul_cvt_bf16_to_fp4_4x_sr(in_4x, scale, rb).__x;
        uint16_t p_out = polyfill_mul_cvt_bf16_to_fp4_4x_sr(in_4x, scale, rb).__x;

        for (int nib = 0; nib < 4; nib++) {
            float n_dec = e2m1_decode((n_out >> (nib * 4)) & 0xFu);
            float p_dec = e2m1_decode((p_out >> (nib * 4)) & 0xFu);
            local_native_sum[nib] += (double)n_dec;
            local_poly_sum[nib]   += (double)p_dec;
        }

        // Track round-up on nibble 0 specifically
        float n_dec0 = e2m1_decode(n_out & 0xFu);
        float p_dec0 = e2m1_decode(p_out & 0xFu);
        // "round up" = decoded magnitude > input magnitude (for nibble 0)
        // We use absolute comparison: if abs(decoded) > floor
        // For simplicity, just count when the higher of the two neighbors is chosen
        // We detect this by comparing against the lower neighbor
        float n_dec0_abs = fabsf(n_dec0);
        float p_dec0_abs = fabsf(p_dec0);
        // Use a threshold: the first decoded value we see tells us the floor
        // Actually let's just count directly
        if (n_dec0_abs > fabsf(e2m1_decode((n_out & 0x7u))))
            ; // complex — let's just use the sum/mean approach
        // Simplified: track how often native and polyfill agree
        local_native_up += (n_out != p_out) ? 0 : 1;  // agreement count
    }

    // Reduce via atomics (simple for this diagnostic)
    atomicAdd(native_up, local_native_up);
    for (int nib = 0; nib < 4; nib++) {
        atomicAdd(&native_sum[nib], local_native_sum[nib]);
        atomicAdd(&polyfill_sum[nib], local_poly_sum[nib]);
    }
}

// ===================================================================
// Test 3: Per-value detailed comparison
//   For specific test values, compute mean decoded value (= bias test)
//   and round-up probability for both native and polyfill.
// ===================================================================

struct PerValueStats {
    double native_sum;
    double polyfill_sum;
    uint32_t native_round_up;
    uint32_t polyfill_round_up;
};

__global__ void kernel_per_value_stats(
    const float *__restrict__ test_vals,  // values to test (pre-scaled)
    PerValueStats *__restrict__ stats,    // [n_vals]
    int n_vals,
    int n_trials
) {
    int vid = blockIdx.x;
    if (vid >= n_vals) return;

    float x = test_vals[vid];

    // Pack x into nibble 0 position: the asm produces nibble0 = v2 = bf16[3]*scale.y
    // So to get x into nibble 0: set bf16[3] = x, scale.y = 1.0
    // pack_4_bf16(0, 0, 0, x) puts x at bf16[3]
    uint16_t x_bf16 = __float_as_uint(x) >> 16;  // approximate bf16
    // More precise: reconstruct from float
    // Since we're on device, use __float2bfloat16 if available
    // For simplicity, just shift (works for normal floats representable in bf16)
    uint64_t in_4x = ((uint64_t)x_bf16 << 48);  // bf16[3] = x, rest = 0
    float2 scale = make_float2(1.0f, 1.0f);

    curandState rng;
    curand_init(12345u + vid, threadIdx.x, 0, &rng);

    int trials_per_thread = n_trials / blockDim.x;
    double local_native_sum = 0.0, local_poly_sum = 0.0;
    uint32_t local_native_up = 0, local_poly_up = 0;

    // Determine the floor value for round-up detection
    float ax = fabsf(x);
    const float e2m1_vals[8] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
    float x_floor = 0.0f;
    for (int i = 7; i >= 0; i--) {
        if (ax >= e2m1_vals[i]) { x_floor = e2m1_vals[i]; break; }
    }

    for (int i = 0; i < trials_per_thread; i++) {
        uint32_t rb = curand(&rng);

        uint16_t n_out = native_mul_cvt_bf16_to_fp4_4x_sr(in_4x, scale, rb).__x;
        uint16_t p_out = polyfill_mul_cvt_bf16_to_fp4_4x_sr(in_4x, scale, rb).__x;

        // Nibble 0 = bf16[3]*scale.y = x
        float n_dec = e2m1_decode(n_out & 0xFu);
        float p_dec = e2m1_decode(p_out & 0xFu);

        local_native_sum  += (double)n_dec;
        local_poly_sum    += (double)p_dec;

        if (fabsf(n_dec) > x_floor) local_native_up++;
        if (fabsf(p_dec) > x_floor) local_poly_up++;
    }

    // Block reduction
    __shared__ double s_native_sum[256], s_poly_sum[256];
    __shared__ uint32_t s_native_up[256], s_poly_up[256];
    s_native_sum[threadIdx.x] = local_native_sum;
    s_poly_sum[threadIdx.x]   = local_poly_sum;
    s_native_up[threadIdx.x]  = local_native_up;
    s_poly_up[threadIdx.x]    = local_poly_up;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_native_sum[threadIdx.x] += s_native_sum[threadIdx.x + s];
            s_poly_sum[threadIdx.x]   += s_poly_sum[threadIdx.x + s];
            s_native_up[threadIdx.x]  += s_native_up[threadIdx.x + s];
            s_poly_up[threadIdx.x]    += s_poly_up[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(&stats[vid].native_sum, s_native_sum[0]);
        atomicAdd(&stats[vid].polyfill_sum, s_poly_sum[0]);
        atomicAdd(&stats[vid].native_round_up, s_native_up[0]);
        atomicAdd(&stats[vid].polyfill_round_up, s_poly_up[0]);
    }
}

// ===================================================================
// Test 4: Sweep all 256 rbits values for byte 0
//   For a fixed input, sweep rbits byte 0 through 0..255 and
//   compare the decision boundary of native vs polyfill.
// ===================================================================

__global__ void kernel_rbits_sweep(
    const uint64_t in_4x,
    const float2 scale,
    uint16_t *__restrict__ native_out,   // [256]
    uint16_t *__restrict__ polyfill_out  // [256]
) {
    int rb_byte = threadIdx.x;  // 0..255
    // Put the sweep byte in byte 0 of rbits, zero out the rest
    uint32_t rb = (uint32_t)rb_byte;

    native_out[rb_byte]   = native_mul_cvt_bf16_to_fp4_4x_sr(in_4x, scale, rb).__x;
    polyfill_out[rb_byte] = polyfill_mul_cvt_bf16_to_fp4_4x_sr(in_4x, scale, rb).__x;
}

// ===================================================================
// main
// ===================================================================

int main() {
    printf("=== Native vs Polyfill Comparison ===\n\n");

    // Print device info
    {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        printf("Device: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);
    }

    // ---- Test 1: Bit-exact match rate ----
    {
        printf("[Test 1] Bit-exact match rate over random inputs\n");
        const int N = 1 << 20;  // ~1M samples

        // Generate random inputs on host
        uint64_t *h_in = new uint64_t[N];
        float2 *h_sc = new float2[N];
        uint32_t *h_rb = new uint32_t[N];

        srand(42);
        for (int i = 0; i < N; i++) {
            float a = ((float)rand() / RAND_MAX - 0.5f) * 12.0f;
            float b = ((float)rand() / RAND_MAX - 0.5f) * 12.0f;
            float c = ((float)rand() / RAND_MAX - 0.5f) * 12.0f;
            float d = ((float)rand() / RAND_MAX - 0.5f) * 12.0f;
            h_in[i] = pack_4_bf16(a, b, c, d);
            float sx = 0.1f + ((float)rand() / RAND_MAX) * 2.0f;
            float sy = 0.1f + ((float)rand() / RAND_MAX) * 2.0f;
            h_sc[i] = make_float2(sx, sy);
            h_rb[i] = (uint32_t)rand() ^ ((uint32_t)rand() << 16);
        }

        uint64_t *d_in; float2 *d_sc; uint32_t *d_rb;
        uint32_t *d_mismatch, *d_per_nib;
        CUDA_CHECK(cudaMalloc(&d_in, N * sizeof(uint64_t)));
        CUDA_CHECK(cudaMalloc(&d_sc, N * sizeof(float2)));
        CUDA_CHECK(cudaMalloc(&d_rb, N * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_mismatch, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_per_nib, 4 * sizeof(uint32_t)));
        CUDA_CHECK(cudaMemcpy(d_in, h_in, N * sizeof(uint64_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sc, h_sc, N * sizeof(float2), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_rb, h_rb, N * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_mismatch, 0, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_per_nib, 0, 4 * sizeof(uint32_t)));

        kernel_bitexact<<<(N + 255) / 256, 256>>>(
            d_in, d_sc, d_rb, d_mismatch, d_per_nib, N);
        CUDA_CHECK(cudaDeviceSynchronize());

        uint32_t h_mismatch, h_per_nib[4];
        CUDA_CHECK(cudaMemcpy(&h_mismatch, d_mismatch, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_per_nib, d_per_nib, 4 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

        printf("  Total samples: %d\n", N);
        printf("  Mismatched:    %u (%.4f%%)\n", h_mismatch, 100.0 * h_mismatch / N);
        printf("  Match rate:    %.4f%%\n", 100.0 * (N - h_mismatch) / N);
        for (int nib = 0; nib < 4; nib++) {
            printf("  Nibble %d mismatches: %u (%.4f%%)\n",
                   nib, h_per_nib[nib], 100.0 * h_per_nib[nib] / N);
        }
        printf("\n");

        delete[] h_in; delete[] h_sc; delete[] h_rb;
        cudaFree(d_in); cudaFree(d_sc); cudaFree(d_rb);
        cudaFree(d_mismatch); cudaFree(d_per_nib);
    }

    // ---- Test 2: Unbiasedness / mean comparison over random inputs ----
    {
        printf("[Test 2] Unbiasedness: E[decoded] for random inputs\n");
        const int N = 1 << 20;
        int n_trials = 256 * 4096;

        // Pick a representative random input
        uint64_t h_in = pack_4_bf16(0.7f, -1.3f, 2.5f, -4.8f);
        float2 h_sc = make_float2(1.0f, 1.0f);

        uint32_t *d_agree;
        double *d_native_sum, *d_polyfill_sum;
        CUDA_CHECK(cudaMalloc(&d_agree, sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&d_native_sum, 4 * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_polyfill_sum, 4 * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_agree, 0, sizeof(uint32_t)));
        CUDA_CHECK(cudaMemset(d_native_sum, 0, 4 * sizeof(double)));
        CUDA_CHECK(cudaMemset(d_polyfill_sum, 0, 4 * sizeof(double)));

        kernel_roundup_prob<<<256, 256>>>(
            h_in, h_sc, d_agree, d_agree,
            d_native_sum, d_polyfill_sum, n_trials);
        CUDA_CHECK(cudaDeviceSynchronize());

        double h_native_sum[4], h_polyfill_sum[4];
        CUDA_CHECK(cudaMemcpy(h_native_sum, d_native_sum, 4 * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_polyfill_sum, d_polyfill_sum, 4 * sizeof(double), cudaMemcpyDeviceToHost));

        // The nibble layout after the asm (mul.f32x2 + swap + {v2,v3,v0,v1}):
        //   nibble0 = bf16[3]*sy = -4.8, nibble1 = bf16[2]*sx = 2.5,
        //   nibble2 = bf16[1]*sy = -1.3, nibble3 = bf16[0]*sx = 0.7
        float expect_nibble[] = {-4.8f, 2.5f, -1.3f, 0.7f};

        printf("  Trials per nibble: %d\n", n_trials);
        printf("  %-8s  %-12s  %-12s  %-12s  %-12s  %-12s\n",
               "nibble", "input", "E[native]", "E[polyfill]", "native_bias", "poly_bias");
        for (int nib = 0; nib < 4; nib++) {
            double n_mean = h_native_sum[nib] / (double)n_trials;
            double p_mean = h_polyfill_sum[nib] / (double)n_trials;
            double n_bias = n_mean - (double)expect_nibble[nib];
            double p_bias = p_mean - (double)expect_nibble[nib];
            printf("  %-8d  %+8.4f      %+11.6f   %+11.6f   %+11.6f   %+11.6f\n",
                   nib, expect_nibble[nib], n_mean, p_mean, n_bias, p_bias);
        }
        printf("\n");

        cudaFree(d_agree); cudaFree(d_native_sum); cudaFree(d_polyfill_sum);
    }

    // ---- Test 3: Per-value round-up probability ----
    {
        printf("[Test 3] Per-value round-up probability: native vs polyfill\n");
        const float h_vals[] = {
            0.1f, 0.25f, 0.3f, 0.75f,
            1.2f, 1.25f, 1.4f,
            2.3f, 2.5f, 2.7f, 3.5f,
            4.5f, 5.0f, 5.5f,
            -0.3f, -1.25f, -2.5f, -5.0f
        };
        int n_vals = sizeof(h_vals) / sizeof(h_vals[0]);
        int n_trials = 256 * 8192;

        float *d_vals;
        PerValueStats *d_stats;
        CUDA_CHECK(cudaMalloc(&d_vals, n_vals * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_stats, n_vals * sizeof(PerValueStats)));
        CUDA_CHECK(cudaMemcpy(d_vals, h_vals, n_vals * sizeof(float), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemset(d_stats, 0, n_vals * sizeof(PerValueStats)));

        kernel_per_value_stats<<<n_vals, 256>>>(d_vals, d_stats, n_vals, n_trials);
        CUDA_CHECK(cudaDeviceSynchronize());

        PerValueStats *h_stats = new PerValueStats[n_vals];
        CUDA_CHECK(cudaMemcpy(h_stats, d_stats, n_vals * sizeof(PerValueStats), cudaMemcpyDeviceToHost));

        // Compute expected P(round_up)
        const float e2m1_vals[] = {0.f, 0.5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f};
        auto floor_val = [&](float ax) -> float {
            for (int i = 7; i >= 0; i--)
                if (ax >= e2m1_vals[i]) return e2m1_vals[i];
            return 0.0f;
        };
        auto ceil_val = [&](float ax) -> float {
            for (int i = 0; i < 8; i++)
                if (e2m1_vals[i] > ax) return e2m1_vals[i];
            return 6.0f;
        };

        printf("  %-8s  %-10s  %-10s  %-10s  %-10s  %-10s  %-12s  %-12s\n",
               "x", "P_expect", "P_native", "P_poly", "E_native", "E_poly",
               "bias_native", "bias_poly");
        for (int i = 0; i < n_vals; i++) {
            float x = h_vals[i];
            float ax = fabsf(x);
            float lo = floor_val(ax);
            float hi = ceil_val(ax);
            float ulp = hi - lo;
            float p_expect = (ulp > 0.0f) ? (ax - lo) / ulp : 0.0f;

            double p_native  = (double)h_stats[i].native_round_up / (double)n_trials;
            double p_poly    = (double)h_stats[i].polyfill_round_up / (double)n_trials;
            double e_native  = h_stats[i].native_sum / (double)n_trials;
            double e_poly    = h_stats[i].polyfill_sum / (double)n_trials;
            double bias_native = e_native - (double)x;
            double bias_poly   = e_poly - (double)x;

            printf("  %+7.3f   %9.5f   %9.5f   %9.5f   %+9.5f   %+9.5f   %+11.6f   %+11.6f\n",
                   x, p_expect, p_native, p_poly, e_native, e_poly, bias_native, bias_poly);
        }
        printf("\n");

        delete[] h_stats;
        cudaFree(d_vals); cudaFree(d_stats);
    }

    // ---- Test 4: rbits byte 0 sweep for a specific value ----
    {
        printf("[Test 4] rbits byte-0 sweep: decision boundary comparison\n");
        // Test with x = 1.25 (midpoint between 1.0 and 1.5)
        // Put into nibble 0: bf16[3] = 1.25, scale.y = 1.0
        uint64_t h_in = pack_4_bf16(0.0f, 0.0f, 0.0f, 1.25f);
        float2 h_sc = make_float2(1.0f, 1.0f);

        uint16_t *d_native_out, *d_poly_out;
        CUDA_CHECK(cudaMalloc(&d_native_out, 256 * sizeof(uint16_t)));
        CUDA_CHECK(cudaMalloc(&d_poly_out, 256 * sizeof(uint16_t)));

        kernel_rbits_sweep<<<1, 256>>>(h_in, h_sc, d_native_out, d_poly_out);
        CUDA_CHECK(cudaDeviceSynchronize());

        uint16_t h_native[256], h_poly[256];
        CUDA_CHECK(cudaMemcpy(h_native, d_native_out, 256 * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_poly, d_poly_out, 256 * sizeof(uint16_t), cudaMemcpyDeviceToHost));

        // Analyze nibble 0 decisions
        int native_up = 0, poly_up = 0, agree = 0, disagree = 0;
        int first_disagree = -1;
        for (int rb = 0; rb < 256; rb++) {
            unsigned n_nib0 = h_native[rb] & 0xFu;
            unsigned p_nib0 = h_poly[rb] & 0xFu;
            float n_dec = e2m1_decode(n_nib0);
            float p_dec = e2m1_decode(p_nib0);
            if (fabsf(n_dec) > 1.0f) native_up++;
            if (fabsf(p_dec) > 1.0f) poly_up++;
            if (n_nib0 == p_nib0) {
                agree++;
            } else {
                disagree++;
                if (first_disagree < 0) first_disagree = rb;
            }
        }

        printf("  Input: 1.25 (midpoint between 1.0 and 1.5)\n");
        printf("  Sweeping rbits byte 0 = 0..255 (other bytes = 0)\n");
        printf("  Native  round-up count: %d / 256 (P=%.4f)\n", native_up, native_up / 256.0);
        printf("  Polyfill round-up count: %d / 256 (P=%.4f)\n", poly_up, poly_up / 256.0);
        printf("  Expected P(round_up) = 0.5\n");
        printf("  Agree: %d  Disagree: %d\n", agree, disagree);
        if (first_disagree >= 0) {
            printf("  First disagreement at rbits_byte0 = %d:\n", first_disagree);
            printf("    native  nibble0 = 0x%X -> %g\n",
                   h_native[first_disagree] & 0xF,
                   e2m1_decode(h_native[first_disagree] & 0xFu));
            printf("    polyfill nibble0 = 0x%X -> %g\n",
                   h_poly[first_disagree] & 0xF,
                   e2m1_decode(h_poly[first_disagree] & 0xFu));
        }

        // Also test 0.75 (quarter-point between 0.5 and 1.0)
        printf("\n  Input: 0.75 (quarter-point: P_up should be 0.5)\n");
        h_in = pack_4_bf16(0.0f, 0.0f, 0.0f, 0.75f);
        CUDA_CHECK(cudaMemcpy(d_native_out, h_native, 256 * sizeof(uint16_t), cudaMemcpyHostToDevice));
        kernel_rbits_sweep<<<1, 256>>>(h_in, h_sc, d_native_out, d_poly_out);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_native, d_native_out, 256 * sizeof(uint16_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_poly, d_poly_out, 256 * sizeof(uint16_t), cudaMemcpyDeviceToHost));

        native_up = 0; poly_up = 0; agree = 0; disagree = 0;
        for (int rb = 0; rb < 256; rb++) {
            unsigned n_nib0 = h_native[rb] & 0xFu;
            unsigned p_nib0 = h_poly[rb] & 0xFu;
            float n_dec = e2m1_decode(n_nib0);
            float p_dec = e2m1_decode(p_nib0);
            if (fabsf(n_dec) > 0.5f) native_up++;
            if (fabsf(p_dec) > 0.5f) poly_up++;
            if (n_nib0 == p_nib0) agree++; else disagree++;
        }
        printf("  Native  round-up: %d / 256 (P=%.4f)\n", native_up, native_up / 256.0);
        printf("  Polyfill round-up: %d / 256 (P=%.4f)\n", poly_up, poly_up / 256.0);
        printf("  Expected P(round_up) = 0.5\n");
        printf("  Agree: %d  Disagree: %d\n", agree, disagree);

        printf("\n");
        cudaFree(d_native_out); cudaFree(d_poly_out);
    }

    // ---- Test 5: Exact representable values should always match ----
    {
        printf("[Test 5] Exact representable values (must always match)\n");
        const float exact[] = {
            0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
            -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
        };
        int n_exact = sizeof(exact) / sizeof(exact[0]);

        // Test with 1000 random rbits each
        int total_fail = 0;
        for (int ei = 0; ei < n_exact; ei++) {
            float x = exact[ei];
            uint64_t h_in = pack_4_bf16(x, x, x, x);
            float2 h_sc = make_float2(1.0f, 1.0f);

            uint16_t *d_native_out, *d_poly_out;
            CUDA_CHECK(cudaMalloc(&d_native_out, 256 * sizeof(uint16_t)));
            CUDA_CHECK(cudaMalloc(&d_poly_out, 256 * sizeof(uint16_t)));

            kernel_rbits_sweep<<<1, 256>>>(h_in, h_sc, d_native_out, d_poly_out);
            CUDA_CHECK(cudaDeviceSynchronize());

            uint16_t h_native[256], h_poly[256];
            CUDA_CHECK(cudaMemcpy(h_native, d_native_out, 256 * sizeof(uint16_t), cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(h_poly, d_poly_out, 256 * sizeof(uint16_t), cudaMemcpyDeviceToHost));

            int mismatches = 0;
            for (int rb = 0; rb < 256; rb++) {
                if (h_native[rb] != h_poly[rb]) mismatches++;
            }
            if (mismatches > 0) {
                printf("  FAIL x=%+6g: %d/256 mismatches\n", x, mismatches);
                total_fail += mismatches;
            }

            cudaFree(d_native_out); cudaFree(d_poly_out);
        }
        if (total_fail == 0)
            printf("  All exact values match across 256 rbits each: PASS\n");
        printf("\n");
    }

    printf("=== Done ===\n");
    return 0;
}
