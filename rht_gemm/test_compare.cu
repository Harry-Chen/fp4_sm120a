// Comparison test: our SM120 kernel vs the original TE SM100 reference.
//
// Both kernels run on the SAME SM100 GPU (e.g. GB200) with identical inputs.
// The two kernels live in separate translation units to avoid include conflicts
// between sr.sm120.cuh and TE's ptx.cuh.
//
// Build: make test_compare.exe   (requires compute_100a)
// Run:   on a GB200 or B200 machine

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

// Defined in test_compare_ours.cu
extern void run_ours(int m, int n,
                     const __nv_bfloat16* A, const __nv_bfloat16* B,
                     uint8_t* C, uint8_t* SFC,
                     const float* global_amax, const size_t* rng_state,
                     uint32_t sm_count, cudaStream_t stream);

// Defined in test_compare_ref.cu
extern void run_ref(int m, int n,
                    const __nv_bfloat16* A, const __nv_bfloat16* B,
                    uint8_t* C, uint8_t* SFC,
                    const float* global_amax, const size_t* rng_state,
                    uint32_t sm_count, cudaStream_t stream);

#define CHECK_CUDA(call)                                                    \
    do {                                                                    \
        cudaError_t err = (call);                                           \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,         \
                    __LINE__, cudaGetErrorString(err));                      \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

static uint8_t get_nibble(const uint8_t* data, int idx) {
    return (idx % 2 == 0) ? (data[idx / 2] & 0xF) : (data[idx / 2] >> 4);
}

// Benchmark a kernel: warmup + timed runs, returns average milliseconds.
static float benchmark_kernel(
    void (*fn)(int, int, const __nv_bfloat16*, const __nv_bfloat16*,
               uint8_t*, uint8_t*, const float*, const size_t*, uint32_t, cudaStream_t),
    int M, int N,
    const __nv_bfloat16* d_A, const __nv_bfloat16* d_B,
    uint8_t* d_C, uint8_t* d_SFC,
    const float* d_amax, const size_t* d_rng,
    uint32_t sm_count)
{
    constexpr int WARMUP = 10;
    constexpr int RUNS   = 50;

    for (int i = 0; i < WARMUP; i++)
        fn(M, N, d_A, d_B, d_C, d_SFC, d_amax, d_rng, sm_count, 0);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CHECK_CUDA(cudaEventCreate(&t0));
    CHECK_CUDA(cudaEventCreate(&t1));
    CHECK_CUDA(cudaEventRecord(t0));
    for (int i = 0; i < RUNS; i++)
        fn(M, N, d_A, d_B, d_C, d_SFC, d_amax, d_rng, sm_count, 0);
    CHECK_CUDA(cudaEventRecord(t1));
    CHECK_CUDA(cudaDeviceSynchronize());

    float ms;
    CHECK_CUDA(cudaEventElapsedTime(&ms, t0, t1));
    CHECK_CUDA(cudaEventDestroy(t0));
    CHECK_CUDA(cudaEventDestroy(t1));
    return ms / RUNS;
}

// Fill the 16x16 Hadamard matrix (scaled by 0.25)
static void fill_hadamard(std::vector<__nv_bfloat16>& h_B) {
    h_B.resize(16 * 16);
    for (int r = 0; r < 16; r++)
        for (int c = 0; c < 16; c++) {
            int sign = __builtin_popcount(r & c) % 2 == 0 ? 1 : -1;
            h_B[r * 16 + c] = __float2bfloat16(sign * 0.25f);
        }
}

bool run_comparison(int M, int N,
                    const char* label = nullptr,
                    std::vector<__nv_bfloat16>* custom_A = nullptr,
                    float global_amax_val = 4.0f) {
    if (label)
        printf("--- %s (%d x %d, amax=%.4g) ---\n", label, M, N, global_amax_val);
    else
        printf("--- %d x %d ---\n", M, N);

    std::vector<__nv_bfloat16> h_A_default, h_B;
    fill_hadamard(h_B);

    if (!custom_A) {
        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);
        h_A_default.resize(M * N);
        for (auto& v : h_A_default) v = __float2bfloat16(dist(gen));
        custom_A = &h_A_default;
    }

    size_t rng_h[2] = {12345, 0};

    __nv_bfloat16 *d_A, *d_B;
    float *d_amax;
    size_t *d_rng;
    uint8_t *d_C_ours, *d_SFC_ours, *d_C_ref, *d_SFC_ref;

    size_t c_bytes = M * N / 2;
    size_t sfc_bytes = M * (N / 16);

    CHECK_CUDA(cudaMalloc(&d_A, M * N * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_B, 256 * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_amax, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_rng, 2 * sizeof(size_t)));
    CHECK_CUDA(cudaMalloc(&d_C_ours, c_bytes));
    CHECK_CUDA(cudaMalloc(&d_SFC_ours, sfc_bytes));
    CHECK_CUDA(cudaMalloc(&d_C_ref, c_bytes));
    CHECK_CUDA(cudaMalloc(&d_SFC_ref, sfc_bytes));

    CHECK_CUDA(cudaMemcpy(d_A, custom_A->data(), M * N * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B.data(), 256 * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_amax, &global_amax_val, sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_rng, rng_h, 2 * sizeof(size_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_C_ours, 0, c_bytes));
    CHECK_CUDA(cudaMemset(d_SFC_ours, 0, sfc_bytes));
    CHECK_CUDA(cudaMemset(d_C_ref, 0, c_bytes));
    CHECK_CUDA(cudaMemset(d_SFC_ref, 0, sfc_bytes));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    uint32_t sm_count = prop.multiProcessorCount;

    // ---- Correctness comparison ----
    printf("  Correctness check...\n");
    run_ours(M, N, d_A, d_B, d_C_ours, d_SFC_ours, d_amax, d_rng, sm_count, 0);
    CHECK_CUDA(cudaDeviceSynchronize());
    run_ref(M, N, d_A, d_B, d_C_ref, d_SFC_ref, d_amax, d_rng, sm_count, 0);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<uint8_t> h_C_ours(c_bytes), h_C_ref(c_bytes);
    std::vector<uint8_t> h_SFC_ours(sfc_bytes), h_SFC_ref(sfc_bytes);
    CHECK_CUDA(cudaMemcpy(h_C_ours.data(), d_C_ours, c_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_C_ref.data(), d_C_ref, c_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_SFC_ours.data(), d_SFC_ours, sfc_bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_SFC_ref.data(), d_SFC_ref, sfc_bytes, cudaMemcpyDeviceToHost));

    int sfc_mm = 0;
    for (size_t i = 0; i < sfc_bytes; i++)
        if (h_SFC_ours[i] != h_SFC_ref[i]) {
            if (++sfc_mm <= 5) {
                int row = i / (N / 16), grp = i % (N / 16);
                printf("  SFC mismatch row=%d group=%d: ours=0x%02X ref=0x%02X\n",
                       row, grp, h_SFC_ours[i], h_SFC_ref[i]);
            }
        }

    int fp4_mm = 0;
    for (int i = 0; i < M * N; i++)
        if (get_nibble(h_C_ours.data(), i) != get_nibble(h_C_ref.data(), i)) {
            if (++fp4_mm <= 5) {
                int r = i / N, c = i % N;
                printf("  FP4 mismatch (%d,%d): ours=0x%X ref=0x%X\n",
                       r, c, get_nibble(h_C_ours.data(), i), get_nibble(h_C_ref.data(), i));
            }
        }

    printf("  SFC: %d / %zu mismatches (%.4f%%)\n", sfc_mm, sfc_bytes,
           sfc_bytes ? 100.0 * sfc_mm / sfc_bytes : 0.0);
    printf("  FP4: %d / %d mismatches (%.4f%%)\n", fp4_mm, M * N,
           M * N ? 100.0 * fp4_mm / (M * N) : 0.0);
    bool ok = (fp4_mm == 0 && sfc_mm == 0);
    printf("  Correctness: %s\n", ok ? "MATCH" : "MISMATCH");

    // ---- Performance benchmark (only if correctness passes) ----
    if (ok) {
        float ms_ours = benchmark_kernel(run_ours, M, N, d_A, d_B,
                                         d_C_ours, d_SFC_ours, d_amax, d_rng, sm_count);
        float ms_ref  = benchmark_kernel(run_ref,  M, N, d_A, d_B,
                                         d_C_ref,  d_SFC_ref,  d_amax, d_rng, sm_count);

        // FLOP count: N/16 independent 16x16 matmuls per row, M rows.
        // Each 16x16 matmul = 2*16*16*16 = 8192 FLOP (FMA counted as 2).
        double flops = 2.0 * M * N * 16;
        double gflops_ours = flops / (ms_ours * 1e-3) / 1e9;
        double gflops_ref  = flops / (ms_ref  * 1e-3) / 1e9;

        // Effective bandwidth: read A (BF16) + write C (FP4) + write SFC (FP8)
        double bytes = (double)M * N * 2 + (double)M * N / 2 + (double)M * (N / 16) + 512;
        double bw_ours = bytes / (ms_ours * 1e-3) / 1e9;
        double bw_ref  = bytes / (ms_ref  * 1e-3) / 1e9;

        printf("  Performance:\n");
        printf("    Ours (WMMA):  %8.3f ms  %8.1f GFLOPS  %8.1f GB/s\n",
               ms_ours, gflops_ours, bw_ours);
        printf("    Ref  (UMMA):  %8.3f ms  %8.1f GFLOPS  %8.1f GB/s\n",
               ms_ref, gflops_ref, bw_ref);
        printf("    Speedup: %.2fx\n", ms_ref / ms_ours);
    }
    printf("\n");

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_amax));
    CHECK_CUDA(cudaFree(d_rng));
    CHECK_CUDA(cudaFree(d_C_ours));
    CHECK_CUDA(cudaFree(d_SFC_ours));
    CHECK_CUDA(cudaFree(d_C_ref));
    CHECK_CUDA(cudaFree(d_SFC_ref));
    return ok;
}

int main() {
    printf("=== RHT GEMM: SM120 kernel vs TE SM100 reference ===\n\n");

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    if (prop.major < 10) {
        fprintf(stderr, "Error: SM100+ GPU required (got SM %d.%d).\n",
                prop.major, prop.minor);
        return 1;
    }

    int pass = 0, total = 0;
    auto test = [&](int m, int n) { total++; if (run_comparison(m, n)) pass++; };

    // Standard random data
    test(256, 128);
    test(1024, 1024);
    test(8192, 5120);

    // Extreme data tests — compare both kernels with edge-case inputs
    printf("\n=== Extreme Data Tests ===\n");
    constexpr int EM = 256, EN = 128;
    std::mt19937 gen(42);
    auto extreme = [&](const char* label, auto fill_fn, float amax) {
        std::vector<__nv_bfloat16> A(EM * EN);
        fill_fn(A);
        total++;
        if (run_comparison(EM, EN, label, &A, amax)) pass++;
    };

    extreme("all_zeros", [](auto& A) {
        for (auto& v : A) v = __float2bfloat16(0.0f);
    }, 1.0f);

    extreme("all_zeros_amax0", [](auto& A) {
        for (auto& v : A) v = __float2bfloat16(0.0f);
    }, 0.0f);

    extreme("very_small", [&](auto& A) {
        for (auto& v : A) v = __float2bfloat16((gen()%2?1:-1) * 1e-6f * (1+(gen()%100)/100.f));
    }, 1e-4f);

    extreme("very_large", [&](auto& A) {
        for (auto& v : A) v = __float2bfloat16((gen()%2?1:-1) * (500.f + gen()%1000));
    }, 1000.0f);

    extreme("very_large_small_amax", [&](auto& A) {
        for (auto& v : A) v = __float2bfloat16((gen()%2?1:-1) * (500.f + gen()%1000));
    }, 1.0f);

    extreme("sparse_outliers", [&](auto& A) {
        for (int i = 0; i < EM * EN; i++)
            A[i] = __float2bfloat16(gen()%256==0 ? (gen()%2?100.f:-100.f) : 0.f);
    }, 100.0f);

    extreme("uniform_one", [](auto& A) {
        for (auto& v : A) v = __float2bfloat16(1.0f);
    }, 4.0f);

    extreme("uniform_one_tiny_amax", [](auto& A) {
        for (auto& v : A) v = __float2bfloat16(1.0f);
    }, 0.001f);

    extreme("contains_inf", [&](auto& A) {
        uint16_t pos_inf = 0x7F80, neg_inf = 0xFF80;
        for (int i = 0; i < EM * EN; i++) {
            if (gen() % 256 == 0) {
                uint16_t bits = (gen() % 2) ? pos_inf : neg_inf;
                A[i] = *reinterpret_cast<__nv_bfloat16*>(&bits);
            } else {
                A[i] = __float2bfloat16(std::normal_distribution<float>(0, 1)(gen));
            }
        }
    }, 4.0f);

    extreme("contains_nan", [&](auto& A) {
        uint16_t nan_bits = 0x7FC0;
        for (int i = 0; i < EM * EN; i++) {
            if (gen() % 256 == 0) {
                A[i] = *reinterpret_cast<__nv_bfloat16*>(&nan_bits);
            } else {
                A[i] = __float2bfloat16(std::normal_distribution<float>(0, 1)(gen));
            }
        }
    }, 4.0f);

    printf("\n=== %d / %d tests matched ===\n", pass, total);
    return (pass == total) ? 0 : 1;
}
