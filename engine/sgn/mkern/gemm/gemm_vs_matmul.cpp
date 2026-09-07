// gemm_vs_matmul.cpp - mkern/gemm_i8 vs ops.cpp matmul 本体同口径对比（欠账②清偿）
//
// "matmul 本体" = sgn_autograd::matmul_forward（dispatch → matmul_fwd_avx2_fma，
// 引擎现役 float GEMM 内核：AVX2+FMA + OpenMP，Tensor 级入口含输出分配）。
// 口径：相同 M/N/K，MACs/s（每对元素一次乘加）；两侧均单线程（OMP_NUM_THREADS=1，
// 与 benchmark_mnist8 的口径结论一致——小 batch 下多线程受宿主省电状态干扰）。
// 各对标自身保守峰值：f32 AVX2-FMA ~16 MAC/cycle（2 FMA 口 × 8 lane）；
// int8 AVX2-VNNI ≥32 MAC/cycle（dpbusd 32 MAC/条，实测 >1 条/周期）。
//
// 诚实注记：int8 与 float 是不同数值域——本对比回答"量化 GEMM 相对现役 float
// 内核的吞吐量级"（R3 验收的"同口径对比"），不是精度替代证明。matmul_forward
// 每次 call 含 Tensor 输出分配（op 语义一部分），gemm_i8 写调用方 C（分配在外）。
//
// 构建：gemm standalone CMakeLists 的 gemm_vs_matmul target（链接 ops/tensor/registry）。
// 运行：cd mkern/gemm/build && ./gemm_vs_matmul.exe

#include "mkern/gemm/gemm_api.h"
#include "autograd/ops.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(_OPENMP)
#include <omp.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

int main() {
#if defined(_OPENMP)
    omp_set_num_threads(1);   // 两侧同口径单线程（见文件头注）
#endif
    printf("=== gemm_i8 (mkern, int8-VNNI) vs matmul_forward (ops.cpp, float-FMA) ===\n");
    printf("口径: 同 M/N/K, MACs/s, 两侧单线程 (OMP_NUM_THREADS=1)\n");
    printf("注: matmul_forward 每次 call 含输出 Tensor 分配 (op 语义); gemm_i8 C 在计时区外\n\n");

    struct Shape { int64_t M, N, K; const char* tag; int iters; };
    const Shape shapes[] = {
        {64, 64, 576, "小(L1 内)", 200},
        {256, 256, 576, "中(L2)", 30},
        {1024, 1024, 4096, "大(出 L2)", 3},
    };

    for (const auto& sh : shapes) {
        const double macs = static_cast<double>(sh.M) * sh.N * sh.K;
        printf("M=%lld N=%lld K=%lld (%s, %.2f GMAC/次)\n",
               static_cast<long long>(sh.M), static_cast<long long>(sh.N),
               static_cast<long long>(sh.K), sh.tag, macs / 1e9);

        // ---- gemm_i8 ----
        std::vector<uint8_t> A8(static_cast<size_t>(sh.M * sh.K));
        std::vector<int8_t> B8(static_cast<size_t>(sh.K * sh.N));
        for (auto& v : A8) v = static_cast<uint8_t>(rand());
        for (auto& v : B8) v = static_cast<int8_t>(rand());
        std::vector<int8_t> Bp(static_cast<size_t>(
            sgn::mkern::gemm::pack_b_i8_bytes(sh.N, sh.K)), 0);
        sgn::mkern::gemm::pack_b_i8(Bp.data(), B8.data(), sh.N, sh.K);
        std::vector<int32_t> C8(static_cast<size_t>(sh.M * sh.N), 0);
        auto run_gemm = [&]() {
            sgn::mkern::gemm::gemm_i8(C8.data(), A8.data(), Bp.data(),
                                      sh.M, sh.N, sh.K, false);
        };
        for (int i = 0; i < 3; ++i) run_gemm();
#if defined(__x86_64__) || defined(_M_X64)
        unsigned long long c0 = __rdtsc();
#endif
        double t0 = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        for (int i = 0; i < sh.iters; ++i) run_gemm();
        double t1 = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
#if defined(__x86_64__) || defined(_M_X64)
        unsigned long long c1 = __rdtsc();
        const double cyc = static_cast<double>(c1 - c0) / sh.iters;
#endif
        const double gemm_ms = (t1 - t0) / sh.iters;
        const double gemm_macs = macs / ((t1 - t0) / 1000.0);

        // ---- matmul_forward 本体（float）----
        std::vector<float> Af(static_cast<size_t>(sh.M * sh.K));
        std::vector<float> Bf(static_cast<size_t>(sh.K * sh.N));
        for (auto& v : Af) v = static_cast<float>(rand() % 2000 - 1000) / 512.0f;
        for (auto& v : Bf) v = static_cast<float>(rand() % 2000 - 1000) / 512.0f;
        auto run_matmul = [&]() {
            sgn_autograd::Tensor a({sh.M, sh.K}, Af.data());
            sgn_autograd::Tensor b({sh.K, sh.N}, Bf.data());
            sgn_autograd::Tensor c = sgn_autograd::matmul_forward(a, b);
            return c.data()[0];  // 防 DCE
        };
        volatile float sink = 0.0f;
        for (int i = 0; i < 3; ++i) sink = sink + run_matmul();
        double m0 = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        for (int i = 0; i < sh.iters; ++i) sink = sink + run_matmul();
        double m1 = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
        const double mm_ms = (m1 - m0) / sh.iters;
        const double mm_macs = macs / ((m1 - m0) / 1000.0);
        (void)sink;

        printf("  gemm_i8 tiled        %10.2f ms  %10.2f GMAC/s",
               gemm_ms, gemm_macs / 1e9);
#if defined(__x86_64__) || defined(_M_X64)
        printf("  %.1f MACs/cycle", macs / cyc);
#endif
        printf("\n");
        printf("  matmul_forward(f32)  %10.2f ms  %10.2f GMAC/s\n", mm_ms, mm_macs / 1e9);
        printf("  → int8/float 本体 = %.2fx（吞吐量级对比，非精度替代）\n\n",
               gemm_macs / mm_macs);
    }
    return 0;
}
