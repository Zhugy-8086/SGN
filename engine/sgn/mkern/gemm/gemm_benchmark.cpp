// gemm_benchmark.cpp - mkern/gemm 性能基准（roofline 标注 + 同口径对照）
//
// 对照组（同 M/N/K 口径）：
//   1. gemm_i8 tiled（mkern/gemm VNNI 面板内核，pack_b 时间单列不摊入）
//   2. gemm_i8 naive：逐输出 simd::dot8（B 简单转置 Bt[N,K]）——现状代理
//      （msint narrow_dot 每对 (a,c) 一次 dot8 的同型结构，无寄存器 tile）
//   3. gemm_i16 tiled vs naive（逐输出 simd::dot16，B 简单转置）
//   4. float 参照：朴素三重循环（-mavx2 -mfma 编译，代表 float matmul 量级参照，
//      非 ops.cpp 本体——同口径 M/N/K 的量级对照）
//
// roofline：MACs/cycle 经 __rdtsc 实测；理论峰值按 dpbusd 每条 32 MAC（AVX2）/
// 64 MAC（AVX512）× 1 条/周期保守计（实际微架构吞吐可能更高，故标注为「≥」）。
//
// 构建：独立可执行（mkern/gemm/CMakeLists.txt）；本文件 per-file -mavx2 -mfma
// 仅供 float 参照基线向量化（整型原语不受浮点收缩影响）。

#include "mkern/gemm/gemm_api.h"
#include "mkern/simd/simd_api.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace {

double now_ms() {
    return std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

unsigned int g_rng = 0xdeadbeefu;
inline unsigned int rng() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}

struct Row {
    std::string name;
    double macs_per_s;     // MAC/s
    double macs_per_cycle; // 实测（rdtsc）
    double ms;
};

// 简单转置 B[K×N] → Bt[N×K]（naive 基线与 float 参照用的 k 连续布局）
template <typename T>
void transpose_simple(std::vector<T>& Bt, const std::vector<T>& B, int64_t N, int64_t K) {
    Bt.assign(static_cast<size_t>(N * K), T{0});
    for (int64_t k = 0; k < K; ++k) {
        for (int64_t n = 0; n < N; ++n) {
            Bt[n * K + k] = B[k * N + n];
        }
    }
}

Row bench_i8_tiled(int64_t M, int64_t N, int64_t K, int iters) {
    using namespace sgn::mkern::gemm;
    std::vector<uint8_t> A(static_cast<size_t>(M * K));
    std::vector<int8_t> B(static_cast<size_t>(K * N));
    for (auto& v : A) v = static_cast<uint8_t>(rng());
    for (auto& v : B) v = static_cast<int8_t>(rng());
    std::vector<int8_t> Bp(static_cast<size_t>(pack_b_i8_bytes(N, K)), 0);
    const double t_pk0 = now_ms();
    pack_b_i8(Bp.data(), B.data(), N, K);
    const double t_pack = now_ms() - t_pk0;
    std::vector<int32_t> C(static_cast<size_t>(M * N), 0);
    for (int i = 0; i < 3; ++i) gemm_i8(C.data(), A.data(), Bp.data(), M, N, K, false);
    (void)t_pack;  // pack 时间单独打印，不摊入内核计时
#if defined(__x86_64__) || defined(_M_X64)
    unsigned long long cyc0 = __rdtsc();
#endif
    const double t0 = now_ms();
    for (int i = 0; i < iters; ++i) gemm_i8(C.data(), A.data(), Bp.data(), M, N, K, false);
    const double t1 = now_ms();
#if defined(__x86_64__) || defined(_M_X64)
    unsigned long long cyc1 = __rdtsc();
    const double cyc = static_cast<double>(cyc1 - cyc0) / iters;
#endif
    const double macs = static_cast<double>(M) * N * K;
    Row r;
    r.name = "gemm_i8 tiled";
    r.ms = (t1 - t0) / iters;
    r.macs_per_s = macs / ((t1 - t0) / 1000.0);
#if defined(__x86_64__) || defined(_M_X64)
    r.macs_per_cycle = macs / cyc;
#else
    r.macs_per_cycle = 0;
#endif
    (void)t_pack;
    return r;
}

Row bench_i8_naive(int64_t M, int64_t N, int64_t K, int iters) {
    std::vector<uint8_t> A(static_cast<size_t>(M * K));
    std::vector<int8_t> B(static_cast<size_t>(K * N));
    for (auto& v : A) v = static_cast<uint8_t>(rng());
    for (auto& v : B) v = static_cast<int8_t>(rng());
    std::vector<int8_t> Bt;
    transpose_simple(Bt, B, N, K);
    std::vector<int32_t> C(static_cast<size_t>(M * N), 0);
    auto run = [&]() {
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                C[m * N + n] = static_cast<int32_t>(
                    sgn::simd::dot8(A.data() + m * K, Bt.data() + n * K, K));
            }
        }
    };
    run();
    const double t0 = now_ms();
    for (int i = 0; i < iters; ++i) run();
    const double t1 = now_ms();
    const double macs = static_cast<double>(M) * N * K;
    Row r;
    r.name = "gemm_i8 naive(dot8)";
    r.ms = (t1 - t0) / iters;
    r.macs_per_s = macs / ((t1 - t0) / 1000.0);
    r.macs_per_cycle = 0;
    return r;
}

Row bench_i16_tiled(int64_t M, int64_t N, int64_t K, int iters) {
    using namespace sgn::mkern::gemm;
    std::vector<int16_t> A(static_cast<size_t>(M * K));
    std::vector<int16_t> B(static_cast<size_t>(K * N));
    for (auto& v : A) v = static_cast<int16_t>(rng());
    for (auto& v : B) v = static_cast<int16_t>(rng());
    std::vector<int64_t> C(static_cast<size_t>(M * N), 0);
    for (int i = 0; i < 3; ++i) gemm_i16(C.data(), A.data(), B.data(), M, N, K, false);
#if defined(__x86_64__) || defined(_M_X64)
    unsigned long long cyc0 = __rdtsc();
#endif
    const double t0 = now_ms();
    for (int i = 0; i < iters; ++i) gemm_i16(C.data(), A.data(), B.data(), M, N, K, false);
    const double t1 = now_ms();
#if defined(__x86_64__) || defined(_M_X64)
    unsigned long long cyc1 = __rdtsc();
    const double cyc = static_cast<double>(cyc1 - cyc0) / iters;
#endif
    const double macs = static_cast<double>(M) * N * K;
    Row r;
    r.name = "gemm_i16 tiled";
    r.ms = (t1 - t0) / iters;
    r.macs_per_s = macs / ((t1 - t0) / 1000.0);
#if defined(__x86_64__) || defined(_M_X64)
    r.macs_per_cycle = macs / cyc;
#endif
    return r;
}

Row bench_i16_naive(int64_t M, int64_t N, int64_t K, int iters) {
    std::vector<int16_t> A(static_cast<size_t>(M * K));
    std::vector<int16_t> B(static_cast<size_t>(K * N));
    for (auto& v : A) v = static_cast<int16_t>(rng());
    for (auto& v : B) v = static_cast<int16_t>(rng());
    std::vector<int16_t> Bt;
    transpose_simple(Bt, B, N, K);
    std::vector<int64_t> C(static_cast<size_t>(M * N), 0);
    auto run = [&]() {
        for (int64_t m = 0; m < M; ++m) {
            for (int64_t n = 0; n < N; ++n) {
                C[m * N + n] =
                    sgn::simd::dot16(A.data() + m * K, Bt.data() + n * K, K);
            }
        }
    };
    run();
    const double t0 = now_ms();
    for (int i = 0; i < iters; ++i) run();
    const double t1 = now_ms();
    const double macs = static_cast<double>(M) * N * K;
    Row r;
    r.name = "gemm_i16 naive(dot16)";
    r.ms = (t1 - t0) / iters;
    r.macs_per_s = macs / ((t1 - t0) / 1000.0);
    r.macs_per_cycle = 0;
    return r;
}

Row bench_f32_ref(int64_t M, int64_t N, int64_t K, int iters) {
    std::vector<float> A(static_cast<size_t>(M * K));
    std::vector<float> B(static_cast<size_t>(K * N));
    for (auto& v : A) v = static_cast<float>(rng() % 1000) / 512.0f;
    for (auto& v : B) v = static_cast<float>(rng() % 1000) / 512.0f;
    std::vector<float> Bt;
    transpose_simple(Bt, B, N, K);
    std::vector<float> C(static_cast<size_t>(M * N), 0.0f);
    auto run = [&]() {
        for (int64_t m = 0; m < M; ++m) {
            const float* a = A.data() + m * K;
            for (int64_t n = 0; n < N; ++n) {
                const float* b = Bt.data() + n * K;
                float s = 0.0f;
                for (int64_t k = 0; k < K; ++k) s += a[k] * b[k];
                C[m * N + n] = s;
            }
        }
    };
    run();
    const double t0 = now_ms();
    for (int i = 0; i < iters; ++i) run();
    const double t1 = now_ms();
    const double macs = static_cast<double>(M) * N * K;
    Row r;
    r.name = "f32 ref(朴素三重循环)";
    r.ms = (t1 - t0) / iters;
    r.macs_per_s = macs / ((t1 - t0) / 1000.0);
    r.macs_per_cycle = 0;
    return r;
}

} // namespace

int main() {
    std::printf("=== mkern/gemm benchmark ===\n");
    std::printf("gemm backend: %s | simd backend: %s\n",
                sgn::mkern::gemm::active_gemm_backend_name(),
                sgn::simd::active_backend_name());
    std::printf("roofline 参考：AVX2-VNNI dpbusd 每条 32 MAC，AVX512-VNNI 每条 64 MAC"
                "（按 1 条/周期保守计，实测 MACs/cycle 为下界证据）\n\n");

    struct Shape { int64_t M, N, K; const char* tag; int iters; };
    const Shape shapes[] = {
        {64, 64, 576, "小(L1 内)", 200},
        {256, 256, 576, "中(L2)", 30},
        {1024, 1024, 4096, "大(出 L2)", 2},
    };

    for (const auto& sh : shapes) {
        const double macs = static_cast<double>(sh.M) * sh.N * sh.K;
        std::printf("M=%lld N=%lld K=%lld  (%s, %.2f GMAC/次)\n",
                    static_cast<long long>(sh.M), static_cast<long long>(sh.N),
                    static_cast<long long>(sh.K), sh.tag, macs / 1e9);
        Row rows[] = {
            bench_i8_tiled(sh.M, sh.N, sh.K, sh.iters),
            bench_i8_naive(sh.M, sh.N, sh.K, sh.iters),
            bench_i16_tiled(sh.M, sh.N, sh.K, sh.iters),
            bench_i16_naive(sh.M, sh.N, sh.K, sh.iters),
            bench_f32_ref(sh.M, sh.N, sh.K, sh.iters),
        };
        for (const auto& r : rows) {
            std::printf("  %-26s %10.2f ms   %12.2f GMAC/s",
                        r.name.c_str(), r.ms, r.macs_per_s / 1e9);
            if (r.macs_per_cycle > 0) {
                std::printf("   %.2f MACs/cycle", r.macs_per_cycle);
            }
            std::printf("\n");
        }
        if (rows[0].macs_per_s > 0 && rows[1].macs_per_s > 0) {
            std::printf("  → tiled/naive(i8) = %.2fx\n",
                        rows[0].macs_per_s / rows[1].macs_per_s);
        }
        if (rows[2].macs_per_s > 0 && rows[3].macs_per_s > 0) {
            std::printf("  → tiled/naive(i16) = %.2fx\n",
                        rows[2].macs_per_s / rows[3].macs_per_s);
        }
        std::printf("\n");
    }
    return 0;
}
