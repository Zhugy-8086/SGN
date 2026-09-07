// nested_benchmark.cpp - mkern/nested benchmark（设计档 §二.3：探针 v2 正式版）
//
// 误差-成本双口径（对应方向 A 三证据的正式化）：
//   误差：嵌套（一次 quant + 4 档 dequant）vs 独立（每档一次 SR 量化，独立随机位）
//         的每档 MSE——满幅归一化口径（±2^31·u，对齐 SGN 实际用法）。
//   成本：同 ISA 公平对照（nested 标量 vs 独立标量，隔离"单码多档"结构收益）+
//         生产后端（dispatch 自动选）数字。
// 独立基线的每档随机位独立化：seed 混入档位 tag（计数器式 RNG 的直系用法）。
//
// 计时：steady_clock，预热后取 min（交随机/频率漂移）。
// 构建：见 CMakeLists（nested_benchmark 目标）；运行：build/nested_benchmark.exe。

#include "mkern/nested/nested_api.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

namespace {

constexpr int64_t kN = 1 << 20;
constexpr double kU = 8.0 / 32767.0;
constexpr int kLevels[4] = {4, 8, 16, 32};
constexpr int kReps = 7;

uint64_t g_rng = 0x243F6A8885A308D3ULL;
inline uint64_t rng64() {
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return g_rng >> 33;
}
inline double rng_uniform() {
    return (static_cast<double>(rng64() >> 7) + 0.5) * 0x1.0p-24;
}
inline double rng_gauss() {
    const double u1 = rng_uniform(), u2 = rng_uniform();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

// 规格 a 的单元素随机数（独立基线用；档位 tag 混入 seed 实现档位间独立）
inline double level_uniform(uint64_t seed, int level, int64_t i) {
    uint64_t z = seed + 0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(level) +
                 0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(i);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return static_cast<double>(z >> 11) * 0x1.0p-53;
}

// 独立 SR 量化（探针 independent_codes 口径）：floor(h/s + U)·s，值域直出
void indep_sr(float* out, const float* h, int64_t n, double s,
              uint64_t seed, int level) {
    for (int64_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(h[i]) / s;
        out[i] = static_cast<float>(
            std::floor(x + level_uniform(seed, level, i)) * s);
    }
}

double mse(const float* a, const float* b, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        s += d * d;
    }
    return s / static_cast<double>(n);
}

template <typename F>
double time_min(F&& f) {
    f();  // 预热
    double best = 1e300;
    for (int r = 0; r < kReps; ++r) {
        const auto t0 = std::chrono::steady_clock::now();
        f();
        const auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best) best = ms;
    }
    return best;
}

} // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    using sgn::mkern::nested::nested_quant_i32;
    using sgn::mkern::nested::nested_dequant;
    using sgn::mkern::nested::nested_quant_i32_scalar;
    using sgn::mkern::nested::nested_dequant_scalar;

    std::printf("=== mkern/nested benchmark ===\n");
    std::printf("backend: %s | n=%lld | u=%.4e | regime=full-scale (±2^31·u)\n\n",
                sgn::mkern::nested::active_nested_backend_name(),
                (long long)kN, kU);

    // 满幅归一化信号（A1 复测口径）：gauss·U(0.2,2) → max|h| 映到 ±2^31·u
    std::vector<float> h(static_cast<size_t>(kN));
    {
        std::vector<float> raw(static_cast<size_t>(kN));
        float mx = 0.f;
        for (auto& v : raw) {
            v = static_cast<float>(rng_gauss() * (rng_uniform() * 1.8 + 0.2));
            mx = std::fabs(v) > mx ? std::fabs(v) : mx;
        }
        const float scale = static_cast<float>(2147483648.0 * kU) / mx;
        for (size_t i = 0; i < h.size(); ++i) h[i] = raw[i] * scale;
    }

    std::vector<int64_t> code(static_cast<size_t>(kN));
    std::vector<float> dn(static_cast<size_t>(kN));
    std::vector<float> di(static_cast<size_t>(kN));

    // ---- 误差表（nested vs independent，每档 MSE）----
    double err_n[4] = {0, 0, 0, 0}, err_i[4] = {0, 0, 0, 0};
    nested_quant_i32(code.data(), h.data(), kN, kU, 42);
    for (int j = 0; j < 4; ++j) {
        nested_dequant(dn.data(), code.data(), kN, kU, kLevels[j]);
        err_n[j] = mse(dn.data(), h.data(), kN);
        const double s = kU * static_cast<double>(int64_t(1) << (32 - kLevels[j]));
        indep_sr(di.data(), h.data(), kN, s, 42, kLevels[j]);
        err_i[j] = mse(di.data(), h.data(), kN);
    }
    std::printf("%-22s %13s %13s %13s %13s\n", "bits", "4", "8", "16", "32");
    std::printf("%-22s", "nested MSE");
    for (int j = 0; j < 4; ++j) std::printf(" %12.6g", err_n[j]);
    std::printf("\n%-22s", "independent MSE");
    for (int j = 0; j < 4; ++j) std::printf(" %12.6g", err_i[j]);
    std::printf("\n%-22s", "nested/independent");
    for (int j = 0; j < 4; ++j) std::printf(" %12.4f", err_n[j] / err_i[j]);
    std::printf("\n(full-scale theory 0.5: RTN s^2/12 vs SR s^2/6)\n\n");

    // ---- 成本（同 ISA 公平对照 + 生产后端）----
    const double t_nq_scalar = time_min([&] {
        nested_quant_i32_scalar(code.data(), h.data(), kN, kU, 42);
    });
    double t_nd_scalar = 0.0;
    for (int j = 0; j < 4; ++j) {
        t_nd_scalar += time_min([&] {
            nested_dequant_scalar(dn.data(), code.data(), kN, kU, kLevels[j]);
        });
    }
    const double t_nq_simd = time_min([&] {
        nested_quant_i32(code.data(), h.data(), kN, kU, 42);
    });
    double t_nd_simd = 0.0;
    for (int j = 0; j < 4; ++j) {
        t_nd_simd += time_min([&] {
            nested_dequant(dn.data(), code.data(), kN, kU, kLevels[j]);
        });
    }
    // 独立基线：4 档各一次 SR 量化（标量，探针同构）
    double t_indep = 0.0;
    for (int j = 0; j < 4; ++j) {
        const double s = kU * static_cast<double>(int64_t(1) << (32 - kLevels[j]));
        t_indep += time_min([&] {
            indep_sr(di.data(), h.data(), kN, s, 42, kLevels[j]);
        });
    }

    std::printf("cost (n=%lld, min of %d reps)\n", (long long)kN, kReps);
    std::printf("  nested  scalar: quant %8.3f ms + 4x dequant %8.3f ms = %8.3f ms\n",
                t_nq_scalar, t_nd_scalar, t_nq_scalar + t_nd_scalar);
    std::printf("  nested  %-7s: quant %8.3f ms + 4x dequant %8.3f ms = %8.3f ms\n",
                sgn::mkern::nested::active_nested_backend_name(),
                t_nq_simd, t_nd_simd, t_nq_simd + t_nd_simd);
    std::printf("  indep   scalar: 4x SR quant              = %8.3f ms\n", t_indep);
    std::printf("  structural ratio (indep/nested, both scalar) = %.2fx\n",
                t_indep / (t_nq_scalar + t_nd_scalar));
    return 0;
}
