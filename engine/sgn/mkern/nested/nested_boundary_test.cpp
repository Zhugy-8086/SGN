// nested_boundary_test.cpp - mkern/nested 边界/逐位对拍测试（验收 N1-N4）
//
// 验收判据（预先登记，nested_quant设计档_2026_09_05.md §三）：
//   N1 同 seed 同输入码字逐位一致（kBitExact）：确定性、dispatch vs 标量锚点 vs
//      【独立重写参考实现】三方对拍、seed 敏感性、码字 int32 域；
//   N2 格嵌套不变量：dequant_b ∈ u·2^(32−b)·ℤ 且为 dequant_32 的最近粗格点
//      （≤ 半格 + 对 ±1 邻格最优性 + tie 半偶规则）、截位可组合、level 32 恒等、
//      h=0 吸收态（设计档 §四）；
//   N3 切换一致性（探针口径复现，比值 > 1e5 或嵌套零跳变）+ A1 归一化口径
//      嵌套不劣于独立 SR（逐元素支配 ⇒ 聚合 MSE ≤，理论 0.5）+ A1 原始口径
//      b=16 粗档 4×（修订判据 2026-09-05 拍板：原预先登记 64× 与冻结契约不相容，
//      见执行记录 §四；原始口径诊断值 0.119 ≈ 8.4×）；
//   N4 boundary：n=0、非法 level 无操作、域外饱和、往返误差界
//      （|dequant_b − h| ≤ s_b/2 + u，域内输入）。
//
// 统计段（N3）为测试 LCG 下的确定性复现；N3 的 Python 探针正本
// （独立 Python 探针 run_nested_probe.py，seeded）复现 A2 = 1,575,676×。
//
// 构建：独立可执行（见 mkern/nested/CMakeLists.txt），不依赖 pybind11/Python；
// UBSan 变体同一源（-fsanitize=undefined，N4）。

#include "mkern/nested/nested_api.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// 标量锚点手动声明（同 gemm_boundary_test 纪律：锚点定义于实现文件、不进公共
// 接口头；必须放全局作用域具名 namespace，放匿名 namespace 会构造假命名空间）。
namespace sgn::mkern::nested {
void nested_quant_i32_scalar(int64_t*, const float*, int64_t, float, uint64_t);
void nested_dequant_scalar(float*, const int64_t*, int64_t, float, int);
} // namespace sgn::mkern::nested

namespace {

int g_total = 0, g_failed = 0;

#define CHECK(cond, msg)                                                        \
    do {                                                                        \
        ++g_total;                                                              \
        if (!(cond)) {                                                          \
            ++g_failed;                                                         \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, msg);           \
        }                                                                       \
    } while (0)

// ---- 测试 LCG（输入生成专用，与契约 RNG 无关）：64 位 LCG + PCG 式输出 ----
uint64_t g_rng = 0x243F6A8885A308D3ULL;
inline uint64_t rng64() {
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return g_rng >> 33;
}
inline uint32_t rng32() { return static_cast<uint32_t>(rng64() >> 32); }
// (0,1) 均匀——rng64() 返回 31 位（state>>33），取高 24 位；+0.5 避开 0
inline double rng_uniform() {
    return (static_cast<double>(rng64() >> 7) + 0.5) * 0x1.0p-24;
}
inline double rng_gauss() {  // Box-Muller
    const double u1 = rng_uniform(), u2 = rng_uniform();
    return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

// ----------------------------------------------------------------------------
// 独立重写参考实现（对拍正本）：按 nested_api.h 冻结规格另行实现一遍，
// 与 scalar.cpp 无共享代码——两侧一致才能证明"规格被正确实现"而非"实现自洽"。
namespace ref {

inline double uniform(uint64_t seed, int64_t i) {   // 规格 a
    uint64_t z = seed + 0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(i);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z ^= z >> 31;
    return static_cast<double>(z >> 11) * 0x1.0p-53;
}

void quant(int64_t* code, const float* h, int64_t n, float u, uint64_t seed) {
    for (int64_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(h[i]) / static_cast<double>(u);
        int64_t v;
        if (x >= 2147483647.0)      v = 2147483647;    // 规格 c
        else if (x <= -2147483648.0) v = -2147483648;
        else {
            const double fl = std::floor(x);
            v = static_cast<int64_t>(fl);
            if (uniform(seed, i) < x - fl) ++v;        // 规格 b
        }
        code[i] = v;
    }
}

// 规格 d 的整数域 RTN（half-to-even）。m ∈ [0,28]。
int64_t rtn(int64_t I, int m) {
    int64_t q = I >> m;
    if (m > 0) {
        const int64_t r = I - (q << m);
        const int64_t half = int64_t(1) << (m - 1);
        if (r > half || (r == half && (q & 1))) ++q;
    }
    return q << m;
}

void dequant(float* out, const int64_t* code, int64_t n, float u, int level) {
    if (level != 4 && level != 8 && level != 16 && level != 32) return;
    const int m = 32 - level;
    for (int64_t i = 0; i < n; ++i) {
        // 与 scalar.cpp 同规格：ldexp 精确值重建（int64 域无溢出）
        const int64_t I = code[i];
        int64_t q = I >> m;
        if (m > 0) {
            const int64_t r = I - (q << m);
            const int64_t half = int64_t(1) << (m - 1);
            if (r > half || (r == half && (q & 1))) ++q;
        }
        out[i] = static_cast<float>(std::ldexp(static_cast<double>(q), m) *
                                    static_cast<double>(u));
    }
}

} // namespace ref

// 独立 SR 量化 MSE（N3 独立基线，探针 independent_codes 口径）：
// floor(h/s + U)·s，U 取自测试 LCG（与契约 RNG 无关——独立基线本来就是另一次
// 独立随机化）。
double indep_sr_mse(const std::vector<float>& h, double s) {
    double se = 0.0;
    for (float v : h) {
        const double x = static_cast<double>(v) / s;
        const double q = std::floor(x + rng_uniform());
        se += (q * s - static_cast<double>(v)) * (q * s - static_cast<double>(v));
    }
    return se / static_cast<double>(h.size());
}

constexpr double kU = 8.0 / 32767.0;              // 探针最细格距
constexpr int64_t kProbeN = 256, kProbeTrials = 400;

// 探针口径信号：h = gauss(N)·U(0.2, 2)
std::vector<float> probe_signal() {
    std::vector<float> h(static_cast<size_t>(kProbeN));
    for (auto& v : h)
        v = static_cast<float>(rng_gauss() * (rng_uniform() * 1.8 + 0.2));
    return h;
}

double mse(const std::vector<float>& a, const std::vector<float>& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) {
        const double d = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        s += d * d;
    }
    return s / static_cast<double>(a.size());
}

} // namespace

// ----------------------------------------------------------------------------
// N1：bit-exact / 确定性
static void test_n1_bitexact() {
    using sgn::mkern::nested::nested_quant_i32;
    using sgn::mkern::nested::nested_quant_i32_scalar;
    using sgn::mkern::nested::nested_dequant;
    using sgn::mkern::nested::nested_dequant_scalar;

    const int64_t n = 1000;
    std::vector<float> h(static_cast<size_t>(n));
    for (auto& v : h) {
        // 宽幅输入：对数均匀幅值 × 随机符号（覆盖饱和域/常规域/次格距）
        const double mag = std::pow(10.0, rng_uniform() * 14.0 - 4.0); // 1e-4..1e10
        v = static_cast<float>(mag * (rng32() % 2 ? 1.0 : -1.0));
    }
    const std::vector<float> us = {kU, 1e-6f, 1e-3f, 1.0f, 1e3f, 1e6f};

    for (float u : us) {
        for (uint64_t seed : {0ULL, 1ULL, 0xDEADBEEFCAFEBABEULL}) {
            std::vector<int64_t> c1(static_cast<size_t>(n), -99);
            std::vector<int64_t> c2(static_cast<size_t>(n), -99);
            std::vector<int64_t> c3(static_cast<size_t>(n), -99);
            nested_quant_i32(c1.data(), h.data(), n, u, seed);          // dispatch
            nested_quant_i32_scalar(c2.data(), h.data(), n, u, seed);   // 锚点
            ref::quant(c3.data(), h.data(), n, u, seed);                // 独立参考
            CHECK(std::memcmp(c1.data(), c2.data(), sizeof(int64_t) * static_cast<size_t>(n)) == 0,
                  "N1 dispatch vs scalar anchor mismatch");
            CHECK(std::memcmp(c1.data(), c3.data(), sizeof(int64_t) * static_cast<size_t>(n)) == 0,
                  "N1 primitive vs independent reference mismatch");
            bool in_range = true;   // 码字恒在 int32 域（规格 c）
            for (auto v : c1)
                in_range = in_range && v >= -2147483648LL && v <= 2147483647LL;
            CHECK(in_range, "N1 codeword out of int32 range");

            for (int level : {4, 8, 16, 32, 5, 0, -4, 33}) {  // dequant 三方对拍
                std::vector<float> o1(static_cast<size_t>(n), -99.f);
                std::vector<float> o2(static_cast<size_t>(n), -99.f);
                std::vector<float> o3(static_cast<size_t>(n), -99.f);
                nested_dequant(o1.data(), c1.data(), n, u, level);
                nested_dequant_scalar(o2.data(), c1.data(), n, u, level);
                ref::dequant(o3.data(), c1.data(), n, u, level);
                CHECK(std::memcmp(o1.data(), o2.data(), sizeof(float) * static_cast<size_t>(n)) == 0,
                      "N1 dequant dispatch vs scalar anchor mismatch");
                CHECK(std::memcmp(o1.data(), o3.data(), sizeof(float) * static_cast<size_t>(n)) == 0,
                      "N1 dequant vs independent reference mismatch");
            }
        }
        {   // 确定性：同 seed 重跑逐位一致
            std::vector<int64_t> c1(static_cast<size_t>(n)), c2(static_cast<size_t>(n));
            nested_quant_i32(c1.data(), h.data(), n, u, 42);
            nested_quant_i32(c2.data(), h.data(), n, u, 42);
            CHECK(std::memcmp(c1.data(), c2.data(), sizeof(int64_t) * static_cast<size_t>(n)) == 0,
                  "N1 same-seed rerun not bit-identical");
        }
    }
    {   // seed 敏感性：不同 seed 宽幅随机输入下码字不同（seed 未进随机流则恒同）
        std::vector<int64_t> c1(static_cast<size_t>(n)), c2(static_cast<size_t>(n));
        nested_quant_i32(c1.data(), h.data(), n, kU, 7);
        nested_quant_i32(c2.data(), h.data(), n, kU, 8);
        CHECK(std::memcmp(c1.data(), c2.data(), sizeof(int64_t) * static_cast<size_t>(n)) != 0,
              "N1 different seeds produced identical codewords (seed inert)");
    }
}

// ----------------------------------------------------------------------------
// N1 扩展：SIMD 后端直连对拍（按 active 后端名判定本机 ISA，机器安全——
// scalar(forced) 或非 x86 时跳过；avx512 蕴含 avx2 可用，两者都直连测）
static void test_n1_cross_backend() {
    using namespace sgn::mkern::nested;
    const char* name = active_nested_backend_name();
    const bool has_avx2   = std::strcmp(name, "avx2") == 0 ||
                            std::strcmp(name, "avx512") == 0;
    const bool has_avx512 = std::strcmp(name, "avx512") == 0;
    if (!has_avx2 && !has_avx512) {
        std::printf("  [N1-x] active backend '%s': no SIMD backend to compare\n",
                    name);
        return;
    }
    const int64_t n = 997;   // 非 4/8 倍数：向量体 + 尾路径同测
    std::vector<float> h(static_cast<size_t>(n));
    for (auto& v : h) {
        const double mag = std::pow(10.0, rng_uniform() * 12.0 - 3.0);
        v = static_cast<float>(mag * (rng32() % 2 ? 1.0 : -1.0));
    }
    const std::vector<float> us = {kU, 1e-3f, 1.0f, 1e3f};
    std::vector<int64_t> cs(static_cast<size_t>(n)), cx(static_cast<size_t>(n));
    std::vector<float> os(static_cast<size_t>(n)), ox(static_cast<size_t>(n));
    for (int sel = 0; sel < 2; ++sel) {
        if (sel == 0 && !has_avx2) continue;
        if (sel == 1 && !has_avx512) continue;
        const char* bname = sel == 0 ? "avx2" : "avx512";
        bool q_ok = true, d_ok = true;
        for (float u : us) {
            for (uint64_t seed : {0ULL, 0xDEADBEEFCAFEBABEULL}) {
                nested_quant_i32_scalar(cs.data(), h.data(), n, u, seed);
                if (sel == 0) nested_quant_i32_avx2(cx.data(), h.data(), n, u, seed);
                else          nested_quant_i32_avx512(cx.data(), h.data(), n, u, seed);
                if (std::memcmp(cs.data(), cx.data(),
                                sizeof(int64_t) * static_cast<size_t>(n)) != 0)
                    q_ok = false;
                for (int level : {4, 8, 16, 32}) {
                    nested_dequant_scalar(os.data(), cs.data(), n, u, level);
                    if (sel == 0)
                        nested_dequant_avx2(ox.data(), cs.data(), n, u, level);
                    else
                        nested_dequant_avx512(ox.data(), cs.data(), n, u, level);
                    if (std::memcmp(os.data(), ox.data(),
                                    sizeof(float) * static_cast<size_t>(n)) != 0)
                        d_ok = false;
                }
            }
        }
        CHECK(q_ok, "N1-x SIMD quant vs scalar anchor mismatch");
        CHECK(d_ok, "N1-x SIMD dequant vs scalar anchor mismatch");
        std::printf("  [N1-x] %s vs scalar anchor: bit-exact (quant+dequant, "
                    "n=%lld incl. tail)\n", bname, (long long)n);
    }
}

// ----------------------------------------------------------------------------
// N2：格嵌套不变量（dequant_b ∈ u·2^(32−b)·ℤ 且为 dequant_32 最近粗格点）
static void test_n2_nesting() {
    using sgn::mkern::nested::nested_quant_i32;
    using sgn::mkern::nested::nested_dequant;

    const int64_t n = 2000;
    std::vector<float> h(static_cast<size_t>(n));
    for (auto& v : h) {
        const double mag = std::pow(10.0, rng_uniform() * 10.0 - 4.0); // 1e-4..1e6
        v = static_cast<float>(mag * (rng32() % 2 ? 1.0 : -1.0));
    }
    const std::vector<float> us = {kU, 1e-3f, 1.0f};
    std::vector<int64_t> code(static_cast<size_t>(n));
    std::vector<float> d32(static_cast<size_t>(n));
    std::vector<float> db(static_cast<size_t>(n));

    for (float u : us) {
        nested_quant_i32(code.data(), h.data(), n, u, 20260905ULL);
        nested_dequant(d32.data(), code.data(), n, u, 32);

        {   // level 32 恒等：out == I·u（同一整数重建表达式，逐位相等）
            bool ok = true;
            for (int64_t i = 0; i < n && ok; ++i) {
                const float expect = static_cast<float>(
                    static_cast<double>(code[static_cast<size_t>(i)]) *
                    static_cast<double>(u));
                ok = (d32[static_cast<size_t>(i)] == expect);
            }
            CHECK(ok, "N2 dequant_32 not exact I*u identity");
        }
        {   // h=0 吸收态：量化 0 → 码字 0（任意 seed）；码字 0 → 全档 dequant 0
            bool ok = true;
            for (uint64_t seed : {0ULL, 42ULL, 0xFFFFFFFFFFFFFFFFULL}) {
                const float zero = 0.0f;
                int64_t c = -1;
                nested_quant_i32(&c, &zero, 1, u, seed);
                ok = ok && (c == 0);
            }
            float o[4] = {-1.f, -1.f, -1.f, -1.f};
            const int64_t zero_code = 0;
            for (int li = 0; li < 4; ++li) {
                nested_dequant(&o[li], &zero_code, 1, u, 4 << li);
            }
            ok = ok && o[0] == 0.f && o[1] == 0.f && o[2] == 0.f && o[3] == 0.f;
            CHECK(ok, "N2 h=0 absorption violated");
        }

        for (int level : {4, 8, 16}) {
            const int m = 32 - level;
            const int64_t cell = int64_t(1) << m;
            const int64_t half = int64_t(1) << (m - 1);
            nested_dequant(db.data(), code.data(), n, u, level);

            bool lattice_ok = true, nearest_ok = true, optimal_ok = true,
                 tie_parity_ok = true, compose_ok = true;
            int64_t ties = 0;
            for (int64_t i = 0; i < n; ++i) {
                const int64_t I = code[static_cast<size_t>(i)];
                const int64_t trunc = ref::rtn(I, m);      // 独立 RTN 截位
                // (i) 格点隶属：out == (float)(trunc·u)，trunc = q·2^m ∈ 2^m·ℤ
                const float expect_val = static_cast<float>(
                    static_cast<double>(trunc) * static_cast<double>(u));
                if (db[static_cast<size_t>(i)] != expect_val) lattice_ok = false;
                // (ii) 最近粗格点：|I − trunc| ≤ 2^(m−1)（tie 取等）
                const int64_t q = trunc >> m;
                const int64_t dist = I - trunc;
                const int64_t adist = dist < 0 ? -dist : dist;
                if (adist > half) nearest_ok = false;
                // (iii) 对 ±1 邻格最优性（唯一最近或 tie 并列）
                const int64_t d_m = I - (trunc - cell);
                const int64_t d_p = I - (trunc + cell);
                const int64_t adm = d_m < 0 ? -d_m : d_m;
                const int64_t adp = d_p < 0 ? -d_p : d_p;
                if (adist > adm || adist > adp) optimal_ok = false;
                if (adist == half) {   // tie：half-to-even 冻结规则（取偶）
                    ++ties;
                    if (q & 1) tie_parity_ok = false;
                }
                // (iv) 截位组合界：RTN(RTN(I,2^16),2^m) 与直接 RTN(I,2^m) 至多差
                //   一个粗格（中间截位恰落粗格中点 tie 时可差一格——任何确定性
                //   tie 规则不可避免；契约 dequant 是从 I 直接一步定义，不依赖
                //   多跳组合，故这里断言界而非恒等）。
                if (ref::rtn(ref::rtn(I, 16), m) - trunc > cell ||
                    trunc - ref::rtn(ref::rtn(I, 16), m) > cell) compose_ok = false;
            }
            CHECK(lattice_ok, "N2 dequant_b not on lattice u*2^(32-b)*Z");
            CHECK(nearest_ok, "N2 dequant_b farther than half coarse cell");
            CHECK(optimal_ok, "N2 dequant_b not nearest among +/-1 neighbor grid points");
            CHECK(tie_parity_ok, "N2 tie not resolved half-to-even");
            CHECK(compose_ok, "N2 truncation composition broken (prefix property)");
            if (ties > 0) {
                std::printf("  [N2] level=%d u=%g: %lld tie cases covered\n",
                            level, static_cast<double>(u), (long long)ties);
            }
        }
    }

    // 任意 int64 码字：dequant 良定义。分两段——
    // (a) RTN 值在 int64 可表示域（I ≤ INT64_MAX − 2^28；INT64_MIN 恰可表示）：
    //     整型域最近性全检查；
    // (b) 极端码字（RTN 值超 int64，q·2^m 乘积不可用）：总和性 + ldexp 值域检查。
    {
        std::vector<int64_t> wild;
        wild.push_back(-9223372036854775807LL - 1);   // INT64_MIN：RTN 恰可表示
        wild.push_back(-2147483648LL);
        wild.push_back(2147483647LL);
        wild.push_back(-65536LL);
        wild.push_back(65536LL);
        wild.push_back(-1LL);
        wild.push_back(1LL);
        wild.push_back(0LL);
        for (int64_t k = -50; k <= 50; ++k) {
            for (int m : {28, 24, 16}) {
                wild.push_back(k * (int64_t(1) << m) + (int64_t(1) << (m - 1)));
                wild.push_back(k * (int64_t(1) << m));
            }
        }
    const int64_t nw = static_cast<int64_t>(wild.size());
    for (int level : {4, 8, 16, 32}) {
        const int m = 32 - level;
        const int64_t half = m > 0 ? int64_t(1) << (m - 1) : 0;
        std::vector<float> out(wild.size(), 0.f);
        nested_dequant(out.data(), wild.data(), nw, 1.0f, level);
        bool ok = true;
            for (int64_t i = 0; i < nw && ok; ++i) {
                const int64_t trunc = ref::rtn(wild[static_cast<size_t>(i)], m);
                const float expect = static_cast<float>(
                    static_cast<double>(trunc) * 1.0);
                if (out[static_cast<size_t>(i)] != expect) ok = false;
                if (m > 0) {
                    const int64_t d = wild[static_cast<size_t>(i)] - trunc;
                    const int64_t ad = d < 0 ? -d : d;
                    if (ad > half) ok = false;
                }
            }
            CHECK(ok, "N2 wild int64 codewords: dequant not RTN-nearest");
        }
        // (b) 极端码字：INT64_MAX 的 RTN（2^63）在 int64 不可表示——原语经
        //     ldexp double 域重建值，良定义无 UB；测试用同域公式算期望值。
        {
            const int64_t extremes[2] = {9223372036854775807LL,
                                         -9223372036854775807LL - 1};
            bool ok = true;
            for (int level : {4, 8, 16, 32}) {
                const int m = 32 - level;
                for (int ei = 0; ei < 2; ++ei) {
                    const int64_t I = extremes[ei];
                    float o = 0.f;
                    nested_dequant(&o, &I, 1, 1.0f, level);
                    int64_t q = I >> m;
                    if (m > 0) {
                        const int64_t r = I - (q << m);
                        const int64_t half2 = int64_t(1) << (m - 1);
                        if (r > half2 || (r == half2 && (q & 1))) ++q;
                    }
                    const float expect = static_cast<float>(
                        std::ldexp(static_cast<double>(q), m) * 1.0);
                    if (o != expect) ok = false;
                }
            }
            CHECK(ok, "N2 extreme codewords: dequant not ldexp-domain RTN value");
        }
    }
}

// ----------------------------------------------------------------------------
// N3：切换一致性 + A1 归一化口径（统计段，测试 LCG 下确定性）
static void test_n3_switching_and_a1() {
    using sgn::mkern::nested::nested_quant_i32;
    using sgn::mkern::nested::nested_dequant;

    // ---- A2 切换一致性（探针 regime：raw gauss·U(0.2,2)，u = 8/32767）----
    //   nested |d16 − d8|² vs 独立两次 SR |s16 − s8|²：嵌套同码截位确定性 ⇒
    //   粗格独立 SR 偶发跳格而嵌套截位永不跳格 ⇒ 比值 > 1e5（或嵌套零跳变）。
    for (uint64_t trial_seed : {1ULL, 2ULL, 3ULL, 4ULL}) {
        g_rng = 0x9E3779B97F4A7C15ULL * trial_seed + 12345ULL;
        double sw_n = 0.0, sw_i = 0.0;
        int64_t crossings = 0;
        for (int64_t t = 0; t < kProbeTrials; ++t) {
            std::vector<float> h = probe_signal();
            std::vector<int64_t> code(static_cast<size_t>(kProbeN));
            std::vector<float> d16(static_cast<size_t>(kProbeN));
            std::vector<float> d8(static_cast<size_t>(kProbeN));
            nested_quant_i32(code.data(), h.data(), kProbeN, kU, 42);
            nested_dequant(d16.data(), code.data(), kProbeN, kU, 16);
            nested_dequant(d8.data(), code.data(), kProbeN, kU, 8);
            for (int64_t i = 0; i < kProbeN; ++i) {
                const double dd = static_cast<double>(d16[static_cast<size_t>(i)]) -
                                  static_cast<double>(d8[static_cast<size_t>(i)]);
                if (dd != 0.0) ++crossings;
                sw_n += dd * dd;
            }
            // 独立基线：16bit / 8bit 两次独立 SR（各自重采样，探针 independent_codes）
            std::vector<float> s16(static_cast<size_t>(kProbeN));
            std::vector<float> s8(static_cast<size_t>(kProbeN));
            const double sp16 = kU * 65536.0, sp8 = kU * 16777216.0;
            for (int64_t i = 0; i < kProbeN; ++i) {
                const double x16 = static_cast<double>(h[static_cast<size_t>(i)]) / sp16;
                s16[static_cast<size_t>(i)] = static_cast<float>(std::floor(x16 + rng_uniform()) * sp16);
                const double x8 = static_cast<double>(h[static_cast<size_t>(i)]) / sp8;
                s8[static_cast<size_t>(i)] = static_cast<float>(std::floor(x8 + rng_uniform()) * sp8);
            }
            for (int64_t i = 0; i < kProbeN; ++i) {
                const double dd = static_cast<double>(s16[static_cast<size_t>(i)]) -
                                  static_cast<double>(s8[static_cast<size_t>(i)]);
                sw_i += dd * dd;
            }
        }
        const double denom = static_cast<double>(kProbeTrials * kProbeN);
        sw_n /= denom;
        sw_i /= denom;
        char ratio_buf[32];
        if (sw_n > 0.0) std::snprintf(ratio_buf, sizeof(ratio_buf), "%.3g", sw_i / sw_n);
        else            std::snprintf(ratio_buf, sizeof(ratio_buf), "inf(no-jump)");
        const bool pass = (sw_n == 0.0 && sw_i > 0.0) || (sw_i / sw_n > 1e5);
        CHECK(pass, "N3 switching consistency ratio below 1e5");
        std::printf("  [N3-A2] seed=%llu: nested_switch_mse=%g indep=%g ratio=%s "
                    "(nonzero-jump elements=%lld)\n",
                    (unsigned long long)trial_seed, sw_n, sw_i, ratio_buf,
                    (long long)crossings);
    }

    // ---- A1 归一化口径：per-tensor scale 归一化到嵌套满幅 ±2^31·u ----
    //   同格同信号下 RTN 与 SR 逐元素误差支配：E[RTN²|θ] = s²·min(θ,1−θ)² ≤
    //   s²·θ(1−θ) = E[SR²|θ] ⇒ 聚合 nested_MSE ≤ indep_MSE（可证明方向，断言）。
    //   实测比值打印落档（全幅信号理论值 ≈ 0.5：RTN s²/12 vs SR s²/6）。
    {
        g_rng = 0xC0FFEE123456789ULL;
        double err_n[3] = {0.0, 0.0, 0.0}, err_i[3] = {0.0, 0.0, 0.0};
        const int levels[3] = {4, 8, 16};
        const double full_scale = 2147483648.0 * kU;   // ±2^31·u
        for (int64_t t = 0; t < kProbeTrials; ++t) {
            std::vector<float> raw = probe_signal();
            float mx = 0.f;
            for (float v : raw) mx = std::fabs(v) > mx ? std::fabs(v) : mx;
            const float scale = static_cast<float>(full_scale) / mx;
            std::vector<float> hn(static_cast<size_t>(kProbeN));
            for (size_t i = 0; i < hn.size(); ++i) hn[i] = raw[i] * scale;

            std::vector<int64_t> code(static_cast<size_t>(kProbeN));
            std::vector<float> d(static_cast<size_t>(kProbeN));
            nested_quant_i32(code.data(), hn.data(), kProbeN, kU, 7);
            for (int j = 0; j < 3; ++j) {
                nested_dequant(d.data(), code.data(), kProbeN, kU, levels[j]);
                err_n[j] += mse(d, hn);
                err_i[j] += indep_sr_mse(hn, kU * static_cast<double>(
                            int64_t(1) << (32 - levels[j])));
            }
        }
        for (int j = 0; j < 3; ++j) {
            err_n[j] /= static_cast<double>(kProbeTrials);
            err_i[j] /= static_cast<double>(kProbeTrials);
            CHECK(err_n[j] <= err_i[j] * 1.000001,
                  "N3-A1 nested MSE exceeds independent SR (dominance broken)");
            std::printf("  [N3-A1] bits=%d: nested=%g indep=%g ratio=%.4f "
                        "(full-scale theory 0.5)\n",
                        levels[j], err_n[j], err_i[j], err_n[j] / err_i[j]);
        }
    }
    // ---- A1 原始口径粗档复现（修订判据第二支，2026-09-05 用户拍板）----
    //   raw gauss·U(0.2,2) 信号 ≪ 粗格胞：嵌套截位确定性留在最近格点（永不跳格），
    //   独立 SR 以 |h|/s16 概率跳整格 → b=16 比值 ≈ 0.119（诊断记录 1.49/12.56）。
    //   断言 < 0.25（2× 统计余量；诊断值 0.119，跨 LCG 种子波动 ±10%）。
    {
        g_rng = 0xABCD1234EF567890ULL;
        double en = 0.0, ei = 0.0;
        const double s16 = kU * 65536.0;
        for (int64_t t = 0; t < kProbeTrials; ++t) {
            std::vector<float> h = probe_signal();
            std::vector<int64_t> code(static_cast<size_t>(kProbeN));
            std::vector<float> d(static_cast<size_t>(kProbeN));
            nested_quant_i32(code.data(), h.data(), kProbeN, kU, 42);
            nested_dequant(d.data(), code.data(), kProbeN, kU, 16);
            en += mse(d, h);
            ei += indep_sr_mse(h, s16);
        }
        en /= static_cast<double>(kProbeTrials);
        ei /= static_cast<double>(kProbeTrials);
        CHECK(en < ei * 0.25, "N3-A1 raw-regime b16 dominance weaker than 4x");
        std::printf("  [N3-A1raw] bits=16: nested=%g indep=%g ratio=%.4f "
                    "(diagnostic 0.119)\n", en, ei, en / ei);
    }
}

// ----------------------------------------------------------------------------
// N4：boundary / UB 卫生
static void test_n4_boundary() {
    using sgn::mkern::nested::nested_quant_i32;
    using sgn::mkern::nested::nested_dequant;

    {   // n = 0：无操作不崩溃（未初始化缓冲不得被写）
        int64_t c = 123;
        float o = 123.f;
        nested_quant_i32(&c, nullptr, 0, kU, 1);
        nested_dequant(&o, nullptr, 0, kU, 8);
        CHECK(c == 123 && o == 123.f, "N4 n=0 must be a no-op");
    }
    {   // 非法 level：无操作（out 不修改）
        const int64_t c = 777;
        float o = 5.f;
        nested_dequant(&o, &c, 1, 1.0f, 5);
        CHECK(o == 5.f, "N4 invalid level must leave out untouched");
    }
    {   // 饱和：±FLT_MAX、域外确定值 → 码字 int32 边界；dequant = 边界·u
        const float u = kU;
        const float big = 2147483648.0f * 2.0f;   // 2^32（float 精确）≫ 满幅
        const float hs[4] = {3.4028235e38f, -3.4028235e38f, big * u, -big * u};
        const int64_t expect[4] = {2147483647, -2147483648, 2147483647, -2147483648};
        int64_t c[4] = {0, 0, 0, 0};
        nested_quant_i32(c, hs, 4, u, 42);
        bool ok = true;
        for (int i = 0; i < 4; ++i) ok = ok && (c[i] == expect[i]);
        CHECK(ok, "N4 saturation bounds violated");
        float o[4] = {0.f, 0.f, 0.f, 0.f};
        nested_dequant(o, c, 4, u, 32);
        ok = true;
        for (int i = 0; i < 4; ++i) {
            ok = ok && (o[i] == static_cast<float>(static_cast<double>(c[i]) *
                                                   static_cast<double>(u)));
        }
        CHECK(ok, "N4 saturated codeword dequant mismatch");
    }
    {   // 往返误差界（域内输入）：|dequant_b(h) − h| ≤ s_b/2 + u
        //   （SR 误差 ≤ 1 细格 + RTN 误差 ≤ 半粗格；信号限幅在 ±0.999·满幅内）
        const double amp = 2147483648.0 * kU * 0.999 / 6.0;   // 6σ ≈ 0.999 满幅
        for (int level : {4, 8, 16, 32}) {
            const double s = kU * static_cast<double>(int64_t(1) << (32 - level));
            const double bound = s / 2.0 + kU * 1.0000001;
            double worst = 0.0;
            bool ok = true;
            for (int64_t t = 0; t < 40 && ok; ++t) {
                std::vector<float> h(static_cast<size_t>(64));
                for (auto& v : h) v = static_cast<float>(rng_gauss() * amp);
                const int64_t nn = static_cast<int64_t>(h.size());
                std::vector<int64_t> code(h.size());
                std::vector<float> d(h.size());
                nested_quant_i32(code.data(), h.data(), nn, kU, 99);
                nested_dequant(d.data(), code.data(), nn, kU, level);
                for (size_t i = 0; i < h.size(); ++i) {
                    const double e = std::fabs(static_cast<double>(d[i]) -
                                               static_cast<double>(h[i]));
                    if (e > worst) worst = e;
                    if (e > bound) { ok = false; break; }
                }
            }
            CHECK(ok, "N4 round-trip error beyond s_b/2 + u");
            std::printf("  [N4] level=%d: worst round-trip error %g (bound %g)\n",
                        level, worst, bound);
        }
    }
}

// ----------------------------------------------------------------------------
// Phase 2：view_codes 整数视图码（MSint 点积域入口，S3 衔接评估 B3/B4）
static void test_view_codes() {
    using sgn::mkern::nested::nested_view_codes;
    using sgn::mkern::nested::nested_dequant;

    const int64_t n = 1000;
    std::vector<int64_t> code(static_cast<size_t>(n));
    // 宽幅码字：int32 全域均匀 + 饱和界 + 构造 tie 模式（m=0 无 tie 语义，跳过）
    std::vector<int64_t> wild;
    wild.push_back(-2147483648LL);
    wild.push_back(2147483647LL);
    for (int64_t k = -40; k <= 40; ++k) {
        for (int m : {28, 24, 16, 0}) {
            if (m > 0) {
                wild.push_back(k * (int64_t(1) << m) + (int64_t(1) << (m - 1)));
            }
            wild.push_back(k * (int64_t(1) << m));
        }
    }
    const int64_t nw = static_cast<int64_t>(wild.size());

    for (int level : {4, 8, 16, 32}) {
        const int m = 32 - level;
        // 视图整数码域 = [−2^(b−1), 2^(b−1)]（非对称 2^b+1 值：饱和顶码
        // I = 2³¹−1 的 RTN 上取整可达 +2^(b−1)——int32 域不对称的必然结果）
        const int64_t lo_bound = -(int64_t(1) << (level - 1));
        const int64_t hi_bound = (int64_t(1) << (level - 1));
        // 域内随机码 + 极端码合并测试
        std::vector<int64_t> code(wild);
        for (int64_t i = 0; i < n; ++i) {
            code.push_back(static_cast<int64_t>(rng64() % 2000000) - 1000000);
        }
        const int64_t nc = static_cast<int64_t>(code.size());
        std::vector<int64_t> vc(static_cast<size_t>(nc), -99);
        std::vector<float> d(static_cast<size_t>(nc), -99.f);
        nested_view_codes(code.data(), nc, level, vc.data());
        nested_dequant(d.data(), code.data(), nc, 1.0f, level);
        bool ok = true, range_ok = true, deq_ok = true, ident_ok = true;
        for (int64_t i = 0; i < nc && (ok || range_ok || deq_ok); ++i) {
            const int64_t I = code[static_cast<size_t>(i)];
            const int64_t trunc = ref::rtn(I, m);
            const int64_t expect_vc = trunc >> m;
            if (vc[static_cast<size_t>(i)] != expect_vc) ok = false;
            // 非对称域断言仅限 int32 域内码字（wild 超域码的商本就外推，
            // dequant/view_codes 对任意 int64 良定义但域主张不适用）
            if (I >= -2147483648LL && I <= 2147483647LL) {
                if (expect_vc < lo_bound || expect_vc > hi_bound) range_ok = false;
            }
            // dequant 一致性：out == (float)(ldexp(vc, m)·u)
            const float expect_d = static_cast<float>(
                std::ldexp(static_cast<double>(expect_vc), m) * 1.0);
            if (d[static_cast<size_t>(i)] != expect_d) deq_ok = false;
            if (level == 32 && vc[static_cast<size_t>(i)] != I) ident_ok = false;
        }
        CHECK(ok, "view_codes mismatch vs independent RTN reference");
        CHECK(range_ok, "view_codes outside asymmetric b-bit domain");
        CHECK(deq_ok, "view_codes/dequant reconstruction mismatch");
        if (level == 32) CHECK(ident_ok, "view_codes level-32 identity broken");
    }
    // n=0 与非法 level 无操作
    {
        std::vector<int64_t> vc(3, -99);
        nested_view_codes(nullptr, 0, 8, vc.data());
        CHECK(vc[0] == -99 && vc[1] == -99 && vc[2] == -99,
              "view_codes n=0 must be a no-op");
        const int64_t c = 5;
        nested_view_codes(&c, 1, 5, vc.data());
        CHECK(vc[0] == -99, "view_codes invalid level must be a no-op");
    }
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);   // 无缓冲：UBSan 陷阱时日志不丢
    std::printf("=== mkern/nested boundary test ===\n");
    std::printf("nested backend: %s\n\n",
                sgn::mkern::nested::active_nested_backend_name());
    test_n1_bitexact();
    test_n1_cross_backend();
    test_n2_nesting();
    test_view_codes();
    test_n3_switching_and_a1();
    test_n4_boundary();
    std::printf("\n%d/%d checks passed\n", g_total - g_failed, g_total);
    if (g_failed == 0) {
        std::printf("ALL NESTED BOUNDARY TESTS PASSED\n");
        return 0;
    }
    std::printf("NESTED BOUNDARY TESTS FAILED\n");
    return 1;
}
