// scalar.cpp - mkern/nested 标量实现（bit-exact 参考锚点，全平台常驻编译）
//
// 与 mkern/gemm/scalar.cpp 同纪律：标量锚点常驻编译，供 nested_dispatch.cpp 运行时
// 表回退，并作为 boundary 测试的逐位参照（N1）。规格（RNG/SR 判式/饱和/tie）见
// nested_api.h 头注释——实现必须逐条对应，任何偏离即违反 kBitExact 契约。
//
// 背景：内部档案 §一/§五.3。

#include "mkern/nested/nested_api.h"

#include <cmath>

namespace sgn::mkern::nested {

namespace {

// 契约 RNG（规格 a）：计数器式 SplitMix64 终混，U ∈ [0,1) 53 位。
inline double contract_uniform(uint64_t seed, int64_t i) {
    uint64_t z = seed + 0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(i);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return static_cast<double>(z >> 11) * 0x1.0p-53;
}

// 规格 d 的 RTN 商（half-to-even）——dequant 与 view_codes 的**唯一整型定义点**
// （单一定义纪律，同 sr_kernel F5 教训：两处手写 RTN 会静默破坏逐位一致）。
inline int64_t rtn_quotient(int64_t I, int m) {
    int64_t q = I >> m;                     // 算术移位 = floor(I/2^m)
    if (m > 0) {                            // m=0（level 32）恒等，跳过
        const int64_t r = I - (q << m);     // ∈ [0, 2^m)，q·2^m ≤ I 无溢出
        const int64_t half = int64_t(1) << (m - 1);
        if (r > half || (r == half && (q & 1))) ++q;
    }
    return q;
}

} // namespace

void nested_quant_i32_scalar(int64_t* code, const float* h, int64_t n,
                             float u, uint64_t seed) {
    const double ud = static_cast<double>(u);
    for (int64_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(h[i]) / ud;   // 规格 b：f64 口径
        int64_t I;
        if (x >= 2147483647.0) {                            // 规格 c：饱和
            I = 2147483647;
        } else if (x <= -2147483648.0) {
            I = -2147483648;
        } else {
            const double xf = std::floor(x);
            const double frac = x - xf;                     // [0,1)，|x|<2^52 内精确
            I = static_cast<int64_t>(xf);
            if (contract_uniform(seed, i) < frac) ++I;      // 规格 b：Bernoulli SR
        }
        code[i] = I;
    }
}

void nested_dequant_scalar(float* out, const int64_t* code, int64_t n,
                           float u, int level) {
    // level ∈ {4,8,16,32} → m ∈ {28,24,16,0}；其他值无操作（防御性，非 UB，
    // 见 nested_api.h 前置条件——m 被 shift 使用，域外直接返回保 UBSan 干净）。
    if (level != 4 && level != 8 && level != 16 && level != 32) return;
    const int m = 32 - level;
    const double ud = static_cast<double>(u);
    const double vpow2m = std::ldexp(1.0, m);   // 精确 2^m（宿主侧一次）
    for (int64_t i = 0; i < n; ++i) {
        const int64_t I = code[i];
        // 规格 d：RTN 商经 rtn_quotient（单一定义点）；值重建走 double 域
        // （q ≤ 2^35+1，53 位尾数精确覆盖）——int64 q<<m 在 I 接近 INT64_MAX
        // 时会符号溢出（UB），严禁用。
        const int64_t q = rtn_quotient(I, m);
        // (double)q · 2^m 与 ldexp((double)q, m) 数学恒等（幂乘只改指数域，
        // |q·2^m| ≤ 2^63 无溢出）——避免逐元素 libm ldexp 调用
        out[i] = static_cast<float>(static_cast<double>(q) * vpow2m * ud);
    }
}

void nested_view_codes(const int64_t* code, int64_t n, int level, int64_t* out) {
    // level 档视图整数码：out[i] = RTN 商（b bit 有符号，[−2^(b−1), 2^(b−1)−1]）。
    // 与 dequant 共享 rtn_quotient（单一定义点）；标量即终态（整数 shift+cast，
    // 无 SIMD 热路径）；level 校验同 dequant（域外无操作）。
    if (level != 4 && level != 8 && level != 16 && level != 32) return;
    const int m = 32 - level;
    for (int64_t i = 0; i < n; ++i) {
        out[i] = rtn_quotient(code[i], m);
    }
}

} // namespace sgn::mkern::nested
