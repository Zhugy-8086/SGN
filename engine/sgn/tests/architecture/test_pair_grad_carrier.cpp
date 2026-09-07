// test_pair_grad_carrier.cpp - int8 对（h,l）梯度载体 C++ 测试（独立 exe）
//
// 验证项（对应 msint_int8_pair_grad_carrier_2026_08_31.md §七/§八）：
//   P1 编解码往返 bit-exact：q = 256·h_u8 + l_u8 − 32768 对网格全域（含极值）
//   P2 SR 双路一致性：sr_quantize_to_pair（pair 路径）vs sr_quantize_grad
//      （现有 float 路径，同种子重置）输出 float 梯度 bit-exact 相同
//      —— pair 化不改训练动态（V7 dev=0 的实现侧兑现）
//   P3 fine dot8 消费：256·dot8(h,w8) + dot8(l,w8) − 32768·Sw == 标量 Σq·w
//      （K 含非对齐尺寸 + 255×127 满幅对抗，int64 精确）
//   P4 coarse dot8 消费：256·(dot8(h,w8) − 128·Sw) == 标量 Σ(256·h_s)·w
//   P5 浮点解码：fine q·scale 与 P2 参考一致；coarse = 256·h_s·scale
//
// 编译：由 engine/sgn/CMakeLists.txt test_pair_grad_carrier target 管理

#include "msint/pair_grad_carrier.h"
#include "autograd/backward_strategy.h"

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

using sgn_msint::PairGradCarrier;
using sgn_msint::sr_quantize_to_pair;
using sgn_msint::decode_pair_fine_f32;
using sgn_msint::decode_pair_coarse_f32;
using sgn_msint::pair_dot_fine;
using sgn_msint::pair_dot_coarse;
using sgn_msint::pair_decode_q;
using sgn_autograd::set_sr_seed;
using sgn_autograd::sr_quantize_grad;

static int failures = 0;
#define CHECK(cond, msg) do { \
    if (!(cond)) { ++failures; std::printf("FAIL: %s (line %d)\n", msg, __LINE__); } \
} while (0)

// ---- P1: 编解码往返（网格全域 + 极值 + 随机）----
static void test_roundtrip() {
    // 极值与边界 q 值（含 -32767：h_u8=0 必须配 l_u8≥1，h=0,l=0 不可达）
    for (int32_t q : {-32767, -32513, -256, -255, -129, -128, -1, 0, 1,
                      127, 128, 255, 256, 32512, 32766, 32767}) {
        PairGradCarrier p;
        p.n = 1; p.scale = 1.0f;
        const int32_t h_s = q >> 8;
        p.h.push_back(static_cast<uint8_t>(h_s + 128));
        p.l.push_back(static_cast<uint8_t>(q & 0xFF));
        CHECK(pair_decode_q(p, 0) == q, "roundtrip mismatch");
    }
    // 随机全域
    std::mt19937 rng(20260902);
    PairGradCarrier p;
    p.n = 70000; p.scale = 1.0f;
    p.h.resize(p.n); p.l.resize(p.n);
    for (size_t i = 0; i < p.n; ++i) {
        int32_t q = static_cast<int32_t>(rng() % 65535) - 32767;  // [-32767, 32767]
        p.h[i] = static_cast<uint8_t>((q >> 8) + 128);
        p.l[i] = static_cast<uint8_t>(q & 0xFF);
        if (pair_decode_q(p, i) != q) {
            CHECK(false, "roundtrip random mismatch");
            break;
        }
    }
}

// ---- P2: SR 双路 bit-exact（pair 路径 vs 现有 float 路径，同种子重置）----
static void test_sr_equivalence() {
    const size_t K = 4096;
    std::mt19937 gen(20260902);
    std::vector<float> g(K);
    for (auto& x : g) {
        x = std::exp(static_cast<float>(gen() % 2000) / 300.0f - 4.0f)
            * ((gen() % 2) ? 1.0f : -1.0f);  // 类梯度分布（跨数量级 + 符号）
    }
    const int bits = 16;
    const float clip_sigma = 4.0f;

    std::vector<float> ref(K);
    set_sr_seed(42);                                   // 同一种子
    std::vector<float> g1 = g;
    sr_quantize_grad(g1.data(), K, bits, clip_sigma);  // 现有 float 路径
    ref = g1;

    set_sr_seed(42);                                   // 同种子重置
    PairGradCarrier p;
    sr_quantize_to_pair(g.data(), K, clip_sigma, p);   // pair 路径

    CHECK(p.n == K, "pair size mismatch");
    std::vector<float> dec(K);
    decode_pair_fine_f32(p, dec.data());
    for (size_t i = 0; i < K; ++i) {
        if (std::memcmp(&dec[i], &ref[i], sizeof(float)) != 0) {
            CHECK(false, "SR dual-path float mismatch (q or RNG sequence diverged)");
            std::printf("  first diff at i=%zu: pair=%a ref=%a\n", i, dec[i], ref[i]);
            return;
        }
    }
    // 显式 scale 一致性
    float scale_ref = 0.0f;
    for (size_t i = 0; i < K; ++i) {
        float a = std::fabs(g[i]);
        if (a > scale_ref) scale_ref = a;
    }
    scale_ref /= 32767.0f;
    CHECK(std::fabs(p.scale - scale_ref) / scale_ref < 1e-7f, "scale mismatch");
}

// ---- P3/P4: dot8 消费恒等式 vs int64 标量参考 ----
static void test_dot_consumption() {
    std::mt19937 gen(20260902);
    for (size_t K : {size_t(0), size_t(1), size_t(7), size_t(31), size_t(33),
                     size_t(63), size_t(1000), size_t(4096), size_t(65536)}) {
        // 构造 pair（直接按编码规则，覆盖满幅对抗：h/l 全 0/255 交替）
        PairGradCarrier p;
        p.n = K; p.scale = 1.0f;
        p.h.resize(K); p.l.resize(K);
        std::vector<int16_t> q(K);
        std::vector<int8_t> w8(K);
        for (size_t i = 0; i < K; ++i) {
            if (i % 7 == 0)      q[i] = -32767;                 // 满幅负
            else if (i % 7 == 1) q[i] = 32767;                  // 满幅正
            else                 q[i] = static_cast<int16_t>(gen() % 65535) - 32767;
            p.h[i] = static_cast<uint8_t>((q[i] >> 8) + 128);
            p.l[i] = static_cast<uint8_t>(q[i] & 0xFF);
            w8[i] = (i % 11 == 0) ? -128
                  : (i % 11 == 3) ? 127
                  : static_cast<int8_t>(gen() % 256);
        }
        // 标量参考（int64 精确）
        int64_t ref_fine = 0, ref_coarse = 0;
        for (size_t i = 0; i < K; ++i) {
            const int32_t h_s = q[i] >> 8;
            ref_fine   += static_cast<int64_t>(q[i]) * w8[i];
            ref_coarse += static_cast<int64_t>(256 * h_s) * w8[i];
        }
        CHECK(pair_dot_fine(p, w8.data()) == ref_fine,
              "fine dot8 identity mismatch");
        CHECK(pair_dot_coarse(p, w8.data()) == ref_coarse,
              "coarse dot8 identity mismatch");
    }
    // 大 K 满幅守卫（2026-09-05 dot8 审查）：K=1M 越过 dot8 修复前的 int32
    // 溢出阈值（vnni 526,296 / avx2 1,052,631）。q=+32767（h_u8=255, l_u8=255），
    // w8=127 → fine 真值 = Σq·w = 32767×127×K = 4,161,409×K ≈ 4.36e12 > 2^31，
    // 修复前 dot8 内部回绕必 FAIL。
    {
        const size_t K = 1048576;
        PairGradCarrier p;
        p.n = K; p.scale = 1.0f;
        p.h.assign(K, 255);   // h_s = 127（q=+32767）
        p.l.assign(K, 255);   // l' = 255
        std::vector<int8_t> w8(K, 127);
        CHECK(pair_dot_fine(p, w8.data()) == 32767LL * 127 * static_cast<int64_t>(K),
              "fine large-K full-scale overflow guard");
        // coarse：q_c = 256×h_s = 32512 → 真值 = 32512×127×K
        CHECK(pair_dot_coarse(p, w8.data()) == 32512LL * 127 * static_cast<int64_t>(K),
              "coarse large-K full-scale overflow guard");
    }
}

// ---- P5: 浮点解码语义 ----
static void test_decode_f32() {
    const size_t K = 1000;
    std::mt19937 gen(20260902);
    std::vector<float> g(K);
    for (auto& x : g) x = static_cast<float>(gen() % 100000) / 50000.0f - 1.0f;
    set_sr_seed(7);
    PairGradCarrier p;
    sr_quantize_to_pair(g.data(), K, 4.0f, p);
    std::vector<float> fine(K), coarse(K);
    decode_pair_fine_f32(p, fine.data());
    decode_pair_coarse_f32(p, coarse.data());
    for (size_t i = 0; i < K; ++i) {
        const int32_t q = pair_decode_q(p, i);
        if (fine[i] != static_cast<float>(q) * p.scale) {
            CHECK(false, "fine decode != q*scale");
            return;
        }
        const int32_t h_s = static_cast<int32_t>(p.h[i]) - 128;
        if (coarse[i] != static_cast<float>(256 * h_s) * p.scale) {
            CHECK(false, "coarse decode != 256*h_s*scale");
            return;
        }
    }
}

int main() {
    test_roundtrip();
    test_sr_equivalence();
    test_dot_consumption();
    test_decode_f32();
    if (failures == 0) {
        std::printf("ALL PAIR GRAD CARRIER TESTS PASSED\n");
        return 0;
    }
    std::printf("%d FAILURES\n", failures);
    return 1;
}
