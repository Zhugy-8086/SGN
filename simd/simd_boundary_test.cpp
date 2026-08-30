// simd_boundary_test.cpp - simd 原语层边界条件 + 内存安全审查（独立测试 exe，非 pyd）
//
// 目的（2026-08-29 审查）：
//   1. 全部 10 个可调度原语的边界尺寸扫描（0/1/尾部非对齐/跨块），SIMD 路径 vs 标量锚点逐位对比；
//   2. 极值覆盖（dot16 的 -32768、dot8 的 255×127 满幅——即原实现注释中破坏 bit-exact 的两个边界）；
//   3. 用 -fsanitize=undefined 编译抓未定义行为（索引越界、有符号溢出等）。
// 编译：clang++ -O1 -mavx2 -mavxvnni -std=c++23 -fsanitize=undefined -I<sgn root>
// 用 -O1 而非 -O3：sanitizer 下保留可诊断性。

#include "simd/simd_api.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

using namespace sgn::simd;

// 标量锚点（内部命名，见 scalar.cpp / simd_dispatch.cpp；测试直接对比 SIMD vs 标量）
namespace sgn::simd {
int64_t dot16_scalar(const int16_t*, const int16_t*, size_t);
int64_t dot8_scalar(const uint8_t*, const int8_t*, size_t);
int64_t dot4_scalar(const uint8_t*, const int8_t*, size_t);
void decode_i16_f32_scalar(const uint64_t*, int, float, float*);
void reverse_bytes8_scalar(uint64_t, int, int64_t*);
void batch_reverse_u8_scalar(const uint64_t*, int, int, int64_t*);
float sum_f32_scalar(const float*, int64_t, int64_t);
float sum_sq_dev_f32_scalar(const float*, int64_t, int64_t, float);
void sum_sumprod_f32_scalar(const float*, const float*, int64_t, int64_t, float*, float*);
void accum_f32_scalar(float*, const float*, int64_t);
}

static int failures = 0;

#define CHECK(cond, msg)                                              \
    do {                                                              \
        if (!(cond)) {                                                \
            ++failures;                                               \
            std::printf("FAIL: %s (line %d)\n", msg, __LINE__);       \
        }                                                             \
    } while (0)

// 伪随机但可复现
static std::mt19937_64 rng(12345);

static void test_dot16() {
    // 尺寸扫描：0,1,7,8,15,16,17,31,32,33,1000
    for (size_t K : {0u, 1u, 7u, 8u, 15u, 16u, 17u, 31u, 32u, 33u, 1000u}) {
        std::vector<int16_t> a(K), b(K);
        // 满幅覆盖：含 -32768（madd_epi16 破坏位）与普通值
        for (size_t i = 0; i < K; ++i) {
            a[i] = (i % 3 == 0) ? INT16_MIN : static_cast<int16_t>(rng() % 65536);
            b[i] = (i % 5 == 0) ? INT16_MIN : static_cast<int16_t>(rng() % 65536);
        }
        int64_t simd_v = dot16(a.data(), b.data(), K);
        int64_t ref_v = dot16_scalar(a.data(), b.data(), K);
        CHECK(simd_v == ref_v, "dot16 SIMD vs scalar mismatch");
    }
    // 全 -32768 极值（历史上 madd_epi16 恰好溢出的用例）
    {
        std::vector<int16_t> a(64, INT16_MIN), b(64, INT16_MIN);
        CHECK(dot16(a.data(), b.data(), 64) == dot16_scalar(a.data(), b.data(), 64),
              "dot16 all-INT16_MIN mismatch");
    }
}

static void test_dot8_dot4() {
    for (size_t K : {0u, 1u, 7u, 8u, 31u, 32u, 33u, 1000u}) {
        std::vector<uint8_t> u(K);
        std::vector<int8_t> s(K);
        for (size_t i = 0; i < K; ++i) {
            u[i] = (i % 4 == 0) ? 255 : static_cast<uint8_t>(rng() % 256);  // 满幅 255
            s[i] = (i % 6 == 0) ? 127 : (i % 7 == 0 ? -128 : static_cast<int8_t>(rng() % 256));
        }
        CHECK(dot8(u.data(), s.data(), K) == dot8_scalar(u.data(), s.data(), K),
              "dot8 SIMD vs scalar mismatch");
        CHECK(dot4(u.data(), s.data(), K) == dot4_scalar(u.data(), s.data(), K),
              "dot4 SIMD vs scalar mismatch");
    }
    // 满幅 255×127（maddubs 饱和破坏位的用例）
    {
        std::vector<uint8_t> u(64, 255);
        std::vector<int8_t> s(64, 127);
        CHECK(dot8(u.data(), s.data(), 64) == dot8_scalar(u.data(), s.data(), 64),
              "dot8 full-scale mismatch");
    }
}

static void test_decode() {
    for (int n : {0, 1, 3, 7, 8, 9, 15, 16, 100}) {
        std::vector<uint64_t> pv(n);
        for (int i = 0; i < n; ++i) {
            pv[i] = rng() & 0xFFFF;  // 低 16 位有效
            if (i % 3 == 0) pv[i] = 0x8000;   // 负边界
            if (i % 5 == 0) pv[i] = 0xFFFF;   // -1
        }
        std::vector<float> r1(n), r2(n);
        decode_i16_f32(pv.data(), n, 0.25f, r1.data());
        decode_i16_f32_scalar(pv.data(), n, 0.25f, r2.data());
        CHECK(std::memcmp(r1.data(), r2.data(), sizeof(float) * n) == 0,
              "decode_i16_f32 SIMD vs scalar mismatch");
    }
}

static void test_reverse() {
    // reverse_bytes8：n=1..8
    for (int n = 1; n <= 8; ++n) {
        uint64_t packed = rng();
        std::vector<int64_t> r1(n), r2(n);
        reverse_bytes8(packed, n, r1.data());
        reverse_bytes8_scalar(packed, n, r2.data());
        CHECK(std::memcmp(r1.data(), r2.data(), sizeof(int64_t) * n) == 0,
              "reverse_bytes8 mismatch");
    }
    // batch_reverse_u8：n_values 覆盖 4 块边界（0,1,3,4,5,9），n_slots 1..8
    for (int nv : {0, 1, 3, 4, 5, 9}) {
        for (int ns = 1; ns <= 8; ++ns) {
            std::vector<uint64_t> pv(nv);
            for (int i = 0; i < nv; ++i) pv[i] = rng();
            std::vector<int64_t> r1(static_cast<size_t>(nv) * ns),
                r2(static_cast<size_t>(nv) * ns);
            batch_reverse_u8(pv.data(), nv, ns, r1.data());
            batch_reverse_u8_scalar(pv.data(), nv, ns, r2.data());
            CHECK(std::memcmp(r1.data(), r2.data(), sizeof(int64_t) * nv * ns) == 0,
                  "batch_reverse_u8 mismatch");
        }
    }
}

static void test_sum_series() {
    // stride=1 连续 与 stride=5 列方向；n 覆盖 8 边界
    for (int64_t n : {0, 1, 7, 8, 9, 23, 100}) {
        for (int64_t stride : {1, 5}) {
            int64_t total = (n == 0) ? 0 : (n - 1) * stride + 1;
            std::vector<float> p(total);
            // 幅度限制在 ±100：保证 d² ≤ 103²、部分和 ≤ 100×103² ≈ 1.06e6 < 2^24，
            // 所有中间量精确可表示 → SIMD(分块)与标量(顺序)的逐位对比才有确定期望
            for (auto& x : p) x = static_cast<float>(rng() % 201) - 100.0f;

            CHECK(sum_f32(p.data(), n, stride) == sum_f32_scalar(p.data(), n, stride),
                  "sum_f32 mismatch");
            // 浮点归约 SIMD(8 路分块)与标量(顺序累加)是舍入级差异(kRounding)，
            // 仅在数据可精确表示时逐位一致——用整数 mu 使所有中间量精确，
            // 才能做逐位对比（分数 mu 下顺序差异属预期，非 bug）。
            CHECK(sum_sq_dev_f32(p.data(), n, stride, 3.0f) ==
                      sum_sq_dev_f32_scalar(p.data(), n, stride, 3.0f),
                  "sum_sq_dev_f32 mismatch");

            std::vector<float> q(total);
            for (auto& x : q) x = static_cast<float>(rng() % 101) - 50.0f;
            float s1, sp1, s2, sp2;
            sum_sumprod_f32(p.data(), q.data(), n, stride, &s1, &sp1);
            sum_sumprod_f32_scalar(p.data(), q.data(), n, stride, &s2, &sp2);
            CHECK(s1 == s2 && sp1 == sp2, "sum_sumprod_f32 mismatch");
        }
    }
    // accum_f32
    for (int64_t n : {0, 1, 7, 8, 9, 100}) {
        std::vector<float> d1(n, 1.0f), d2(n, 1.0f), s(n);
        for (auto& x : s) x = static_cast<float>(rng() % 100);
        accum_f32(d1.data(), s.data(), n);
        accum_f32_scalar(d2.data(), s.data(), n);
        CHECK(std::memcmp(d1.data(), d2.data(), sizeof(float) * n) == 0,
              "accum_f32 mismatch");
    }
}

static void test_backend_name() {
    const char* n = active_backend_name();
    std::printf("active backend: %s\n", n);
    CHECK(n != nullptr, "backend name null");
}

int main() {
    test_dot16();
    test_dot8_dot4();
    test_decode();
    test_reverse();
    test_sum_series();
    test_backend_name();
    if (failures == 0) {
        std::printf("ALL BOUNDARY TESTS PASSED\n");
        return 0;
    }
    std::printf("%d FAILURES\n", failures);
    return 1;
}
