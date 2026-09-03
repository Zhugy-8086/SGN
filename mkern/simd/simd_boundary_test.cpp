// simd_boundary_test.cpp - simd 原语层边界条件 + 内存安全审查（独立测试 exe，非 pyd）
//
// 目的（2026-08-29 审查）：
//   1. 全部 10 个可调度原语的边界尺寸扫描（0/1/尾部非对齐/跨块），SIMD 路径 vs 标量锚点逐位对比；
//   2. 极值覆盖（dot16 的 -32768、dot8 的 255×127 满幅——即原实现注释中破坏 bit-exact 的两个边界）；
//   3. 用 -fsanitize=undefined 编译抓未定义行为（索引越界、有符号溢出等）。
// 编译：clang++ -O1 -mavx2 -mavxvnni -std=c++23 -fsanitize=undefined -I<sgn root>
// 用 -O1 而非 -O3：sanitizer 下保留可诊断性。

#include "mkern/simd/simd_api.h"

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
// AVX2 中间档（avx2_dot.cpp，综合执行计划 §一）——定点直测（绕开 dispatch 的 vnni
// 优先级），验证 vpmaddubsw 中间档的饱和边界位 bit-exact。
int64_t dot8_avx2(const uint8_t*, const int8_t*, size_t);
int64_t dot4_avx2(const uint8_t*, const int8_t*, size_t);
// R2 4 位打包点积（avx2_dot / avxvnni / avx512vnni / scalar）——直接消费打包 nibble
int64_t dot4_packed(const uint8_t*, const int8_t*, size_t);
int64_t dot4_packed_scalar(const uint8_t*, const int8_t*, size_t);
// 连续 int16 解码（avx2_decode.cpp AVX2 + scalar.cpp 标量锚点，布局优化）——定点直测
void decode_i16_f32_packed16(const int16_t*, int, float, float*);
void decode_i16_f32_packed16_scalar(const int16_t*, int, float, float*);
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

// R2 4 位打包点积专项（mkern微内核层实施计划 §3.3）：
// 直接消费 nibble 打包布局（K/2 字节），验证 dot4_packed 各后端 bit-exact。
// 打包格式：byte[j] = (elem[2j+1] << 4) | elem[2j]；a 无符号 0-15、b 有符号 -8..7。
static void test_dot4_packed() {
    for (size_t K : {0u, 1u, 2u, 3u, 31u, 32u, 63u, 64u, 65u, 127u, 128u, 129u, 1000u}) {
        const size_t nb = (K + 1) / 2;
        std::vector<uint8_t> a_packed(nb, 0), b_packed(nb, 0);
        std::vector<int> a(K), b(K);
        int64_t ref = 0;
        for (size_t i = 0; i < K; ++i) {
            const int ai = static_cast<int>(rng() % 16);        // a 无符号 0-15
            const int bi = static_cast<int>(rng() % 16) - 8;    // b 有符号 -8..7
            a[i] = ai; b[i] = bi;
            ref += static_cast<int64_t>(ai) * static_cast<int64_t>(bi);
            const size_t j = i >> 1;
            const int sh = 4 * (static_cast<int>(i) & 1);
            if (i & 1) {
                a_packed[j] |= static_cast<uint8_t>((ai & 0x0F) << 4);
                b_packed[j] |= static_cast<uint8_t>((bi & 0x0F) << 4);
            } else {
                a_packed[j] = static_cast<uint8_t>(ai & 0x0F);
                b_packed[j] = static_cast<uint8_t>(bi & 0x0F);
            }
        }
        const int8_t* bp = reinterpret_cast<const int8_t*>(b_packed.data());
        CHECK(dot4_packed(a_packed.data(), bp, K) == ref,
              "dot4_packed dispatch vs elementwise ref mismatch");
        CHECK(dot4_packed(a_packed.data(), bp, K) ==
                  dot4_packed_scalar(a_packed.data(), bp, K),
              "dot4_packed dispatch vs scalar mismatch");
    }
    // 满幅边界：a=15 全、b=-8/+7 全
    {
        const size_t K = 128, nb = 64;
        std::vector<uint8_t> ap(nb, 0xFF), bp(nb, 0xFF);  // a 全 15
        std::vector<uint8_t> bp_neg(nb, 0x88);            // b 全 -8（0x8 位模式）
        int64_t ref = 0;
        for (size_t i = 0; i < K; ++i) ref += 15 * -8;    // 15 × -8
        CHECK(dot4_packed(ap.data(), reinterpret_cast<const int8_t*>(bp_neg.data()), K) == ref,
              "dot4_packed full-scale -8 mismatch");
        std::vector<uint8_t> ap2(nb, 0xFF), bp2(nb, 0x77);  // a=15, b=7
        int64_t ref2 = 0;
        for (size_t i = 0; i < K; ++i) ref2 += 15 * 7;
        CHECK(dot4_packed(ap2.data(), reinterpret_cast<const int8_t*>(bp2.data()), K) == ref2,
              "dot4_packed full-scale +7 mismatch");
    }
}

// AVX2 中间档饱和对抗专项（综合执行计划 §一，2026-09-02）：
// 直接调 dot8_avx2 / dot4_avx2（绕开 dispatch 的 vnni 优先级），验证 vpmaddubsw
// 中间档在饱和边界位的 bit-exact：
//   - a 全 ≤127（对和 ≤ 32258 < 32768 永不饱和）→ maddubs 全速路径
//   - a 含 ≥128（触发 cvtep8+madd_epi16 精确回退路径）
//   - 批内混排触发饱和 vs 接近不饱和
static void test_dot8_avx2_mid() {
    // 用例组：每元素 a/b 由"饱和触发源"决定
    //   0: 普通随机（走 maddubs 安全路径）
    //   1: a=127 b=127（对和 32258，接近但不饱和的上界——maddubs 路径正确）
    //   2: a=255 b=127（对和 64770 > 32767，maddubs 饱和 → 必须走精确回退）
    //   3: a=255 b=-128（对和 -65280，负饱和 → 精确回退）
    //   4: a 含 128（边界：对和 128*127*2=32512 < 32768 仍安全，但 a≥128 触发回退路径）
    for (size_t K : {0u, 1u, 31u, 32u, 33u, 63u, 64u, 65u, 1000u}) {
        std::vector<uint8_t> u(K);
        std::vector<int8_t> s(K);
        for (size_t i = 0; i < K; ++i) {
            switch (i % 5) {
                case 0:
                    u[i] = static_cast<uint8_t>(rng() % 128);  // 0..127
                    s[i] = static_cast<int8_t>(rng() % 256);
                    break;
                case 1: u[i] = 127; s[i] = 127; break;   // 近饱和上界，maddubs 安全
                case 2: u[i] = 255; s[i] = 127; break;   // 正饱和，须回退
                case 3: u[i] = 255; s[i] = -128; break;  // 负饱和，须回退
                case 4: u[i] = 128; s[i] = 127; break;   // a=128 触发回退，但对和仍安全
            }
        }
        CHECK(dot8_avx2(u.data(), s.data(), K) == dot8_scalar(u.data(), s.data(), K),
              "dot8_avx2 mid vs scalar mismatch");
        CHECK(dot4_avx2(u.data(), s.data(), K) == dot4_scalar(u.data(), s.data(), K),
              "dot4_avx2 mid vs scalar mismatch");
    }
    // 全 a=127 b=127（整段 maddubs 安全路径，对和固定在 32258）
    {
        std::vector<uint8_t> u(256, 127);
        std::vector<int8_t> s(256, 127);
        CHECK(dot8_avx2(u.data(), s.data(), 256) == dot8_scalar(u.data(), s.data(), 256),
              "dot8_avx2 all-safe mismatch");
    }
    // 全 a=255 b=127（整段饱和 → 全回退精确路径）
    {
        std::vector<uint8_t> u(256, 255);
        std::vector<int8_t> s(256, 127);
        CHECK(dot8_avx2(u.data(), s.data(), 256) == dot8_scalar(u.data(), s.data(), 256),
              "dot8_avx2 all-sat mismatch");
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
        // n=0 时 data() 返回 nullptr，memcmp 参数声明 nonnull（size=0 亦为 UB，
        // 2026-09-02 远程复核 EPYC 9K65 报告问题 2）→ n>0 才比较
        CHECK(n == 0 || std::memcmp(r1.data(), r2.data(), sizeof(float) * n) == 0,
              "decode_i16_f32 SIMD vs scalar mismatch");
    }
}

static void test_decode_packed16() {
    // 连续 int16 布局解码：尺寸扫描 + 满幅/负边界，AVX2 版 vs 标量锚点逐位一致
    for (int n : {0, 1, 3, 7, 8, 9, 15, 16, 100}) {
        std::vector<int16_t> src(n);
        for (int i = 0; i < n; ++i) {
            src[i] = static_cast<int16_t>(rng() % 65536 - 32768);
            if (i % 3 == 0) src[i] = INT16_MIN;  // 负边界
            if (i % 5 == 0) src[i] = INT16_MAX;  // 正满幅
        }
        std::vector<float> r1(n), r2(n);
        decode_i16_f32_packed16(src.data(), n, 0.25f, r1.data());
        decode_i16_f32_packed16_scalar(src.data(), n, 0.25f, r2.data());
        CHECK(n == 0 || std::memcmp(r1.data(), r2.data(), sizeof(float) * n) == 0,
              "decode_i16_f32_packed16 AVX2 vs scalar mismatch");
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
            // nv=0 时指针为 nullptr（memcmp nonnull UB，同上）
            CHECK(nv == 0 ||
                  std::memcmp(r1.data(), r2.data(), sizeof(int64_t) * nv * ns) == 0,
                  "batch_reverse_u8 mismatch");
        }
    }
}

static void test_sum_series() {
    // stride=1 连续 与 stride=5 列方向；n 覆盖 8 边界
    // R1（2026-09-03）后 sum 系采用固定 8 路 + 固定归约树 + 固定尾，scalar 与 avx2
    // 逐位一致——**任意数据（含分数、非整数 mu）**均可直接对拍，不再依赖整数精确。
    for (int64_t n : {0, 1, 7, 8, 9, 23, 100}) {
        for (int64_t stride : {1, 5}) {
            int64_t total = (n == 0) ? 0 : (n - 1) * stride + 1;
            std::vector<float> p(total);
            // 任意 float（含分数），避免大数消去掩盖问题，但仍保留 ±100 幅度
            for (auto& x : p) x = static_cast<float>(rng() % 2001 - 1000) / 7.0f;

            CHECK(sum_f32(p.data(), n, stride) == sum_f32_scalar(p.data(), n, stride),
                  "sum_f32 mismatch");
            // 分数 mu（R1 固定树下与标量逐位一致，不再限定整数 mu）
            const float mu = 1.7f;
            CHECK(sum_sq_dev_f32(p.data(), n, stride, mu) ==
                      sum_sq_dev_f32_scalar(p.data(), n, stride, mu),
                  "sum_sq_dev_f32 mismatch");

            std::vector<float> q(total);
            for (auto& x : q) x = static_cast<float>(rng() % 2001 - 1000) / 13.0f;
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
        // n=0 时指针为 nullptr（memcmp nonnull UB，同上）
        CHECK(n == 0 || std::memcmp(d1.data(), d2.data(), sizeof(float) * n) == 0,
              "accum_f32 mismatch");
    }
}

static void test_backend_name() {
    const char* n = active_backend_name();
    std::printf("active backend: %s\n", n);
    CHECK(n != nullptr, "backend name null");
}

// 精度档位查询回归（R1 可复现化 2026-09-03）：整型/字节原语 + sum 系/accum 均 kBitExact
// （sum 系统一固定 8 路 + 固定归约树 + 固定尾，scalar/avx2/avx512 逐位一致）
static void test_precision_classes() {
    using P = PrecisionClass;
    CHECK(primitive_precision_class(PrimitiveId::Dot16) == P::kBitExact, "dot16 class");
    CHECK(primitive_precision_class(PrimitiveId::Dot8)  == P::kBitExact, "dot8 class");
    CHECK(primitive_precision_class(PrimitiveId::Dot4)  == P::kBitExact, "dot4 class");
    CHECK(primitive_precision_class(PrimitiveId::Dot4Packed) == P::kBitExact, "dot4_packed class");
    CHECK(primitive_precision_class(PrimitiveId::DecodeI16) == P::kBitExact, "decode class");
    CHECK(primitive_precision_class(PrimitiveId::DecodeI16Packed) == P::kBitExact, "decode16 packed class");
    CHECK(primitive_precision_class(PrimitiveId::Reverse) == P::kBitExact, "reverse class");
    CHECK(primitive_precision_class(PrimitiveId::BatchReverse) == P::kBitExact, "batch class");
    CHECK(primitive_precision_class(PrimitiveId::Sum) == P::kBitExact, "sum class");
    CHECK(primitive_precision_class(PrimitiveId::SumSqDev) == P::kBitExact, "sumsq class");
    CHECK(primitive_precision_class(PrimitiveId::SumSumprod) == P::kBitExact, "sumsumprod class");
    CHECK(primitive_precision_class(PrimitiveId::Accum) == P::kBitExact, "accum class");
    std::printf("precision classes verified\n");
}

// ----------------------------------------------------------------------------
// CPUID 位定义回归测试（2026-09-02）：检测逻辑与手工独立 CPUID 读数交叉验证。
//
// 背景：AVX-VNNI 检测曾双重错位（旧读 leaf7 sub0 ECX[4]=OSPKE 位，正确为
// sub1 EAX[4]）——OSPKE=0 机器 dot8/dot4 静默落标量、EPYC/Linux 因 OSPKE=1
// 侥幸误判掩盖（见 docs/SGN_ArrowLake速度测试归档_2026_09_02.md §2）。
// 本测试在测试内**独立**读 CPUID（不复用 simd_dispatch 内部代码），按与
// simd_backend() 相同的选择逻辑推导期望后端名，与 active_backend_name() 精确
// 比对——位定义/subleaf/寄存器再读错即 FAIL，防复发。
// 注意：两处 CPUID 读法若同时错且错得一样则测不出（同一作者同错概率低，且
// 本文件的读取直接按 Intel SDM 位表写死，不参考 simd_dispatch 实现）。
// ----------------------------------------------------------------------------
#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
#include <intrin.h>
static void raw_cpuid(int leaf, int sub, int* r) { __cpuidex(r, leaf, sub); }
static uint64_t raw_xcr0() { return _xgetbv(0); }
#else
#include <cpuid.h>
static void raw_cpuid(int leaf, int sub, int* r) {
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
}
static uint64_t raw_xcr0() {
    uint32_t lo, hi;
    __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (uint64_t(hi) << 32) | lo;
}
#endif

static void test_cpuid_caps() {
    // 强制后端模式（SGN_KERNEL_BACKEND=scalar/sse2）：simd_backend 直接走标量锚点，
    // CPUID 推导不适用——期望 scalar(forced)，仅保留"无误选 VNNI"断言。
    const char* forced = std::getenv("SGN_KERNEL_BACKEND");
    const bool forced_scalar =
        forced && (std::strcmp(forced, "scalar") == 0 ||
                   std::strcmp(forced, "sse2") == 0);

    int r[4];
    // —— 按 Intel SDM 位表独立读取（注释即位定义出处）——
    raw_cpuid(1, 0, r);
    const bool osxsave = (r[2] & (1 << 27)) != 0;  // leaf1 ECX[27]
    const bool cpu_avx = (r[2] & (1 << 28)) != 0;  // leaf1 ECX[28]
    const bool ssse3   = (r[2] & (1 << 9)) != 0;   // leaf1 ECX[9]
    (void)cpu_avx;
    const uint64_t xcr0 = osxsave ? raw_xcr0() : 0;
    const bool os_ymm = (xcr0 & 0x6) == 0x6;       // XMM+YMM 状态

    bool avx2 = false, avx512f = false, avx512bw = false, avx512vnni = false,
         avx_vnni = false;
    if (os_ymm) {
        raw_cpuid(7, 0, r);  // leaf7 sub0
        avx2      = (r[1] & (1 << 5)) != 0;    // EBX[5]
        avx512f   = (r[1] & (1 << 16)) != 0;   // EBX[16]
        avx512bw  = (r[1] & (1 << 30)) != 0;   // EBX[30]
        avx512vnni = (r[2] & (1 << 11)) != 0;  // ECX[11]
        const bool os_zmm = (xcr0 & 0xE6) == 0xE6;  // opmask+ZMM hi256
        if (avx512f && !os_zmm) { avx512f = avx512bw = false; avx512vnni = false; }
        raw_cpuid(7, 1, r);  // leaf7 sub1
        avx_vnni = (r[0] & (1 << 4)) != 0;     // EAX[4] ← 2026-09-02 修复位
    }

    const char* actual = active_backend_name();

    if (forced_scalar) {
        std::printf("cpuid caps: avx2=%d avx_vnni(sub1.EAX[4])=%d avx512vnni=%d "
                    "(forced scalar mode)\n", avx2, avx_vnni, avx512vnni);
        CHECK(std::strcmp(actual, "scalar(forced)") == 0,
              "forced scalar mode but backend is not scalar(forced)");
    } else {
        // —— 期望后端名（与 simd_dispatch.cpp simd_backend() 选择逻辑逐条一致）——
        const char* expect;
        if (avx512vnni && avx512bw)      expect = "avx512vnni";
        else if (avx512f && avx512bw)    expect = "avx512f";
        else if (avx_vnni)               expect = "avxvnni";
        else if (avx2)                   expect = "avx2";
        else if (ssse3)                  expect = "ssse3";
        else                             expect = "scalar";

        std::printf("cpuid caps: avx2=%d avx_vnni(sub1.EAX[4])=%d avx512vnni=%d "
                    "expect=%s\n", avx2, avx_vnni, avx512vnni, expect);
        CHECK(std::strcmp(actual, expect) == 0,
              "active_backend_name != independent CPUID expectation");

        // 防错位回归断言：
        // 1. CPUID 报告有 256 位 VNNI → 后端必须是 vnni 系（dot8 不得静默落标量）
        if (avx_vnni) {
            CHECK(std::strcmp(actual, "avxvnni") == 0 ||
                  std::strcmp(actual, "avx512vnni") == 0,
                  "AVX-VNNI present but dot8 backend not vnni (detection misread?)");
        }
    }
    // 2.（两种模式均适用）CPUID 报告无 256/512 VNNI → 后端不得是 vnni 系
    //   （防误选 illegal instruction；强制标量下天然满足，仍断言防回归）
    if (!avx_vnni && !(avx512vnni && avx512bw)) {
        CHECK(std::strcmp(actual, "avxvnni") != 0 &&
              std::strcmp(actual, "avx512vnni") != 0,
              "VNNI absent but backend claims vnni (detection misread?)");
    }
}
#else
static void test_cpuid_caps() {
    // 非 x86：仅检查后端名为标量系
    CHECK(std::strcmp(active_backend_name(), "scalar(forced)") == 0 ||
          std::strcmp(active_backend_name(), "scalar") == 0,
          "non-x86 backend should be scalar");
}
#endif

int main() {
    test_dot16();
    test_dot8_dot4();
    test_dot4_packed();
    test_dot8_avx2_mid();
    test_decode();
    test_decode_packed16();
    test_reverse();
    test_sum_series();
    test_backend_name();
    test_precision_classes();
    test_cpuid_caps();
    if (failures == 0) {
        std::printf("ALL BOUNDARY TESTS PASSED\n");
        return 0;
    }
    std::printf("%d FAILURES\n", failures);
    return 1;
}
