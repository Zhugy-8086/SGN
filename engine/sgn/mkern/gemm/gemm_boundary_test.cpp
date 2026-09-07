// gemm_boundary_test.cpp - mkern/gemm 边界/逐位对拍测试
//
// 覆盖（同 simd_boundary_test 纪律）：
//   - gemm_i8 / gemm_i16 / gemv_i8：dispatch vs 标量锚点 vs 朴素 int64 三重循环
//     三方对拍；M/N/K 尾部（1..17 / 补零面板 / K%4、K%8、K%32 尾组）、accum 双态、
//     K=0 边界、满幅值（255×-128 / 32767×-32768）、K=65536 全幅（文档安全域上界）；
//   - pack_b_i8：面板布局与朴素转置语义对拍（补零区为 0）；
//   - gemm_i16 的 -32768 边界（mullo 逐乘积路径无 pair 溢出问题，专项满幅验证）。
//
// 构建：独立可执行（见 mkern/gemm/CMakeLists.txt），不依赖 pybind11/Python。

#include "mkern/gemm/gemm_api.h"
#include "mkern/simd/simd_api.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

// 标量锚点手动声明（同 simd_boundary_test 对 dot*_scalar 的模式：锚点定义于
// 后端实现文件、不进公共接口头，测试直接对拍用）。注意必须放在全局作用域的
// 具名 namespace 内——放匿名 namespace 里会构造出 ::{anonymous}::sgn::... 假命名空间。
namespace sgn::mkern::gemm {
void gemm_i8_scalar(int32_t*, const uint8_t*, const int8_t*, int64_t, int64_t, int64_t, bool);
void gemm_i16_scalar(int64_t*, const int16_t*, const int16_t*, int64_t, int64_t, int64_t, bool);
} // namespace sgn::mkern::gemm
namespace sgn::simd {
int64_t dot8_scalar(const uint8_t*, const int8_t*, size_t);
} // namespace sgn::simd

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

unsigned int g_rng = 0x12345678u;
inline unsigned int rng() {
    g_rng = g_rng * 1664525u + 1013904223u;
    return g_rng;
}

// 朴素 int64 参照（i8：Bp 面板布局逐字节收集；等价于 A×B 语义）
int64_t ref_dot8_packed(const uint8_t* a, const int8_t* Bp, int64_t N, int64_t n,
                        int64_t K) {
    const int64_t ng = (K + 3) / 4;
    const int8_t* vb = Bp + (n / 16) * ng * 64 + (n % 16) * 4;
    int64_t sum = 0;
    for (int64_t g = 0; g < ng; ++g) {
        for (int64_t j = 0; j < 4; ++j) {
            const int64_t k = g * 4 + j;
            if (k < K) sum += static_cast<int64_t>(a[k]) * static_cast<int64_t>(vb[g * 64 + j]);
        }
    }
    return sum;
}

} // namespace

static void test_pack_b_i8() {
    for (int64_t N : {1, 15, 16, 17, 33, 64}) {
        for (int64_t K : {1, 3, 4, 5, 7, 8, 33, 100}) {
            std::vector<int8_t> B(static_cast<size_t>(K * N));
            for (auto& v : B) v = static_cast<int8_t>(rng() % 256) - 128;  // 满幅
            std::vector<int8_t> Bp(static_cast<size_t>(
                sgn::mkern::gemm::pack_b_i8_bytes(N, K)), 0);
            sgn::mkern::gemm::pack_b_i8(Bp.data(), B.data(), N, K);
            // 布局对拍：byte[t*4+j'] == B[(4g+j')*N + 16p+t]，补零区为 0
            const int64_t np = (N + 15) / 16, ng = (K + 3) / 4;
            bool ok = true;
            for (int64_t p = 0; p < np && ok; ++p) {
                for (int64_t g = 0; g < ng && ok; ++g) {
                    for (int64_t t = 0; t < 16 && ok; ++t) {
                        for (int64_t j = 0; j < 4 && ok; ++j) {
                            const int64_t k = g * 4 + j, n = p * 16 + t;
                            const int8_t expect =
                                (k < K && n < N) ? B[k * N + n] : 0;
                            ok = (Bp[(p * ng + g) * 64 + t * 4 + j] == expect);
                        }
                    }
                }
            }
            CHECK(ok, "pack_b_i8 layout mismatch");
        }
    }
}

static void test_gemm_i8() {
    using namespace sgn::mkern::gemm;
    for (int64_t M : {1, 2, 3, 4, 5, 7, 8, 17, 33}) {
        for (int64_t N : {1, 3, 15, 16, 17, 31, 33, 64}) {
            for (int64_t K : {1, 3, 4, 5, 7, 8, 31, 32, 33, 100, 257}) {
                std::vector<uint8_t> A(static_cast<size_t>(M * K));
                std::vector<int8_t> B(static_cast<size_t>(K * N));
                for (auto& v : A) v = static_cast<uint8_t>(rng() % 256);      // 满幅
                for (auto& v : B) v = static_cast<int8_t>(rng() % 256) - 128; // 满幅
                std::vector<int8_t> Bp(static_cast<size_t>(pack_b_i8_bytes(N, K)), 0);
                pack_b_i8(Bp.data(), B.data(), N, K);

                for (bool accum : {false, true}) {
                    std::vector<int32_t> c1(static_cast<size_t>(M * N), accum ? 7 : 0);
                    std::vector<int32_t> c2(static_cast<size_t>(M * N), accum ? 7 : 0);
                    std::vector<int64_t> ref(static_cast<size_t>(M * N), accum ? 7 : 0);
                    gemm_i8(c1.data(), A.data(), Bp.data(), M, N, K, accum);
                    gemm_i8_scalar(c2.data(), A.data(), Bp.data(), M, N, K, accum);
                    for (int64_t m = 0; m < M; ++m) {
                        for (int64_t n = 0; n < N; ++n) {
                            int64_t s = ref_dot8_packed(
                                A.data() + m * K, Bp.data(), N, n, K);
                            ref[m * N + n] += static_cast<int32_t>(s);
                        }
                    }
                    int bad1 = -1, bad2 = -1;
                    for (int64_t i = 0; i < M * N; ++i) {
                        if (c1[i] != ref[i] && bad1 < 0) bad1 = static_cast<int>(i);
                        if (c2[i] != ref[i] && bad2 < 0) bad2 = static_cast<int>(i);
                    }
                    if (bad1 >= 0 || bad2 >= 0) {
                        std::printf("  [M=%lld N=%lld K=%lld accum=%d] bad1=%d bad2=%d "
                                    "c1=%d c2=%d ref=%lld\n",
                                    (long long)M, (long long)N, (long long)K, (int)accum,
                                    bad1, bad2,
                                    bad1 >= 0 ? c1[bad1] : 0, bad2 >= 0 ? c2[bad2] : 0,
                                    bad1 >= 0 ? (long long)ref[bad1] : (long long)ref[bad2]);
                    }
                    CHECK(bad1 < 0, "gemm_i8 dispatch vs naive ref mismatch");
                    CHECK(bad2 < 0, "gemm_i8 scalar vs naive ref mismatch");
                }
            }
        }
    }
    // K=65536 全幅（文档安全域上界：65536×255×128 = 2.139e9 < 2^31）
    {
        const int64_t M = 4, N = 16, K = 65536;
        std::vector<uint8_t> A(static_cast<size_t>(M * K), 255);
        std::vector<int8_t> B(static_cast<size_t>(K * N), -128);
        std::vector<int8_t> Bp(static_cast<size_t>(pack_b_i8_bytes(N, K)), 0);
        pack_b_i8(Bp.data(), B.data(), N, K);
        std::vector<int32_t> c(static_cast<size_t>(M * N), 0);
        gemm_i8(c.data(), A.data(), Bp.data(), M, N, K, false);
        const int32_t expect = static_cast<int32_t>(65536LL * 255 * -128);
        bool ok = true;
        for (auto v : c) ok = ok && (v == expect);
        CHECK(ok, "gemm_i8 K=65536 full-scale mismatch");
    }
    // K=0 边界：覆盖写清零 / 累加保持
    {
        const int64_t M = 3, N = 17;
        std::vector<uint8_t> A(4, 1);
        std::vector<int8_t> Bp(static_cast<size_t>(pack_b_i8_bytes(N, 0)), 0);
        std::vector<int32_t> c1(static_cast<size_t>(M * N), 5), c2(static_cast<size_t>(M * N), 5);
        gemm_i8(c1.data(), A.data(), Bp.data(), M, N, 0, false);
        gemm_i8(c2.data(), A.data(), Bp.data(), M, N, 0, true);
        bool ok1 = true, ok2 = true;
        for (auto v : c1) ok1 = ok1 && (v == 0);
        for (auto v : c2) ok2 = ok2 && (v == 5);
        CHECK(ok1, "gemm_i8 K=0 overwrite must zero C");
        CHECK(ok2, "gemm_i8 K=0 accumulate must keep C");
    }
}

static void test_gemm_i16() {
    using namespace sgn::mkern::gemm;
    for (int64_t M : {1, 2, 3, 4, 5, 8, 17}) {
        for (int64_t N : {1, 3, 7, 8, 9, 16, 33}) {
            for (int64_t K : {1, 3, 8, 16, 17, 100, 257}) {
                std::vector<int16_t> A(static_cast<size_t>(M * K));
                std::vector<int16_t> B(static_cast<size_t>(K * N));
                for (auto& v : A) v = static_cast<int16_t>(rng() % 65536) - 32768; // 满幅含 -32768
                for (auto& v : B) v = static_cast<int16_t>(rng() % 65536) - 32768;
                for (bool accum : {false, true}) {
                    std::vector<int64_t> c1(static_cast<size_t>(M * N), accum ? -9 : 0);
                    std::vector<int64_t> c2(static_cast<size_t>(M * N), accum ? -9 : 0);
                    std::vector<int64_t> ref(static_cast<size_t>(M * N), accum ? -9 : 0);
                    gemm_i16(c1.data(), A.data(), B.data(), M, N, K, accum);
                    gemm_i16_scalar(c2.data(), A.data(), B.data(), M, N, K, accum);
                    for (int64_t m = 0; m < M; ++m) {
                        for (int64_t n = 0; n < N; ++n) {
                            int64_t s = 0;
                            for (int64_t k = 0; k < K; ++k) {
                                s += static_cast<int64_t>(A[m * K + k]) *
                                     static_cast<int64_t>(B[k * N + n]);
                            }
                            ref[m * N + n] += s;
                        }
                    }
                    CHECK(std::memcmp(c1.data(), ref.data(), sizeof(int64_t) * M * N) == 0,
                          "gemm_i16 dispatch vs naive ref mismatch");
                    CHECK(std::memcmp(c2.data(), ref.data(), sizeof(int64_t) * M * N) == 0,
                          "gemm_i16 scalar vs naive ref mismatch");
                }
            }
        }
    }
    // 满幅专项：全 -32768（mullo 逐乘积路径无 pair 溢出；int64 累加精确）
    {
        const int64_t M = 4, N = 8, K = 1024;
        std::vector<int16_t> A(static_cast<size_t>(M * K), -32768);
        std::vector<int16_t> B(static_cast<size_t>(K * N), -32768);
        std::vector<int64_t> c(static_cast<size_t>(M * N), 0);
        gemm_i16(c.data(), A.data(), B.data(), M, N, K, false);
        const int64_t expect = 1024LL * 32768LL * 32768LL;   // 2^40，int64 内
        bool ok = true;
        for (auto v : c) ok = ok && (v == expect);
        CHECK(ok, "gemm_i16 all -32768 full-scale mismatch");
    }
    // K=0 边界
    {
        std::vector<int64_t> c1(6, 3), c2(6, 3);
        gemm_i16(c1.data(), nullptr, nullptr, 2, 3, 0, false);
        gemm_i16(c2.data(), nullptr, nullptr, 2, 3, 0, true);
        bool ok1 = true, ok2 = true;
        for (auto v : c1) ok1 = ok1 && (v == 0);
        for (auto v : c2) ok2 = ok2 && (v == 3);
        CHECK(ok1, "gemm_i16 K=0 overwrite must zero C");
        CHECK(ok2, "gemm_i16 K=0 accumulate must keep C");
    }
}

static void test_gemv_i8() {
    using namespace sgn::mkern::gemm;
    for (int64_t M : {1, 3, 8, 33}) {
        for (int64_t K : {1, 7, 32, 33, 1000}) {
            std::vector<uint8_t> A(static_cast<size_t>(M * K));
            std::vector<int8_t> x(static_cast<size_t>(K));
            for (auto& v : A) v = static_cast<uint8_t>(rng() % 256);
            for (auto& v : x) v = static_cast<int8_t>(rng() % 256) - 128;
            std::vector<int64_t> y(static_cast<size_t>(M), 0);
            gemv_i8(y.data(), A.data(), x.data(), M, K);
            for (int64_t m = 0; m < M; ++m) {
                const int64_t expect =
                    sgn::simd::dot8_scalar(A.data() + m * K, x.data(), K);
                CHECK(y[m] == expect, "gemv_i8 vs dot8_scalar mismatch");
            }
        }
    }
}

int main() {
    std::printf("=== mkern/gemm boundary test ===\n");
    std::printf("gemm backend: %s | simd backend: %s\n\n",
                sgn::mkern::gemm::active_gemm_backend_name(),
                sgn::simd::active_backend_name());
    test_pack_b_i8();
    test_gemm_i8();
    test_gemm_i16();
    test_gemv_i8();
    std::printf("\n%d/%d checks passed\n", g_total - g_failed, g_total);
    if (g_failed == 0) {
        std::printf("ALL GEMM BOUNDARY TESTS PASSED\n");
        return 0;
    }
    std::printf("GEMM BOUNDARY TESTS FAILED\n");
    return 1;
}
