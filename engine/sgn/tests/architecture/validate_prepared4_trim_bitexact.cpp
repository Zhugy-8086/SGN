// validate_prepared4_trim_bitexact.cpp - narrow_dot_prepared4 + trim_high_diag 跨编译器 bit-exact 复验
//
// 目标（对应 H3 摊销路径：prepare_nibble 一次 + M 输出复用 dot_prepared4）：
//   1. narrow_dot_prepared4(trim=False) 与标量 __int128 参考 Σ w_i·x_i bit-exact
//      （fuse_128 的 (hi, lo) 与参考 128 位逐位一致）
//   2. narrow_dot_prepared4(trim=True) 高位对角（m>=n）全 0、低位与 trim=False 逐位一致
//   3. 跨编译器一致性：Clang / GCC 分别编译运行，输出 FINAL_HASH 必须一致
//      （FNV-1a 64 位，遍历全部 full/tri partials 字节）
//
// 测试数据：
//   - random：5 seeds，每 seed 覆盖 K = 1,2,8,31,32,33,63,64,100,513
//     （32/33/63/64 覆盖 vpdpbusd 主循环 + 零填充尾部边界）
//   - skewed：base 值 ±4 扰动（高相似、位模式集中）
//   - boundary：全 0 / 全 int32_max / 全 int32_min / 极值混合 / ±1
//
// 编译（需支持 AVX-VNNI，本机 Clang 22.1.8 / GCC 16.1.0 均满足）：
//   clang++ -O3 -mavx2 -mavxvnni -std=c++23 validate_prepared4_trim_bitexact.cpp -o ../../../build/vp4_clang.exe
//   g++     -O3 -mavx2 -mavxvnni -std=c++23 validate_prepared4_trim_bitexact.cpp -o ../../../build/vp4_gcc.exe
// 运行：两个 exe 输出到文件后 diff FINAL_HASH 与 ALL PASS 行。
#include "../../msint/split_dot.cpp"

#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

using namespace sgn;

static uint64_t g_hash = 1469598103934665603ULL;  // FNV-1a 64 offset basis

static void hash_bytes(const void* p, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    for (size_t i = 0; i < n; ++i) {
        g_hash ^= b[i];
        g_hash *= 1099511628211ULL;
    }
}

// 权重/激活侧统一构建：split_parts + pack_narrow_value → prepare_nibble
static NibblePrepared make_prepared(const std::vector<int64_t>& v) {
    const int total_bits = 32, sb = 4;
    const int n = total_bits / sb;
    const size_t K = v.size();
    NarrowParts np = alloc_narrow_parts(sb, n, K);
    for (size_t i = 0; i < K; ++i) {
        auto parts = SplitDot::split_parts(v[i], total_bits, sb);
        for (int a = 0; a < n; ++a) {
            pack_narrow_value(np, static_cast<size_t>(a), i, parts[static_cast<size_t>(a)]);
        }
    }
    return prepare_nibble(np);
}

static bool check(const std::vector<int64_t>& w, const std::vector<int64_t>& x,
                  const char* label) {
    auto pw = make_prepared(w);
    auto px = make_prepared(x);
    auto full = narrow_dot_prepared4(pw, px, false);
    auto tri = narrow_dot_prepared4(pw, px, true);

    // 参考：__int128 精确累加 Σ w_i·x_i → 拆 (hi, lo) 与 fuse_128 对比
    __int128 ref = 0;
    for (size_t i = 0; i < w.size(); ++i) {
        ref += static_cast<__int128>(w[i]) * static_cast<__int128>(x[i]);
    }
    auto [hi, lo] = SplitDot::fuse_128(full, 4);
    const int64_t  ref_hi = static_cast<int64_t>(ref >> 64);
    const uint64_t ref_lo = static_cast<uint64_t>(ref & 0xFFFFFFFFFFFFFFFFULL);
    const bool ref_ok = (hi == ref_hi && lo == ref_lo);

    // trim 语义：高位（m>=n）归零、低位（m<n）与 full 逐位一致
    const int n = 8;
    bool hi_zero = true, lo_same = true;
    for (int m = n; m < 2 * n - 1; ++m) {
        if (tri[static_cast<size_t>(m)] != 0) hi_zero = false;
    }
    for (int m = 0; m < n; ++m) {
        if (tri[static_cast<size_t>(m)] != full[static_cast<size_t>(m)]) lo_same = false;
    }

    const bool ok = ref_ok && hi_zero && lo_same;
    for (int64_t v : full) hash_bytes(&v, sizeof(v));
    for (int64_t v : tri) hash_bytes(&v, sizeof(v));

    if (!ok) {
        std::printf("[FAIL] %s ref=%c hi_zero=%d lo_same=%d "
                    "(hi=%lld lo=%llu ref_hi=%lld ref_lo=%llu)\n",
                    label, ref_ok ? 'Y' : 'N', hi_zero ? 1 : 0, lo_same ? 1 : 0,
                    static_cast<long long>(hi), static_cast<unsigned long long>(lo),
                    static_cast<long long>(ref_hi), static_cast<unsigned long long>(ref_lo));
    }
    return ok;
}

int main() {
    int fails = 0;
    const std::vector<size_t> Ks = {1, 2, 8, 31, 32, 33, 63, 64, 100, 513};
    const int64_t LO = -2147483648LL, HI = 2147483647LL;
    char buf[96];

    // 1) random multi-seed
    std::mt19937_64 rng64(12345);
    for (int seed = 0; seed < 5; ++seed) {
        std::mt19937 rng(static_cast<unsigned>(rng64()));
        std::uniform_int_distribution<int64_t> d(LO, HI);
        for (size_t K : Ks) {
            std::vector<int64_t> w(K), x(K);
            for (size_t i = 0; i < K; ++i) { w[i] = d(rng); x[i] = d(rng); }
            std::snprintf(buf, sizeof(buf), "random seed=%d K=%zu", seed, K);
            if (!check(w, x, buf)) ++fails;
        }
    }

    // 2) skewed：base ±4 扰动（位模式集中）
    {
        std::mt19937 rng(999);
        std::uniform_int_distribution<int64_t> d(LO, HI);
        for (size_t K : Ks) {
            std::vector<int64_t> w(K), x(K);
            const int64_t bw = d(rng), bx = d(rng);
            for (size_t i = 0; i < K; ++i) {
                w[i] = bw + static_cast<int64_t>(rng() % 9) - 4;
                x[i] = bx + static_cast<int64_t>(rng() % 9) - 4;
            }
            std::snprintf(buf, sizeof(buf), "skewed K=%zu", K);
            if (!check(w, x, buf)) ++fails;
        }
    }

    // 3) boundary：全同/极值/混合
    const std::vector<std::pair<int64_t, int64_t>> bvals = {
        {0, 0}, {HI, HI}, {LO, LO}, {HI, LO}, {-1, 1}};
    for (const auto& [bw, bx] : bvals) {
        for (size_t K : Ks) {
            std::vector<int64_t> w(K, bw), x(K, bx);
            std::snprintf(buf, sizeof(buf), "boundary w=%lld x=%lld K=%zu",
                          static_cast<long long>(bw), static_cast<long long>(bx), K);
            if (!check(w, x, buf)) ++fails;
        }
    }

    std::printf("FINAL_HASH 0x%016llx\n", static_cast<unsigned long long>(g_hash));
    if (fails == 0) {
        std::printf("ALL PASS\n");
        return 0;
    }
    std::printf("%d failures total\n", fails);
    return 1;
}
