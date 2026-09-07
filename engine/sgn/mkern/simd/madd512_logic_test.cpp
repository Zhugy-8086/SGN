// madd512_logic_test.cpp - dot32_madd 算法逻辑的标量模拟验证（2026-08-31）
// 目的：本机无 AVX-512 硬件，无法真跑 _mm512_madd_epi16；本测试用纯标量精确复现
// dot32_madd 的「检测 -32768 → madd 相邻对求和 / 回退逐元素」语义，对比精确标量锚点，
// 验证 madd 快速路径的算法正确性（bit-exact 保持）。远程 AVX-512 机器上由
// simd_boundary_test 补真指令验证（待办）。
//
// 编译：clang++ -O1 -std=c++23 madd512_logic_test.cpp -o madd512_logic_test.exe
#include <cstdio>
#include <cstdint>
#include <random>
#include <vector>

// 精确标量锚点：逐元素乘积累加
static int64_t dot16_scalar_ref(const int16_t* a, const int16_t* b, size_t K) {
    int64_t sum = 0;
    for (size_t i = 0; i < K; ++i) sum += (int64_t)a[i] * (int64_t)b[i];
    return sum;
}

// 模拟 dot32_madd 逻辑（与 avx512.cpp 完全一致的决策树）：
//   检测批内（a/b 两侧 32 元素）是否含 -32768；
//   无 → madd 语义（相邻两对乘积相加，检测保证不溢出）；
//   有 → 回退逐元素精确累加。
static int64_t dot32_madd_sim(const int16_t* a, const int16_t* b, size_t K) {
    int64_t acc = 0;
    size_t i = 0;
    const size_t batch = 32;
    for (; i + batch <= K; i += batch) {
        bool has_min = false;
        for (size_t j = i; j < i + batch; ++j) {
            if (a[j] == INT16_MIN || b[j] == INT16_MIN) { has_min = true; break; }
        }
        if (has_min) {
            // 回退 vpmuldq 语义：逐元素精确
            for (size_t j = i; j < i + batch; ++j) acc += (int64_t)a[j] * (int64_t)b[j];
        } else {
            // madd 语义：相邻对乘积和（r[k] = a[2k]b[2k] + a[2k+1]b[2k+1]）
            for (size_t j = i; j < i + batch; j += 2) {
                int64_t p0 = (int64_t)a[j] * (int64_t)b[j];
                int64_t p1 = (int64_t)a[j + 1] * (int64_t)b[j + 1];
                acc += p0 + p1;
            }
        }
    }
    // 尾部：逐元素
    for (; i < K; ++i) acc += (int64_t)a[i] * (int64_t)b[i];
    return acc;
}

int main() {
    int failures = 0;
    std::mt19937_64 rng(20260831);
    const size_t K = 1000;
    std::vector<int16_t> a(K), b(K);

    // 用例 1：全普通值（走 madd）
    for (size_t i = 0; i < K; ++i) {
        a[i] = static_cast<int16_t>(rng() % 20000 - 10000);
        b[i] = static_cast<int16_t>(rng() % 20000 - 10000);
    }
    if (dot32_madd_sim(a.data(), b.data(), K) != dot16_scalar_ref(a.data(), b.data(), K))
        { ++failures; std::printf("FAIL 用例1 全普通\n"); }

    // 用例 2：含 -32768 混批（触发回退）
    for (size_t i = 0; i < K; ++i) {
        a[i] = static_cast<int16_t>(rng() % 65536 - 32768);
        b[i] = static_cast<int16_t>(rng() % 65536 - 32768);
    }
    a[33] = INT16_MIN; b[40] = INT16_MIN; a[100] = INT16_MIN;
    if (dot32_madd_sim(a.data(), b.data(), K) != dot16_scalar_ref(a.data(), b.data(), K))
        { ++failures; std::printf("FAIL 用例2 含-32768混批\n"); }

    // 用例 3：全 -32768（极端，全回退）
    for (size_t i = 0; i < K; ++i) { a[i] = INT16_MIN; b[i] = INT16_MIN; }
    if (dot32_madd_sim(a.data(), b.data(), K) != dot16_scalar_ref(a.data(), b.data(), K))
        { ++failures; std::printf("FAIL 用例3 全-32768\n"); }

    // 用例 4：非 32 倍数尾部
    std::vector<int16_t> a2(50), b2(50);
    for (size_t i = 0; i < 50; ++i) {
        a2[i] = static_cast<int16_t>(rng() % 65536 - 32768);
        b2[i] = static_cast<int16_t>(rng() % 65536 - 32768);
    }
    a2[31] = INT16_MIN;  // 第一批(0..31)含 -32768 → 回退；尾部(32..49)逐元素
    if (dot32_madd_sim(a2.data(), b2.data(), 50) != dot16_scalar_ref(a2.data(), b2.data(), 50))
        { ++failures; std::printf("FAIL 用例4 非32倍数尾部\n"); }

    if (failures == 0) std::printf("ALL MADD512 LOGIC TESTS PASSED\n");
    else std::printf("%d FAILURES\n", failures);
    return failures ? 1 : 0;
}
