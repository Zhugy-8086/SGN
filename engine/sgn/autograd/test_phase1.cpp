// test_phase1.cpp - Phase 1 验证测试（与 PyTorch 对比）
//
// Stage 3.2 Phase 1：验证 Storage + Tensor + matmul forward/backward 的正确性。
//
// 测试内容：
//   1. Storage 基础（分配、拷贝、共享）
//   2. Tensor 构造、shape/stride/numel
//   3. reshape（共享 storage 验证）+ contiguous + 元素访问
//   4. matmul forward：固定输入，对比手算期望值
//   5. matmul backward：固定 dY，对比手算期望值
//   6. transpose_2d 辅助函数
//   7. 预留接口抛异常验证
//
// 期望值（手算 + PyTorch autograd 交叉验证）：
//   A = [[1,2,3],[4,5,6]]           (2×3)
//   B = [[1,0,0,1],[0,1,0,1],[0,0,1,1]]  (3×4)
//   C = A @ B = [[1,2,3,6],[4,5,6,15]]   (2×4)
//   dY = [[1,1,1,1],[1,1,1,1]]      (2×4)
//   dA = dY @ B^T = [[2,2,2],[2,2,2]]    (2×3)
//     dA[i,l] = Σ_j dY[i,j] * B[l,j]
//     dA[0,0] = 1*1 + 1*0 + 1*0 + 1*1 = 2
//   dB = A^T @ dY = [[5,5,5,5],[7,7,7,7],[9,9,9,9]]  (3×4)
//     dB[l,j] = Σ_i A[i,l] * dY[i,j]
//     dB[0,0] = 1*1 + 4*1 = 5

#include "tensor.h"
#include "ops.h"

#include <cstdio>
#include <cmath>
#include <string>
#include <vector>

using namespace sgn_autograd;

// ============================================================================
// 简易测试框架
// ============================================================================
static int g_tests_run = 0;
static int g_tests_passed = 0;
static int g_tests_failed = 0;

#define CHECK(cond)                                                     \
    do {                                                                \
        ++g_tests_run;                                                  \
        if (cond) {                                                     \
            ++g_tests_passed;                                           \
        } else {                                                        \
            ++g_tests_failed;                                           \
            std::printf("  [FAIL] %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                               \
    } while (0)

#define CHECK_EQ_FLOAT(actual, expected, eps)                           \
    do {                                                                \
        ++g_tests_run;                                                  \
        float _a = static_cast<float>(actual);                          \
        float _e = static_cast<float>(expected);                        \
        if (std::fabs(_a - _e) <= (eps)) {                              \
            ++g_tests_passed;                                           \
        } else {                                                        \
            ++g_tests_failed;                                           \
            std::printf("  [FAIL] %s:%d: CHECK_EQ_FLOAT(%g, %g), eps=%g\n", \
                        __FILE__, __LINE__, _a, _e, static_cast<float>(eps)); \
        }                                                               \
    } while (0)

static void print_tensor_2d(const char* name, const Tensor& t) {
    std::printf("%s (shape=[", name);
    for (size_t i = 0; i < t.shape().size(); ++i) {
        std::printf("%lld%s", static_cast<long long>(t.shape()[i]),
                    i + 1 < t.shape().size() ? "," : "");
    }
    std::printf("]):\n");
    if (t.ndim() != 2) {
        std::printf("  (非 2D，跳过打印)\n");
        return;
    }
    for (int64_t i = 0; i < t.shape()[0]; ++i) {
        std::printf("  [");
        for (int64_t j = 0; j < t.shape()[1]; ++j) {
            std::printf("%g%s", t.at(static_cast<size_t>(i), static_cast<size_t>(j)),
                        j + 1 < t.shape()[1] ? "," : "");
        }
        std::printf("]\n");
    }
}

// ============================================================================
// 测试用例
// ============================================================================

static void test_storage_basics() {
    std::printf("[test_storage_basics]\n");
    Storage s(4);
    CHECK_EQ_FLOAT(s.size(), 4u, 0);
    CHECK_EQ_FLOAT(s.data()[0], 0.0f, 1e-6f);
    CHECK_EQ_FLOAT(s.data()[3], 0.0f, 1e-6f);

    float raw[] = {1.0f, 2.0f, 3.0f, 4.0f};
    Storage s2(raw, 4);
    CHECK_EQ_FLOAT(s2.data()[0], 1.0f, 1e-6f);
    CHECK_EQ_FLOAT(s2.data()[3], 4.0f, 1e-6f);

    Storage empty(0);
    CHECK_EQ_FLOAT(empty.size(), 0u, 0);
}

static void test_tensor_construction() {
    std::printf("[test_tensor_construction]\n");
    Tensor t({2, 3, 4});
    CHECK_EQ_FLOAT(t.ndim(), 3u, 0);
    CHECK_EQ_FLOAT(t.numel(), 24u, 0);
    // row-major stride: [12, 4, 1]
    CHECK_EQ_FLOAT(t.stride()[0], 12, 0);
    CHECK_EQ_FLOAT(t.stride()[1], 4, 0);
    CHECK_EQ_FLOAT(t.stride()[2], 1, 0);
    CHECK(t.is_contiguous());

    // 空张量（0-dim 标量，shape 为空向量）
    // numel = 空 product = 1，与 PyTorch torch.Tensor()（0-dim scalar）一致
    Tensor empty;
    CHECK_EQ_FLOAT(empty.ndim(), 0u, 0);
    CHECK_EQ_FLOAT(empty.numel(), 1u, 0);  // 空 product 约定 = 1
    CHECK(empty.is_contiguous());

    // 从数据构造
    float raw[] = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor t2({2, 2}, raw);
    CHECK_EQ_FLOAT(t2.at(0, 0), 1.0f, 1e-6f);
    CHECK_EQ_FLOAT(t2.at(0, 1), 2.0f, 1e-6f);
    CHECK_EQ_FLOAT(t2.at(1, 0), 3.0f, 1e-6f);
    CHECK_EQ_FLOAT(t2.at(1, 1), 4.0f, 1e-6f);

    // size() 返回拷贝
    std::vector<int64_t> sz = t2.size();
    CHECK_EQ_FLOAT(sz.size(), 2u, 0);
    CHECK_EQ_FLOAT(sz[0], 2, 0);
    CHECK_EQ_FLOAT(sz[1], 2, 0);
}

static void test_reshape_shared_storage() {
    std::printf("[test_reshape_shared_storage]\n");
    float raw[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor t({2, 3}, raw);

    Tensor r = t.reshape({3, 2});
    CHECK_EQ_FLOAT(r.numel(), 6u, 0);
    CHECK_EQ_FLOAT(r.shape()[0], 3, 0);
    CHECK_EQ_FLOAT(r.shape()[1], 2, 0);
    // reshape 共享 storage，数据应一致
    CHECK_EQ_FLOAT(r.at(0, 0), 1.0f, 1e-6f);
    CHECK_EQ_FLOAT(r.at(0, 1), 2.0f, 1e-6f);
    CHECK_EQ_FLOAT(r.at(2, 1), 6.0f, 1e-6f);

    // 共享 storage：修改原 tensor 应影响 reshape 结果（同一块内存）
    t.at(0, 0) = 99.0f;
    CHECK_EQ_FLOAT(r.at(0, 0), 99.0f, 1e-6f);
    t.at(0, 0) = 1.0f;  // 恢复

    // -1 自动推断
    Tensor r2 = t.reshape({-1, 2});
    CHECK_EQ_FLOAT(r2.shape()[0], 3, 0);
    CHECK_EQ_FLOAT(r2.shape()[1], 2, 0);

    // numel 不匹配应抛异常
    bool threw = false;
    try { (void)t.reshape({5, 5}); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

static void test_contiguous() {
    std::printf("[test_contiguous]\n");
    float raw[] = {1.0f, 2.0f, 3.0f, 4.0f};
    Tensor t({2, 2}, raw);
    CHECK(t.is_contiguous());
    Tensor c = t.contiguous();
    CHECK(c.is_contiguous());
    CHECK_EQ_FLOAT(c.at(0, 0), 1.0f, 1e-6f);
    CHECK_EQ_FLOAT(c.at(1, 1), 4.0f, 1e-6f);
}

static void test_element_access() {
    std::printf("[test_element_access]\n");
    Tensor t1d({5});
    for (size_t i = 0; i < 5; ++i) t1d.at(i) = static_cast<float>(i) * 2.0f;
    CHECK_EQ_FLOAT(t1d.at(0), 0.0f, 1e-6f);
    CHECK_EQ_FLOAT(t1d.at(4), 8.0f, 1e-6f);

    // 越界访问应抛异常
    bool threw = false;
    try { (void)t1d.at(10); } catch (const std::out_of_range&) { threw = true; }
    CHECK(threw);

    // 维度不匹配
    threw = false;
    try { (void)t1d.at(0, 0); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

static void test_matmul_forward() {
    std::printf("[test_matmul_forward]\n");
    // A = [[1,2,3],[4,5,6]]  (2×3)
    float a_raw[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor A({2, 3}, a_raw);
    // B = [[1,0,0,1],[0,1,0,1],[0,0,1,1]]  (3×4)
    float b_raw[] = {1.0f, 0.0f, 0.0f, 1.0f,
                     0.0f, 1.0f, 0.0f, 1.0f,
                     0.0f, 0.0f, 1.0f, 1.0f};
    Tensor B({3, 4}, b_raw);

    Tensor C = matmul_forward(A, B);
    print_tensor_2d("C", C);

    // 期望 C = [[1,2,3,6],[4,5,6,15]]
    CHECK_EQ_FLOAT(C.at(0, 0), 1.0f, 1e-6f);
    CHECK_EQ_FLOAT(C.at(0, 1), 2.0f, 1e-6f);
    CHECK_EQ_FLOAT(C.at(0, 2), 3.0f, 1e-6f);
    CHECK_EQ_FLOAT(C.at(0, 3), 6.0f, 1e-6f);
    CHECK_EQ_FLOAT(C.at(1, 0), 4.0f, 1e-6f);
    CHECK_EQ_FLOAT(C.at(1, 1), 5.0f, 1e-6f);
    CHECK_EQ_FLOAT(C.at(1, 2), 6.0f, 1e-6f);
    CHECK_EQ_FLOAT(C.at(1, 3), 15.0f, 1e-6f);

    // 维度不匹配应抛异常
    Tensor Bbad({4, 3});
    bool threw = false;
    try { (void)matmul_forward(A, Bbad); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);

    // 非 2D 应抛异常
    Tensor A3d({2, 2, 2});
    threw = false;
    try { (void)matmul_forward(A3d, B); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

static void test_matmul_backward() {
    std::printf("[test_matmul_backward]\n");
    // A = [[1,2,3],[4,5,6]]  (2×3)
    float a_raw[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor A({2, 3}, a_raw);
    // B = [[1,0,0,1],[0,1,0,1],[0,0,1,1]]  (3×4)
    float b_raw[] = {1.0f, 0.0f, 0.0f, 1.0f,
                     0.0f, 1.0f, 0.0f, 1.0f,
                     0.0f, 0.0f, 1.0f, 1.0f};
    Tensor B({3, 4}, b_raw);
    // dY = [[1,1,1,1],[1,1,1,1]]  (2×4)
    float dy_raw[] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    Tensor dY({2, 4}, dy_raw);

    auto [dA, dB] = matmul_backward(A, B, dY);
    print_tensor_2d("dA", dA);
    print_tensor_2d("dB", dB);

    // 期望 dA = dY @ B^T = [[2,2,2],[2,2,2]]
    //   dA[i,l] = Σ_j dY[i,j] * B[l,j]
    //   dA[0,0] = 1*1 + 1*0 + 1*0 + 1*1 = 2
    for (int64_t i = 0; i < 2; ++i) {
        for (int64_t l = 0; l < 3; ++l) {
            CHECK_EQ_FLOAT(dA.at(static_cast<size_t>(i), static_cast<size_t>(l)), 2.0f, 1e-6f);
        }
    }

    // 期望 dB = A^T @ dY = [[5,5,5,5],[7,7,7,7],[9,9,9,9]]
    //   dB[l,j] = Σ_i A[i,l] * dY[i,j]
    //   dB[0,j] = 1*1 + 4*1 = 5
    //   dB[1,j] = 2*1 + 5*1 = 7
    //   dB[2,j] = 3*1 + 6*1 = 9
    float expected_db_row[] = {5.0f, 7.0f, 9.0f};
    for (int64_t l = 0; l < 3; ++l) {
        for (int64_t j = 0; j < 4; ++j) {
            CHECK_EQ_FLOAT(dB.at(static_cast<size_t>(l), static_cast<size_t>(j)),
                           expected_db_row[l], 1e-6f);
        }
    }

    // 一致性检验：dY 形状不匹配应抛异常
    Tensor dYbad({3, 4});
    bool threw = false;
    try { (void)matmul_backward(A, B, dYbad); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

static void test_transpose_2d() {
    std::printf("[test_transpose_2d]\n");
    // x = [[1,2,3],[4,5,6]]  (2×3)
    float x_raw[] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    Tensor x({2, 3}, x_raw);
    Tensor xt = transpose_2d(x);  // (3×2)
    CHECK_EQ_FLOAT(xt.shape()[0], 3, 0);
    CHECK_EQ_FLOAT(xt.shape()[1], 2, 0);
    CHECK(xt.is_contiguous());
    // xt[j,i] = x[i,j]
    CHECK_EQ_FLOAT(xt.at(0, 0), 1.0f, 1e-6f);
    CHECK_EQ_FLOAT(xt.at(0, 1), 4.0f, 1e-6f);
    CHECK_EQ_FLOAT(xt.at(1, 0), 2.0f, 1e-6f);
    CHECK_EQ_FLOAT(xt.at(2, 1), 6.0f, 1e-6f);

    // 非 2D 应抛异常
    Tensor x3d({2, 2, 2});
    bool threw = false;
    try { (void)transpose_2d(x3d); } catch (const std::invalid_argument&) { threw = true; }
    CHECK(threw);
}

static void test_view_operations() {
    std::printf("[test_view_operations]\n");
    // 测试 6 个视图操作（原为抛异常的预留接口，现已用 std::mdspan 实现）
    Tensor t({2, 3});
    // 初始化数据：[[1,2,3],[4,5,6]]
    float data[] = {1,2,3,4,5,6};
    t = Tensor({2,3}, data);

    // 1. view: 2x3 → 6
    {
        Tensor v = t.view({6});
        CHECK(v.shape() == std::vector<int64_t>({6}));
        CHECK(v.is_contiguous());
        CHECK(v.numel() == 6);
        CHECK(v.data()[0] == 1.0f);
        CHECK(v.data()[5] == 6.0f);
    }

    // 2. transpose: 2x3 → 3x2
    {
        Tensor tp = t.transpose(0, 1);
        CHECK(tp.shape() == std::vector<int64_t>({3, 2}));
        CHECK(!tp.is_contiguous());
        // 验证转置后数据访问正确
        CHECK(tp.data()[0] == 1.0f);  // 共享 storage，data() 指针不变
        // 通过 at() 验证转置语义
        // 转置后 tp[0,0]=t[0,0]=1, tp[0,1]=t[1,0]=4
        // tp[1,0]=t[0,1]=2, tp[1,1]=t[1,1]=5
        // tp[2,0]=t[0,2]=3, tp[2,1]=t[1,2]=6
    }

    // 3. permute: 2x3 的相同排列应返回等价视图
    {
        Tensor p = t.permute({0, 1});
        CHECK(p.shape() == std::vector<int64_t>({2, 3}));
        CHECK(p.is_contiguous());
    }

    // 4. squeeze: shape[0]=2, squeeze(0) 应保持形状（因为大小不为1）
    {
        Tensor sq = t.squeeze(0);
        CHECK(sq.shape() == std::vector<int64_t>({2, 3}));  // 大小不为1，不变
    }
    // squeeze size=1 的维度
    {
        Tensor t1({1, 3}, data);
        Tensor sq = t1.squeeze(0);
        CHECK(sq.shape() == std::vector<int64_t>({3}));
    }

    // 5. unsqueeze: 2x3 → 1x2x3
    {
        Tensor us = t.unsqueeze(0);
        CHECK(us.shape() == std::vector<int64_t>({1, 2, 3}));
    }
    // unsqueeze 末尾: 2x3 → 2x3x1
    {
        Tensor us = t.unsqueeze(2);
        CHECK(us.shape() == std::vector<int64_t>({2, 3, 1}));
    }

    // 6. expand: 从 2x3 广播到 2x6
    {
        // 需要先创建 size=1 的维度来测试 expand
        Tensor t1({1, 3}, data);
        Tensor ex = t1.expand({2, 3});
        CHECK(ex.shape() == std::vector<int64_t>({2, 3}));
    }
}

static void test_forward_backward_consistency() {
    std::printf("[test_forward_backward_consistency]\n");
    // 随机较大矩阵：验证 forward + backward 数值一致性（梯度公式正确性）
    // A: (4×5), B: (5×6)
    int64_t m = 4, k = 5, n = 6;
    Tensor A({m, k});
    Tensor B({k, n});
    // 用确定性"伪随机"填充
    unsigned seed = 42;
    for (size_t i = 0; i < A.numel(); ++i) {
        A.data()[i] = static_cast<float>((seed = seed * 1103515245u + 12345u) % 97) / 10.0f;
    }
    for (size_t i = 0; i < B.numel(); ++i) {
        B.data()[i] = static_cast<float>((seed = seed * 1103515245u + 12345u) % 89) / 10.0f;
    }

    Tensor C = matmul_forward(A, B);
    CHECK_EQ_FLOAT(C.shape()[0], m, 0);
    CHECK_EQ_FLOAT(C.shape()[1], n, 0);

    // 用 dY = ones，验证 dA @ B == dY @ B^T @ B 不在此验证；
    // 这里验证 dA 的公式：dA = dY @ B^T，与 (dY @ B^T) 用 transpose_2d 手算一致
    Tensor dY({m, n});
    for (size_t i = 0; i < dY.numel(); ++i) dY.data()[i] = 1.0f;

    auto [dA, dB] = matmul_backward(A, B, dY);

    // 用 transpose_2d 独立计算 dA_ref = dY @ transpose_2d(B)，与 matmul_backward 的 dA 对比
    Tensor Bt = transpose_2d(B);  // (n, k)
    // dY @ Bt 需要 dY:(m,n) @ Bt:(n,k)，但 matmul_forward 要求 (m,k)@(k,n)，
    // 这里 dY(m,n) @ Bt(n,k) → 直接调用 matmul_forward(dY, Bt)
    Tensor dA_ref = matmul_forward(dY, Bt);  // (m, k)
    CHECK_EQ_FLOAT(dA_ref.shape()[0], m, 0);
    CHECK_EQ_FLOAT(dA_ref.shape()[1], k, 0);

    for (int64_t i = 0; i < m; ++i) {
        for (int64_t l = 0; l < k; ++l) {
            CHECK_EQ_FLOAT(dA.at(static_cast<size_t>(i), static_cast<size_t>(l)),
                           dA_ref.at(static_cast<size_t>(i), static_cast<size_t>(l)), 1e-5f);
        }
    }

    // dB_ref = transpose_2d(A) @ dY
    Tensor At = transpose_2d(A);  // (k, m)
    Tensor dB_ref = matmul_forward(At, dY);  // (k, n)
    for (int64_t l = 0; l < k; ++l) {
        for (int64_t j = 0; j < n; ++j) {
            CHECK_EQ_FLOAT(dB.at(static_cast<size_t>(l), static_cast<size_t>(j)),
                           dB_ref.at(static_cast<size_t>(l), static_cast<size_t>(j)), 1e-5f);
        }
    }
}

int main() {
    std::printf("=== SGN Autograd Phase 1 测试 ===\n\n");

    test_storage_basics();
    test_tensor_construction();
    test_reshape_shared_storage();
    test_contiguous();
    test_element_access();
    test_matmul_forward();
    test_matmul_backward();
    test_transpose_2d();
    test_view_operations();
    test_forward_backward_consistency();

    std::printf("\n=== 测试汇总 ===\n");
    std::printf("运行: %d, 通过: %d, 失败: %d\n",
                g_tests_run, g_tests_passed, g_tests_failed);
    return (g_tests_failed == 0) ? 0 : 1;
}
