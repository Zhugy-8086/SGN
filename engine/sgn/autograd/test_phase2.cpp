// test_phase2.cpp - tape-based Autograd 引擎验证
//
// Stage 3.2 Phase 2 Task A2.6：验证 autograd backward 正确性。
//
// 验证策略：
//   - 用 Phase 1 已验证的 matmul_backward（137 断言与 PyTorch 一致）作为 ground truth
//   - 对比 Tape autograd 的 backward 结果与手动链式法则计算
//
// 测试用例：
//   1. 基本链式法则: e = a @ b @ c, 验证 grad_a/grad_b/grad_c
//   2. 同一 tensor 多次使用: f = a @ a, 验证 grad 累加
//   3. requires_grad=False: a 不需梯度, 验证 grad_a=nullptr
//   4. 非录制模式: tape 未 start, matmul 不记录

#include "autograd.h"
#include "ops.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace sgn_autograd;

// ============================================================================
// 辅助：浮点近似比较
// ============================================================================
static bool approx(float a, float b, float eps = 1e-5f) {
    return std::fabs(a - b) < eps;
}

static bool tensor_eq(const Tensor& a, const Tensor& b, float eps = 1e-5f) {
    if (a.shape() != b.shape()) return false;
    size_t n = a.numel();
    if (n != b.numel()) return false;
    for (size_t i = 0; i < n; ++i) {
        if (!approx(a.data()[i], b.data()[i], eps)) return false;
    }
    return true;
}

// ============================================================================
// 辅助：创建填充数据的 Tensor
// ============================================================================
static Tensor make_tensor(const std::vector<int64_t>& shape, const std::vector<float>& data) {
    return Tensor(shape, data.data());
}

// ============================================================================
// 测试 1: 基本链式法则 e = a @ b @ c
// ============================================================================
static int test_chain_rule() {
    printf("[test_chain_rule] e = a @ b @ c\n");
    Tape& tape = Tape::current();
    tape.clear();

    // a: (2,3), b: (3,4), c: (4,5)
    std::vector<float> a_data = {1,2,3, 4,5,6};           // 2x3
    std::vector<float> b_data = {1,0,1,0, 0,1,0,1, 1,1,1,1}; // 3x4
    std::vector<float> c_data = {
        1,2,3,4,5,
        0,1,0,1,0,
        1,0,1,0,1,
        2,1,2,1,2
    }; // 4x5

    Tensor a = make_tensor({2,3}, a_data); a.set_requires_grad(true);
    Tensor b = make_tensor({3,4}, b_data); b.set_requires_grad(true);
    Tensor c = make_tensor({4,5}, c_data); c.set_requires_grad(true);

    // 前向（录制）
    tape.start_recording();
    Tensor d = matmul(a, b);   // d: (2,4)
    Tensor e = matmul(d, c);   // e: (2,5)
    tape.stop_recording();

    printf("  tape size = %zu (期望 2)\n", tape.size());
    if (tape.size() != 2) { printf("  FAIL: tape size\n"); return 1; }

    // backward: grad_e = ones(2,5)
    std::vector<float> grad_e_data(10, 1.0f);
    Tensor grad_e = make_tensor({2,5}, grad_e_data);
    tape.backward(e, grad_e);

    // === 手动计算参考梯度（用已验证的 matmul_backward）===
    // d = a @ b
    Tensor d_ref = matmul_forward(a, b);
    // e = d @ c
    // grad_d = grad_e @ c^T, grad_c = d^T @ grad_e
    auto [grad_d_ref, grad_c_ref] = matmul_backward(d_ref, c, grad_e);
    // grad_a = grad_d @ b^T, grad_b = a^T @ grad_d
    auto [grad_a_ref, grad_b_ref] = matmul_backward(a, b, grad_d_ref);

    // === 对比 autograd 结果 ===
    const Tensor* ga = tape.grad(a.id());
    const Tensor* gb = tape.grad(b.id());
    const Tensor* gc = tape.grad(c.id());

    if (!ga) { printf("  FAIL: grad_a is null\n"); return 1; }
    if (!gb) { printf("  FAIL: grad_b is null\n"); return 1; }
    if (!gc) { printf("  FAIL: grad_c is null\n"); return 1; }

    if (!tensor_eq(*ga, grad_a_ref)) { printf("  FAIL: grad_a mismatch (vs matmul_backward)\n"); return 1; }
    if (!tensor_eq(*gb, grad_b_ref)) { printf("  FAIL: grad_b mismatch (vs matmul_backward)\n"); return 1; }
    if (!tensor_eq(*gc, grad_c_ref)) { printf("  FAIL: grad_c mismatch (vs matmul_backward)\n"); return 1; }

    // === 直接对比 PyTorch 期望值（独立验证，非间接依赖 Phase 1）===
    // PyTorch 输出（e.backward(ones)）：
    //   grad_a = [[18,10,28],[18,10,28]]
    //   grad_b = [[75,10,15,40],[105,14,21,56],[135,18,27,72]]
    //   grad_c = [[14,14,14,14,14],[16,16,16,16,16],[14,14,14,14,14],[16,16,16,16,16]]
    std::vector<float> pytorch_grad_a = {18,10,28, 18,10,28};
    std::vector<float> pytorch_grad_b = {75,10,15,40, 105,14,21,56, 135,18,27,72};
    std::vector<float> pytorch_grad_c = {
        14,14,14,14,14,
        16,16,16,16,16,
        14,14,14,14,14,
        16,16,16,16,16
    };
    Tensor pa = make_tensor({2,3}, pytorch_grad_a);
    Tensor pb = make_tensor({3,4}, pytorch_grad_b);
    Tensor pc = make_tensor({4,5}, pytorch_grad_c);
    if (!tensor_eq(*ga, pa)) { printf("  FAIL: grad_a mismatch (vs PyTorch)\n"); return 1; }
    if (!tensor_eq(*gb, pb)) { printf("  FAIL: grad_b mismatch (vs PyTorch)\n"); return 1; }
    if (!tensor_eq(*gc, pc)) { printf("  FAIL: grad_c mismatch (vs PyTorch)\n"); return 1; }

    printf("  grad_a OK (vs PyTorch + matmul_backward, shape: %lldx%lld)\n", (long long)ga->shape()[0], (long long)ga->shape()[1]);
    printf("  grad_b OK (vs PyTorch + matmul_backward, shape: %lldx%lld)\n", (long long)gb->shape()[0], (long long)gb->shape()[1]);
    printf("  grad_c OK (vs PyTorch + matmul_backward, shape: %lldx%lld)\n", (long long)gc->shape()[0], (long long)gc->shape()[1]);
    printf("  PASS\n\n");
    return 0;
}

// ============================================================================
// 测试 2: 同一 tensor 多次使用 f = a @ a（grad 累加）
// ============================================================================
static int test_reuse_same_tensor() {
    printf("[test_reuse_same_tensor] f = a @ a (grad 累加)\n");
    Tape& tape = Tape::current();
    tape.clear();

    // a: (3,3) 方阵
    std::vector<float> a_data = {1,2,3, 0,1,4, 5,6,0}; // 3x3
    Tensor a = make_tensor({3,3}, a_data); a.set_requires_grad(true);

    tape.start_recording();
    Tensor f = matmul(a, a);  // f: (3,3)
    tape.stop_recording();

    // backward: grad_f = ones(3,3)
    std::vector<float> grad_f_data(9, 1.0f);
    Tensor grad_f = make_tensor({3,3}, grad_f_data);
    tape.backward(f, grad_f);

    // 手动参考：f = a @ a
    // matmul_backward(a, a, grad_f) → {dA, dB}
    //   dA = grad_f @ a^T  (来自第二个 a，作为 A 的位置)
    //   dB = a^T @ grad_f  (来自第一个 a，作为 B 的位置)
    // 因为两个 a 是同一 tensor，grad_a = dA + dB
    auto [dA_ref, dB_ref] = matmul_backward(a, a, grad_f);
    // grad_a_ref = dA + dB
    Tensor grad_a_ref({3,3});
    for (size_t i = 0; i < 9; ++i) {
        grad_a_ref.data()[i] = dA_ref.data()[i] + dB_ref.data()[i];
    }

    const Tensor* ga = tape.grad(a.id());
    if (!ga) { printf("  FAIL: grad_a is null\n"); return 1; }
    if (!tensor_eq(*ga, grad_a_ref)) {
        printf("  FAIL: grad_a mismatch (累加)\n");
        printf("  autograd grad_a:\n");
        for (int i = 0; i < 3; ++i) {
            printf("    ");
            for (int j = 0; j < 3; ++j) printf("%g ", ga->at(i,j));
            printf("\n");
        }
        printf("  reference grad_a:\n");
        for (int i = 0; i < 3; ++i) {
            printf("    ");
            for (int j = 0; j < 3; ++j) printf("%g ", grad_a_ref.at(i,j));
            printf("\n");
        }
        return 1;
    }

    printf("  grad_a (累加) OK\n");
    printf("  PASS\n\n");
    return 0;
}

// ============================================================================
// 测试 3: requires_grad=False 的 input 无梯度
// ============================================================================
static int test_no_grad_input() {
    printf("[test_no_grad_input] a.requires_grad=false, b.requires_grad=true\n");
    Tape& tape = Tape::current();
    tape.clear();

    std::vector<float> a_data = {1,2,3, 4,5,6};      // 2x3
    std::vector<float> b_data = {1,0,1,0, 0,1,0,1, 1,1,1,1}; // 3x4

    Tensor a = make_tensor({2,3}, a_data); a.set_requires_grad(false);  // 不需梯度
    Tensor b = make_tensor({3,4}, b_data); b.set_requires_grad(true);

    tape.start_recording();
    Tensor c = matmul(a, b);  // c: (2,4)
    tape.stop_recording();

    // c.requires_grad 应为 true（因为 b 需要）
    if (!c.requires_grad()) { printf("  FAIL: c.requires_grad should be true\n"); return 1; }

    std::vector<float> grad_c_data(8, 1.0f);
    Tensor grad_c = make_tensor({2,4}, grad_c_data);
    tape.backward(c, grad_c);

    // a 无梯度
    const Tensor* ga = tape.grad(a.id());
    if (ga != nullptr) { printf("  FAIL: grad_a should be null\n"); return 1; }

    // b 有梯度
    const Tensor* gb = tape.grad(b.id());
    if (!gb) { printf("  FAIL: grad_b should not be null\n"); return 1; }

    // 验证 grad_b 正确：matmul_backward(a, b, grad_c) → {dA, dB}
    auto [dA_ref, dB_ref] = matmul_backward(a, b, grad_c);
    if (!tensor_eq(*gb, dB_ref)) { printf("  FAIL: grad_b mismatch\n"); return 1; }

    printf("  grad_a = null (正确)\n");
    printf("  grad_b OK\n");
    printf("  PASS\n\n");
    return 0;
}

// ============================================================================
// 测试 4: 非录制模式下 matmul 不记录
// ============================================================================
static int test_no_recording() {
    printf("[test_no_recording] tape 未 start_recording\n");
    Tape& tape = Tape::current();
    tape.clear();

    std::vector<float> a_data = {1,2,3, 4,5,6};
    std::vector<float> b_data = {1,0,1,0, 0,1,0,1, 1,1,1,1};

    Tensor a = make_tensor({2,3}, a_data); a.set_requires_grad(true);
    Tensor b = make_tensor({3,4}, b_data); b.set_requires_grad(true);

    // 不 start_recording
    Tensor c = matmul(a, b);

    if (tape.size() != 0) { printf("  FAIL: tape should be empty\n"); return 1; }
    if (c.requires_grad()) { printf("  FAIL: c.requires_grad should be false (no recording)\n"); return 1; }

    // 前向结果仍应正确
    Tensor c_ref = matmul_forward(a, b);
    if (!tensor_eq(c, c_ref)) { printf("  FAIL: forward result wrong\n"); return 1; }

    printf("  tape empty, forward correct, no recording\n");
    printf("  PASS\n\n");
    return 0;
}

// ============================================================================
// 测试 5: 前向数值与手动 matmul_forward 一致（录制不影响前向）
// ============================================================================
static int test_forward_unchanged() {
    printf("[test_forward_unchanged] 录制模式下前向数值不变\n");
    Tape& tape = Tape::current();
    tape.clear();

    std::vector<float> a_data = {1,2,3, 4,5,6};
    std::vector<float> b_data = {1,0,1,0, 0,1,0,1, 1,1,1,1};

    Tensor a = make_tensor({2,3}, a_data); a.set_requires_grad(true);
    Tensor b = make_tensor({3,4}, b_data); b.set_requires_grad(true);

    tape.start_recording();
    Tensor c_autograd = matmul(a, b);
    tape.stop_recording();

    Tensor c_ref = matmul_forward(a, b);

    if (!tensor_eq(c_autograd, c_ref)) { printf("  FAIL: forward changed by recording\n"); return 1; }

    printf("  forward values identical\n");
    printf("  PASS\n\n");
    return 0;
}

// ============================================================================
// main
// ============================================================================
int main() {
    printf("========================================\n");
    printf("Phase 2: tape-based Autograd 测试\n");
    printf("========================================\n\n");

    int failures = 0;
    failures += test_chain_rule();
    failures += test_reuse_same_tensor();
    failures += test_no_grad_input();
    failures += test_no_recording();
    failures += test_forward_unchanged();

    printf("========================================\n");
    if (failures == 0) {
        printf("全部测试通过 (5/5)\n");
    } else {
        printf("%d 个测试失败\n", failures);
    }
    printf("========================================\n");
    return failures;
}
