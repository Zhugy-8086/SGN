// test_phase3.cpp - 神经网络算子验证（Phase 3）
//
// Stage 3.2 Phase 3 Task A3.7：验证 linear/relu/bn/conv2d/maxpool forward+backward
//
// 验证策略：
//   - forward 直接对比 PyTorch 期望值（硬编码）
//   - backward 用手动链式法则 + PyTorch 期望值双验证
//
// 测试用例：
//   1. linear forward + backward (vs PyTorch)
//   2. relu forward + backward (vs PyTorch)
//   3. bn forward + backward (训练模式, vs PyTorch)
//   4. conv2d forward + backward (vs PyTorch)
//   5. maxpool forward + backward (vs PyTorch)

#include "ops_nn.h"
#include "ops.h"

#include <cmath>
#include <cstdio>
#include <vector>

using namespace sgn_autograd;

// ============================================================================
// 辅助：浮点近似比较
// ============================================================================
static bool approx(float a, float b, float eps = 1e-4f) {
    return std::fabs(a - b) < eps;
}

static bool tensor_eq(const Tensor& a, const Tensor& b, float eps = 1e-4f) {
    if (a.shape() != b.shape()) return false;
    size_t n = a.numel();
    if (n != b.numel()) return false;
    for (size_t i = 0; i < n; ++i) {
        if (!approx(a.data()[i], b.data()[i], eps)) return false;
    }
    return true;
}

static Tensor make_tensor(const std::vector<int64_t>& shape, const std::vector<float>& data) {
    return Tensor(shape, data.data());
}

// ============================================================================
// 测试 1: linear forward + backward
// ============================================================================
static int test_linear() {
    printf("[test_linear] Y = X @ W^T + b\n");
    // X: (2, 3), W: (4, 3), b: (4,)
    std::vector<float> x_data = {1,2,3, 4,5,6};        // 2x3
    std::vector<float> w_data = {
        1,0,1, 0,1,0, 1,1,1, 2,1,2
    }; // 4x3
    std::vector<float> b_data = {0.1f, 0.2f, 0.3f, 0.4f};

    Tensor X = make_tensor({2,3}, x_data);
    Tensor W = make_tensor({4,3}, w_data);
    Tensor b = make_tensor({4}, b_data);

    // forward
    Tensor Y = linear_forward(X, W, b);

    // PyTorch 期望值: Y = X @ W^T + b
    //   Y[0,0] = 1*1+2*0+3*1 + 0.1 = 4.1
    //   Y[0,1] = 1*0+2*1+3*0 + 0.2 = 2.2
    //   Y[0,2] = 1*1+2*1+3*1 + 0.3 = 6.3
    //   Y[0,3] = 1*2+2*1+3*2 + 0.4 = 10.4
    //   Y[1,0] = 4*1+5*0+6*1 + 0.1 = 10.1
    //   Y[1,1] = 4*0+5*1+6*0 + 0.2 = 5.2
    //   Y[1,2] = 4*1+5*1+6*1 + 0.3 = 15.3
    //   Y[1,3] = 4*2+5*1+6*2 + 0.4 = 25.4
    std::vector<float> y_expected = {4.1f, 2.2f, 6.3f, 10.4f, 10.1f, 5.2f, 15.3f, 25.4f};
    Tensor Y_exp = make_tensor({2,4}, y_expected);
    if (!tensor_eq(Y, Y_exp)) {
        printf("  FAIL: forward mismatch\n");
        printf("  got:    ");
        for (size_t i = 0; i < 8; ++i) printf("%g ", Y.data()[i]);
        printf("\n  expect: ");
        for (size_t i = 0; i < 8; ++i) printf("%g ", y_expected[i]);
        printf("\n");
        return 1;
    }
    printf("  forward OK\n");

    // backward: dY = ones(2,4)
    std::vector<float> dy_data(8, 1.0f);
    Tensor dY = make_tensor({2,4}, dy_data);
    auto [dX, dW, db] = linear_backward(X, W, b, dY);

    // PyTorch 期望值 (dY=ones):
    //   dX = dY @ W → dX[i,l] = sum_j W[j,l]
    //     dX[0,0] = 1+0+1+2 = 4, dX[0,1] = 0+1+1+1 = 3, dX[0,2] = 1+0+1+2 = 4
    //     dX[1,0] = 4, dX[1,1] = 3, dX[1,2] = 4
    //   dW = dY^T @ X → dW[j,l] = sum_i X[i,l]
    //     dW[0,0] = 1+4=5, dW[0,1] = 2+5=7, dW[0,2] = 3+6=9
    //     dW[1,0] = 5, dW[1,1] = 7, dW[1,2] = 9  (所有行相同)
    //   db = sum_rows(dY) = [2, 2, 2, 2]
    std::vector<float> dx_expected = {4,3,4, 4,3,4};
    std::vector<float> dw_expected = {5,7,9, 5,7,9, 5,7,9, 5,7,9};
    std::vector<float> db_expected = {2,2,2,2};
    if (!tensor_eq(dX, make_tensor({2,3}, dx_expected))) { printf("  FAIL: dX mismatch\n"); return 1; }
    if (!tensor_eq(dW, make_tensor({4,3}, dw_expected))) { printf("  FAIL: dW mismatch\n"); return 1; }
    if (!tensor_eq(db, make_tensor({4}, db_expected))) { printf("  FAIL: db mismatch\n"); return 1; }
    printf("  backward OK (dX/dW/db vs PyTorch)\n");
    printf("  PASS\n\n");
    return 0;
}

// ============================================================================
// 测试 2: relu forward + backward
// ============================================================================
static int test_relu() {
    printf("[test_relu] Y = max(0, X)\n");
    std::vector<float> x_data = {-1, 2, -3, 4, 0, -5, 6, -7};
    Tensor X = make_tensor({2,4}, x_data);

    Tensor Y = relu_forward(X);
    std::vector<float> y_expected = {0, 2, 0, 4, 0, 0, 6, 0};
    if (!tensor_eq(Y, make_tensor({2,4}, y_expected))) { printf("  FAIL: forward mismatch\n"); return 1; }
    printf("  forward OK\n");

    // backward: dY = ones
    std::vector<float> dy_data(8, 1.0f);
    Tensor dY = make_tensor({2,4}, dy_data);
    Tensor dX = relu_backward(X, dY);
    // dX = dY * (X > 0) = [0,1,0,1,0,0,1,0]
    std::vector<float> dx_expected = {0,1,0,1,0,0,1,0};
    if (!tensor_eq(dX, make_tensor({2,4}, dx_expected))) { printf("  FAIL: backward mismatch\n"); return 1; }
    printf("  backward OK\n");
    printf("  PASS\n\n");
    return 0;
}

// ============================================================================
// 测试 3: bn forward + backward (训练模式, dim=0, BatchNorm1d)
// ============================================================================
static int test_bn() {
    printf("[test_bn] BatchNorm1d train mode (dim=0)\n");
    // X: (4, 3) — 4 个样本，3 个特征
    std::vector<float> x_data = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12
    };
    Tensor X = make_tensor({4,3}, x_data);
    std::vector<float> g_data = {1.0f, 1.0f, 1.0f};  // gamma=1
    std::vector<float> be_data = {0.0f, 0.0f, 0.0f}; // beta=0
    Tensor gamma = make_tensor({3}, g_data);
    Tensor beta = make_tensor({3}, be_data);
    Tensor rm = make_tensor({3}, {0.0f, 0.0f, 0.0f});
    Tensor rv = make_tensor({3}, {1.0f, 1.0f, 1.0f});

    BNContext ctx;
    float momentum = 0.1f;
    float eps = 1e-5f;
    Tensor Y = bn_forward_train(X, gamma, beta, rm, rv, momentum, eps, 0, ctx);

    // 每列均值: [5.5, 6.5, 7.5]
    // 每列方差: ((1-5.5)^2+(4-5.5)^2+(7-5.5)^2+(10-5.5)^2)/4 = (20.25+2.25+2.25+20.25)/4 = 45/4 = 11.25
    // 同理列2,3 方差也是 11.25
    // rstd = 1/sqrt(11.25+eps) ≈ 0.29814
    // Y = (X - mean) * rstd (gamma=1, beta=0)
    float expected_mean[3] = {5.5f, 6.5f, 7.5f};
    float expected_var = 11.25f;
    float expected_rstd = 1.0f / std::sqrt(expected_var + eps);

    // 验证 mean 和 rstd
    for (int f = 0; f < 3; ++f) {
        if (!approx(ctx.mean.data()[f], expected_mean[f])) {
            printf("  FAIL: mean[%d] = %g, expect %g\n", f, ctx.mean.data()[f], expected_mean[f]);
            return 1;
        }
        if (!approx(ctx.rstd.data()[f], expected_rstd)) {
            printf("  FAIL: rstd[%d] = %g, expect %g\n", f, ctx.rstd.data()[f], expected_rstd);
            return 1;
        }
    }
    printf("  mean/rstd OK (mean=[5.5,6.5,7.5], var=11.25)\n");

    // 验证 Y
    bool y_ok = true;
    for (int i = 0; i < 4 && y_ok; ++i) {
        for (int j = 0; j < 3; ++j) {
            float expected = (x_data[i*3+j] - expected_mean[j]) * expected_rstd;
            if (!approx(Y.data()[i*3+j], expected)) {
                printf("  FAIL: Y[%d,%d] = %g, expect %g\n", i, j, Y.data()[i*3+j], expected);
                y_ok = false;
                break;
            }
        }
    }
    if (!y_ok) return 1;
    printf("  forward OK\n");

    // backward: dY = ones(4,3)
    std::vector<float> dy_data(12, 1.0f);
    Tensor dY = make_tensor({4,3}, dy_data);
    auto [dX, dgamma, dbeta] = bn_backward_train(ctx, gamma, dY, 0);

    // PyTorch 期望值 (gamma=1, beta=0, dY=ones):
    //   dgamma = sum(dY * x_norm) 沿 batch 维
    //   dbeta = sum(dY) = 4 (每列)
    //   dX = (1/N) * rstd * (N*dY - sum(dY) - x_norm * sum(dY*x_norm))
    //      N=4, dY=1, sum(dY)=4, sum(dY*x_norm) = sum(x_norm)
    //      dX[i,j] = (rstd/4) * (4 - 4 - x_norm[i,j] * sum(x_norm[:,j]))
    //              = -(rstd/4) * x_norm[i,j] * sum(x_norm[:,j])
    //   sum(x_norm[:,j]) = 0 (归一化后均值为0)
    //   所以 dX[i,j] = 0
    for (size_t i = 0; i < 12; ++i) {
        if (!approx(dX.data()[i], 0.0f, 1e-3f)) {
            printf("  FAIL: dX[%zu] = %g, expect ~0 (sum(x_norm)=0)\n", i, dX.data()[i]);
            return 1;
        }
    }
    // dbeta = 4
    for (int f = 0; f < 3; ++f) {
        if (!approx(dbeta.data()[f], 4.0f)) {
            printf("  FAIL: dbeta[%d] = %g, expect 4\n", f, dbeta.data()[f]);
            return 1;
        }
    }
    // dgamma = sum(x_norm) = 0 (归一化后均值为0)
    for (int f = 0; f < 3; ++f) {
        if (!approx(dgamma.data()[f], 0.0f, 1e-3f)) {
            printf("  FAIL: dgamma[%d] = %g, expect ~0\n", f, dgamma.data()[f]);
            return 1;
        }
    }
    printf("  backward OK (dX~0, dgamma~0, dbeta=4 vs PyTorch)\n");
    printf("  PASS\n\n");
    return 0;
}

// ============================================================================
// 测试 4: conv2d forward + backward
// ============================================================================
static int test_conv2d() {
    printf("[test_conv2d] conv2d stride=1, padding=1\n");
    // X: (1, 1, 3, 3) — 单样本单通道 3x3
    std::vector<float> x_data = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9
    };
    Tensor X = make_tensor({1, 1, 3, 3}, x_data);
    // W: (1, 1, 3, 3) — 单输出通道，3x3 卷积核
    std::vector<float> w_data = {
        1, 0, 0,
        0, 1, 0,
        0, 0, 1
    };
    Tensor W = make_tensor({1, 1, 3, 3}, w_data);
    Tensor b = make_tensor({1}, {0.0f});

    Conv2DContext ctx;
    Tensor Y = conv2d_forward(X, W, b, 1, 1, ctx);
    // out_h = out_w = (3 + 2 - 3)/1 + 1 = 3
    // Y: (1, 1, 3, 3)
    // 手动计算 (padding=1, 所以输入周围补0):
    //   padded X:
    //     0 0 0 0 0
    //     0 1 2 3 0
    //     0 4 5 6 0
    //     0 7 8 9 0
    //     0 0 0 0 0
    //   Y[0,0] = 0*1+0*0+0*0 + 0*0+1*1+2*0 + 0*0+4*0+5*1 = 1+5 = 6
    //   Y[0,1] = 0*1+0*0+0*0 + 1*0+2*1+3*0 + 4*0+5*0+6*1 = 2+6 = 8
    //   Y[0,2] = 0*1+0*0+0*0 + 2*0+3*1+0*0 + 5*0+6*0+0*1 = 3
    //   Y[1,0] = 0*1+1*0+2*0 + 0*0+4*1+5*0 + 0*0+7*0+8*1 = 4+8 = 12
    //   Y[1,1] = 1*1+2*0+3*0 + 4*0+5*1+6*0 + 7*0+8*0+9*1 = 1+5+9 = 15
    //   Y[1,2] = 2*1+3*0+0*0 + 5*0+6*1+0*0 + 8*0+9*0+0*1 = 2+6 = 8
    //   Y[2,0] = 0*1+4*0+5*0 + 0*0+7*1+8*0 + 0*0+0*0+0*1 = 7
    //   Y[2,1] = 4*1+5*0+6*0 + 7*0+8*1+9*0 + 0*0+0*0+0*1 = 4+8 = 12
    //   Y[2,2] = 5*1+6*0+0*0 + 8*0+9*1+0*0 + 0*0+0*0+0*1 = 5+9 = 14
    std::vector<float> y_expected = {6, 8, 3, 12, 15, 8, 7, 12, 14};
    if (!tensor_eq(Y, make_tensor({1,1,3,3}, y_expected))) {
        printf("  FAIL: forward mismatch\n");
        printf("  got:    ");
        for (int i = 0; i < 9; ++i) printf("%g ", Y.data()[i]);
        printf("\n  expect: ");
        for (int i = 0; i < 9; ++i) printf("%g ", y_expected[i]);
        printf("\n");
        return 1;
    }
    printf("  forward OK\n");

    // backward: dY = ones(1,1,3,3)
    std::vector<float> dy_data(9, 1.0f);
    Tensor dY = make_tensor({1,1,3,3}, dy_data);
    auto [dX, dW, db] = conv2d_backward(ctx, X, W, dY);

    // db = sum(dY) = 9
    if (!approx(db.data()[0], 9.0f)) {
        printf("  FAIL: db = %g, expect 9\n", db.data()[0]);
        return 1;
    }
    printf("  backward db OK (=9)\n");

    // 验证 dW: dW = dY_col @ x_col^T
    // 由于实现复杂，这里用 PyTorch 期望值验证
    // PyTorch conv2d backward (dY=ones, padding=1):
    //   dW[0,0,0,0] = sum of X values where kernel[0,0] touches
    //   实际计算：dW[i,j] = sum over all (b, oh, ow) of dY[b,oc,oh,ow] * X_padded[b,ic,oh+i,ow+j]
    //   dY=1 时，dW[i,j] = sum of X_padded at positions where kernel[i,j] aligns
    //   padded X (5x5):
    //     0 0 0 0 0
    //     0 1 2 3 0
    //     0 4 5 6 0
    //     0 7 8 9 0
    //     0 0 0 0 0
    //   dW[0,0] = sum of padded X at (0,0),(0,1),(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2)
    //           = 0+0+0+0+1+2+0+4+5 = 12
    //   dW[0,1] = 0+0+0+1+2+3+4+5+6 = 21
    //   dW[0,2] = 0+0+0+2+3+0+5+6+0 = 16
    //   dW[1,0] = 0+1+2+0+4+5+0+7+8 = 27
    //   dW[1,1] = 1+2+3+4+5+6+7+8+9 = 45
    //   dW[1,2] = 2+3+0+5+6+0+8+9+0 = 33
    //   dW[2,0] = 0+4+5+0+7+8+0+0+0 = 24
    //   dW[2,1] = 4+5+6+7+8+9+0+0+0 = 39
    //   dW[2,2] = 5+6+0+8+9+0+0+0+0 = 28
    std::vector<float> dw_expected = {12, 21, 16, 27, 45, 33, 24, 39, 28};
    if (!tensor_eq(dW, make_tensor({1,1,3,3}, dw_expected))) {
        printf("  FAIL: dW mismatch\n");
        printf("  got:    ");
        for (int i = 0; i < 9; ++i) printf("%g ", dW.data()[i]);
        printf("\n  expect: ");
        for (int i = 0; i < 9; ++i) printf("%g ", dw_expected[i]);
        printf("\n");
        return 1;
    }
    printf("  backward dW OK (vs PyTorch)\n");

    // 验证 dX: dX = col2im(W_col^T @ dY_col)
    // PyTorch 实际输出 (dY=ones, W=identity diagonal, padding=1):
    //   dX = [[2,2,1],[2,3,2],[1,2,2]]
    std::vector<float> dx_expected = {2,2,1, 2,3,2, 1,2,2};
    if (!tensor_eq(dX, make_tensor({1,1,3,3}, dx_expected))) {
        printf("  FAIL: dX mismatch\n");
        printf("  got:    ");
        for (int i = 0; i < 9; ++i) printf("%g ", dX.data()[i]);
        printf("\n  expect: ");
        for (int i = 0; i < 9; ++i) printf("%g ", dx_expected[i]);
        printf("\n");
        return 1;
    }
    printf("  backward dX OK (vs PyTorch: [2,2,1,2,3,2,1,2,2])\n");
    printf("  PASS\n\n");
    return 0;
}

// ============================================================================
// 测试 5: maxpool forward + backward
// ============================================================================
static int test_maxpool() {
    printf("[test_maxpool] maxpool2d kernel=2, stride=2\n");
    // X: (1, 1, 4, 4)
    std::vector<float> x_data = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12,
        13, 14, 15, 16
    };
    Tensor X = make_tensor({1,1,4,4}, x_data);

    MaxPoolContext ctx;
    Tensor Y = maxpool2d_forward(X, 2, 2, ctx);
    // out_h = out_w = (4-2)/2 + 1 = 2
    // Y: (1,1,2,2)
    // Y[0,0] = max(1,2,5,6) = 6
    // Y[0,1] = max(3,4,7,8) = 8
    // Y[1,0] = max(9,10,13,14) = 14
    // Y[1,1] = max(11,12,15,16) = 16
    std::vector<float> y_expected = {6, 8, 14, 16};
    if (!tensor_eq(Y, make_tensor({1,1,2,2}, y_expected))) { printf("  FAIL: forward mismatch\n"); return 1; }
    printf("  forward OK\n");

    // backward: dY = ones(1,1,2,2)
    std::vector<float> dy_data(4, 1.0f);
    Tensor dY = make_tensor({1,1,2,2}, dy_data);
    Tensor dX = maxpool2d_backward(ctx, dY);
    // dX 应只在 argmax 位置有值
    // argmax: [6 在 (0,0)→索引 5, 8 在 (0,1)→索引 7, 14 在 (1,0)→索引 13, 16 在 (1,1)→索引 15]
    // dX = [0,0,0,0, 0,1,0,1, 0,0,0,0, 0,1,0,1]
    std::vector<float> dx_expected = {0,0,0,0, 0,1,0,1, 0,0,0,0, 0,1,0,1};
    if (!tensor_eq(dX, make_tensor({1,1,4,4}, dx_expected))) {
        printf("  FAIL: backward mismatch\n");
        printf("  got:    ");
        for (int i = 0; i < 16; ++i) printf("%g ", dX.data()[i]);
        printf("\n  expect: ");
        for (int i = 0; i < 16; ++i) printf("%g ", dx_expected[i]);
        printf("\n");
        return 1;
    }
    printf("  backward OK (gradient routing correct)\n");
    printf("  PASS\n\n");
    return 0;
}

// ============================================================================
// main
// ============================================================================
int main() {
    printf("========================================\n");
    printf("Phase 3: 神经网络算子测试\n");
    printf("========================================\n\n");

    int failures = 0;
    failures += test_linear();
    failures += test_relu();
    failures += test_bn();
    failures += test_conv2d();
    failures += test_maxpool();

    printf("========================================\n");
    if (failures == 0) {
        printf("全部测试通过 (5/5)\n");
    } else {
        printf("%d 个测试失败\n", failures);
    }
    printf("========================================\n");
    return failures;
}
