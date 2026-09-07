// autograd_nn.cpp - autograd-aware 神经网络算子实现（Phase 5 + Stage 3.x 策略感知）
//
// Stage 3.2 Phase 5：为 linear/relu/conv2d/maxpool/bn 添加 autograd-aware 版本。
// Stage 3.x：添加反向传播策略感知（FLOAT32 / STE），其他策略为桩。
//
// 实现模式（与 autograd.cpp 的 matmul 一致）：
//   1. 调用 forward 计算（根据策略选择 FLOAT32 或 STE 实现）
//   2. 若 tape 在记录 且 needs_grad，构造 Record
//   3. backward_fn 捕获所需输入和 context
//   4. record 到 tape

#include "autograd_nn.h"
#include "ops_activation.h"
#include "ops_norm.h"
#include "ops.h"

#include <stdexcept>
#include <utility>

namespace sgn_autograd {

// ============================================================================
// 策略桩：未实现策略抛出异常
// ============================================================================
namespace {
void check_strategy_supported(BackwardStrategy s) {
    switch (s) {
        case BackwardStrategy::FLOAT32:
        case BackwardStrategy::STE:
        case BackwardStrategy::GEF:
        case BackwardStrategy::SR:
        case BackwardStrategy::A1:
            return;  // 已实现
        case BackwardStrategy::HC16:
        case BackwardStrategy::EF_SGD:
        case BackwardStrategy::MSINT:
        case BackwardStrategy::LEVEL_AMP:
            throw std::runtime_error(
                std::string("BackwardStrategy::") + strategy_name(s) +
                " 尚未实现。当前支持 FLOAT32 / STE / GEF / SR。"
            );
        default:
            throw std::runtime_error("Unknown BackwardStrategy");
    }
}
}  // anonymous namespace

// ============================================================================
// LinearNode: linear 的 backward 逻辑（P2 type-erased NodeBase）
// ============================================================================
// 捕获 X/W/b 浅拷贝。apply: dX/dW/db = linear_backward。
class LinearNode final : public NodeBase {
public:
    LinearNode(Tensor X, Tensor W, Tensor b)
        : X_(std::move(X)), W_(std::move(W)), b_(std::move(b)) {}

    std::vector<Tensor> apply(const Tensor& dY) override {
        auto [dX, dW, db] = linear_backward(X_, W_, b_, dY);
        return {std::move(dX), std::move(dW), std::move(db)};
    }

private:
    Tensor X_, W_, b_;
};

// ============================================================================
// linear: autograd-aware（策略感知）
// ============================================================================
Tensor linear(const Tensor& X, const Tensor& W, const Tensor& b) {
    BackwardStrategy s = StrategyContext::get();
    check_strategy_supported(s);

    Tensor Y;
    if (s == BackwardStrategy::STE || s == BackwardStrategy::A1) {
        Y = linear_forward_ste(X, W, b, StrategyContext::quant_config());
    } else {  // FLOAT32
        Y = linear_forward(X, W, b);
    }

    bool needs_grad = X.requires_grad() || W.requires_grad() || b.requires_grad();
    Tape& tape = Tape::current();
    if (tape.is_recording() && needs_grad) {
        Y.set_requires_grad(true);

        Record rec;
        rec.op_type = "linear";
        rec.output_id = Y.id();
        rec.input_ids = {X.id(), W.id(), b.id()};
        rec.input_requires_grad = {X.requires_grad(), W.requires_grad(), b.requires_grad()};
        rec.backward_node = std::make_unique<LinearNode>(X, W, b);

        tape.record(std::move(rec));
    }
    return Y;
}

// ============================================================================
// ReluNode: relu 的 backward 逻辑（P2 type-erased NodeBase）
// ============================================================================
// 捕获 X 浅拷贝。apply: dX = relu_backward。
class ReluNode final : public NodeBase {
public:
    explicit ReluNode(Tensor X) : X_(std::move(X)) {}

    std::vector<Tensor> apply(const Tensor& dY) override {
        return {relu_backward(X_, dY)};
    }

private:
    Tensor X_;
};

// ============================================================================
// relu: autograd-aware
// ============================================================================
Tensor relu(const Tensor& X) {
    Tensor Y = relu_forward(X);

    bool needs_grad = X.requires_grad();
    Tape& tape = Tape::current();
    if (tape.is_recording() && needs_grad) {
        Y.set_requires_grad(true);

        Record rec;
        rec.op_type = "relu";
        rec.output_id = Y.id();
        rec.input_ids = {X.id()};
        rec.input_requires_grad = {X.requires_grad()};
        rec.backward_node = std::make_unique<ReluNode>(X);

        tape.record(std::move(rec));
    }
    return Y;
}

// ============================================================================
// 激活族节点：sigmoid/tanh 捕获输出 Y（y*(1-y) / 1-y*y）；gelu/silu 捕获
// 输入 X（解析式需 x）。节点薄、内核在 ops_activation.cpp（模块化）。
// ============================================================================
class SigmoidNode final : public NodeBase {
public:
    explicit SigmoidNode(Tensor Y) : Y_(std::move(Y)) {}

    std::vector<Tensor> apply(const Tensor& dY) override {
        return {sigmoid_backward(Y_, dY)};
    }

private:
    Tensor Y_;
};

Tensor sigmoid(const Tensor& X) {
    Tensor Y = sigmoid_forward(X);

    Tape& tape = Tape::current();
    if (tape.is_recording() && X.requires_grad()) {
        Y.set_requires_grad(true);

        Record rec;
        rec.op_type = "sigmoid";
        rec.output_id = Y.id();
        rec.input_ids = {X.id()};
        rec.input_requires_grad = {X.requires_grad()};
        rec.backward_node = std::make_unique<SigmoidNode>(Y);

        tape.record(std::move(rec));
    }
    return Y;
}

class TanhNode final : public NodeBase {
public:
    explicit TanhNode(Tensor Y) : Y_(std::move(Y)) {}

    std::vector<Tensor> apply(const Tensor& dY) override {
        return {tanh_backward(Y_, dY)};
    }

private:
    Tensor Y_;
};

Tensor tanh(const Tensor& X) {
    Tensor Y = tanh_forward(X);

    Tape& tape = Tape::current();
    if (tape.is_recording() && X.requires_grad()) {
        Y.set_requires_grad(true);

        Record rec;
        rec.op_type = "tanh";
        rec.output_id = Y.id();
        rec.input_ids = {X.id()};
        rec.input_requires_grad = {X.requires_grad()};
        rec.backward_node = std::make_unique<TanhNode>(Y);

        tape.record(std::move(rec));
    }
    return Y;
}

class GeluNode final : public NodeBase {
public:
    explicit GeluNode(Tensor X) : X_(std::move(X)) {}

    std::vector<Tensor> apply(const Tensor& dY) override {
        return {gelu_backward(X_, dY)};
    }

private:
    Tensor X_;
};

Tensor gelu(const Tensor& X) {
    Tensor Y = gelu_forward(X);

    Tape& tape = Tape::current();
    if (tape.is_recording() && X.requires_grad()) {
        Y.set_requires_grad(true);

        Record rec;
        rec.op_type = "gelu";
        rec.output_id = Y.id();
        rec.input_ids = {X.id()};
        rec.input_requires_grad = {X.requires_grad()};
        rec.backward_node = std::make_unique<GeluNode>(X);

        tape.record(std::move(rec));
    }
    return Y;
}

class SiluNode final : public NodeBase {
public:
    explicit SiluNode(Tensor X) : X_(std::move(X)) {}

    std::vector<Tensor> apply(const Tensor& dY) override {
        return {silu_backward(X_, dY)};
    }

private:
    Tensor X_;
};

Tensor silu(const Tensor& X) {
    Tensor Y = silu_forward(X);

    Tape& tape = Tape::current();
    if (tape.is_recording() && X.requires_grad()) {
        Y.set_requires_grad(true);

        Record rec;
        rec.op_type = "silu";
        rec.output_id = Y.id();
        rec.input_ids = {X.id()};
        rec.input_requires_grad = {X.requires_grad()};
        rec.backward_node = std::make_unique<SiluNode>(X);

        tape.record(std::move(rec));
    }
    return Y;
}

// ============================================================================
// AddNode: add 的 backward 逻辑（P2 type-erased NodeBase）
// ============================================================================
// 加法梯度直通：dA = dY, dB = dY（无需捕获前向输入）。
class AddNode final : public NodeBase {
public:
    std::vector<Tensor> apply(const Tensor& dY) override {
        // 返回两个共享 dY storage 的浅拷贝（Tape 按 input_id 分别累加）
        return {dY, dY};
    }
};

// ============================================================================
// add: autograd-aware（ResNet 残差连接）
// ============================================================================
Tensor add(const Tensor& A, const Tensor& B) {
    Tensor Y = add_forward(A, B);

    bool needs_grad = A.requires_grad() || B.requires_grad();
    Tape& tape = Tape::current();
    if (tape.is_recording() && needs_grad) {
        Y.set_requires_grad(true);

        Record rec;
        rec.op_type = "add";
        rec.output_id = Y.id();
        rec.input_ids = {A.id(), B.id()};
        rec.input_requires_grad = {A.requires_grad(), B.requires_grad()};
        rec.backward_node = std::make_unique<AddNode>();

        tape.record(std::move(rec));
    }
    return Y;
}

// ============================================================================
// MulNode: 乘法 backward（捕获另一输入；dA = dY·B，dB = dY·A）
// ============================================================================
class MulNode final : public NodeBase {
public:
    MulNode(Tensor A, Tensor B) : A_(std::move(A)), B_(std::move(B)) {}

    std::vector<Tensor> apply(const Tensor& dY) override {
        Tensor dA = mul_forward(dY, B_);
        Tensor dB = mul_forward(dY, A_);
        return {dA, dB};
    }

private:
    Tensor A_;
    Tensor B_;
};

// ============================================================================
// mul: autograd-aware（逐元素乘；Dropout/掩码/缩放的地基）
// ============================================================================
Tensor mul(const Tensor& A, const Tensor& B) {
    Tensor Y = mul_forward(A, B);

    bool needs_grad = A.requires_grad() || B.requires_grad();
    Tape& tape = Tape::current();
    if (tape.is_recording() && needs_grad) {
        Y.set_requires_grad(true);

        Record rec;
        rec.op_type = "mul";
        rec.output_id = Y.id();
        rec.input_ids = {A.id(), B.id()};
        rec.input_requires_grad = {A.requires_grad(), B.requires_grad()};
        rec.backward_node = std::make_unique<MulNode>(A, B);

        tape.record(std::move(rec));
    }
    return Y;
}

// ============================================================================
// BNNode: bn_train 的 backward 逻辑（P2 type-erased NodeBase）
// ============================================================================
// 捕获 ctx（含 mean/rstd/x_normalized）和 gamma（backward 公式需要）。
class BNNode final : public NodeBase {
public:
    BNNode(BNContext ctx, Tensor gamma, int dim)
        : ctx_(std::move(ctx)), gamma_(std::move(gamma)), dim_(dim) {}

    std::vector<Tensor> apply(const Tensor& dY) override {
        auto [dX, dgamma, dbeta] = bn_backward_train(ctx_, gamma_, dY, dim_);
        return {std::move(dX), std::move(dgamma), std::move(dbeta)};
    }

private:
    BNContext ctx_;
    Tensor gamma_;
    int dim_;
};

// ============================================================================
// bn_train: autograd-aware
// ============================================================================
// 注意：running_mean/var 是引用参数，会被原地修改。
// backward 不需要 running stats，只需要 ctx 中的 mean/rstd/x_normalized。
// 但 gamma 需要 backward，所以作为 input 记录。
// beta 不参与 backward 公式中的 dX 计算，但需要 dbeta。
Tensor bn_train(
    const Tensor& X, const Tensor& gamma, const Tensor& beta,
    Tensor& running_mean, Tensor& running_var,
    float momentum, float eps, int dim
) {
    // 在 stack 上创建临时 context，backward_node 会持有其拷贝
    BNContext ctx;
    Tensor Y = bn_forward_train(X, gamma, beta, running_mean, running_var,
                                 momentum, eps, dim, ctx);

    bool needs_grad = X.requires_grad() || gamma.requires_grad() || beta.requires_grad();
    Tape& tape = Tape::current();
    if (tape.is_recording() && needs_grad) {
        Y.set_requires_grad(true);

        Record rec;
        rec.op_type = "bn_train";
        rec.output_id = Y.id();
        rec.input_ids = {X.id(), gamma.id(), beta.id()};
        rec.input_requires_grad = {X.requires_grad(), gamma.requires_grad(), beta.requires_grad()};
        rec.backward_node = std::make_unique<BNNode>(ctx, gamma, dim);

        tape.record(std::move(rec));
    }
    return Y;
}

// ============================================================================
// batchnorm2d: autograd-aware（4D 输入的 BN2d 包装）
// ============================================================================
// 输入 X: (B, C, H, W)，gamma/beta/running_mean/var: (C,)
// 直接调 bn_forward_train_4d 逐通道计算（免去 reshape 到 (C,B*H*W) 的深拷贝）。
// backward 走 BN2dNode → bn_backward_train_4d。
// ============================================================================
// BN2dNode: batchnorm2d 的 backward 逻辑（P2 type-erased NodeBase）
// ============================================================================
// 捕获 ctx、gamma；apply 直接调 bn_backward_train_4d（4D 布局，无 reshape 拷贝）。
class BN2dNode final : public NodeBase {
public:
    BN2dNode(BNContext ctx, Tensor gamma)
        : ctx_(std::move(ctx)), gamma_(std::move(gamma)) {}

    std::vector<Tensor> apply(const Tensor& dY) override {
        auto [dX, dgamma, dbeta] = bn_backward_train_4d(ctx_, gamma_, dY);
        return {std::move(dX), std::move(dgamma), std::move(dbeta)};
    }

private:
    BNContext ctx_;
    Tensor gamma_;
};

Tensor batchnorm2d(
    const Tensor& X, const Tensor& gamma, const Tensor& beta,
    Tensor& running_mean, Tensor& running_var,
    float momentum, float eps
) {
    // 直接在 (B,C,H,W) 布局上逐通道计算，免去 permute+contiguous 深拷贝（2026-08-16）
    BNContext ctx;
    Tensor Y = bn_forward_train_4d(X, gamma, beta, running_mean, running_var,
                                    momentum, eps, ctx);

    bool needs_grad = X.requires_grad() || gamma.requires_grad() || beta.requires_grad();
    Tape& tape = Tape::current();
    if (tape.is_recording() && needs_grad) {
        Y.set_requires_grad(true);

        Record rec;
        rec.op_type = "batchnorm2d";
        rec.output_id = Y.id();
        rec.input_ids = {X.id(), gamma.id(), beta.id()};
        rec.input_requires_grad = {X.requires_grad(), gamma.requires_grad(), beta.requires_grad()};
        rec.backward_node = std::make_unique<BN2dNode>(ctx, gamma);

        tape.record(std::move(rec));
    }
    return Y;
}

// ============================================================================
// Conv2dNode: conv2d 的 backward 逻辑（P2 type-erased NodeBase）
// ============================================================================
// 捕获 ctx（含 im2col 结果）、X、W（backward 需要）。
class Conv2dNode final : public NodeBase {
public:
    Conv2dNode(Conv2DContext ctx, Tensor X, Tensor W)
        : ctx_(std::move(ctx)), X_(std::move(X)), W_(std::move(W)) {}

    std::vector<Tensor> apply(const Tensor& dY) override {
        auto [dX, dW, db] = conv2d_backward(ctx_, X_, W_, dY);
        return {std::move(dX), std::move(dW), std::move(db)};
    }

private:
    Conv2DContext ctx_;
    Tensor X_, W_;
};

// ============================================================================
// conv2d: autograd-aware（策略感知）
// ============================================================================
Tensor conv2d(
    const Tensor& X, const Tensor& W, const Tensor& b,
    int stride, int padding
) {
    BackwardStrategy s = StrategyContext::get();
    check_strategy_supported(s);

    Conv2DContext ctx;
    Tensor Y;
    if (s == BackwardStrategy::STE || s == BackwardStrategy::A1) {
        Y = conv2d_forward_ste(X, W, b, stride, padding, ctx, StrategyContext::quant_config());
    } else {  // FLOAT32
        Y = conv2d_forward(X, W, b, stride, padding, ctx);
    }

    bool needs_grad = X.requires_grad() || W.requires_grad() || b.requires_grad();
    Tape& tape = Tape::current();
    if (tape.is_recording() && needs_grad) {
        Y.set_requires_grad(true);

        Record rec;
        rec.op_type = "conv2d";
        rec.output_id = Y.id();
        rec.input_ids = {X.id(), W.id(), b.id()};
        rec.input_requires_grad = {X.requires_grad(), W.requires_grad(), b.requires_grad()};
        rec.backward_node = std::make_unique<Conv2dNode>(ctx, X, W);

        tape.record(std::move(rec));
    }
    return Y;
}

// ============================================================================
// MaxPoolNode: maxpool2d 的 backward 逻辑（P2 type-erased NodeBase）
// ============================================================================
// 捕获 MaxPoolContext（含 argmax 索引）。
class MaxPoolNode final : public NodeBase {
public:
    explicit MaxPoolNode(MaxPoolContext ctx) : ctx_(std::move(ctx)) {}

    std::vector<Tensor> apply(const Tensor& dY) override {
        return {maxpool2d_backward(ctx_, dY)};
    }

private:
    MaxPoolContext ctx_;
};

// ============================================================================
// maxpool2d: autograd-aware
// ============================================================================
Tensor maxpool2d(const Tensor& X, int kernel, int stride_arg) {
    int stride = (stride_arg < 0) ? kernel : stride_arg;  // 默认 stride=kernel
    MaxPoolContext ctx;
    Tensor Y = maxpool2d_forward(X, kernel, stride, ctx);

    bool needs_grad = X.requires_grad();
    Tape& tape = Tape::current();
    if (tape.is_recording() && needs_grad) {
        Y.set_requires_grad(true);

        Record rec;
        rec.op_type = "maxpool2d";
        rec.output_id = Y.id();
        rec.input_ids = {X.id()};
        rec.input_requires_grad = {X.requires_grad()};
        rec.backward_node = std::make_unique<MaxPoolNode>(ctx);

        tape.record(std::move(rec));
    }
    return Y;
}

// ============================================================================
// AvgPoolNode: avgpool2d 的 backward 逻辑（P2 type-erased NodeBase）
// ============================================================================
// 捕获 AvgPoolContext（形状信息）；backward dX = 窗口内均分 dY / k²。
class AvgPoolNode final : public NodeBase {
public:
    explicit AvgPoolNode(AvgPoolContext ctx) : ctx_(std::move(ctx)) {}

    std::vector<Tensor> apply(const Tensor& dY) override {
        return {avgpool2d_backward(ctx_, dY)};
    }

private:
    AvgPoolContext ctx_;
};

// ============================================================================
// avgpool2d: autograd-aware（GAP）
// ============================================================================
Tensor avgpool2d(const Tensor& X, int kernel, int stride_arg) {
    int stride = (stride_arg < 0) ? kernel : stride_arg;  // 默认 stride=kernel
    AvgPoolContext ctx;
    Tensor Y = avgpool2d_forward(X, kernel, stride, ctx);

    bool needs_grad = X.requires_grad();
    Tape& tape = Tape::current();
    if (tape.is_recording() && needs_grad) {
        Y.set_requires_grad(true);

        Record rec;
        rec.op_type = "avgpool2d";
        rec.output_id = Y.id();
        rec.input_ids = {X.id()};
        rec.input_requires_grad = {X.requires_grad()};
        rec.backward_node = std::make_unique<AvgPoolNode>(ctx);

        tape.record(std::move(rec));
    }
    return Y;
}

// ============================================================================
// Conv2dReluNode: conv2d_relu 的 backward 逻辑（P2 type-erased NodeBase）
// ============================================================================
// 捕获 ctx（含 Conv2DContext 和 relu_mask）、X、W（backward 需要）。
class Conv2dReluNode final : public NodeBase {
public:
    Conv2dReluNode(Conv2DReluContext ctx, Tensor X, Tensor W)
        : ctx_(std::move(ctx)), X_(std::move(X)), W_(std::move(W)) {}

    std::vector<Tensor> apply(const Tensor& dY) override {
        auto [dX, dW, db] = conv2d_relu_backward(ctx_, X_, W_, dY);
        return {std::move(dX), std::move(dW), std::move(db)};
    }

private:
    Conv2DReluContext ctx_;
    Tensor X_, W_;
};

// ============================================================================
// conv2d_relu: autograd-aware（算子融合）
// ============================================================================
// 将 conv2d forward + relu forward 合并为一个 tape 记录。
// backward 时调用 conv2d_relu_backward，在 col2im 前对 dY 应用 relu mask，
// 省去一次独立的 relu_backward 调用。
Tensor conv2d_relu(
    const Tensor& X, const Tensor& W, const Tensor& b,
    int stride, int padding
) {
    Conv2DReluContext ctx;
    Tensor Y = conv2d_relu_forward(X, W, b, stride, padding, ctx);

    bool needs_grad = X.requires_grad() || W.requires_grad() || b.requires_grad();
    Tape& tape = Tape::current();
    if (tape.is_recording() && needs_grad) {
        Y.set_requires_grad(true);

        Record rec;
        rec.op_type = "conv2d_relu";
        rec.output_id = Y.id();
        rec.input_ids = {X.id(), W.id(), b.id()};
        rec.input_requires_grad = {X.requires_grad(), W.requires_grad(), b.requires_grad()};
        rec.backward_node = std::make_unique<Conv2dReluNode>(ctx, X, W);

        tape.record(std::move(rec));
    }
    return Y;
}

}

// ============================================================================
namespace sgn_autograd {
// LNNode: layernorm backward（捕获 ctx + gamma；返回 dX/dgamma/dbeta）
// ============================================================================
class LNNode final : public sgn_autograd::NodeBase {
public:
    explicit LNNode(sgn_autograd::LayerNormContext ctx, sgn_autograd::Tensor gamma)
        : ctx_(std::move(ctx)), gamma_(std::move(gamma)) {}

    std::vector<sgn_autograd::Tensor> apply(const sgn_autograd::Tensor& dY) override {
        return sgn_autograd::layernorm_backward(dY, gamma_, ctx_);
    }

private:
    sgn_autograd::LayerNormContext ctx_;
    sgn_autograd::Tensor gamma_;
};

// ============================================================================
// layernorm: autograd-aware（2D (B,C) 沿末维归一化；gamma/beta 可学习）
// ============================================================================
sgn_autograd::Tensor layernorm(const sgn_autograd::Tensor& X,
                               const sgn_autograd::Tensor& gamma,
                               const sgn_autograd::Tensor& beta, float eps) {
    sgn_autograd::LayerNormContext ctx;
    sgn_autograd::Tensor Y = sgn_autograd::layernorm_forward(X, gamma, beta, eps, ctx);

    bool needs_grad = X.requires_grad() || gamma.requires_grad() ||
                      beta.requires_grad();
    sgn_autograd::Tape& tape = sgn_autograd::Tape::current();
    if (tape.is_recording() && needs_grad) {
        Y.set_requires_grad(true);

        sgn_autograd::Record rec;
        rec.op_type = "layernorm";
        rec.output_id = Y.id();
        rec.input_ids = {X.id(), gamma.id(), beta.id()};
        rec.input_requires_grad = {X.requires_grad(), gamma.requires_grad(),
                                   beta.requires_grad()};
        rec.backward_node = std::make_unique<LNNode>(std::move(ctx), gamma);

        tape.record(std::move(rec));
    }
    return Y;
}
}  // namespace sgn_autograd

// ============================================================================
