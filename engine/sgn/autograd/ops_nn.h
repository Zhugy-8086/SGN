// ops_nn.h - 神经网络算子声明（Phase 3）
//
// GPU 移植注意事项（2026-08-02）：
//   GPU 不支持 C++ 异常（throw），当前所有算子函数的错误处理使用 throw。
//   未来 GPU 移植时，推荐使用 std::expected<Tensor, OpError> 替代异常：
//     enum class OpError { InvalidShape, NotContiguous, DimMismatch };
//     std::expected<Tensor, OpError> linear_forward(...);
//   这样 kernel 接口可在 CPU 和 GPU 间共享，无需重写错误处理路径。
//   std::expected 是 C++23 标准库特性，Clang 22 已支持。
//   详见 内部档案
//
// Stage 3.2 Phase 3：适配 6 层 CNN+BN 反向传播需求的算子集合。
//
// 算子清单（从 cnn6_cifar10.py 实际架构推导，非照搬 PyTorch）：
//   - linear (fc1/fc2/fc3): matmul + bias
//   - relu (激活): 朴素逐元素
//   - bn (BatchNorm2d/1d): 训练/推理模式
//   - conv2d (conv1/conv2/conv3): im2col + matmul
//   - maxpool (pool): 梯度路由（只传到前向最大值位置）
//
// 设计原则：
//   1. 只实现 6 层 CNN 需要的算子，不提前造冗余抽象
//   2. forward 只做计算，autograd 记录在 nn.h/cpp 的包装函数里（与 matmul 一致的模式）
//   3. conv2d backward 的 col2im 部分复用 sgn.col2im_add（已验证的 C++ 实现）

#pragma once

#include "tensor.h"
#include "backward_strategy.h"

#include <cstddef>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace sgn_autograd {

// ============================================================================
// linear: Y = X @ W^T + b  （PyTorch nn.Linear 约定）
// ============================================================================
//   X: (m, in_features), W: (out_features, in_features), b: (out_features,)
//   Y: (m, out_features)
//   注：W 存储为 (out, in)，计算时需转置。这里直接接受 (out, in) 的 W，
//       内部按 Y[i,j] = sum_l X[i,l] * W[j,l] + b[j] 计算。
Tensor linear_forward(const Tensor& X, const Tensor& W, const Tensor& b);

// linear backward:
//   dX = dY @ W        (m×out @ out×in = m×in)
//   dW = dY^T @ X      (out×m @ m×in = out×in)
//   db = sum_rows(dY)  (out,)
std::tuple<Tensor, Tensor, Tensor> linear_backward(
    const Tensor& X, const Tensor& W, const Tensor& b, const Tensor& dY
);

// ============================================================================
// relu: Y = max(0, X)  （逐元素）
// ============================================================================
Tensor relu_forward(const Tensor& X);

// relu backward: dX = dY * (X > 0 ? 1 : 0)
Tensor relu_backward(const Tensor& X, const Tensor& dY);

// ============================================================================
// add: Y = A + B  （逐元素，ResNet 残差连接；A 与 B 同形状，或 B 为标量 1 元素）
// ============================================================================
Tensor add_forward(const Tensor& A, const Tensor& B);
Tensor mul_forward(const Tensor& A, const Tensor& B);

// ============================================================================
// batch_norm: 训练模式 forward
// ============================================================================
//   输入 X 形状 (m, features)（BatchNorm1d）或 (m, c, h, w)（BatchNorm2d）
//   归一化维度：BatchNorm1d 沿 batch 维 (dim 0)；BatchNorm2d 沿 (N, H, W) 维
//
// 为简化实现，统一按 2D 处理：
//   - BatchNorm1d: X 直接是 (m, features)
//   - BatchNorm2d: 调用方先把 X reshape 成 (m*h*w, c)，归一化后再 reshape 回去
//   这样 bn_forward 只需处理 2D 情况，dim 参数指定归一化维度（0 或 1）
//
// 参数：
//   X: (m, n)，gamma/beta: (n,) 或 (m,) 取决于 dim
//   running_mean/var: (特征维度,)，训练时更新
//   momentum: running stats 更新系数 (PyTorch 约定: running = (1-momentum)*running + momentum*batch)
//   eps: 数值稳定
//   dim: 归一化维度（0 = 沿行归一化每列，1 = 沿列归一化每行）
//   training: true=用 batch 统计并更新 running；false=用 running 统计
struct BNContext {
    Tensor mean;       // batch 均值（backward 需要）
    Tensor rstd;       // 1/sqrt(var+eps)（backward 需要）
    Tensor x_normalized;  // 归一化后的 X（backward 需要）
};

Tensor bn_forward_train(
    const Tensor& X, const Tensor& gamma, const Tensor& beta,
    Tensor& running_mean, Tensor& running_var,
    float momentum, float eps, int dim,
    BNContext& ctx
);

// bn backward（训练模式）:
//   dgamma = sum(dY * x_normalized)
//   dbeta = sum(dY)
//   dX = (1/N) * rstd * (N*dY - sum(dY) - x_normalized * sum(dY*x_normalized))
//   其中 N 是归一化方向的元素数
std::tuple<Tensor, Tensor, Tensor> bn_backward_train(
    const BNContext& ctx, const Tensor& gamma, const Tensor& dY, int dim
);

// 4D batchnorm2d forward: Y = gamma * (X - mean) / sqrt(var + eps) + beta
// X: (B, C, H, W), gamma/beta/rm/rv: (C,)
// 直接在 4D 布局上逐通道计算，免去 permute+contiguous 深拷贝（2026-08-16）
Tensor bn_forward_train_4d(
    const Tensor& X, const Tensor& gamma, const Tensor& beta,
    Tensor& running_mean, Tensor& running_var,
    float momentum, float eps, BNContext& ctx
);

// 4D batchnorm2d backward（与 bn_forward_train_4d 配对）
std::tuple<Tensor, Tensor, Tensor> bn_backward_train_4d(
    const BNContext& ctx, const Tensor& gamma, const Tensor& dY
);

// ============================================================================
// conv2d: Y = im2col(X) @ W_col + b  （朴素实现）
// ============================================================================
//   X: (batch, in_channels, in_h, in_w)
//   W: (out_channels, in_channels, kh, kw)
//   b: (out_channels,)
//   stride: 步长（默认 1）
//   padding: 零填充（默认 0）
//   Y: (batch, out_channels, out_h, out_w)
//       out_h = (in_h + 2*padding - kh) / stride + 1
//       out_w = (in_w + 2*padding - kw) / stride + 1
//
// 内部用 im2col + matmul：
//   im2col: (batch*in_channels*kh*kw, out_h*out_w) → 转成列矩阵
//   W_col: (out_channels, in_channels*kh*kw)
//   Y_col = W_col @ im2col + b → (out_channels, out_h*out_w) per batch
struct Conv2DContext {
    int64_t batch, in_channels, in_h, in_w;
    int64_t out_channels, kh, kw;
    int64_t out_h, out_w;
    int stride, padding;
    Tensor x_col;  // im2col 结果：(in_channels*kh*kw, batch*out_h*out_w)
                    // 或 (batch, in_channels*kh*kw, out_h*out_w) 视实现
};

Tensor conv2d_forward(
    const Tensor& X, const Tensor& W, const Tensor& b,
    int stride, int padding,
    Conv2DContext& ctx
);

// conv2d backward:
//   dW: (out_channels, in_channels*kh*kw) = dY_col @ im2col^T
//   db: sum over (batch, out_h*out_w)
//   dX: col2im(dW_col^T @ dY_col)  — 复用 sgn.col2im_add
std::tuple<Tensor, Tensor, Tensor> conv2d_backward(
    const Conv2DContext& ctx, const Tensor& X, const Tensor& W,
    const Tensor& dY
);

// ============================================================================
// maxpool2d: 沿 (kernel_h, kernel_w) 窗口取最大值
// ============================================================================
//   X: (batch, channels, in_h, in_w)
//   kernel: 窗口大小（正方形 kernel×kernel）
//   stride: 步长（默认 = kernel，无重叠）
//   Y: (batch, channels, out_h, out_w)
struct MaxPoolContext {
    int64_t batch, channels, in_h, in_w;
    int kernel, stride;
    int64_t out_h, out_w;
    std::vector<int64_t> argmax;  // 记录每个输出位置的最大值在输入中的线性索引（backward 路由用）
};

Tensor maxpool2d_forward(
    const Tensor& X, int kernel, int stride,
    MaxPoolContext& ctx
);

// ============================================================================
// avgpool2d: Y = 窗口均值（全局平均池化 GAP，ResNet 分类头）
// ============================================================================
struct AvgPoolContext {
    int64_t batch, channels, in_h, in_w;
    int kernel, stride;
    int64_t out_h, out_w;
};

Tensor avgpool2d_forward(
    const Tensor& X, int kernel, int stride,
    AvgPoolContext& ctx
);

// avgpool backward: dX 在窗口内均分 dY（dX += dY / k²，累加语义，调用方保证 dX 新分配）
Tensor avgpool2d_backward(const AvgPoolContext& ctx, const Tensor& dY);

// maxpool backward: 梯度路由，只传到前向最大值位置
Tensor maxpool2d_backward(const MaxPoolContext& ctx, const Tensor& dY);

// ============================================================================
// BufferPool: 线程局部临时缓冲区（已简化为空壳，见 ops_nn.cpp 注释）
// ============================================================================
// P1 将 Storage 接入全局可插拔分配器后，conv2d 临时矩阵的 Tensor 构造/释放
// 已由 PoolAllocator 自动复用内存；BufferPool 的对象缓存与之重叠且无收益。
// 保留 API（acquire/release/clear）以最小化调用方改动，实现为空壳。
class BufferPool {
public:
    static Tensor acquire(const std::vector<int64_t>& shape);
    static void release(Tensor&& t);
    static void clear();  // 清空池（无缓存状态，空操作）
};

// ============================================================================
// STE 变体：前向量化 + 反向 float32 直通
// ============================================================================
// 这些变体仅在前向对输入/权重做量化+反量化，反向与 float32 版本完全相同。
// 用途：在训练中模拟量化推理环境，同时保持反向精度。

// linear STE: Y = deq(Q(X @ W^T)) + b
//   内部对 matmul 结果做量化+反量化，bias 不量化
Tensor linear_forward_ste(const Tensor& X, const Tensor& W, const Tensor& b,
                          const QuantConfig& qcfg = QuantConfig{});

// conv2d STE: Y = deq(Q(im2col(X) @ W_col)) + b
//   内部对 matmul 结果做量化+反量化，bias 不量化
Tensor conv2d_forward_ste(
    const Tensor& X, const Tensor& W, const Tensor& b,
    int stride, int padding,
    Conv2DContext& ctx,
    const QuantConfig& qcfg = QuantConfig{}
);

// ============================================================================
// 算子融合：conv2d + relu
// ============================================================================
// 将 conv2d forward + relu forward 合并，以及 conv2d backward + relu gradient
// 合并到 col2im 步骤中，减少一次 dX 的写入-读取往返。

// conv2d + relu 融合 forward: 先 conv2d forward，再 relu 激活
//   返回 {Y, relu_mask}，其中 Y 是 relu 后的输出，relu_mask 用于 backward 融合
struct Conv2DReluContext {
    Conv2DContext conv_ctx;
    Tensor relu_mask;  // 前向 relu 激活的 mask（X_col > 0 ? 1.0 : 0.0），用于 backward 融合
};

Tensor conv2d_relu_forward(
    const Tensor& X, const Tensor& W, const Tensor& b,
    int stride, int padding,
    Conv2DReluContext& ctx
);

// conv2d + relu 融合 backward: 在 col2im 累加步骤中同时应用 relu 梯度
//   等价于 conv2d_backward + relu_backward，但减少一次内存遍历
std::tuple<Tensor, Tensor, Tensor> conv2d_relu_backward(
    const Conv2DReluContext& ctx, const Tensor& X, const Tensor& W,
    const Tensor& dY
);

// ============================================================================
// 全局策略上下文（进程级 static，2026-09-02 注释修正：非线程局部）
// ============================================================================
// 当前活跃的反向传播策略。由 autograd_nn.cpp 的包装函数在 forward 时读取，
// 决定使用哪种 forward/backward 算子。
// 默认 = FLOAT32（纯 float32）。
//
// ⚠️ 作用域注记（2026-09-02 审查修正）：本类各字段为【进程级 static】，与
// Tape 的 thread_local（每线程独立 tape）作用域不同。多线程各自 backward 时
// 共享同一策略/开关——项目训练用法为单线程 per-step 设置，语义自洽；若未来
// 引入多线程异构策略训练，需先将本类改 thread_local（当前不做，收益不抵风险）。
//
// 量化配置分离（链路核查 2026-08-21 P0-2 修复）：
//   - quant_config()    → 前向 STE 量化（linear/conv2d 的 *_forward_ste），
//                          由 set_ste_quant_config 设置，默认 bits=8
//   - bwd_quant_config() → 反向梯度量化（Tape::backward 的 GEF/SR/A1），
//                          由 set_quant_config 设置，默认 bits=16
//   此前两者共用同一 qcfg_，导致 set_ste_quant_config 每步切换前向位宽时
//   反向 SR 位宽被同步覆写（文档声明"反向 SR 16bit 默认"不成立）。
class StrategyContext {
public:
    static BackwardStrategy get() { return strategy_; }
    static void set(BackwardStrategy s) { strategy_ = s; }
    static QuantConfig& quant_config() { return qcfg_; }
    static QuantConfig& bwd_quant_config() { return bwd_qcfg_; }

    // int8 对（h,l）叶梯度存储开关（2026-09-02，A″ 三层访问方案）：
    //   默认 false = 现状 float32 叶梯度（逐位不变，零风险）。
    //   true 且策略 ∈ {SR, A1} 且 bwd bits=16 时：叶梯度（单路径）经
    //   sr_quantize_to_pair 一步出 pair 存储（2B/元素，显存 -50%），
    //   grad() 透明解码（惰性缓存，每 step 一次）；grad_pair() 显式视图
    //   （PairGradView，研究入口：肢分析 / fine-coarse 双读 / dot8 消费）。
    //   其他策略 / bwd bits≠16 / 多路径叶（权重共享）自动回退 float 路径。
    //   见 msint/pair_grad_carrier.h 与主报告 §八.4（入口决定瓶颈）。
    static bool pair_grad_store() { return pair_grad_store_; }
    static void set_pair_grad_store(bool v) { pair_grad_store_ = v; }

private:
    static BackwardStrategy strategy_;
    static QuantConfig qcfg_;       // 前向 STE（默认 8bit）
    static QuantConfig bwd_qcfg_;   // 反向 GEF/SR/A1（默认 16bit）
    static bool pair_grad_store_;
};

}  // namespace sgn_autograd
