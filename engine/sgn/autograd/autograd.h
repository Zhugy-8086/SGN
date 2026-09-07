// autograd.h - tape-based Autograd 引擎
//
// Stage 3.2 Phase 2：基于 tape 的自动微分引擎。
//
// 设计（详见 spec.md 决策 2）：
//   1. tape-based Autograd（PyTorch v0.4 之后的现代设计）
//      - 前向时 record() 记录操作到 vector（tape）
//      - backward() 逆序遍历 tape，调用各算子的 backward 逻辑
//      - 无循环引用，实现简单，内存随 tape clear 自动释放
//   2. grad 存在 Tape 的 map 中（按 tensor id 索引），不在 Tensor 里
//      - 避免修改 Tensor 的内存模型（Phase 1 已定型）
//      - 多个 op 的 grad 自动累加到同一 id
//   3. thread_local Tape（Tape::current()），多线程训练各线程独立，
//      测试间用 clear() 清空。
//
// P2 改造（2026-08-15，见 engine_infrastructure_plan P2）：
//   - backward 逻辑从 std::function 改为 type-erased NodeBase（Record::backward_node），
//     避免 std::function 对较大捕获（BN/Conv 的 ctx + 多个 Tensor）的堆分配与间接调用开销。
//   - 中间梯度在 backward 中用 move 语义消费（每个 output grad 恰好被其所属 record
//     消费一次），backward 完成后自动释放 graph（records_ 清空），叶梯度保留在
//     grads_ 供 grad(id) 查询——替代手动 clear() 的"自动管理"。

#pragma once

#include "tensor.h"
#include "msint/pair_grad_carrier.h"

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace sgn_autograd {

// ============================================================================
// NodeBase: 类型擦除的 backward 逻辑（P2 替代 std::function）
// ============================================================================
// 每个前向算子构造其派生 Node，捕获 backward 所需的输入（浅拷贝 Tensor）与
// 中间 context。Tape::backward() 逆序遍历 records 时调用 apply(grad_output)，
// 返回各 input 的梯度（顺序与 Record::input_ids 对应）。
// 派生类在 autograd.cpp / autograd_nn.cpp 中定义（就近于算子实现）。
//
// ⚠️ 重要约束（安全审计 2026-08-16 AG-6）：Node 捕获的是 Tensor 浅拷贝
// （共享 storage）。forward 记录之后、backward 之前，【禁止原地修改】被捕获
// 的输入（如提前用优化器更新权重）——否则 backward 读到更新后的数据，
// 梯度静默错误（本项目暂无 version counter 检测机制）。优化器必须在
// backward 之后更新参数。
class NodeBase {
public:
    virtual ~NodeBase() = default;
    // 计算各 input 的梯度（顺序与 input_ids 对应）
    virtual std::vector<Tensor> apply(const Tensor& grad_output) = 0;
};

// ============================================================================
// Record: tape 的一条记录（一次前向操作）
// ============================================================================
struct Record {
    std::string op_type;                 // 操作类型（"matmul", "relu", ...）
    int64_t output_id = 0;               // 输出 Tensor 的 id
    std::vector<int64_t> input_ids;      // 各输入 Tensor 的 id（用于 grad 累加）
    std::vector<bool> input_requires_grad;  // 各 input 是否需要梯度（不需则跳过累加）
    // backward_node: 类型擦除的 backward 逻辑（P2 替代 std::function）
    //   接受 output 的 grad，返回各 input 的 grad（顺序与 input_ids 对应）
    std::unique_ptr<NodeBase> backward_node;
};

// ============================================================================
// Tape: 操作记录带（tape-based Autograd 核心引擎）
// ============================================================================
class Tape {
public:
    // thread_local tape 单例（每线程独立，多线程训练安全）
    static Tape& current();

    // === 录制控制 ===
    void start_recording() { recording_ = true; }
    void stop_recording() { recording_ = false; }
    bool is_recording() const { return recording_; }

    // === 前向记录 ===
    // 记录一次前向操作到 tape（仅在 is_recording() 时调用）
    void record(Record r);

    // === 反向传播 ===
    // 从 output 开始，以 grad_output 为初始梯度，逆序遍历 tape 累加梯度。
    // backward 完成后，各输入的梯度可通过 grad(tensor.id()) 获取。
    // P2：backward 消费完毕后自动释放 graph（records_ 清空），叶梯度保留在 grads_。
    //
    // ⚠️ 线程约束（安全审计 2026-08-16 AG-5）：Tape 是 thread_local——
    // forward 与 backward 必须在同一线程。跨线程 backward 找不到本线程的
    // records，会静默无梯度（无报错）。
    //
    // ⚠️ 重复调用（审计 AG-9）：backward 消费 tape 后再次调用（未重新
    // forward/clear）将抛 std::runtime_error，防止静默无操作。
    void backward(const Tensor& output, Tensor grad_output);

    // === 梯度查询 ===
    // 获取某个 Tensor 的梯度（backward 后调用）。
    // 返回 nullptr 表示该 Tensor 无梯度（未参与计算、不需要梯度、
    // 或梯度已被 backward move 消费——中间结果的梯度不保留，审计 AG-1）。
    //
    // pair_grad_store 开启时：单路径叶梯度以 int8 对存储（pair_grads_），
    // 本函数透明解码返回（惰性缓存于 pair_decoded_，同一 step 内多次查询
    // 共享一次解码；clear() 失效）。fine 读法 ≡ Q16-SR 精确恢复（V7 dev=0）。
    const Tensor* grad(int64_t tensor_id) const;

    // 可变梯度访问（zero_grad 原地清零等场景的显式接口，
    // 替代调用方 const_cast——安全审计 2026-08-16 M-3）。
    // 返回 nullptr 语义同 grad()。
    // 注意：pair 存储的叶梯度无 float 常驻副本，本函数返回 nullptr
    // （Python optimizer 走 tensor.grad 属性 → grad()，不受影响）。
    Tensor* grad_mutable(int64_t tensor_id);

    // === int8 对叶梯度显式视图（研究入口，2026-09-02 A″） ===
    // 返回该叶梯度的 pair 载体（拷贝，独立于 Tape 生命周期）；非 pair 存储
    // 的 id 返回 nullptr。通过 pybind 暴露为 PairGradView（肢 numpy 视图 /
    // fine-coarse 双读 / dot8 消费），见 autograd_bindings.cpp。
    const sgn_msint::PairGradCarrier* grad_pair(int64_t tensor_id) const;

    // === 重置 ===
    void clear();  // 清空 records 和 grads（测试间重置）

    size_t size() const { return records_.size(); }
    bool empty() const { return records_.empty(); }

    // === 内省访问器（Phase 4 调试工具用） ===
    const std::vector<Record>& records() const { return records_; }

private:
    std::vector<Record> records_;
    std::unordered_map<int64_t, Tensor> grads_;  // 按 tensor id 索引的梯度表
    bool recording_ = false;
    bool consumed_ = false;  // backward 是否已消费 tape（防重复 backward，审计 AG-9）

    // --- int8 对叶梯度存储（2026-09-02 A″，见 StrategyContext::pair_grad_store） ---
    // pair_grads_：单路径叶梯度的 pair 载体（shape 保留供 grad() 解码重建 Tensor）。
    // pair_decoded_：grad() 惰性解码缓存（mutable——grad() 为 const；clear() 失效）。
    struct PairGradEntry {
        sgn_msint::PairGradCarrier carrier;
        std::vector<int64_t> shape;
    };
    std::unordered_map<int64_t, PairGradEntry> pair_grads_;
    mutable std::unordered_map<int64_t, Tensor> pair_decoded_;

    // 累加梯度到 grads_（若 id 已存在则 element-wise 相加）
    void accumulate_grad(int64_t id, Tensor g);
};

// ============================================================================
// Autograd-aware 算子
// ============================================================================
// matmul: 自动微分版 matmul
//   - 始终执行前向计算（matmul_forward）
//   - 若 Tape 正在记录 且 任一输入 requires_grad，则记录到 tape
//   - 输出的 requires_grad = A.requires_grad || B.requires_grad
Tensor matmul(const Tensor& A, const Tensor& B);

// reshape: 自动微分版 reshape（tape 记录，backward 将梯度 reshape 回原形状）
//   用于 flatten (B,C,H,W)→(B,C*H*W) 等场景
Tensor reshape(const Tensor& X, const std::vector<int64_t>& new_shape);

}  // namespace sgn_autograd
