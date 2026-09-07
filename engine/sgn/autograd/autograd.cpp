// autograd.cpp - tape-based Autograd 引擎实现
//
// Stage 3.2 Phase 2：Tape 类 + autograd-aware matmul。
//
// 实现说明：
//   - Tape::current() 返回 thread_local 单例（每线程独立，多线程训练安全）
//   - backward() 逆序遍历 records_，跳过无梯度的 output
//   - accumulate_grad() 处理同 id 的多次累加（同一 Tensor 参与多个 op）
//   - matmul() 构造 MatmulNode（P2：type-erased NodeBase，捕获 A/B 浅拷贝）
//
// P2 改造（2026-08-15，见 engine_infrastructure_plan P2）：
//   - backward 逻辑从 std::function 改为 NodeBase 派生类（MatmulNode/ReshapeNode）。
//   - backward 中 output grad 用 move 语义消费（每个 output grad 恰好被其 record 消费一次），
//     避免中间梯度拷贝；backward 完成后自动释放 records_（graph），叶梯度保留在 grads_。

#include "autograd.h"
#include "ops.h"
#include "ops_nn.h"
#include "backward_strategy.h"
#include "mkern/simd/simd_api.h"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <utility>

namespace sgn_autograd {

// int8 对叶梯度存储（A″）：pair 原语在 sgn_msint 命名空间（pair_grad_carrier.h）
using sgn_msint::PairGradCarrier;
using sgn_msint::sr_quantize_to_pair;
using sgn_msint::decode_pair_fine_f32;

// ============================================================================
// Tape 单例
// ============================================================================
Tape& Tape::current() {
    // thread_local：每线程独立实例，多线程训练互不干扰
    static thread_local Tape instance;
    return instance;
}

// ============================================================================
// StrategyContext 静态成员定义
// ============================================================================
// 集中在本文件：test_phase2 等不编译 ops_nn.cpp 的测试目标也能解析
// backward() 中的 StrategyContext::get() / quant_config() 引用（2026-08-15）。
BackwardStrategy StrategyContext::strategy_ = BackwardStrategy::FLOAT32;
QuantConfig StrategyContext::qcfg_ = QuantConfig{};
// 反向梯度量化配置（GEF/SR/A1），默认 16bit——与前向 STE（默认 8bit）分离
// （链路核查 2026-08-21 P0-2 修复：此前共用 qcfg_，前向切位宽会覆写反向）
QuantConfig StrategyContext::bwd_qcfg_ = QuantConfig{16, 4.0f};
// int8 对叶梯度存储开关（默认关 = 现状 float32 逐位不变；见 ops_nn.h 注释）
bool StrategyContext::pair_grad_store_ = false;

// ============================================================================
// record: 记录一次前向操作
// ============================================================================
void Tape::record(Record r) {
    if (!recording_) {
        return;  // 未在录制，忽略
    }
    records_.push_back(std::move(r));
    consumed_ = false;  // 新的前向记录使 tape 重新可用（审计 AG-9）
}

// ============================================================================
// backward: 逆序遍历 tape，累加梯度
// ============================================================================
void Tape::backward(const Tensor& output, Tensor grad_output) {
    // 防"backward 后未重新 forward/clear 再次 backward"静默无操作（审计 AG-9）
    if (consumed_) {
        throw std::runtime_error(
            "Tape::backward(): tape already consumed by a previous backward; "
            "record new forward ops (or call clear()) before backward again");
    }

    // 设置 output 的初始梯度
    grads_[output.id()] = std::move(grad_output);

    // F1（2026-09-02 审查）：清上一轮 pair 存储与解码缓存。
    // backward 是新计算起点——AG-9 允许"backward 后重新 forward 再 backward（不
    // clear）"，此时 pair_grads_ 若残留上一 step 的叶梯度，grad() 会优先命中返回
    // 过期梯度（且第二次该叶可能不再满足 pair 条件 → 缓存与 grads_ 彻底错位）。
    // 与 float 路径的 grads_ 累加语义不同（原引擎二次 backward 混沌语义的延续，
    // 不扩大战场），pair 路径取覆盖语义，入口清空最简单且封闭。
    pair_grads_.clear();
    pair_decoded_.clear();

    // had_records：本次 backward 是否消费了 tape 记录。仅消费过 records 才
    // 置 consumed_——纯叶梯度播种（无 records 的 backward）不占用"消费"状态，
    // 允许后续再次播种（审计 AG-9）
    const bool had_records = !records_.empty();

    // 读取当前反向传播策略（GEF/SR/A1 需要在梯度返回后做量化）
    BackwardStrategy s = StrategyContext::get();
    bool quantize_grads = (s == BackwardStrategy::GEF || s == BackwardStrategy::SR
                           || s == BackwardStrategy::A1);
    // 反向梯度量化使用独立的 bwd 配置（默认 16bit），不受前向 STE 位宽切换影响
    // （链路核查 2026-08-21 P0-2 修复：此前读 quant_config() 与前向共用）
    int bits = StrategyContext::bwd_quant_config().bits;
    float clip_sigma = StrategyContext::bwd_quant_config().clip_sigma;

    // int8 对叶梯度存储分流（2026-09-02 A″，见 StrategyContext::pair_grad_store）：
    // 仅 SR/A1 + bits=16 生效；叶判定 = 出现于 input_ids 且不作任何 record 的
    // output_id 且单路径（count==1，多路径叶自动回退 float 规避 pair 域累加语义）。
    const bool pair_active = StrategyContext::pair_grad_store() &&
                             (s == BackwardStrategy::SR || s == BackwardStrategy::A1) &&
                             bits == 16;
    std::unordered_map<int64_t, int> pair_leaf_ids;
    if (pair_active) {
        for (const auto& rec : records_) {
            for (size_t i = 0; i < rec.input_ids.size(); ++i) {
                if (i < rec.input_requires_grad.size() && !rec.input_requires_grad[i]) {
                    continue;
                }
                ++pair_leaf_ids[rec.input_ids[i]];
            }
        }
        for (const auto& rec : records_) {
            pair_leaf_ids.erase(rec.output_id);  // 中间结果（后续消费）不走 pair
        }
        for (auto it = pair_leaf_ids.begin(); it != pair_leaf_ids.end();) {
            if (it->second > 1) {
                it = pair_leaf_ids.erase(it);  // 多路径（权重共享）回退 float
            } else {
                ++it;
            }
        }
    }

    // 异常安全（审计 AG-4）：apply 可能抛异常（shape 不匹配等），必须保证
    // records_ 在异常路径也被清空——否则残留的半消费 records 会混入下一次
    // backward，访问 moved-from 的梯度 Tensor。
    try {
        // 逆序遍历 records（从最后执行的操作向前回溯）
        for (auto it = records_.rbegin(); it != records_.rend(); ++it) {
            Record& rec = *it;

            // 查找当前 output 的梯度
            auto grad_it = grads_.find(rec.output_id);
            if (grad_it == grads_.end()) {
                continue;  // 该 output 无梯度（不影响最终 loss），跳过
            }
            if (!rec.backward_node) {
                continue;  // 空 Node（防御），跳过
            }

            // P2：move 语义消费 output grad——每个 output grad 恰好被其所属 record
            // 消费一次，move 出 map 后该 id 的梯度不再需要（中间结果），避免拷贝。
            Tensor output_grad = std::move(grad_it->second);
            std::vector<Tensor> input_grads = rec.backward_node->apply(output_grad);

            // GEF/SR/A1 策略：对每个返回的梯度张量做量化（原地修改）
            // A1 主线（2026-08-19）：反向梯度用 SR（无偏白噪声，路线图主推）
            std::vector<size_t> pair_taken;  // F2：被 pair 消费的 input 索引
            if (quantize_grads) {
                for (size_t gi = 0; gi < input_grads.size(); ++gi) {
                    auto& g = input_grads[gi];
                    // pair 分流：单路径叶 → 一步出 pair（同 SRNG 同舍入序列，
                    // 直接编码整数肢，不经 float32 网格往返；见 pair_grad_carrier.h）
                    if (pair_active && gi < rec.input_ids.size() &&
                        pair_leaf_ids.count(rec.input_ids[gi]) > 0) {
                        PairGradEntry entry;
                        entry.shape = g.shape();
                        sr_quantize_to_pair(g.data(), g.numel(), clip_sigma,
                                            entry.carrier);
                        pair_grads_[rec.input_ids[gi]] = std::move(entry);
                        pair_taken.push_back(gi);
                        continue;  // F2：不置空、不落入下方累加（避免 grads_ 空脏条目）
                    }
                    if (s == BackwardStrategy::GEF) {
                        gef_quantize_grad(g.data(), g.numel(), bits, clip_sigma);
                    } else {  // SR / A1
                        sr_quantize_grad(g.data(), g.numel(), bits, clip_sigma);
                    }
                }
            }

            // 累加到各 input（跳过 requires_grad=false 的 input、被 pair 消费的项）
            size_t n = std::min(rec.input_ids.size(), input_grads.size());
            for (size_t i = 0; i < n; ++i) {
                if (i < rec.input_requires_grad.size() && !rec.input_requires_grad[i]) {
                    continue;  // 该 input 不需要梯度
                }
                if (std::find(pair_taken.begin(), pair_taken.end(), i)
                    != pair_taken.end()) {
                    continue;  // F2：pair 叶不走 float 累加（grad() 走 pair 解码）
                }
                accumulate_grad(rec.input_ids[i], std::move(input_grads[i]));
            }
        }
    } catch (...) {
        records_.clear();
        consumed_ = true;
        throw;
    }

    // P2：backward 消费完毕，自动释放 graph（records_），替代手动 clear()。
    // 叶梯度保留在 grads_ 供 grad(id) 查询；下次 forward 前由用户 clear() 重置。
    records_.clear();
    if (had_records) {
        consumed_ = true;  // 消费过 tape：再次 backward 需新的 forward 记录
    }
}

// ============================================================================
// grad: 查询某个 Tensor 的梯度
// ============================================================================
const Tensor* Tape::grad(int64_t tensor_id) const {
    // pair 存储：惰性解码 + 缓存（同一 step 内多次查询共享一次解码；clear 失效）。
    // fine 读法 ≡ Q16-SR 精确恢复（V7 dev=0，数学层 bit-exact）。
    auto pit = pair_grads_.find(tensor_id);
    if (pit != pair_grads_.end()) {
        auto cit = pair_decoded_.find(tensor_id);
        if (cit == pair_decoded_.end()) {
            const auto& e = pit->second;
            std::vector<float> buf(e.carrier.n);
            decode_pair_fine_f32(e.carrier, buf.data());
            cit = pair_decoded_
                      .emplace(tensor_id, Tensor(e.shape, buf.data()))
                      .first;
        }
        return &cit->second;
    }
    auto it = grads_.find(tensor_id);
    if (it == grads_.end()) {
        return nullptr;
    }
    // moved-from entry（被 backward move 消费的中间梯度）或空梯度：
    // defunct（storage_ 空）或 numel 为 0，返回 nullptr 防调用方误用
    // （审计 AG-1；M5 修复：moved-from 的 numel()==1，原防线失效）
    if (it->second.defunct() || it->second.numel() == 0) {
        return nullptr;
    }
    return &it->second;
}

Tensor* Tape::grad_mutable(int64_t tensor_id) {
    // pair 存储的叶梯度无 float 常驻副本：返回 nullptr（防调用方原地清零
    // 只清缓存不清 pair 主存造成语义不一致；见 autograd.h 注释）
    if (pair_grads_.count(tensor_id) > 0) {
        return nullptr;
    }
    auto it = grads_.find(tensor_id);
    if (it == grads_.end() || it->second.defunct() || it->second.numel() == 0) {
        return nullptr;
    }
    return &it->second;
}

// ============================================================================
// grad_pair: int8 对叶梯度显式视图（研究入口，A″；见 autograd.h）
// ============================================================================
const sgn_msint::PairGradCarrier* Tape::grad_pair(int64_t tensor_id) const {
    auto it = pair_grads_.find(tensor_id);
    if (it == pair_grads_.end()) {
        return nullptr;
    }
    return &it->second.carrier;
}

// ============================================================================
// clear: 清空 tape
// ============================================================================
void Tape::clear() {
    records_.clear();
    grads_.clear();
    pair_grads_.clear();
    pair_decoded_.clear();
    recording_ = false;
    consumed_ = false;
}

// ============================================================================
// accumulate_grad: 累加梯度（element-wise add）
// ============================================================================
void Tape::accumulate_grad(int64_t id, Tensor g) {
    auto it = grads_.find(id);
    if (it == grads_.end()) {
        // 首次出现：直接存入
        grads_[id] = std::move(g);
    } else {
        // 已存在：element-wise 累加
        Tensor& existing = it->second;
        size_t n = existing.numel();
        if (n != g.numel()) {
            throw std::runtime_error(
                "accumulate_grad: numel mismatch (" + std::to_string(n) +
                " vs " + std::to_string(g.numel()) + ") for tensor id " +
                std::to_string(id));
        }
        float* dst = existing.data();
        const float* src = g.data();
        // AVX2 向量化累加迁出到 simd::accum_f32（AVX2 8 路 + 标量尾，见 simd 原语层）
        sgn::simd::accum_f32(dst, src, static_cast<int64_t>(n));
    }
}

// ============================================================================
// MatmulNode: matmul 的 backward 逻辑（P2 type-erased NodeBase）
// ============================================================================
// 捕获 A/B 的浅拷贝（Tensor 拷贝共享 storage，不复制数据）。
// apply: 计算 dA = dY @ B^T, dB = A^T @ dY。
class MatmulNode final : public NodeBase {
public:
    MatmulNode(Tensor A, Tensor B) : A_(std::move(A)), B_(std::move(B)) {}

    std::vector<Tensor> apply(const Tensor& dY) override {
        auto [dA, dB] = matmul_backward(A_, B_, dY);
        return {std::move(dA), std::move(dB)};
    }

private:
    Tensor A_, B_;
};

// ============================================================================
// matmul: autograd-aware matmul
// ============================================================================
Tensor matmul(const Tensor& A, const Tensor& B) {
    // 始终执行前向计算
    Tensor C = matmul_forward(A, B);

    // 判断是否需要记录到 tape
    bool needs_grad = A.requires_grad() || B.requires_grad();
    Tape& tape = Tape::current();
    if (tape.is_recording() && needs_grad) {
        // 设置输出 requires_grad
        C.set_requires_grad(true);

        // 构造 Record（backward 逻辑用 MatmulNode 类型擦除）
        Record rec;
        rec.op_type = "matmul";
        rec.output_id = C.id();
        rec.input_ids = {A.id(), B.id()};
        rec.input_requires_grad = {A.requires_grad(), B.requires_grad()};
        rec.backward_node = std::make_unique<MatmulNode>(A, B);

        tape.record(std::move(rec));
    }

    return C;
}

// ============================================================================
// ReshapeNode: reshape 的 backward 逻辑（P2 type-erased NodeBase）
// ============================================================================
// 捕获原 shape。apply: dX = dY reshape 回原 shape（拷贝，避免共享 storage 的累加问题）。
class ReshapeNode final : public NodeBase {
public:
    explicit ReshapeNode(std::vector<int64_t> orig_shape)
        : orig_shape_(std::move(orig_shape)) {}

    std::vector<Tensor> apply(const Tensor& dY) override {
        // 拷贝梯度数据到原 shape 的新 Tensor（避免共享 storage 的累加问题）。
        // dY 可能非连续（上游 view/permute 过），先 materialize 再 memcpy，
        // 否则按连续布局拷贝会读到错误数据（安全审计 2026-08-16 AG-7）
        Tensor dY_c = dY.is_contiguous() ? dY : dY.contiguous();
        Tensor dX(orig_shape_);
        std::memcpy(dX.data(), dY_c.data(), dY_c.numel() * sizeof(float));
        return {std::move(dX)};
    }

private:
    std::vector<int64_t> orig_shape_;
};

// ============================================================================
// reshape: autograd-aware reshape
// ============================================================================
// forward: Y = X.reshape(new_shape)（共享 storage）
// backward: dX = dY reshape 回原 shape（拷贝，避免共享 storage 导致的累加问题）
Tensor reshape(const Tensor& X, const std::vector<int64_t>& new_shape) {
    Tensor Y = X.reshape(new_shape);  // forward（共享 storage）

    bool needs_grad = X.requires_grad();
    Tape& tape = Tape::current();
    if (tape.is_recording() && needs_grad) {
        Y.set_requires_grad(true);

        Record rec;
        rec.op_type = "reshape";
        rec.output_id = Y.id();
        rec.input_ids = {X.id()};
        rec.input_requires_grad = {X.requires_grad()};
        rec.backward_node = std::make_unique<ReshapeNode>(X.shape());

        tape.record(std::move(rec));
    }
    return Y;
}

}  // namespace sgn_autograd
