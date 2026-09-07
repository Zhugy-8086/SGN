// tensor.h - 简化 Tensor 类（Storage + Shape + Stride）
//
// Stage 3.2 Phase 1：C++ Autograd 框架的基础张量类。
// 采用"算子优先"策略，先实现最小可用 Tensor 与 matmul 算子互相验证。
//
// 设计要点：
//   1. Storage = 连续内存块，shared_ptr 管理引用计数
//   2. Tensor = shared_ptr<Storage> + Shape + Stride
//   3. 基础操作：reshape / contiguous / 元素访问
//   4. 视图操作（view/transpose/permute/squeeze/unsqueeze/expand）：
//      用 std::mdspan 的 layout_stride 实现非 contiguous 视图。
//      mdspan 是内部实现细节，不暴露到公开 API。保持公开接口不变。
//      目标：减少未来 GPU 移植时 Tensor 数据视图层的重写量。
//      CUDA 12.x 的 CCCL 提供 cuda::std::mdspan，接口兼容。
//   5. Stride 以"元素数"为单位（非字节数），与 PyTorch 内部约定一致

#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>
#include <string>
#include <stdexcept>
#include <mdspan>

namespace sgn_autograd {

// ============================================================================
// Storage: 连续内存块（shared_ptr 管理，引用计数）
// ============================================================================
//
// 仅持有 float32 连续内存。通过 shared_ptr 实现多 Tensor 间共享同一块内存
// （例如 reshape 共享 storage）。生命周期由引用计数自动管理。
// 内存分配经 sgn::allocate_floats / sgn::deallocate_floats（common/allocator.h）
// ——可插拔分配器（默认 stdlib；可接 PoolAllocator 或外部后端分配器）。
class Storage {
public:
    // 构造：分配 n 个 float 的连续内存，初始化为 0
    explicit Storage(size_t n);

    // 从现有数据构造（拷贝 n 个 float）
    Storage(const float* data, size_t n);

    // 访问
    float* data() { return data_.get(); }
    const float* data() const { return data_.get(); }
    size_t size() const { return size_; }

private:
    // shared_ptr<float> + 自定义删除器（经 sgn 分配器释放）。get() 即 float*。
    std::shared_ptr<float> data_;
    size_t size_;
};

// ============================================================================
// Tensor: 多维张量（Storage + Shape + Stride）
// ============================================================================
//
// 内部使用 std::mdspan 的 layout_stride 实现视图操作。
// shape_/stride_ 缓存保留（选项 A），确保公开 API 向后兼容。
// 例如 shape=[2,3,4] → stride=[12,4,1]。
class Tensor {
public:
    // 构造函数
    Tensor();  // 空张量
    explicit Tensor(const std::vector<int64_t>& shape);  // 按形状分配（元素初始化为 0）
    Tensor(const std::vector<int64_t>& shape, const float* data);  // 从数据构造（拷贝）

    // 属性
    const std::vector<int64_t>& shape() const { return shape_; }
    const std::vector<int64_t>& stride() const { return stride_; }
    size_t ndim() const { return shape_.size(); }
    size_t numel() const;  // 元素总数
    size_t dim() const { return ndim(); }  // 别名（兼容 PyTorch 习惯）
    std::vector<int64_t> size() const;  // 返回 shape 的拷贝（兼容 PyTorch 习惯）

    // === Autograd 支持（Phase 2）===
    // id_：全局唯一标识，用于 Tape 的 grads_ map 索引。
    //   拷贝构造保持原 id（共享 storage 视为同一张量，grad 累加到同一处）。
    //   新构造分配新 id。
    int64_t id() const { return id_; }
    bool requires_grad() const { return requires_grad_; }
    void set_requires_grad(bool rg) { requires_grad_ = rg; }

    // 数据访问
    float* data() { return storage_->data(); }
    const float* data() const { return storage_->data(); }
    Storage& storage() { return *storage_; }
    const Storage& storage() const { return *storage_; }

    // 元素访问（1D 和 2D 便捷接口，要求 contiguous）
    float& at(size_t i);  // 1D
    float at(size_t i) const;
    float& at(size_t i, size_t j);  // 2D
    float at(size_t i, size_t j) const;

    // === 基础操作（已实现）===
    Tensor reshape(const std::vector<int64_t>& new_shape) const;  // 返回新 Tensor（共享 storage）
    // contiguous()：若已连续返回浅拷贝（共享 storage，但分配【新 id】，不继承
    // tape 记录——需要梯度链连续时用 autograd 层的 sgn_autograd::reshape 等
    // free function，审计 T-3）；否则深拷贝为连续内存
    Tensor contiguous() const;
    bool is_contiguous() const;
    // moved-from 判据（M5 修复 2026-09-07）：move 后 storage_ 为空——此时
    // shape_ 亦空而 numel() 返回 1（空积初值），numel==0 防线失效；
    // 标量梯度（shape 合法）不受影响——defunct 只认 storage_。
    bool defunct() const noexcept { return !static_cast<bool>(storage_); }
    // clone()：深拷贝到新 storage + 新 id（与拷贝构造的"共享 storage + 同 id"
    // 语义区分）。不记录到 tape（叶子副本）——安全审计 2026-08-16 T-7
    Tensor clone() const;

    // === 视图操作（用 std::mdspan layout_stride 实现，共享 storage）===
    Tensor view(const std::vector<int64_t>& new_shape) const;  // 用 mdspan 计算 stride，共享 storage
    Tensor transpose(int64_t dim0, int64_t dim1) const;  // 交换 stride（零拷贝）
    Tensor permute(const std::vector<int64_t>& dims) const;  // 任意维度排列（零拷贝）
    Tensor squeeze(int64_t dim) const;  // 去除大小为 1 的维度
    Tensor unsqueeze(int64_t dim) const;  // 插入大小为 1 的维度
    Tensor expand(const std::vector<int64_t>& new_shape) const;  // 广播（stride=0）

private:
    std::shared_ptr<Storage> storage_;
    std::vector<int64_t> shape_;
    std::vector<int64_t> stride_;  // 各维度的步长（元素数，非字节数）
    int64_t id_ = 0;               // 全局唯一标识（Phase 2 Autograd）
    bool requires_grad_ = false;   // 是否需要梯度（Phase 2 Autograd）

    void compute_stride();  // 根据 shape 计算 contiguous stride
    void check_shape_match(const std::vector<int64_t>& expected) const;
};

// ============================================================================
// 初始化辅助函数（原地修改 Tensor 数据）
// ============================================================================

// Kaiming uniform 初始化（He 初始化）
//   fan_in: 输入神经元数
//   a: ReLU 的负斜率（默认 sqrt(5) 对应 Kaiming uniform）
//   bound = sqrt(6 / ((1 + a^2) * fan_in)) * gain
// ⚠️ rng 为 thread_local：进程 fork（如 Python multiprocessing 的 fork 模式）
// 后子进程会继承相同 rng 状态导致重复随机序列——fork 后需重新初始化
// （安全审计 2026-08-16 T-8）
inline void kaiming_uniform_(Tensor& t, int64_t fan_in, float a = 2.2360679775f /* sqrt(5) */) {
    float gain = std::sqrt(2.0f / (1.0f + a * a));
    float bound = gain * std::sqrt(3.0f / static_cast<float>(fan_in));
    std::uniform_real_distribution<float> dist(-bound, bound);
    static thread_local std::mt19937 rng(std::random_device{}());
    float* data = t.data();
    size_t n = t.numel();
    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }
}

// 均匀分布初始化
inline void uniform_(Tensor& t, float low, float high) {
    std::uniform_real_distribution<float> dist(low, high);
    static thread_local std::mt19937 rng(std::random_device{}());
    float* data = t.data();
    size_t n = t.numel();
    for (size_t i = 0; i < n; ++i) {
        data[i] = dist(rng);
    }
}

// 常量填充
inline void fill_(Tensor& t, float value) {
    size_t n = t.numel();
    float* data = t.data();
    for (size_t i = 0; i < n; ++i) {
        data[i] = value;
    }
}

}  // namespace sgn_autograd
