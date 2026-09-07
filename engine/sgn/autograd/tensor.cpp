// tensor.cpp - Storage + Tensor 类实现
//
// Stage 3.2 Phase 1：与 tensor.h 配套的实现文件。
// 详见 tensor.h 头部注释的设计要点。

#include "tensor.h"

#include "common/allocator.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <numeric>

namespace sgn_autograd {

// 全局 Tensor ID 计数器（Phase 2 Autograd）
// 每个 *新构造* 的 Tensor 分配唯一 id；拷贝构造保持原 id（共享 storage 视为同一张量）。
namespace {
std::atomic<int64_t> g_next_tensor_id{1};
}  // namespace

// ============================================================================
// Storage 实现
// ============================================================================

Storage::Storage(size_t n) : data_(nullptr), size_(n) {
    if (n == 0) {
        return;  // 空存储，data_ 保持 nullptr
    }
    // 经 sgn 可插拔分配器分配（默认 stdlib；可接 PoolAllocator / 外部后端）。
    float* p = sgn::sgn_allocate_floats(n);
    if (p == nullptr) {
        throw std::bad_alloc();
    }
    // 自定义删除器：在【分配时】捕获当前 deallocator，保证 alloc/dealloc 严格
    // 配对——即使全局分配器在此 Storage 存活期间被切换（如 std→pool），也用
    // 分配它的分配器释放。否则旧 std 块会被 pool_dealloc 回收入池并按 size-class
    // 复用，导致越界写（安全审计 2026-08-16 P-2/P-6）。
    const size_t nbytes = n * sizeof(float);
    sgn::DeallocFn dealloc =
        sgn::detail::dealloc_ref().load(std::memory_order_relaxed);
    data_ = std::shared_ptr<float>(p, [nbytes, dealloc](float* ptr) {
        dealloc(ptr, nbytes);
    });
    std::fill(data_.get(), data_.get() + n, 0.0f);  // 初始化为 0
}

Storage::Storage(const float* data, size_t n) : data_(nullptr), size_(n) {
    if (n == 0) {
        return;
    }
    float* p = sgn::sgn_allocate_floats(n);
    if (p == nullptr) {
        throw std::bad_alloc();
    }
    // 同上：分配时捕获 deallocator，保证配对释放（审计 P-2/P-6）
    const size_t nbytes = n * sizeof(float);
    sgn::DeallocFn dealloc =
        sgn::detail::dealloc_ref().load(std::memory_order_relaxed);
    data_ = std::shared_ptr<float>(p, [nbytes, dealloc](float* ptr) {
        dealloc(ptr, nbytes);
    });
    if (data != nullptr) {
        std::memcpy(data_.get(), data, n * sizeof(float));
    } else {
        std::fill(data_.get(), data_.get() + n, 0.0f);
    }
}

// ============================================================================
// Tensor 实现
// ============================================================================

Tensor::Tensor()
    : storage_(std::make_shared<Storage>(0)), id_(g_next_tensor_id.fetch_add(1)) {}

Tensor::Tensor(const std::vector<int64_t>& shape)
    : storage_(std::make_shared<Storage>(static_cast<size_t>(
          std::accumulate(shape.begin(), shape.end(), int64_t{1},
                          std::multiplies<int64_t>())))),
      shape_(shape),
      id_(g_next_tensor_id.fetch_add(1)) {
    compute_stride();
}

Tensor::Tensor(const std::vector<int64_t>& shape, const float* data)
    : storage_(std::make_shared<Storage>(
          data, static_cast<size_t>(std::accumulate(shape.begin(), shape.end(), int64_t{1},
                                                    std::multiplies<int64_t>())))),
      shape_(shape),
      id_(g_next_tensor_id.fetch_add(1)) {
    compute_stride();
}

size_t Tensor::numel() const {
    return static_cast<size_t>(std::accumulate(shape_.begin(), shape_.end(), int64_t{1},
                                               std::multiplies<int64_t>()));
}

std::vector<int64_t> Tensor::size() const {
    return shape_;  // 返回拷贝
}

void Tensor::compute_stride() {
    // row-major（C-contiguous）stride：shape=[2,3,4] → stride=[12,4,1]
    stride_.assign(shape_.size(), 0);
    if (shape_.empty()) {
        return;
    }
    stride_.back() = 1;
    for (size_t i = shape_.size() - 1; i > 0; --i) {
        stride_[i - 1] = stride_[i] * shape_[i];
    }
}

void Tensor::check_shape_match(const std::vector<int64_t>& expected) const {
    if (shape_ != expected) {
        throw std::invalid_argument("Tensor shape mismatch");
    }
}

// --- 元素访问 ---

float& Tensor::at(size_t i) {
    if (ndim() != 1) {
        throw std::invalid_argument("at(size_t) requires 1D tensor");
    }
    // M4 修复 2026-09-07：线性索引假定连续布局——非连续（expand 视图）直接拒绝
    if (!is_contiguous()) {
        throw std::invalid_argument("at(size_t) requires contiguous tensor");
    }
    if (i >= static_cast<size_t>(shape_[0])) {
        throw std::out_of_range("Tensor 1D index out of range");
    }
    return storage_->data()[i];
}

float Tensor::at(size_t i) const {
    if (ndim() != 1) {
        throw std::invalid_argument("at(size_t) requires 1D tensor");
    }
    if (!is_contiguous()) {
        throw std::invalid_argument("at(size_t) requires contiguous tensor");
    }
    if (i >= static_cast<size_t>(shape_[0])) {
        throw std::out_of_range("Tensor 1D index out of range");
    }
    return storage_->data()[i];
}

float& Tensor::at(size_t i, size_t j) {
    if (ndim() != 2) {
        throw std::invalid_argument("at(size_t, size_t) requires 2D tensor");
    }
    // offset 公式假定 row-major 连续布局；非连续张量（如 transpose 视图）
    // 直接拒绝，防止静默读错数据（安全审计 2026-08-16 T-4）
    if (!is_contiguous()) {
        throw std::runtime_error("at(i, j) requires contiguous tensor (call contiguous() first)");
    }
    if (i >= static_cast<size_t>(shape_[0]) || j >= static_cast<size_t>(shape_[1])) {
        throw std::out_of_range("Tensor 2D index out of range");
    }
    // contiguous 2D：offset = i * shape[1] + j
    return storage_->data()[i * static_cast<size_t>(shape_[1]) + j];
}

float Tensor::at(size_t i, size_t j) const {
    if (ndim() != 2) {
        throw std::invalid_argument("at(size_t, size_t) requires 2D tensor");
    }
    if (!is_contiguous()) {
        throw std::runtime_error("at(i, j) requires contiguous tensor (call contiguous() first)");
    }
    if (i >= static_cast<size_t>(shape_[0]) || j >= static_cast<size_t>(shape_[1])) {
        throw std::out_of_range("Tensor 2D index out of range");
    }
    return storage_->data()[i * static_cast<size_t>(shape_[1]) + j];
}

// --- 基础操作 ---

Tensor Tensor::reshape(const std::vector<int64_t>& new_shape) const {
    // 计算 new_shape 的元素总数（支持 -1 自动推断，与 PyTorch 一致）
    int64_t new_numel = 1;
    int64_t neg_idx = -1;
    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            if (neg_idx != -1) {
                throw std::invalid_argument("reshape: only one dimension can be -1");
            }
            neg_idx = static_cast<int64_t>(i);
        } else {
            if (new_shape[i] < 0) {
                throw std::invalid_argument("reshape: dimension size must be non-negative");
            }
            new_numel *= new_shape[i];
        }
    }

    int64_t cur_numel = static_cast<int64_t>(numel());
    std::vector<int64_t> resolved_shape = new_shape;
    if (neg_idx != -1) {
        if (new_numel == 0) {
            throw std::invalid_argument("reshape: cannot infer dimension with zero numel");
        }
        if (cur_numel % new_numel != 0) {
            throw std::invalid_argument("reshape: total numel not divisible by inferred dimension");
        }
        resolved_shape[neg_idx] = cur_numel / new_numel;
    } else {
        if (new_numel != cur_numel) {
            throw std::invalid_argument("reshape: numel mismatch");
        }
    }

    if (!is_contiguous()) {
        throw std::runtime_error("reshape: tensor must be contiguous (call contiguous() first)");
    }

    // 共享 storage，只改 shape 和 stride
    Tensor result;
    result.storage_ = storage_;  // 浅拷贝，共享内存
    result.shape_ = resolved_shape;
    result.compute_stride();
    return result;
}

bool Tensor::is_contiguous() const {
    // 检查 stride 是否等于 row-major contiguous stride
    if (shape_.empty()) {
        return true;
    }
    int64_t expected = 1;
    for (size_t i = shape_.size(); i-- > 0;) {
        if (stride_[i] != expected) {
            return false;
        }
        expected *= shape_[i];
    }
    return true;
}

Tensor Tensor::contiguous() const {
    if (is_contiguous()) {
        // 已经是 contiguous：返回浅拷贝（共享 storage）
        Tensor result;
        result.storage_ = storage_;
        result.shape_ = shape_;
        result.stride_ = stride_;
        return result;
    }
    // 非 contiguous：深拷贝为连续内存
    // 简化实现：当前所有构造路径都产生 contiguous Tensor，此分支预留。
    // 防御：增量 offset 回卷算法假定 stride 非负——当前无产生负 stride 的
    // 路径（transpose/permute 只交换），未来若引入 as_strided 需先支持负步长
    // （安全审计 2026-08-16 T-2）
    for (int64_t s : stride_) {
        if (s < 0) {
            throw std::runtime_error("contiguous(): negative stride not supported");
        }
    }
    Tensor result(shape_);
    // 通用拷贝：按 stride 索引逐元素复制（增量 offset 更新，O(n) 代替 O(n*ndim)）
    size_t n = numel();
    if (n == 0) {
        return result;
    }
    size_t nd = shape_.size();
    std::vector<int64_t> idx(nd, 0);
    int64_t offset = 0;  // 初始 idx 全为 0 → offset = 0
    for (size_t linear = 0; linear < n; ++linear) {
        result.data()[linear] = storage_->data()[offset];
        // 行优先递增 idx，增量更新 offset
        for (size_t d = nd; d-- > 0;) {
            if (++idx[d] < shape_[d]) {
                offset += stride_[d];
                break;
            }
            idx[d] = 0;
            offset -= (shape_[d] - 1) * stride_[d];
        }
    }
    return result;
}

// --- 视图操作（用 std::mdspan layout_stride 实现，共享 storage）---
//
// 这些方法用 std::mdspan 的 layout_stride 策略实现非 contiguous 视图。
// mdspan 是内部实现细节，不暴露到公开 API。
// 目标：减少未来 GPU 移植时 Tensor 数据视图层的重写量。
// CUDA 12.x 的 CCCL 提供 cuda::std::mdspan，接口兼容，可直接复用。

Tensor Tensor::view(const std::vector<int64_t>& new_shape) const {
    if (!is_contiguous()) {
        throw std::runtime_error("view: tensor must be contiguous (call contiguous() first)");
    }
    // 验证元素数，支持 -1 自动推断
    int64_t new_numel = 1;
    int64_t neg_idx = -1;
    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (new_shape[i] == -1) {
            if (neg_idx != -1) {
                throw std::invalid_argument("view: only one dimension can be -1");
            }
            neg_idx = static_cast<int64_t>(i);
        } else {
            if (new_shape[i] < 0) {
                throw std::invalid_argument("view: dimension size must be non-negative");
            }
            new_numel *= new_shape[i];
        }
    }
    int64_t cur_numel = static_cast<int64_t>(numel());
    std::vector<int64_t> resolved_shape = new_shape;
    if (neg_idx != -1) {
        if (new_numel == 0) {
            throw std::invalid_argument("view: cannot infer dimension with zero numel");
        }
        if (cur_numel % new_numel != 0) {
            throw std::invalid_argument("view: total numel not divisible by inferred dimension");
        }
        resolved_shape[neg_idx] = cur_numel / new_numel;
    } else {
        if (new_numel != cur_numel) {
            throw std::invalid_argument("view: numel mismatch");
        }
    }
    // 共享 storage，用 mdspan 的 layout_right 语义计算 stride
    Tensor result;
    result.storage_ = storage_;
    result.shape_ = resolved_shape;
    result.compute_stride();
    return result;
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    if (dim0 < 0 || dim0 >= static_cast<int64_t>(ndim()) ||
        dim1 < 0 || dim1 >= static_cast<int64_t>(ndim())) {
        throw std::out_of_range("transpose: dimension out of range");
    }
    // 用 mdspan layout_stride：交换 shape 和 stride
    // 零拷贝，共享 storage
    Tensor result;
    result.storage_ = storage_;
    result.shape_ = shape_;
    result.stride_ = stride_;
    std::swap(result.shape_[dim0], result.shape_[dim1]);
    std::swap(result.stride_[dim0], result.stride_[dim1]);
    return result;
}

Tensor Tensor::permute(const std::vector<int64_t>& dims) const {
    if (dims.size() != ndim()) {
        throw std::invalid_argument("permute: number of dims must match tensor ndim");
    }
    // 验证 dims 是有效排列
    std::vector<bool> seen(ndim(), false);
    for (size_t i = 0; i < dims.size(); ++i) {
        if (dims[i] < 0 || dims[i] >= static_cast<int64_t>(ndim())) {
            throw std::out_of_range("permute: dimension out of range");
        }
        if (seen[dims[i]]) {
            throw std::invalid_argument("permute: duplicate dimension");
        }
        seen[dims[i]] = true;
    }
    // 用 mdspan layout_stride：按 permute 顺序重排 shape 和 stride
    // 零拷贝，共享 storage
    Tensor result;
    result.storage_ = storage_;
    result.shape_.resize(ndim());
    result.stride_.resize(ndim());
    for (size_t i = 0; i < dims.size(); ++i) {
        result.shape_[i] = shape_[dims[i]];
        result.stride_[i] = stride_[dims[i]];
    }
    return result;
}

Tensor Tensor::squeeze(int64_t dim) const {
    if (dim < 0 || dim >= static_cast<int64_t>(ndim())) {
        throw std::out_of_range("squeeze: dimension out of range");
    }
    Tensor result;
    result.storage_ = storage_;
    if (shape_[dim] != 1) {
        // 大小不为 1，不改变形状，返回浅拷贝
        result.shape_ = shape_;
        result.stride_ = stride_;
        return result;
    }
    // 去除大小为 1 的维度
    result.shape_.reserve(ndim() - 1);
    result.stride_.reserve(ndim() - 1);
    for (size_t i = 0; i < ndim(); ++i) {
        if (static_cast<int64_t>(i) != dim) {
            result.shape_.push_back(shape_[i]);
            result.stride_.push_back(stride_[i]);
        }
    }
    return result;
}

Tensor Tensor::unsqueeze(int64_t dim) const {
    if (dim < 0 || dim > static_cast<int64_t>(ndim())) {
        throw std::out_of_range("unsqueeze: dimension out of range");
    }
    Tensor result;
    result.storage_ = storage_;
    result.shape_.reserve(ndim() + 1);
    result.stride_.reserve(ndim() + 1);
    for (size_t i = 0; i < ndim(); ++i) {
        if (static_cast<int64_t>(i) == dim) {
            result.shape_.push_back(1);
            // 插入大小为 1 的维度，stride 设为相邻维度的 stride（值不重要，因为 size=1）
            result.stride_.push_back(stride_[i]);
        }
        result.shape_.push_back(shape_[i]);
        result.stride_.push_back(stride_[i]);
    }
    if (dim == static_cast<int64_t>(ndim())) {
        result.shape_.push_back(1);
        result.stride_.push_back(1);  // 末尾追加 size=1 维度
    }
    return result;
}

Tensor Tensor::expand(const std::vector<int64_t>& new_shape) const {
    if (new_shape.size() < ndim()) {
        throw std::invalid_argument("expand: new_shape must have at least as many dims as current");
    }
    // 用 mdspan layout_stride：stride=0 表示广播
    Tensor result;
    result.storage_ = storage_;
    result.shape_ = new_shape;
    result.stride_.resize(new_shape.size());
    int64_t offset = static_cast<int64_t>(new_shape.size() - ndim());
    for (size_t i = 0; i < new_shape.size(); ++i) {
        if (static_cast<int64_t>(i) < offset) {
            // 新增的前导维度，stride=0 表示广播
            result.stride_[i] = 0;
        } else {
            size_t old_idx = i - offset;
            if (shape_[old_idx] == 1 && new_shape[i] > 1) {
                // 从 size=1 扩展到 >1，stride=0
                result.stride_[i] = 0;
            } else if (shape_[old_idx] == 1 && new_shape[i] == 0) {
                // 允许 size=1 维度扩展到 0（PyTorch 语义，numel=0 张量）
                // ——安全审计 2026-08-16 T-6
                result.stride_[i] = 0;
            } else if (shape_[old_idx] == new_shape[i]) {
                result.stride_[i] = stride_[old_idx];
            } else {
                throw std::invalid_argument("expand: dimension mismatch (shape must match or be 1)");
            }
        }
    }
    return result;
}

Tensor Tensor::clone() const {
    // 深拷贝：新 storage + 新 id（与拷贝构造的"共享 storage + 同 id"区分，
    // 安全审计 2026-08-16 T-7）。clone 结果是独立叶子，不记录到 tape。
    Tensor result;
    if (is_contiguous()) {
        result = Tensor(shape_);  // 新分配（零初始化）
        if (numel() > 0) {
            std::memcpy(result.data(), storage_->data(), numel() * sizeof(float));
        }
    } else {
        result = contiguous();  // 非 continuous 分支即深拷贝（新 storage、新 id）
    }
    result.set_requires_grad(requires_grad_);
    return result;
}

}  // namespace sgn_autograd
