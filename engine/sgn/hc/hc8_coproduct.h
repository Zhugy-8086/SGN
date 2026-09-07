// hc8_coproduct.h - HC8 余积存储 C++ 实现
//
// 对标 exp40_three_component_integration.py 的 HC8 余积 6 字节布局
//
// 设计依据:
//   - exp38: HC8 余积存储概念 (6 字节中 4 字节可存储方差统计)
//   - exp39: HC8 卡尔曼滤波 (N=256 步后 Δb=6.43)
//   - exp40: 三组件整合协同效应 383 倍, 存储 61.4%
//
// 6 字节布局 (per-element 容器):
//   - Byte 0-1: 梯度 (int16, 可 bitsplit 为 V_8² 多视角读取)
//   - Byte 2-3: 方差统计 (int16, 滑动窗口内的梯度方差, 驱动 Level_b)
//   - Byte 4-5: 卡尔曼状态 (int16, Q15 定点, 累积 N 步后的状态)
//
// 数学依据:
//   - 梯度量化: scale = max(|g|)/32767, int16 补码
//   - 方差统计: 滑动窗口方差, int16 量化 (相对值)
//   - 卡尔曼 Q15: x̂_k = x̂_{k-1} + (K_k·(z_k - x̂_{k-1})) >> 15
//   - Δb 理论值: ½log₂(N) (N=256 → Δb=4)
//
// 复用关系:
//   - 与 PackedBackend 互补: PackedBackend 打包多槽位, HC8Coproduct 打包多状态
//   - 与 Level_b 协同: 方差统计驱动 Level_b 零延迟调度
//   - 卡尔曼: 时间换精度, 可选启用
#pragma once

#include <cstdint>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <utility>
#include <vector>
#include <string>

namespace sgn {

// ============================================================
// HC8Coproduct - 单元素 6 字节余积存储
// ============================================================
//
// 内存布局 (6 字节, 小端序):
//   [0][1] gradient  (int16, 补码)
//   [2][3] variance  (int16, 无符号, 表示方差相对值)
//   [4][5] kalman    (int16, 补码, Q15 定点状态)
//
// 设计说明:
//   - 6 字节对齐到 2 字节边界, 可用 3 次 int16 读写
//   - 与 exp40 的字节布局完全一致
//   - 不可复制 (含状态语义), 但可移动

class HC8Coproduct {
public:
    // 默认构造: 全零 (未初始化状态)
    HC8Coproduct() : bytes_{0, 0, 0, 0, 0, 0} {}

    // 从原始 6 字节构造 (深拷贝)
    explicit HC8Coproduct(const uint8_t raw[6]) {
        for (int i = 0; i < 6; ++i) bytes_[i] = raw[i];
    }

    // 从三个 int16 字段构造
    HC8Coproduct(int16_t gradient, uint16_t variance, int16_t kalman) {
        set_gradient(gradient);
        set_variance(variance);
        set_kalman(kalman);
    }

    // ============================================================
    // 梯度 (Byte 0-1)
    // ============================================================

    // 读取梯度 (signed int16, 补码)
    int16_t gradient() const {
        // 小端序: 低字节在前
        return static_cast<int16_t>(
            (static_cast<uint16_t>(bytes_[1]) << 8) | bytes_[0]
        );
    }

    // 写入梯度
    void set_gradient(int16_t value) {
        uint16_t u = static_cast<uint16_t>(value);
        bytes_[0] = static_cast<uint8_t>(u & 0xFF);
        bytes_[1] = static_cast<uint8_t>((u >> 8) & 0xFF);
    }

    // 梯度的 bitsplit 视角 (exp37/38):
    //   - V_8_high: 高字节 (前向 int8 路径)
    //   - V_8_low:  低字节 (反向 int16 concat)
    // 返回 (high, low) 元组, high 为 signed int8
    std::pair<int8_t, uint8_t> gradient_bitsplit() const {
        uint8_t low = bytes_[0];
        uint8_t high_u = bytes_[1];
        int8_t high_s = static_cast<int8_t>(high_u);
        return {high_s, low};
    }

    // 从 bitsplit 写入梯度
    void set_gradient_bitsplit(int8_t high, uint8_t low) {
        bytes_[0] = low;
        bytes_[1] = static_cast<uint8_t>(high);
    }

    // ============================================================
    // 方差统计 (Byte 2-3)
    // ============================================================

    // 读取方差统计 (unsigned int16, 相对值 0~32767)
    uint16_t variance() const {
        return static_cast<uint16_t>(
            (static_cast<uint16_t>(bytes_[3]) << 8) | bytes_[2]
        );
    }

    // 写入方差统计
    void set_variance(uint16_t value) {
        bytes_[2] = static_cast<uint8_t>(value & 0xFF);
        bytes_[3] = static_cast<uint8_t>((value >> 8) & 0xFF);
    }

    // 方差统计 → log(var) 反馈信号 (exp10-recheck power law)
    // 返回 log2(variance + 1), 范围 [0, 16]
    // 用途: 驱动 Level_b 调度 (方差大 → bits 增加)
    // 注: +1 避免 log2(0) 未定义, 且使 variance=0 → log2(1)=0
    float variance_log2() const {
        uint16_t v = variance();
        // log2(v + 1), v ∈ [0, 65535] → 结果 ∈ [0, 16]
        uint32_t v_plus_1 = static_cast<uint32_t>(v) + 1;
        // 整数 log2
        int int_log2 = 0;
        uint32_t tmp = v_plus_1;
        while (tmp > 1) { tmp >>= 1; int_log2++; }
        // 线性插值细化
        uint32_t base = static_cast<uint32_t>(1) << int_log2;
        float frac = static_cast<float>(v_plus_1 - base) / static_cast<float>(base);
        return static_cast<float>(int_log2) + frac;
    }

    // ============================================================
    // 卡尔曼状态 (Byte 4-5)
    // ============================================================

    // 读取卡尔曼状态 (signed int16, Q15 定点)
    int16_t kalman() const {
        return static_cast<int16_t>(
            (static_cast<uint16_t>(bytes_[5]) << 8) | bytes_[4]
        );
    }

    // 写入卡尔曼状态
    void set_kalman(int16_t value) {
        uint16_t u = static_cast<uint16_t>(value);
        bytes_[4] = static_cast<uint8_t>(u & 0xFF);
        bytes_[5] = static_cast<uint8_t>((u >> 8) & 0xFF);
    }

    // ============================================================
    // 原始字节访问
    // ============================================================

    const uint8_t* raw_bytes() const { return bytes_.data(); }
    uint8_t* mutable_raw_bytes() { return bytes_.data(); }
    static constexpr int byte_size() { return 6; }

    // 序列化 (16 进制字符串, 便于调试)
    std::string serialize() const {
        char buf[16];
        snprintf(buf, sizeof(buf), "%02x%02x%02x%02x%02x%02x",
                 bytes_[0], bytes_[1], bytes_[2],
                 bytes_[3], bytes_[4], bytes_[5]);
        return std::string(buf);
    }

    // 相等比较
    bool operator==(const HC8Coproduct& other) const {
        return bytes_ == other.bytes_;
    }
    bool operator!=(const HC8Coproduct& other) const {
        return !(*this == other);
    }

private:
    std::array<uint8_t, 6> bytes_;
};

// ============================================================
// HC8CoproductArray - 批量余积存储 (连续内存)
// ============================================================
//
// 设计:
//   - 所有元素的 6 字节连续存储 (SOA→AOS 布局)
//   - 支持批量梯度/方差/卡尔曼读写 (numpy 互操作)
//   - 内存: n_elements * 6 字节
//
// 对标 exp40 的 per-layer 容器:
//   - 梯度按元素存 (n_elements * 2 bytes)
//   - 方差 per-layer (1 * 2 bytes)  ← 注意: exp40 中方差是 per-layer
//   - 卡尔曼 per-element (n_elements * 2 bytes)
//
// 本实现提供两种模式:
//   - per_element_variance=True: 方差也按元素存 (精细, 6 bytes/elem)
//   - per_element_variance=False: 方差 per-layer (exp40 原始模式, 2 bytes/elem + 2 bytes/layer)

class HC8CoproductArray {
public:
    // 构造: n_elements 个元素, 全零初始化
    // per_element_variance=True: 6 bytes/elem (梯度+方差+卡尔曼)
    // per_element_variance=False: 4 bytes/elem + 2 bytes/layer (exp40 模式)
    explicit HC8CoproductArray(int n_elements, bool per_element_variance = true)
        : n_elements_(n_elements),
          per_element_variance_(per_element_variance) {
        if (per_element_variance_) {
            // 6 bytes/elem, 连续存储
            data_.resize(static_cast<size_t>(n_elements) * 6, 0);
        } else {
            // 4 bytes/elem (梯度+卡尔曼) + 2 bytes/layer (方差)
            data_.resize(static_cast<size_t>(n_elements) * 4 + 2, 0);
        }
    }

    int n_elements() const { return n_elements_; }
    bool per_element_variance() const { return per_element_variance_; }

    // 总字节数
    size_t total_bytes() const { return data_.size(); }

    // ============================================================
    // per_element_variance=True 模式 (6 bytes/elem)
    // ============================================================

    // 下标边界检查：单元素访问方法统一调用（安全审计 2026-08-16 M5-R——
    // 原实现 i<0 时 static_cast<size_t>(i) 变巨大值，data_[offset] 严重越界）
    void check_bounds(int i) const {
        if (i < 0 || static_cast<size_t>(i) >= static_cast<size_t>(n_elements_)) {
            throw std::out_of_range(
                "HC8CoproductArray: index " + std::to_string(i) +
                " out of range [0, " + std::to_string(n_elements_) + ")");
        }
    }

    // 获取第 i 个元素的完整 HC8Coproduct 引用
    HC8Coproduct at(int i) const {
        check_bounds(i);
        if (per_element_variance_) {
            return HC8Coproduct(&data_[static_cast<size_t>(i) * 6]);
        } else {
            // per-layer variance 模式: 组装 6 字节
            int16_t grad = get_gradient(i);
            uint16_t var = get_layer_variance();
            int16_t kal = get_kalman(i);
            return HC8Coproduct(grad, var, kal);
        }
    }

    // 设置第 i 个元素的完整 HC8Coproduct
    void set_at(int i, const HC8Coproduct& cop) {
        check_bounds(i);
        if (per_element_variance_) {
            const uint8_t* src = cop.raw_bytes();
            std::copy(src, src + 6, &data_[static_cast<size_t>(i) * 6]);
        } else {
            set_gradient(i, cop.gradient());
            set_kalman(i, cop.kalman());
            // variance 在 per-layer 模式下由 set_layer_variance 设置
        }
    }

    // ============================================================
    // 梯度批量读写
    // ============================================================

    int16_t get_gradient(int i) const {
        check_bounds(i);
        size_t offset = per_element_variance_
            ? static_cast<size_t>(i) * 6
            : static_cast<size_t>(i) * 4;
        return static_cast<int16_t>(
            (static_cast<uint16_t>(data_[offset + 1]) << 8) | data_[offset]
        );
    }

    void set_gradient(int i, int16_t value) {
        check_bounds(i);
        size_t offset = per_element_variance_
            ? static_cast<size_t>(i) * 6
            : static_cast<size_t>(i) * 4;
        uint16_t u = static_cast<uint16_t>(value);
        data_[offset] = static_cast<uint8_t>(u & 0xFF);
        data_[offset + 1] = static_cast<uint8_t>((u >> 8) & 0xFF);
    }

    // ============================================================
    // 方差批量读写
    // ============================================================

    // per_element_variance=True: 读取第 i 个元素的方差
    uint16_t get_variance(int i) const {
        check_bounds(i);
        if (!per_element_variance_) {
            return get_layer_variance();
        }
        size_t offset = static_cast<size_t>(i) * 6 + 2;
        return static_cast<uint16_t>(
            (static_cast<uint16_t>(data_[offset + 1]) << 8) | data_[offset]
        );
    }

    // per_element_variance=True: 设置第 i 个元素的方差
    void set_variance(int i, uint16_t value) {
        check_bounds(i);
        if (!per_element_variance_) {
            throw std::runtime_error(
                "per_element_variance=False 模式下请用 set_layer_variance");
        }
        size_t offset = static_cast<size_t>(i) * 6 + 2;
        data_[offset] = static_cast<uint8_t>(value & 0xFF);
        data_[offset + 1] = static_cast<uint8_t>((value >> 8) & 0xFF);
    }

    // per_element_variance=False: 读取 per-layer 方差
    uint16_t get_layer_variance() const {
        if (per_element_variance_) {
            throw std::runtime_error(
                "per_element_variance=True 模式下请用 get_variance(i)");
        }
        size_t offset = static_cast<size_t>(n_elements_) * 4;
        return static_cast<uint16_t>(
            (static_cast<uint16_t>(data_[offset + 1]) << 8) | data_[offset]
        );
    }

    // per_element_variance=False: 设置 per-layer 方差
    void set_layer_variance(uint16_t value) {
        if (per_element_variance_) {
            throw std::runtime_error(
                "per_element_variance=True 模式下请用 set_variance(i, value)");
        }
        size_t offset = static_cast<size_t>(n_elements_) * 4;
        data_[offset] = static_cast<uint8_t>(value & 0xFF);
        data_[offset + 1] = static_cast<uint8_t>((value >> 8) & 0xFF);
    }

    // ============================================================
    // 卡尔曼状态批量读写
    // ============================================================

    int16_t get_kalman(int i) const {
        check_bounds(i);
        size_t offset = per_element_variance_
            ? static_cast<size_t>(i) * 6 + 4
            : static_cast<size_t>(i) * 4 + 2;
        return static_cast<int16_t>(
            (static_cast<uint16_t>(data_[offset + 1]) << 8) | data_[offset]
        );
    }

    void set_kalman(int i, int16_t value) {
        check_bounds(i);
        size_t offset = per_element_variance_
            ? static_cast<size_t>(i) * 6 + 4
            : static_cast<size_t>(i) * 4 + 2;
        uint16_t u = static_cast<uint16_t>(value);
        data_[offset] = static_cast<uint8_t>(u & 0xFF);
        data_[offset + 1] = static_cast<uint8_t>((u >> 8) & 0xFF);
    }

    // ============================================================
    // 批量 numpy 互操作接口 (由 bindings.cpp 调用)
    // ============================================================

    // 批量读取梯度 → int16 数组 (调用方预分配)
    void batch_get_gradient(int16_t* out) const {
        for (int i = 0; i < n_elements_; ++i) {
            out[i] = get_gradient(i);
        }
    }

    // 批量写入梯度 ← int16 数组
    void batch_set_gradient(const int16_t* in) {
        for (int i = 0; i < n_elements_; ++i) {
            set_gradient(i, in[i]);
        }
    }

    // 批量读取卡尔曼 → int16 数组
    void batch_get_kalman(int16_t* out) const {
        for (int i = 0; i < n_elements_; ++i) {
            out[i] = get_kalman(i);
        }
    }

    // 批量写入卡尔曼 ← int16 数组
    void batch_set_kalman(const int16_t* in) {
        for (int i = 0; i < n_elements_; ++i) {
            set_kalman(i, in[i]);
        }
    }

    // 原始字节访问
    const uint8_t* raw_bytes() const { return data_.data(); }
    uint8_t* mutable_raw_bytes() { return data_.data(); }

private:
    int n_elements_;
    bool per_element_variance_;
    std::vector<uint8_t> data_;
};

// ============================================================
// HC8KalmanFilter - 整数卡尔曼滤波器 (Q15 定点)
// ============================================================
//
// 对标 exp39/exp40 的 integer_kalman_with_convergence
//
// 更新公式 (Q15 定点):
//   K_k = round(P * 2^15 / (P + R))
//   x̂_k = x̂_{k-1} + (K_k * (z_k - x̂_{k-1})) >> 15
//   P_k = P_{k-1} - (K_k * P_{k-1}) >> 15
//
// Δb 理论值: ½log₂(N) (N=256 → Δb=4)
// exp40 实测: Δb=3.903 (接近理论)

class HC8KalmanFilter {
public:
    // 构造: n_elements 个状态, 初始 P 和 R (Q15 定点)
    HC8KalmanFilter(int n_elements, int16_t P0_int, int16_t R_int)
        : n_elements_(n_elements),
          P_(n_elements, P0_int),
          R_(R_int) {}

    // 单步更新 (批量)
    // x_int16: 估计值 (in/out, int16)
    // z_int16: 观测值 (in, int16)
    // 返回更新后的估计值 (与 x_int16 相同指针)
    void update(int16_t* x_int16, const int16_t* z_int16) {
        const int32_t Q15 = 32768;
        for (int i = 0; i < n_elements_; ++i) {
            int32_t P = P_[i];
            int32_t P_plus_R = P + R_;
            int32_t P_plus_R_safe = (P_plus_R > 1) ? P_plus_R : 1;
            int32_t K = (P * Q15) / P_plus_R_safe;
            if (K < 0) K = 0;
            if (K > 32767) K = 32767;

            int32_t x = x_int16[i];
            int32_t z = z_int16[i];
            int32_t innov = z - x;
            int32_t x_new = x + (K * innov) / Q15;
            // clamp to int16
            if (x_new < -32768) x_new = -32768;
            if (x_new > 32767) x_new = 32767;
            x_int16[i] = static_cast<int16_t>(x_new);

            int32_t P_new = P - (K * P) / Q15;
            if (P_new < 0) P_new = 0;
            if (P_new > 32767) P_new = 32767;
            P_[i] = static_cast<int16_t>(P_new);
        }
    }

    // 获取当前 P (协方差估计)
    const std::vector<int16_t>& covariance() const { return P_; }

    // 计算 Δb (等效 bits 提升)
    // Δb = ½log₂(R_initial / P_current)
    // 需要 R_initial 和当前 P 的均值
    static float compute_delta_bits(int16_t R_int, int16_t P_current_mean) {
        if (P_current_mean <= 0 || R_int <= 0) return 0.0f;
        return 0.5f * std::log2(static_cast<float>(R_int) /
                                static_cast<float>(P_current_mean));
    }

    int n_elements() const { return n_elements_; }
    int16_t R() const { return R_; }

private:
    int n_elements_;
    std::vector<int16_t> P_;  // 协方差估计 (Q15)
    int16_t R_;                // 观测噪声 (Q15)
};

} // namespace sgn
