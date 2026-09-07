// leveled_split_dot.h - MSInt 逐元素异构粒度拆分点积（组合落地）
//
// 对标范式文档 docs/msint_multisplit_paradigm/MSint多精度拆分计算范式.md
//   §1.2/H4（1:N 多精度解释） + §3.2/H4（Level 逐元素精度选择）的组合：
//   - 对一批 w/x 元素，每个元素按"重要性"独立选择拆分粒度（PrecisionSelector 语义）
//   - 同一批数据因此被分流到多个精度组（细粒度 / 中粒度 / 粗粒度）
//   - 各组内部用 SplitDot 做标准多精度拆分点积（多输出 partial[m]），
//     体现"一次搬运，多精度解释"（1:N）
//   - 单融合时逐组 128 位精确融合再累加，得到 y ≡ Σ_i w_i·x_i（bit-exact 可逆）
//
// 数学依据（异构粒度下位拆分仍严格可逆）：
//   元素 i 选粒度 b_i，则
//     w_i = Σ_k w_i[k]·2^(k·b_i)，x_i = Σ_l x_i[l]·2^(l·b_i)
//     w_i·x_i = Σ_{k,l} w_i[k]·x_i[l]·2^((k+l)·b_i)   （128 位内精确）
//   所有元素贡献直接相加（整数加法），故
//     y = Σ_i w_i·x_i  位精确等价于原始点积（不随粒度选择而改变）。
//   → 这同时验证范式假设 H1（数值等价）在异构逐元素粒度下仍成立。
//
// 注意：本文件仅含纯标量计算，不依赖平台向量指令；
//       128 位融合复用 SplitDot::fuse_128（含非 __int128 回退路径）。
#pragma once

#include <cstdint>
#include <map>
#include <vector>

#include "split_dot.h"

namespace sgn {

class LeveledSplitDot {
public:
    // 按重要性为每个元素选择拆分粒度（options 从粗到细，thresholds 升序，
    // 长度 = options.size()-1；语义与 PrecisionSelector::select 一致）。
    // 返回长度 = importance.size()，第 i 个元素所选 split_bits。
    static std::vector<int> select_levels(
        const std::vector<int64_t>& importance,
        int total_bits,
        const std::vector<int>& options,
        const std::vector<int64_t>& thresholds);

    // 异构粒度多输出拆分点积：w/x/importance 等长 K。
    // 每个元素 i 按 select_levels 选粒度 b_i，拆分 w_i/x_i 到 b_i，
    // 再按粒度分组，组内做标准 dot_split 多输出。
    // 返回 {b: partials_b}，partials_b 长度 = 2*(total_bits/b)-1。
    //   b=16 → [fine, cross, coarse]；b=8 → 7 个 partial；b=4 → 15 个 partial。
    // 体现"一次搬运多精度解释"：同批数据按重要性分流到不同精度组，各组独立 1:N。
    // trim_high_diag=True 时，各组透传裁剪到组内 n² 点积（跳过高位对角 m>=n，
    // 对 int32 截断目标 bit-exact，4 位档省 44% ALU；默认 false 保持全量兼容）。
    static std::map<int, std::vector<int64_t>> dot_split_leveled(
        const std::vector<int64_t>& w, const std::vector<int64_t>& x,
        const std::vector<int64_t>& importance,
        int total_bits,
        const std::vector<int>& options,
        const std::vector<int64_t>& thresholds,
        bool trim_high_diag = false);

    // 异构粒度单融合点积：fused = Σ_b fuse_128(partials_b, b)，128 位精确累加，
    // 返回低 64 位（有符号 int64 位模式）。
    // 数值等价于原始点积 Σ_i w_i·x_i（bit-exact，H1 在异构下成立）。
    static int64_t dot_fused_leveled(
        const std::vector<int64_t>& w, const std::vector<int64_t>& x,
        const std::vector<int64_t>& importance,
        int total_bits,
        const std::vector<int>& options,
        const std::vector<int64_t>& thresholds);

    // 降档决策：按重要性选择「精度位数」p ∈ options（Level 精度调度语义，
    // 替代 select_levels 的「拆分粒度」——低重要度 Level 降档到低精度）。
    // options 为精度位数列表，从「低精度（最少位数）→ 高精度（最多位数）」
    // 排列（如 {8, 16, 32}），thresholds 升序（长度 = options.size()-1）。
    // 映射与 select_levels 同向：options[0] 给最低重要度桶——重要性越低，
    // 精度位数越少（降档越多），高重要度 → 全精度。
    // 注意与粒度选择器的语义差异：select_levels 的 options[0] 是「粗粒度」
    // （少部分数），而这里的 options[0] 是「低精度」（少位数）。
    static std::vector<int> select_precision(
        const std::vector<int64_t>& importance,
        const std::vector<int>& options,
        const std::vector<int64_t>& thresholds);

    // 降档摊销多输出点积：按 select_precision 分组，每组「保留高 p 位」
    // （keep_top(v,p) = (v>>(32-p))<<(32-p)，int8/int16 量化存储形态）后右移成
    // p-bit 值 v'=v>>(32-p)，走 4 位摊销路径（prepare_nibble_from_raw(p) +
    // narrow_dot_prepared4）。返回 {p: partials_p}，partials_p 长度 2n-1（n=p/4），
    // 为 **p-bit 原始 partials（未缩放）**——组内 fused（fuse_128, split_bits=4）
    // == Σ_i∈组 (w_i>>(32-p))·(x_i>>(32-p))；恢复量纲需左移 2(32-p)（见
    // dot_fused_leveled_downcast）。w/x/importance 等长 K。
    static std::map<int, std::vector<int64_t>> dot_split_leveled_downcast(
        const std::vector<int64_t>& w, const std::vector<int64_t>& x,
        const std::vector<int64_t>& importance,
        int total_bits,
        const std::vector<int>& options,
        const std::vector<int64_t>& thresholds);

    // 降档摊销单融合点积：跨精度组把 partials 融合（fuse_128, split_bits=4）后
    // 左移 2(32-p) 恢复量纲，再 128 位累加，返回低 64 位（有符号）。
    // 数值上 ≈ Σ_i w_i·x_i，截断误差由「量化到 p 位的粒度」决定 ≈ 2^(1-p)
    // （p 为各元素降档精度；p=32 组精确无误差）。
    static int64_t dot_fused_leveled_downcast(
        const std::vector<int64_t>& w, const std::vector<int64_t>& x,
        const std::vector<int64_t>& importance,
        int total_bits,
        const std::vector<int>& options,
        const std::vector<int64_t>& thresholds);

    // 降档预解包融合（摊销 prepare 阶段 Python 层开销，见
    // prepare_x_python_overhead_fusion_plan_2026_08_14.md）：
    // 一次传入原始 w 与 importance，C++ 内部分组（select_precision）→ keep_top
    // 截断（右移成 p-bit 值）→ prepare_nibble_from_raw(p)，返回 {p: NibblePrepared}。
    // 消除 Python 层三笔开销（group_elems 分组循环 + trunc 列表推导 + 每组重复
    // pybind list→vector 转换）——prepare 的 Python 观测值 2.84ms 中 C++ 内核仅
    // ~0.18ms（6%），其余 ~94% 在 Python 层。
    // 权重侧每行离线调用一次、激活侧每前向调用一次；返回的 NibblePrepared 组与
    // dot_split_leveled_downcast 的分组语义一致（n=p/4 组合）。
    // 要求 w/importance 等长 K，p ∈ options 且整除 total_bits。
    static std::map<int, NibblePrepared> prepare_downcast(
        const std::vector<int64_t>& w, const std::vector<int64_t>& importance,
        int total_bits,
        const std::vector<int>& options,
        const std::vector<int64_t>& thresholds);

    // 裸指针版本（阶段 2：numpy 零拷贝绑定的底层接口）。
    // 语义与 prepare_downcast 完全一致，但直接读 w/importance 的连续内存指针，
    // 避免 pybind 从 Python list 逐元素转 std::vector<int64_t>（该转换本身
    // K=16384 时 ~0.26~0.57ms，是阶段 1 后剩余的 prepare 主头）。
    // 由 vector 版委托；绑定层用 py::array_t<int64_t>.unchecked().data() 传入。
    static std::map<int, NibblePrepared> prepare_downcast_ptr(
        const int64_t* w, const int64_t* importance, size_t K,
        int total_bits,
        const std::vector<int>& options,
        const std::vector<int64_t>& thresholds);

    // 便捷重载：默认 total_bits=32, options={16,8,4}, thresholds={100,1000}
    static std::vector<int> select_levels_default(
        const std::vector<int64_t>& importance, int total_bits = 32);

    static std::map<int, std::vector<int64_t>> dot_split_leveled_default(
        const std::vector<int64_t>& w, const std::vector<int64_t>& x,
        const std::vector<int64_t>& importance, int total_bits = 32);

    static int64_t dot_fused_leveled_default(
        const std::vector<int64_t>& w, const std::vector<int64_t>& x,
        const std::vector<int64_t>& importance, int total_bits = 32);

    // 降档便捷重载：默认 total_bits=32, options={8,16,32}, thresholds={100,1000}
    // （从低精度到高精度排列——低重要度降档到少位数，高重要度全精度）。
    static std::vector<int> select_precision_default(
        const std::vector<int64_t>& importance);

    static std::map<int, std::vector<int64_t>> dot_split_leveled_downcast_default(
        const std::vector<int64_t>& w, const std::vector<int64_t>& x,
        const std::vector<int64_t>& importance, int total_bits = 32);

    static int64_t dot_fused_leveled_downcast_default(
        const std::vector<int64_t>& w, const std::vector<int64_t>& x,
        const std::vector<int64_t>& importance, int total_bits = 32);
};

} // namespace sgn
