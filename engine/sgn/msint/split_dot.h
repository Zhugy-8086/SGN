// split_dot.h - MSInt 前向多精度拆分点积 C++ 实现
//
// 对标范式文档 docs/msint_multisplit_paradigm/MSint多精度拆分计算范式.md §2
//   - 一次数据搬运，多精度解释：把 int 值按 split_bits 拆成 n 个部分，
//     各自参与独立粒度的点积，产出 2n-1 层不同粒度的子点积（多输出），
//     或融合为一个等价原始点积的结果（单融合）。
//
// 数学依据（位拆分严格可逆）：
//   value = Σ_{k=0}^{n-1} part_k * 2^(k*split_bits)
//   其中低位 part_0..part_{n-2} 为无符号（0..2^p-1），最高位 part_{n-1} 为符号位扩展。
//   该分解对 int 值 bit-exact（含负数，如 -1 → low=0xFFFF, high=-1）。
//
//   对点积：
//     y = Σ_i w_i * x_i
//       = Σ_{a,b} (Σ_i part_w[a][i]*part_x[b][i]) * 2^((a+b)*split_bits)
//       = Σ_{m=0}^{2n-2} partial[m] * 2^(m*split_bits)
//   其中 partial[m] = Σ_{a+b=m} Σ_i part_w[a][i]*part_x[b][i]。
//
//   多输出模式：直接产出 partial[m]（m=0..2n-2），即 coarse/cross/fine 等多尺度表示。
//   单融合模式：fused = Σ_m partial[m]*2^(m*split_bits)，数值等价于原始点积。
//
// 注意：本文件仅包含拆分+点积的纯计算逻辑，不依赖平台向量指令；
//       所有累加使用 int64，避免拆分后子点积在 K 较大时 int32 溢出。
#pragma once

#include <cstdint>
#include <vector>
#include <utility>

namespace sgn {

class SplitDot {
public:
    // 把一个 int 值按 split_bits 拆成 n 部分（n = total_bits / split_bits）。
    // 返回 vector 长度为 n，低位在前（part_0 为最低 split_bits 位）。
    //   低位 part_0..part_{n-2}：无符号，0..2^split_bits-1
    //   最高位 part_{n-1}：符号位扩展（算术右移），可为负
    // 该分解保证 value = Σ_k part_k * 2^(k*split_bits) 对任何 int64 输入 bit-exact。
    // 要求 total_bits > 0, split_bits > 0, total_bits 能被 split_bits 整除。
    static std::vector<int64_t> split_parts(int64_t value, int total_bits, int split_bits);

    // 多输出模式：前向多精度拆分点积。
    // w、x 等长（长度为 K），各自按 (total_bits, split_bits) 拆成 n 部分，
    // 返回长度为 2n-1 的 partial[m]（m=0..2n-2）。
    //   partial[m] = Σ_{a+b=m} Σ_i part_w[a][i] * part_x[b][i]
    // 对 n=2（如 int32 拆 int16）：
    //   partial[0] = fine   = Σ w_l * x_l
    //   partial[1] = cross  = Σ (w_h*x_l + w_l*x_h)
    //   partial[2] = coarse = Σ w_h * x_h
    // 各 partial 均为 int64 累加（含符号）。
    // trim_high_diag=true 时跳过高位对角（m=a+c>=n）——该项对「截断到
    // total_bits=n*split_bits 位」的结果无贡献（是 2^total_bits 的倍数），可安全裁剪
    // （4 位档 n=8 节省 44% ALU）。裁剪后 partials[m]（m>=n）保持 0，仅供低位截断消费者使用；
    // 多输出/多尺度消费者应保持默认 false（完整 partials）。
    static std::vector<int64_t> dot_split(
        const std::vector<int64_t>& w,
        const std::vector<int64_t>& x,
        int total_bits, int split_bits,
        bool trim_high_diag = false);

    // 单融合模式：fused = Σ_m partial[m] * 2^(m*split_bits)。
    // 内部先算 dot_split，再融合为 int64（结果超出 int64 时截断低位）。
    // 数值上等价于原始点积（在结果范围内 bit-exact）。
    static int64_t dot_fused(
        const std::vector<int64_t>& w,
        const std::vector<int64_t>& x,
        int total_bits, int split_bits);

    // 单融合 int32 截断模式：只保留 m<n（移位 < 32）的 partials 融合并截断为 int32。
    // 高位对角（m>=n）是 2^32 的倍数，对低 32 位无贡献 → 内部走裁剪内核
    // （trim_high_diag=true），4 位档（n=8）可省 44% ALU；
    // 结果与 dot_split 全融合后再 int32 截断 bit-exact 一致。
    // 要求 32 能被 split_bits 整除。
    static int32_t dot_fused_i32(
        const std::vector<int64_t>& w,
        const std::vector<int64_t>& x,
        int split_bits = 4);

    // 精确融合（128 位）：把 partial[m] 融合为 128 位有符号结果 (hi, lo)，
    // 避免 int64 溢出。lo 为低 64 位无符号，hi 为高 64 位（含符号）。
    static std::pair<int64_t, uint64_t> fuse_128(
        const std::vector<int64_t>& partials, int split_bits);
};

// 把 int64 值拆到调用方提供的固定缓冲（避免逐元素分配 vector）。
// 要求 n = total_bits / split_bits，out 容量 >= n；语义与 SplitDot::split_parts 一致：
// 低位 part_0..part_{n-2} 无符号（0..2^split_bits-1），最高位 part_{n-1} 符号位扩展。
// 供 dot_split 窄路径与 LeveledSplitDot 分组直写复用：配合 pack_narrow_value，
// 在拆分阶段直接把部分值写入预分配的窄缓冲（分组阶段去堆分配）。
void split_parts_fixed(int64_t value, int total_bits, int split_bits, int n, int64_t* out);

// 窄精度多输出点积（方案 A 三档窄路径 + 标量回退）。
// 供 SplitDot::dot_split 与 LeveledSplitDot 组内点积共享（leveled 侧先转置为部分主序）。
// parts_w[a] / parts_x[c]：n 个部分，各为长度 group_size 的 int64 数组
// （部分主序：parts_w[a][i] = 第 a 个部分在第 i 个元素上的值）。
// 返回 2n-1 个 partial[m] = Σ_{a+c=m} Σ_i parts_w[a][i]*parts_x[c][i]。
// split_bits=16 走 simd::dot16（avx2/avx512 后端），8/4 走 simd::dot8 / dot4 预解包
// （仅 avxvnni/avx512vnni 后端）；窄/宽门控由 is_narrow_simd() 运行时判定
// （后台 = simd_dispatch 的 CPUID，非编译期宏），其余 或标量后端自动落标量三重循环；
// 窄路径经偏置法修正，与标量逐元素 bit-exact 一致。
// trim_high_diag=true 时跳过高位对角（m>=n），语义同 SplitDot::dot_split（默认 false）。
std::vector<int64_t> narrow_group_dot(
    int split_bits,
    const std::vector<std::vector<int64_t>>& parts_w,
    const std::vector<std::vector<int64_t>>& parts_x,
    bool trim_high_diag = false);

// ============================================================
// 窄精度打包数据 + 直写接口（方案 A split_parts 直写窄数组）
// ============================================================
// NarrowParts：窄精度拆分点积的打包数据（部分主序，n 个部分 × K 元素），
// 由 alloc_narrow_parts 分配、pack_narrow_value 填充，供 narrow_dot 做 n² 点积。
// 相比 int64 部分数组，拆分阶段直接写入窄缓冲，省去 int64 中间存储与
// int64→窄二次转换（内存占用降到 1/2 / 1/4 / 1/8）。
struct NarrowParts {
    int split_bits = 0;      // 16 / 8 / 4（窄路径）
    size_t n = 0;            // 部分数 = total_bits / split_bits
    size_t K = 0;            // 元素数（group_size）
    std::vector<std::vector<int16_t>> s16;  // split_bits==16：有符号（含偏置修正）
    std::vector<std::vector<uint8_t>> su8;  // split_bits==8：无符号部分
    std::vector<std::vector<int8_t>>  ss8;  // split_bits==8：有符号部分
    std::vector<std::vector<uint8_t>> su4;  // split_bits==4：无符号 nibble 打包（长度 (K+1)/2）
    std::vector<std::vector<uint8_t>> ss4;  // split_bits==4：有符号 nibble 打包
    std::vector<int64_t> Sw, Sx;            // 偏置修正和（拆包阶段顺带累加）
};

// 分配空窄缓冲（n 部分 × K 元素，部分主序；nibble 数组按 (K+1)/2 零初始化）。
NarrowParts alloc_narrow_parts(int split_bits, size_t n, size_t K);

// 把第 a 个部分在第 i 个元素上的部分值（int64，见 SplitDot::split_parts 语义）
// 直接写入窄缓冲并累加 Sw/Sx。is_low（a+1<n 的无符号低位）内部判定并做偏置修正。
// 一次遍历完成拆分+打包，省去 int64→窄二次转换。
void pack_narrow_value(NarrowParts& dst, size_t a, size_t i, int64_t value);

// 从 int64 部分（部分主序 parts_w[a][i]）整体打包为窄表示。
NarrowParts pack_narrow(int split_bits, const std::vector<std::vector<int64_t>>& parts_w);

// 从 int64 部分（元素主序 parts_em[i][a]，Leveled 分组产物）整体打包为窄表示，
// 跳过 int64 部分主序转置。
NarrowParts pack_narrow_element_major(int split_bits,
                                      const std::vector<std::vector<int64_t>>& parts_em);

// 对两组窄打包做 n² 点积 → 2n-1 个 partial[m]（偏置法修正后与标量 bit-exact 一致）。
// trim_high_diag=true 时跳过高位对角（m=a+c>=n）——该项是 2^total_bits 的倍数，
// 对低位截断结果无贡献，可安全裁剪（4 位档 n=8 省 44% ALU）；裁剪后 partials[m>=n] 保持 0。
std::vector<int64_t> narrow_dot(const NarrowParts& w, const NarrowParts& x,
                                bool trim_high_diag = false);

// ============================================================
// 4 位摊销接口 — prepare_nibble + narrow_dot_prepared4
// ============================================================
// M 输出摊销场景（H3：同一激活 x 复用于 M 个输出行）下，narrow_dot 每次调用
// 都把 x 的 nibble 重新解包一次（M 次重复解包）。本接口把「预解包」提升为可复用的
// 一等操作——权重侧离线 prepare_nibble 一次、激活侧每前向 prepare_nibble 一次，
// 热路径 M 个输出全部复用同一预解包结果，n² 点积走纯 dpbusd（无解包指令）。
// 与 narrow_dot 内部 4 位预解包 bit-exact 一致（复用同一 unpack_nibble_u/s 语义）。
struct NibblePrepared {
    int n = 0;                               // 部分数 = total_bits / split_bits
    size_t K = 0;                            // 元素数（group_size）
    std::vector<std::vector<uint8_t>> u8;    // 无符号 nibble → 字节（w 侧预解包结果）
    std::vector<std::vector<int8_t>> s8;     // 有符号 nibble → 字节（x 侧预解包结果）
    std::vector<int64_t> Sw, Sx;             // 偏置修正和（与 NarrowParts 一致）
};

// 4 位 nibble 预解包（权重离线一次 / 激活每前向一次），返回字节数组缓存。
NibblePrepared prepare_nibble(const NarrowParts& p);

// 用预解包结果做 4 位 n² 点积（热路径 M×n² 次调用，无解包开销）。
// trim_high_diag 语义同 narrow_dot：true 时跳过高位对角（m>=n），partials[m>=n] 保持 0。
std::vector<int64_t> narrow_dot_prepared4(const NibblePrepared& w, const NibblePrepared& x,
                                          bool trim_high_diag = false);

// 从原始 int64 数组直接构建 4 位预解包缓存（权重离线 / 激活每前向一次）。
// 与按元素逐次 split_parts 的版本语义 bit-exact 一致，但用 split_parts_fixed 栈缓冲
// + alloc_narrow_parts 连续预分配 + pack_narrow_value 直写，消除 K 次 split_parts
// 的逐元素 vector 堆分配（K=16384 时 prepare 主导端到端耗时，是该路径最大优化点，
// 与 leveled 分组阶段堆分配问题同型）。
NibblePrepared prepare_nibble_from_raw(const std::vector<int64_t>& w, int total_bits = 32);

// 批量 M 输出摊销：同一预解包激活 x 复用于 M 个预解包权重行。
// 每行产出 2n-1 个 partial[m]（语义与 narrow_dot_prepared4 逐行一致）。
// M 循环在 C++ 内部执行（无 Python 逐行调用/绑定开销），且 x 侧 u8/s8 字节数组
// 跨行驻留缓存复用。trim_high_diag 语义同 narrow_dot_prepared4。
std::vector<std::vector<int64_t>> matmul_prepared4(
    const std::vector<NibblePrepared>& W,
    const NibblePrepared& x,
    bool trim_high_diag = false);

// split_bits 是否走窄 SIMD 直写路径（16→simd::dot16，8/4→simd::dot8/dot4；
// 由 simd::active_backend_name() 运行时判定，标量后端返回 false）。
bool is_narrow_simd(int split_bits);

} // namespace sgn
