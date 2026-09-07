/**
 * @file hc8_net.h
 * @brief HC8 神经网络运算扩展（阶段 1.3 全整数路径）
 * @version 1.3.0
 *
 * 主程序 pysgn 缺失的批量/矩阵运算 API，在此独立扩展中实现：
 *   - HC8 对称量化/反量化（scale + offset 128）
 *   - HC8 整数矩阵乘（int8 × int8 → int32 累加 → 反量化 → 重新量化）
 *   - HC8 整数 ReLU（基于偏移表示）
 *   - HC8 矩阵扁平存储互转
 *
 * 设计原则：
 *   - 不修改 engine/hc/ 下任何现有文件
 *   - 仅 include 现有头文件（hc8.h），复用 hc8_t 结构
 *   - 独立编译为 pysgn_net 扩展模块
 *   - 通过 bytes 接口与 pysgn.HC8 互操作
 *
 * 依赖：
 *   - engine/hc/sgn/include/hc/hc8.h（hc8_t 定义）
 *   - engine/hc/sgn/include/hc/hc.h（overflow_t 等基础类型）
 *
 * 参考：
 *   - 阶段 1.3 纯 Python 实现：legacy/traditional/stage_1_3_int_path/hc_adapter.py + hc_matmul.py
 *   - 静态预测试报告：legacy/traditional/stage_1_3_int_path/stage_1_3_preflight_static_review.md
 */

#ifndef SGN_HC8_NET_H
#define SGN_HC8_NET_H

#include "hc/hc.h"
#include "hc/hc8.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 量化方案：对称线性量化 + 偏移 128
 * ============================================================================
 *
 * 流程：
 *   1. 对称量化：scale = max(|w|) / 127, q = round(w / scale) ∈ [-127, 127]
 *   2. 偏移到无符号：q_u = q + 128 ∈ [1, 255]（0 留给"未初始化"特判）
 *   3. HC8 只用 v[0] 层存 q_u，v[1..5] = 0
 *
 * 反量化：
 *   1. q_u = hc8.v[0]
 *   2. q = q_u - 128
 *   3. w = q * scale
 */

/**
 * 对称量化方案参数
 */
typedef struct {
    float    scale;       /**< 量化 scale（float → int 时的缩放因子） */
    int32_t  qmin;        /**< 对称量化下界（-127） */
    int32_t  qmax;        /**< 对称量化上界（127，避免 -128 不对称） */
    int32_t  offset;      /**< 偏移到无符号（128，使 q_u ∈ [1, 255]） */
} hc8_quant_schema_t;

/**
 * 从 float 数组推导量化 scale
 *   scale = max(|w[i]|) / 127，全零时返回 1.0
 *
 * @param w        float 输入数组
 * @param n        数组长度
 * @return         量化 scale
 */
float hc8_quant_compute_scale(const float* w, uint32_t n);

/**
 * 量化 float 数组到 HC8 数组
 *   每个元素：q = round(w[i] / scale), clamp(-127, 127), q_u = q + 128
 *   存入 out[i].v[0]，out[i].v[1..5] = 0
 *
 * @param w        float 输入数组
 * @param n        数组长度
 * @param scale    量化 scale（由 hc8_quant_compute_scale 推导）
 * @param schema   量化方案（提供 qmin/qmax/offset）
 * @param out      输出 HC8 数组（长度 n，需预分配）
 */
void hc8_quantize(const float* w, uint32_t n,
                  float scale,
                  const hc8_quant_schema_t* schema,
                  hc8_t* out);

/**
 * 反量化 HC8 数组到 float 数组
 *   每个元素：q_u = hc8.v[0], q = q_u - offset, w = q * scale
 *
 * @param h        HC8 输入数组
 * @param n        数组长度
 * @param scale    量化 scale
 * @param schema   量化方案（提供 offset）
 * @param out      输出 float 数组（长度 n，需预分配）
 */
void hc8_dequantize(const hc8_t* h, uint32_t n,
                    float scale,
                    const hc8_quant_schema_t* schema,
                    float* out);

/* ============================================================================
 * HC8 矩阵乘（整数路径）
 * ============================================================================
 *
 * C = A @ B，其中 A: m×k, B: k×n, C: m×n（行优先）
 *
 * 整数路径：
 *   1. 从 HC8 提取量化整数（去偏移）：a_q, b_q ∈ [-127, 127]
 *   2. 在 int32 空间累加：c_acc[i][j] = sum_l(a_q[i][l] * b_q[l][j])
 *   3. 反量化累加结果：c_float = c_acc * a_scale * b_scale
 *   4. 重新量化到 HC8：基于 c_float 的 max(|w|) 推导新 scale，量化到 HC8
 *
 * 注意：输出 scale 是基于 c_float 重新计算的，与输入 scale 不同。
 */

/**
 * HC8 矩阵乘
 *
 * @param a            输入矩阵 A（m×k，行优先，HC8 数组，长度 m*k）
 * @param b            输入矩阵 B（k×n，行优先，HC8 数组，长度 k*n）
 * @param m, k, n      矩阵维度
 * @param a_scale      A 的量化 scale
 * @param b_scale      B 的量化 scale
 * @param schema       量化方案（用于重新量化输出）
 * @param out          输出矩阵 C（m×n，HC8 数组，长度 m*n，需预分配）
 * @param out_scale    输出 scale（输出参数，由函数写入）
 */
void hc8_matmul(const hc8_t* a, const hc8_t* b,
                uint32_t m, uint32_t k, uint32_t n,
                float a_scale, float b_scale,
                const hc8_quant_schema_t* schema,
                hc8_t* out, float* out_scale);

/* ============================================================================
 * HC8 整数 ReLU
 * ============================================================================
 *
 * HC8 是无符号偏移表示，q_u = q + 128：
 *   - q < 0 → q_u < 128 → ReLU 后应为 0 → q_u = 128
 *   - q >= 0 → q_u >= 128 → ReLU 后不变
 *
 * 偏移 128 来自 schema->offset
 */

/**
 * HC8 整数 ReLU
 *
 * @param x            输入矩阵 X（m×n，行优先，HC8 数组）
 * @param m, n         矩阵维度
 * @param schema       量化方案（提供 offset）
 * @param out          输出矩阵（m×n，HC8 数组，需预分配）
 */
void hc8_relu(const hc8_t* x, uint32_t m, uint32_t n,
              const hc8_quant_schema_t* schema,
              hc8_t* out);

/* ============================================================================
 * HC8 矩阵扁平存储互转
 * ============================================================================
 *
 * HC8 矩阵以行优先扁平 hc8_t 数组存储。
 * 提供 bytes ↔ 矩阵 的互转，方便与 Python/PyTorch 交互。
 */

/**
 * HC8 数组 → bytes（每个 HC8 6 字节）
 *
 * @param h        HC8 数组
 * @param n        数组长度
 * @param out      输出 bytes 缓冲区（长度 6*n，需预分配）
 */
void hc8_array_to_bytes(const hc8_t* h, uint32_t n, uint8_t* out);

/**
 * bytes → HC8 数组
 *
 * @param bytes    输入 bytes 缓冲区（长度 6*n）
 * @param n        HC8 数组长度
 * @param out      输出 HC8 数组（长度 n，需预分配）
 */
void hc8_bytes_to_array(const uint8_t* bytes, uint32_t n, hc8_t* out);

/* ============================================================================
 * 默认量化方案
 * ============================================================================ */

/**
 * 获取默认量化方案（qmin=-127, qmax=127, offset=128）
 */
hc8_quant_schema_t hc8_quant_default_schema(void);

/* ============================================================================
 * UFP-1 残差量化：v[0..depth] 存量化残差，实现 48-bit 存储精度
 * ============================================================================
 *
 * 设计原则（见 hc_ufp_1_residual_design.md）：
 *   - 每层存上一层的量化残差，递归 depth 次
 *   - scale 是组级共享（整个矩阵每层一个 scale，存在外部）
 *   - depth=0 等价于 hc8_quantize（只用 v[0]）
 *   - depth=5 是完整 UFP-1（v[0..5]，48-bit 存储精度）
 *
 * 编码：
 *   remaining = w
 *   for layer in 0..depth:
 *       scale_layer = max(|remaining_array|) / 127
 *       q = round(remaining / scale_layer), clamp(-127, 127)
 *       v[layer] = q + 128
 *       remaining = remaining - q * scale_layer
 *
 * 解码：
 *   w_approx = sum_{layer=0..depth} (v[layer] - 128) * scales[layer]
 *
 * 矩阵乘采用方案 C（存储高精度，运算 v[0]）：
 *   - 权重存储用 v[0..depth]（48-bit 精度）
 *   - 前向运算时反量化到 float，重新量化到 v[0]，再做矩阵乘
 *   - 这样 UFP-1 的运算速度与阶段 1.3 相同，只是存储精度提升
 */

/**
 * 残差 scale 集：每层一个 scale，未使用的层为 0
 */
typedef struct {
    float scales[6];  /**< 6 层 scale，scales[i] 对应 v[i] */
} hc8_residual_scales_t;

/**
 * 残差量化：float 数组 → HC8 数组（v[0..depth] 填充，v[depth+1..5]=0）
 *
 * @param w            float 输入数组
 * @param n            数组长度
 * @param depth        残差深度（0~5，0 等价于 hc8_quantize）
 * @param schema       量化方案
 * @param out          输出 HC8 数组（长度 n，需预分配）
 * @param out_scales   输出 scale 集（out_scales->scales[0..depth] 填充，[depth+1..5]=0）
 */
void hc8_quantize_residual(const float* w, uint32_t n,
                           int depth,
                           const hc8_quant_schema_t* schema,
                           hc8_t* out,
                           hc8_residual_scales_t* out_scales);

/**
 * 残差反量化：HC8 数组 → float 数组（累加 v[0..depth]）
 *
 * @param h            HC8 输入数组
 * @param n            数组长度
 * @param depth        残差深度
 * @param scales       scale 集
 * @param schema       量化方案（提供 offset）
 * @param out          输出 float 数组（长度 n，需预分配）
 */
void hc8_dequantize_residual(const hc8_t* h, uint32_t n,
                             int depth,
                             const hc8_residual_scales_t* scales,
                             const hc8_quant_schema_t* schema,
                             float* out);

/**
 * 残差矩阵乘（方案 C：存储高精度，运算 v[0]）
 *
 * 流程：
 *   1. A、B 反量化到 float（用 v[0..depth]）
 *   2. 基于 float A、B 重新量化到 v[0]（用各自的 scale_0）
 *   3. 做 v[0] 矩阵乘（hc8_matmul）
 *   4. 输出是 v[0] 格式（深度 0）
 *
 * 注意：输出的 depth 总是 0（运算精度是 8-bit），UFP-1 只提升存储精度。
 *       要在前向使用 v[0..depth] 运算，需要 UFP-2（未来工作）。
 *
 * @param a            输入矩阵 A（m×k，HC8 残差格式）
 * @param b            输入矩阵 B（k×n，HC8 残差格式）
 * @param m, k, n      矩阵维度
 * @param a_depth      A 的残差深度
 * @param b_depth      B 的残差深度
 * @param a_scales     A 的 scale 集
 * @param b_scales     B 的 scale 集
 * @param schema       量化方案
 * @param out          输出矩阵 C（m×n，HC8 v[0] 格式，需预分配）
 * @param out_scale    输出 scale（输出参数）
 */
void hc8_residual_matmul(const hc8_t* a, const hc8_t* b,
                         uint32_t m, uint32_t k, uint32_t n,
                         int a_depth, int b_depth,
                         const hc8_residual_scales_t* a_scales,
                         const hc8_residual_scales_t* b_scales,
                         const hc8_quant_schema_t* schema,
                         hc8_t* out, float* out_scale);

/**
 * 残差 ReLU：对 v[0] 做 ReLU，v[1..5] 清零
 *
 * ReLU 是非线性运算，跨层残差不再有效，所以输出总是 depth=0。
 *
 * @param x            输入矩阵 X（m×n，HC8 残差格式）
 * @param m, n         矩阵维度
 * @param schema       量化方案
 * @param out          输出矩阵（m×n，HC8 v[0] 格式，需预分配）
 */
void hc8_residual_relu(const hc8_t* x, uint32_t m, uint32_t n,
                       const hc8_quant_schema_t* schema,
                       hc8_t* out);

/* ============================================================================
 * UFP-2 方案 B：整数域累加矩阵乘（运算精度提升到 16-48 bit）
 * ============================================================================
 *
 * 设计见 hc_ufp_2_scheme_b_design.md。
 *
 * 与方案 C（hc8_residual_matmul）的区别：
 *   - 方案 C：反量化到 float32 → 重新量化到 v[0] → int8 矩阵乘
 *             精度损失：float32 只有 24 位尾数，48-bit 存储会截断
 *   - 方案 B：直接用 v[0..depth] 的 int8 值做矩阵乘
 *             整数累加精确，用 double 合并（52 位尾数，够 48-bit）
 *
 * 数学：
 *   C[i][j] = Σ_l Σ_m [a_scales[l] * b_scales[m]] * [Σ_k (a.v[l]-off) * (b.v[m]-off)]
 *                          ↑ double combined_scale          ↑ int32 精确累加
 *
 * 计算量：(a_depth+1) * (b_depth+1) 次 int8 矩阵乘
 *   depth=0: 1 次（与方案 C 等价）
 *   depth=5: 36 次
 */

/**
 * UFP-2 方案 B 残差矩阵乘（整数域累加 + double 合并）
 *
 * 流程：
 *   1. 对每对 (l, m) where a_scales[l]>0 and b_scales[m]>0:
 *      a. int8 × int8 → int32 累加（k 维度）
 *      b. acc_double += (double)int32_acc * (a_scales[l] * b_scales[m])
 *   2. 重新量化 c_double 到 v[0]（输出 depth=0）
 *
 * 注意：输出 depth 总是 0（从 double 量化到 8-bit）。
 *       要输出 v[0..depth]，需要 UFP-3（未来工作）。
 *
 * @param a            输入矩阵 A（m×k，HC8 残差格式）
 * @param b            输入矩阵 B（k×n，HC8 残差格式）
 * @param m, k, n      矩阵维度
 * @param a_depth      A 的残差深度
 * @param b_depth      B 的残差深度
 * @param a_scales     A 的 scale 集
 * @param b_scales     B 的 scale 集
 * @param schema       量化方案
 * @param out          输出矩阵 C（m×n，HC8 v[0] 格式，需预分配）
 * @param out_scale    输出 scale（输出参数）
 */
void hc8_residual_matmul_b(const hc8_t* a, const hc8_t* b,
                           uint32_t m, uint32_t k, uint32_t n,
                           int a_depth, int b_depth,
                           const hc8_residual_scales_t* a_scales,
                           const hc8_residual_scales_t* b_scales,
                           const hc8_quant_schema_t* schema,
                           hc8_t* out, float* out_scale);

/* ============================================================================
 * HC4 非对称拆分：int8 → int4+int4 运算路径
 * ============================================================================
 *
 * 设计见 hc_v1.4_asymmetric_split_design.md。
 *
 * 核心数学（UFP 前瞻定理 1.1）：
 *   HC8 的 v[l] ∈ [0,255] 拆为 h_high = v[l]>>4 ∈ [0,15], h_low = v[l]&0x0F ∈ [0,15]
 *   矩阵乘 a_q*b_q = (a_h*16+a_l-128)*(b_h*16+b_l-128)
 *   分解为 4 次 int4×int4 累加 + 偏移修正项
 *
 * 关键发现：hc4_t.packed[l] 与 hc8_t.v[l] 二进制完全相同。
 *   拆分/合并是零成本操作（只是类型转换），区别在矩阵乘的解读方式。
 *
 * 位宽安全：uint4×uint4 最大 225（8 bit），累加 k 次需 8+log2(k) bit。
 *   MNIST k=784 → 18 bit，CIFAR-10 k=3072 → 20 bit，int32 安全。
 */

/**
 * HC4 类型：12×int4 紧缩存储（6 字节，与 hc8_t 二进制相同）
 *
 * packed[l] 的 high nibble = h[2l]（即 v[l] 的高 4 bit）
 * packed[l] 的 low nibble  = h[2l+1]（即 v[l] 的低 4 bit）
 *
 * 注意：packed[l] 的二进制值 == hc8_t.v[l]，两者可互转（零成本）
 */
#pragma pack(push, 1)
typedef struct {
    uint8_t packed[6];  /* 12 个 int4 紧缩为 6 字节 */
} hc4_t;
#pragma pack(pop)  // C1 补全修复 2026-09-07：pack() 只重置默认不清 push 栈，Clang 仍报 unterminated

/**
 * HC8 → HC4 拆分（零成本，二进制相同）
 *
 * @param hc8  输入 HC8 数组（长度 n）
 * @param n    元素数
 * @param out  输出 HC4 数组（长度 n，需预分配）
 */
void hc8_split_to_hc4(const hc8_t* hc8, uint32_t n, hc4_t* out);

/**
 * HC4 → HC8 合并（零成本，二进制相同，无损双射）
 *
 * @param hc4  输入 HC4 数组（长度 n）
 * @param n    元素数
 * @param out  输出 HC8 数组（长度 n，需预分配）
 */
void hc4_merge_to_hc8(const hc4_t* hc4, uint32_t n, hc8_t* out);

/**
 * HC4 残差矩阵乘（int4×int4→int32 累加 + double 合并）
 *
 * 数学：
 *   对每对 (l, m_idx) where a_scales[l]>0 and b_scales[m_idx]>0:
 *     1. 拆分 a.packed[l] → a_high, a_low
 *     2. 拆分 b.packed[m_idx] → b_high, b_low
 *     3. 4 次 int4×int4 矩阵乘：S_hh, S_hl, S_lh, S_ll
 *     4. 偏移修正：acc = 256*S_hh + 16*S_hl + 16*S_lh + S_ll
 *                  - 2048*sum_ah - 128*sum_al - 2048*sum_bh - 128*sum_bl + 16384*k
 *     5. c_double += (double)acc * (a_scales[l] * b_scales[m_idx])
 *   重新量化 c_double 到 v[0]（输出 depth=0）
 *
 * 与 hc8_residual_matmul_b 数学等价：相同输入 → 相同输出（max_diff=0）
 *
 * @param a            输入矩阵 A（m×k，HC4 格式）
 * @param b            输入矩阵 B（k×n，HC4 格式）
 * @param m, k, n      矩阵维度
 * @param a_depth      A 的残差深度
 * @param b_depth      B 的残差深度
 * @param a_scales     A 的 scale 集（与 HC8 共享）
 * @param b_scales     B 的 scale 集
 * @param schema       量化方案
 * @param out          输出矩阵 C（m×n，HC8 v[0] 格式，需预分配）
 * @param out_scale    输出 scale
 */
void hc4_residual_matmul_b(const hc4_t* a, const hc4_t* b,
                           uint32_t m, uint32_t k, uint32_t n,
                           int a_depth, int b_depth,
                           const hc8_residual_scales_t* a_scales,
                           const hc8_residual_scales_t* b_scales,
                           const hc8_quant_schema_t* schema,
                           hc8_t* out, float* out_scale);

/* ============================================================================
 * SIMD 加速：SoA 数据布局 + AVX-VNNI/AVX2 kernel
 * ============================================================================
 *
 * 设计见 hc_simd_avx_vnni_design.md v1.2。
 *
 * E 阶段（数据布局重排）：
 *   - HC8 SoA per-layer：把 hc8_t.v[l] 跨步 6 字节 → 连续 uint8 数组 layer[l][N]
 *   - HC4 SoA per-nibble：把 hc4_t.packed[l] 解包为 high[l][N] + low[l][N]
 *   - 通用 uint8 矩阵转置：B(k×n) → B_T(n×k)，让列连续
 *
 * A 阶段（SIMD kernel）：
 *   - HC8：AVX-VNNI _mm256_dpbusd_epi32（uint8×int8→int32 累加）
 *   - HC4：AVX2 _mm256_maddubs_epi16 + _mm256_madd_epi16（int4×int4→int32 累加）
 */

/**
 * HC8 SoA 结构：每层一个连续 uint8 数组
 *
 * layer[l] 指向 N 个 uint8（第 l 层数据，值 1-255）
 * layer[l] 在 hc8_soa_free 时释放
 */
typedef struct {
    uint8_t* layer[6];  /* layer[l] 指向 N 个 uint8（第 l 层） */
    uint32_t n;         /* 元素数 */
} hc8_soa_t;

/**
 * HC8 AoS → SoA 转换
 *
 * 把 hc8_t[N].v[l] 重排为 layer[l][N]（每层连续）
 *
 * @param aos  输入 AoS 数组（长度 n）
 * @param n    元素数
 * @param out  输出 SoA 结构（内部 malloc 6 个 layer 数组，用 hc8_soa_free 释放）
 */
void hc8_aos_to_soa(const hc8_t* aos, uint32_t n, hc8_soa_t* out);

/**
 * HC8 SoA 释放（释放内部 malloc 的 layer 数组，不释放 out 本身）
 */
void hc8_soa_free(hc8_soa_t* soa);

/**
 * HC4 SoA 结构：每层 2 个 nibble 数组（high + low）
 *
 * high[l] / low[l] 指向 N 个 uint8（值 0-15）
 * 在 hc4_soa_free 时释放
 */
typedef struct {
    uint8_t* high[6];  /* high[l] 指向 N 个 uint8（第 l 层高 nibble） */
    uint8_t* low[6];   /* low[l] 指向 N 个 uint8（第 l 层低 nibble） */
    uint32_t n;        /* 元素数 */
} hc4_soa_t;

/**
 * HC4 AoS → SoA 预解包
 *
 * 把 hc4_t[N].packed[l] 解包为 high[l][N]（高 nibble）+ low[l][N]（低 nibble）
 * high[l][i] = (packed[l] >> 4) & 0x0F，low[l][i] = packed[l] & 0x0F
 *
 * @param aos    输入 AoS 数组（长度 n）
 * @param n      元素数
 * @param depth  残差深度（只解包 l=0..depth，其余层不分配）
 * @param out    输出 SoA 结构（内部 malloc，用 hc4_soa_free 释放）
 */
void hc4_unpack_to_soa(const hc4_t* aos, uint32_t n, int depth, hc4_soa_t* out);

/**
 * HC4 SoA 释放
 */
void hc4_soa_free(hc4_soa_t* soa);

/**
 * 通用 uint8 矩阵转置：k×n（行优先）→ n×k（行优先）
 *
 * @param src   输入 k×n 矩阵（行优先）
 * @param k, n  维度
 * @return      新 malloc 的 n×k 矩阵（行优先），调用者负责 free
 */
uint8_t* hc8_transpose_u8(const uint8_t* src, uint32_t k, uint32_t n);

/**
 * HC8 残差矩阵乘 SIMD 版（AVX-VNNI 加速）
 *
 * 与 hc8_residual_matmul_b 数学等价（max_diff=0），内部用 SoA 布局 + VNNI kernel。
 * 无 AVX-VNNI 时回退到标量版。
 *
 * 接口与 hc8_residual_matmul_b 完全一致。
 */
void hc8_residual_matmul_b_simd(const hc8_t* a, const hc8_t* b,
                                uint32_t m, uint32_t k, uint32_t n,
                                int a_depth, int b_depth,
                                const hc8_residual_scales_t* a_scales,
                                const hc8_residual_scales_t* b_scales,
                                const hc8_quant_schema_t* schema,
                                hc8_t* out, float* out_scale);

/**
 * HC4 残差矩阵乘 SIMD 版（AVX2 maddubs 加速）
 *
 * 与 hc4_residual_matmul_b 数学等价（max_diff=0），内部用 SoA per-nibble + AVX2 kernel。
 * 无 AVX2 时回退到标量版。
 *
 * 接口与 hc4_residual_matmul_b 完全一致。
 */
void hc4_residual_matmul_b_simd(const hc4_t* a, const hc4_t* b,
                                uint32_t m, uint32_t k, uint32_t n,
                                int a_depth, int b_depth,
                                const hc8_residual_scales_t* a_scales,
                                const hc8_residual_scales_t* b_scales,
                                const hc8_quant_schema_t* schema,
                                hc8_t* out, float* out_scale);

/* ============================================================================
 * A-HC8 阶段：AVX-VNNI SBE C 化（2026-07-22）
 *
 * SBE（语义块编码）per-block 量化 + VNNI 加速 matmul
 * 设计见 hc_simd_avx_vnni_design.md §4
 * ============================================================================ */

/**
 * SBE 权重 per-block 量化 + 预处理
 *
 * 把 (k, n) 权重矩阵按 k 维度分 groups 块，每块独立量化。
 * 每块预处理为 VNNI 友好格式（转置 + 有符号转换 + 预计算 sum_b）。
 *
 * @param w              k×n, row-major, float 输入权重
 * @param groups         分块数
 * @param k_block        每块列数
 * @param k, n           矩阵形状
 * @param w_signed_flat  输出: groups*n*k_block 个 int8（每块 n×k_block, row-major）
 * @param w_sum_b_flat   输出: groups*n 个 int32（每块 n 个 sum_b 值）
 * @param w_scales       输出: groups 个 float（每块的量化 scale）
 */
void sbe_quantize_weight_blocks_c(
    const float* w,
    uint32_t groups, uint32_t k_block, uint32_t k, uint32_t n,
    int8_t* w_signed_flat,
    int32_t* w_sum_b_flat,
    float* w_scales
);

/**
 * SBE 分块 matmul（C 循环 + VNNI kernel + float 累加）
 *
 * 对每个 group g：
 *   1. 提取 x_block + per-block 量化
 *   2. VNNI matmul（AVX-VNNI _mm256_dpbusd_epi32 加速）
 *   3. float 累加到输出 y
 *
 * @param x            m×k, row-major, float 输入
 * @param w_signed     groups*n*k_block 个 int8（由 sbe_quantize_weight_blocks_c 预处理）
 * @param w_sum_b      groups*n 个 int32（由 sbe_quantize_weight_blocks_c 预计算）
 * @param w_scales     groups 个 float（由 sbe_quantize_weight_blocks_c 预计算）
 * @param groups       分块数
 * @param k_block      每块列数
 * @param m, k, n      矩阵形状
 * @param y            m×n, row-major, float 输出（调用者负责清零）
 */
void sbe_matmul_c(
    const float* x,
    const int8_t* w_signed,
    const int32_t* w_sum_b,
    const float* w_scales,
    uint32_t groups, uint32_t k_block, uint32_t m, uint32_t k, uint32_t n,
    float* y
);

/**
 * SBE Conv2d 前向融合（v2.0.0-conv-fusion，2026-07-28）
 *
 * 在 C 层融合 im2col + SBE matmul + bias add，消除 Python 层开销。
 * 数学等价于：im2col(x) → sbe_matmul → reshape + bias。
 *
 * @param x            (B, C_in, H, W), row-major, float32
 * @param w_signed     groups*n*k_block 个 int8（由 sbe_quantize_weight_blocks_c 预处理）
 * @param w_sum_b      groups*n 个 int32（由 sbe_quantize_weight_blocks_c 预计算）
 * @param w_scales     groups 个 float（由 sbe_quantize_weight_blocks_c 预计算）
 * @param B, C_in, H, W  输入尺寸
 * @param C_out, kh, kw, stride, padding  卷积参数
 * @param groups, k_block  SBE 分块参数
 * @param bias         (C_out,) 或 NULL（不加 bias）
 * @param y            (B, C_out, H_out, W_out), row-major, float32, 调用者分配
 */
void sbe_conv2d_forward_c(
    const float* x,
    const int8_t* w_signed,
    const int32_t* w_sum_b,
    const float* w_scales,
    uint32_t B, uint32_t C_in, uint32_t H, uint32_t W,
    uint32_t C_out, uint32_t kh, uint32_t kw,
    uint32_t stride, uint32_t padding,
    uint32_t groups, uint32_t k_block,
    const float* bias,
    float* y
);

/**
 * SBE + Smoothing C 分块 matmul（Smoothing 融合到 C 扩展，v1.7.0，2026-07-23）
 *
 * 与 sbe_matmul_c 数学等价（当 mean_shift 不改变量化舍入时）但更精确：
 *   对每个 group g 的 x_block 做 per-row mean shift 后再量化，
 *   主项 INT8 matmul + 修正项 float 累加。
 *
 * 数学：
 *   y = Σ_g x_block_g @ w_block_g
 *     = Σ_g [(x_block_g - c_mean_g) @ w_block_g + c_mean_g @ w_block_g]
 *       ↑ INT8 matmul (shifted, 量化更精确)    ↑ float 修正项
 *
 *   修正项优化: c_mean @ w = c_mean * w_sum
 *     其中 w_sum[j] = w_scale * Σ_k w_int8[j][k] = w_scale * w_sum_b[j]
 *
 * 流程（对每个 group g）：
 *   1. per-row mean: c_mean[i] = mean(x_block[i, :])
 *   2. shift: x_shifted[i, :] = x_block[i, :] - c_mean[i]
 *   3. per-block scale + quantize(x_shifted) → x_u (uint8)
 *   4. VNNI matmul(x_u, w_signed_T) → c_int32
 *   5. float 累加: y += c_int32 * x_scale * w_scale
 *   6. 修正项累加: y[i, j] += c_mean[i] * (w_sum_b[j] * w_scale)
 *
 * @param x            m×k, row-major, float 输入
 * @param w_signed     groups*n*k_block 个 int8（由 sbe_quantize_weight_blocks_c 预处理）
 * @param w_sum_b      groups*n 个 int32（由 sbe_quantize_weight_blocks_c 预计算）
 * @param w_scales     groups 个 float（由 sbe_quantize_weight_blocks_c 预计算）
 * @param groups       分块数
 * @param k_block      每块列数
 * @param m, k, n      矩阵形状
 * @param y            m×n, row-major, float 输出（调用者负责清零）
 */
void sbe_matmul_smoothed_c(
    const float* x,
    const int8_t* w_signed,
    const int32_t* w_sum_b,
    const float* w_scales,
    uint32_t groups, uint32_t k_block, uint32_t m, uint32_t k, uint32_t n,
    float* y
);

/* ============================================================================
 * A-HC8 阶段：per-channel 量化 SBE C 化（2026-07-23）
 *
 * 与 per-block SBE 的区别：w_scales 从 (groups,) 改为 (groups, n)，
 * 每个 block 内每个输出通道独立 scale，量化精度更高。
 *
 * 设计动机：CNN SBE 精度瓶颈（70.09% vs 71.25% 阈值），
 * per-channel 量化进一步降低量化噪声，与梯度累积正交可叠加。
 *
 * 数学等价性：per-channel 是 per-block 的精细化版本，
 *   per-block scale = max_j(per-channel scale[j])
 *   per-channel 量化误差 ≤ per-block 量化误差
 * ============================================================================ */

/**
 * SBE 权重 per-channel 量化 + 预处理
 *
 * 把 (k, n) 权重矩阵按 k 维度分 groups 块，每块内每个输出通道独立量化。
 *
 * 与 sbe_quantize_weight_blocks_c 的区别：
 *   - w_scales 输出形状从 (groups,) 改为 (groups, n)
 *   - 量化 scale 从 per-block（整个 block 共用）改为 per-channel（每列独立）
 *   - w_signed 和 w_sum_b 格式不变（复用 VNNI kernel）
 *
 * @param w              k×n, row-major, float 输入权重
 * @param groups         分块数
 * @param k_block        每块列数
 * @param k, n           矩阵形状
 * @param w_signed_flat  输出: groups*n*k_block 个 int8（与 per-block 格式相同）
 * @param w_sum_b_flat   输出: groups*n 个 int32（与 per-block 格式相同）
 * @param w_scales       输出: groups*n 个 float（per-channel，布局 (groups, n)）
 */
void sbe_quantize_weight_blocks_perchannel_c(
    const float* w,
    uint32_t groups, uint32_t k_block, uint32_t k, uint32_t n,
    int8_t* w_signed_flat,
    int32_t* w_sum_b_flat,
    float* w_scales  /* groups*n, 布局 (groups, n) */
);

/**
 * SBE per-channel 分块 matmul（C 循环 + VNNI kernel + per-channel float 累加）
 *
 * 与 sbe_matmul_c 的区别：
 *   - w_scales 是 (groups, n) 而非 (groups,)
 *   - float 累加用 per-channel scale 向量，而非标量广播
 *   - VNNI kernel 不变（int8×int8→int32 累加与 scale 无关）
 *
 * @param x            m×k, row-major, float 输入
 * @param w_signed     groups*n*k_block 个 int8（与 per-block 格式相同）
 * @param w_sum_b      groups*n 个 int32（与 per-block 格式相同）
 * @param w_scales     groups*n 个 float（per-channel，布局 (groups, n)）
 * @param groups       分块数
 * @param k_block      每块列数
 * @param m, k, n      矩阵形状
 * @param y            m×n, row-major, float 输出（调用者负责清零）
 */
void sbe_matmul_perchannel_c(
    const float* x,
    const int8_t* w_signed,
    const int32_t* w_sum_b,
    const float* w_scales,  /* groups*n, 布局 (groups, n) */
    uint32_t groups, uint32_t k_block, uint32_t m, uint32_t k, uint32_t n,
    float* y
);

/* ============================================================================
 * Triple-int8 缩放 C 化（v1.9.0，2026-07-24）
 *
 * 将 float 输入分解为 3 个 int8 分量（24-bit 精度），供 WEF+Triple 使用。
 * 数学等价于 sbe_conv2d.py 的 rescale_to_triple_int8_sbe，但用 AVX2 + OpenMP 加速。
 *
 * @param x          n 个 float，输入矩阵（任意形状，扁平化为 1D）
 * @param n          元素数
 * @param C_high     n 个 float 输出，∈ [-127, 127]（主量化）
 * @param C_mid      n 个 float 输出，∈ [-128, 128]（中位量化）
 * @param C_low      n 个 float 输出，∈ [-128, 128]（低位量化）
 * @param out_scale  输出标量 = max(|x|) / 127
 */
void sbe_rescale_to_triple_c(
    const float* x, uint32_t n,
    float* C_high, float* C_mid, float* C_low, float* out_scale
);

/* ============================================================================
 * 正交多视角 matmul（HC 树并行解读，v1.6.0，2026-07-23）
 *
 * 设计见 hc_tree_multiview_unified.md（方案 C 核心交付物）。
 * Python 参考实现：legacy/traditional/stage_2_3_int_path/hc_tree_unified.py
 *
 * 核心思想（与 UFP 串行残差链的区别）：
 *   - UFP 串行：分解原始 int8 输入为残差链 v[0],v[1]...（视角间有依赖）
 *   - 正交并行：分解累加值 C (int64) 为位段视角 c_i = (C >> 8i) & 0xFF（独立）
 *
 * 多视角 matmul（后续层用，首层用标准 int8×int8）：
 *   C_next = Σ_i (c_i @ b_int8) * 2^(8i)
 *
 * 其中 c_i 是 C_acc 的第 i 个字节（uint8, 0-255），b_int8 是有符号权重（-127~127）。
 * 复杂度：n_views 次 uint8×int8 matmul（无交叉项）。
 *
 * 重要约束（4 视角符号丢失）：
 *   - 4 视角覆盖 32 bit，要求 C_acc >= 0（ReLU 后）
 *   - 负数 int64 高 32 位是 0xFFFFFFFF，4 视角会丢失符号
 *   - 8 视角才能处理有符号 int64
 *   - 真实流程中每层 matmul 后都有 ReLU_mv，所以 C_acc >= 0
 * ============================================================================ */

/**
 * 正交多视角 matmul（后续层：分解累加值）
 *
 * 分解累加值 C_acc 为 n_views 个 uint8 视角，权重保持 int8：
 *   C_next[i][j] = Σ_v [ Σ_l c_v[i][l] * b[l][j] ] * 2^(8v)
 *   其中 c_v[i][l] = (C_acc[i][l] >> 8v) & 0xFF
 *
 * 与 Python hc_tree_unified.matmul_multiview 数学等价（max_diff=0）。
 *
 * @param c_acc    输入累加值（m×k，行优先，int64，ReLU 后 >= 0）
 * @param b_int8   输入权重（k×n，行优先，int8，有符号 [-127, 127]）
 * @param m, k, n  矩阵维度
 * @param n_views  视角数（4 覆盖 int32，8 覆盖 int64）
 * @param out      输出累加值（m×n，行优先，int64，函数内部清零，需预分配）
 */
void hc8_multiview_matmul(
    const int64_t* c_acc,
    const int8_t* b_int8,
    uint32_t m, uint32_t k, uint32_t n,
    int n_views,
    int64_t* out
);

/* ============================================================================
 * AVX-VNNI 运行时检测（v1.5.1，2026-07-22）
 *
 * 两阶段检测：
 *   1. CPUID 检查：CPUID.07H.01H:EAX[4] = 1？
 *   2. SEH 探针：实际执行 _mm256_dpbusd_epi32，捕获 ILLEGAL_INSTRUCTION
 *
 * 需要两阶段的原因：部分 Zhaoxin/Hygon CPU 在 CPUID 中报告 AVX-VNNI 支持，
 * 但实际执行 VPDPBUSD 指令会触发非法指令异常。
 *
 * @return 1 = 支持 AVX-VNNI, 0 = 不支持（回退标量路径）
 * ============================================================================ */
int hc_detect_avx_vnni(void);

/* 调试用：返回 CPUID 7.1 的原始 EAX/EBX/ECX/EDX 值 */
void hc_cpuid_7_1_raw(int* eax, int* ebx, int* ecx, int* edx);

/* ============================================================================
 * SSSE3 优化函数
 * ============================================================================ */

/**
 * PABSB 绝对值 L1 距离（int8 向量）
 * @param a, b  输入向量（uint8 数组，长度 n）
 * @param n     向量长度
 * @return      Σ |a[i] - b[i]|
 */
int32_t hc8_l1_distance_avx2(const uint8_t* a, const uint8_t* b, uint32_t n);

/**
 * PSIGNB 符号门控更新（原地修改 delta）
 * @param delta  更新值（uint8 数组，原地修改）
 * @param mask   门控 mask（int8 数组，0=清零, >0=保持, <0=取反）
 * @param n      数组长度
 */
void hc8_sign_gated_update(uint8_t* delta, const int8_t* mask, uint32_t n);

#ifdef __cplusplus
}
#endif

#endif /* SGN_HC8_NET_H */
