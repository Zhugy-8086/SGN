/**
 * @file hc16_net.h
 * @brief HC16 神经网络运算扩展（MSInt 位宽链中间层）
 * @version 1.5.0
 *
 * 基于 MSInt 位宽重新解释研究（2026-07-30），HC16 填补 int32↔int16↔int8↔int4
 * 位宽链的 int16 中间层，利用 AVX2 `_mm256_madd_epi16` 指令实现高效 int16×int16→int32 累加。
 *
 * 设计原则：
 *   - 不修改 engine/hc/ 下任何现有文件
 *   - 独立编译为 pysgn_hc16 扩展模块
 *   - 通过 pybind11 与 Python 交互
 *   - 存储直接用 int16_t（有符号，无需 offset 偏移）
 *
 * 与 HC8 的关键区别：
 *   - HC8: uint8 + offset 128（无符号偏移表示），int8×int8→int32 累加
 *   - HC16: int16 有符号直接存储，int16×int16→int32/int64 累加
 *   - 精度提升 258x（32767/127），适用于反向梯度存储和高精度计算
 *
 * 数学基础（已通过 msint_math_validation_2026_07_30.py 验证）：
 *   1. 量化：scale = max(|w|) / 32767, q = round(w / scale) ∈ [-32768, 32767]
 *   2. matmul 路径 C：int16×int16→int64 累加（防溢出，K=1024 时 int32 溢出）
 *   3. AVX2 路径：_mm256_madd_epi16 横向累加 4×int32，分块累加到 int64
 *
 * 依赖：
 *   - 无 engine/hc 依赖（独立 int16 存储，不复用 hc8_t）
 *   - AVX2 intrinsics（编译时 /arch:AVX2）
 *
 * 参考：
 *   - 数学验证：内部验证记录（结论已内联于本模块注释）
 *   - 指令集兼容性：内部档案
 *   - SIMD 设计：内部档案
 *   - HC8 接口规范：hc8_net.h（设计模式参考）
 */

#ifndef SGN_HC16_NET_H
#define SGN_HC16_NET_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 量化方案：对称线性量化（int16 有符号直接存储）
 * ============================================================================
 *
 * 流程：
 *   1. 对称量化：scale = max(|w|) / 32767, q = round(w / scale) ∈ [-32768, 32767]
 *   2. 直接存 int16_t（有符号，无需 offset 偏移）
 *
 * 反量化：
 *   1. q = int16 值
 *   2. w = q * scale
 *
 * 与 HC8 的区别：
 *   - HC8 需要 offset 128 偏移到无符号（uint8 存储）
 *   - HC16 直接用 int16 有符号存储，省去偏移步骤
 */

/**
 * HC16 对称量化方案参数
 */
typedef struct {
    float    scale;       /**< 量化 scale（float → int 时的缩放因子） */
    int32_t  qmin;        /**< 对称量化下界（-32768） */
    int32_t  qmax;        /**< 对称量化上界（32767） */
} hc16_quant_schema_t;

/**
 * 从 float 数组推导量化 scale
 *   scale = max(|w[i]|) / 32767，全零时返回 1.0
 *
 * @param w        float 输入数组
 * @param n        数组长度
 * @return         量化 scale
 */
float hc16_quant_compute_scale(const float* w, uint32_t n);

/**
 * 量化 float 数组到 int16 数组
 *   每个元素：q = round(w[i] / scale), clamp(-32768, 32767)
 *   直接存为 int16_t（有符号，无需偏移）
 *
 * @param w        float 输入数组
 * @param n        数组长度
 * @param scale    量化 scale（由 hc16_quant_compute_scale 推导）
 * @param schema   量化方案（提供 qmin/qmax）
 * @param out      输出 int16 数组（长度 n，需预分配）
 */
void hc16_quantize(const float* w, uint32_t n,
                   float scale,
                   const hc16_quant_schema_t* schema,
                   int16_t* out);

/**
 * 反量化 int16 数组到 float 数组
 *   每个元素：w = q * scale
 *
 * @param h        int16 输入数组
 * @param n        数组长度
 * @param scale    量化 scale
 * @param out      输出 float 数组（长度 n，需预分配）
 */
void hc16_dequantize(const int16_t* h, uint32_t n,
                     float scale,
                     float* out);

/* ============================================================================
 * HC16 矩阵乘（整数路径）
 * ============================================================================
 *
 * C = A @ B，其中 A: m×k, B: k×n, C: m×n（行优先）
 *
 * 整数路径（路径 C，已验证防溢出）：
 *   1. int16 × int16 → int64 累加（K 维度）
 *      c_acc[i][j] = sum_l(a[i][l] * b[l][j])
 *   2. 反量化：c_float = c_acc * a_scale * b_scale
 *
 * 溢出分析：
 *   - int16×int16 最大 32767^2 ≈ 1.07e9
 *   - K=1024 时最大累加 1.07e9 * 1024 ≈ 1.1e12，远超 int32 上限（2.1e9）
 *   - 必须用 int64 累加器（上限 9.2e18，安全）
 *
 * AVX2 优化路径（_mm256_madd_epi16）：
 *   - 单指令完成 16× int16×int16 → 8× int32 横向累加
 *   - 分块策略：每块 K_BLOCK 个元素用 int32 横向累加，块间用 int64 累加
 *   - K_BLOCK 安全阈值：int32 上限 / (32767^2) ≈ 2.1e9 / 1.07e9 ≈ 1.96
 *     即每块最多 1 个元素用 int32 累加（不安全！）
 *   - 实际策略：_mm256_madd_epi16 输出 int32 后立即转 int64 累加
 *
 * 性能预期：
 *   - 标量路径：~10x 慢于 numpy float32 BLAS
 *   - AVX2 路径：~1.5-2x 慢于 numpy float32 BLAS（int16 数据量减半补偿指令开销）
 *   - 优势在批量量化场景（梯度存储精度提升 258x）
 */

/**
 * HC16 矩阵乘（标量路径，int64 累加器）
 *
 * @param a            输入矩阵 A（m×k，行优先，int16 数组，长度 m*k）
 * @param b            输入矩阵 B（k×n，行优先，int16 数组，长度 k*n）
 * @param m, k, n      矩阵维度
 * @param a_scale      A 的量化 scale
 * @param b_scale      B 的量化 scale
 * @param out          输出矩阵 C（m×n，float 数组，长度 m*n，需预分配）
 */
void hc16_matmul_scalar(const int16_t* a, const int16_t* b,
                        uint32_t m, uint32_t k, uint32_t n,
                        float a_scale, float b_scale,
                        float* out);

/**
 * HC16 矩阵乘（AVX2 优化路径，_mm256_madd_epi16 + int64 累加）
 *
 * 运行时检测 AVX2 支持，不支持时自动回退到 hc16_matmul_scalar。
 *
 * @param a            输入矩阵 A（m×k，行优先，int16 数组，长度 m*k）
 * @param b            输入矩阵 B（k×n，行优先，int16 数组，长度 k*n）
 * @param m, k, n      矩阵维度
 * @param a_scale      A 的量化 scale
 * @param b_scale      B 的量化 scale
 * @param out          输出矩阵 C（m×n，float 数组，长度 m*n，需预分配）
 */
void hc16_matmul_avx2(const int16_t* a, const int16_t* b,
                      uint32_t m, uint32_t k, uint32_t n,
                      float a_scale, float b_scale,
                      float* out);

/**
 * HC16 矩阵乘（自动选择最优路径）
 *
 * 运行时检测 AVX2 支持，选择最优路径。
 * - AVX2 可用：调用 hc16_matmul_avx2
 * - AVX2 不可用：回退到 hc16_matmul_scalar
 *
 * @param a            输入矩阵 A（m×k，行优先，int16 数组，长度 m*k）
 * @param b            输入矩阵 B（k×n，行优先，int16 数组，长度 k*n）
 * @param m, k, n      矩阵维度
 * @param a_scale      A 的量化 scale
 * @param b_scale      B 的量化 scale
 * @param out          输出矩阵 C（m×n，float 数组，长度 m*n，需预分配）
 */
void hc16_matmul(const int16_t* a, const int16_t* b,
                 uint32_t m, uint32_t k, uint32_t n,
                 float a_scale, float b_scale,
                 float* out);

/* ============================================================================
 * 便捷封装：float → float 的 HC16 量化 matmul
 * ============================================================================
 *
 * 一步完成：float 输入 → 量化 → int16 matmul → 反量化 → float 输出
 * 用于与 Python numpy 路径对比验证数学等价性
 */

/**
 * HC16 量化 matmul（float → float，自动量化+反量化）
 *
 * @param x            输入矩阵 X（m×k，float 数组）
 * @param w            输入矩阵 W（k×n，float 数组）
 * @param m, k, n      矩阵维度
 * @param schema       量化方案（用于量化 X 和 W）
 * @param out          输出矩阵 Y（m×n，float 数组，需预分配）
 */
void hc16_quantized_matmul(const float* x, const float* w,
                           uint32_t m, uint32_t k, uint32_t n,
                           const hc16_quant_schema_t* schema,
                           float* out);

/* ============================================================================
 * 默认量化方案
 * ============================================================================ */

/**
 * 获取默认量化方案（qmin=-32768, qmax=32767）
 */
hc16_quant_schema_t hc16_quant_default_schema(void);

/* ============================================================================
 * 运行时检测
 * ============================================================================ */

/**
 * 检测 CPU 是否支持 AVX2
 * @return 1=支持, 0=不支持
 */
int hc16_detect_avx2(void);

/* ============================================================================
 * Per-channel scale 扩展（v1.5.0 新增，用于 2.6 非 STE 反向传播）
 * ============================================================================
 *
 * 背景：
 *   - per-tensor scale（上文 API）用一个标量 scale 量化整个张量
 *   - per-channel scale 为每行（行优先视角）提供独立 scale
 *   - 反向场景：grad_output (M×N) 按列（输出通道）per-channel scale，
 *               weight (N×K) 按行（输出通道）per-channel scale
 *   - 通过外部转置可统一为 "每行一个 scale" 的模型
 *
 * 数学定义（per-channel）：
 *   量化：对第 i 行，q[i][j] = round(w[i][j] / scales[i])，clamp(qmin, qmax)
 *   反量化：对第 i 行，w[i][j] = q[i][j] * scales[i]
 *   matmul：C[i][j] = sum_l(A[i][l] * B[l][j]) * a_scales[i] * b_scales[j]
 *           （即 A 按行 per-channel，B 按列 per-channel，
 *              B 行优先存储时 b_scales[j] 对应 B 的第 j 列，
 *              等价于 B 转置后按行 per-channel）
 *
 * 维度约定：
 *   - A: m×k 行优先，a_scales 长度 m（每行一个 scale）
 *   - B: k×n 行优先，b_scales 长度 n（每列一个 scale）
 *   - C: m×n 行优先，C[i][j] 用 a_scales[i] * b_scales[j] 组合
 *
 * 与 per-tensor 路径的关系：
 *   - 当 a_scales 全部相等且 b_scales 全部相等时，per-channel 退化为 per-tensor
 *   - per-channel 不影响整数累加路径（AVX2 madd + int64 累加器不变），
 *     仅在输出阶段按 (i, j) 组合 scale
 *
 * 溢出分析：
 *   - 与 per-tensor 路径一致，整数累加部分完全相同
 *   - 仅 scale 组合在 float 阶段进行，无额外溢出风险
 */

/**
 * Per-channel 量化 float 数组到 int16 数组
 *   对每个 "行"（row）使用独立 scale 量化
 *   每个元素：q[i][j] = round(w[i][j] / scales[i]), clamp(qmin, qmax)
 *
 * 内存布局（行优先）：
 *   w:     rows × per_row_size，按行存储
 *   scales: rows 个 scale，scales[i] 对应第 i 行
 *   out:   与 w 同布局
 *
 * @param w              float 输入数组（长度 rows * per_row_size）
 * @param rows           行数
 * @param per_row_size   每行元素数
 * @param scales         per-channel scale 数组（长度 rows）
 * @param schema         量化方案（提供 qmin/qmax）
 * @param out            输出 int16 数组（长度 rows * per_row_size，需预分配）
 */
void hc16_quantize_per_channel(const float* w, uint32_t rows, uint32_t per_row_size,
                               const float* scales,
                               const hc16_quant_schema_t* schema,
                               int16_t* out);

/**
 * Per-channel 反量化 int16 数组到 float 数组
 *   对每个 "行"（row）使用独立 scale 反量化
 *   每个元素：w[i][j] = q[i][j] * scales[i]
 *
 * @param h              int16 输入数组（长度 rows * per_row_size）
 * @param rows           行数
 * @param per_row_size   每行元素数
 * @param scales         per-channel scale 数组（长度 rows）
 * @param out            输出 float 数组（长度 rows * per_row_size，需预分配）
 */
void hc16_dequantize_per_channel(const int16_t* h, uint32_t rows, uint32_t per_row_size,
                                 const float* scales,
                                 float* out);

/**
 * HC16 矩阵乘（per-channel scale，自动选择最优路径）
 *
 * C[i][j] = sum_l(A[i][l] * B[l][j]) * a_scales[i] * b_scales[j]
 *
 * @param a            输入矩阵 A（m×k，行优先，int16，长度 m*k）
 * @param b            输入矩阵 B（k×n，行优先，int16，长度 k*n）
 * @param m, k, n      矩阵维度
 * @param a_scales     A 的 per-row scale 数组（长度 m）
 * @param b_scales     B 的 per-column scale 数组（长度 n）
 * @param out          输出矩阵 C（m×n，float，长度 m*n，需预分配）
 */
void hc16_matmul_per_channel(const int16_t* a, const int16_t* b,
                             uint32_t m, uint32_t k, uint32_t n,
                             const float* a_scales, const float* b_scales,
                             float* out);

/**
 * HC16 矩阵乘（per-channel scale，标量路径，int64 累加器）
 *   与 hc16_matmul_per_channel 相同语义，强制走标量路径（用于验证/调试）
 */
void hc16_matmul_per_channel_scalar(const int16_t* a, const int16_t* b,
                                    uint32_t m, uint32_t k, uint32_t n,
                                    const float* a_scales, const float* b_scales,
                                    float* out);

/**
 * HC16 矩阵乘（per-channel scale，AVX2 优化路径）
 *   与 hc16_matmul_per_channel 相同语义，强制走 AVX2 路径
 *   AVX2 不可用时自动回退到标量路径
 */
void hc16_matmul_per_channel_avx2(const int16_t* a, const int16_t* b,
                                  uint32_t m, uint32_t k, uint32_t n,
                                  const float* a_scales, const float* b_scales,
                                  float* out);

#ifdef __cplusplus
}
#endif

#endif /* SGN_HC16_NET_H */
