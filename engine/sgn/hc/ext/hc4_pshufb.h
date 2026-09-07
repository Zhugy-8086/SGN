/**
 * @file hc4_pshufb.h
 * @brief HC4 PSHUFB LUT - int4×int4→int8 查表乘法（AVX2 _mm256_shuffle_epi8）
 * @version 1.6.0
 *
 * 基于 PSHUFB（_mm256_shuffle_epi8）实现 int4×int4→int8 的高效乘法。
 *
 * 核心思想：
 *   预计算 16×16 乘法 LUT（256 字节，16 个 16 字节的表）
 *   LUT[b][a] = a * b，其中 a, b ∈ [0, 15]（uint4）
 *   用 _mm256_shuffle_epi8 一次查表 32 个乘积
 *
 * PSHUFB 查表原理：
 *   _mm256_shuffle_epi8(LUT, idx) 对 idx 的每个字节 i：
 *     - 若 idx[i] 高位为 1（≥128），结果为 0
 *     - 否则结果 = LUT[idx[i] & 0x0F]
 *   即：result[i] = LUT[idx[i] & 0x0F]（idx[i] < 128 时）
 *
 * 多 b 值处理（关键设计）：
 *   PSHUFB 只能用一个 16 字节的 LUT，但输入 b 有 16 种可能值。
 *   方案：对每个 b 值 v（0-15），迭代处理：
 *     1. 创建掩码 mask = (b == v) ? 0xFF : 0x00（_mm256_cmpeq_epi8）
 *     2. 用 LUT_v 做 PSHUFB 查表：prod = PSHUFB(LUT_v, a)
 *     3. 用掩码过滤：prod &= mask
 *     4. 累加：result |= prod（每位置只命中一次 b 值，用 OR 即可）
 *
 * 与 nibble split 的关系：
 *   nibble split：int8 = high_nibble * 16 + low_nibble，展开后用 int8 乘法
 *   PSHUFB LUT：直接查表 int4×int4 乘积，避免乘法指令
 *   两者数学等价（已在 msint_math_validation 中验证）
 *
 * 性能特点：
 *   - 16 次迭代（每个 b 值一次），每次 4 条 AVX2 指令
 *   - 一次处理 32 个 (a, b) 乘积对
 *   - 总指令数：64 条处理 32 个乘积
 *   - 比 _mm256_maddubs_epi16（1 条指令处理 32 个 int8×int8）慢
 *   - 但在某些场景（如低精度推理）可能有优势
 *
 * 参考：
 *   - 数学验证：architecture/msint_math_validation_results_2026_07_30.md
 *   - HC4 接口：hc8_net.h 的 hc4_residual_matmul_b
 *   - AVX2 文档：Intel Intrinsics Guide
 */

#ifndef SGN_HC4_PSHUFB_H
#define SGN_HC4_PSHUFB_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * 16×16 乘法 LUT
 * ============================================================================
 *
 * LUT[b*16 + a] = (uint8_t)(a * b)
 *   a, b ∈ [0, 15]（uint4）
 *   乘积 ∈ [0, 225]（8 bit 足够）
 *
 * 256 字节，16 个 16 字节的表（每个对应一个 b 值）
 */

/* LUT 访问宏：LUT[b][a] = a * b */
#define HC4_PSHUFB_LUT(b, a) ((uint8_t)((a) * (b)))

/* ============================================================================
 * PSHUFB 单批次乘法（32 个 int4×int4→int8）
 * ============================================================================ */

/**
 * PSHUFB 批量乘法：32 个 uint4 × uint4 → 32 个 uint8
 *
 * 输入：a 和 b 各 32 字节，每个字节是一个 uint4（0-15）
 * 输出：32 字节乘积，每个字节是 a[i] * b[i]
 *
 * 实现：对每个 b 值 v（0-15），用 PSHUFB 查 LUT_v
 *
 * @param a         32 字节输入 a（每个字节是 uint4）
 * @param b         32 字节输入 b（每个字节是 uint4）
 * @param out       32 字节输出（乘积）
 */
void hc4_pshufb_mul_32(const uint8_t* a, const uint8_t* b, uint8_t* out);

/* ============================================================================
 * HC4 PSHUFB matmul（int4×int4→int32 累加）
 * ============================================================================
 *
 * C = A @ B，其中 A: m×k, B: k×n, C: m×n（行优先）
 *
 * 输入：A 和 B 的元素是 uint4（0-15），存储为 uint8（每字节一个 uint4）
 * 运算：C[i][j] = sum_l A[i][l] * B[l][j]，累加到 int32
 *
 * 注意：本实现是 uint4 × uint4（无符号），不适用于有符号 int4
 *
 * @param a            输入矩阵 A（m×k，uint8 数组，每个字节是 uint4）
 * @param b            输入矩阵 B（k×n，uint8 数组，每个字节是 uint4）
 * @param m, k, n      矩阵维度
 * @param out          输出矩阵 C（m×n，int32 数组）
 */
void hc4_pshufb_matmul(const uint8_t* a, const uint8_t* b,
                        uint32_t m, uint32_t k, uint32_t n,
                        int32_t* out);

/* ============================================================================
 * 便捷封装：float → float 的 HC4 PSHUFB 量化 matmul
 * ============================================================================
 *
 * 将 float 输入量化到 uint4（0-15），做 PSHUFB matmul，反量化回 float
 *
 * 量化方案：uint4 对称量化
 *   scale = max(|w|) / 7（4-bit 有符号范围 [-7, 7]，但本实现用 uint4 [0, 15]）
 *   实际：把 float [-1, 1] 映射到 uint4 [0, 15]，中心 8 对应 0
 *   q = round((w / scale) + 8), clamp to [0, 15]
 *   w_approx = (q - 8) * scale
 *
 * @param x            输入矩阵 X（m×k，float 数组）
 * @param w            输入矩阵 W（k×n，float 数组）
 * @param m, k, n      矩阵维度
 * @param out          输出矩阵 Y（m×n，float 数组）
 */
void hc4_pshufb_quantized_matmul(const float* x, const float* w,
                                  uint32_t m, uint32_t k, uint32_t n,
                                  float* out);

/* ============================================================================
 * 运行时检测
 * ============================================================================ */

/**
 * 检测 CPU 是否支持 AVX2
 * @return 1=支持, 0=不支持
 */
int hc4_pshufb_detect_avx2(void);

#ifdef __cplusplus
}
#endif

#endif /* SGN_HC4_PSHUFB_H */
