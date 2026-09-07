#ifdef __cplusplus
#include <bit>
#endif

/**
 * @file hc16ms.h
 * @brief HC16MS - MSInt int16 容器多视角存储（方案 B）
 * @version 1.5.0
 *
 * 基于 MSInt 位宽重新解释特性，用 2 字节 int16 容器替代 hc8_t 的 6 字节存储，
 * 实现 3x 存储压缩 + 3 档精度零拷贝切换。
 *
 * 核心思想：同一块 2 字节内存，按需以不同位宽读取
 *   ├── HC16 视角: 1 × int16  （高精度，65534 级，反向梯度存储）
 *   ├── HC8  视角: 2 × int8   （标准精度，254 级，前向激活值）
 *   └── HC4  视角: 4 × int4   （低精度，30 级，超低精度推理）
 *
 * vs hc8_t (6 字节):
 *   - 存储压缩 3x（6→2 字节）
 *   - 基础 HC8 利用率 16.7% → 100%（无浪费）
 *   - 支持 HC16 高精度档位（hc8_t 不支持）
 *
 * 数学基础（MSInt 位宽链）：
 *   int16 值 V 的二进制表示: [b15..b8][b7..b0]
 *   - HC16 视角: V（1 个 int16）
 *   - HC8  视角: [b15..b8] = high_byte, [b7..b0] = low_byte（2 个 int8）
 *   - HC4  视角: [b15..b12]=h0, [b11..b8]=h1, [b7..b4]=h2, [b3..b0]=h3（4 个 int4）
 *
 * 矩阵组织（关键设计）：
 *   hc16ms_t 数组 a[m×k] 在不同视角下等效于：
 *   - HC16 视角: int16 矩阵 (m×k)，1 个元素/hc16ms_t
 *   - HC8  视角: int8  矩阵 (m×2k)，2 个元素/hc16ms_t（高低字节展开）
 *   - HC4  视角: int8  矩阵 (m×4k)，4 个元素/hc16ms_t（4 nibble 展开）
 *
 * 编译：/arch:AVX2（启用 _mm256_madd_epi16, _mm256_maddubs_epi16）
 *
 * 参考：
 *   - 存储优化研究：architecture/msint_storage_optimization_research_2026_07_30.md
 *   - HC16 C 扩展：hc16_net.h（HC16 matmul 实现参考）
 *   - HC8 接口规范：hc8_net.h（设计模式参考）
 */

#ifndef SGN_HC16MS_H
#define SGN_HC16MS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * hc16ms_t 类型：2 字节 MSInt int16 容器
 * ============================================================================ */

#pragma pack(push, 1)
typedef struct {
    int16_t raw;  /**< 2 字节存储，按位宽重新解释得到不同精度 */
} hc16ms_t;
#pragma pack()

/* ============================================================================
 * 字节序/nibble 序交换工具（读取顺序切换）
 * ============================================================================
 *
 * 对同一块 hc16ms_t 内存，支持按相反字节序/nibble 序读取：
 *   - bswap16: int16 字节交换（0x1234 → 0x3412）
 *   - nswap8 : uint8 nibble 交换（0x3C → 0xC3）
 */

/* int16 字节交换：0x1234 → 0x3412 */
static inline int16_t hc16ms_bswap16(int16_t v) {
#ifdef __cplusplus
    return std::byteswap(v);
#else
    /* C 回退：手动位操作 */
    uint16_t u = (uint16_t)v;
    return (int16_t)((u >> 8) | (u << 8));
#endif
}

/* int8 nibble 交换：0x3C → 0xC3 */
static inline uint8_t hc16ms_nswap8(uint8_t v) {
    return ((v & 0x0F) << 4) | ((v & 0xF0) >> 4);
}

/* ============================================================================
 * 多视角读取：从 hc16ms_t 提取不同位宽的值
 * ============================================================================
 *
 * 内存布局（小端序，x86/ARM 默认）：
 *   raw = 0xHHLL
 *   字节 0 (低地址): LL (低字节)
 *   字节 1 (高地址): HH (高字节)
 *
 * HC16 视角: 直接读 int16
 * HC8  视角: high = HH, low = LL（2 个 int8）
 * HC4  视角: h0=HH>>4, h1=HH&0xF, h2=LL>>4, h3=LL&0xF（4 个 uint4）
 */

/**
 * HC16 视角读取：1 个 int16
 */
static inline int16_t hc16ms_read_hc16(const hc16ms_t* h) {
    return h->raw;
}

/**
 * HC8 视角读取：2 个 int8（高字节 + 低字节）
 *
 * @param h        hc16ms_t 输入
 * @param high     输出高字节 int8
 * @param low      输出低字节 int8
 */
static inline void hc16ms_read_hc8(const hc16ms_t* h,
                                    int8_t* high, int8_t* low) {
    const int8_t* p = (const int8_t*)&h->raw;
    *low  = p[0];  /* 小端序：低地址 = 低字节 */
    *high = p[1];  /* 小端序：高地址 = 高字节 */
}

/**
 * HC4 视角读取：4 个 uint4（范围 [0, 15]）
 *
 * 顺序：h0=高字节高 nibble, h1=高字节低 nibble,
 *       h2=低字节高 nibble, h3=低字节低 nibble
 *
 * @param h        hc16ms_t 输入
 * @param h0..h3   输出 4 个 uint4（范围 [0, 15]）
 */
static inline void hc16ms_read_hc4(const hc16ms_t* h,
                                    uint8_t* h0, uint8_t* h1,
                                    uint8_t* h2, uint8_t* h3) {
    const uint8_t* p = (const uint8_t*)&h->raw;
    *h2 = (p[0] >> 4) & 0xF;  /* 低字节高 nibble */
    *h3 = p[0] & 0xF;         /* 低字节低 nibble */
    *h0 = (p[1] >> 4) & 0xF;  /* 高字节高 nibble */
    *h1 = p[1] & 0xF;         /* 高字节低 nibble */
}

/* ============================================================================
 * 切换视角读取（读取顺序切换）：按相反字节序/nibble 序读取同一块内存
 * ============================================================================
 *
 * 与正向读取的关系：
 *   - hc16ms_read_hc16_swapped: 返回 hc16ms_bswap16(h->raw)
 *   - hc16ms_read_hc8_swapped : 与正向 read_hc8 相比 high/low 对调
 *                               （正向 *high=高字节, *low=低字节;
 *                                切换 *high=低字节, *low=高字节）
 *   - hc16ms_read_hc4_swapped : nibble 顺序反转，正向 [h0,h1,h2,h3]
 *                               切换返回 [h3,h2,h1,h0]
 *
 * 验证点（raw=0x1234，小端序内存：低字节0x34, 高字节0x12）：
 *   - 正向 HC4: [1,2,3,4]（h0=1,h1=2 来自高字节0x12; h2=3,h3=4 来自低字节0x34）
 *   - 切换 HC4: [4,3,2,1]（反转）
 */

/**
 * HC16 切换视角读取：字节交换后的 int16
 */
int16_t hc16ms_read_hc16_swapped(const hc16ms_t* h);

/**
 * HC8 切换视角读取：交换高低字节顺序（与正向 high/low 对调）
 *
 * @param h        hc16ms_t 输入
 * @param high     输出（切换视角下 = 低字节 int8）
 * @param low      输出（切换视角下 = 高字节 int8）
 */
void hc16ms_read_hc8_swapped(const hc16ms_t* h, int8_t* high, int8_t* low);

/**
 * HC4 切换视角读取：nibble 顺序反转（正向 [h0,h1,h2,h3] → [h3,h2,h1,h0]）
 *
 * @param h        hc16ms_t 输入
 * @param h0..h3   输出 4 个反转后的 uint4（h0=低字节低nibble, ..., h3=高字节高nibble）
 */
void hc16ms_read_hc4_swapped(const hc16ms_t* h,
                              uint8_t* h0, uint8_t* h1,
                              uint8_t* h2, uint8_t* h3);

/* ============================================================================
 * 多视角写入：从不同位宽的值构建 hc16ms_t
 * ============================================================================ */

/**
 * HC16 视角写入：1 个 int16 → hc16ms_t
 */
static inline void hc16ms_write_hc16(hc16ms_t* h, int16_t val) {
    h->raw = val;
}

/**
 * HC8 视角写入：2 个 int8 → hc16ms_t
 */
static inline void hc16ms_write_hc8(hc16ms_t* h, int8_t high, int8_t low) {
    int8_t* p = (int8_t*)&h->raw;
    p[0] = low;
    p[1] = high;
}

/**
 * HC4 视角写入：4 个 uint4 → hc16ms_t
 */
static inline void hc16ms_write_hc4(hc16ms_t* h,
                                     uint8_t h0, uint8_t h1,
                                     uint8_t h2, uint8_t h3) {
    uint8_t* p = (uint8_t*)&h->raw;
    p[0] = (h2 << 4) | (h3 & 0xF);  /* 低字节 */
    p[1] = (h0 << 4) | (h1 & 0xF);  /* 高字节 */
}

/* ============================================================================
 * 量化方案：float → hc16ms_t（不同视角）
 * ============================================================================
 */

/**
 * HC16 视角量化：float → hc16ms_t（1 个 int16）
 *   scale = max(|w|) / 32767, q = round(w / scale) ∈ [-32767, 32767]
 *
 * @param w        float 输入数组
 * @param n        数组长度
 * @param scale    量化 scale
 * @param out      输出 hc16ms_t 数组（长度 n，HC16 视角）
 */
void hc16ms_quantize_hc16(const float* w, uint32_t n,
                           float scale,
                           hc16ms_t* out);

/**
 * HC8 视角量化：float → hc16ms_t（2 个 int8，高低字节独立量化）
 *
 * 语义：每个 hc16ms_t 存 2 个独立的 HC8 值
 *   - 输入 w 长度 = 2n（n 个 hc16ms_t，每个存 2 个 int8）
 *   - w[2i] → 低字节, w[2i+1] → 高字节
 *   - scale = max(|w|) / 127（共用 scale）
 *
 * @param w        float 输入数组（长度 2n）
 * @param n        hc16ms_t 输出长度（实际存 2n 个 int8）
 * @param scale    量化 scale
 * @param out      输出 hc16ms_t 数组（长度 n，HC8 视角）
 */
void hc16ms_quantize_hc8(const float* w, uint32_t n,
                          float scale,
                          hc16ms_t* out);

/**
 * 从 float 数组推导 HC16 量化 scale
 *   scale = max(|w[i]|) / 32767，全零时返回 1.0
 */
float hc16ms_quant_compute_scale_hc16(const float* w, uint32_t n);

/**
 * 从 float 数组推导 HC8 量化 scale
 *   scale = max(|w[i]|) / 127，全零时返回 1.0
 */
float hc16ms_quant_compute_scale_hc8(const float* w, uint32_t n);

/* ============================================================================
 * 反量化：hc16ms_t → float（不同视角）
 * ============================================================================ */

/**
 * HC16 视角反量化：hc16ms_t → float（1 个 int16 → 1 个 float）
 *
 * @param h        hc16ms_t 输入数组（长度 n）
 * @param n        数组长度
 * @param scale    HC16 量化 scale
 * @param out      输出 float 数组（长度 n）
 */
void hc16ms_dequantize_hc16(const hc16ms_t* h, uint32_t n,
                             float scale,
                             float* out);

/**
 * HC8 视角反量化：hc16ms_t → float（2 个 int8 → 2 个 float）
 *
 * @param h        hc16ms_t 输入数组（长度 n）
 * @param n        hc16ms_t 数组长度
 * @param scale    HC8 量化 scale
 * @param out      输出 float 数组（长度 2n）
 */
void hc16ms_dequantize_hc8(const hc16ms_t* h, uint32_t n,
                            float scale,
                            float* out);

/* ============================================================================
 * 多视角 matmul
 * ============================================================================
 *
 * C = A @ B，矩阵组织按视角不同：
 *
 * HC16 视角:
 *   A: (m×k) hc16ms_t 数组，每个元素 1 个 int16
 *   B: (k×n) hc16ms_t 数组，每个元素 1 个 int16
 *   C: (m×n) float 输出
 *   运算: int16×int16→int64 累加 → 反量化
 *
 * HC8 视角:
 *   A: (m×k) hc16ms_t 数组，每个元素 2 个 int8 → 等效 (m×2k) int8 矩阵
 *   B: (k×n) hc16ms_t 数组，每个元素 2 个 int8 → 等效 (2k×n) int8 矩阵
 *   C: (m×n) float 输出
 *   运算: int8×int8→int32 累加 → 反量化
 *
 * 注意：HC8 视角下，A 的 2k 维必须等于 B 的 2k 维（自动满足，因为都来自 k 个 hc16ms_t）
 */

/**
 * HC16 视角 matmul（int16×int16→int64 累加）
 *
 * 复用 hc16_net.c 的 AVX2 _mm256_madd_epi16 路径。
 *
 * @param a            输入矩阵 A（m×k，hc16ms_t 数组，长度 m*k）
 * @param b            输入矩阵 B（k×n，hc16ms_t 数组，长度 k*n）
 * @param m, k, n      矩阵维度（hc16ms_t 元素数）
 * @param a_scale      A 的 HC16 量化 scale
 * @param b_scale      B 的 HC16 量化 scale
 * @param out          输出矩阵 C（m×n，float 数组，长度 m*n）
 */
void hc16ms_matmul_hc16(const hc16ms_t* a, const hc16ms_t* b,
                         uint32_t m, uint32_t k, uint32_t n,
                         float a_scale, float b_scale,
                         float* out);

/**
 * HC16 切换视角 matmul（先对 A、B 做 bswap16，再调用 hc16ms_matmul_hc16）
 *
 * 与正向 hc16ms_matmul_hc16 的区别：先把 A、B 每个 hc16ms_t 的 raw 做字节交换，
 * 然后用交换后的临时缓冲区调用正向 matmul。语义上等价于按相反字节序读取
 * 同一块内存再做 int16×int16 累加。
 *
 * @param a            输入矩阵 A（m×k，hc16ms_t 数组，长度 m*k）
 * @param b            输入矩阵 B（k×n，hc16ms_t 数组，长度 k*n）
 * @param m, k, n      矩阵维度（hc16ms_t 元素数）
 * @param a_scale      A 的 HC16 量化 scale
 * @param b_scale      B 的 HC16 量化 scale
 * @param out          输出矩阵 C（m×n，float 数组，长度 m*n）
 */
void hc16ms_matmul_hc16_swapped(const hc16ms_t* a, const hc16ms_t* b,
                                 uint32_t m, uint32_t k, uint32_t n,
                                 float a_scale, float b_scale, float* out);

/**
 * HC8 视角 matmul（int8×int8→int32 累加）
 *
 * 每个 hc16ms_t 展开为 2 个 int8，矩阵 A (m×k) → (m×2k) int8，B (k×n) → (2k×n) int8
 * 运算: int8×int8→int32 累加（k_effective = 2k），反量化
 *
 * @param a            输入矩阵 A（m×k，hc16ms_t 数组，长度 m*k）
 * @param b            输入矩阵 B（k×n，hc16ms_t 数组，长度 k*n）
 * @param m, k, n      矩阵维度（hc16ms_t 元素数）
 * @param a_scale      A 的 HC8 量化 scale
 * @param b_scale      B 的 HC8 量化 scale
 * @param out          输出矩阵 C（m×n，float 数组，长度 m*n）
 */
void hc16ms_matmul_hc8(const hc16ms_t* a, const hc16ms_t* b,
                        uint32_t m, uint32_t k, uint32_t n,
                        float a_scale, float b_scale,
                        float* out);

/* ============================================================================
 * 便捷封装：float → float 的多视角量化 matmul
 * ============================================================================ */

/**
 * HC16 视角量化 matmul（float → float，一步完成）
 *
 * @param x            输入矩阵 X（m×k，float 数组）
 * @param w            输入矩阵 W（k×n，float 数组）
 * @param m, k, n      矩阵维度
 * @param out          输出矩阵 Y（m×n，float 数组）
 */
void hc16ms_quantized_matmul_hc16(const float* x, const float* w,
                                   uint32_t m, uint32_t k, uint32_t n,
                                   float* out);

/**
 * HC8 视角量化 matmul（float → float，一步完成）
 *
 * 注意：HC8 视角下每个 hc16ms_t 存 2 个 int8，所以输入 float 数组长度为 2*m*k 和 2*k*n
 *
 * @param x            输入矩阵 X（m×2k，float 数组，长度 2*m*k）
 * @param w            输入矩阵 W（2k×n，float 数组，长度 2*k*n）
 * @param m, k, n      矩阵维度（k 是 hc16ms_t 维度，float 维度是 2k）
 * @param out          输出矩阵 Y（m×n，float 数组）
 */
void hc16ms_quantized_matmul_hc8(const float* x, const float* w,
                                  uint32_t m, uint32_t k, uint32_t n,
                                  float* out);

/* ============================================================================
 * 运行时检测
 * ============================================================================ */

/**
 * 检测 CPU 是否支持 AVX2
 * @return 1=支持, 0=不支持
 */
int hc16ms_detect_avx2(void);

#ifdef __cplusplus
}
#endif

#pragma pack(pop)

#endif /* SGN_HC16MS_H */
