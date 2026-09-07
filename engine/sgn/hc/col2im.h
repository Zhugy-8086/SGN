// col2im.h - HC 扩展 C->C++ 迁移模板（col2im）
//
// Stage 3.0 Task 1.3 第一步：将 col2im 从 C 迁移到 C++，作为 HC 扩展
// (hc8_net / hc16_net / hc4_pshufb / hc16ms) C->C++ 迁移的最小验证模板。
//
// 设计要点（供后续迁移复用）：
//   1. 保留 extern "C" 接口（向后兼容已有 C 调用方）
//   2. 用 C++ 类封装核心逻辑（统一抽象层入口）
//   3. 底层实现复用原始 C 代码（见 col2im.cpp 通过相对路径引用原 C 源）
//   4. 头文件自包含，不依赖原 C 头文件（避免中文路径传染到绑定层）

#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

/* col2im C 扩展（累加重叠区域）- 与原始 col2im_c.h 接口完全一致
 *
 * 等价于 numpy _col2im 的核心操作：
 *   for i in range(kh):
 *       for j in range(kw):
 *           x_padded[:, :, i:i+stride*H_out:stride, j:j+stride*W_out:stride] +=
 *               x_col_reshaped[:, :, i, j, :, :]
 *
 * 参数：
 *   x_col:     (B, C, kh, kw, H_out, W_out) 已 reshape 的梯度列矩阵（C-contiguous）
 *   x_padded:  (B, C, H_padded, W_padded) 输出（累加到已有值，C-contiguous）
 *   B, C, kh, kw, H_out, W_out, stride, H_padded, W_padded: 维度参数
 */
void col2im_add_c(const float* x_col, float* x_padded,
                  int B, int C, int kh, int kw,
                  int H_out, int W_out, int stride,
                  int H_padded, int W_padded);

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus

/* HC 迁移封装类
 *
 * Col2im 提供 C++ 接口，内部委托给 col2im_add_c（原始 C 实现）。
 * 保留 extern "C" 函数 col2im_add_c 以便向后兼容直接调用 C 接口的代码。 */
class Col2im {
public:
    Col2im() = default;

    /* 执行 col2im scatter-add（原地累加到 x_padded）
     *
     * 数学等价：
     *   x_padded[b, c, i + ho*stride, j + wo*stride] += x_col[b, c, i, j, ho, wo]
     *
     * Args:
     *   x_col:     (B, C, kh, kw, H_out, W_out) C-contiguous float32
     *   x_padded:  (B, C, H_padded, W_padded) C-contiguous float32（原地修改）
     *   其余为维度参数
     *
     * Note: x_padded 必须预先清零或包含需要累加的初始值（执行 += 而非 =）。 */
    void execute(const float* x_col, float* x_padded,
                 int B, int C, int kh, int kw,
                 int H_out, int W_out, int stride,
                 int H_padded, int W_padded);
};

#endif
