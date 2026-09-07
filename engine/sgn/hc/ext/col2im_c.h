#ifndef COL2IM_C_H
#define COL2IM_C_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* col2im C 扩展（累加重叠区域）
 *
 * 等价于 numpy _col2im 的核心操作：
 *   for i in range(kh):
 *       for j in range(kw):
 *           x_padded[:, :, i:i+stride*H_out:stride, j:j+stride*W_out:stride] +=
 *               x_col_reshaped[:, :, i, j, :, :]
 *
 * 参数：
 *   x_col: (B, C, kh, kw, H_out, W_out) 已 reshape 的梯度列矩阵（C-contiguous）
 *   x_padded: (B, C, H_padded, W_padded) 输出（累加到已有值，C-contiguous）
 *   B, C, kh, kw, H_out, W_out, stride, H_padded, W_padded: 维度参数
 */
void col2im_add_c(const float* x_col, float* x_padded,
                  int B, int C, int kh, int kw,
                  int H_out, int W_out, int stride,
                  int H_padded, int W_padded);

#ifdef __cplusplus
}
#endif

#endif /* COL2IM_C_H */
