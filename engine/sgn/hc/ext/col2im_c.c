#include "col2im_c.h"
#include <string.h>

/* col2im C 扩展（累加重叠区域）
 *
 * 等价于 numpy _col2im 的核心操作：
 *   for i in range(kh):
 *       for j in range(kw):
 *           x_padded[:, :, i:i+stride*H_out:stride, j:j+stride*W_out:stride] +=
 *               x_col_reshaped[:, :, i, j, :, :]
 *
 * 内存布局（均 C-contiguous）：
 *   x_col:     (B, C, kh, kw, H_out, W_out)
 *   x_padded:  (B, C, H_padded, W_padded)
 *
 * 对每个 (b, c, i, j, ho, wo)：
 *   x_padded[b, c, i + ho*stride, j + wo*stride] += x_col[b, c, i, j, ho, wo]
 *
 * OpenMP 并行化（batch x channel 维度），MSVC /openmp 支持 OpenMP 2.0
 * （循环变量需为 signed int，collapse 子句受限于 2.0 但 collapse(2) 支持）。
 */
void col2im_add_c(const float* x_col, float* x_padded,
                  int B, int C, int kh, int kw,
                  int H_out, int W_out, int stride,
                  int H_padded, int W_padded) {
    /* 预计算各维度步长（元素数） */
    const long BC = (long)B * (long)C;
    const int kh_kw_Hout_Wout = kh * kw * H_out * W_out;
    const int kw_Hout_Wout = kw * H_out * W_out;
    const int Hout_Wout = H_out * W_out;
    const int Hp_Wp = H_padded * W_padded;

    /* OpenMP 并行化 batch x channel 维度（扁平化 bc 索引，互不写冲突）。
     * MSVC /openmp 仅支持 OpenMP 2.0：循环变量需在外部声明为 signed 类型，
     * 不支持 collapse 子句。用 bc = b*C + c 扁平索引获得 B*C 并行度。 */
    long bc;
    #pragma omp parallel for schedule(static)
    for (bc = 0; bc < BC; ++bc) {
        const int b = (int)(bc / C);
        const int c = (int)(bc % C);
        {
            /* x_padded[b, c, :, :] 的起始指针 */
            float* dst_bc = x_padded + (size_t)b * C * Hp_Wp + (size_t)c * Hp_Wp;
            /* x_col[b, c, :, :, :, :] 的起始指针 */
            const float* src_bc = x_col + (size_t)b * C * kh_kw_Hout_Wout
                                        + (size_t)c * kh_kw_Hout_Wout;

            for (int i = 0; i < kh; ++i) {
                /* dst_bc + i*W_padded 指向 x_padded[b,c, i, :] */
                float* dst_i = dst_bc + (size_t)i * W_padded;
                const float* src_i = src_bc + (size_t)i * kw_Hout_Wout;

                for (int j = 0; j < kw; ++j) {
                    /* x_col[b, c, i, j, :, :] 的起始指针 */
                    const float* src_ij = src_i + (size_t)j * Hout_Wout;
                    /* 目标列起点偏移 j（行内偏移） */
                    float* dst_ij_col = dst_i + j;

                    if (stride == 1) {
                        /* stride==1 快速路径：目标行连续，可整行拷贝累加 */
                        for (int ho = 0; ho < H_out; ++ho) {
                            float* dst_row = dst_ij_col + (size_t)ho * W_padded;
                            const float* src_row = src_ij + (size_t)ho * W_out;
                            /* dst_row[0..W_out-1] += src_row[0..W_out-1] */
                            int wo = 0;
                            for (; wo + 4 <= W_out; wo += 4) {
                                dst_row[wo]     += src_row[wo];
                                dst_row[wo + 1] += src_row[wo + 1];
                                dst_row[wo + 2] += src_row[wo + 2];
                                dst_row[wo + 3] += src_row[wo + 3];
                            }
                            for (; wo < W_out; ++wo) {
                                dst_row[wo] += src_row[wo];
                            }
                        }
                    } else {
                        /* stride>1：目标列按 stride 步进 */
                        for (int ho = 0; ho < H_out; ++ho) {
                            float* dst_row = dst_ij_col
                                           + (size_t)ho * stride * W_padded;
                            const float* src_row = src_ij + (size_t)ho * W_out;
                            for (int wo = 0; wo < W_out; ++wo) {
                                dst_row[wo * stride] += src_row[wo];
                            }
                        }
                    }
                }
            }
        }
    }
}
