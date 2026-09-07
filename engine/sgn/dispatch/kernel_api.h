// kernel_api.h - 平台无关内核接口（纯声明，无实现）
//
// 设计背景：内部调研文档（conv2d backward im2col/col2im/transpose 优化调研）
//   节奏 0：搭接口骨架，不改任何内核逻辑。
//
// 本文件只声明签名与契约，不含任何实现。算子层（ops.cpp / ops_nn.cpp）
// 只 include 本文件并通过 registry 调用端口，禁止 include 任何平台头
// （immintrin.h / target 属性 / __builtin_cpu_supports）——平台细节封死
// 在后端目录内，避免复制粘贴改内核（见调研文档 §2）。
//
// 契约（所有端口共同遵守，勿在实现中违背）：
//   1. 输入/输出均为行主序 float；所有尺寸用 int64_t；
//   2. 输出缓冲由【调用方】分配/持有，后端只写、不拥有、不得缓存指针；
//   3. col2im 是累加语义：dX_pad 调用前必须已清零（调用方保证，后端不负责）；
//   4. 除特别标注外，不假设任意指针对齐（统一 loadu/storeu）；
//   5. 端口默认【不抛异常】；GPU 后端除外（CUDA 失败映射异常）；
//   6. 形状/contiguous/清零等校验是【算子层】职责——后端不做重复校验
//      （防每后端复制粘贴校验代码），后端可带 assert 级 debug 断言；
//   7. 并行是【后端内部】职责：内核内自行决定 OpenMP 划分（见调研文档 §6.4）。
//      调用方【禁止】在自己的 parallel 域内直接调端口——嵌套并行会线程
//      爆炸（或被 omp_get_max_threads 截断后性能崩塌）。
//
// 数值档位：BITEXACT = 与 ref_scalar 逐位一致；ROUNDING = 舍入级差异（有声明）。

#pragma once

#include "autograd/tensor.h"

#include <cstdint>
#include <utility>

namespace sgn_autograd {

enum class NumLevel : uint8_t { kBitExact, kRounding };

// ============================================================================
// matmul 族端口（节奏 0 接线现有内核；沿用 Tensor 式签名以不改内核逻辑）
// 注意：后续节奏加入 conv2d 族端口（dw/dx/db/dycol/col2im/im2col）时，将按
// 契约改用裸指针 + int64 尺寸签名（见调研文档 §1.1），并放入各算子族 KernelSet。
// ============================================================================

// C = A @ B（Tensor 式：A/B/C 均由调用方持有的 Tensor 承载）
using MatmulFwdFn = Tensor (*)(const Tensor& A, const Tensor& B);

// C = A @ B^T
using MatmulTransposeBFn = Tensor (*)(const Tensor& A, const Tensor& B);

// C = A^T @ B
using MatmulTransposeAFn = Tensor (*)(const Tensor& A, const Tensor& B);

// {dA, dB} = {dY @ B^T, A^T @ dY}
using MatmulBackwardFn = std::pair<Tensor, Tensor> (*)(
    const Tensor& A, const Tensor& B, const Tensor& dY);

// 一个算子族的内核集合 + 数值档位声明（matmul 族，节奏 0 起；
// conv2d 族独立 KernelSet 见下，节奏 1 起逐步迁入）
struct KernelSet {
    MatmulFwdFn       matmul_fwd;
    MatmulTransposeBFn matmul_transpose_b;
    MatmulTransposeAFn matmul_transpose_a;
    MatmulBackwardFn  matmul_backward;
    NumLevel          num_level;   // 当前活跃后端的数值档位
    const char*       name;        // 后端名（诊断用，Python 侧可查询）
};

// ============================================================================
// conv2d 族端口（裸指针 + int64 尺寸，符合契约 §1.1；输出缓冲由调用方分配）
// 节奏 1：先迁入 dW 端口（dw_scalar / dw_avx2），其余 5 端口暂为 stub（调用即抛）。
// 数值档位：dW 的 avx2 内核采用"8 路并行累加 + 末尾横向归约"（l 步长 8），
//           与标量串行归约次序不同 → avx2 标 kRounding；scalar 为 bit-exact 锚点。
// ============================================================================

// dW：C[oc, k] = sum_l A[oc, l] * B[k, l]   （A=dY_col (out_c×bs), B=x_col (k×bs)）
// 注意：§1.1 的 DwFn 参数名 out_c/bs/k 与本注释中 k=in_c*kh*kw 的"k"为同一量纲
//       （B 的行数 = in_c*kh*kw）。
using DwFn = void (*)(const float* A, const float* B, float* C,
                      int64_t out_c, int64_t bs, int64_t k);

// dx_col：C[r, c] = sum_l A[l, r] * B[l, c] （A=W_col^T (r×out_c), B=dY_col (out_c×bs)）
using DxFn = void (*)(const float* A, const float* B, float* C,
                      int64_t r, int64_t out_c, int64_t bs);

// db 行和：db[oc] = sum_l dY_col[oc, l]（反向步骤 ③）
using DbSumFn = void (*)(const float* dY_col, float* db,
                         int64_t out_c, int64_t bs);

// dY→dY_col 转置（反向步骤 ①）：dY(b,oc,s) → dY_col(oc, b·s)（纯搬运）
using TransposeBcsFn = void (*)(const float* dY, float* dY_col,
                                int64_t batch, int64_t out_c, int64_t spatial);

// col2im（零填充缓冲版）：dx_col(r×bs) 累加进 dX_pad（调用前必须清零，见契约 3）
// 所有维度显式传入；padded_h = in_h + 2*pad
using Col2ImFn = void (*)(const float* dx_col, float* dX_pad,
                          int64_t batch, int64_t in_c,
                          int64_t out_h, int64_t out_w,
                          int64_t bs, int64_t kh, int64_t kw,
                          int64_t stride, int64_t pad,
                          int64_t padded_h, int64_t padded_w);

// im2col（前向）：X(b,in_c,in_h,in_w) → x_col(r×bs)，越界填 0
using Im2ColFn = void (*)(const float* X, float* x_col,
                          int64_t batch, int64_t in_c,
                          int64_t in_h, int64_t in_w,
                          int64_t out_h, int64_t out_w,
                          int64_t kh, int64_t kw, int64_t stride, int64_t pad);

// conv2d 族内核集合（节奏 1 仅 dw 已实现，其余 stub 占位显式暴露）
struct Conv2dKernelSet {
    DwFn           dw;
    DxFn           dx;
    DbSumFn        db;
    TransposeBcsFn dycol;
    Col2ImFn       col2im;
    Im2ColFn       im2col;
    NumLevel       num_level;
    const char*    name;
};

}  // namespace sgn_autograd
