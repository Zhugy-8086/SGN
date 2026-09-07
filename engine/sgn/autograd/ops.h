// ops.h - 算子声明（matmul forward / backward）
//
// Stage 3.2 Phase 1：matmul 算子（朴素 C++ 实现，float32）。
// AVX2 优化预留接口待性能优化阶段补充。
//
// 数学定义：
//   forward:  C = A @ B,  C[i,j] = sum_l A[i,l] * B[l,j]
//   backward: dA = dY @ B^T,  dB = A^T @ dY

#pragma once

#include "tensor.h"

#include <utility>

namespace sgn_autograd {

// matmul forward: C = A @ B
//   A: (m, k), B: (k, n), C: (m, n)
// 要求 A 和 B 都是 2D contiguous Tensor。
Tensor matmul_forward(const Tensor& A, const Tensor& B);

// matmul backward:
//   dA = dY @ B^T   (dA: m×k, dY: m×n, B: k×n → B^T: n×k)
//     dA[i,l] = sum_j dY[i,j] * B[l,j]
//   dB = A^T @ dY   (dB: k×n, A: m×k → A^T: k×m, dY: m×n)
//     dB[l,j] = sum_i A[i,l] * dY[i,j]
// 返回 {dA, dB}。
std::pair<Tensor, Tensor> matmul_backward(
    const Tensor& A, const Tensor& B, const Tensor& dY
);

// matmul forward with transpose_b: C = A @ B^T
//   A: (m, k), B: (n, k), C: (m, n)
//   C(i,j) = sum_l A(i,l) * B(j,l)
// B 以原始布局 (n,k) 传入，无需显式转置拷贝。
// 要求 A 和 B 都是 2D contiguous Tensor。
Tensor matmul_forward_transpose_b(const Tensor& A, const Tensor& B);

// matmul forward with transpose_a: C = A^T @ B
//   A: (m, k), B: (m, n), C: (k, n)
//   C(i,j) = sum_l A(l,i) * B(l,j)
// A 以原始布局 (m,k) 传入，无需显式转置拷贝。
// 要求 A 和 B 都是 2D contiguous Tensor。
Tensor matmul_forward_transpose_a(const Tensor& A, const Tensor& B);

// 辅助：转置 2D Tensor（返回新的 contiguous Tensor，非 view）
//   x: (r, c) → 返回 (c, r)，result[j,i] = x[i,j]
Tensor transpose_2d(const Tensor& x);

}  // namespace sgn_autograd
