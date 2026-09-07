// ops.cpp - matmul 算子实现（float32）
//
// Stage 3.2 Phase 1：朴素三重循环实现，i-k-j 顺序（缓存友好）。
// 节奏 0（2026-08-16）：dispatch 逻辑迁至 dispatch/registry.cpp，本文件只保留
// 内核实现 + 薄适配函数（*_scalar / *_avx2），公开入口经 kernel_registry()
// 一次性函数指针调用端口（内核逻辑不变，见调研文档 §2）。
//
// 实现说明：
//   - 所有算子要求输入为 2D contiguous Tensor，否则抛 std::invalid_argument
//   - 内部直接用裸指针访问，避免 at() 的边界检查开销
//   - 行优先布局：x[i,j] 位于 data()[i * ncols + j]
//   - 平台细节（CPU 能力检测 / 后端选择）全部封死在 dispatch/registry.cpp，
//     本文件禁止 include 平台头后散落 if-else 分支

#include "ops.h"

#include "dispatch/registry.h"

#include <immintrin.h>
#include <stdexcept>

namespace sgn_autograd {

// ============================================================================
// AVX2 内核前向声明（供后端适配函数引用；定义见下文各处）
// ============================================================================
Tensor matmul_fwd_transpose_b_avx2_fma(const Tensor& A, const Tensor& B);
Tensor matmul_fwd_transpose_b_avx2_fma_small_k(const Tensor& A, const Tensor& B);
Tensor matmul_fwd_transpose_a_avx2_fma(const Tensor& A, const Tensor& B);
Tensor matmul_fwd_transpose_a_avx2_fma_small_k(const Tensor& A, const Tensor& B);
Tensor matmul_bwd_dA_avx2_fma(const Tensor& dY, const Tensor& B);
Tensor matmul_bwd_dB_avx2_fma(const Tensor& A, const Tensor& dY);

// ============================================================================
// AVX2 FMA: float32 matmul forward with k-tiling + register blocking, 8-wide
// C = A @ B, A: (m,k), B: (k,n), C: (m,n)
//
// 寄存器分块策略：
//   - 外层 k-tiling (T_k=64) 保证 B tile 在 L1/L2 cache 中命中
//   - 内层 i-blocking (I_BLOCK=4)：一次处理 4 行 i，将 C 的 4×8 块保持在
//     4 个 YMM 寄存器中，跨 k-tile 的 l 循环累加
//   - 循环顺序：l0→i0→j→l，B 只加载一次供 4 行 C 共享
//   - 寄存器使用：4 (C) + 4 (A broadcasts) + 1 (B) = 9/16 YMM
// ============================================================================
Tensor matmul_fwd_avx2_fma(const Tensor& A, const Tensor& B) {
    int64_t m = A.shape()[0], k = A.shape()[1], n = B.shape()[1];
    Tensor C({m, n});
    const float* a_ptr = A.data();
    const float* b_ptr = B.data();
    float* c_ptr = C.data();

    const int64_t T_k = 64;     // k-tile size
    const int64_t I_BLOCK = 4;  // register blocking: 4 rows of C in YMM regs

    // k-tiling: keep B[tile, :] in L1/L2 cache
    for (int64_t l0 = 0; l0 < k; l0 += T_k) {
        int64_t l_end = (l0 + T_k < k) ? l0 + T_k : k;

        // Register-blocked i loop: process I_BLOCK rows at once
        // C[i0:i0+I_BLOCK, j:j+8] stays in YMM registers across the l loop
        const int64_t i_blocked = (m / I_BLOCK) * I_BLOCK;
        // OpenMP: i 块独立写 C，k-tile 内串行累加 → 逐位一致（2026-08-16）
        #pragma omp parallel for schedule(guided) if(m * k * n >= (1 << 20))
        for (int64_t i0 = 0; i0 < i_blocked; i0 += I_BLOCK) {
            // j-l ordering: j is outer, l is inner
            // This keeps C[i0:i0+4, j:j+8] in registers across the k-tile
            int64_t j = 0;
            for (; j + 8 <= n; j += 8) {
                // Load C registers for I_BLOCK rows (1 load per row)
                __m256 c0 = _mm256_loadu_ps(&c_ptr[i0 * n + j]);
                __m256 c1 = _mm256_loadu_ps(&c_ptr[(i0 + 1) * n + j]);
                __m256 c2 = _mm256_loadu_ps(&c_ptr[(i0 + 2) * n + j]);
                __m256 c3 = _mm256_loadu_ps(&c_ptr[(i0 + 3) * n + j]);

                // Accumulate over the k-tile
                for (int64_t l = l0; l < l_end; ++l) {
                    // B[l, j:j+8] loaded once, shared by all I_BLOCK rows
                    __m256 b = _mm256_loadu_ps(&b_ptr[l * n + j]);
                    __m256 a0 = _mm256_set1_ps(a_ptr[i0 * k + l]);
                    __m256 a1 = _mm256_set1_ps(a_ptr[(i0 + 1) * k + l]);
                    __m256 a2 = _mm256_set1_ps(a_ptr[(i0 + 2) * k + l]);
                    __m256 a3 = _mm256_set1_ps(a_ptr[(i0 + 3) * k + l]);
                    c0 = _mm256_fmadd_ps(a0, b, c0);
                    c1 = _mm256_fmadd_ps(a1, b, c1);
                    c2 = _mm256_fmadd_ps(a2, b, c2);
                    c3 = _mm256_fmadd_ps(a3, b, c3);
                }

                // Store C registers (1 store per row, after all l in tile)
                _mm256_storeu_ps(&c_ptr[i0 * n + j], c0);
                _mm256_storeu_ps(&c_ptr[(i0 + 1) * n + j], c1);
                _mm256_storeu_ps(&c_ptr[(i0 + 2) * n + j], c2);
                _mm256_storeu_ps(&c_ptr[(i0 + 3) * n + j], c3);
            }

            // Remainder j columns (scalar, n % 8)
            // Process I_BLOCK rows at once with scalar accumulation
            for (; j < n; ++j) {
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                for (int64_t l = l0; l < l_end; ++l) {
                    float b_val = b_ptr[l * n + j];
                    s0 += a_ptr[i0 * k + l] * b_val;
                    s1 += a_ptr[(i0 + 1) * k + l] * b_val;
                    s2 += a_ptr[(i0 + 2) * k + l] * b_val;
                    s3 += a_ptr[(i0 + 3) * k + l] * b_val;
                }
                c_ptr[i0 * n + j] += s0;
                c_ptr[(i0 + 1) * n + j] += s1;
                c_ptr[(i0 + 2) * n + j] += s2;
                c_ptr[(i0 + 3) * n + j] += s3;
            }
        }

        // Remainder i rows (single-row processing, original i-k-j style)
        for (int64_t i = i_blocked; i < m; ++i) {
            for (int64_t l = l0; l < l_end; ++l) {
                float a_val = a_ptr[i * k + l];
                __m256 a_reg = _mm256_set1_ps(a_val);
                int64_t j = 0;
                for (; j + 8 <= n; j += 8) {
                    __m256 b_vec = _mm256_loadu_ps(&b_ptr[l * n + j]);
                    __m256 c_vec = _mm256_loadu_ps(&c_ptr[i * n + j]);
                    _mm256_storeu_ps(&c_ptr[i * n + j], _mm256_fmadd_ps(a_reg, b_vec, c_vec));
                }
                for (; j < n; ++j) {
                    c_ptr[i * n + j] += a_val * b_ptr[l * n + j];
                }
            }
        }
    }
    return C;
}

// ============================================================================
// AVX2 FMA: float32 matmul forward, small-K optimized path
// C = A @ B, A: (m,k), B: (k,n), C: (m,n)
//
// 针对小 K 维度（K ≤ 128）优化：去掉 k-tiling 外层循环，减少 I_BLOCK 到 2，
// 降低寄存器压力并减少尾部行开销。
// ============================================================================
Tensor matmul_fwd_avx2_fma_small_k(const Tensor& A, const Tensor& B) {
    int64_t m = A.shape()[0], k = A.shape()[1], n = B.shape()[1];
    Tensor C({m, n});
    const float* a_ptr = A.data();
    const float* b_ptr = B.data();
    float* c_ptr = C.data();

    const int64_t I_BLOCK = 2;

    // K < 8: fallback to scalar triple loop
    if (k < 8) {
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t l = 0; l < k; ++l) {
                float a_il = a_ptr[i * k + l];
                for (int64_t j = 0; j < n; ++j) {
                    c_ptr[i * n + j] += a_il * b_ptr[l * n + j];
                }
            }
        }
        return C;
    }

    // K >= 8: AVX2 FMA, no k-tiling, I_BLOCK=2
    const int64_t i_blocked = (m / I_BLOCK) * I_BLOCK;
    // OpenMP: i 块独立写 C → 逐位一致（2026-08-16）
    #pragma omp parallel for schedule(guided) if(m * k * n >= (1 << 20))
    for (int64_t i0 = 0; i0 < i_blocked; i0 += I_BLOCK) {
        int64_t j = 0;
        for (; j + 8 <= n; j += 8) {
            __m256 c0 = _mm256_loadu_ps(&c_ptr[i0 * n + j]);
            __m256 c1 = _mm256_loadu_ps(&c_ptr[(i0 + 1) * n + j]);

            for (int64_t l = 0; l < k; ++l) {
                __m256 b = _mm256_loadu_ps(&b_ptr[l * n + j]);
                __m256 a0 = _mm256_set1_ps(a_ptr[i0 * k + l]);
                __m256 a1 = _mm256_set1_ps(a_ptr[(i0 + 1) * k + l]);
                c0 = _mm256_fmadd_ps(a0, b, c0);
                c1 = _mm256_fmadd_ps(a1, b, c1);
            }

            _mm256_storeu_ps(&c_ptr[i0 * n + j], c0);
            _mm256_storeu_ps(&c_ptr[(i0 + 1) * n + j], c1);
        }

        // Remainder j columns (scalar)
        for (; j < n; ++j) {
            float s0 = 0.0f, s1 = 0.0f;
            for (int64_t l = 0; l < k; ++l) {
                float b_val = b_ptr[l * n + j];
                s0 += a_ptr[i0 * k + l] * b_val;
                s1 += a_ptr[(i0 + 1) * k + l] * b_val;
            }
            c_ptr[i0 * n + j] += s0;
            c_ptr[(i0 + 1) * n + j] += s1;
        }
    }

    // Remainder i rows (single-row, SIMD)
    for (int64_t i = i_blocked; i < m; ++i) {
        for (int64_t l = 0; l < k; ++l) {
            float a_val = a_ptr[i * k + l];
            __m256 a_reg = _mm256_set1_ps(a_val);
            int64_t j = 0;
            for (; j + 8 <= n; j += 8) {
                __m256 b_vec = _mm256_loadu_ps(&b_ptr[l * n + j]);
                __m256 c_vec = _mm256_loadu_ps(&c_ptr[i * n + j]);
                _mm256_storeu_ps(&c_ptr[i * n + j], _mm256_fmadd_ps(a_reg, b_vec, c_vec));
            }
            for (; j < n; ++j) {
                c_ptr[i * n + j] += a_val * b_ptr[l * n + j];
            }
        }
    }

    return C;
}

// ============================================================================
// AVX-512 FMA: float32 matmul forward, 16-wide
// ============================================================================
#ifdef __AVX512F__
__attribute__((target("avx512f,fma")))
Tensor matmul_fwd_avx512f(const Tensor& A, const Tensor& B) {
    int64_t m = A.shape()[0], k = A.shape()[1], n = B.shape()[1];
    Tensor C({m, n});
    const float* a_ptr = A.data();
    const float* b_ptr = B.data();
    float* c_ptr = C.data();

    const int64_t T_k = 128;  // AVX-512: 16-wide, larger tile to amortize 512-bit FMA overhead

    for (int64_t l0 = 0; l0 < k; l0 += T_k) {
        int64_t l_end = (l0 + T_k < k) ? l0 + T_k : k;
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t l = l0; l < l_end; ++l) {
                __m512 a_reg = _mm512_set1_ps(a_ptr[i * k + l]);
                int64_t j = 0;
                for (; j + 16 <= n; j += 16) {
                    __m512 b_vec = _mm512_loadu_ps(&b_ptr[l * n + j]);
                    __m512 c_vec = _mm512_loadu_ps(&c_ptr[i * n + j]);
                    _mm512_storeu_ps(&c_ptr[i * n + j], _mm512_fmadd_ps(a_reg, b_vec, c_vec));
                }
                for (; j < n; ++j) {
                    c_ptr[i * n + j] += a_ptr[i * k + l] * b_ptr[l * n + j];
                }
            }
        }
    }
    return C;
}
#endif  // __AVX512F__

// ============================================================================
// 标量后端（ref_scalar，bit-exact 锚点）
// 供 dispatch/registry.cpp 通过函数指针调用
// ============================================================================

// C = A @ B, scalar i-k-j
Tensor matmul_fwd_scalar(const Tensor& A, const Tensor& B) {
    int64_t m = A.shape()[0], k = A.shape()[1], n = B.shape()[1];
    Tensor C({m, n});
    const float* a_ptr = A.data();
    const float* b_ptr = B.data();
    float* c_ptr = C.data();
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t l = 0; l < k; ++l) {
            float a_il = a_ptr[i * k + l];
            for (int64_t j = 0; j < n; ++j) {
                c_ptr[i * n + j] += a_il * b_ptr[l * n + j];
            }
        }
    }
    return C;
}

// C = A @ B^T, scalar
Tensor matmul_transpose_b_scalar(const Tensor& A, const Tensor& B) {
    int64_t m = A.shape()[0], k = A.shape()[1], n = B.shape()[0];
    Tensor C({m, n});
    const float* a_ptr = A.data();
    const float* b_ptr = B.data();
    float* c_ptr = C.data();
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int64_t l = 0; l < k; ++l) {
                sum += a_ptr[i * k + l] * b_ptr[j * k + l];
            }
            c_ptr[i * n + j] = sum;
        }
    }
    return C;
}

// C = A^T @ B, scalar
Tensor matmul_transpose_a_scalar(const Tensor& A, const Tensor& B) {
    int64_t m = A.shape()[0], k = A.shape()[1], n = B.shape()[1];
    Tensor C({k, n});
    const float* a_ptr = A.data();
    const float* b_ptr = B.data();
    float* c_ptr = C.data();
    for (int64_t i = 0; i < k; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int64_t l = 0; l < m; ++l) {
                sum += a_ptr[l * k + i] * b_ptr[l * n + j];
            }
            c_ptr[i * n + j] = sum;
        }
    }
    return C;
}

// {dA, dB} = {dY @ B^T, A^T @ dY}, scalar
std::pair<Tensor, Tensor> matmul_backward_scalar(
    const Tensor& A, const Tensor& B, const Tensor& dY) {
    int64_t m = A.shape()[0], k = A.shape()[1], n = B.shape()[1];
    const float* dy_ptr = dY.data();
    const float* b_ptr = B.data();
    const float* a_ptr = A.data();

    Tensor dA({m, k});
    float* da_ptr = dA.data();
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t l = 0; l < k; ++l) {
            float sum = 0.0f;
            for (int64_t j = 0; j < n; ++j) {
                sum += dy_ptr[i * n + j] * b_ptr[l * n + j];
            }
            da_ptr[i * k + l] = sum;
        }
    }

    Tensor dB({k, n});
    float* db_ptr = dB.data();
    for (int64_t l = 0; l < k; ++l) {
        for (int64_t j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int64_t i = 0; i < m; ++i) {
                sum += a_ptr[i * k + l] * dy_ptr[i * n + j];
            }
            db_ptr[l * n + j] = sum;
        }
    }

    return {dA, dB};
}

// ============================================================================
// AVX2 后端适配（含小 k 选择 / dA+dB 组合）
// 供 dispatch/registry.cpp 通过函数指针调用
// ============================================================================

Tensor matmul_fwd_avx2(const Tensor& A, const Tensor& B) {
    const int64_t k = A.shape()[1];
    if (k <= 128) return matmul_fwd_avx2_fma_small_k(A, B);
    return matmul_fwd_avx2_fma(A, B);
}

Tensor matmul_transpose_b_avx2(const Tensor& A, const Tensor& B) {
    const int64_t k = A.shape()[1];
    if (k <= 128) return matmul_fwd_transpose_b_avx2_fma_small_k(A, B);
    return matmul_fwd_transpose_b_avx2_fma(A, B);
}

Tensor matmul_transpose_a_avx2(const Tensor& A, const Tensor& B) {
    const int64_t reduction = A.shape()[0];
    if (reduction <= 128) return matmul_fwd_transpose_a_avx2_fma_small_k(A, B);
    return matmul_fwd_transpose_a_avx2_fma(A, B);
}

std::pair<Tensor, Tensor> matmul_backward_avx2(
    const Tensor& A, const Tensor& B, const Tensor& dY) {
    return {matmul_bwd_dA_avx2_fma(dY, B), matmul_bwd_dB_avx2_fma(A, dY)};
}

// ============================================================================
// matmul forward: C = A @ B
// ============================================================================
Tensor matmul_forward(const Tensor& A, const Tensor& B) {
    if (A.ndim() != 2 || B.ndim() != 2) {
        throw std::invalid_argument("matmul_forward: inputs must be 2D");
    }
    if (!A.is_contiguous() || !B.is_contiguous()) {
        throw std::invalid_argument("matmul_forward: inputs must be contiguous");
    }

    int64_t k = A.shape()[1];
    int64_t k2 = B.shape()[0];

    if (k != k2) {
        throw std::invalid_argument("matmul_forward: inner dimension mismatch");
    }

    const KernelSet& ks = kernel_registry();
    return ks.matmul_fwd(A, B);
}

// ============================================================================
// AVX2 FMA: matmul forward with transpose_b: C = A @ B^T
// A: (m, k), B: (n, k), C: (m, n)
// C(i,j) = sum_l A(i,l) * B(j,l)
//
// B 以原始布局 (n,k) 传入，B(j,l) 在地址 B[j*k + l] 处，步长 k。
// 用 AVX2 gather (_mm256_i32gather_ps) 沿 B 的列方向读取。
// ============================================================================
Tensor matmul_fwd_transpose_b_avx2_fma(const Tensor& A, const Tensor& B) {
    int64_t m = A.shape()[0], k = A.shape()[1], n = B.shape()[0];
    Tensor C({m, n});
    const float* a_ptr = A.data();
    const float* b_ptr = B.data();
    float* c_ptr = C.data();

    const int64_t T_k = 64;
    const int64_t I_BLOCK = 4;

    for (int64_t l0 = 0; l0 < k; l0 += T_k) {
        int64_t l_end = (l0 + T_k < k) ? l0 + T_k : k;

        const int64_t i_blocked = (m / I_BLOCK) * I_BLOCK;
        // OpenMP: i 块独立写 C，k-tile 内串行累加 → 逐位一致（2026-08-16）
        #pragma omp parallel for schedule(guided) if(m * k * n >= (1 << 20))
        for (int64_t i0 = 0; i0 < i_blocked; i0 += I_BLOCK) {
            int64_t j = 0;
            for (; j + 8 <= n; j += 8) {
                __m256 c0 = _mm256_loadu_ps(&c_ptr[i0 * n + j]);
                __m256 c1 = _mm256_loadu_ps(&c_ptr[(i0 + 1) * n + j]);
                __m256 c2 = _mm256_loadu_ps(&c_ptr[(i0 + 2) * n + j]);
                __m256 c3 = _mm256_loadu_ps(&c_ptr[(i0 + 3) * n + j]);

                // Precompute base offsets for B[j:j+8, l]
                // B is (n, k), B[j, l] at B[j*k + l]
                // vindex[i] = (j+i)*k + l, base_offs = (j+i)*k
                __m256i base_offs = _mm256_setr_epi32(
                    (int)(j * k), (int)((j + 1) * k),
                    (int)((j + 2) * k), (int)((j + 3) * k),
                    (int)((j + 4) * k), (int)((j + 5) * k),
                    (int)((j + 6) * k), (int)((j + 7) * k));

                for (int64_t l = l0; l < l_end; ++l) {
                    __m256 a0 = _mm256_set1_ps(a_ptr[i0 * k + l]);
                    __m256 a1 = _mm256_set1_ps(a_ptr[(i0 + 1) * k + l]);
                    __m256 a2 = _mm256_set1_ps(a_ptr[(i0 + 2) * k + l]);
                    __m256 a3 = _mm256_set1_ps(a_ptr[(i0 + 3) * k + l]);

                    __m256i lv = _mm256_set1_epi32((int)l);
                    __m256i vidx = _mm256_add_epi32(base_offs, lv);
                    __m256 b = _mm256_i32gather_ps((const void*)b_ptr, vidx, 4);

                    c0 = _mm256_fmadd_ps(a0, b, c0);
                    c1 = _mm256_fmadd_ps(a1, b, c1);
                    c2 = _mm256_fmadd_ps(a2, b, c2);
                    c3 = _mm256_fmadd_ps(a3, b, c3);
                }

                _mm256_storeu_ps(&c_ptr[i0 * n + j], c0);
                _mm256_storeu_ps(&c_ptr[(i0 + 1) * n + j], c1);
                _mm256_storeu_ps(&c_ptr[(i0 + 2) * n + j], c2);
                _mm256_storeu_ps(&c_ptr[(i0 + 3) * n + j], c3);
            }

            // Remainder j columns (scalar)
            for (; j < n; ++j) {
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                for (int64_t l = l0; l < l_end; ++l) {
                    float b_val = b_ptr[j * k + l];
                    s0 += a_ptr[i0 * k + l] * b_val;
                    s1 += a_ptr[(i0 + 1) * k + l] * b_val;
                    s2 += a_ptr[(i0 + 2) * k + l] * b_val;
                    s3 += a_ptr[(i0 + 3) * k + l] * b_val;
                }
                c_ptr[i0 * n + j] += s0;
                c_ptr[(i0 + 1) * n + j] += s1;
                c_ptr[(i0 + 2) * n + j] += s2;
                c_ptr[(i0 + 3) * n + j] += s3;
            }
        }

        // Remainder i rows (single-row, gather-based)
        for (int64_t i = i_blocked; i < m; ++i) {
            int64_t j = 0;
            for (; j + 8 <= n; j += 8) {
                __m256i base_offs = _mm256_setr_epi32(
                    (int)(j * k), (int)((j + 1) * k),
                    (int)((j + 2) * k), (int)((j + 3) * k),
                    (int)((j + 4) * k), (int)((j + 5) * k),
                    (int)((j + 6) * k), (int)((j + 7) * k));
                for (int64_t l = l0; l < l_end; ++l) {
                    __m256 a_reg = _mm256_set1_ps(a_ptr[i * k + l]);
                    __m256i lv = _mm256_set1_epi32((int)l);
                    __m256i vidx = _mm256_add_epi32(base_offs, lv);
                    __m256 b_vec = _mm256_i32gather_ps((const void*)b_ptr, vidx, 4);
                    __m256 c_vec = _mm256_loadu_ps(&c_ptr[i * n + j]);
                    _mm256_storeu_ps(&c_ptr[i * n + j],
                                     _mm256_fmadd_ps(a_reg, b_vec, c_vec));
                }
            }
            for (; j < n; ++j) {
                for (int64_t l = l0; l < l_end; ++l) {
                    c_ptr[i * n + j] += a_ptr[i * k + l] * b_ptr[j * k + l];
                }
            }
        }
    }
    return C;
}

// ============================================================================
// AVX2 FMA: matmul forward with transpose_b, small-K optimized path
// C = A @ B^T, A: (m,k), B: (n,k), C: (m,n)
// C(i,j) = sum_l A(i,l) * B(j,l)
//
// 针对小 K 维度（K ≤ 128）优化：去掉 k-tiling，I_BLOCK=2。
// ============================================================================
Tensor matmul_fwd_transpose_b_avx2_fma_small_k(const Tensor& A, const Tensor& B) {
    int64_t m = A.shape()[0], k = A.shape()[1], n = B.shape()[0];
    Tensor C({m, n});
    const float* a_ptr = A.data();
    const float* b_ptr = B.data();
    float* c_ptr = C.data();

    const int64_t I_BLOCK = 2;

    if (k < 8) {
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int64_t l = 0; l < k; ++l) {
                    sum += a_ptr[i * k + l] * b_ptr[j * k + l];
                }
                c_ptr[i * n + j] = sum;
            }
        }
        return C;
    }

    const int64_t i_blocked = (m / I_BLOCK) * I_BLOCK;
    // OpenMP: i 块独立写 C → 逐位一致（2026-08-16）
    #pragma omp parallel for schedule(guided) if(m * k * n >= (1 << 20))
    for (int64_t i0 = 0; i0 < i_blocked; i0 += I_BLOCK) {
        int64_t j = 0;
        for (; j + 8 <= n; j += 8) {
            __m256 c0 = _mm256_loadu_ps(&c_ptr[i0 * n + j]);
            __m256 c1 = _mm256_loadu_ps(&c_ptr[(i0 + 1) * n + j]);

            __m256i base_offs = _mm256_setr_epi32(
                (int)(j * k), (int)((j + 1) * k),
                (int)((j + 2) * k), (int)((j + 3) * k),
                (int)((j + 4) * k), (int)((j + 5) * k),
                (int)((j + 6) * k), (int)((j + 7) * k));

            for (int64_t l = 0; l < k; ++l) {
                __m256 a0 = _mm256_set1_ps(a_ptr[i0 * k + l]);
                __m256 a1 = _mm256_set1_ps(a_ptr[(i0 + 1) * k + l]);

                __m256i lv = _mm256_set1_epi32((int)l);
                __m256i vidx = _mm256_add_epi32(base_offs, lv);
                __m256 b = _mm256_i32gather_ps((const void*)b_ptr, vidx, 4);

                c0 = _mm256_fmadd_ps(a0, b, c0);
                c1 = _mm256_fmadd_ps(a1, b, c1);
            }

            _mm256_storeu_ps(&c_ptr[i0 * n + j], c0);
            _mm256_storeu_ps(&c_ptr[(i0 + 1) * n + j], c1);
        }

        // Remainder j columns (scalar)
        for (; j < n; ++j) {
            float s0 = 0.0f, s1 = 0.0f;
            for (int64_t l = 0; l < k; ++l) {
                float b_val = b_ptr[j * k + l];
                s0 += a_ptr[i0 * k + l] * b_val;
                s1 += a_ptr[(i0 + 1) * k + l] * b_val;
            }
            c_ptr[i0 * n + j] += s0;
            c_ptr[(i0 + 1) * n + j] += s1;
        }
    }

    // Remainder i rows (single-row, gather-based)
    for (int64_t i = i_blocked; i < m; ++i) {
        int64_t j = 0;
        for (; j + 8 <= n; j += 8) {
            __m256i base_offs = _mm256_setr_epi32(
                (int)(j * k), (int)((j + 1) * k),
                (int)((j + 2) * k), (int)((j + 3) * k),
                (int)((j + 4) * k), (int)((j + 5) * k),
                (int)((j + 6) * k), (int)((j + 7) * k));
            for (int64_t l = 0; l < k; ++l) {
                __m256 a_reg = _mm256_set1_ps(a_ptr[i * k + l]);
                __m256i lv = _mm256_set1_epi32((int)l);
                __m256i vidx = _mm256_add_epi32(base_offs, lv);
                __m256 b_vec = _mm256_i32gather_ps((const void*)b_ptr, vidx, 4);
                __m256 c_vec = _mm256_loadu_ps(&c_ptr[i * n + j]);
                _mm256_storeu_ps(&c_ptr[i * n + j],
                                 _mm256_fmadd_ps(a_reg, b_vec, c_vec));
            }
        }
        for (; j < n; ++j) {
            for (int64_t l = 0; l < k; ++l) {
                c_ptr[i * n + j] += a_ptr[i * k + l] * b_ptr[j * k + l];
            }
        }
    }

    return C;
}

// ============================================================================
// AVX2 FMA: matmul forward with transpose_a: C = A^T @ B
// A: (m, k), B: (m, n), C: (k, n)
// C(i,j) = sum_l A(l,i) * B(l,j)
//
// A 以原始布局 (m,k) 传入，A(l,i) 在地址 A[l*k + i] 处。
// 对于固定 i、变化 l，A(l,i) 步长 k（不连续），但只需标量广播。
// B(l, j:j+8) 连续，正常加载。
// ============================================================================
Tensor matmul_fwd_transpose_a_avx2_fma(const Tensor& A, const Tensor& B) {
    int64_t m = A.shape()[0], k = A.shape()[1], n = B.shape()[1];
    // C = A^T @ B, A(m,k), B(m,n) → C(k,n)
    Tensor C({k, n});
    const float* a_ptr = A.data();
    const float* b_ptr = B.data();
    float* c_ptr = C.data();

    const int64_t T_m = 64;   // tile over m (reduction dimension)
    const int64_t I_BLOCK = 4;

    for (int64_t l0 = 0; l0 < m; l0 += T_m) {
        int64_t l_end = (l0 + T_m < m) ? l0 + T_m : m;

        const int64_t i_blocked = (k / I_BLOCK) * I_BLOCK;
        // OpenMP: i 块独立写 C，m-tile 内串行累加 → 逐位一致（2026-08-16）
        #pragma omp parallel for schedule(guided) if(m * k * n >= (1 << 20))
        for (int64_t i0 = 0; i0 < i_blocked; i0 += I_BLOCK) {
            int64_t j = 0;
            for (; j + 8 <= n; j += 8) {
                __m256 c0 = _mm256_loadu_ps(&c_ptr[i0 * n + j]);
                __m256 c1 = _mm256_loadu_ps(&c_ptr[(i0 + 1) * n + j]);
                __m256 c2 = _mm256_loadu_ps(&c_ptr[(i0 + 2) * n + j]);
                __m256 c3 = _mm256_loadu_ps(&c_ptr[(i0 + 3) * n + j]);

                for (int64_t l = l0; l < l_end; ++l) {
                    __m256 b = _mm256_loadu_ps(&b_ptr[l * n + j]);
                    __m256 a0 = _mm256_set1_ps(a_ptr[l * k + i0]);
                    __m256 a1 = _mm256_set1_ps(a_ptr[l * k + i0 + 1]);
                    __m256 a2 = _mm256_set1_ps(a_ptr[l * k + i0 + 2]);
                    __m256 a3 = _mm256_set1_ps(a_ptr[l * k + i0 + 3]);
                    c0 = _mm256_fmadd_ps(a0, b, c0);
                    c1 = _mm256_fmadd_ps(a1, b, c1);
                    c2 = _mm256_fmadd_ps(a2, b, c2);
                    c3 = _mm256_fmadd_ps(a3, b, c3);
                }

                _mm256_storeu_ps(&c_ptr[i0 * n + j], c0);
                _mm256_storeu_ps(&c_ptr[(i0 + 1) * n + j], c1);
                _mm256_storeu_ps(&c_ptr[(i0 + 2) * n + j], c2);
                _mm256_storeu_ps(&c_ptr[(i0 + 3) * n + j], c3);
            }

            // Remainder j columns (scalar)
            for (; j < n; ++j) {
                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                for (int64_t l = l0; l < l_end; ++l) {
                    float b_val = b_ptr[l * n + j];
                    s0 += a_ptr[l * k + i0] * b_val;
                    s1 += a_ptr[l * k + i0 + 1] * b_val;
                    s2 += a_ptr[l * k + i0 + 2] * b_val;
                    s3 += a_ptr[l * k + i0 + 3] * b_val;
                }
                c_ptr[i0 * n + j] += s0;
                c_ptr[(i0 + 1) * n + j] += s1;
                c_ptr[(i0 + 2) * n + j] += s2;
                c_ptr[(i0 + 3) * n + j] += s3;
            }
        }

        // Remainder i rows (single-row)
        for (int64_t i = i_blocked; i < k; ++i) {
            for (int64_t l = l0; l < l_end; ++l) {
                float a_val = a_ptr[l * k + i];
                __m256 a_reg = _mm256_set1_ps(a_val);
                int64_t j = 0;
                for (; j + 8 <= n; j += 8) {
                    __m256 b_vec = _mm256_loadu_ps(&b_ptr[l * n + j]);
                    __m256 c_vec = _mm256_loadu_ps(&c_ptr[i * n + j]);
                    _mm256_storeu_ps(&c_ptr[i * n + j],
                                     _mm256_fmadd_ps(a_reg, b_vec, c_vec));
                }
                for (; j < n; ++j) {
                    c_ptr[i * n + j] += a_val * b_ptr[l * n + j];
                }
            }
        }
    }
    return C;
}

// ============================================================================
// AVX2 FMA: matmul forward with transpose_a, small-K optimized path
// C = A^T @ B, A: (m,k), B: (m,n), C: (k,n)
// C(i,j) = sum_l A(l,i) * B(l,j)
//
// 针对小 reduction 维度（m ≤ 128）优化：去掉 m-tiling，I_BLOCK=2。
// ============================================================================
Tensor matmul_fwd_transpose_a_avx2_fma_small_k(const Tensor& A, const Tensor& B) {
    int64_t m = A.shape()[0], k = A.shape()[1], n = B.shape()[1];
    Tensor C({k, n});
    const float* a_ptr = A.data();
    const float* b_ptr = B.data();
    float* c_ptr = C.data();

    const int64_t I_BLOCK = 2;

    if (m < 8) {
        for (int64_t i = 0; i < k; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                float sum = 0.0f;
                for (int64_t l = 0; l < m; ++l) {
                    sum += a_ptr[l * k + i] * b_ptr[l * n + j];
                }
                c_ptr[i * n + j] = sum;
            }
        }
        return C;
    }

    const int64_t i_blocked = (k / I_BLOCK) * I_BLOCK;
    // OpenMP: i 块独立写 C → 逐位一致（2026-08-16）
    #pragma omp parallel for schedule(guided) if(m * k * n >= (1 << 20))
    for (int64_t i0 = 0; i0 < i_blocked; i0 += I_BLOCK) {
        int64_t j = 0;
        for (; j + 8 <= n; j += 8) {
            __m256 c0 = _mm256_loadu_ps(&c_ptr[i0 * n + j]);
            __m256 c1 = _mm256_loadu_ps(&c_ptr[(i0 + 1) * n + j]);

            for (int64_t l = 0; l < m; ++l) {
                __m256 b = _mm256_loadu_ps(&b_ptr[l * n + j]);
                __m256 a0 = _mm256_set1_ps(a_ptr[l * k + i0]);
                __m256 a1 = _mm256_set1_ps(a_ptr[l * k + i0 + 1]);
                c0 = _mm256_fmadd_ps(a0, b, c0);
                c1 = _mm256_fmadd_ps(a1, b, c1);
            }

            _mm256_storeu_ps(&c_ptr[i0 * n + j], c0);
            _mm256_storeu_ps(&c_ptr[(i0 + 1) * n + j], c1);
        }

        // Remainder j columns (scalar)
        for (; j < n; ++j) {
            float s0 = 0.0f, s1 = 0.0f;
            for (int64_t l = 0; l < m; ++l) {
                float b_val = b_ptr[l * n + j];
                s0 += a_ptr[l * k + i0] * b_val;
                s1 += a_ptr[l * k + i0 + 1] * b_val;
            }
            c_ptr[i0 * n + j] += s0;
            c_ptr[(i0 + 1) * n + j] += s1;
        }
    }

    // Remainder i rows (single-row)
    for (int64_t i = i_blocked; i < k; ++i) {
        for (int64_t l = 0; l < m; ++l) {
            float a_val = a_ptr[l * k + i];
            __m256 a_reg = _mm256_set1_ps(a_val);
            int64_t j = 0;
            for (; j + 8 <= n; j += 8) {
                __m256 b_vec = _mm256_loadu_ps(&b_ptr[l * n + j]);
                __m256 c_vec = _mm256_loadu_ps(&c_ptr[i * n + j]);
                _mm256_storeu_ps(&c_ptr[i * n + j],
                                 _mm256_fmadd_ps(a_reg, b_vec, c_vec));
            }
            for (; j < n; ++j) {
                c_ptr[i * n + j] += a_val * b_ptr[l * n + j];
            }
        }
    }

    return C;
}

// ============================================================================
// matmul forward with transpose_b: C = A @ B^T  (dispatch)
// ============================================================================
Tensor matmul_forward_transpose_b(const Tensor& A, const Tensor& B) {
    if (A.ndim() != 2 || B.ndim() != 2) {
        throw std::invalid_argument("matmul_forward_transpose_b: inputs must be 2D");
    }
    if (!A.is_contiguous() || !B.is_contiguous()) {
        throw std::invalid_argument("matmul_forward_transpose_b: inputs must be contiguous");
    }

    int64_t k_a = A.shape()[1];
    int64_t k_b = B.shape()[1];

    if (k_a != k_b) {
        throw std::invalid_argument("matmul_forward_transpose_b: inner dimension mismatch "
            "(A.shape[1]=" + std::to_string(k_a) + " vs B.shape[1]=" + std::to_string(k_b) + ")");
    }

    const KernelSet& ks = kernel_registry();
    return ks.matmul_transpose_b(A, B);
}

// ============================================================================
// matmul forward with transpose_a: C = A^T @ B  (dispatch)
// ============================================================================
Tensor matmul_forward_transpose_a(const Tensor& A, const Tensor& B) {
    if (A.ndim() != 2 || B.ndim() != 2) {
        throw std::invalid_argument("matmul_forward_transpose_a: inputs must be 2D");
    }
    if (!A.is_contiguous() || !B.is_contiguous()) {
        throw std::invalid_argument("matmul_forward_transpose_a: inputs must be contiguous");
    }

    int64_t m_a = A.shape()[0];
    int64_t m_b = B.shape()[0];

    if (m_a != m_b) {
        throw std::invalid_argument("matmul_forward_transpose_a: inner dimension mismatch "
            "(A.shape[0]=" + std::to_string(m_a) + " vs B.shape[0]=" + std::to_string(m_b) + ")");
    }

    const KernelSet& ks = kernel_registry();
    return ks.matmul_transpose_a(A, B);
}

// ============================================================================
// AVX2 FMA: dA = dY @ B^T, (m,k) = (m,n) @ (k,n)^T
// dA[i,l] = sum_j dY[i,j] * B[l,j], vectorized over j with horizontal sum
//
// 注：尝试过委托 matmul_forward(dY, transpose_2d(B))，但 transpose_2d 的
// O(k*n) 拷贝开销抵消了 matmul_forward 的优化收益（dA 的 m 很小）。保留水平
// 求和方案，该方案避免拷贝且 B[l,:] 访问顺序友好。
// ============================================================================
Tensor matmul_bwd_dA_avx2_fma(const Tensor& dY, const Tensor& B) {
    int64_t m = dY.shape()[0], n = dY.shape()[1], k = B.shape()[0];
    Tensor dA({m, k});
    const float* dy_ptr = dY.data();
    const float* b_ptr = B.data();
    float* da_ptr = dA.data();

    // OpenMP: 每行 dA 独立（内层 k 串行累加）→ 逐位一致（2026-08-16）
    #pragma omp parallel for schedule(guided) if(m * k * n >= (1 << 20))
    for (int64_t i = 0; i < m; ++i) {
        for (int64_t l = 0; l < k; ++l) {
            __m256 sum = _mm256_setzero_ps();
            int64_t j = 0;
            for (; j + 8 <= n; j += 8) {
                __m256 dy_vec = _mm256_loadu_ps(&dy_ptr[i * n + j]);
                __m256 b_vec = _mm256_loadu_ps(&b_ptr[l * n + j]);
                sum = _mm256_fmadd_ps(dy_vec, b_vec, sum);
            }
            // Horizontal sum of 8 floats
            float hsum[8];
            _mm256_storeu_ps(hsum, sum);
            float total = hsum[0] + hsum[1] + hsum[2] + hsum[3]
                        + hsum[4] + hsum[5] + hsum[6] + hsum[7];
            for (; j < n; ++j) total += dy_ptr[i * n + j] * b_ptr[l * n + j];
            da_ptr[i * k + l] = total;
        }
    }
    return dA;
}

// ============================================================================
// AVX2 FMA: dB = A^T @ dY, (k,n) = (m,k)^T @ (m,n)
// dB[l,j] = sum_i A[i,l] * dY[i,j], uses i-k-j style accumulation over j
// ============================================================================
Tensor matmul_bwd_dB_avx2_fma(const Tensor& A, const Tensor& dY) {
    int64_t m = A.shape()[0], k = A.shape()[1], n = dY.shape()[1];
    Tensor dB({k, n});
    const float* a_ptr = A.data();
    const float* dy_ptr = dY.data();
    float* db_ptr = dB.data();

    // Iterate over l (row of dB), i (row of A), vectorize over j
    // OpenMP: 每行 dB 独立（内层 i 串行累加）→ 逐位一致（2026-08-16）
    #pragma omp parallel for schedule(guided) if(m * k * n >= (1 << 20))
    for (int64_t l = 0; l < k; ++l) {
        for (int64_t i = 0; i < m; ++i) {
            float a_val = a_ptr[i * k + l];
            __m256 a_reg = _mm256_set1_ps(a_val);
            int64_t j = 0;
            for (; j + 8 <= n; j += 8) {
                __m256 dy_vec = _mm256_loadu_ps(&dy_ptr[i * n + j]);
                __m256 db_vec = _mm256_loadu_ps(&db_ptr[l * n + j]);
                _mm256_storeu_ps(&db_ptr[l * n + j], _mm256_fmadd_ps(a_reg, dy_vec, db_vec));
            }
            for (; j < n; ++j) {
                db_ptr[l * n + j] += a_val * dy_ptr[i * n + j];
            }
        }
    }
    return dB;
}

// ============================================================================
// matmul backward: dA = dY @ B^T, dB = A^T @ dY
// ============================================================================
std::pair<Tensor, Tensor> matmul_backward(
    const Tensor& A, const Tensor& B, const Tensor& dY
) {
    // 检查 2D + contiguous
    if (A.ndim() != 2 || B.ndim() != 2 || dY.ndim() != 2) {
        throw std::invalid_argument("matmul_backward: inputs must be 2D");
    }
    if (!A.is_contiguous() || !B.is_contiguous() || !dY.is_contiguous()) {
        throw std::invalid_argument("matmul_backward: inputs must be contiguous");
    }

    int64_t m = A.shape()[0];
    int64_t k = A.shape()[1];
    int64_t k2 = B.shape()[0];
    int64_t n = B.shape()[1];
    int64_t m_y = dY.shape()[0];
    int64_t n_y = dY.shape()[1];

    if (k != k2) {
        throw std::invalid_argument("matmul_backward: A/B inner dimension mismatch");
    }
    if (m != m_y || n != n_y) {
        throw std::invalid_argument("matmul_backward: dY shape must match (m, n)");
    }

    const KernelSet& ks = kernel_registry();
    return ks.matmul_backward(A, B, dY);
}

// ============================================================================
// transpose_2d: 返回新的 contiguous 转置 Tensor（非 view）
// 分块转置（T=64）提高 cache 命中率：每个 tile 内 x_ptr 连续访问，r_ptr 跨度小
// ============================================================================
Tensor transpose_2d(const Tensor& x) {
    if (x.ndim() != 2) {
        throw std::invalid_argument("transpose_2d: input must be 2D");
    }
    int64_t r = x.shape()[0];
    int64_t c = x.shape()[1];

    Tensor result({c, r});
    const float* x_ptr = x.data();
    float* r_ptr = result.data();

    const int64_t T = 64;  // tile size (cache-friendly)
    for (int64_t i0 = 0; i0 < r; i0 += T) {
        int64_t i_end = (i0 + T < r) ? i0 + T : r;
        for (int64_t j0 = 0; j0 < c; j0 += T) {
            int64_t j_end = (j0 + T < c) ? j0 + T : c;
            for (int64_t i = i0; i < i_end; ++i) {
                for (int64_t j = j0; j < j_end; ++j) {
                    r_ptr[j * r + i] = x_ptr[i * c + j];
                }
            }
        }
    }
    return result;
}

}  // namespace sgn_autograd
