// scalar.cpp - mkern/gemm 标量实现（bit-exact 参考锚点，全平台常驻编译）
//
// 与 mkern/simd/scalar.cpp 同纪律：标量锚点常驻编译（不受 SIMD 宏屏蔽），供
// gemm_dispatch.cpp 运行时表在无对应 SIMD 实现时回退，并作为 boundary 测试的
// 逐位参照。命名统一后缀 _scalar。pack_b_i8 为平台无关纯标量搬运（同
// unpack_nibble 定位），无 SIMD 版本。
//
// 背景：fixes_相关修复/mkern微内核层实施计划_2026_09_03.md §3.2（R3）。

#include "mkern/gemm/gemm_api.h"
#include "mkern/simd/simd_api.h"

namespace sgn::mkern::gemm {

int64_t pack_b_i8_bytes(int64_t N, int64_t K) {
    return ((N + 15) / 16) * ((K + 3) / 4) * 64;
}

void pack_b_i8_scalar(int8_t* Bp_out, const int8_t* B_in, int64_t N, int64_t K) {
    const int64_t np = (N + 15) / 16;   // 列面板（末面板对 N 之外零填充）
    const int64_t ng = (K + 3) / 4;     // k 组数（尾组对 k ≥ K 零填充）
    for (int64_t p = 0; p < np; ++p) {
        for (int64_t grp = 0; grp < ng; ++grp) {
            int8_t* dst = Bp_out + (p * ng + grp) * 64;
            for (int64_t t = 0; t < 16; ++t) {
                for (int64_t j = 0; j < 4; ++j) {
                    const int64_t k = grp * 4 + j;
                    const int64_t n = p * 16 + t;
                    dst[t * 4 + j] = (k < K && n < N) ? B_in[k * N + n] : 0;
                }
            }
        }
    }
}

void gemm_i8_scalar(int32_t* C, const uint8_t* A, const int8_t* Bp,
                    int64_t M, int64_t N, int64_t K, bool accum) {
    // 消费 VNNI 面板布局（与 SIMD 内核同输入契约）：列 n → 面板 p=n/16、
    // 面板内列 t=n%16，逐组累加 4 个 k（k ≥ K 的补零字节积为 0，跳过）。整数
    // 累加顺序无关 → bit-exact。M/N 尾部天然覆盖（面板已零填充到 16 对齐）。
    const int64_t ng = (K + 3) / 4;
    for (int64_t m = 0; m < M; ++m) {
        const uint8_t* a = A + m * K;
        for (int64_t n = 0; n < N; ++n) {
            const int8_t* vb = Bp + (n / 16) * ng * 64 + (n % 16) * 4;
            int64_t sum = 0;
            for (int64_t grp = 0; grp < ng; ++grp) {
                const int8_t* v = vb + grp * 64;
                for (int64_t j = 0; j < 4; ++j) {
                    const int64_t k = grp * 4 + j;
                    if (k < K) sum += static_cast<int64_t>(a[k]) * static_cast<int64_t>(v[j]);
                }
            }
            if (accum) C[m * N + n] += static_cast<int32_t>(sum);
            else       C[m * N + n]  = static_cast<int32_t>(sum);
        }
    }
}

void gemm_i16_scalar(int64_t* C, const int16_t* A, const int16_t* B,
                     int64_t M, int64_t N, int64_t K, bool accum) {
    for (int64_t m = 0; m < M; ++m) {
        const int16_t* a = A + m * K;
        for (int64_t n = 0; n < N; ++n) {
            int64_t sum = 0;
            for (int64_t k = 0; k < K; ++k) {
                sum += static_cast<int64_t>(a[k]) * static_cast<int64_t>(B[k * N + n]);
            }
            if (accum) C[m * N + n] += sum;
            else       C[m * N + n]  = sum;
        }
    }
}

void gemv_i8_scalar(int64_t* y, const uint8_t* A, const int8_t* x, int64_t M, int64_t K) {
    // 逐行消费 simd::dot8（标量锚点表在 simd 层内部调度；分层见 gemm_api.h 契约）
    for (int64_t m = 0; m < M; ++m) {
        y[m] = sgn::simd::dot8(A + m * K, x, K);
    }
}

} // namespace sgn::mkern::gemm
