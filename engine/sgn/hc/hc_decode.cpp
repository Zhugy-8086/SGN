// hc_decode.cpp - hc(B,L) 分层编码/解码实现（R3-B 落地）
//
// 从 R2b 基准（tests/hc_ext/bench_r2b_decode.c）提升为可复用组件。
// SIMD 逐层累加解码（B=8 用 8 位 lane，B=16 用 16 位 lane）；部分解码 L'<L 只读
// L'·n 字节（A 路径）。数值档位：kRounding（float32 累加，与 R0 网格差 ≤ 数 ulp）。
//
// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 zhugy-8086
#include "hc/hc_decode.h"

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

namespace sgn_autograd {

namespace {

int64_t ipow64(int64_t b, int e) {
    int64_t r = 1;
    for (int i = 0; i < e; ++i) r *= b;
    return r;
}

// floor 除法（C 的 / 向零截断，digit 分解须向下取整，与 R2a Python // 一致）
int64_t fldiv(int64_t a, int64_t b) {
    int64_t q = a / b;
    if ((a % b != 0) && ((a < 0) != (b < 0))) q -= 1;
    return q;
}

}  // namespace

void hc_encode_n(int B, int L, const float* x, int64_t n,
                 float* scale_out, uint8_t* layers) {
    float amax = 0.0f;
    for (int64_t i = 0; i < n; ++i) amax = std::fmax(amax, std::fabs(x[i]));
    if (amax == 0.0f) amax = 1.0f;
    const int64_t b = (B == 16) ? 65536 : 256;
    const double top = (B == 16) ? 32767.0 : 127.0;
    *scale_out = (float)(amax / top);
    const int64_t bL1 = ipow64(b, L - 1);
    for (int64_t i = 0; i < n; ++i) {
        // 直接量化到完整网格再拆层（勿先量化到单层——小数层会全为 0，R2b 教训）
        int64_t N = (int64_t)llround((double)x[i] / amax * top * (double)bL1);
        int64_t rem = N;
        for (int j = 0; j < L; ++j) {
            int64_t p = ipow64(b, L - 1 - j);
            int64_t d = fldiv(rem, p);
            if (B == 16) {
                ((uint16_t*)layers)[(int64_t)j * n + i] = (uint16_t)(d & 0xFFFF);
            } else {
                layers[(int64_t)j * n + i] = (uint8_t)(d & 0xFF);
            }
            rem -= d * p;
        }
    }
}

void hc_decode_n(int B, int L, int Lprime, const uint8_t* layers,
                 float scale, float* out, int64_t n) {
    (void)L;  // 仅 Lprime 决定解码层数（A 路径）
    if (B == 16) {
        // 每层权重 1/b^j（b=65536）；ws 尺寸与 B=8 对称（支持 Lprime ≤ 6，R3-B 审计）
        float ws[7];
        ws[0] = 1.0f; { double a = 1.0; for (int j = 1; j < 7; ++j) { a /= 65536.0; ws[j] = (float)a; } }
        const uint16_t* lay = (const uint16_t*)layers;
        __m256 sc = _mm256_set1_ps(scale);
        int64_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 acc = _mm256_cvtepi32_ps(
                _mm256_cvtepi16_epi32(_mm_loadu_si128((const __m128i*)(lay + 0 * n + i))));
            for (int j = 1; j < Lprime; ++j) {
                __m256 v = _mm256_cvtepi32_ps(
                    _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*)(lay + (int64_t)j * n + i))));
                acc = _mm256_fmadd_ps(v, _mm256_set1_ps(ws[j]), acc);
            }
            _mm256_storeu_ps(out + i, _mm256_mul_ps(acc, sc));
        }
        for (; i < n; ++i) {
            float a = (float)(int16_t)lay[0 * n + i];
            for (int j = 1; j < Lprime; ++j) a += (float)(uint16_t)lay[(int64_t)j * n + i] * ws[j];
            out[i] = a * scale;
        }
    } else {  // B == 8
        float ws[7];
        ws[0] = 1.0f; { double a = 1.0; for (int j = 1; j < 7; ++j) { a /= 256.0; ws[j] = (float)a; } }
        __m256 sc = _mm256_set1_ps(scale);
        int64_t i = 0;
        for (; i + 8 <= n; i += 8) {
            __m256 acc = _mm256_cvtepi32_ps(
                _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(layers + 0 * n + i))));
            for (int j = 1; j < Lprime; ++j) {
                __m256 v = _mm256_cvtepi32_ps(
                    _mm256_cvtepu8_epi32(_mm_loadl_epi64((const __m128i*)(layers + (int64_t)j * n + i))));
                acc = _mm256_fmadd_ps(v, _mm256_set1_ps(ws[j]), acc);
            }
            _mm256_storeu_ps(out + i, _mm256_mul_ps(acc, sc));
        }
        for (; i < n; ++i) {
            float a = (float)(int8_t)layers[0 * n + i];
            for (int j = 1; j < Lprime; ++j) a += (float)(uint8_t)layers[(int64_t)j * n + i] * ws[j];
            out[i] = a * scale;
        }
    }
}

}  // namespace sgn_autograd
