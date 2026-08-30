// scalar.cpp - SIMD 原语库标量实现（bit-exact/参考锚点，全平台常驻编译）
//
// P2（HAL 雏形）：本文件全部原语标量实现常驻编译（不再受 SIMD 宏屏蔽），
// 供 simd_dispatch.cpp 的运行时表在无对应 SIMD 实现时回退。命名统一后缀 _scalar，
// 与 x86/*.cpp 的 SIMD 实现（_avx2/_vnni/_ssse3）区分。
// unpack_nibble_u/s 为平台无关纯标量语义，无 SIMD 版本，保持原名（不调度）。
//
// 背景：fixes_相关修复/simd指令集加速文件拆分计划_2026_08_29.md Step 0-3 + P2。

#include "simd/simd_api.h"

namespace sgn::simd {

void unpack_nibble_u(const uint8_t* su_p, size_t K, uint8_t* u8) {
    for (size_t i = 0; i < K; ++i) {
        const int sh = 4 * (static_cast<int>(i) & 1);
        u8[i] = static_cast<uint8_t>((su_p[i >> 1] >> sh) & 0x0F);
    }
}

void unpack_nibble_s(const uint8_t* ss_p, size_t K, int8_t* s8) {
    for (size_t i = 0; i < K; ++i) {
        const int sh = 4 * (static_cast<int>(i) & 1);
        uint8_t nib = static_cast<uint8_t>((ss_p[i >> 1] >> sh) & 0x0F);
        s8[i] = static_cast<int8_t>(static_cast<int8_t>(nib << 4) >> 4);
    }
}

// ---- 整型点积标量锚点（bit-exact 参照）----

int64_t dot16_scalar(const int16_t* a, const int16_t* b, size_t K) {
    int64_t sum = 0;
    for (size_t i = 0; i < K; ++i) {
        sum += static_cast<int64_t>(a[i]) * static_cast<int64_t>(b[i]);
    }
    return sum;
}

int64_t dot8_scalar(const uint8_t* a, const int8_t* b, size_t K) {
    int64_t sum = 0;
    for (size_t i = 0; i < K; ++i) {
        sum += static_cast<int64_t>(a[i]) * static_cast<int64_t>(b[i]);
    }
    return sum;
}

int64_t dot4_scalar(const uint8_t* u8, const int8_t* s8, size_t K) {
    int64_t sum = 0;
    for (size_t i = 0; i < K; ++i) {
        sum += static_cast<int64_t>(u8[i]) * static_cast<int64_t>(s8[i]);
    }
    return sum;
}

// ---- packed 解码标量锚点（Step 2）----

void decode_i16_f32_scalar(const uint64_t* pv_ptr, int n_values, float scale, float* res_ptr) {
    for (int i = 0; i < n_values; ++i) {
        int16_t val16 = static_cast<int16_t>(pv_ptr[i] & 0xFFFF);
        res_ptr[i] = static_cast<float>(val16) * scale;
    }
}

void reverse_bytes8_scalar(uint64_t packed, int n, int64_t* out) {
    for (int i = 0; i < n; ++i) {
        out[i] = static_cast<int64_t>((packed >> (8 * (n - 1 - i))) & 0xFF);
    }
}

void batch_reverse_u8_scalar(const uint64_t* packed_values, int n_values, int n_slots,
                             int64_t* result) {
    for (int i = 0; i < n_values; ++i) {
        uint64_t pv = packed_values[i];
        const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&pv);
        for (int j = 0; j < n_slots; ++j) {
            result[static_cast<size_t>(i) * n_slots + j] =
                static_cast<int64_t>(bytes[n_slots - 1 - j]);
        }
    }
}

// ---- float 归约/累加标量锚点（Step 3）----

float sum_f32_scalar(const float* p, int64_t n, int64_t stride) {
    float sum = 0.0f;
    for (int64_t i = 0; i < n; ++i) sum += p[i * stride];
    return sum;
}

float sum_sq_dev_f32_scalar(const float* p, int64_t n, int64_t stride, float mu) {
    float sum_sq = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        float d = p[i * stride] - mu;
        sum_sq += d * d;
    }
    return sum_sq;
}

void sum_sumprod_f32_scalar(const float* a, const float* b, int64_t n, int64_t stride,
                            float* out_sum, float* out_sumprod) {
    float sd = 0.0f, sdp = 0.0f;
    for (int64_t i = 0; i < n; ++i) {
        sd += a[i * stride];
        sdp += a[i * stride] * b[i * stride];
    }
    *out_sum = sd;
    *out_sumprod = sdp;
}

void accum_f32_scalar(float* dst, const float* src, int64_t n) {
    for (int64_t i = 0; i < n; ++i) dst[i] += src[i];
}

} // namespace sgn::simd
