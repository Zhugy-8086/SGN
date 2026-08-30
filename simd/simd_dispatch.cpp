// simd_dispatch.cpp - simd 原语层运行时调度（HAL 雏形，P2）
//
// 背景：fixes_相关修复/simd指令集加速文件拆分计划_2026_08_29.md P2。
// 结构仿 dispatch/registry.cpp：
//   - 每个原语的公开接口（simd_api.h 声明）在本文件实现为"调度入口"，
//     经 simd_backend() 的表中函数指针选择最优实现；
//   - SIMD 实现（x86/*.cpp）与标量锚点（scalar.cpp）双层化，命名 _avx2/_vnni/_ssse3/_scalar；
//   - 能力解析【进程生命周期只执行一次】（C++11 magic static，线程安全），
//     执行路径上是零分支的间接调用——回退通道存在但不影响主内核速度。
//
// 选择规则（每原语独立取最优可用实现）：
//   - dot8/dot4          → AVX-VNNI（vpdpbusd）> scalar
//   - dot16/decode/sum 系 → AVX2 > scalar
//   - reverse/batch      → SSSE3 > scalar
// 编译期回退保持：非 x86 / 无 AVX 平台，对应 SIMD 实现文件 #if 不编译，
// 表中仅标量项 → 跨平台可移植性不变。
//
// 注意：当前构建仍全局 -mavx2 -mavxvnni（binary 硬性要求 AVX2，见拆分计划 D2）；
// SGN_KERNEL_BACKEND=scalar 环境变量可强制回退标量（测试钩子，与 dispatch/registry 一致）。

#include "simd/simd_api.h"

#include <cstdlib>
#include <cstring>

namespace sgn::simd {

// ============================================================================
// 内部实现前向声明（定义在 scalar.cpp / x86/*.cpp）
// ============================================================================

// 标量锚点（常驻编译）
int64_t dot16_scalar(const int16_t*, const int16_t*, size_t);
int64_t dot8_scalar(const uint8_t*, const int8_t*, size_t);
int64_t dot4_scalar(const uint8_t*, const int8_t*, size_t);
void decode_i16_f32_scalar(const uint64_t*, int, float, float*);
void reverse_bytes8_scalar(uint64_t, int, int64_t*);
void batch_reverse_u8_scalar(const uint64_t*, int, int, int64_t*);
float sum_f32_scalar(const float*, int64_t, int64_t);
float sum_sq_dev_f32_scalar(const float*, int64_t, int64_t, float);
void sum_sumprod_f32_scalar(const float*, const float*, int64_t, int64_t, float*, float*);
void accum_f32_scalar(float*, const float*, int64_t);

// SIMD 实现（编译期宏保护；非支持平台不编译 → 表中无此项）
#if defined(__AVXVNNI__)
int64_t dot8_vnni(const uint8_t*, const int8_t*, size_t);
int64_t dot4_vnni(const uint8_t*, const int8_t*, size_t);
#endif
#if defined(__AVX2__)
int64_t dot16_avx2(const int16_t*, const int16_t*, size_t);
void decode_i16_f32_avx2(const uint64_t*, int, float, float*);
float sum_f32_avx2(const float*, int64_t, int64_t);
float sum_sq_dev_f32_avx2(const float*, int64_t, int64_t, float);
void sum_sumprod_f32_avx2(const float*, const float*, int64_t, int64_t, float*, float*);
void accum_f32_avx2(float*, const float*, int64_t);
#endif
#if defined(__SSSE3__)
void reverse_bytes8_ssse3(uint64_t, int, int64_t*);
void batch_reverse_u8_ssse3(const uint64_t*, int, int, int64_t*);
#endif

namespace {

// CPU 能力快照（一次性填充；编译期宏优先，未以 SIMD 编译时运行时 CPUID 检测——仅 x86）。
// 与 dispatch/registry.cpp 的 CpuCaps 同构，但只关注 simd 原语层依赖的指令集。
struct CpuCaps {
    bool avx2;     // 覆盖 dot16 / decode / sum 系
    bool avx_vnni; // 覆盖 dot8 / dot4
    bool ssse3;    // 覆盖 reverse / batch
};

CpuCaps cpu_caps() {
    CpuCaps caps = {false, false, false};
#if defined(__AVX2__)
    caps.avx2 = true;
#endif
#if defined(__AVXVNNI__)
    caps.avx_vnni = true;
#endif
#if defined(__SSSE3__)
    caps.ssse3 = true;
#endif
#if !defined(__AVX2__) && (defined(__x86_64__) || defined(_M_X64))
    if (__builtin_cpu_supports("avx2")) caps.avx2 = true;
    if (__builtin_cpu_supports("avxvnni")) caps.avx_vnni = true;
    if (__builtin_cpu_supports("ssse3")) caps.ssse3 = true;
#endif
    return caps;
}

}  // anonymous namespace

const SimdBackend& simd_backend() noexcept {
    // magic static: 仅第一次进入时求值一次，此后直接返回缓存引用（无分支）
    static const SimdBackend s = [] {
        // 环境变量强制后端（测试钩子，不做运行期热切换；与 dispatch/registry 一致）。
        // 仅支持强制 scalar（sse2 同样落到标量原语）。
        static char forced_buf[32] = {0};
        size_t required = 0;
        getenv_s(&required, forced_buf, sizeof(forced_buf), "SGN_KERNEL_BACKEND");
        const char* forced = (required > 0 && required <= sizeof(forced_buf))
                                 ? forced_buf
                                 : nullptr;

        SimdBackend b;
        if (forced && (std::strcmp(forced, "scalar") == 0 ||
                       std::strcmp(forced, "sse2") == 0)) {
            // 全标量回退（bit-exact 锚点）
            b.dot16            = dot16_scalar;
            b.dot8             = dot8_scalar;
            b.dot4             = dot4_scalar;
            b.decode_i16_f32   = decode_i16_f32_scalar;
            b.reverse_bytes8   = reverse_bytes8_scalar;
            b.batch_reverse_u8 = batch_reverse_u8_scalar;
            b.sum_f32          = sum_f32_scalar;
            b.sum_sq_dev_f32   = sum_sq_dev_f32_scalar;
            b.sum_sumprod_f32  = sum_sumprod_f32_scalar;
            b.accum_f32        = accum_f32_scalar;
            b.name             = "scalar(forced)";
            return b;
        }

        const CpuCaps& caps = cpu_caps();
        // 默认全标量（跨平台回退兜底），再按可用性覆盖各原语
        b.dot16            = dot16_scalar;
        b.dot8             = dot8_scalar;
        b.dot4             = dot4_scalar;
        b.decode_i16_f32   = decode_i16_f32_scalar;
        b.reverse_bytes8   = reverse_bytes8_scalar;
        b.batch_reverse_u8 = batch_reverse_u8_scalar;
        b.sum_f32          = sum_f32_scalar;
        b.sum_sq_dev_f32   = sum_sq_dev_f32_scalar;
        b.sum_sumprod_f32  = sum_sumprod_f32_scalar;
        b.accum_f32        = accum_f32_scalar;

#if defined(__AVXVNNI__)
        if (caps.avx_vnni) {
            b.dot8 = dot8_vnni;
            b.dot4 = dot4_vnni;
        }
#endif
#if defined(__AVX2__)
        if (caps.avx2) {
            b.dot16           = dot16_avx2;
            b.decode_i16_f32  = decode_i16_f32_avx2;
            b.sum_f32         = sum_f32_avx2;
            b.sum_sq_dev_f32  = sum_sq_dev_f32_avx2;
            b.sum_sumprod_f32 = sum_sumprod_f32_avx2;
            b.accum_f32       = accum_f32_avx2;
        }
#endif
#if defined(__SSSE3__)
        if (caps.ssse3) {
            b.reverse_bytes8   = reverse_bytes8_ssse3;
            b.batch_reverse_u8 = batch_reverse_u8_ssse3;
        }
#endif

        // 整体后端名 = 最高可用指令集（诊断用）
        if (caps.avx_vnni)   b.name = "avxvnni";
        else if (caps.avx2)  b.name = "avx2";
        else if (caps.ssse3) b.name = "ssse3";
        else                 b.name = "scalar";
        return b;
    }();
    return s;
}

const char* active_backend_name() noexcept {
    return simd_backend().name;
}

// ============================================================================
// 调度入口（公开接口，simd_api.h 声明；调用方零改动）
// ============================================================================

int64_t dot16(const int16_t* a, const int16_t* b, size_t K) {
    return simd_backend().dot16(a, b, K);
}

int64_t dot8(const uint8_t* a, const int8_t* b, size_t K) {
    return simd_backend().dot8(a, b, K);
}

int64_t dot4(const uint8_t* u8, const int8_t* s8, size_t K) {
    return simd_backend().dot4(u8, s8, K);
}

void decode_i16_f32(const uint64_t* pv_ptr, int n_values, float scale, float* res_ptr) {
    simd_backend().decode_i16_f32(pv_ptr, n_values, scale, res_ptr);
}

void reverse_bytes8(uint64_t packed, int n, int64_t* out) {
    simd_backend().reverse_bytes8(packed, n, out);
}

void batch_reverse_u8(const uint64_t* packed_values, int n_values, int n_slots,
                      int64_t* result) {
    simd_backend().batch_reverse_u8(packed_values, n_values, n_slots, result);
}

float sum_f32(const float* p, int64_t n, int64_t stride) {
    return simd_backend().sum_f32(p, n, stride);
}

float sum_sq_dev_f32(const float* p, int64_t n, int64_t stride, float mu) {
    return simd_backend().sum_sq_dev_f32(p, n, stride, mu);
}

void sum_sumprod_f32(const float* a, const float* b, int64_t n, int64_t stride,
                     float* out_sum, float* out_sumprod) {
    simd_backend().sum_sumprod_f32(a, b, n, stride, out_sum, out_sumprod);
}

void accum_f32(float* dst, const float* src, int64_t n) {
    simd_backend().accum_f32(dst, src, n);
}

}  // namespace sgn::simd
