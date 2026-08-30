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
// CPUID 检测用内联 intrinsic（跨平台）：
//   - Windows/MSVC target: <intrin.h> 提供 __cpuid/__cpuidex/_xgetbv
//   - Linux (GCC/Clang):    <cpuid.h> 提供 __get_cpuid/__get_cpuid_count；
//                           _xgetbv 在 <immintrin.h>（-mavx 下可用）。
// 置于全局命名空间：这些 intrinsic 是编译器内建，在匿名 namespace 内声明可能影响解析。
#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#include <immintrin.h>  // _xgetbv（需 -mavx 或更高）
#endif
#endif

namespace sgn::simd {
namespace {

// 跨平台读取环境变量（避免 MSVC 对 getenv 的弃用警告；与 common/logger.h 的
// sgn_getenv 同模式）：MSVC 下用 C11 getenv_s，其余平台用 std::getenv。
// 仅进程启动早期读取一次（magic static 求值时），返回指针生命周期足够。
inline const char* simd_getenv(const char* name) {
#if defined(_MSC_VER)
    static thread_local char buf[64];
    size_t len = 0;
    if (getenv_s(&len, buf, sizeof(buf), name) != 0) {
        return nullptr;
    }
    return (len == 0) ? nullptr : buf;
#else
    return std::getenv(name);
#endif
}
}  // namespace

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

// AVX-512 实现（服务器加速，P1；见 simd服务器加速计划_2026_08_30.md）。
// 无条件声明：avx512.cpp / avx512vnni.cpp 由 CMake 单独加 -mavx512f -mavx512vnni
// -mavx512bw（文件级选项），符号常驻产出；本文件基础编译，仅经 CPUID 运行时选入。
// decode_i16_f32 无 512 位版（带宽型，512 位打包收益有限）→ 落到 AVX2 版。
int64_t dot16_avx512(const int16_t*, const int16_t*, size_t);
int64_t dot8_avx512vnni(const uint8_t*, const int8_t*, size_t);
int64_t dot4_avx512vnni(const uint8_t*, const int8_t*, size_t);
float sum_f32_avx512(const float*, int64_t, int64_t);
float sum_sq_dev_f32_avx512(const float*, int64_t, int64_t, float);
void sum_sumprod_f32_avx512(const float*, const float*, int64_t, int64_t, float*, float*);
void accum_f32_avx512(float*, const float*, int64_t);

namespace {

// CPU 能力快照（一次性填充；编译期宏优先，未以 SIMD 编译时运行时 CPUID 检测——仅 x86）。
// 与 dispatch/registry.cpp 的 CpuCaps 同构，但只关注 simd 原语层依赖的指令集。
// AVX-512 字段恒走运行时检测（本文件基础编译无 __AVX512F__ 宏；avx512 符号常驻）。
struct CpuCaps {
    bool avx2;        // 覆盖 dot16 / decode / sum 系
    bool avx_vnni;    // 覆盖 dot8 / dot4
    bool ssse3;       // 覆盖 reverse / batch
    bool avx512f;     // 覆盖 dot16 / sum 系（512 位版）
    bool avx512_vnni; // 覆盖 dot8 / dot4（512 位 VNNI 版）
    bool avx512_bw;   // dot16 的 _mm512_mul_epi32 / 字节指令需 AVX512BW
};

#if defined(__x86_64__) || defined(_M_X64)
// CPUID 运行时检测。不用 __builtin_cpu_supports——其依赖 __cpu_model 运行时符号，
// -nostdlib 链接的 .pyd 无法解析（同 pysgn_net.cpp 的 _cpu_has_avx2 结论）。
// 跨平台：Windows 用 <intrin.h> 的 __cpuid/__cpuidex/_xgetbv；
//          Linux 用 <cpuid.h> 的 __get_cpuid/__get_cpuid_count（_xgetbv 同 intrinsic）。
// 位定义：leaf1 ECX: SSSE3=9, AVX=28, XSAVE=27；leaf7 sub0 EBX: AVX2=5, AVX512F=16,
//         AVX512BW=30；leaf7 sub0 ECX: AVX512VNNI=11, AVX-VNNI=4；XCR0: XMM=1, YMM=2,
//         opmask=4, ZMM hi=8。
static void cpuid_leaf(int leaf, int* r) {
#if defined(_MSC_VER)
    __cpuid(r, leaf);
#else
    __cpuid_count(leaf, 0, r[0], r[1], r[2], r[3]);
#endif
}
static CpuCaps cpu_caps_x86() {
    CpuCaps c = {false, false, false, false, false, false};
    int r[4];
    cpuid_leaf(1, r);
    const bool os_xsave = (r[2] & (1 << 27)) != 0;
    const bool cpu_avx  = (r[2] & (1 << 28)) != 0;
    c.ssse3 = (r[2] & (1 << 9)) != 0;
    if (os_xsave && cpu_avx && (_xgetbv(0) & 0x6) == 0x6) {
        // XMM+YMM 状态已由 OS 使能（AVX/AVX2 前提）
        cpuid_leaf(7, r);  // leaf7 subleaf0：一次调用读全 EBX/ECX/EDX
        c.avx2      = (r[1] & (1 << 5)) != 0;
        c.avx512f   = (r[1] & (1 << 16)) != 0;
        c.avx512_bw = (r[1] & (1 << 30)) != 0;
        // AVX-512 还需 opmask + ZMM hi256 状态（XCR0 0xE6）
        if (c.avx512f && (_xgetbv(0) & 0xE6) != 0xE6) {
            c.avx512f = c.avx512_bw = false;
        }
        // AVX-VNNI（256 位 VNNI）：leaf7 sub0 ECX bit4
        c.avx_vnni = (r[2] & (1 << 4)) != 0;
        // AVX512-VNNI：leaf7 sub0 ECX bit11（2026-08-31 修正——此前误读 subleaf1，
        // 而 sub1 ECX 恒为 0 导致 VNNI 永不检测到；复用上方 r[2]，无需再查 CPUID）。
        c.avx512_vnni = (r[2] & (1 << 11)) != 0;
    }
    return c;
}
#endif

CpuCaps cpu_caps() {
    CpuCaps caps = {false, false, false, false, false, false};
#if defined(__AVX2__)
    caps.avx2 = true;
#endif
#if defined(__AVXVNNI__)
    caps.avx_vnni = true;
#endif
#if defined(__SSSE3__)
    caps.ssse3 = true;
#endif
// AVX-512：无条件运行时检测（x86）——即使全局 -mavx2 编译下宏已定义 avx2，
// avx512 也只能经 CPUID 判定（本文件不随 avx512 文件加编译选项）。
#if defined(__x86_64__) || defined(_M_X64)
    const CpuCaps rt = cpu_caps_x86();
    caps.avx2        = caps.avx2        || rt.avx2;
    caps.avx_vnni    = caps.avx_vnni    || rt.avx_vnni;
    caps.ssse3       = caps.ssse3       || rt.ssse3;
    caps.avx512f     = rt.avx512f;
    caps.avx512_bw   = rt.avx512_bw;
    caps.avx512_vnni = rt.avx512_vnni;
#endif
    return caps;
}

}  // anonymous namespace

const SimdBackend& simd_backend() noexcept {
    // magic static: 仅第一次进入时求值一次，此后直接返回缓存引用（无分支）
    static const SimdBackend s = [] {
        // 环境变量强制后端（测试钩子，不做运行期热切换；与 dispatch/registry 一致）。
        // 仅支持强制 scalar（sse2 同样落到标量原语）。
        const char* forced = simd_getenv("SGN_KERNEL_BACKEND");

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
        // AVX-512（P1 服务器加速）：优先于 AVX-VNNI/AVX2 覆盖 compute-bound 原语。
        // decode_i16_f32 无 512 位版，保持 AVX2；reverse/batch 无 512 位版，保持 SSSE3。
        // 运行时 CPUID 检测（本文件基础编译，avx512 符号常驻由 CMake 文件级选项产出）。
        if (caps.avx512_vnni && caps.avx512_bw) {
            b.dot8 = dot8_avx512vnni;
            b.dot4 = dot4_avx512vnni;
        }
        if (caps.avx512f && caps.avx512_bw) {
            b.dot16           = dot16_avx512;
            b.sum_f32         = sum_f32_avx512;
            b.sum_sq_dev_f32  = sum_sq_dev_f32_avx512;
            b.sum_sumprod_f32 = sum_sumprod_f32_avx512;
            b.accum_f32       = accum_f32_avx512;
        }

        // 整体后端名 = 最高可用指令集（诊断用）
        if (caps.avx512_vnni) b.name = "avx512vnni";
        else if (caps.avx512f) b.name = "avx512f";
        else if (caps.avx_vnni) b.name = "avxvnni";
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
