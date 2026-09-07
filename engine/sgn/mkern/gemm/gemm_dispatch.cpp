// gemm_dispatch.cpp - mkern/gemm 运行时调度（magic static + CPUID + 环境变量钩子）
//
// 与 mkern/simd/simd_dispatch.cpp 同构：
//   - 编译期：无对应 ISA 的实现不编译 → 表仅标量项（跨平台回退保持）；
//   - 运行时：一次性 CPU 检测选后端；SGN_GEMM_BACKEND=scalar 环境变量强制回退
//     标量（测试钩子；其他取值告警并忽略，同 SGN_KERNEL_BACKEND 的 2026-09-02
//     EPYC 复核修正——不静默回退自动检测）。
//
// 后端选择链：
//   gemm_i8  : scalar → avx2vnni（AVX2+VNNI）→ avx512vnni（AVX512BW+VNNI）
//   gemm_i16 : scalar → avx2（AVX512 机器落 avx2 内核，专用版为后续项）
//   gemv_i8  : 标量循环消费 simd::dot8（dot8 自身在 simd 层调度，无独立后端）
//   pack_b_i8: 标量常驻（纯搬运，无 SIMD 版）

#include "mkern/gemm/gemm_api.h"
#include "mkern/simd/simd_api.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace sgn::mkern::gemm {
// 声明各后端实现（定义见 scalar.cpp / x86/*.cpp；无 ISA 的文件不参与链接）。
// 注意：标量锚点定义于 scalar.cpp 的具名 namespace（外部链接，boundary 测试也引用），
// 前置声明必须在具名 namespace 内，不能放下方匿名 namespace（否则链接错位）。
void pack_b_i8_scalar(int8_t*, const int8_t*, int64_t, int64_t);
void gemm_i8_scalar(int32_t*, const uint8_t*, const int8_t*, int64_t, int64_t, int64_t, bool);
void gemm_i16_scalar(int64_t*, const int16_t*, const int16_t*, int64_t, int64_t, int64_t, bool);
void gemv_i8_scalar(int64_t*, const uint8_t*, const int8_t*, int64_t, int64_t);

namespace {

#if defined(__x86_64__) || defined(_M_X64)
struct CpuCaps {
    bool avx2;
    bool avx_vnni;
    bool avx512_bw;
    bool avx512_vnni;
};

// CPUID 运行时检测——不用 __builtin_cpu_supports：其依赖 __cpu_model 运行时符号，
// Windows/clang+lld-link（及 -nostdlib 的 .pyd）无法解析（同 simd_dispatch.cpp
// 注释结论 / pysgn_net.cpp 的 _cpu_has_avx2）。位定义与 simd_dispatch 完全一致：
//   leaf1 ECX: XSAVE=27, AVX=28；leaf7.0 EBX: AVX2=5, AVX512BW=30；
//   leaf7.0 ECX: AVX512VNNI=11；leaf7.1 EAX: AVX-VNNI=4；XCR0 需 0x6（XMM+YMM）。
#if defined(_MSC_VER)
#include <intrin.h>
static void cpuid_leaf(int leaf, int sub, int* r) { __cpuidex(r, leaf, sub); }
static unsigned long long xgetbv0() { return _xgetbv(0); }
#else
#include <cpuid.h>
static void cpuid_leaf(int leaf, int sub, int* r) {
    __cpuid_count(leaf, sub, r[0], r[1], r[2], r[3]);
}
static unsigned long long xgetbv0() {
    unsigned int lo, hi;
    __asm__ __volatile__("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    return (static_cast<unsigned long long>(hi) << 32) | lo;
}
#endif

CpuCaps cpu_caps() {
    CpuCaps caps{};
    int r[4];
    cpuid_leaf(1, 0, r);
    const bool os_xsave = (r[2] & (1 << 27)) != 0;
    const bool cpu_avx  = (r[2] & (1 << 28)) != 0;
    if (os_xsave && cpu_avx && (xgetbv0() & 0x6) == 0x6) {
        cpuid_leaf(7, 0, r);
        caps.avx2      = (r[1] & (1 << 5)) != 0;
        caps.avx512_bw = (r[1] & (1 << 30)) != 0;
        caps.avx512_vnni = (r[2] & (1 << 11)) != 0 && caps.avx512_bw;
        cpuid_leaf(7, 1, r);
        caps.avx_vnni = (r[0] & (1 << 4)) != 0;
    }
    return caps;
}
#endif  // x86

// SIMD 后端实现声明已在 gemm_api.h（本文件已 include，具名 namespace）——
// per-file ISA 编译、符号常驻产出。与 simd_dispatch 2026-08-31 修正同纪律：
// 此处【无条件】引用，绝不包 #if defined(__AVX2__) 编译期短路——dispatch 文件
// 自身无 ISA 旗标，宏恒不成立会导致 x86 内核永远选不进去（本次首写即踩，
// boundary 测试的 backend 打印抓出）；运行时选入由 cpu_caps 的 CPUID 决定。
// 注意不可在本匿名 namespace 内重复声明（会与具名 namespace 的同名声明构成
// 二义重载集）。

}  // anonymous namespace

const GemmBackend& gemm_backend() noexcept {
    static const GemmBackend s = [] {
        // 环境变量强制后端（测试钩子）：仅支持 scalar；其他取值告警并忽略
        const char* forced = std::getenv("SGN_GEMM_BACKEND");
        if (forced && *forced != '\0' && std::strcmp(forced, "scalar") != 0) {
            std::fprintf(stderr,
                         "[sgn::mkern::gemm] warning: unknown SGN_GEMM_BACKEND='%s', "
                         "ignoring (supported: scalar); using CPUID auto-detect\n",
                         forced);
        }

        GemmBackend b{};
        b.gemm_i8    = gemm_i8_scalar;
        b.gemm_i16   = gemm_i16_scalar;
        b.gemv_i8    = gemv_i8_scalar;
        b.pack_b_i8  = pack_b_i8_scalar;
        b.name       = "scalar";

        if (forced && std::strcmp(forced, "scalar") == 0) {
            b.name = "scalar(forced)";
            return b;
        }

#if defined(__x86_64__) || defined(_M_X64)
        const CpuCaps caps = cpu_caps();
        // gemm_i16：仅要求 AVX2（mullo 路径）；AVX512 机器落 avx2 内核照常可用
        if (caps.avx2) {
            b.gemm_i16 = gemm_i16_avx2;
            b.name     = "avx2";
        }
        // gemm_i8：AVX2+VNNI → avx2vnni；AVX512BW+VNNI 覆盖为 avx512vnni
        if (caps.avx2 && caps.avx_vnni) {
            b.gemm_i8 = gemm_i8_avx2vnni;
            b.name    = "avx2vnni";
        }
        if (caps.avx512_vnni && caps.avx512_bw) {
            b.gemm_i8 = gemm_i8_avx512vnni;
            b.name    = "avx512vnni";
        }
#endif  // x86
        return b;
    }();
    return s;
}

const char* active_gemm_backend_name() noexcept {
    return gemm_backend().name;
}

// ---- 公共调度入口 ----

void pack_b_i8(int8_t* Bp_out, const int8_t* B_in, int64_t N, int64_t K) {
    gemm_backend().pack_b_i8(Bp_out, B_in, N, K);
}

void gemm_i8(int32_t* C, const uint8_t* A, const int8_t* Bp,
             int64_t M, int64_t N, int64_t K, bool accum) {
    gemm_backend().gemm_i8(C, A, Bp, M, N, K, accum);
}

void gemm_i16(int64_t* C, const int16_t* A, const int16_t* B,
              int64_t M, int64_t N, int64_t K, bool accum) {
    gemm_backend().gemm_i16(C, A, B, M, N, K, accum);
}

void gemv_i8(int64_t* y, const uint8_t* A, const int8_t* x, int64_t M, int64_t K) {
    gemm_backend().gemv_i8(y, A, x, M, K);
}

} // namespace sgn::mkern::gemm
