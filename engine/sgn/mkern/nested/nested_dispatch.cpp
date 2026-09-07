// nested_dispatch.cpp - mkern/nested 运行时调度（magic static + CPUID + 环境变量钩子）
//
// 与 mkern/gemm/gemm_dispatch.cpp 同构：
//   - 编译期：无对应 ISA 的实现不编译 → 表仅标量项（跨平台回退保持）；
//   - 运行时：一次性 CPU 检测选后端；SGN_NESTED_BACKEND=scalar 强制回退标量
//     （测试钩子；其他取值告警并忽略，同 SGN_GEMM_BACKEND 的 2026-09-02 EPYC
//     复核修正——不静默回退自动检测）。
//
// 后端选择链：scalar → avx2（AVX2）→ avx512（AVX512F+DQ；mullo_epi64/cvtqq2pd
// 等 DQ 原生指令是 avx512 版的依赖，见 x86/nested_avx512.cpp 头注释）。

#include "mkern/nested/nested_api.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace sgn::mkern::nested {

namespace {

#if defined(__x86_64__) || defined(_M_X64)
struct CpuCaps {
    bool avx2;
    bool avx512_f;
    bool avx512_dq;
};

// CPUID 运行时检测——不用 __builtin_cpu_supports：其依赖 __cpu_model 运行时符号，
// Windows/clang+lld-link（及 -nostdlib 的 .pyd）无法解析（同 gemm_dispatch.cpp
// 注释结论）。位定义与 gemm_dispatch 一致，另加 DQ：
//   leaf1 ECX: XSAVE=27, AVX=28；leaf7.0 EBX: AVX2=5, AVX512F=16, AVX512DQ=17；
//   XCR0 需 0x6（XMM+YMM）；AVX512 需 XCR0 bit7 (opmask) + bit6 (ZMM) = 0xE0。
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
        caps.avx2     = (r[1] & (1 << 5)) != 0;
        const bool f  = (r[1] & (1 << 16)) != 0;
        const bool dq = (r[1] & (1 << 17)) != 0;
        if (f && dq && (xgetbv0() & 0xE0) == 0xE0) {
            caps.avx512_f  = true;
            caps.avx512_dq = true;
        }
    }
    return caps;
}
#endif  // x86

}  // anonymous namespace

const NestedBackend& nested_backend() noexcept {
    static const NestedBackend s = [] {
        const char* forced = std::getenv("SGN_NESTED_BACKEND");
        if (forced && *forced != '\0' && std::strcmp(forced, "scalar") != 0) {
            std::fprintf(stderr,
                         "[sgn::mkern::nested] warning: unknown SGN_NESTED_BACKEND='%s', "
                         "ignoring (supported: scalar); using CPUID auto-detect\n",
                         forced);
        }

        NestedBackend b{};
        b.nested_quant_i32 = nested_quant_i32_scalar;
        b.nested_dequant   = nested_dequant_scalar;
        b.name             = "scalar";

        if (forced && std::strcmp(forced, "scalar") == 0) {
            b.name = "scalar(forced)";
            return b;
        }

#if defined(__x86_64__) || defined(_M_X64)
        const CpuCaps caps = cpu_caps();
        if (caps.avx2) {
            b.nested_quant_i32 = nested_quant_i32_avx2;
            b.nested_dequant   = nested_dequant_avx2;
            b.name             = "avx2";
        }
        if (caps.avx512_f && caps.avx512_dq) {
            b.nested_quant_i32 = nested_quant_i32_avx512;
            b.nested_dequant   = nested_dequant_avx512;
            b.name             = "avx512";
        }
#endif  // x86
        return b;
    }();
    return s;
}

const char* active_nested_backend_name() noexcept {
    return nested_backend().name;
}

// ---- 公共调度入口 ----

void nested_quant_i32(int64_t* code, const float* h, int64_t n,
                      float u, uint64_t seed) {
    nested_backend().nested_quant_i32(code, h, n, u, seed);
}

void nested_dequant(float* out, const int64_t* code, int64_t n,
                    float u, int level) {
    nested_backend().nested_dequant(out, code, n, u, level);
}

} // namespace sgn::mkern::nested
