// registry.cpp - 内核注册表（一次性能力解析）
//
// 节奏 0：搭接口骨架，不改任何内核逻辑。现有内核（ops.cpp 中定义）经薄适配
// 函数接入注册表；适配函数只做"小 k 选择 / dA+dB 组合"等调度，不改内核实现。
//
// 能力解析【进程生命周期只执行一次】（C++11 magic static，线程安全）。
// 若写成"每次调用 if-else 分支"，违背一次性检测承诺（调研文档 §2.5 #3 已修正）。

#include "dispatch/registry.h"
#include "common/logger.h"

// CPUID 检测用内联 intrinsic（跨平台，与 simd/simd_dispatch.cpp 同纪律）：
//   - Windows/MSVC target: <intrin.h> 提供 __cpuid/__cpuidex/_xgetbv
//   - Linux (GCC/Clang):    <cpuid.h> 提供 __cpuid_count；_xgetbv 在 <immintrin.h>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

#if defined(__x86_64__) || defined(_M_X64)
#if defined(_MSC_VER)
#include <intrin.h>
#else
#include <cpuid.h>
#include <immintrin.h>  // _xgetbv
#endif
#endif

namespace sgn_autograd {

// ============================================================================
// 适配函数前向声明（实现在 ops.cpp 定义；薄包装现有内核，不改内核逻辑）
// 标量后端（ref_scalar，bit-exact 锚点）
// ============================================================================
Tensor matmul_fwd_scalar(const Tensor& A, const Tensor& B);
Tensor matmul_transpose_b_scalar(const Tensor& A, const Tensor& B);
Tensor matmul_transpose_a_scalar(const Tensor& A, const Tensor& B);
std::pair<Tensor, Tensor> matmul_backward_scalar(
    const Tensor& A, const Tensor& B, const Tensor& dY);

// AVX2 后端（含小 k 选择 / dA+dB 组合）
Tensor matmul_fwd_avx2(const Tensor& A, const Tensor& B);
Tensor matmul_transpose_b_avx2(const Tensor& A, const Tensor& B);
Tensor matmul_transpose_a_avx2(const Tensor& A, const Tensor& B);
std::pair<Tensor, Tensor> matmul_backward_avx2(
    const Tensor& A, const Tensor& B, const Tensor& dY);

// AVX-512 后端（仅 forward 有独立内核；transpose_a/b 与 backward 复用 avx2，
// 因为支持 AVX-512 的 CPU 必然支持 AVX2）
Tensor matmul_fwd_avx512f(const Tensor& A, const Tensor& B);

// conv2d 族端口（实现在 ops_nn.cpp 定义；裸指针版）
void dw_scalar(const float* A, const float* B, float* C,
               int64_t out_c, int64_t bs, int64_t k);
void dw_avx2(const float* dy, const float* xc, float* c,
             int64_t out_c, int64_t bs, int64_t k);
// dx_col（P0-A j-tile 重排 + 标量参照）
void dx_scalar(const float* A, const float* B, float* C,
               int64_t r, int64_t out_c, int64_t bs);
void dx_avx2(const float* A, const float* B, float* C,
             int64_t r, int64_t out_c, int64_t bs);
// db 行和（反向步骤 ③，通道验收收口 → 6/6 全实现）
void db_scalar(const float* dY_col, float* db, int64_t out_c, int64_t bs);
void db_avx2(const float* dY_col, float* db, int64_t out_c, int64_t bs);
// col2im 行条带法（P0-B + 标量参照；累加语义，调用前 dX_pad 须清零）
void col2im_scalar(const float* dx_col, float* dX_pad,
                   int64_t batch, int64_t in_c,
                   int64_t out_h, int64_t out_w,
                   int64_t bs, int64_t kh, int64_t kw,
                   int64_t stride, int64_t pad,
                   int64_t padded_h, int64_t padded_w);
void col2im_avx2(const float* dx_col, float* dX_pad,
                 int64_t batch, int64_t in_c,
                 int64_t out_h, int64_t out_w,
                 int64_t bs, int64_t kh, int64_t kw,
                 int64_t stride, int64_t pad,
                 int64_t padded_h, int64_t padded_w);
// dY→dY_col 转置（P1-C 纯搬运 + 标量参照）
void dycol_scalar(const float* dY, float* dY_col,
                  int64_t batch, int64_t out_c, int64_t spatial);
void dycol_avx2(const float* dY, float* dY_col,
                int64_t batch, int64_t out_c, int64_t spatial);
// im2col 行条带法（P1-E + 标量参照；纯拷贝越界填 0）
void im2col_scalar(const float* X, float* x_col,
                   int64_t batch, int64_t in_c,
                   int64_t in_h, int64_t in_w,
                   int64_t out_h, int64_t out_w,
                   int64_t kh, int64_t kw, int64_t stride, int64_t pad);
void im2col_avx2(const float* X, float* x_col,
                 int64_t batch, int64_t in_c,
                 int64_t in_h, int64_t in_w,
                 int64_t out_h, int64_t out_w,
                 int64_t kh, int64_t kw, int64_t stride, int64_t pad);

namespace {

// 未实现端口的占位：调用即抛异常（端口"存在但未填"显式暴露）
// 通道验收（2026-08-19）后 conv2d 族 6 端口已全实现，not_impl 保留
// 供未来新算子族（bn/pool/HC 等）未落地端口占位。
[[maybe_unused]] void not_impl() { throw std::logic_error("kernel backend not implemented"); }

// CPU 能力快照（一次性填充；2026-08-31 阶段 2 修正：与 simd/simd_dispatch.cpp 同纪律——
// 纯运行时 CPUID 检测，编译期宏不再参与。此前编译期短路会在全局 -mavx2 -mavxvnni
// 编译下无条件置位；而用 __builtin_cpu_supports 则依赖 __cpu_model 运行时符号，
// -nostdlib 链接的 .pyd 无法解析）。本快照仅服务内核层（浮点 matmul/conv 内核选择），
// 与 ops.cpp 内核的 target() 属性独立编译保持一致（avx2 符号常驻，运行时决定选否）。
struct CpuCaps {
    bool avx2_fma;
    bool avx512f;
};

// 位定义：leaf1 ECX: XSAVE=27, AVX=28, FMA=12；leaf7 sub0 EBX: AVX2=5, AVX512F=16；
// XCR0: XMM=1, YMM=2, opmask=4, ZMM hi=8。
#if defined(__x86_64__) || defined(_M_X64)
static void cpuid_leaf(int leaf, int* r) {
#if defined(_MSC_VER)
    __cpuid(r, leaf);
#else
    __cpuid_count(leaf, 0, r[0], r[1], r[2], r[3]);
#endif
}
static void cpu_caps_x86(CpuCaps& c) {
    int r[4];
    cpuid_leaf(1, r);
    const bool os_xsave = (r[2] & (1 << 27)) != 0;
    const bool cpu_avx  = (r[2] & (1 << 28)) != 0;
    const bool cpu_fma  = (r[2] & (1 << 12)) != 0;
    if (os_xsave && cpu_avx && (_xgetbv(0) & 0x6) == 0x6) {
        cpuid_leaf(7, r);  // leaf7 subleaf0（同 simd_dispatch 注释：sub1 ECX 恒 0 不可用）
        c.avx2_fma = ((r[1] & (1 << 5)) != 0) && cpu_fma;
        c.avx512f  = (r[1] & (1 << 16)) != 0;
        // AVX-512 还需 opmask + ZMM hi256 状态（XCR0 0xE6）
        if (c.avx512f && (_xgetbv(0) & 0xE6) != 0xE6) c.avx512f = false;
    }
}
#endif

CpuCaps cpu_caps() {
    CpuCaps caps = {false, false};
#if defined(__x86_64__) || defined(_M_X64)
    cpu_caps_x86(caps);
#endif
    return caps;
}

}  // anonymous namespace

const KernelSet& kernel_registry() {
    // magic static: 仅第一次进入时求值一次，此后直接返回缓存引用（无分支）
    static const KernelSet ks = [] {
        // 环境变量强制后端（测试钩子，不做运行期热切换）。首次解析【之前】
        // 读取一次，之后进程内不再看环境变量。仅支持强制 scalar。
        static char forced_buf[32] = {0};
        size_t required = 0;
        getenv_s(&required, forced_buf, sizeof(forced_buf), "SGN_KERNEL_BACKEND");
        const char* forced = (required > 0 && required <= sizeof(forced_buf))
                                 ? forced_buf
                                 : nullptr;

        KernelSet k;
        if (forced && std::strcmp(forced, "scalar") == 0) {
            k.matmul_fwd         = matmul_fwd_scalar;
            k.matmul_transpose_b = matmul_transpose_b_scalar;
            k.matmul_transpose_a = matmul_transpose_a_scalar;
            k.matmul_backward    = matmul_backward_scalar;
            k.num_level          = NumLevel::kBitExact;
            k.name               = "ref_scalar(forced)";
            return k;
        }
        // SSE2 兼容后端（L1 层）：纯标量实现（bit-exact），与 ref_scalar 共用内核。
        // 强制选择用于在支持 AVX2 的机器上验证 L1 回退路径（通道验收测试钩子）。
        if (forced && std::strcmp(forced, "sse2") == 0) {
            k.matmul_fwd         = matmul_fwd_scalar;
            k.matmul_transpose_b = matmul_transpose_b_scalar;
            k.matmul_transpose_a = matmul_transpose_a_scalar;
            k.matmul_backward    = matmul_backward_scalar;
            k.num_level          = NumLevel::kBitExact;
            k.name               = "x86_sse2(forced)";
            return k;
        }

        const CpuCaps& caps = cpu_caps();
#if defined(__AVX512F__)
        if (caps.avx512f) {
            k.matmul_fwd         = matmul_fwd_avx512f;
            k.matmul_transpose_b = matmul_transpose_b_avx2;
            k.matmul_transpose_a = matmul_transpose_a_avx2;
            k.matmul_backward    = matmul_backward_avx2;
            k.num_level          = NumLevel::kRounding;
            k.name               = "x86_avx512f";
        } else
#endif  // __AVX512F__
        if (caps.avx2_fma) {
            k.matmul_fwd         = matmul_fwd_avx2;
            k.matmul_transpose_b = matmul_transpose_b_avx2;
            k.matmul_transpose_a = matmul_transpose_a_avx2;
            k.matmul_backward    = matmul_backward_avx2;
            k.num_level          = NumLevel::kRounding;
            k.name               = "x86_avx2";
        } else {
            // x86-64 基线 SSE2：无 AVX2 时回退到 L1 层（纯标量实现，bit-exact）。
            // 非 x86（未来 ARM 等）由各自后端目录 + 注册接入，不落入此分支。
            k.matmul_fwd         = matmul_fwd_scalar;
            k.matmul_transpose_b = matmul_transpose_b_scalar;
            k.matmul_transpose_a = matmul_transpose_a_scalar;
            k.matmul_backward    = matmul_backward_scalar;
            k.num_level          = NumLevel::kBitExact;
            k.name               = "x86_sse2";
        }
        SGN_LOG_INFO("kernel backend selected: %s", k.name);
        return k;
    }();
    return ks;
}

const Conv2dKernelSet& conv2d_registry() {
    // magic static：与 kernel_registry() 相同的单次求值语义
    static const Conv2dKernelSet ks = [] {
        // 与 matmul 共用同一个环境变量强制后端（保持测试钩子一致）
        static char forced_buf[32] = {0};
        size_t required = 0;
        getenv_s(&required, forced_buf, sizeof(forced_buf), "SGN_KERNEL_BACKEND");
        const char* forced = (required > 0 && required <= sizeof(forced_buf))
                                 ? forced_buf
                                 : nullptr;

        Conv2dKernelSet k;
        // 通道验收（2026-08-19）：db 收口后 6/6 端口全实现。
        // 节奏 1：dw；节奏 3：dx(P0-A) + col2im(P0-B) + dycol(P1-C) + im2col(P1-E)。
        if (forced && std::strcmp(forced, "scalar") == 0) {
            k.dw        = dw_scalar;
            k.dx        = dx_scalar;
            k.db        = db_scalar;
            k.dycol     = dycol_scalar;
            k.col2im    = col2im_scalar;
            k.im2col    = im2col_scalar;
            k.num_level = NumLevel::kBitExact;
            k.name      = "ref_scalar(forced)";
            return k;
        }
        // SSE2 兼容后端（L1 层）：纯标量实现（bit-exact），与 ref_scalar 共用内核。
        // 强制选择用于在支持 AVX2 的机器上验证 L1 回退路径（通道验收测试钩子）。
        if (forced && std::strcmp(forced, "sse2") == 0) {
            k.dw        = dw_scalar;
            k.dx        = dx_scalar;
            k.db        = db_scalar;
            k.dycol     = dycol_scalar;
            k.col2im    = col2im_scalar;
            k.im2col    = im2col_scalar;
            k.num_level = NumLevel::kBitExact;
            k.name      = "x86_sse2(forced)";
            return k;
        }

        const CpuCaps& caps = cpu_caps();
        if (caps.avx2_fma) {
            k.dw        = dw_avx2;
            k.dx        = dx_avx2;
            k.db        = db_avx2;
            k.dycol     = dycol_avx2;
            k.col2im    = col2im_avx2;
            k.im2col    = im2col_avx2;
            k.num_level = NumLevel::kRounding;   // avx2 dW/dx/db 归约序与标量不同
            k.name      = "x86_avx2";
        } else {
            // x86-64 基线 SSE2：无 AVX2 时回退到 L1 层（纯标量实现，bit-exact）。
            k.dw        = dw_scalar;
            k.dx        = dx_scalar;
            k.db        = db_scalar;
            k.dycol     = dycol_scalar;
            k.col2im    = col2im_scalar;
            k.im2col    = im2col_scalar;
            k.num_level = NumLevel::kBitExact;
            k.name      = "x86_sse2";
        }
        return k;
    }();
    return ks;
}

}  // namespace sgn_autograd