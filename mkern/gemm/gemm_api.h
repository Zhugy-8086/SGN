// gemm_api.h - mkern 矩阵级微内核接口（纯声明，无实现）
//
// 背景：fixes_相关修复/mkern微内核层实施计划_2026_09_03.md §3.2（R3）。
// mkern 层定位：dispatch（Tensor 算子级）与 simd（一维原语级）之间的矩阵级层——
// 本文件只声明签名与契约，不含实现，且禁止 include 任何平台头。
// 各 ISA 实现位于 gemm/x86/{gemm_avx2vnni,gemm_avx2,gemm_avx512vnni}.cpp 与
// gemm/scalar.cpp；调用方只 include 本接口头，平台细节封死在后端文件内。
//
// 契约（与 simd_api.h 同纪律）：
//   1. 输入/输出为裸指针 + int64 尺寸，行主序连续；输出缓冲由【调用方】分配/持有，
//      后端只写、不拥有、不得缓存指针、不抛异常（无内部分配）；
//   2. 不假设任意指针对齐（统一 loadu/storeu）；
//   3. 整型运算 bit-exact：整数加法可交换/结合，SIMD 分块/多累加器不改变结果；
//      各原语的安全域（K 上界）逐函数注释声明。

#pragma once

#include <cstdint>

namespace sgn::mkern::gemm {

// ----------------------------------------------------------------------------
// pack_b_i8：把行主 B[K×N] 打包成 VNNI 面板布局（标量常驻，纯搬运一次摊销）。
//
//   布局：P = ceil(N/16) 个列面板（末面板对 N 之外的列零填充），G = ceil(K/4)
//   个 k 组（尾组对 k ≥ K 零填充）。面板 p、组 g 的 64 字节：
//     byte[t*4 + j'] = (4g+j' < K && 16p+t < N) ? B[(4g+j')*N + 16p + t] : 0
//   （t = 面板内列 0..15，j' = 组内 k 0..3 —— 16 列 × 4 k 的转置块）
//   零填充乘积为 0 → 内核对补零区照算不存即可，bit-exact 不破坏。
//
//   为什么是这个布局：AVX2/AVX512-VNNI 的 vpdpbusd 沿字节位置配对累加——
//   va 广播 A 的 4 字节 k 组、vb 取该 64 字节块时，dpbusd 的 16 个 int32 lane
//   恰好对应 16 个输出列（lane t = Σ_{j'<4} A[m,4g+j']·B[4g+j',16p+t]），
//   累加器即 C 列向量，免水平归约、dpbusd 端口成为唯一瓶颈（每次 32/64 MAC）。
//   简单转置（Bp[n*K+k]）做不到这一点（lane 对应 k 组而非列）。
//
//   大小：ceil(N/16)·ceil(K/4)·64 字节（pack_b_i8_bytes）。调用方分配/持有 Bp；
//   同一 B 多次 gemm 调用可复用（同 prepare_nibble 摊销哲学）。
int64_t pack_b_i8_bytes(int64_t N, int64_t K);
void pack_b_i8(int8_t* Bp_out, const int8_t* B_in, int64_t N, int64_t K);

// ----------------------------------------------------------------------------
// gemm_i8：C[M×N] int32 = A[M×K] uint8 × B[K×N] int8（B 经 pack_b_i8 预打包）
//   C[m*N + n] = Σ_{k<K} A[m*K + k] * B[k*N + n]
//   accum=false 覆盖写 C；accum=true 累加（C += A@B^T）。
//   内核：4 行 × 16/32 列寄存器 tile（4×2=8 个 int32 向量累加器），A 的 4 字节
//   k 组广播复用于全部列、B 面板组被 4 行复用；末面板补零列照算不存、M%4
//   尾行走同构向量路径（行数 1..3，vb 组仍复用；勿用标量收集——实测有 67× 悬崖）。
//   安全域：K ≤ 65536——每 int32 lane 累计全部 K 个乘积，全幅 255×128 时
//   ≤ 65536·32640 = 2.139e9 < 2^31；更大 K 调用方自行 K 分块（分块累加语义
//   与 accum=true 相同）。结果真值超 int32 时按补码截断（调用方保证量纲）。
//   后端链：scalar → avx2vnni → avx512vnni（AVX2 无 VNNI 落标量，maddubs
//   中间档为后续项）。bit-exact。
void gemm_i8(int32_t* C, const uint8_t* A, const int8_t* Bp,
             int64_t M, int64_t N, int64_t K, bool accum);

// ----------------------------------------------------------------------------
// gemm_i16：C[M×N] int64 = A[M×K] int16 × B[K×N] int16（B 行主直接消费，无需打包）
//   C[m*N + n] = Σ_{k<K} A[m*K + k] * B[k*N + n]（int64 全程精确，无 K 上界——
//   与 simd::dot16 同语义；int64 输出是 MSint 拆分点积融合路径的精确性要求）。
//   内核（2026-09-03 madd-quad 重设计，取代首版 mullo N 轴方案）：4 行 × 8 列
//   tile，k 按对处理——va 广播 A 的 (a[k0],a[k1]) 4 字节、vb 由 B 相邻两 k 行
//   4 列切片 unpacklo 交错，_mm_madd_epi16 一条完成 4 列 × 2 k = 8 MAC
//   （lane=列）→ 扩 int64 累加。溢出守卫：pair 和溢出当且仅当 a[k0]=a[k1]=
//   -32768（两积同时 +2^30；负向不可能）——调用前一次性预扫描 A（O(M·K)），
//   干净输入走无守卫热路径，脏输入逐 (行,k对) 回退标量。M%4 尾行同构向量
//   路径；N%8 尾列标量（无越界，修复首版部分列向量全宽 load 读穿行尾的
//   潜在 OOB）。后端链：scalar → avx2（AVX2 上 int16 无字节 VNNI 等价物，
//   uop 效率天花板 ≈ naive dot16；真正的加速在 AVX512-VNNI vpdpwssd，
//   EPYC 后续项）。bit-exact。
void gemm_i16(int64_t* C, const int16_t* A, const int16_t* B,
              int64_t M, int64_t N, int64_t K, bool accum);

// ----------------------------------------------------------------------------
// gemv_i8：y[M] int64 = A[M×K] uint8 × x[K] int8（矩阵-向量，M 个独立 K 维点积）
//   y[m] = Σ_{k<K} A[m*K + k] * x[k]。GEMV 无矩阵级数据复用可挖（每输出一个
//   独立 K 维点积），逐行消费 simd::dot8（mkern 向下消费 simd 原语的分层示范，
//   见计划 §一）；x 驻 L1、A 流式——即 dot8 的最优使用形态。bit-exact，
//   无 K 上界（同 dot8 契约）。
void gemv_i8(int64_t* y, const uint8_t* A, const int8_t* x, int64_t M, int64_t K);

// ---- 后端标识 / 调度（与 simd_dispatch 同构：magic static + CPUID + 环境变量钩子）----
// 编译期：非 x86 / 无对应 ISA 的实现不编译 → 表仅标量项；运行时一次性 CPU 检测选后端，
// SGN_GEMM_BACKEND=scalar 环境变量可强制回退标量（测试钩子，同 SGN_KERNEL_BACKEND 纪律）。
struct GemmBackend {
    void (*gemm_i8)(int32_t*, const uint8_t*, const int8_t*, int64_t, int64_t, int64_t, bool);
    void (*gemm_i16)(int64_t*, const int16_t*, const int16_t*, int64_t, int64_t, int64_t, bool);
    void (*gemv_i8)(int64_t*, const uint8_t*, const int8_t*, int64_t, int64_t);
    void (*pack_b_i8)(int8_t*, const int8_t*, int64_t, int64_t);
    const char* name;   // "avx512vnni" / "avx2vnni" / "avx2" / "scalar"(forced)
};

const GemmBackend& gemm_backend() noexcept;
const char* active_gemm_backend_name() noexcept;

// ---- x86 实现声明（定义见 gemm/x86/*.cpp；命名约定同 simd 的 _avx2/_vnni 后缀）----
void gemm_i8_avx2vnni(int32_t*, const uint8_t*, const int8_t*, int64_t, int64_t, int64_t, bool);
void gemm_i8_avx512vnni(int32_t*, const uint8_t*, const int8_t*, int64_t, int64_t, int64_t, bool);
void gemm_i16_avx2(int64_t*, const int16_t*, const int16_t*, int64_t, int64_t, int64_t, bool);

} // namespace sgn::mkern::gemm
