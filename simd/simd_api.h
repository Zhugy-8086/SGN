// simd_api.h - SGN SIMD 平台无关内核原语接口（纯声明，无实现）
//
// 背景：fixes_相关修复/simd指令集加速文件拆分计划_2026_08_29.md（Step 0 骨架）
//
// 本文件只声明签名与契约，不含任何实现，且禁止 include 任何平台头
// （immintrin.h / arm_neon.h / target 属性 / __builtin_cpu_supports）。
// 各 ISA 实现位于 simd/x86/{avx2,avxvnni,ssse3}.cpp 与 simd/scalar.cpp；
// 调用方（split_dot.cpp / packed_backend*.cpp / ops*.cpp）只 include 本接口头，
// 平台细节封死在后端文件内（对齐 dispatch/kernel_api.h 的纪律）。
//
// 契约（所有原语共同遵守）：
//   1. 输入/输出为裸指针 + size_t 尺寸；输出缓冲由【调用方】分配/持有，
//      后端只写、不拥有、不得缓存指针；
//   2. 除特别标注外，不假设任意指针对齐（统一 loadu/storeu）；
//   3. 端口不抛异常；
//   4. 数值档位逐函数注释声明：bit-exact（与标量逐位一致）或注明差异。
//
// 分步实施状态：
//   - Step 0（本文件）：接口契约冻结。
//   - Step 1：dot16/dot8/dot4 的 SIMD 实现（avx2.cpp / avxvnni.cpp）
//             与标量锚点（scalar.cpp，#if !defined(__AVX2__/__AVXVNNI__) 保护）
//             由 msint/split_dot.cpp 三内核迁移填充。

#pragma once

#include <cstdint>
#include <cstddef>

namespace sgn::simd {

// ---- 精度档位（综合执行计划 §二.3 · EPYC 9K65 复核问题 ④）----
// 每个可调度原语属于一个精度类别，供上层在跨后端可复现/断点续训场景查询：
//   - kBitExact：与标量锚点逐位一致（结果与后端无关）。整型原语（dot 系 / reverse /
//     batch_reverse / decode 系）恒属此类——整数累加可交换/结合，SIMD 展开不改变结果。
//   - kRounding：浮点归约有舍入级差异（结果依赖后端累加器路数与顺序）。SIMD 后端
//     （avx2/avx512 的 sum 系多路累加器 + 水平归约）与标量锚点（逐序累加）非逐位一致，
//     SIMD 通常更准但仍属 kRounding；多后端混用会导致训练/断点结果轻微漂移。
// 查询入口 primitive_precision_class(Id)：Id 见下方枚举，返回精度类别（后端无关）。
// 保守口径：浮点归约原语无论当前后端统一声明 kRounding（标量桩是 bit-exact 参照，
// 但作为"低档回退"其数值与 SIMD 后端不同，跨后端可复现需以后端为准）；整型原语
// 无论后端统一 kBitExact（整数累加可交换/结合，SIMD 展开不改变结果）。
enum class PrimitiveId {
    Dot16, Dot8, Dot4,          // 整型点积 → kBitExact
    DecodeI16, DecodeI16Packed, // 整数解码 → kBitExact
    Reverse, BatchReverse,      // 字节重排 → kBitExact
    Sum, SumSqDev, SumSumprod,  // float 归约 → kRounding
    Accum,                      // float 累加 → kRounding
};
enum class PrecisionClass { kBitExact, kRounding };
PrecisionClass primitive_precision_class(PrimitiveId id) noexcept;

// ---- 整型点积原语（源：msint/split_dot.cpp；Step 1 迁移）----

// int16[K] × int16[K] → int64 精确点积。
//   AVX2/AVX-512 实现（simd/x86/avx2.cpp / avx512.cpp）：madd 快速路径 + vpmuldq 回退
//   （2026-08-31 数据实验定案，见 simd服务器加速计划 §7/§8）——批内检测 -32768，
//   无则 _mm256/_mm512_madd_epi16 全速（相邻对乘积和，无 -32768 时 pair 和 < 2^31 不溢出），
//   有则该批回退 vpmuldq 精确累加（保持 bit-exact）。
//   bit-exact：整数加法可交换/结合，累加顺序不影响结果。
int64_t dot16(const int16_t* a, const int16_t* b, size_t K);

// uint8[K] × int8[K] → int64 精确点积。后端选择链：
//   scalar → avx2(vpmaddubsw 中间档, 2026-09-02 新增) → avx_vnni(vpdpbusd) → avx512vnni
//   - AVX2 中间档（avx2_dot.cpp）：u8×s8→dot8_avx2，a≤127 批 vpmaddubsw 安全全速，
//     含 ≥128 批 cvtep8+madd_epi16 精确回退（综合执行计划 §一）。bit-exact。
//   - AVX-VNNI（avxvnni.cpp）：vpdpbusd 直接 int32 精确累加
//     （禁裸用 maddubs_epi16：255*128 pair 和饱和破坏 bit-exact）。
//   bit-exact。
int64_t dot8(const uint8_t* a, const int8_t* b, size_t K);

// 4 位预解包点积：u8[K](uint8) × s8[K](int8) → int64。与 dot8 同实现载体
// （avx2_dot.cpp / avxvnni.cpp / avx512vnni.cpp），语义等价逐元素 u8×s8 求和。
// 注意：dot4 不负责 nibble 解包——输入须已由 unpack_nibble_u/s 预解包为满宽字节
// （解包在 4 位路径由调用方负责，故 dot4 与 dot8 同实现而非自带 4 位特化）。bit-exact。
int64_t dot4(const uint8_t* u8, const int8_t* s8, size_t K);

// ---- nibble 字节解包（平台无关标量语义，simd/scalar.cpp 常驻；供 prepare_nibble 复用）----
// 打包格式：byte[j] = (elem[2j+1] << 4) | elem[2j]，每字节 2 个 nibble；
// su/ss 数组大小为 int8 路径的一半（内存带宽减半）。
// 仅解无符号 nibble → uint8（w 侧；n² 点积取 w 的无符号字节）
void unpack_nibble_u(const uint8_t* su_p, size_t K, uint8_t* u8);
// 仅解有符号 nibble → int8（x 侧；4 位符号扩展 0..7→0..7, 8..15→-8..-1）
void unpack_nibble_s(const uint8_t* ss_p, size_t K, int8_t* s8);

// ---- packed 解码原语（源：msint/packed_backend*.cpp；Step 2 迁移）----

// 把 packed uint64 数组解码为 int16 值 × scale 的 float 数组（backward_int16 8+8 schema）。
// 前提：每个 uint64 的低 16 位为一个 int16（由 2 个 8-bit 槽位 concat），高 48 位无效；
//       res[i] = float(int16(pv[i] & 0xFFFF)) * scale。
//   AVX2 实现（x86/avx2.cpp）：SSSE3 PSHUFB+PALIGNR+PMOVSXWD 热路径；AVX2 原始路径回退。
//   标量锚点在 scalar.cpp（非 AVX2 平台）。float 乘 scale 用 mul 不融合，与标量逐位一致。
void decode_i16_f32(const uint64_t* pv_ptr, int n_values, float scale, float* res_ptr);

// 连续 int16 解码（布局优化，综合执行计划 §二.5；EPYC 9K65 复核问题 ①）：
//   src[i] = 连续 int16（每 16 位 1 个有效值，无 uint64 的 48 位填充），
//   res[i] = float(src[i]) * scale。
// 与 decode_i16_f32（uint64 每 8 字节仅 16 位有效 → 4× 带宽浪费）相比，本原语输入
// 连续布局，内存带宽利用 100%——是问题 ① 的正解（改上层布局）。实现于
// x86/avx2_decode.cpp（AVX2 宽读 + cvtepi16_epi32 + cvtepi32_ps，标量尾部）；
// 非 AVX2 平台由调用方落到标量循环（scalar.cpp 提供 decode_i16_f32_packed16_scalar，见注）。
void decode_i16_f32_packed16(const int16_t* src, int n_values, float scale, float* res_ptr);

// 反转 packed 的低 n 字节（n ∈ [1,8]）到 out[0..n-1]：out[i] = (packed >> (8*(n-1-i))) & 0xFF。
//   SSSE3 实现（x86/ssse3.cpp）：单 128 位 PSHUFB。标量锚点在 scalar.cpp。bit-exact。
void reverse_bytes8(uint64_t packed, int n, int64_t* out);

// 批量反转：对每个 uint64 反转字节序，取前 n_slots 字节，
// result[i*n_slots + j] = 第 i 个 uint64 的字节 (n_slots-1-j)。
// 前提：调用方已判定 n_slots ∈ [1,8] 且全 8-bit 等宽（SSSE3/AVX2 路径逐 4 个 uint64 批量）。
//   SSSE3 实现（x86/ssse3.cpp）：AVX2 PSHUFB 批量（提取偏移 8-n_slots，2026-08-29 修复
//   原代码 n_slots<8 取错槽位的潜伏 bug，见拆分计划"后续审查"节）。标量锚点在 scalar.cpp。
//   bit-exact（纯字节重排，无算术）。
void batch_reverse_u8(const uint64_t* packed_values, int n_values, int n_slots, int64_t* result);

// ---- float 归约/累加原语（源：autograd/ops_nn.cpp + autograd/autograd.cpp；Step 3 迁移）----
// 这些原语是 BatchNorm 水平归约与梯度累加的可复用数值积木；浮点累加顺序与迁移前
// 完全一致（AVX2 8 路累加器 + 水平归约次序 + 标量尾），故结果与原地实现逐位一致。

// 等步长求和：sum = Σ_{i=0}^{n-1} p[i*stride]。
//   stride=1 连续 → AVX2 loadu；stride>1 列方向 → _mm256_set_ps 构建（与 bn_forward_train
//   dim=0/1 两分支一致）。AVX2 实现（x86/avx2.cpp），标量锚点在 scalar.cpp。
float sum_f32(const float* p, int64_t n, int64_t stride);

// 等步长偏差平方和：sum_sq = Σ_{i=0}^{n-1} (p[i*stride] - mu)²。AVX2/标量同上。
float sum_sq_dev_f32(const float* p, int64_t n, int64_t stride, float mu);

// 单遍双累加（BN backward 的 dβ=Σa 与 dγ=Σa·b 一次遍历算完，避免 2 次数据遍历）：
//   *out_sum     = Σ_{i} a[i*stride]
//   *out_sumprod = Σ_{i} a[i*stride] * b[i*stride]
// 与原 bn_backward_train 的 sd256/sdx256 双累加器 + 各自水平归约次序一致。AVX2/标量同上。
void sum_sumprod_f32(const float* a, const float* b, int64_t n, int64_t stride,
                     float* out_sum, float* out_sumprod);

// 逐元素累加：dst[i] += src[i]（i < n）。AVX2 8 路加法 + 标量尾（与 accumulate_grad 一致）。
void accum_f32(float* dst, const float* src, int64_t n);

// ---- 后端标识 / HAL 雏形（P2：simd 原语层运行时调度）----
// 每个原语的 SIMD 实现与标量锚点双层化（见 simd_dispatch.cpp）；公开接口（上文各函数）是
// 运行时调度入口，经 simd_backend() 的表选择最优实现。与 dispatch/registry 同构：
//   - 编译期：非支持平台（非 x86 无 AVX）SIMD 实现不编译 → 表仅标量项（跨平台回退保持）；
//   - 运行时：一次性 CPU 检测选后端，SGN_KERNEL_BACKEND=scalar 环境变量可强制回退标量。
// 注意：当前构建仍全局 -mavx2 -mavxvnni（binary 硬性要求 AVX2，见拆分计划 D2）；
// 摘除全局标志的真·无 AVX2 二进制需全部 intrinsic target 化，属后续独立改造。
//
// AVX-512（服务器加速，见 simd服务器加速计划_2026_08_30.md P1）：
//   avx512.cpp / avx512vnni.cpp 由 CMake 单独加 -mavx512f -mavx512vnni -mavx512bw
//   （set_source_files_properties，文件级选项），故符号常驻编译、可被本文件无条件声明；
//   运行时经 __builtin_cpu_supports("avx512*") 检测，本机（无 AVX-512）自动落低档实现。
//   多累加器展开保持整数加法可交换/结合，bit-exact 不破坏。

// ---- AVX-512 实现声明（定义见 simd/x86/avx512.cpp / avx512vnni.cpp）----
// 命名约定：_avx512 / _avx512vnni 后缀，与 _avx2/_vnni/_ssse3/_scalar 平级。
// 仅在目标 CPU 支持 AVX-512 时被 simd_dispatch.cpp 选入；本机编译仅保证符号存在。
int64_t dot16_avx512(const int16_t* a, const int16_t* b, size_t K);
int64_t dot8_avx512vnni(const uint8_t* a, const int8_t* b, size_t K);
int64_t dot4_avx512vnni(const uint8_t* u8, const int8_t* s8, size_t K);
void decode_i16_f32_avx512(const uint64_t* pv_ptr, int n_values, float scale, float* res_ptr);
float sum_f32_avx512(const float* p, int64_t n, int64_t stride);
float sum_sq_dev_f32_avx512(const float* p, int64_t n, int64_t stride, float mu);
void sum_sumprod_f32_avx512(const float* a, const float* b, int64_t n, int64_t stride,
                            float* out_sum, float* out_sumprod);
void accum_f32_avx512(float* dst, const float* src, int64_t n);

// 当前进程活跃的 simd 后端（只读，进程生命周期内不变；magic static 单次求值）。
struct SimdBackend {
    int64_t (*dot16)(const int16_t*, const int16_t*, size_t);
    int64_t (*dot8)(const uint8_t*, const int8_t*, size_t);
    int64_t (*dot4)(const uint8_t*, const int8_t*, size_t);
    void (*decode_i16_f32)(const uint64_t*, int, float, float*);
    void (*reverse_bytes8)(uint64_t, int, int64_t*);
    void (*batch_reverse_u8)(const uint64_t*, int, int, int64_t*);
    float (*sum_f32)(const float*, int64_t, int64_t);
    float (*sum_sq_dev_f32)(const float*, int64_t, int64_t, float);
    void (*sum_sumprod_f32)(const float*, const float*, int64_t, int64_t, float*, float*);
    void (*accum_f32)(float*, const float*, int64_t);
    const char* name;   // 整体后端名（诊断）："avx512vnni" / "avxvnni" / "avx2" / "ssse3" / "scalar"(forced)
};

const SimdBackend& simd_backend() noexcept;

// 返回当前活跃后端名（等价 simd_backend().name）："avx512vnni" / "avxvnni" / "avx2" / "ssse3" / "scalar"。
const char* active_backend_name() noexcept;

} // namespace sgn::simd
