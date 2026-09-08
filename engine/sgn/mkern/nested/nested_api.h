// nested_api.h - mkern 嵌套量化原语接口（纯声明，无实现）
//
// 背景：fixes_相关修复/level_scheduler_2_0/nested_quant立项_2026_09_05.md §一
// （接口契约冻结稿，签名勿改）。方向 A 判决（同目录 方向A判决_嵌套量化_2026_09_05.md）
// A-GO 三证据：切换一致性 1.58M×（探针 seeded 复现）+ 归一化口径精度 + 单码多档成本。
//
// 结构（v2 设计）：一次 SR 量化到最细格 u 产出单一 int32 码字 I；各档 dequant =
// 对 I 逐级 round-to-nearest 截位。spacing 整数倍嵌套 s_b = u·2^(32−b)（粗格点 ⊂
// 细格点），max_range 全档一致 ±2^31·u（范围不缩、精度嵌套——NestQuant 式，区别于
// 旧 Level 档位的范围缩窄）。level ∈ {4,8,16,32}：b 档码字是 32 档码字的最高 b 位
// 前缀（截位可组合：RTN(RTN(I,2^m1),2^m2) = RTN(I,2^m2)，m1 ≤ m2）。
//
// 契约（与 gemm_api.h 同纪律）：
//   1. 裸指针 + int64 尺寸，输出缓冲调用方分配/持有，后端只写不拥有、不抛异常、
//      无内部分配；
//   2. 不假设指针对齐（标量逐元素，天然满足；SIMD 后端统一 loadu/storeu）；
//   3. 整型 bit-exact（kBitExact，N1）：码字是下列【冻结规格】的纯函数，跨后端
//      逐位一致——规格钉死后端可自由实现，任何偏离即违约：
//
//   冻结规格（bit-exact 的全部自由度在此钉死）：
//   a. RNG（计数器式，per-element 恰 1 个随机数，无顺序依赖——SIMD 后端可按 lane
//      独立重算）：元素 i 的 U_i ∈ [0,1) 由 SplitMix64 终混派生：
//        z = seed + 0x9E3779B97F4A7C15 * (uint64_t)i
//        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9
//        z = (z ^ (z >> 27)) * 0x94D049BB133111EB
//        z =  z ^ (z >> 31)                    // 全程 uint64 回绕
//        U_i = (z >> 11) * 0x1.0p-53           // 53 位尾数，[0,1)
//   b. SR 判式（Bernoulli 形式，与 sr_kernel.cpp 同构）：x = (double)h[i] / (double)u
//      （f64 口径）；I = floor(x)，若 U_i < (x − floor(x)) 则 I += 1。与立项注释的
//      floor(x + U[0,1)) 同分布同 SR 语义（floor(x+U) = floor(x) + [1−U < frac]），
//      采用 Bernoulli 形式是冻结决策：|x| 接近 2^31 时字面形式在 f64 下丢失 U 低
//      位（ulp(x) ≈ 2^−21），Bernoulli 形式对 |x| < 2^52 精确。
//   c. 饱和（max_range 全档一致 ±2^31·u）：x ≥ 2147483647.0 → I = 2147483647；
//      x ≤ −2147483648.0 → I = −2147483648（设计域外饱和，域内 SR 精确）。
//      码字恒在 int32 域（nested_quant_i32 之名）。
//      注（2026-09-06 审查）：精确 SR 域实为 [−2^31·u, (2^31−1)·u]——正侧比
//      负侧短 1 个最细格 u（int32 正顶 2^31−1 的本征不对称）。x ∈ [2^31−1, 2^31]
//      被钳到 I = 2^31−1，不做 SR。数值上无碍（x = −2^31 恰为格点，frac=0）。
//   d. dequant 截位（纯整数域，f64 只出现在最后一步值重建）：m = 32 − level；
//      q = floor(I / 2^m) 的最近整数，tie（I ≡ 2^(m−1) mod 2^m）取偶数
//     （round-half-to-even，对齐探针 np.round 口径与 IEEE 默认；tie 规则是冻结
//      决策，测试覆盖）。out[i] = (float)((double)(q·2^m) * (double)u)。
//      level 32 → m = 0 → out = I·u（精确恒等）。
//
//   前置条件（调用方保证，热路径不校验）：
//     - h 元素有限（NaN/Inf 行为未定义）；u > 0（u ≤ 0 行为未定义）；
//     - level ∈ {4,8,16,32}（其他值：nested_dequant 无操作，out 不修改——这一项
//       是防御性检查而非 UB，shift 安全）；
//     - n ≥ 0（n = 0 合法，无操作）；
//     - **code 输入域（2026-09-08 B3 补，docs/问题追踪/B3判读_输入域契约与超域
//       行为_2026_09_08.md）**：dequant/view_codes 的 code[i] 须 ∈ [−2^31, 2^31−1]
//       （nested_quant_i32 的产出保证）。**域外（wild int64）输入行为良定义、
//       无 UB、但超契约**，两条精确边界（sympy 符号验证）：
//         (a) rtn_quotient 内部 floor 商路径（`q = I>>m` 与 frac 的 `q<<m`）对
//             **任意** int64 无溢出——`q<<m = I − r ∈ [I−2^m+1, I]`，|值| < 2^63；
//         (b) 值重建**必须走 double 域**（`(double)q · 2^m`，现行为）：若改用
//             int64 `q_rtn<<m`，wild 输入 I 接近 INT64_MAX 时 q_rtn = 2^39、
//             `q_rtn<<m = 2^63` 恰回绕 INT64_MIN——**有符号左移溢出 UB**（三档
//             m ∈ {28,24,16} 全触发）。这是**实现约束而非规格自由度**，任何
//             "优化"改回 int64 重建即引入 UB，禁止。
//       精度注记：域内 |q| ≤ 2^15、wild |q| ≤ 2^39+1 均 < 2^53 ⟹ (double)q
//       精确；q·2^m 为 2 的幂（double 精确）；wild 下输出值可超 float 域
//       （→ ±inf，IEEE 良定义）。boundary N2 已覆盖 wild/extreme 码字。
//   已知吸收态（立项 §四）：h = 0 → frac = 0 → I = 0 → 全档 dequant = 0
//   （SR 在格点上无随机性，残差恒 0），调用方文档注明。

#pragma once

#include <cstdint>

namespace sgn::mkern::nested {

// ----------------------------------------------------------------------------
// nested_quant_i32：一次 SR 量化到最细格，产出嵌套 int32 码字。
//   code[i] = clip_int32( floor(h[i]/u) + [U_i < frac(h[i]/u)] )，规格见头注释 a-c。
//   单码多档：码字本身即全部档位的表示（各档 = 最高 b 位前缀），量化一次 O(n)。
void nested_quant_i32(int64_t* code, const float* h, int64_t n,
                      float u, uint64_t seed);

// ----------------------------------------------------------------------------
// nested_dequant：level 档 dequant = 对码字 RTN 截位（规格 d）。
//   level ∈ {4,8,16,32} → spacing u·2^(32−level)；输出 ∈ u·2^(32−level)·ℤ
//   且为 code[i]·u 的最近粗格点（N2 格嵌套不变量）。
void nested_dequant(float* out, const int64_t* code, int64_t n,
                    float u, int level);

// ----------------------------------------------------------------------------
// nested_view_codes：level 档视图的**整数码**（RTN 商）。
//   out[i] = RTN(code[i], 2^(32−level)) >> (32−level)——与 nested_dequant 共享
//   同一 RTN 定义点（单一定义纪律）；dequant 值 = out·2^(32−level)·u。
//   域 = [−2^(b−1), 2^(b−1)]（非对称 2^b+1 值：饱和顶码 I = 2³¹−1 的 RTN
//   上取整可达 +2^(b−1)，超出有符号 b-bit 一格——dot8 消费方注意 int8 域）。
//   用途：把档位视图喂给 MSint 点积域（Q8 → int8/dot8、Q16 → int16/pair 载体、
//   Q4 → int4/dot4——S3 衔接评估 B3/B4 的计算域入口，嵌套 Q16 视图码 ∈ int16
//   直接过 pair 载体恒等式）。标量即终态（整数 shift+cast，无热路径，无 SIMD
//   计划）；level 校验同 dequant（域外无操作）；n=0 合法。
//   承载注记（2026-09-08 B2 判读，docs/问题追踪/B2判读_nested视图码域与承载_
//   2026_09_08.md）：本 API 的 RTN 码域含 2^b+1 个值，**b-bit 容器不可承载**
//   （int_b 与 u8 偏置均溢出，计数原理 2^b+1 > 2^b）。**floor 读法**
//   （`code >> (32−level)`，算术右移）域收窄为 [−2^(b−1), 2^(b−1)−1] 恰 2^b 值，
//   此时 u8 偏置（q_u8 = q + 2^(b−1) ∈ [0, 2^b−1]）可承载——层 2 消费即走此路
//   （点积恒等式 2^m·Σq_f·w = 2^m·(dot(q_u8,w) − 2^(b−1)·Σw)，bit-exact）。
//   若需 RTN 忠实的 b-bit 视图消费，必须走 int16/pair 承载（层 2 案 b）。
void nested_view_codes(const int64_t* code, int64_t n, int level, int64_t* out);

// ---- 后端标识 / 调度（与 gemm_dispatch 同构：magic static + CPUID + 环境变量钩子）----
// 编译期：非 x86 / 无对应 ISA 的实现不编译 → 表仅标量项；运行时一次性 CPU 检测
// 选后端，SGN_NESTED_BACKEND=scalar 可强制回退标量（测试钩子，同 SGN_GEMM_BACKEND
// 纪律）。后端链：scalar → avx2（AVX2）→ avx512（AVX512F+DQ）。
struct NestedBackend {
    void (*nested_quant_i32)(int64_t*, const float*, int64_t, float, uint64_t);
    void (*nested_dequant)(float*, const int64_t*, int64_t, float, int);
    const char* name;   // "avx512" / "avx2" / "scalar"(forced)
};

const NestedBackend& nested_backend() noexcept;
const char* active_nested_backend_name() noexcept;

// ---- 标量锚点声明（定义见 scalar.cpp；boundary 测试直接对拍用，同 gemm 纪律：
//      锚点定义于实现文件、不进公共接口头，测试内具名 namespace 手动声明）----
void nested_quant_i32_scalar(int64_t* code, const float* h, int64_t n,
                             float u, uint64_t seed);
void nested_dequant_scalar(float* out, const int64_t* code, int64_t n,
                           float u, int level);

// ---- x86 实现声明（定义见 nested/x86/*.cpp；命名约定同 gemm/simd 后端后缀）----
// 位等同性论证见各文件头注释；与标量锚点逐位一致由 boundary 测试 N1 实证。
void nested_quant_i32_avx2(int64_t*, const float*, int64_t, float, uint64_t);
void nested_dequant_avx2(float*, const int64_t*, int64_t, float, int);
void nested_quant_i32_avx512(int64_t*, const float*, int64_t, float, uint64_t);
void nested_dequant_avx512(float*, const int64_t*, int64_t, float, int);

} // namespace sgn::mkern::nested
