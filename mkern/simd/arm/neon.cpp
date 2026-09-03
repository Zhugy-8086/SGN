// neon.cpp - ARM NEON 原语实现（远期预留 / TODO 占位）
//
// 背景：fixes_相关修复/simd指令集加速文件拆分计划_2026_08_29.md
//
// ⚠️ TODO 占位（综合执行计划 §二.4，EPYC 9K65 复核问题 ⑥）：当前为**空壳**，
// 无任何实际实现——ARM 平台所有原语（含 dot16/dot8/dot4）均走 simd/scalar.cpp
// 标量锚点，不代表"ARM 已支持"。目标平台为 Apple Silicon（AArch64 NEON + 可选
// SVE），待真实 ARM 硬件需求出现后，按 simd/simd_api.h 已冻结接口实现：
//   - dot16：int16 × int16 → int64（vmlaq_s32 两段式，注意 bit-exact）
//   - dot8/dot4：u8 × s8 → int64（NEON 无 vpdpbusd 等价物——vaddvq/vmlal 标量路径，
//     或 AArch64 的 SDOT/MADD（SDOT 为 u8/s8 点积，最优），视可用指令选择）
// 未实现落地前，编译期 __ARM_NEON 分支保持空，运行时自动落标量锚点（正确性不受影响）。

#include "mkern/simd/simd_api.h"

#if defined(__ARM_NEON)
#include <arm_neon.h>
namespace sgn::simd {
// 远期填充（TODO）：见上。
} // namespace sgn::simd
#endif
