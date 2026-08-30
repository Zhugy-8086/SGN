// neon.cpp - ARM NEON 原语实现（远期预留）
//
// 背景：fixes_相关修复/simd指令集加速文件拆分计划_2026_08_29.md
//
// Step 0：空壳。接口契约已在 simd/simd_api.h 冻结；待 ARM 硬件需求出现后，
// 按同一接口实现 dot16/dot8/dot4（NEON 无 VNNI 等价物时标量锚点生效）。

#include "simd/simd_api.h"

#if defined(__ARM_NEON)
namespace sgn::simd {
// 远期填充。
} // namespace sgn::simd
#endif
