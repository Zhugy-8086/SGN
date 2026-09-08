# ISA 声明表（G1 阶段 1c，2026-09-08 建档）

> **纪律：新 SIMD 路径先登记再实现。** 本表是每条 SIMD 路径的四元组权威登记点：
> **实现文件 × per-file 编译参数 × CPUID 检测位 × 标量回退/验证**。
> 修改任一列时必须同步本表 + 对应文件；review 按表逐列核对。
> 动机：外部审查 §一.8/9——CPUID 检测 + per-file ISA + 回退 + 验证四件事
> 分散四处，靠纪律非结构保证（Arrow Lake OSPKE 误读 2026-09-02 为前车之鉴）。

## CPUID 位定义总表（与 simd_dispatch.cpp `caps_from_registers` 逐位对应）

| 特性 | leaf | subleaf | 寄存器 | bit |
|---|---|---|---|---|
| SSSE3 | 1 | 0 | ECX | 9 |
| XSAVE（OS 启用） | 1 | 0 | ECX | 27 |
| AVX | 1 | 0 | ECX | 28 |
| AVX2 | 7 | 0 | EBX | 5 |
| AVX512F | 7 | 0 | EBX | 16 |
| AVX512BW | 7 | 0 | EBX | 30 |
| AVX512-VNNI | 7 | 0 | ECX | 11 |
| **AVX-VNNI（256 位 vpdpbusd）** | **7** | **1** | **EAX** | **4** |
| XCR0 AVX 状态 | XCR0 | — | — | 0x6 |
| XCR0 AVX-512（opmask+ZMM hi256） | XCR0 | — | — | 0xE6 |

⚠️ 历史教训：AVX-VNNI 曾双重读错（subleaf 0 非 1、寄存器 ECX 非 EAX），
OSPKE=0 机器 dot8/dot4 静默落标量。检测位变更必须：查 Intel SDM →
更新本表 → 更新 `caps_from_registers` → 更新 `test_cpuid_caps_mock`。

## 路径登记表

### mkern/simd（原语层）

| 原语族 | 实现文件 | per-file 编译参数 | CPUID 门控 | 标量回退 | bit-exact 验证 |
|---|---|---|---|---|---|
| dot16（u16×i16 madd） | `x86/avx2.cpp`；AVX-512 变体 `x86/avx512.cpp` | avx512.cpp: `-mavx512f -mavx512bw -mavx512vl -ffp-contract=off` | AVX；avx512 变体另需 AVX512F + XCR0 0xE6 | `scalar.cpp` | sgn_benchmark 238 项 + simd_boundary |
| dot8（u8×s8→i32） | `x86/avx2_dot.cpp`（vpmaddubsw 中间档）→ `x86/avxvnni.cpp`（vpdpbusd）→ `x86/avx512vnni.cpp` | avx2_dot: `-mavx2`；avxvnni: `-mavx2 -mavxvnni`；avx512vnni: `-mavx512f -mavx512bw -mavx512vnni` | AVX2 → AVX-VNNI(leaf7.1.EAX[4]) → AVX512-VNNI(leaf7.0.ECX[11]) | `scalar.cpp` | 大 K 满幅 10 档 + 238 项 |
| dot4 / dot4_packed（nibble） | `x86/avx2.cpp`（随 dot8 系文件） | 同 dot8 系 | 同 dot8 | `scalar.cpp` | 238 项 |
| reverse_bytes8 | `x86/ssse3.cpp` | （需 SSSE3；batch 变体 _mm256 需 AVX2） | SSSE3 | `scalar.cpp` | 238 项 |
| batch_reverse_u8 | `x86/ssse3.cpp` | 同上 | SSSE3/AVX2 | `scalar.cpp` | 238 项 |
| decode_i16_f32（i16→f32） | `x86/avx2_decode.cpp` | `-mavx2`（隐含于文件级） | AVX2 | `scalar.cpp::decode_i16_f32_packed16_scalar` | V7 dev=0（pair 载体等价） |
| unpack_nibble_u / _s | `x86/ssse3.cpp` | 同 reverse 系 | SSSE3/AVX2 | `scalar.cpp` | W5–W7（leveled_state） |
| 浮点归约（avx2_reduce） | `x86/avx2_reduce.cpp` | `-mavx2 -ffp-contract=off`（浮点禁收缩锁死） | AVX2 | `scalar.cpp`（同旗标） | 归约专项 |

### mkern/gemm（矩阵微内核）

| 原语 | 实现文件 | per-file 编译参数 | CPUID 门控 | 回退 | 验证 |
|---|---|---|---|---|---|
| gemm_avx2 / _avx2vnni / _avx512vnni | `mkern/gemm/x86/gemm_{avx2,avx2vnni,avx512vnni}.cpp` | `-mavx2` / `-mavx2 -mavxvnni` / `-mavx512f -mavx512bw -mavx512vnni` | 同 simd 系 | gemm 标量 | gemm 4819 项 |

### mkern/nested（嵌套量化）

| 原语 | 实现文件 | per-file 编译参数 | CPUID 门控 | 回退 | 验证 |
|---|---|---|---|---|---|
| nested_avx2 / _avx512 | `mkern/nested/x86/nested_{avx2,avx512}.cpp` | `-mavx2` / `-mavx512f -mavx512dq`（mullo_epi64/cvtqq2pd 需 **DQ**，见 nested_avx512.cpp 头注） | 同 simd 系（avx512 另需 DQ 覆盖的指令在 F+DQ） | nested scalar | nested_boundary 438 + gemm 4819 |

## 通用纪律（每条新路径的 checklist）

1. **登记**：本表加行（四元组填全才许开工）；
2. **编译**：CMakeLists `set_source_files_properties` 加文件级选项（禁止全局
   `-mavx*` 回潮，见 全局AVX编译参数移除调查_2026_08_31.md）；
3. **检测**：`caps_from_registers` 加位 + `test_cpuid_caps_mock` 用例
   （检测位变更三处同步：Intel SDM 核对 → dispatch → mock）；
4. **回退**：`scalar.cpp` 标量锚点（非 x86/无 ISA 平台安全降级）；
5. **验证**：bit-exact 对抗（跨后端逐位一致 + 满幅/大 K 对抗）；
6. **浮点注意**：涉及浮点归约的文件必须 `-ffp-contract=off`（R1 修订：
   -mavx512f 隐含 FMA，Clang 默认收缩会破坏 kBitExact）。
