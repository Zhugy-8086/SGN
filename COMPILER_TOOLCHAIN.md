# 编译器工具链说明

> **位置**：项目根目录最显眼处
> **最后更新**：2026-09-08（G1 阶段 1：声明式 preset 与 ISA 声明表互链）
> **用途**：本项目 C/C++ 开发所用的全部编译器、构建工具的安装位置与使用方式

---

## ⭐ 权威声明点（G1 阶段 1，2026-09-08 起）

- **配置声明**：`engine/sgn/CMakePresets.json`——Clang 路径/生成器/C++23/libomp
  已声明式入仓（`cmake --preset windows-clang-release` 一步配置；覆盖方式见
  preset 文件头注释）。本文件描述安装位置与原理，**配置以 preset 为准**。
- **ISA 声明表**：`engine/sgn/mkern/simd/isa_registry.md`——每条 SIMD 路径的
  四元组（实现文件 × per-file 编译参数 × CPUID 检测位 × 标量回退/验证）权威
  登记点；**新 SIMD 路径先登记再实现**，CPUID 位定义以该表与 Intel SDM 逐位
  核对为准。

---

## 0. 一句话索引

所有编译器工具链统一部署在 **`c:\kaffj\`** 目录下，共 4 套工具：

| 工具 | 版本 | 路径 | 角色 |
|------|------|------|------|
| **Clang/LLVM** | 22.1.8 | `c:\kaffj\clang+llvm-22.1.8-x86_64-pc-windows-msvc\bin\` | 🔴 底层优化开发（主力） |
| **GCC (MinGW64)** | 16.1.0 | `c:\kaffj\msys64\mingw64\bin\` | 🟡 后期生产测试 |
| **TCC** | 0.9.27 | `c:\kaffj\tcc\tcc\tcc.exe` | 🟢 快速 C 原型 |
| **CMake** | 4.4.2 | `c:\kaffj\cmake\bin\` | 🔵 构建系统 |

另：**MSVC 19.44** 已安装于 VS 2022 BuildTools，仅作为 Windows SDK 链接备用。

---

## 1. 目录结构

```
c:\kaffj\
├── clang+llvm-22.1.8-x86_64-pc-windows-msvc\   ← Clang/LLVM (主力开发)
│   └── bin\
│       ├── clang.exe          ← C 编译器
│       ├── clang++.exe        ← C++ 编译器
│       ├── clang-cl.exe       ← MSVC 兼容接口
│       ├── lld.exe            ← LLVM 链接器
│       ├── lld-link.exe       ← MSVC 兼容链接器
│       ├── llvm-ar.exe        ← 静态库归档
│       ├── llvm-nm.exe        ← 符号表查看
│       ├── llvm-objdump.exe   ← 反汇编
│       ├── opt.exe            ← IR 优化器（底层优化分析）
│       ├── llc.exe            ← IR→机器码
│       ├── clang-tidy.exe     ← 静态检查
│       └── clang-format.exe   ← 代码格式化
├── msys64\                                      ← MSYS2 环境
│   └── mingw64\bin\                             ← GCC 16.1.0
│       ├── gcc.exe
│       ├── g++.exe
│       ├── ar.exe / as.exe / ld.exe / nm.exe
│       └── gdb.exe            ← 调试器
├── tcc\                                         ← Tiny C Compiler
│   └── tcc\tcc.exe
└── cmake\                                       ← CMake 4.4.2
    └── bin\cmake.exe
```

---

## 2. PATH 配置

以下 4 条路径已写入**用户环境变量**（2026-08-01 配置完成）：

```text
c:\kaffj\cmake\bin
c:\kaffj\clang+llvm-22.1.8-x86_64-pc-windows-msvc\bin
c:\kaffj\msys64\mingw64\bin
c:\kaffj\tcc\tcc
```

**注意**：新开终端窗口才会生效。当前终端可手动执行：

```powershell
$env:Path += ";c:\kaffj\cmake\bin;c:\kaffj\clang+llvm-22.1.8-x86_64-pc-windows-msvc\bin;c:\kaffj\msys64\mingw64\bin;c:\kaffj\tcc\tcc"
```

---

## 3. 双编译器策略（用户决策 2026-08-01）

### 3.1 策略定义

| 编译器 | 角色 | 理由 |
|--------|------|------|
| **Clang 22.1.8** | 底层优化开发 | 更严格，适合内存分配/结构体/指令集适配优化 |
| **GCC 16.1.0** | 后期生产测试 | 严谨程度足够，适合跑测试和调试 |

### 3.2 Stage 3.0 使用方式

- **开发阶段**：Clang（严格检查，捕获内存/指令集问题）
- **测试阶段**：GCC（生产环境验证）
- **指令集**：AVX2 / AVX-VNNI / AVX-512（Clang 完整支持）

### 3.3 为什么不用 MSVC 作主力

MSVC 存在 **AVX-VNNI 代码生成 bug**（详见 [avx_vnni_compiler_requirement](fixes_相关修复/architecture/avx_vnni_compiler_requirement_2026_07_30.md)），因此底层优化开发改用 Clang。

---

## 4. 使用示例

### 4.1 Clang 开发编译（严格模式）

```powershell
# 单文件编译（含 AVX-VNNI + 严格警告 + 内存检查）
clang++ -O3 -mavx2 -mavxvnni -std=c++23 -Wall -Wextra -Wpedantic -fsanitize=memory -o output.exe source.cpp

# 多文件编译
clang++ -O3 -mavx2 -mavxvnni -std=c++23 -Wall -Wextra -I include/ src/*.cpp -o output.exe

# 生成调试信息
clang++ -g -O0 -std=c++23 -Wall -Wextra -o output_debug.exe source.cpp

# 仅预处理/汇编/IR 查看（底层优化分析）
clang++ -S -emit-llvm -O3 -mavx2 -o output.ll source.cpp    # 查看优化后的 IR
clang++ -S -O3 -mavx2 -o output.s source.cpp                 # 查看汇编
```

### 4.2 GCC 生产测试编译

```powershell
# 标准生产构建
g++ -O3 -mavx2 -mavxvnni -std=c++23 -Wall -Wextra -o output.exe source.cpp

# 带性能分析
g++ -O3 -mavx2 -pg -std=c++23 -o output_prof.exe source.cpp

# 链接 MSYS2 库
g++ -O3 -std=c++23 -I "c:\kaffj\msys64\mingw64\include" -L "c:\kaffj\msys64\mingw64\lib" -o output.exe source.cpp -lstdc++
```

### 4.3 CMake 构建项目（推荐用 Clang）

```powershell
# 配置（指定 Clang）
cmake -B build -G Ninja -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_BUILD_TYPE=Release

# 构建
cmake --build build

# 用 GCC 测试构建
cmake -B build_gcc -G Ninja -DCMAKE_C_COMPILER=gcc -DCMAKE_CXX_COMPILER=g++ -DCMAKE_BUILD_TYPE=Release
cmake --build build_gcc
```

> **项目 ISA 纪律（2026-08-31 起，阶段 1-4 完成）**：`engine/sgn` 的全局编译选项**不再含
> `-mavx2 -mavxvnni`**（全局仅 `-O3 -fopenmp=libomp -fno-lto`）。AVX/AVX-VNNI/FMA 能力全部
> 下沉到 **per-file `COMPILE_OPTIONS`**（`simd/x86/*.cpp`、`hc/`、`autograd/ops(.nn).cpp`）+
> 运行时 CPUID/SEH 检测门控 → 同一二进制在无对应指令集的 CPU 上自动回退，不再
> illegal instruction（此前全局宏会让二进制硬性要求 AVX-VNNI）。手工编译单文件时按指令集
> 显式加对应 flag（本手册 §4.1/4.2 示例即此用法）。详见
> [全局AVX编译参数移除调查](engine/sgn/fixes_相关修复/全局AVX编译参数移除调查_2026_08_31.md)。

### 4.4 TCC 快速 C 原型

```powershell
# 极速 C 编译（无优化但启动快）
tcc -o output.exe source.c

# 带 AVX
tcc -mavx2 -o output.exe source.c
```

**注意**：TCC 仅支持 C，C++ 支持有限，不适合 Stage 3.0 的 C++ 抽象层。

---

## 5. 指令集支持矩阵

| 指令集 | Clang 22.1.8 | GCC 16.1.0 | MSVC 19.44 | TCC 0.9.27 |
|--------|-------------|------------|-----------|-----------|
| AVX2 | ✅ `-mavx2` | ✅ `-mavx2` | ✅ `/arch:AVX2` | ✅ `-mavx2` |
| AVX-VNNI | ✅ `-mavxvnni` | ✅ `-mavxvnni` | ❌ **bug** | ❌ |
| AVX-512 | ✅ `-mavx512f` | ✅ `-mavx512f` | ⚠️ 部分 | ❌ |
| FMA | ✅ `-mfma` | ✅ `-mfma` | ✅ `/arch:AVX2` | ❌ |
| F16C | ✅ `-mf16c` | ✅ `-mf16c` | ✅ `/arch:AVX2` | ❌ |

**结论**：AVX-VNNI 必须用 Clang 或 GCC，不能用 MSVC。

---

## 6. MSVC 备用环境

MSVC 19.44 已安装但**不在 PATH**，需要时通过 vcvarsall.bat 配置：

```powershell
# 方式 1：加载 MSVC 环境到当前终端
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

# 方式 2：打开 "x64 Native Tools Command Prompt for VS 2022"（开始菜单）

# 编译（仅作 Windows SDK 链接备用）
cl /O2 /arch:AVX2 /std:c++23 source.cpp
```

**路径**：`C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Tools\MSVC\14.44.35207\bin\Hostx64\x64\cl.exe`

---

## 7. 已编译的 Python C 扩展

项目共有 **1 个** `.pyd` 扩展，全部已升级到 **Python 3.14** 并统一使用 **Clang 22.1.8** 编译：

| 扩展 | 位置 | 用途 | 编译器 | C++20 支持 | AVX-VNNI |
|------|------|------|--------|-----------|----------|
| `sgn.cp314` | `engine/sgn/build/` | 统一 C++ 抽象层（含 col2im_add、col2im_c.col2im_add、HC8/HC16/HC4/HC16ms 绑定） | **Clang 22.1.8** | ✓ 完善 | ✓ 运行时启用 |

**当前状态**：1/1 全部用 Clang 22.1.8，通过 `engine/sgn/CMakeLists.txt` 统一构建。

**迁移历史**（2026-08-02/06）：
- **Stage 3.1**：4 个 `pysgn_*` 模块（pysgn_net、pysgn_hc16、pysgn_hc4_pshufb、pysgn_hc16ms）从 MSVC 19.44 迁移到 Clang 22.1.8，随后合并到统一 `sgn` 模块
- **Stage 3.0.7**：`pysgn_col2im` 从 MSVC 迁移到 CMake + Clang 22.1.8 构建，保留为独立 .pyd 作为 col2im 性能对比基准
- **2026-08-06**：`pysgn_col2im` 合并到 `sgn` 统一模块（`sgn.col2im_c` 子模块），彻底消除最后一个独立 .pyd
- 关键发现：ASan 与 Python 扩展不兼容（需 `-fno-sanitize=address`）；MSVC `/arch:AVX2` 隐式启用 FMA，Clang 需单独 `-mfma`；pybind11 在 `CMAKE_INTERPROCEDURAL_OPTIMIZATION` 未定义时会自动链接 `pybind11::lto`（`-flto` 落在 `-fno-lto` 之后失效），2026-08-31 起 CMake 显式 `set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)` 关闭
- 旧的 `setup_*.py` 编译脚本已删除，统一通过 CMake 构建

---

## 8. OpenMP 运行时与跨编译器性能对比

### 8.1 运行时选择：libomp vs VCOMP140

| 运行时 | 编译器 | 部署方式 | 特点 |
|--------|--------|---------|------|
| **libomp** | Clang (`-fopenmp=libomp`) | `libomp.dll` 需复制到 `.pyd` 目录 | 线程池 fork/join 开销略高 (~20-50μs) |
| **VCOMP140** | MSVC (`/openmp`) | 系统自带 `vcomp140.dll` | 小规模场景开销更低 |

**关键约束**：Clang 在 Windows 上生成的 OpenMP 符号（`__kmpc_*`）与 VCOMP140 不兼容，必须使用 libomp。尝试链接 `vcomp.lib` 会导致 `undefined symbol: __kmpc_for_static_fini`。

### 8.2 libomp.dll 部署

Python 3.8+ 不从 PATH 加载 DLL，`libomp.dll` 必须与 `.pyd` 同目录。CMakeLists.txt 配置：

```cmake
# 链接 libomp.lib
target_link_libraries(sgn PRIVATE
    "c:/kaffj/clang+llvm-22.1.8-x86_64-pc-windows-msvc/lib/libomp.lib")

# 复制 libomp.dll 到 build 目录
set(LIBOMP_DLL "c:/kaffj/clang+llvm-22.1.8-x86_64-pc-windows-msvc/bin/libomp.dll")
add_custom_command(TARGET sgn POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${LIBOMP_DLL}" "$<TARGET_FILE_DIR:sgn>"
    COMMENT "Copying libomp.dll to build directory")
```

**许可与分发（2026-09-05 开源决策注记）**：`libomp.dll` 来自 LLVM Project（openmp
runtime），许可为 **Apache 2.0 with LLVM Exceptions**（非 LGPL，勿与 GCC 的 libgomp
GPLv3+exception 混淆）。随本项目分发 pyd+dll 无需开源本项目代码，义务仅为：分发物
附 LLVM 许可文本副本 + 保留版权声明（见仓库 THIRD_PARTY_NOTICES.md）。GCC 链的
libgomp 仅存在于本地测试构建，永不进入分发物。

### 8.3 跨编译器性能对比方法论

对比 Clang+libomp (sgn 模块) 与 MSVC+VCOMP140 (pysgn_col2im 独立扩展) 时，发现两个关键问题：

**问题 1：VCOMP140 线程调度方差大**
- 原 C 实现顺序测量时，最小值在 0.559ms~0.921ms 间波动 (±30%)
- libomp 实现稳定在 ~0.77ms

**解决方案：交错基准测试 (Interleaved Benchmark)**
- 每轮交替运行 C++ 和原 C，保证两者经历相同 CPU 频率/缓存状态
- 取各自 30 轮最小值比较，消除系统状态漂移

**问题 2：小规模场景 OpenMP 线程池开销主导**
- BC < 128 时，libomp fork/join 开销 (~20-50μs) 超过计算量
- 串行回退路径：BC < 128 时绕过 OpenMP，直接执行循环

### 8.4 分级容差策略（已收紧）

性能测试 (`engine/sgn/tests/test_col2im_perf.py`) 采用分级容差，反映跨编译器/运行时的合理差异（2026-08-03 收紧）：

| 原 C 耗时 | 容差 | 场景特征 | 理由 |
|-----------|------|---------|------|
| < 0.1ms | 3.0x | OpenMP 开销主导 | libomp fork/join 噪声，ratio 2.2~3.0 波动 |
| < 1ms | 10% | 中等噪声 | 微基准系统噪声，已从 20% 收紧 |
| ≥ 1ms | 5% | 计算主导 | 严格要求，已从 10% 收紧 |

**实测结果**（交错 benchmark，3 次运行）：
- 小规模 B=2 C=32: ratio 2.4~2.9 (容差 3x) ✓
- 中规模 B=4 C=64: ratio 1.1~1.4 (容差 3x) ✓
- 大规模 B=8 C=128: ratio 0.65~0.99 (容差 20%) ✓ — C++ 与原 C 持平或更快

---

## 9. 故障排查

### 9.1 "clang: command not found"

PATH 未生效，新开终端窗口，或手动执行：

```powershell
$env:Path += ";c:\kaffj\clang+llvm-22.1.8-x86_64-pc-windows-msvc\bin"
```

### 8.2 "cannot find -lstdc++"（MinGW 链接错误）

确保使用 MSYS2 mingw64 的 g++，而非系统其他 g++：

```powershell
Get-Command g++ | Select-Object Source
# 应显示 c:\kaffj\msys64\mingw64\bin\g++.exe
```

### 9.3 CMake 找不到编译器

显式指定编译器：

```powershell
cmake -B build -DCMAKE_C_COMPILER="c:/kaffj/clang+llvm-22.1.8-x86_64-pc-windows-msvc/bin/clang.exe" -DCMAKE_CXX_COMPILER="c:/kaffj/clang+llvm-22.1.8-x86_64-pc-windows-msvc/bin/clang++.exe"
```

### 9.4 LLVM 版本确认

```powershell
clang --version
# 应显示: clang version 22.1.8, Target: x86_64-pc-windows-msvc
```

### 9.5 "DLL load failed while importing sgn"

`libomp.dll` 未在 `.pyd` 同目录。Python 3.8+ 不从 PATH 加载 DLL。

```powershell
# 检查 libomp.dll 是否在 build 目录
ls engine\sgn\build\libomp.dll

# 手动复制（CMake POST_BUILD 应自动完成）
copy "c:\kaffj\clang+llvm-22.1.8-x86_64-pc-windows-msvc\bin\libomp.dll" engine\sgn\build\
```

### 9.6 "undefined symbol: __kmpc_for_static_fini"

Clang 生成的 OpenMP 符号 (`__kmpc_*`) 与 MSVC VCOMP140 不兼容。不能链接 `vcomp.lib`，必须使用 libomp：

```cmake
# 正确: 链接 libomp
target_link_libraries(sgn PRIVATE
    "c:/kaffj/clang+llvm-22.1.8-x86_64-pc-windows-msvc/lib/libomp.lib")

# 错误: 不能用 vcomp.lib (符号不兼容)
# target_link_libraries(sgn PRIVATE vcomp.lib)  # ← 会报 __kmpc_* undefined
```

---

## 10. 未来路线图（5 .pyd 统一 + HC 库整理）

> **决策时间**：2026-08-02
> **决策性质**：5 个 .pyd 已全部用 Clang 22.1.8 编译；下一阶段是合并为 1 个统一 `sgn.cp314` 并整理 HC 库

### 10.1 下一阶段目标

全部 5 个 `.pyd` 已用 **Clang 22.1.8** 编译（Stage 3.1 完成，2026-08-02），下一阶段是 **5 个 .pyd 合并为 1 个统一 `sgn.cp314`**。

合并的收益：

**收益 1：统一导入路径**

当前 5 个 `.pyd` 各自是独立模块，Python 侧需分别 `import`。合并后只需 `import sgn`，简化调用方代码。

**收益 2：统一构建与符号管理**

5 个 `pybind11_add_module` 目标合并为 1 个，CMakeLists.txt 更精简，避免重复的编译选项与链接配置。

**收益 3：HC 库整理的契机**

合并时可顺带将 `fixes_相关修复/hc_v1.3_net_extension/` 下的 C 代码整合到 `engine/sgn/hc/`，统一 HC 源码组织。

### 10.2 已完成项总结

截至 2026-08-06，以下编译器相关项已完成：

- ✅ **4 个 pysgn_* 合并到统一 sgn 模块**：pysgn_net、pysgn_hc16、pysgn_hc4_pshufb、pysgn_hc16ms 绑定代码合并到 `sgn`，保留向后兼容别名
- ✅ **pysgn_col2im 合并到统一 sgn 模块**：原独立 .pyd 已合并到 `sgn.col2im_c` 子模块，彻底消除最后一个独立 .pyd（2026-08-06）
- ✅ **pysgn_col2im 迁移到 Clang + CMake**：从 MSVC setup_col2im.py 迁移到 CMake + Clang 22.1.8，保留为独立 .pyd 作为 col2im 性能对比基准
- ✅ **旧 .pyd 文件已清理**：fixes 目录下已合并的旧 .pyd 文件全部删除，避免干扰测试
- ✅ **全局 ASan 已移除**：ASan 与 Python 扩展不兼容（无法 dlopen），CMakeLists.txt 中的全局 ASan 启用逻辑已删除
- ✅ **Debug 模式覆盖 -O3 问题已修复**：CMakeCache.txt 中 `CMAKE_BUILD_TYPE=Debug` 导致 `-O0` 覆盖全局 `-O3`，已通过默认 Release 模式修复（详见 §10.7）
- ✅ **HC C 库迁移到 Clang + Ninja**：`engine/hc/CMakeLists.txt` 已配置 Clang 编译器和 Ninja 构建系统（2026-08-06）

### 10.3 下一阶段的触发条件

Clang 迁移已完成，.pyd 合并已完成（5 个独立 .pyd 全部合并为 1 个统一 `sgn` 模块），HC 源码整合已完成。以下条件满足时，应考虑后续优化：

1. **col2im perf 测试容差收紧**：pysgn_col2im 已合并到 `sgn.col2im_c`，同编译器同模块环境已满足，可收紧为统一 5-10%
2. **Autograd 性能基准测试**：`matmul` 向量化 + OpenMP 并行化已实现，可运行 benchmark 验证加速比

### 10.4 已完成迁移（5 个 .pyd → 1 个统一 .pyd）

5 个 .pyd 合并为 1 个统一 `sgn.cp314` 已完成（2026-08-06）：

1. 4 个 `pysgn_*.cpp` 的绑定代码已合并到 `sgn` 模块（通过 `register_*` 函数）
2. 4 个独立的 `pybind11_add_module` 目标已移除，只保留 `sgn` 一个
3. `pysgn_col2im` 的绑定代码已合并到 `sgn.col2im_c` 子模块（2026-08-06）
4. 统一 Python 导入路径：所有 HC 功能和 col2im 通过 `import sgn` 访问
5. 向后兼容别名可用：`sys.modules['pysgn_net'] = sgn` 等，旧代码无需修改

> **兼容别名移除计划**（安全审计 2026-08-16 决策项 8）：
> - 别名注入位于 `engine/sgn/__init__.py` 的 `_setup_compat_aliases()`，默认启用。
> - 已提供 `SGN_DISABLE_PYSGN_COMPAT=1` 环境变量开关，置位时跳过注入
>   （用于验证/剔除对旧别名的依赖）。
> - **移除时机**：活跃测试与示例对 `pysgn_*` 的依赖迁移完毕后（当前唯一活跃
>   依赖 hc16_gradient_storage_validation 已于 2026-08-16 迁移到 `sgn._native.hc16`），
>   删除 `_setup_compat_aliases()` 及开关，随后清理本文档第 5 条说明。

### 10.5 已完成项与待完成项

**已完成**（2026-08-06）：

- ✅ 4 个 pysgn_* 合并到统一 sgn 模块（pysgn_net、pysgn_hc16、pysgn_hc4_pshufb、pysgn_hc16ms）
- ✅ pysgn_col2im 合并到统一 sgn 模块（sgn.col2im_c 子模块，2026-08-06）
- ✅ 旧 .pyd 文件清理（fixes 目录 + build 目录）
- ✅ CMakeLists.txt 默认 Release 模式
- ✅ 全局 ASan 移除（与 .pyd 不兼容）
- ✅ Debug 模式 `-O0` 覆盖 `-O3` 问题修复（详见 §10.7）
- ✅ HC C 库迁移到 Clang + Ninja（engine/hc/CMakeLists.txt，2026-08-06）

**待完成**：

- ✅ **col2im perf 测试容差已收紧**（2026-08-03）：< 0.1ms 保持 3.0x（OpenMP 开销噪声），< 1ms 从 20%→10%，≥ 1ms 从 10%→5%
- ✅ **Autograd 性能基准测试已完成**（2026-08-03）：Task 6.3，B=4/8/16 下 fwd+bwd 和 fwd only 对比，C++ 比 PyTorch 慢 3-5x（fwd+bwd），瓶颈在 backward
- ✅ HC 源码整合已完成（2026-08-03）：`fixes_相关修复/hc_v1.3_net_extension/` 下 15 个源文件移入 `engine/sgn/hc/ext/`，`hc/col2im.cpp` 头文件引用路径修复，编译验证通过
- ✅ C++23 升级已完成（2026-08-03）：`-std=c++20` → `-std=c++23`，Tensanor 类引入 `std::mdspan` 实现 6 个视图接口，`hc16ms_bswap16` 替换为 `std::byteswap`
- ✅ `batchnorm2d` 零拷贝适配已完成（2026-08-03）：`bn2d_reshape_fwd`/`bn2d_reshape_bwd` 从手动三重循环拷贝改为 `permute` + `reshape` 零拷贝链，消除前向和反向各一次 O(B*C*H*W) 的完整数据拷贝
- ✅ Autograd 性能优化（AVX2/AVX-VNNI/AVX-512 + OpenMP）已完成（2026-08-03）：
  - matmul forward/backward 添加 AVX2 FMA (`_mm256_fmadd_ps`) 8 路向量化 + AVX-512 (`_mm512_fmadd_ps`) 16 路编译路径 + 运行时 CPU 自动调度
  - conv2d im2col/col2im 添加 `#pragma omp parallel for` 并行化
  - BN/relu/maxpool 添加 `#pragma omp parallel for` 并行化
  - CMakeLists.txt 添加 `ENABLE_AVX512` 可选项（默认关闭，不测试）
  - 测试目标（test_phase1/2/3）增加 libomp 链接支持

### 10.6 版本号认知纠正

一个常见的误解：因为 MSVC 版本号是 19.44（数字小于 Clang 22 和 GCC 16），所以 MSVC "版本最低、最旧"。**这是错误的**。

三个编译器使用不同的版本号体系：
- Clang 22 = LLVM 项目的第 22 个大版本
- GCC 16 = GCC 项目的第 16 个大版本
- MSVC 19.44 = `_MSC_VER` 工具集版本，对应 Visual Studio 2022 (17.x)

MSVC 19.44 实际上是 2024-2025 年的版本，与 Clang 22 / GCC 16 大致同期。**版本号数字小 ≠ 版本旧**。MSVC 的问题是 C++20 支持有缺陷，不是版本太旧。

### 10.7 Debug 模式覆盖 -O3 问题（2026-08-02 修复）

**问题**：CMakeCache.txt 中 `CMAKE_BUILD_TYPE=Debug`，导致 `CMAKE_CXX_FLAGS_DEBUG` 的 `-O0` 覆盖了全局 `add_compile_options(-O3)`。所有 .pyd 在无优化状态下运行。

**影响**：C++ Autograd 的 6 层 CNN benchmark 在 Debug 模式下（`-O0`）报告的"13-43x 比 PyTorch 慢"中，有相当一部分是 `-O0` 导致的虚假差距。Release 模式优化后实测为 3-5x（fwd+bwd）。

**修复**：
1. CMakeLists.txt 顶部添加 `if(NOT CMAKE_BUILD_TYPE) set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE) endif()`
2. 删除全局 ASan 启用逻辑（ASan 与 .pyd 不兼容）
3. 删除每个 .pyd target 的 `-fno-sanitize=address`（全局不再启用 ASan，禁用变得多余）
