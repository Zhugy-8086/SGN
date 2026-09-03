# ============================================================
# simd/ 原语层独立构建入口（"引线"）
# ============================================================
# 用途：独立的接口调用入口——仅编译 simd/ 文件夹（scalar / simd_dispatch /
#       x86/* / arm/neon / simd_boundary_test / sgn_benchmark），产出独立可执行，
#       不依赖 pybind11 / libomp / Python。
#
# 与 engine/sgn/ 主构建完全隔离：
#   - 主 CMakeLists.txt 平铺引用 simd/*.cpp（无 add_subdirectory），
#     本文件不会被主构建自动加载，互不影响；
#   - target 名 simd_boundary_test 与 sgn 模块不冲突（避免将来 add_subdirectory 撞名）。
#
# 本文件是 build_remote.sh 的 CMake 化版本；per-file ISA 纪律 = 阶段 1
# （见 fixes_相关修复/全局AVX编译参数移除调查_2026_08_31.md §6）：
#   基础编译无 AVX 宏 → dispatch 纯运行时 CPUID 门控；
#   各实现文件显式声明所需指令集，符号常驻产出（不与全局宏耦合）。
#
# 用法：
#   Windows:  cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang
#             cmake --build build
#   Linux:    cmake -B build -G Ninja && cmake --build build
#   产出:     build/simd_boundary_test （直接运行：./build/simd_boundary_test）
#   构建目录统一用 build/（已被根 .gitignore 的 build/ 规则忽略，不入库）。
# ============================================================

cmake_minimum_required(VERSION 3.20)
project(sgn_simd_standalone LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 默认 Release（-O3，与主构建一致）
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_options(-O3)

add_executable(simd_boundary_test
    scalar.cpp
    simd_dispatch.cpp
    x86/avx2_dot.cpp
    x86/avx2_decode.cpp
    x86/avx2_reduce.cpp
    x86/avxvnni.cpp
    x86/ssse3.cpp
    x86/avx512.cpp
    x86/avx512vnni.cpp
    arm/neon.cpp
    simd_boundary_test.cpp
)

# 源文件以 #include "mkern/simd/simd_api.h" 引用微内核层头文件
# （本文件位于 engine/sgn/mkern/simd/，../.. = engine/sgn，mkern/simd/simd_api.h 自此解析）
target_include_directories(simd_boundary_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../..")

# per-file ISA（阶段 1 纪律，与 engine/sgn/CMakeLists.txt set_source_files_properties 一致）：
#   avx2_dot.cpp / avx2_decode → -mavx2（avx2.cpp 2026-09-02 按域拆分）
#   avx2_reduce.cpp → -mavx2 -ffp-contract=off（R1 修订：浮点归约禁 FMA 收缩保 kBitExact）
#   scalar.cpp      → -ffp-contract=off（同上，标量锚点防御性锁死）
#   avxvnni.cpp     → -mavx2 -mavxvnni
#   ssse3.cpp       → -mssse3 -mavx2   （reverse 用 SSSE3；batch 用 _mm256_shuffle_epi8 需 AVX2）
#   avx512.cpp      → -mavx512f -mavx512bw -mavx512vl -ffp-contract=off（-mavx512f 隐含 FMA，
#                      Clang 默认收缩会把 sum 系标量尾收缩成 vfmadd，破坏跨后端逐位一致）
#   avx512vnni.cpp  → 同上 + -mavx512vnni
set_source_files_properties(
    x86/avx2_dot.cpp
    x86/avx2_decode.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx2")
set_source_files_properties(
    x86/avx2_reduce.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx2;-ffp-contract=off")
set_source_files_properties(
    scalar.cpp
    PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
set_source_files_properties(
    x86/avxvnni.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx2;-mavxvnni")
set_source_files_properties(
    x86/ssse3.cpp
    PROPERTIES COMPILE_OPTIONS "-mssse3;-mavx2")
set_source_files_properties(
    x86/avx512.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-ffp-contract=off")
set_source_files_properties(
    x86/avx512vnni.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512vnni")

# ------------------------------------------------------------
# sgn_benchmark：原语层完整基准（正确性 + 性能 + JSON 输出）
# ------------------------------------------------------------
# 2026-09-02 收编：原件为远程验证归档 _remote_test_decompressed/sgn_benchmark/
# （EPYC 2026-08-31 实测版本，保留不动），本 target 使用收编的 portable 副本
# sgn_benchmark.cpp（chrono 计时 + CWD JSON 路径，计算逻辑与原件一致）。
# per-file ISA 复用上方同一组源文件属性；本文件自身基础编译（仅调调度层）。
# 运行：./build/sgn_benchmark [--json-only]；强制后端：SGN_KERNEL_BACKEND=scalar
add_executable(sgn_benchmark
    scalar.cpp
    simd_dispatch.cpp
    x86/avx2_dot.cpp
    x86/avx2_decode.cpp
    x86/avx2_reduce.cpp
    x86/avxvnni.cpp
    x86/ssse3.cpp
    x86/avx512.cpp
    x86/avx512vnni.cpp
    arm/neon.cpp
    sgn_benchmark.cpp
)

target_include_directories(sgn_benchmark PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../..")