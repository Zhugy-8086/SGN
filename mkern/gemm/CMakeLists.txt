# ============================================================
# gemm/ 矩阵级微内核独立构建入口（"引线"，与 mkern/simd/CMakeLists.txt 同型）
# ============================================================
# 用途：仅编译 gemm/ + 其依赖的 simd/ 原语（gemv_i8 消费 simd::dot8、
#       标量锚点消费 simd::dot8_scalar），产出独立可执行，
#       不依赖 pybind11 / libomp / Python。
#
# 与 engine/sgn/ 主构建完全隔离（主 CMakeLists.txt 平铺引用 gemm/*.cpp，
# 无 add_subdirectory，互不影响）。
#
# 用法：
#   Windows:  cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang
#             cmake --build build
#   Linux:    cmake -B build -G Ninja && cmake --build build
#   产出:     build/gemm_boundary_test（直接运行）
#             build/gemm_benchmark（roofline 标注 + 同口径对照）
# ============================================================

cmake_minimum_required(VERSION 3.20)
project(sgn_gemm_standalone LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_options(-O3)

# gemm 依赖的 simd 原语文件（gemv_i8 标量锚点逐行消费 simd::dot8）
set(SIMD_SOURCES
    ../simd/scalar.cpp
    ../simd/simd_dispatch.cpp
    ../simd/x86/avx2_dot.cpp
    ../simd/x86/avx2_decode.cpp
    ../simd/x86/avx2_reduce.cpp
    ../simd/x86/avxvnni.cpp
    ../simd/x86/ssse3.cpp
    ../simd/x86/avx512.cpp
    ../simd/x86/avx512vnni.cpp
    ../simd/arm/neon.cpp
)

# gemm 源（x86 后端文件级 ISA 见下方 set_source_files_properties）
set(GEMM_SOURCES
    scalar.cpp
    gemm_dispatch.cpp
    x86/gemm_avx2vnni.cpp
    x86/gemm_avx2.cpp
    x86/gemm_avx512vnni.cpp
)

# ------------------------------------------------------------
# per-file ISA（与 engine/sgn/CMakeLists.txt 同纪律）
#   simd 浮点文件 -ffp-contract=off（R1 修订纪律，见实施计划 §五.3/§六.6）
#   gemm_avx2vnni.cpp → -mavx2 -mavxvnni
#   gemm_avx2.cpp     → -mavx2
#   gemm_avx512vnni.cpp → -mavx512f -mavx512bw -mavx512vnni
# ------------------------------------------------------------
set_source_files_properties(
    ../simd/x86/avx2_dot.cpp
    ../simd/x86/avx2_decode.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx2")
set_source_files_properties(
    ../simd/x86/avx2_reduce.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx2;-ffp-contract=off")
set_source_files_properties(
    ../simd/scalar.cpp
    PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
set_source_files_properties(
    ../simd/x86/avxvnni.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx2;-mavxvnni")
set_source_files_properties(
    ../simd/x86/ssse3.cpp
    PROPERTIES COMPILE_OPTIONS "-mssse3;-mavx2")
set_source_files_properties(
    ../simd/x86/avx512.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-ffp-contract=off")
set_source_files_properties(
    ../simd/x86/avx512vnni.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512vnni")
set_source_files_properties(
    x86/gemm_avx2vnni.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx2;-mavxvnni")
set_source_files_properties(
    x86/gemm_avx2.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx2")
set_source_files_properties(
    x86/gemm_avx512vnni.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vnni")

# 源文件以 #include "mkern/gemm/gemm_api.h" / "mkern/simd/simd_api.h" 引用，
# ../.. = engine/sgn，mkern/{gemm,simd}/... 自此解析
set(INC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")

add_executable(gemm_boundary_test
    ${GEMM_SOURCES} ${SIMD_SOURCES}
    gemm_boundary_test.cpp)
target_include_directories(gemm_boundary_test PRIVATE "${INC_DIR}")

add_executable(gemm_benchmark
    ${GEMM_SOURCES} ${SIMD_SOURCES}
    gemm_benchmark.cpp)
target_include_directories(gemm_benchmark PRIVATE "${INC_DIR}")
# float 参照基线向量化（-mfma；整型原语不受浮点收缩影响）
set_source_files_properties(
    gemm_benchmark.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")

# ------------------------------------------------------------
# gemm_vs_matmul：gemm_i8 vs ops.cpp matmul 本体同口径对比（R3 欠账②）
# 链接引擎现役 float matmul 链（ops.cpp + tensor.cpp + dispatch/registry.cpp，
# 依赖 header-only 的 common/{logger,allocator}.h）；ops.cpp 含 #pragma omp，
# per-file -fopenmp=libomp + 链接 LLVM libomp（同主构建模式）。
# ------------------------------------------------------------
if(NOT LLVM_ROOT)
    set(LLVM_ROOT "c:/kaffj/clang+llvm-22.1.8-x86_64-pc-windows-msvc"
        CACHE PATH "LLVM/Clang 安装根目录" FORCE)
endif()
set(LIBOMP_LIB "${LLVM_ROOT}/lib/libomp.lib")
set(LIBOMP_DLL "${LLVM_ROOT}/bin/libomp.dll")

add_executable(gemm_vs_matmul
    ${GEMM_SOURCES} ${SIMD_SOURCES}
    gemm_vs_matmul.cpp
    ../../autograd/ops.cpp
    ../../autograd/tensor.cpp
    ../../autograd/ops_nn.cpp
    ../../dispatch/registry.cpp)
target_include_directories(gemm_vs_matmul PRIVATE "${INC_DIR}")
set_source_files_properties(
    ../../autograd/ops.cpp
    ../../autograd/ops_nn.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma;-fopenmp=libomp")
target_link_libraries(gemm_vs_matmul PRIVATE "${LIBOMP_LIB}")
# libomp.dll 复制到输出目录（运行时依赖，同主构建模式）
add_custom_command(TARGET gemm_vs_matmul POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different "${LIBOMP_DLL}"
            $<TARGET_FILE_DIR:gemm_vs_matmul>)
