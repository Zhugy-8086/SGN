# ============================================================
# nested/ 嵌套量化原语独立构建入口（仿 mkern/gemm/CMakeLists.txt）
# ============================================================
# 用途：仅编译 nested/ 原语（scalar + x86 后端），产出独立可执行，
#       不依赖 pybind11 / libomp / Python，与 engine/sgn 主构建完全隔离。
#
# 用法：
#   cmake -B build -G Ninja -DCMAKE_CXX_COMPILER=clang++ -DCMAKE_C_COMPILER=clang
#   cmake --build build
#   产出: build/nested_boundary_test（N1-N4，直接运行）
#         build/nested_boundary_test_ubsan（同一源 + UBSan 陷阱，N4 卫生验证）
#         build/nested_benchmark（误差-成本双口径，设计档 §二.3）
#
# 纪律：build/ 产物不入库（SGN .gitignore 已覆盖；GCC 交叉验证目录 build_gcc/
# 同为产物，验证完即删）。

cmake_minimum_required(VERSION 3.20)
project(sgn_nested_standalone LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_compile_options(-O3 -Wall -Wextra)

# ------------------------------------------------------------
# per-file ISA（同 simd/gemm 纪律；标量/调度文件无 ISA 旗标）
#   x86/nested_avx2.cpp   → -mavx2
#   x86/nested_avx512.cpp → -mavx512f -mavx512dq（mullo_epi64/cvtqq2pd 依赖 DQ）
# 运行时选入由 nested_dispatch.cpp 的 CPUID 检测决定（含 XCR0 opmask+ZMM 检查）。
# ------------------------------------------------------------
set(NESTED_SOURCES
    scalar.cpp
    nested_dispatch.cpp
    x86/nested_avx2.cpp
    x86/nested_avx512.cpp
)
set_source_files_properties(
    x86/nested_avx2.cpp PROPERTIES COMPILE_OPTIONS "-mavx2")
set_source_files_properties(
    x86/nested_avx512.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512dq")

# 源文件以 #include "mkern/nested/nested_api.h" 引用，../.. = engine/sgn
set(INC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../..")

add_executable(nested_boundary_test
    ${NESTED_SOURCES}
    nested_boundary_test.cpp)
target_include_directories(nested_boundary_test PRIVATE "${INC_DIR}")

# ---- 独立 pyd（mkern-only 分发：Python 绑定不依赖 engine 主构建）----
# 产出 sgn_nested.cp*.pyd（模块名避免与 engine 的 sgn 包冲突；顶层重导出全部
# API，也保留 sgn_nested.mkern_nested.* 路径）。pybind11 路径：
#   -Dpybind11_DIR="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')"
option(NESTED_BUILD_PYD "Build standalone sgn_nested Python extension" OFF)
if(NESTED_BUILD_PYD)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(sgn_nested
        nested_pyd.cpp
        nested_bindings.cpp     # NESTED_SOURCES 是纯原语集（主构建才带绑定文件）
        ${NESTED_SOURCES})
    target_include_directories(sgn_nested PRIVATE "${INC_DIR}")
endif()

add_executable(nested_benchmark
    ${NESTED_SOURCES}
    nested_benchmark.cpp)
target_include_directories(nested_benchmark PRIVATE "${INC_DIR}")

# UBSan 变体（N4 卫生）：陷阱模式——本 LLVM 22.1.8 Windows 发行版的 ubsan 运行时
# 库残缺（clang_rt.ubsan_standalone 引用未随附的 __coe_win 伴侣符号，链接必败），
# 故用 -fsanitize-trap=undefined：所有 UB 检查（float-cast-overflow / shift /
# signed-integer-overflow 等）编译为 CPU 陷阱（ud2），无需运行时库，任何 UB 即崩。
add_executable(nested_boundary_test_ubsan
    ${NESTED_SOURCES}
    nested_boundary_test.cpp)
target_include_directories(nested_boundary_test_ubsan PRIVATE "${INC_DIR}")
set_target_properties(nested_boundary_test_ubsan PROPERTIES
    MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")
target_compile_options(nested_boundary_test_ubsan PRIVATE
    -O1 -g -fsanitize=undefined -fsanitize-trap=undefined)
