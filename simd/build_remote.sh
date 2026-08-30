#!/usr/bin/env bash
# build_remote.sh - Linux 远程验证脚本：仅编译 simd/ 文件夹，产出独立可执行
#
# 背景：simd服务器加速计划_2026_08_30.md §9.3
#   本机（Arrow Lake，无 AVX-512）无法真跑 AVX-512 路径；远程机器（EPYC 9Y24 等）
#   用本脚本在 Linux 上独立编译 simd 原语层 + 边界测试，验证真指令 bit-exact。
#
# 范围：仅编译 simd/ 下的文件（scalar / simd_dispatch / x86/* / arm/neon /
#       simd_boundary_test）。不依赖 pybind11、不链接 libomp、无 Python——
#       纯 C++23，产出单个可执行文件。
#
# 用法：
#   ./build_remote.sh                # Release 构建（默认 AVX-512 全开）
#   ./build_remote.sh --ubsan        # + UBSan（抓未定义行为，-O1）
#   ./build_remote.sh --clang        # 用 clang++（默认 g++）
#   ./build_remote.sh --clean        # 清理 build/ 后退出
#
# 产出：simd/build/simd_boundary_test（直接运行：./simd/build/simd_boundary_test）
#
# 环境变量：
#   CXX       覆盖编译器（默认 g++ / clang++）
#   EXTRA_FLAGS 追加编译选项（如 -march=native 覆盖 AVX-512 检测）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BUILD_DIR="$SCRIPT_DIR/build"
SRCS=(
    scalar.cpp
    simd_dispatch.cpp
    x86/avx2.cpp
    x86/avxvnni.cpp
    x86/ssse3.cpp
    x86/avx512.cpp
    x86/avx512vnni.cpp
    arm/neon.cpp
    simd_boundary_test.cpp
)

# ---- 默认参数 ----
CXX="${CXX:-g++}"
MODE="release"
UBSAN=0
CLEAN=0

for arg in "$@"; do
    case "$arg" in
        --ubsan)  UBSAN=1 ;;
        --clang)  CXX="clang++" ;;
        --clean)  CLEAN=1 ;;
        *) echo "未知参数: $arg" >&2; exit 1 ;;
    esac
done

if [ "$CLEAN" = "1" ]; then
    rm -rf "$BUILD_DIR"
    echo "[simd] cleaned $BUILD_DIR"
    exit 0
fi

mkdir -p "$BUILD_DIR"

# ---- 编译器能力检查 ----
if ! command -v "$CXX" >/dev/null 2>&1; then
    echo "[simd] ERROR: 找不到编译器 '$CXX'（用 CXX=... 或 --clang 指定）" >&2
    exit 1
fi

# ---- 编译选项 ----
# 基础：C++23。⚠️ 必须带 -mavx2 -mavxvnni（与 CMakeLists.txt 全局选项一致）：
#   avx2.cpp / avxvnni.cpp 用 #if defined(__AVX2__/__AVXVNNI__) 保护整个实现，
#   dispatch 的赋值也用同宏保护——若基础编译缺这些宏，AVX2/AVX-VNNI 实现将
#   编译为空、路径永不选中，远程机器只剩标量（2026-08-31 审查发现的 bug）。
# include 路径指向 simd/ 的父目录（engine/sgn/），因源文件用 #include "simd/simd_api.h"。
BASE_FLAGS="-O3 -std=c++23 -mavx2 -mavxvnni -I.."
if [ "$UBSAN" = "1" ]; then
    BASE_FLAGS="-O1 -std=c++23 -mavx2 -mavxvnni -I.. -fsanitize=undefined -fno-sanitize-recover=all"
fi
BASE_FLAGS="$BASE_FLAGS ${EXTRA_FLAGS:-}"

# AVX-512 仅对 avx512 源文件单独启用（per-file，不污染全局编译）
AVX512_FLAGS="-mavx512f -mavx512bw -mavx512vl -mavx512vnni"

echo "[simd] compiler  = $CXX"
echo "[simd] mode      = $([ "$UBSAN" = 1 ] && echo 'ubsan' || echo 'release')"
echo "[simd] build dir = $BUILD_DIR"

# ---- 逐文件编译为 .o（avx512 单独带指令选项）----
OBJS=()
for src in "${SRCS[@]}"; do
    obj="$BUILD_DIR/${src//\//_}.o"
    # shellcheck disable=SC2086
    "$CXX" $BASE_FLAGS -c "$src" -o "$obj"
    if [[ "$src" == x86/avx512* ]]; then
        # 重新编译：AVX-512 版本（带指令选项）
        obj512="$BUILD_DIR/${src//\//_}_avx512.o"
        # shellcheck disable=SC2086
        "$CXX" $BASE_FLAGS $AVX512_FLAGS -c "$src" -o "$obj512"
        OBJS+=("$obj512")
    else
        OBJS+=("$obj")
    fi
    echo "[simd] compiled $src"
done

# ---- 链接 ----
OUT="$BUILD_DIR/simd_boundary_test"
# shellcheck disable=SC2086
"$CXX" $BASE_FLAGS "${OBJS[@]}" -o "$OUT"

echo "[simd] OK → $OUT"
echo "[simd] 运行: $OUT"
echo "[simd] 提示: 远程机器 AVX-512 路径由 simd_dispatch CPUID 检测自动启用；"
echo "[simd]       如需强制某后端对比，设 SGN_KERNEL_BACKEND=scalar/avx2/avx512"
