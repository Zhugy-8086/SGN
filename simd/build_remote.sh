#!/usr/bin/env bash
# build_remote.sh - Linux 远程验证脚本：仅编译 simd/ 文件夹，产出独立可执行
#
# 背景：simd服务器加速计划_2026_08_30.md §9.3 + 全局AVX编译参数移除调查_2026_08_31.md 阶段 1
#   本机（Arrow Lake，无 AVX-512）无法真跑 AVX-512 路径；远程机器（EPYC 9Y24 等）
#   用本脚本在 Linux 上独立编译 simd 原语层 + 边界测试，验证真指令 bit-exact。
#
# 范围：仅编译 simd/ 下的文件（scalar / simd_dispatch / x86/* / arm/neon /
#       simd_boundary_test）。不依赖 pybind11、不链接 libomp、无 Python——
#       纯 C++23，产出单个可执行文件。
#
# 阶段 1 per-file 方案（与 CMakeLists.txt 一致）：
#   基础编译（无 AVX 宏）→ scalar.cpp / simd_dispatch.cpp / arm/neon.cpp /
#                            simd_boundary_test.cpp
#   文件级 ISA 选项 → avx2(-mavx2) / avxvnni(-mavx2 -mavxvnni) /
#                     ssse3(-mssse3 -mavx2) / avx512(-mavx512f -mavx512bw -mavx512vl) /
#                     avx512vnni(同上 + -mavx512vnni)
#   simd_dispatch.cpp 无条件引用各实现符号（运行时 CPUID 决定选入），故所有实现文件
#   必须编译产出符号——不再依赖全局 -mavx2 -mavxvnni。
#
# 用法：
#   ./build_remote.sh                # Release 构建（默认 AVX-512 全开）
#   ./build_remote.sh --bench        # + sgn_benchmark（原语正确性+性能基准，-O3）
#   ./build_remote.sh --ubsan        # + UBSan（抓未定义行为，-O1；与 --bench 互斥）
#   ./build_remote.sh --clang        # 用 clang++（默认 g++）
#   ./build_remote.sh --clean        # 清理 build/ 后退出
#
# 产出：simd/build/simd_boundary_test（直接运行：./simd/build/simd_boundary_test）
#       --bench 时另产出 simd/build/sgn_benchmark（CPUID 修复复核 + dot8 性能）
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

# ---- 参数 ----
CXX="${CXX:-g++}"
MODE="release"
UBSAN=0
CLEAN=0
BENCH=0

for arg in "$@"; do
    case "$arg" in
        --ubsan)  UBSAN=1 ;;
        --clang)  CXX="clang++" ;;
        --clean)  CLEAN=1 ;;
        --bench)  BENCH=1 ;;   # 编译 sgn_benchmark（原语正确性+性能基准，-O3）；
                               # 与 --ubsan 互斥（UBSan 下性能数据无意义）
        *) echo "未知参数: $arg" >&2; exit 1 ;;
    esac
done

if [ "$BENCH" = "1" ] && [ "$UBSAN" = "1" ]; then
    echo "[simd] ERROR: --bench 与 --ubsan 互斥（UBSan -O1 下性能数据无意义）" >&2
    exit 1
fi

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

# ---- 基础编译选项（阶段 1：无 AVX 宏，运行时 CPUID 检测）----
# include 路径指向 simd/ 的父目录（engine/sgn/），因源文件用 #include "simd/simd_api.h"。
BASE_FLAGS="-O3 -std=c++23 -I.."
if [ "$UBSAN" = "1" ]; then
    BASE_FLAGS="-O1 -std=c++23 -I.. -fsanitize=undefined -fno-sanitize-recover=all"
fi
BASE_FLAGS="$BASE_FLAGS ${EXTRA_FLAGS:-}"

# ---- 文件级 ISA 选项（阶段 1，与 CMakeLists.txt set_source_files_properties 一致）----
# dispatch/scalar/neon/测试：基础编译（无 AVX 宏），dispatch 全走运行时 CPUID。
declare -A FILE_FLAGS
FILE_FLAGS[x86/avx2.cpp]="-mavx2"
FILE_FLAGS[x86/avxvnni.cpp]="-mavx2 -mavxvnni"
FILE_FLAGS[x86/ssse3.cpp]="-mssse3 -mavx2"
FILE_FLAGS[x86/avx512.cpp]="-mavx512f -mavx512bw -mavx512vl"
FILE_FLAGS[x86/avx512vnni.cpp]="-mavx512f -mavx512bw -mavx512vl -mavx512vnni"

echo "[simd] compiler  = $CXX"
echo "[simd] mode      = $([ "$UBSAN" = 1 ] && echo 'ubsan' || echo 'release')"
echo "[simd] build dir = $BUILD_DIR"

# ---- 逐文件编译为 .o（按文件级 ISA 选项；无映射的文件走基础编译）----
# --bench 模式额外编译 sgn_benchmark.cpp（两个 main 分开链接）
if [ "$BENCH" = "1" ]; then
    SRCS+=(sgn_benchmark.cpp)
fi
OBJS=()
for src in "${SRCS[@]}"; do
    obj="$BUILD_DIR/${src//\//_}.o"
    flags="${FILE_FLAGS[$src]:-}"
    # shellcheck disable=SC2086
    "$CXX" $BASE_FLAGS $flags -c "$src" -o "$obj"
    OBJS+=("$obj")
    if [ -n "$flags" ]; then
        echo "[simd] compiled $src ($flags)"
    else
        echo "[simd] compiled $src (base)"
    fi
done

# ---- 链接（基础选项即可；各 .o 已带各自指令）----
# 两个 main 互斥链接：boundary exe 排除 bench.o，bench exe 排除 boundary_test.o
OUT="$BUILD_DIR/simd_boundary_test"
BOUNDARY_OBJS=(); BENCH_OBJS=()
for obj in "${OBJS[@]}"; do
    case "$obj" in
        *sgn_benchmark.cpp.o)       BENCH_OBJS+=("$obj") ;;
        *simd_boundary_test.cpp.o)  BOUNDARY_OBJS+=("$obj") ;;
        *)                          BOUNDARY_OBJS+=("$obj"); BENCH_OBJS+=("$obj") ;;
    esac
done
# shellcheck disable=SC2086
"$CXX" $BASE_FLAGS "${BOUNDARY_OBJS[@]}" -o "$OUT"
echo "[simd] OK → $OUT"

if [ "$BENCH" = "1" ]; then
    BOUT="$BUILD_DIR/sgn_benchmark"
    # shellcheck disable=SC2086
    "$CXX" $BASE_FLAGS "${BENCH_OBJS[@]}" -o "$BOUT"
    echo "[simd] OK → $BOUT（性能数据仅 --bench 默认 -O3 口径有效）"
fi
echo "[simd] 运行: $OUT"
echo "[simd] 提示: 远程机器各路径由 simd_dispatch CPUID 检测自动启用（VNNI/AVX2/AVX-512）；"
echo "[simd]       强制后端对比仅支持 SGN_KERNEL_BACKEND=scalar（或 sse2），其余取值告警后忽略"
