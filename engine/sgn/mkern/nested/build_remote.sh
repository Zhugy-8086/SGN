#!/usr/bin/env bash
# build_remote.sh - Linux 远程验证脚本：仅编译 nested/ 原语，产出独立可执行
#
# 背景：nested_quant设计档_2026_09_05.md §五.3/4——本机（Arrow Lake）无 AVX-512，
#   嵌套 AVX512 后端（x86/nested_avx512.cpp，F+DQ）只能编译验证；远程机器
#   （EPYC 9Y24 等）用本脚本在 Linux 独立编译 nested 原语 + 边界测试，
#   验证真指令 bit-exact（N1 跨后端）与 avx512 选入（范式同 mkern/simd/build_remote.sh）。
#
# 范围：仅编译 nested/ 下的文件（scalar / nested_dispatch / x86/nested_avx2 /
#       x86/nested_avx512 / nested_boundary_test [+ nested_benchmark]）。
#       不依赖 pybind11、不链接 libomp、无 Python——纯 C++23。
#
# 用法：
#   ./build_remote.sh                # Release 构建（默认 -O3）
#   ./build_remote.sh --bench        # + nested_benchmark（误差-成本，-O3；与 --ubsan 互斥）
#   ./build_remote.sh --ubsan        # + UBSan（抓未定义行为，-O1；Linux 运行时正常）
#   ./build_remote.sh --clang        # 用 clang++（默认 g++）
#   ./build_remote.sh --clean        # 清理 build/ 后退出
#
# 产出：nested/build/nested_boundary_test（直接运行）
#       --bench 时另产出 nested/build/nested_benchmark
#
# 环境变量：
#   CXX         覆盖编译器（默认 g++ / --clang 切 clang++）
#   EXTRA_FLAGS 追加编译选项
#   SGN_NESTED_BACKEND=scalar 强制标量对照（测试钩子）

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BUILD_DIR="$SCRIPT_DIR/build"
SRCS=(
    scalar.cpp
    nested_dispatch.cpp
    x86/nested_avx2.cpp
    x86/nested_avx512.cpp
    nested_boundary_test.cpp
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
        --bench)  BENCH=1 ;;   # 与 --ubsan 互斥（UBSan 下性能数据无意义）
        *) echo "未知参数: $arg" >&2; exit 1 ;;
    esac
done

if [ "$BENCH" = "1" ] && [ "$UBSAN" = "1" ]; then
    echo "[nested] ERROR: --bench 与 --ubsan 互斥" >&2
    exit 1
fi

if [ "$CLEAN" = "1" ]; then
    rm -rf "$BUILD_DIR"
    echo "[nested] cleaned $BUILD_DIR"
    exit 0
fi

mkdir -p "$BUILD_DIR"

if ! command -v "$CXX" >/dev/null 2>&1; then
    echo "[nested] ERROR: 找不到编译器 '$CXX'（用 CXX=... 或 --clang 指定）" >&2
    exit 1
fi

# ---- 基础编译选项（无 AVX 宏，运行时 CPUID 检测）----
# include 路径指向 engine/sgn/ 根（本脚本位于 engine/sgn/mkern/nested/，../.. = engine/sgn）
BASE_FLAGS="-O3 -std=c++23 -I../.."
if [ "$UBSAN" = "1" ]; then
    # Linux ubsan 运行时正常（Windows 发行版残缺才用陷阱模式，见 CMakeLists.txt）
    BASE_FLAGS="-O1 -std=c++23 -I../.. -fsanitize=undefined -fno-sanitize-recover=all"
fi
BASE_FLAGS="$BASE_FLAGS ${EXTRA_FLAGS:-}"

# ---- 文件级 ISA 选项（与 CMakeLists.txt set_source_files_properties 一致）----
declare -A FILE_FLAGS
FILE_FLAGS[x86/nested_avx2.cpp]="-mavx2"
FILE_FLAGS[x86/nested_avx512.cpp]="-mavx512f -mavx512dq"

echo "[nested] compiler  = $CXX"
echo "[nested] mode      = $([ "$UBSAN" = 1 ] && echo 'ubsan' || echo 'release')"
echo "[nested] build dir = $BUILD_DIR"

if [ "$BENCH" = "1" ]; then
    SRCS+=(nested_benchmark.cpp)
fi
OBJS=()
for src in "${SRCS[@]}"; do
    obj="$BUILD_DIR/${src//\//_}.o"
    flags="${FILE_FLAGS[$src]:-}"
    # shellcheck disable=SC2086
    "$CXX" $BASE_FLAGS $flags -c "$src" -o "$obj"
    OBJS+=("$obj")
    if [ -n "$flags" ]; then
        echo "[nested] compiled $src ($flags)"
    else
        echo "[nested] compiled $src (base)"
    fi
done

# ---- 链接（两个 main 互斥：boundary 排除 bench.o，bench 排除 boundary_test.o）----
OUT="$BUILD_DIR/nested_boundary_test"
BOUNDARY_OBJS=(); BENCH_OBJS=()
for obj in "${OBJS[@]}"; do
    case "$obj" in
        *nested_benchmark.cpp.o)      BENCH_OBJS+=("$obj") ;;
        *nested_boundary_test.cpp.o)  BOUNDARY_OBJS+=("$obj") ;;
        *)                            BOUNDARY_OBJS+=("$obj"); BENCH_OBJS+=("$obj") ;;
    esac
done
# shellcheck disable=SC2086
"$CXX" $BASE_FLAGS "${BOUNDARY_OBJS[@]}" -o "$OUT"
echo "[nested] OK → $OUT"

if [ "$BENCH" = "1" ]; then
    BOUT="$BUILD_DIR/nested_benchmark"
    # shellcheck disable=SC2086
    "$CXX" $BASE_FLAGS "${BENCH_OBJS[@]}" -o "$BOUT"
    echo "[nested] OK → $BOUT（性能数据仅 --bench 默认 -O3 口径有效）"
fi
echo "[nested] 运行: $OUT"
echo "[nested] 提示: 远程机器由 nested_dispatch CPUID 自动选入 avx512（F+DQ）——"
echo "[nested]       N1-x 段将直连对拍 avx512 vs 标量锚点；强制对照 SGN_NESTED_BACKEND=scalar"
