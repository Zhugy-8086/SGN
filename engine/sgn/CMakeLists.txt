cmake_minimum_required(VERSION 3.20)
project(sgn_extension LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ============================================================
# 平台引导（2026-09-07 EPYC 部署教训：Linux 用户误闯主构建会收到
# 混乱的 Clang/libomp 报错——engine 主构建是 Windows 工作流
# （pybind11 + LLVM libomp + Python ABI cp314），Linux 走内核层独立构建）
# ============================================================
if(NOT WIN32)
    message(FATAL_ERROR
        "engine 主构建目前仅支持 Windows（pybind11 + LLVM libomp + Python ABI）。\n"
        "  Linux 请使用内核层独立构建（无 pybind11/libomp/Python 依赖）：\n"
        "    cd engine/sgn/mkern/simd   && ./build_remote.sh --bench\n"
        "    cd engine/sgn/mkern/nested && ./build_remote.sh\n"
        "    cd engine/sgn/mkern/gemm   && cmake -B build -G Ninja && cmake --build build")
endif()

# ============================================================
# LLVM/Clang 安装路径（可被 -DLLVM_ROOT=... 覆盖）
# ============================================================
# 默认指向本机 Clang 22.1.8 安装目录；换机器时用 -DLLVM_ROOT=/path/to/llvm 覆盖。
if(NOT LLVM_ROOT)
    message(FATAL_ERROR
        "LLVM_ROOT 未设置：请指定 LLVM/Clang 22.1.8 安装根目录（含 lib/ 与 bin/，需 libomp），
        例如 -DLLVM_ROOT=<LLVM 安装目录>")
endif()
set(LIBOMP_LIB "${LLVM_ROOT}/lib/libomp.lib")
set(LIBOMP_DLL "${LLVM_ROOT}/bin/libomp.dll")
message(STATUS "LLVM_ROOT = ${LLVM_ROOT}")
message(STATUS "LIBOMP_LIB = ${LIBOMP_LIB}")

# 默认使用 Release 模式，确保 -O3 优化生效
# Debug 模式下 -O0 会覆盖全局 -O3，导致性能下降 10-100x
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# AVX2 / AVX-VNNI 支持（阶段 4 收尾：全局宏已移除，ISA 能力全部下沉到 per-file /
# target 属性，见 内部档案）。
# 各文件 ISA 归属：simd/x86/*.（阶段 1）、hc/（阶段 3）、ops/ops_nn（阶段 4 target 化）、
# placeholder.cpp（测试钩子 VNNI）。
add_compile_options(-O3)

# AVX-512 支持（条件编译，不要求测试通过）
# 当前平台（Intel Core Ultra 5 225, Arrow Lake）不支持 AVX-512。
# 仅在目标 CPU 支持时启用，代码中通过 __AVX512F__ 宏保护。
# 如需测试，在命令行添加 -DENABLE_AVX512=ON 可强制启用（但链接可能失败）。
option(ENABLE_AVX512 "Enable AVX-512F instructions (experimental)" OFF)
if(ENABLE_AVX512)
    message(STATUS "AVX-512F enabled (experimental)")
    add_compile_options(-mavx512f)
endif()

# MemorySanitizer 可选开关 (与 ASan 不兼容，只能选一个)
# 启用方式: -DENABLE_MSAN=ON
option(ENABLE_MSAN "Enable MemorySanitizer" OFF)
if(ENABLE_MSAN)
    message(STATUS "MemorySanitizer enabled")
    add_compile_options(-fsanitize=memory -fno-omit-frame-pointer -g)
    add_link_options(-fsanitize=memory)
endif()

find_package(pybind11 CONFIG REQUIRED)

# OpenMP 支持（col2im 原 C 代码使用 #pragma omp parallel for 并行化）
# Clang/Windows 只能使用 libomp（-fopenmp 仍生成 __kmpc_* 符号，VCOMP140 不兼容）
add_compile_options(-fopenmp=libomp)
# getenv 为文档化后端强制钩子（SGN_GEMM_BACKEND 等）的 CRT 弃用静音
add_compile_definitions(_CRT_SECURE_NO_WARNINGS)

# 阶段 4（2026-08-31）：压制 LTO。
# LTO 与 ops/ops_nn 的 __attribute__((target("avx2,fma"))) 函数（内含 #pragma omp
# outlined 函数）在 lld 22.1.8 后端冲突，链接崩溃（'LLVM ERROR: Do not know how to
# split the result of this operator!'，见 内部档案 §9）。
# 根因：pybind11_add_module 在 CMAKE_INTERPROCEDURAL_OPTIMIZATION 未定义时自动链接
# pybind11::lto（其 INTERFACE -flto 经 target 级注入，落在全局 -fno-lto 之后，
# last-wins 反转为开）。故先定义该变量为 OFF（pybind11 跳过 LTO），再全局 -fno-lto 兜底。
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)
add_compile_options(-fno-lto)
add_link_options(-fno-lto)

# ============================================================
# HC 扩展目录（pysgn_net / pysgn_hc16 / pysgn_hc16ms / pysgn_hc4_pshufb 合并到 sgn）
#   - HC_EXT_DIR: hc/ext/（C 源码 + pybind11 绑定，位于 engine/sgn/hc/ext/）
#   - HC_INCLUDE_DIR: include/（提供 hc/hc8.h, hc/hc.h 类型定义；
#     头文件原在 engine/hc/sgn/include/，2026-08-16 决策项 1 剥离后迁至
#     engine/sgn/include/，engine/hc/ 其余内容已冻结至 legacy/hc/）
# ============================================================
set(HC_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
# 安全审计 2026-08-16 B1-4：头文件目录存在性校验（路径假设失效时
# 给出明确 FATAL_ERROR 而非编译期散乱的 include 报错）
if(NOT EXISTS "${HC_INCLUDE_DIR}/hc/hc.h")
    message(FATAL_ERROR
        "HC_INCLUDE_DIR 无效：${HC_INCLUDE_DIR}/hc/hc.h 不存在。\n"
        "  若 include/hc/ 被移动，请同步更新本文件的 HC_INCLUDE_DIR。")
endif()

pybind11_add_module(sgn
    placeholder.cpp
    level/level_bindings.cpp
    level/level_scheduler.cpp
    hc/col2im.cpp
    hc/col2im_bindings.cpp
    hc/hc8_coproduct_bindings.cpp
    hc/hc_decode.cpp
    hc/hc_decode_bindings.cpp
    common/value_spec_bindings.cpp
    common/unit_value_bindings.cpp
    common/precision_budget_bindings.cpp
    msint/packed_backend.cpp
    msint/packed_backend_bindings.cpp
    msint/msint_view.cpp
    msint/msint_view_bindings.cpp
    msint/split_dot.cpp
    msint/split_dot_bindings.cpp
    msint/multiscale_view.cpp
    msint/multiscale_view_bindings.cpp
    msint/precision_selector.cpp
    msint/precision_selector_bindings.cpp
    msint/leveled_split_dot.cpp
    msint/leveled_split_dot_bindings.cpp
    # int8 对（h,l）梯度载体（U 方案，C++ 内部闭环；见 pair_grad_carrier.h 头注；
    # 2026-09-02：Phase 1 无 Python 绑定，Phase 2 由 autograd 叶梯度存储挂钩调用）
    msint/pair_grad_carrier.cpp
    # SIMD 原语库（平台无关原语层；Step 0 骨架，见 内部档案）
    mkern/simd/scalar.cpp
    mkern/simd/simd_dispatch.cpp
    # SIMD 原语 Python 绑定（sgn.mkern_simd 子模块，层 2 LeveledState 状态消费
    # 计算域入口：dot8/dot4/dot4_packed/unpack_nibble_u,s；层 2 设计 §八 Phase 2a）
    mkern/simd/simd_bindings.cpp
    # AVX2 原语（2026-09-02 按原语域拆分，见 simd_dot8_avx2中间路径设计档_2026_09_02.md §二.5）：
    # dot 域（dot16 + dot8/dot4 AVX2 中间档）· decode 域 · float 归约域
    mkern/simd/x86/avx2_dot.cpp
    mkern/simd/x86/avx2_decode.cpp
    mkern/simd/x86/avx2_reduce.cpp
    mkern/simd/x86/avxvnni.cpp
    mkern/simd/x86/ssse3.cpp
    # 服务器加速（P1，见 内部档案）：
    # avx512*.cpp 单独加 -mavx512* 文件级选项（下方 set_source_files_properties），
    # 符号常驻产出；dispatch 层运行时 CPUID 检测决定是否选入（本机无 AVX-512 自动落低档）。
    mkern/simd/x86/avx512.cpp
    mkern/simd/x86/avx512vnni.cpp
    mkern/simd/arm/neon.cpp
    # mkern/gemm 矩阵级微内核（R3，见 内部档案 §3.2）：
    # gemm_i8（VNNI 面板 tile）/ gemm_i16（N 轴向量化）/ gemv_i8（逐行消费 simd::dot8）
    mkern/gemm/scalar.cpp
    mkern/gemm/gemm_dispatch.cpp
    mkern/gemm/x86/gemm_avx2vnni.cpp
    mkern/gemm/x86/gemm_avx2.cpp
    mkern/gemm/x86/gemm_avx512vnni.cpp
    # mkern/nested 嵌套量化原语（nested_quant 设计档 §二，见 mkern/nested/nested_api.h；
    # 一次 SR 量化多档取，scalar→avx2→avx512 CPUID 链；avx512*.cpp per-file ISA 见下方）
    mkern/nested/scalar.cpp
    mkern/nested/nested_dispatch.cpp
    mkern/nested/nested_bindings.cpp
    mkern/nested/x86/nested_avx2.cpp
    mkern/nested/x86/nested_avx512.cpp
    # Autograd 框架（Stage 3.2 Phase 4-5：Tensor/Tape/算子/nn autograd-aware/pybind11 绑定）
    autograd/tensor.cpp
    autograd/sr_kernel.cpp
    autograd/ops.cpp
    autograd/ops_nn.cpp
    # 激活族算子（模块化：独立成文件，基础设施缺口分析批次 2）
    autograd/ops_activation.cpp
    # 归一化族算子（模块化：独立成文件，批次 4 LayerNorm）
    autograd/ops_norm.cpp
    autograd/autograd.cpp
    autograd/autograd_nn.cpp
    autograd/autograd_bindings.cpp
    # Dispatch 内核注册表（节奏 0：平台无关内核接口 + 一次性能力解析）
    dispatch/registry.cpp
    # 双模式系统：Module 基类 + pybind11 绑定（Phase 1）
    autograd/module.cpp
    autograd/module_bindings.cpp
    # 原 C 实现（col2im_c.c + pysgn_col2im.cpp 绑定）
    hc/ext/col2im_c.c
    hc/ext/pysgn_col2im.cpp
    # 合并的 HC 扩展（pysgn_net + pysgn_hc16 + pysgn_hc16ms + pysgn_hc4_pshufb）
    # 原 4 个独立 .pyd 模块合并到 sgn 子模块：hc8_net / hc16 / hc16ms / hc4
    hc/ext/pysgn_net.cpp
    hc/ext/hc8_net.c
    hc/ext/pysgn_hc16.cpp
    hc/ext/hc16_net.c
    hc/ext/pysgn_hc16ms.cpp
    hc/ext/hc16ms.c
    hc/ext/pysgn_hc4_pshufb.cpp
    hc/ext/hc4_pshufb.c
)
target_include_directories(sgn PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/hc
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/msint
    ${CMAKE_CURRENT_SOURCE_DIR}/autograd
    ${CMAKE_CURRENT_SOURCE_DIR}/level
    # HC 扩展头文件目录（pysgn_*.cpp 内部 #include "hc8_net.h" / "hc16_net.h" 等就近解析）
    ${CMAKE_CURRENT_SOURCE_DIR}/hc/ext
    # engine/hc/sgn/include/（提供 hc/hc8.h, hc/hc.h 类型定义，pysgn_net.cpp 需要）
    ${HC_INCLUDE_DIR}
)

# OpenMP 支持（col2im 原 C 代码使用 #pragma omp parallel for 并行化）
# 直接链接 LLVM libomp.lib（-fopenmp=libomp 需要此库）
target_link_libraries(sgn PRIVATE
    "${LIBOMP_LIB}")

# 运行时依赖：将 libomp.dll 复制到 build 目录（Python 3.8+ 不从 PATH 加载 DLL）
add_custom_command(TARGET sgn POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${LIBOMP_DLL}" "$<TARGET_FILE_DIR:sgn>"
    COMMENT "Copying libomp.dll to build directory")

# 安装到 build 目录便于测试
set_target_properties(sgn PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)

# ============================================================
# sgn target 的 HC 扩展相关编译选项（合并 4 个 pysgn_* 模块后所需）
# ============================================================

# 4 个 .c 文件强制 C 编译器编译（保留 C 语义，匹配原始构建方式）
set_source_files_properties(
    hc/ext/hc8_net.c
    hc/ext/hc16_net.c
    hc/ext/hc16ms.c
    hc/ext/hc4_pshufb.c
    PROPERTIES LANGUAGE C)

# AVX-512 服务器加速（P1，见 内部档案）：
# 仅 avx512*.cpp 启用 AVX-512 指令（文件级选项，不污染全局编译），符号常驻产出，
# 供 simd_dispatch.cpp 无条件链接；运行时 __builtin_cpu_supports 检测决定是否选入。
# 本机（Arrow Lake 无 AVX-512）可编译链接，只是运行时不会选用 AVX-512 路径。
# -ffp-contract=off（R1 修订 2026-09-03）：-mavx512f 隐含 FMA 能力，Clang 默认收缩会把
# sum 系标量尾部的 sum += d*d 收缩成单舍入 vfmadd，破坏与 scalar/avx2 的逐位一致
# （kBitExact）。显式 _mm256_fmadd intrinsic 不受此旗标影响。
set_source_files_properties(
    mkern/simd/x86/avx512.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-ffp-contract=off")
set_source_files_properties(
    mkern/simd/x86/avx512vnni.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512vnni")

# mkern/gemm 矩阵级微内核 per-file ISA（R3，同 simd 层纪律：符号常驻 + 运行时 CPUID）
set_source_files_properties(
    mkern/gemm/x86/gemm_avx2vnni.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx2;-mavxvnni")
set_source_files_properties(
    mkern/gemm/x86/gemm_avx2.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx2")
set_source_files_properties(
    mkern/gemm/x86/gemm_avx512vnni.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vnni")

# mkern/nested 嵌套量化原语 per-file ISA（同 simd/gemm 纪律；avx512 需 DQ：
# mullo_epi64/cvtqq2pd 原生依赖，见 mkern/nested/x86/nested_avx512.cpp 头注释）
set_source_files_properties(
    mkern/nested/x86/nested_avx2.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx2")
set_source_files_properties(
    mkern/nested/x86/nested_avx512.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512dq")

# 浮点归约文件防御性禁收缩（R1 修订 2026-09-03）：avx2_reduce/scalar 当前未启用 FMA
# （无收缩），但未来旗标变动（如 -march=native）会静默引入 vfmadd——直接在文件级锁死。
# 整型/解码/字节原语文件无浮点算术，不涉及。
set_source_files_properties(
    mkern/simd/x86/avx2_reduce.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx2;-ffp-contract=off")
set_source_files_properties(
    mkern/simd/scalar.cpp
    PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")

# AVX2 / AVX-VNNI / SSSE3 实现（阶段 1，见 内部档案）：
# 文件级选项显式声明各文件所需 ISA。当前全局 -mavx2 -mavxvnni 仍存在（冗余但无害），
# 阶段 4 移除全局选项时，此处文件级选项接管，保证符号仍产出。
# avx2_dot/avx2_decode.cpp: 需要 AVX2（2026-09-02 按域拆分）；
# avx2_reduce.cpp 单列（上方 -ffp-contract=off，浮点归约禁收缩）；
# avxvnni.cpp: 需要 AVX-VNNI（隐含 AVX2）；
# ssse3.cpp: reverse 需 SSSE3，batch 用 _mm256_shuffle_epi8 需 AVX2。
set_source_files_properties(
    mkern/simd/x86/avx2_dot.cpp
    mkern/simd/x86/avx2_decode.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx2")
set_source_files_properties(
    mkern/simd/x86/avxvnni.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx2;-mavxvnni")
set_source_files_properties(
    mkern/simd/x86/ssse3.cpp
    PROPERTIES COMPILE_OPTIONS "-mssse3;-mavx2")

# HC 层 per-file ISA（阶段 3，见 内部档案）：
# hc 层为 AVX2 专用实现（各文件自带 CPUID 运行时检测；hc8_net.c 另有 SEH 探针）。
# per-file 显式声明后，阶段 4 移除全局 -mavx2 -mavxvnni 时 hc 层行为不变
# （符号常驻产出，运行时检测门控；与阶段 1 simd 层同纪律）。
# hc8_net.c: AVX2 + AVX-VNNI（SEH 探针执行 _mm256_dpbusd_epi32，__try 捕获非法指令）
#             + FMA（_mm256_fmadd_ps）；
# hc16_net.c / hc4_pshufb.c / hc16ms.c: 仅 AVX2（madd/sad/pshufb/shuffle）；
# hc_decode.cpp / pysgn_net.cpp: 直接使用 AVX2+FMA intrinsic（无宏保护，需文件级 ISA）；
# placeholder.cpp: test_avx_vnni 测试钩子（_mm256_dpbusd_epi32），VNNI intrinsic 需文件级 ISA。
set_source_files_properties(
    hc/ext/hc8_net.c
    PROPERTIES COMPILE_OPTIONS "-mavx2;-mavxvnni;-mfma")
set_source_files_properties(
    hc/ext/hc16_net.c
    hc/ext/hc4_pshufb.c
    hc/ext/hc16ms.c
    PROPERTIES COMPILE_OPTIONS "-mavx2")
set_source_files_properties(
    hc/hc_decode.cpp
    hc/ext/pysgn_net.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
set_source_files_properties(
    placeholder.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx2;-mavxvnni")
# ops.cpp / ops_nn.cpp：matmul/bn/conv 内核含 AVX2+FMA intrinsic 且带 #pragma omp。
# 文件级 ISA（阶段 4，见调查 §9：无全局宏时 target() 属性与 omp outlined 函数组合
# 触发 clang 22.1.8 后端编译崩溃 'Do not know how to split the result of this operator!'，
# 改用 per-file，行为 = 现状全局宏）。
set_source_files_properties(
    autograd/ops.cpp
    autograd/ops_nn.cpp
    PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")

# SGN_PC_EXTENSION 宏（对应原 setup_*.py 的 define_macros，4 个 .c 文件依赖此宏）
target_compile_definitions(sgn PRIVATE SGN_PC_EXTENSION=1)

# FMA: pysgn_net.cpp 用 _mm256_fmadd_pd，hc8_net.c 用 _mm256_fmadd_ps，需要 -mfma。
# MSVC /arch:AVX2 会隐式启用 FMA，但 Clang -mavx2 不会（Clang 把 AVX2/FMA 视为独立 feature）。
# 目标 CPU（Intel Core Ultra 5 225, Arrow Lake）支持 FMA，启用安全。
# （阶段 4：此文件级 FMA 需求已由各文件 per-file COMPILE_OPTIONS 承载，不再全局开启）
# target_compile_options(sgn PRIVATE -mfma)

# ============================================================
# 以下 4 个独立 pysgn_* target 已合并到 sgn 模块（hc8_net / hc16 / hc16ms / hc4 子模块）
# 原 pysgn_hc4_pshufb / pysgn_hc16ms / pysgn_hc16 / pysgn_net 的源文件已并入 sgn target。
# 注意：不再链接 hc8.c / hc.c（评估确认 hc8_net.c 不调用它们，属于死代码）。
# ============================================================

# ============================================================================
# pysgn_col2im 已合并到 sgn 统一模块（2026-08-06）
#   - pysgn_col2im.cpp 绑定已改为 register_col2im_c 注册函数
#   - 注册到 sgn.col2im_c 子模块（原 col2im_add 函数可通过 sgn.col2im_c.col2im_add 访问）
#   - col2im_c.c 源文件已加入 sgn target 编译
# ============================================================================

# ============================================================================
# 测试目标（非 Python 扩展）
#   - 这些目标用于 C++ 层面的独立测试，可以在 Debug + ASan 模式下运行
#   - ASan 仅用于这些非 .pyd 测试目标（.pyd 目标与 Python 内存管理器不兼容，不启用 ASan）
#   - 全局 add_compile_options/add_link_options 不会启用 ASan（仅 MSan 有全局开关）
#   - 仅在 Debug 模式下启用 ASan（Release 模式下 -O3 与 ASan 混用性能差且无意义）
# ============================================================================
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # 测试目标不链接 sgn MODULE（Windows 上不能链接 .pyd），而是直接编译需要的源文件
    # 注意：这些源文件会被 sgn MODULE 和 test_phase* 各编译一次，Debug 模式下可接受

    # 公共 include 路径
    set(AUTOGRAD_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})

    # test_phase1: Storage + Tensor + matmul forward/backward 验证（vs PyTorch）
    # 注意：不使用 -fsanitize=address（ASan + Debug CRT 在 Windows Clang 上不兼容）
    add_executable(test_phase1 autograd/test_phase1.cpp autograd/tensor.cpp autograd/ops.cpp dispatch/registry.cpp)
    target_include_directories(test_phase1 PRIVATE ${AUTOGRAD_INCLUDE_DIRS})
    target_compile_options(test_phase1 PRIVATE -g)

    # test_phase2: tape-based Autograd 引擎验证（链式法则 + 累加）
    add_executable(test_phase2 autograd/test_phase2.cpp autograd/tensor.cpp autograd/ops.cpp autograd/autograd.cpp dispatch/registry.cpp)
    target_include_directories(test_phase2 PRIVATE ${AUTOGRAD_INCLUDE_DIRS})
    target_compile_options(test_phase2 PRIVATE -g)

    # test_phase3: 神经网络算子验证（linear/relu/bn/conv2d/maxpool forward+backward）
    add_executable(test_phase3 autograd/test_phase3.cpp autograd/tensor.cpp autograd/ops.cpp autograd/ops_nn.cpp autograd/autograd.cpp autograd/autograd_nn.cpp dispatch/registry.cpp)
    target_include_directories(test_phase3 PRIVATE ${AUTOGRAD_INCLUDE_DIRS})
    target_compile_options(test_phase3 PRIVATE -g -fopenmp=libomp)
    target_link_libraries(test_phase3 PRIVATE "${LIBOMP_LIB}")

    # test_pair_grad_carrier: int8 对梯度载体验证（P1 往返 / P2 SR 双路 bit-exact /
    # P3-P4 dot8 消费恒等式 / P5 浮点解码；依赖 simd 原语 + 共享 SR 内核 sr_kernel）
    add_executable(test_pair_grad_carrier
        tests/architecture/test_pair_grad_carrier.cpp
        msint/pair_grad_carrier.cpp
        autograd/sr_kernel.cpp
        mkern/simd/scalar.cpp mkern/simd/simd_dispatch.cpp
        mkern/simd/x86/avx2_dot.cpp mkern/simd/x86/avx2_decode.cpp mkern/simd/x86/avx2_reduce.cpp
        mkern/simd/x86/avxvnni.cpp mkern/simd/x86/ssse3.cpp
        mkern/simd/x86/avx512.cpp mkern/simd/x86/avx512vnni.cpp mkern/simd/arm/neon.cpp)
    target_include_directories(test_pair_grad_carrier PRIVATE ${AUTOGRAD_INCLUDE_DIRS})
    # per-file ISA 与主构建一致（simd 原语文件需要各自指令集选项）
    # 注意：本块在主构建属性块之后执行、会覆盖同名文件属性——R1 修订的 -ffp-contract=off
    # 必须在此同步（avx512/avx2_reduce/scalar 三个浮点文件），否则测试目标回退到收缩语义。
    set_source_files_properties(
        mkern/simd/x86/avx2_dot.cpp mkern/simd/x86/avx2_decode.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx2")
    set_source_files_properties(
        mkern/simd/x86/avx2_reduce.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx2;-ffp-contract=off")
    set_source_files_properties(
        mkern/simd/scalar.cpp
        PROPERTIES COMPILE_OPTIONS "-ffp-contract=off")
    set_source_files_properties(
        mkern/simd/x86/avxvnni.cpp PROPERTIES COMPILE_OPTIONS "-mavx2;-mavxvnni")
    set_source_files_properties(
        mkern/simd/x86/ssse3.cpp PROPERTIES COMPILE_OPTIONS "-mssse3;-mavx2")
    set_source_files_properties(
        mkern/simd/x86/avx512.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-ffp-contract=off")
    set_source_files_properties(
        mkern/simd/x86/avx512vnni.cpp PROPERTIES COMPILE_OPTIONS "-mavx512f;-mavx512bw;-mavx512vl;-mavx512vnni")

    # test_tape_hybrid: tape 混合语义守卫（2026-09-07 外部审查 A2/A3/A1 实证）——
    # pair↔float 切换缓存语义（F1 入口清 pair / grads_ 累加）+ QuantConfigGuard
    # RAII 恢复 + clip_sigma 惰性实证（max-scale 下 sigma=4 ≡ sigma=1e30）
    add_executable(test_tape_hybrid
        tests/architecture/test_tape_hybrid.cpp
        autograd/tensor.cpp autograd/ops.cpp autograd/ops_nn.cpp autograd/autograd.cpp
        dispatch/registry.cpp
        msint/pair_grad_carrier.cpp
        autograd/sr_kernel.cpp
        mkern/simd/scalar.cpp mkern/simd/simd_dispatch.cpp
        mkern/simd/x86/avx2_dot.cpp mkern/simd/x86/avx2_decode.cpp mkern/simd/x86/avx2_reduce.cpp
        mkern/simd/x86/avxvnni.cpp mkern/simd/x86/ssse3.cpp
        mkern/simd/x86/avx512.cpp mkern/simd/x86/avx512vnni.cpp mkern/simd/arm/neon.cpp)
    target_include_directories(test_tape_hybrid PRIVATE ${AUTOGRAD_INCLUDE_DIRS})
    # ops_nn.cpp（registry 的 conv AVX2 内核宿主）含 #pragma omp → 与 test_phase3 同模式
    target_compile_options(test_tape_hybrid PRIVATE -fopenmp=libomp)
    target_link_libraries(test_tape_hybrid PRIVATE "${LIBOMP_LIB}")

    # test_cpuid_caps_mock: CPUID caps 纯派生 mock 回归（外部审查 A4）——
    # Arrow Lake OSPKE leaf 误读 bug 固化。测试 TU 直接 include
    # simd_dispatch.cpp 触达匿名命名空间纯派生函数 caps_from_registers，
    # 故本目标**不含** simd_dispatch.cpp 源（防 ODR 重复符号）。
    add_executable(test_cpuid_caps_mock
        mkern/simd/test_cpuid_caps_mock.cpp
        mkern/simd/scalar.cpp
        mkern/simd/x86/avx2_dot.cpp mkern/simd/x86/avx2_decode.cpp mkern/simd/x86/avx2_reduce.cpp
        mkern/simd/x86/avxvnni.cpp mkern/simd/x86/ssse3.cpp
        mkern/simd/x86/avx512.cpp mkern/simd/x86/avx512vnni.cpp mkern/simd/arm/neon.cpp)
    target_include_directories(test_cpuid_caps_mock PRIVATE ${AUTOGRAD_INCLUDE_DIRS})
endif()
