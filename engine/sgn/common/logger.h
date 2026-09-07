// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 zhugy-8086
//
// logger.h - SGN 统一 C++ 日志模块（轻量、零第三方依赖）
//
// 设计目标（见 engine_infrastructure_plan_2026_08_14.md P0）：
//   1. 统一 C++ 侧日志出口，替换散落的 printf/std::cout/snprintf
//   2. 级别控制：DEBUG/INFO/WARN/ERROR，环境变量 SGN_LOG_LEVEL 覆盖（默认 INFO）
//   3. 前缀统一：`[SGN] [LEVEL] <file>:<line> msg`
//   4. header-only + 内联，无额外链接依赖（保持项目零依赖惯例）
//   5. 多线程安全：级别判定与输出用单次 fprintf(stderr)，避免交错
//   6. **可插拔输出目标（sink）**：不把日志封死在 stderr——
//      未来要兼容通用算子库/后端套件（oneDNN / ONNX Runtime / TensorRT 等）
//      时，可通过 set_log_sink() 把日志路由到外部后端的日志系统。
//      当前只提供默认 stderr sink 与注册接口，不预接任何外部库。
//
// 用法：
//   SGN_LOG(SGN_INFO, "shape=[%lld,%lld]", (long long)a, (long long)b);
//   SGN_LOG(SGN_DEBUG, "matmul: %zu x %zu", M, N);
//   通过环境变量 SGN_LOG_LEVEL=DEBUG|INFO|WARN|ERROR 控制（进程内一次读取）。
//
//   // 未来接外部后端日志：实现/包装一个 sink 并注册
//   void my_sink(sgn::LogLevel lv, const char* file, int line, const char* msg);
//   sgn::set_log_sink(&my_sink);   // 替换默认 stderr sink
//
// 级别语义：
//   DEBUG - 详细诊断（默认关闭，需 SGN_LOG_LEVEL=DEBUG 开启）
//   INFO  - 常规信息（默认显示）
//   WARN  - 可恢复的异常情况
//   ERROR - 错误/不可恢复（始终显示）
//
// 注意：本模块只做文本输出；Python 侧统计型日志（sgn.logger.GradLogger）
// 保持独立，C++ 与 Python 两套语义协调但不强行合并。

#pragma once

#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

namespace sgn {

// ============================================================================
// 日志级别
// ============================================================================
enum LogLevel {
    SGN_DEBUG = 0,
    SGN_INFO = 1,
    SGN_WARN = 2,
    SGN_ERROR = 3,
};

// ============================================================================
// 输出目标（sink）——可插拔，避免把日志封死
// ============================================================================
// 输出目标签名：接收级别、文件名、行号、格式化完成的完整消息（不含前缀）。
// 外部后端可注册自己的 sink，把日志路由到其日志系统。
using LogSinkFn = void (*)(LogLevel lv, const char* file, int line, const char* msg);

// 级别名（供 sink 与 impl 使用；前向声明保证 sink 定义在实现之前）
inline const char* sgn_log_level_name(LogLevel lv) {
    switch (lv) {
        case SGN_DEBUG: return "DEBUG";
        case SGN_INFO:  return "INFO";
        case SGN_WARN:  return "WARN";
        case SGN_ERROR: return "ERROR";
        default:        return "INFO";
    }
}

// 当前时间戳 HH:MM:SS.mmm（毫秒级，跨平台 localtime_s / localtime_r）
// 供默认 sink 的前缀使用；外部自定义 sink 不经此函数，格式自定
// （施工备忘 docs/内部研究档/C++日志时间戳打点方案_2026_08_21.md §5.1）
inline void sgn_now_timestamp(char* buf, size_t cap) {
    using namespace std::chrono;
    auto now = system_clock::now();
    auto ms = duration_cast<milliseconds>(now.time_since_epoch()) % 1000;
    std::time_t t = system_clock::to_time_t(now);
    std::tm tm{};
#if defined(_MSC_VER)
    localtime_s(&tm, &t);
#else
    localtime_r(&t, &tm);
#endif
    // HH:MM:SS.mmm
    std::snprintf(buf, cap, "%02d:%02d:%02d.%03d",
                  tm.tm_hour, tm.tm_min, tm.tm_sec,
                  static_cast<int>(ms.count()));
}

// 默认 stderr sink（时间戳 + 前缀：`[SGN] [HH:MM:SS.mmm] [LEVEL] file:line msg`）
// 定义于 log_sink_ref 之前（后者默认指向它——审计 L-3）
inline void sgn_stderr_sink(LogLevel lv, const char* file, int line, const char* msg) {
    char ts[16];
    sgn_now_timestamp(ts, sizeof(ts));
    std::fprintf(stderr, "[SGN] [%s] [%s] %s:%d %s\n",
                 ts, sgn_log_level_name(lv), file, line, msg);
    std::fflush(stderr);
}

namespace detail {
// 共享的全局 sink（header-only 跨 TU 单例：函数内 static 引用，线程安全初始化）
// 默认直接指向 stderr sink（审计 L-3：省去每次 log 调用的 nullptr 回退分支）
inline std::atomic<LogSinkFn>& log_sink_ref() {
    static std::atomic<LogSinkFn> sink{&sgn_stderr_sink};
    return sink;
}
}  // namespace detail

// 设置全局输出目标（线程安全），返回旧的 sink。传 nullptr 恢复默认 stderr。
inline LogSinkFn set_log_sink(LogSinkFn sink) {
    return detail::log_sink_ref().exchange(sink ? sink : &sgn_stderr_sink);
}

// ============================================================================
// 全局级别解析（进程内读取一次环境变量）
// ============================================================================
namespace detail {

// 跨平台读取环境变量（避免 MSVC 对 getenv 的弃用警告，C11 getenv_s 兼容）
inline const char* sgn_getenv(const char* name) {
#if defined(_MSC_VER)
    static thread_local char buf[64];
    size_t len = 0;
    if (getenv_s(&len, buf, sizeof(buf), name) != 0) {
        return nullptr;
    }
    return (len == 0) ? nullptr : buf;
#else
    return std::getenv(name);
#endif
}

}  // namespace detail

inline LogLevel sgn_log_level() {
    // 线程安全：C++11 起函数内 static 初始化是线程安全的。
    // ⚠️ 语义（审计 L-1）：环境变量在首次调用时读取一次并进程级缓存——
    // 运行期修改 SGN_LOG_LEVEL 不会生效，须在程序启动前设置。
    // 非 MSVC 平台的 std::getenv 非线程安全，仅首次初始化期间存在竞争窗口，
    // 实际风险可忽略（建议在单线程阶段触发任意一次 log 以固化初始化）。
    static const LogLevel level = []() -> LogLevel {
        const char* env = detail::sgn_getenv("SGN_LOG_LEVEL");
        if (env == nullptr) {
            return SGN_INFO;
        }
        if (std::strcmp(env, "DEBUG") == 0) return SGN_DEBUG;
        if (std::strcmp(env, "WARN") == 0)  return SGN_WARN;
        if (std::strcmp(env, "ERROR") == 0) return SGN_ERROR;
        // INFO 或缺省/未知值：默认 INFO
        return SGN_INFO;
    }();
    return level;
}

// ============================================================================
// 内部实现：级别名 + 输出
// ============================================================================
// 核心输出函数（供宏调用）。level < 当前级别时不输出。
// L2 修复 2026-09-07：补 format(printf,4,5) 属性——编译器检查格式串与实参
// 匹配（Clang 诊断建议；fmt 为第 4 参、可变参数从第 5 起）
__attribute__((format(printf, 4, 5))) inline void sgn_log_impl(
    LogLevel lv, const char* file, int line, const char* fmt, ...) {
    if (lv < sgn_log_level()) {
        return;
    }
    // 只取 basename（去路径），避免日志过长
    const char* base = file;
    if (const char* slash = std::strrchr(file, '\\')) base = slash + 1;
    if (const char* slash = std::strrchr(base, '/')) base = slash + 1;

    // 先格式化完整消息，再交给当前 sink（外部后端可替换）
    char buf[2048];
    va_list args;
    va_start(args, fmt);
    int need = std::vsnprintf(buf, sizeof(buf), fmt, args);
    va_end(args);
    // 截断检测（审计 L-2）：超长消息静默截断易误导排查，追加显式标记
    if (need < 0) {
        buf[0] = '\0';  // 编码错误：输出空消息
    } else if (static_cast<size_t>(need) >= sizeof(buf)) {
        static constexpr char kTruncMark[] = "...[truncated]";
        constexpr size_t mlen = sizeof(kTruncMark) - 1;
        std::memcpy(buf + sizeof(buf) - 1 - mlen, kTruncMark, mlen + 1);
    }

    // 读当前 sink（原子）；默认已初始化为 stderr sink（审计 L-3）
    LogSinkFn sink = detail::log_sink_ref().load(std::memory_order_relaxed);
    sink(lv, base, line, buf);
}

}  // namespace sgn

// ============================================================================
// 对外宏
// ============================================================================
#define SGN_LOG(lv, ...) \
    ::sgn::sgn_log_impl((lv), __FILE__, __LINE__, __VA_ARGS__)

#define SGN_LOG_DEBUG(...) SGN_LOG(::sgn::SGN_DEBUG, __VA_ARGS__)
#define SGN_LOG_INFO(...)  SGN_LOG(::sgn::SGN_INFO, __VA_ARGS__)
#define SGN_LOG_WARN(...)  SGN_LOG(::sgn::SGN_WARN, __VA_ARGS__)
#define SGN_LOG_ERROR(...) SGN_LOG(::sgn::SGN_ERROR, __VA_ARGS__)

