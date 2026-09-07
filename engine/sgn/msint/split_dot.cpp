// split_dot.cpp - MSInt 前向多精度拆分点积 C++ 实现
//
// 参见 split_dot.h 的数学依据说明。
// 本文件仅保留拆分+点积的纯计算逻辑（标量 int64 运算 + 手动 128 位累加，
// 无 __int128 依赖，保证可移植性）；窄精度 SIMD 内核已迁出到 simd/ 原语层
// （simd_api.h 接口，见 SIMD 原语域拆分设计（内部） Step 1）。
#include "split_dot.h"

#include "mkern/simd/simd_api.h"

#include <cstring>
#include <stdexcept>
#include <string>

namespace sgn {

// 把 int64 值拆到调用方提供的固定缓冲（避免逐元素分配 vector）。
// 要求 n = total_bits / split_bits；out 容量 >= n。语义与 SplitDot::split_parts 一致：
// 低位 part_0..part_{n-2} 无符号，最高位 part_{n-1} 符号位扩展。
void split_parts_fixed(int64_t value, int total_bits, int split_bits, int n,
                       int64_t* out) {
    // M2 加固 2026-09-07：入口按头文件契约（n = total_bits / split_bits、
    // total_bits <= 64）显式校验——否则 (n-1)*split_bits 可能 >= 64，右移 UB
    if (split_bits < 1 || split_bits > 64 || n < 1 || total_bits < 1 ||
        total_bits > 64 || total_bits % split_bits != 0 ||
        n != total_bits / split_bits) {
        throw std::invalid_argument(
            "split_parts_fixed 契约违约：要求 n == total_bits/split_bits 且 "
            "1 <= split_bits, total_bits <= 64");
    }
    uint64_t mask = (split_bits >= 64) ? ~0ULL : ((1ULL << split_bits) - 1);
    for (int k = 0; k < n - 1; ++k) {
        out[k] = static_cast<int64_t>(
            (static_cast<uint64_t>(value) >> (k * split_bits)) & mask);
    }
    out[n - 1] = value >> ((n - 1) * split_bits);
}

// ============================================================
// 窄精度打包数据 + 直写接口（方案 A split_parts 直写窄数组）
// ============================================================

// split_bits 是否走窄 SIMD 直写路径（16→simd::dot16，8/4→simd::dot8/dot4）。
// 2026-08-31（阶段 2，见 全局AVX编译参数移除调查_2026_08_31.md）：不再用编译期宏
// （__AVX2__/__AVXVNNI__）判定——那会在全局 -mavxvnni 编译下误判有 VNNI 的 CPU，
// 而实际 CPU 可能只有 AVX2（无 VNNI 的机器执行 dpbusd 会 illegal instruction）。
// 改为运行时查询 simd 后端名（由 simd_dispatch 的 CPUID 检测决定，与阶段 1 同纪律）。
// 语义：backend 为标量（无任何 SIMD）→ false（窄路径无收益，落标量三重循环）；
//     dot16 需要 AVX2/AVX-512（avx2 及以上后端都提供）；
//     dot8/dot4 需要 AVX-VNNI 或 AVX512-VNNI（仅 vnni 后端提供）。
bool is_narrow_simd(int split_bits) {
    const char* backend = simd::active_backend_name();
    const bool is_scalar = (std::strcmp(backend, "scalar") == 0) ||
                           (std::strcmp(backend, "scalar(forced)") == 0);
    if (is_scalar) return false;
    if (split_bits == 16) return true;  // avx2 / avxvnni / avx512f / avx512vnni 均提供 dot16
    if (split_bits == 8 || split_bits == 4) {
        return std::strcmp(backend, "avxvnni") == 0 ||
               std::strcmp(backend, "avx512vnni") == 0;
    }
    return false;
}

NarrowParts alloc_narrow_parts(int split_bits, size_t n, size_t K) {
    NarrowParts r;
    r.split_bits = split_bits;
    r.n = n;
    r.K = K;
    if (split_bits == 16) {
        r.s16.resize(n);
        for (auto& v : r.s16) v.resize(K);
    } else if (split_bits == 8) {
        r.su8.resize(n);
        r.ss8.resize(n);
        for (auto& v : r.su8) v.resize(K);
        for (auto& v : r.ss8) v.resize(K);
    } else if (split_bits == 4) {
        const size_t nb = (K + 1) / 2;  // 每字节 2 个 nibble
        r.su4.resize(n);
        r.ss4.resize(n);
        for (auto& v : r.su4) v.assign(nb, 0);
        for (auto& v : r.ss4) v.assign(nb, 0);
    }
    r.Sw.assign(n, 0);
    r.Sx.assign(n, 0);
    return r;
}

void pack_narrow_value(NarrowParts& dst, size_t a, size_t i, int64_t value) {
    const bool is_low = (a + 1 < dst.n);  // 无符号低位需偏置修正
    const int sb = dst.split_bits;
    if (sb == 16) {
        const int64_t bias = 32768;  // 2^15
        int16_t s = static_cast<int16_t>(is_low ? (value - bias) : value);
        dst.s16[a][i] = s;
        dst.Sw[a] += s;
        dst.Sx[a] += s;
    } else if (sb == 8) {
        const int64_t bias = 128;  // 2^7
        int8_t s = static_cast<int8_t>(is_low ? (value - bias) : value);
        dst.ss8[a][i] = s;
        dst.su8[a][i] = static_cast<uint8_t>(s + bias);
        dst.Sw[a] += s;
        dst.Sx[a] += s;
    } else if (sb == 4) {
        const int64_t bias = 8;  // 2^3
        int8_t s = static_cast<int8_t>(is_low ? (value - bias) : value);
        dst.Sw[a] += s;
        dst.Sx[a] += s;
        const size_t j = i >> 1;
        const int sh = 4 * (static_cast<int>(i) & 1);
        dst.su4[a][j] |= static_cast<uint8_t>((s + bias) << sh);
        dst.ss4[a][j] |= static_cast<uint8_t>((static_cast<uint8_t>(s) & 0x0F) << sh);
    }
    // 其他 split_bits：窄缓冲为空（不进入窄路径）
}

NarrowParts pack_narrow(int split_bits,
                        const std::vector<std::vector<int64_t>>& parts_w) {
    const size_t n = parts_w.size();
    const size_t K = n ? parts_w[0].size() : 0;
    NarrowParts r = alloc_narrow_parts(split_bits, n, K);
    for (size_t a = 0; a < n; ++a) {
        for (size_t i = 0; i < K; ++i) {
            pack_narrow_value(r, a, i, parts_w[a][i]);
        }
    }
    return r;
}

NarrowParts pack_narrow_element_major(
    int split_bits, const std::vector<std::vector<int64_t>>& parts_em) {
    const size_t group_size = parts_em.size();
    if (group_size == 0) {
        return alloc_narrow_parts(split_bits, 0, 0);
    }
    const size_t n = parts_em[0].size();
    NarrowParts r = alloc_narrow_parts(split_bits, n, group_size);
    for (size_t i = 0; i < group_size; ++i) {
        for (size_t a = 0; a < n; ++a) {
            pack_narrow_value(r, a, i, parts_em[i][a]);
        }
    }
    return r;
}

// ============================================================
// narrow_dot — 对两组窄打包做 n² 点积（偏置法修正，与标量 bit-exact 一致）
// ============================================================

std::vector<int64_t> narrow_dot(const NarrowParts& w, const NarrowParts& x,
                                bool trim_high_diag) {
    const size_t n = w.n;
    const size_t K = w.K;
    if (n == 0 || K == 0) {
        return {};
    }
    const int split_bits = w.split_bits;
    std::vector<int64_t> partials(2 * n - 1, 0);
    // 裁剪条件：trim_high_diag 时只保留低位对角（m=a+c<n）。
    // 高位对角（m>=n）是 2^total_bits 的倍数，对低位截断结果无贡献，可安全跳过。
    const auto keep_pair = [&](size_t a, size_t c) {
        return !trim_high_diag || (a + c < n);
    };

// 2026-08-31（阶段 2）：去掉 `#if defined(__AVX2__)` / `#if defined(__AVXVNNI__)`
    // 编译期短路——simd::dot16/dot8/dot4 是运行时调度（函数指针表），无条件调用即可，
    // 由 simd_dispatch 的 CPUID 决定实际后端（标量锚点常驻）。窄/宽路径门控由调用方
    // is_narrow_simd() 运行时判定，此处不再假设编译期指令集宏。
    if (split_bits == 16) {
        const int64_t bias = 32768;         // 2^15
        const int64_t bias2 = bias * bias;  // 2^30
        // 偏置修正项提升到 (a,c) 热循环之外：按对角线预计算 corr[m]（m = a+c）。
        // 热循环只剩纯 SIMD 点积 + 累加（无 la/lc 分支、无标量修正）。
        // 整数加法可交换/结合，与逐对修正 bit-exact 一致。
        std::vector<int64_t> corr(2 * n - 1, 0);
        for (size_t a = 0; a < n; ++a) {
            const bool la = (a + 1 < n);
            for (size_t c = 0; c < n; ++c) {
                if (!keep_pair(a, c)) continue;
                const bool lc = (c + 1 < n);
                corr[a + c] += (la ? bias * x.Sx[c] : 0) + (lc ? bias * w.Sw[a] : 0)
                             + ((la && lc) ? static_cast<int64_t>(K) * bias2 : 0);
            }
        }
        for (size_t a = 0; a < n; ++a) {
            for (size_t c = 0; c < n; ++c) {
                if (!keep_pair(a, c)) continue;
                partials[a + c] += simd::dot16(w.s16[a].data(), x.s16[c].data(), K);
            }
        }
        for (size_t m = 0; m < partials.size(); ++m) {
            if (!trim_high_diag || m < n) partials[m] += corr[m];
        }
        return partials;
    }
    if (split_bits == 8) {
        const int64_t bias = 128;           // 2^7
        const int64_t bias2 = bias * bias;  // 2^14 = 16384
        // 8 位存储 su8 = ss8 + bias，raw_dot 已含 +bias*Sx[c]；
        // 修正 corr(a,c) = bias*(la-1)*Sx[c] + bias*lc*Sw[a] + K*bias²*la*lc（见上方推导）。
        std::vector<int64_t> corr(2 * n - 1, 0);
        for (size_t a = 0; a < n; ++a) {
            const bool la = (a + 1 < n);
            for (size_t c = 0; c < n; ++c) {
                if (!keep_pair(a, c)) continue;
                const bool lc = (c + 1 < n);
                corr[a + c] += bias * (la ? 0 : -1) * x.Sx[c]
                             + (lc ? bias * w.Sw[a] : 0)
                             + ((la && lc) ? static_cast<int64_t>(K) * bias2 : 0);
            }
        }
        for (size_t a = 0; a < n; ++a) {
            for (size_t c = 0; c < n; ++c) {
                if (!keep_pair(a, c)) continue;
                partials[a + c] += simd::dot8(w.su8[a].data(), x.ss8[c].data(), K);
            }
        }
        for (size_t m = 0; m < partials.size(); ++m) {
            if (!trim_high_diag || m < n) partials[m] += corr[m];
        }
        return partials;
    }
    if (split_bits == 4) {
        const int64_t bias = 8;            // 2^3
        const int64_t bias2 = bias * bias; // 2^6 = 64
        // 预解包：w 无符号字节 + x 有符号字节各一次，消除 n²=64 次重复 nibble 解包
        // （w 侧只需无符号、x 侧只需有符号；结果与逐对 dot4_nibble 解包 bit-exact 一致）
        std::vector<std::vector<uint8_t>> wu(n);
        std::vector<std::vector<int8_t>> xs(n);
        for (size_t a = 0; a < n; ++a) {
            wu[a].resize(K);
            xs[a].resize(K);
            simd::unpack_nibble_u(w.su4[a].data(), K, wu[a].data());
            simd::unpack_nibble_s(x.ss4[a].data(), K, xs[a].data());
        }
        std::vector<int64_t> corr(2 * n - 1, 0);
        for (size_t a = 0; a < n; ++a) {
            const bool la = (a + 1 < n);
            for (size_t c = 0; c < n; ++c) {
                if (!keep_pair(a, c)) continue;
                const bool lc = (c + 1 < n);
                corr[a + c] += bias * (la ? 0 : -1) * x.Sx[c]
                             + (lc ? bias * w.Sw[a] : 0)
                             + ((la && lc) ? static_cast<int64_t>(K) * bias2 : 0);
            }
        }
        for (size_t a = 0; a < n; ++a) {
            for (size_t c = 0; c < n; ++c) {
                if (!keep_pair(a, c)) continue;
                partials[a + c] += simd::dot4(wu[a].data(), xs[c].data(), K);
            }
        }
        for (size_t m = 0; m < partials.size(); ++m) {
            if (!trim_high_diag || m < n) partials[m] += corr[m];
        }
        return partials;
    }

    // 非窄 split_bits（不应经此路径；调用方已按 is_narrow_simd 回退标量）
    return partials;
}

// ============================================================
// 4 位摊销接口 — prepare_nibble + narrow_dot_prepared4
// ============================================================
// 解决 M 输出摊销场景（H3：同一激活 x 复用于 M 个输出行）下，narrow_dot 每次调用
// 都把 x 的 nibble 重新解包一次（M 次重复解包）的低效：把「预解包」提升为可复用的
// 一等操作——权重侧离线 prepare_nibble 一次、激活侧每前向 prepare_nibble 一次，
// 热路径 M 个输出全部复用同一预解包结果，n² 点积走纯 dpbusd（无解包指令）。
// 与 narrow_dot 内部 4 位预解包 bit-exact 一致（复用同一 unpack_nibble_u/s 语义）。

NibblePrepared prepare_nibble(const NarrowParts& p) {
    NibblePrepared r;
    r.n = static_cast<int>(p.n);
    r.K = p.K;
    r.u8.resize(p.n);
    r.s8.resize(p.n);
    for (size_t a = 0; a < p.n; ++a) {
        r.u8[a].resize(p.K);
        r.s8[a].resize(p.K);
        simd::unpack_nibble_u(p.su4[a].data(), p.K, r.u8[a].data());
        simd::unpack_nibble_s(p.ss4[a].data(), p.K, r.s8[a].data());
    }
    r.Sw = p.Sw;
    r.Sx = p.Sx;
    return r;
}

std::vector<int64_t> narrow_dot_prepared4(const NibblePrepared& w, const NibblePrepared& x,
                                          bool trim_high_diag) {
    const int n = w.n;
    const size_t K = w.K;
    const int64_t bias = 8;            // 2^3
    const int64_t bias2 = bias * bias; // 2^6 = 64
    // 偏置修正项提升到 (a,c) 热循环外：按对角线预计算 corr[m]。
    // 8 位推导同样适用（su8 = ss8 + bias 语义相同），热循环只剩纯 dpbusd。
    std::vector<int64_t> corr(static_cast<size_t>(2 * n - 1), 0);
    for (int a = 0; a < n; ++a) {
        const bool la = (a + 1 < n);
        for (int c = 0; c < n; ++c) {
            if (trim_high_diag && (a + c >= n)) continue;  // 高位对角裁剪
            const bool lc = (c + 1 < n);
            corr[static_cast<size_t>(a + c)] += bias * (la ? 0 : -1) * x.Sx[c]
                                               + (lc ? bias * w.Sw[a] : 0)
                                               + ((la && lc) ? static_cast<int64_t>(K) * bias2 : 0);
        }
    }
    std::vector<int64_t> partials(static_cast<size_t>(2 * n - 1), 0);
    for (int a = 0; a < n; ++a) {
        for (int c = 0; c < n; ++c) {
            if (trim_high_diag && (a + c >= n)) continue;  // 高位对角裁剪
            // 回退由 simd::dot4 内部管理：AVX-VNNI 走 vpdpbusd，非 x86 走标量锚点
            partials[static_cast<size_t>(a + c)] += simd::dot4(w.u8[a].data(), x.s8[c].data(), K);
        }
    }
    for (size_t m = 0; m < partials.size(); ++m) {
        if (!trim_high_diag || m < static_cast<size_t>(n)) partials[m] += corr[m];
    }
    return partials;
}

NibblePrepared prepare_nibble_from_raw(const std::vector<int64_t>& w, int total_bits) {
    const int sb = 4;
    if (total_bits <= 0 || total_bits % sb != 0) {
        throw std::invalid_argument("prepare_nibble_from_raw: total_bits(" +
                                    std::to_string(total_bits) + ") 必须 >0 且能被 4 整除");
    }
    const int n = total_bits / sb;
    const size_t K = w.size();
    if (n > 32) {
        throw std::invalid_argument("prepare_nibble_from_raw: 暂不支持拆分数 n>32");
    }
    // 连续预分配窄缓冲 + 栈缓冲拆分直写：避免按元素逐次 split_parts 的 vector
    // 堆分配（K=16384 时该分配是 prepare 主导成本，与 leveled 分组阶段同型问题）。
    NarrowParts pw = alloc_narrow_parts(sb, static_cast<size_t>(n), K);
    int64_t wbuf[32];
    for (size_t i = 0; i < K; ++i) {
        split_parts_fixed(w[i], total_bits, sb, n, wbuf);
        for (int a = 0; a < n; ++a) {
            pack_narrow_value(pw, static_cast<size_t>(a), i, wbuf[a]);
        }
    }
    return prepare_nibble(pw);
}

std::vector<std::vector<int64_t>> matmul_prepared4(
    const std::vector<NibblePrepared>& W,
    const NibblePrepared& x,
    bool trim_high_diag) {
    std::vector<std::vector<int64_t>> out;
    out.reserve(W.size());
    for (const auto& pw : W) {
        out.push_back(narrow_dot_prepared4(pw, x, trim_high_diag));
    }
    return out;
}

// ============================================================
// narrow_group_dot — 窄精度多输出点积（方案 A 三档窄路径 + 标量回退）
// ============================================================
// 部分主序输入：parts_w[a][i] = 第 a 个部分在第 i 个元素上的值；
// parts_w / parts_x 各含 n 个部分（n = parts_w.size()），每个部分长度 = group_size。
// 返回 2n-1 个 partial[m] = Σ_{a+c=m} Σ_i parts_w[a][i]*parts_x[c][i]。
//   split_bits=16 → simd::dot16（avx2/avx512 后端）；8/4 → simd::dot8 / simd::dot4 预解包
//   （仅 avxvnni/avx512vnni 后端提供）；其余 split_bits 或标量后端 → 标量三重循环。
//   窄/宽门控由 is_narrow_simd() 运行时判定（后台 = simd_dispatch CPUID，非编译期宏）。
// 窄路径经偏置法修正，与标量逐元素 bit-exact 一致。
// 供 SplitDot::dot_split 与 LeveledSplitDot 组内点积共享。
std::vector<int64_t> narrow_group_dot(
    int split_bits,
    const std::vector<std::vector<int64_t>>& parts_w,
    const std::vector<std::vector<int64_t>>& parts_x,
    bool trim_high_diag) {
    const size_t n = parts_w.size();
    const size_t K = n ? parts_w[0].size() : 0;
    if (n == 0 || K == 0) {
        return {};
    }

    // 窄路径（16/8/4 位）：整体打包为窄表示（含 Sw/Sx 修正和）再 n² 点积。
    // 与标量逐元素 bit-exact 一致（整数加法可交换/结合 + 偏置修正项为纯整数加法）。
    if (is_narrow_simd(split_bits)) {
        NarrowParts pw = pack_narrow(split_bits, parts_w);
        NarrowParts px = pack_narrow(split_bits, parts_x);
        return narrow_dot(pw, px, trim_high_diag);
    }

    // 标量回退（split_bits=32 等及其他/非 x86 平台）
    std::vector<int64_t> partials(2 * n - 1, 0);
    for (size_t a = 0; a < n; ++a) {
        for (size_t c = 0; c < n; ++c) {
            if (trim_high_diag && (a + c >= n)) continue;  // 高位对角裁剪
            int64_t sum = 0;
            for (size_t i = 0; i < K; ++i) {
                sum += parts_w[a][i] * parts_x[c][i];
            }
            partials[a + c] += sum;
        }
    }
    return partials;
}

// ============================================================
// split_parts — 把 int64 按 split_bits 拆成 n 部分
// ============================================================

std::vector<int64_t> SplitDot::split_parts(int64_t value, int total_bits, int split_bits) {
    if (total_bits <= 0) {
        throw std::invalid_argument("total_bits 必须 > 0");
    }
    if (split_bits <= 0) {
        throw std::invalid_argument("split_bits 必须 > 0");
    }
    if (total_bits % split_bits != 0) {
        throw std::invalid_argument("total_bits(" + std::to_string(total_bits) +
                                    ") 必须能被 split_bits(" +
                                    std::to_string(split_bits) + ") 整除");
    }
    int n = total_bits / split_bits;
    if (n <= 0) {
        throw std::invalid_argument("拆分数必须 >= 1");
    }
    if (total_bits > 64) {
        throw std::invalid_argument("total_bits(" + std::to_string(total_bits) +
                                    ") 超过 64");
    }

    uint64_t mask = (split_bits >= 64) ? ~0ULL : ((1ULL << split_bits) - 1);
    std::vector<int64_t> parts(static_cast<size_t>(n));

    // 低位 part_0..part_{n-2}：无符号提取
    for (int k = 0; k < n - 1; ++k) {
        parts[static_cast<size_t>(k)] =
            static_cast<int64_t>((static_cast<uint64_t>(value) >> (k * split_bits)) & mask);
    }
    // 最高位 part_{n-1}：符号位扩展（算术右移），保证可逆
    parts[static_cast<size_t>(n - 1)] = value >> ((n - 1) * split_bits);
    return parts;
}

// ============================================================
// dot_split — 多输出模式
// ============================================================

std::vector<int64_t> SplitDot::dot_split(
    const std::vector<int64_t>& w,
    const std::vector<int64_t>& x,
    int total_bits, int split_bits,
    bool trim_high_diag) {
    if (w.size() != x.size()) {
        throw std::invalid_argument("w 长度(" + std::to_string(w.size()) +
                                    ") != x 长度(" + std::to_string(x.size()) + ")");
    }
    if (w.empty()) {
        return {};
    }
    int n = total_bits / split_bits;
    if (total_bits % split_bits != 0) {
        throw std::invalid_argument("total_bits 必须能被 split_bits 整除");
    }
    if (n <= 0) {
        throw std::invalid_argument("拆分数必须 >= 1");
    }
    size_t K = w.size();

    // 窄路径（16/8/4 位，x86 AVX2/VNNI）：拆分阶段直写窄缓冲（部分主序），
    // 省去 int64 中间数组与 int64→窄二次转换。split_parts_fixed 用栈缓冲，
    // 避免逐元素分配 vector（K 个元素 × 2 次分配的固定开销）。
    if (is_narrow_simd(split_bits)) {
        if (n > 32) {
            throw std::invalid_argument("窄路径暂不支持拆分数 n>32");
        }
        NarrowParts pw = alloc_narrow_parts(split_bits, static_cast<size_t>(n), K);
        NarrowParts px = alloc_narrow_parts(split_bits, static_cast<size_t>(n), K);
        int64_t wbuf[32], xbuf[32];
        for (size_t i = 0; i < K; ++i) {
            split_parts_fixed(w[i], total_bits, split_bits, n, wbuf);
            split_parts_fixed(x[i], total_bits, split_bits, n, xbuf);
            for (int a = 0; a < n; ++a) {
                pack_narrow_value(pw, static_cast<size_t>(a), i, wbuf[a]);
                pack_narrow_value(px, static_cast<size_t>(a), i, xbuf[a]);
            }
        }
        return narrow_dot(pw, px, trim_high_diag);
    }

    // 标量路径（split_bits 非窄或非 x86 平台）：逐元素拆分（保证可移植），
    // 先整体拆分 w、x 的每一部分（n 个 vector，各长 K），统一由 narrow_group_dot
    // 的标量回退处理。pw/px 已是部分主序（pw[k][i] = 第 k 个部分在第 i 个元素上的值）。
    std::vector<std::vector<int64_t>> pw(n), px(n);
    for (int k = 0; k < n; ++k) {
        pw[static_cast<size_t>(k)].resize(K);
        px[static_cast<size_t>(k)].resize(K);
    }
    for (size_t i = 0; i < K; ++i) {
        auto wparts = split_parts(w[i], total_bits, split_bits);
        auto xparts = split_parts(x[i], total_bits, split_bits);
        for (int k = 0; k < n; ++k) {
            pw[static_cast<size_t>(k)][i] = wparts[static_cast<size_t>(k)];
            px[static_cast<size_t>(k)][i] = xparts[static_cast<size_t>(k)];
        }
    }
    return narrow_group_dot(split_bits, pw, px, trim_high_diag);
}

// ============================================================
// dot_fused — 单融合模式（int64 截断）
// ============================================================

int64_t SplitDot::dot_fused(
    const std::vector<int64_t>& w,
    const std::vector<int64_t>& x,
    int total_bits, int split_bits) {
    auto partials = dot_split(w, x, total_bits, split_bits);
    auto [hi, lo] = fuse_128(partials, split_bits);
    (void)hi;  // 单融合返回低 64 位
    return static_cast<int64_t>(lo);
}

// ============================================================
// dot_fused_i32 — 单融合 int32 截断模式（低位对角裁剪）
// ============================================================

int32_t SplitDot::dot_fused_i32(
    const std::vector<int64_t>& w,
    const std::vector<int64_t>& x,
    int split_bits) {
    const int total_bits = 32;
    if (total_bits % split_bits != 0) {
        throw std::invalid_argument("32 必须能被 split_bits(" + std::to_string(split_bits) +
                                    ") 整除");
    }
    const int n = total_bits / split_bits;
    if (n <= 0) {
        throw std::invalid_argument("拆分数必须 >= 1");
    }
    // 走裁剪内核（trim_high_diag=true）：高位对角（m>=n）是 2^32 的倍数，
    // 对低 32 位无贡献，裁剪后 partials[m>=n] 保持 0，省 44% ALU（4 位档 n=8）。
    auto partials = dot_split(w, x, total_bits, split_bits, /*trim_high_diag=*/true);
    // 低 32 位融合：result = Σ_{m<n} partials[m] * 2^(m*split_bits)（模 2^32）。
    // 逐项取低 32 位再求和，与全融合后 int32 截断 bit-exact 一致
    //（裁剪掉的 m>=n 项均为 2^32 的倍数，对低 32 位贡献为 0）。
    uint32_t acc = 0;
    for (int m = 0; m < n; ++m) {
        const int shift = m * split_bits;
        acc += static_cast<uint32_t>(static_cast<uint64_t>(partials[static_cast<size_t>(m)]) << shift);
    }
    return static_cast<int32_t>(acc);
}

// ============================================================
// fuse_128 — 128 位精确融合
// ============================================================

std::pair<int64_t, uint64_t> SplitDot::fuse_128(
    const std::vector<int64_t>& partials, int split_bits) {
    // fused = Σ_m partial[m] * 2^(m*split_bits)，128 位有符号结果 (hi:lo)。

#if defined(__SIZEOF_INT128__)
    // 主路径：__int128（Clang/GCC x86-64 支持），位拆分+位移保证 bit-exact
    __int128 acc = 0;
    for (size_t m = 0; m < partials.size(); ++m) {
        int shift = static_cast<int>(m) * split_bits;
        acc += static_cast<__int128>(partials[m]) << shift;
    }
    return {static_cast<int64_t>(acc >> 64), static_cast<uint64_t>(acc)};
#else
    // 回退路径：手动 128 位有符号累加（无 __int128 平台，如部分非 x86）
    // term = p * 2^shift，表示为 (term_hi:term_lo)。
    // 把 p 视为 128 位有符号 (hi_p = p>>63, lo_p = (uint64)p)，左移 shift 位。
    uint64_t acc_lo = 0, acc_hi = 0;
    for (size_t m = 0; m < partials.size(); ++m) {
        int shift = static_cast<int>(m) * split_bits;
        int64_t p = partials[m];

        uint64_t term_lo = 0, term_hi = 0;
        if (shift < 64) {
            uint64_t lo_p = static_cast<uint64_t>(p);
            term_lo = lo_p << shift;
            if (p < 0) {
                // 负值：高字符号扩展为全 1
                term_hi = ~0ULL;
            } else {
                term_hi = (shift == 0) ? 0 : (lo_p >> (64 - shift));
            }
        } else if (shift == 64) {
            term_lo = 0;
            term_hi = static_cast<uint64_t>(p);
        } else {
            // shift > 64：仅当高位不溢出 64 位时精确（超出部分截断，文档注明）
            term_lo = 0;
            term_hi = static_cast<uint64_t>(p) << (shift - 64);
        }

        // acc += term（带进位）
        uint64_t sum_lo = acc_lo + term_lo;
        uint64_t carry = (sum_lo < acc_lo) ? 1 : 0;
        acc_hi += term_hi + carry;
        acc_lo = sum_lo;
    }
    return {static_cast<int64_t>(acc_hi), acc_lo};
#endif
}

} // namespace sgn
