// leveled_split_dot.cpp - MSInt 逐元素异构粒度拆分点积（组合落地）
//
// 参见 leveled_split_dot.h 的数学依据说明。
// 本实现仅使用标量 int64 运算 + 手动 128 位累加（复用 SplitDot::fuse_128，
// 含非 __int128 回退路径），保证可移植性，符合项目对非 x86 平台回退路径的约定。
#include "leveled_split_dot.h"

#include <stdexcept>
#include <string>

namespace sgn {

namespace {

std::vector<int> default_options() { return {16, 8, 4}; }
std::vector<int64_t> default_thresholds() { return {100, 1000}; }
// 降档默认：精度位数从低精度到高精度排列（低重要度 → 少位数），阈值同粒度默认。
std::vector<int> default_precision_options() { return {8, 16, 32}; }
std::vector<int64_t> default_precision_thresholds() { return {100, 1000}; }

// 128 位有符号累加器（跨精度组融合用），低 64 位无符号、高 64 位有符号。
struct Int128Acc {
    int64_t hi = 0;
    uint64_t lo = 0;

    void add(int64_t h, uint64_t l) {
        uint64_t sum_lo = lo + l;
        uint64_t carry = (sum_lo < lo) ? 1 : 0;
        hi += h + static_cast<int64_t>(carry);
        lo = sum_lo;
    }

    int64_t low64() const { return static_cast<int64_t>(lo); }
};

// 128 位左移 sh 位（复用 Int128Acc 的 (hi, lo) 表示，sh < 128）。
// 用于降档恢复量纲：组内 p-bit 融合结果左移 2*(total_bits-p)。
std::pair<int64_t, uint64_t> shl128(int64_t hi, uint64_t lo, int sh) {
    if (sh == 0) return {hi, lo};
    if (sh < 0 || sh >= 128) return {0, 0};  // 负移位 UB 防御（审计 M1-2）
    uint64_t n_lo = (sh < 64) ? (lo << sh) : 0;
    int64_t n_hi;
    if (sh < 64) {
        n_hi = static_cast<int64_t>((static_cast<uint64_t>(hi) << sh) |
                                    (lo >> (64 - sh)));
    } else if (sh == 64) {
        n_hi = static_cast<int64_t>(lo);
    } else {
        n_hi = static_cast<int64_t>(lo << (sh - 64));
    }
    return {n_hi, n_lo};
}

// 按精度位数分组：返回 {p: (w 组, x 组)}（std::map 升序遍历，保组内原始顺序）。
std::map<int, std::pair<std::vector<int64_t>, std::vector<int64_t>>>
group_by_precision(const std::vector<int64_t>& w,
                   const std::vector<int64_t>& x,
                   const std::vector<int>& levels) {
    std::map<int, std::pair<std::vector<int64_t>, std::vector<int64_t>>> groups;
    for (size_t i = 0; i < w.size(); ++i) {
        auto& g = groups[static_cast<int>(levels[i])];
        g.first.push_back(w[i]);
        g.second.push_back(x[i]);
    }
    return groups;
}

} // namespace

// ============================================================
// select_levels — 按重要性为每个元素选择拆分粒度
// ============================================================

std::vector<int> LeveledSplitDot::select_levels(
    const std::vector<int64_t>& importance,
    int total_bits,
    const std::vector<int>& options,
    const std::vector<int64_t>& thresholds) {
    if (options.empty()) {
        throw std::invalid_argument("options 不能为空");
    }
    if (thresholds.size() != options.size() - 1) {
        throw std::invalid_argument("thresholds 长度必须等于 options 长度-1");
    }
    // options 值域校验：count_of_b/slot_of_b 以 total_bits+1 为大小、按 b 直接
    // 下标（见 dot_split_leveled），b 超界即 OOB 写（安全审计 2026-08-16 M1-1）
    if (total_bits <= 0) {
        throw std::invalid_argument("total_bits 必须为正");
    }
    for (int b : options) {
        if (b <= 0 || b > total_bits) {
            throw std::invalid_argument(
                "options 中的精度位数必须在 (0, total_bits] 区间");
        }
        if (total_bits % b != 0) {
            throw std::invalid_argument(
                "options 中的精度位数必须整除 total_bits");
        }
    }

    std::vector<int> levels;
    levels.reserve(importance.size());
    for (int64_t imp : importance) {
        size_t idx = 0;
        for (size_t k = 0; k < thresholds.size(); ++k) {
            if (imp >= thresholds[k]) {
                idx = k + 1;
            } else {
                break;
            }
        }
        levels.push_back(options[idx]);
    }
    return levels;
}

// ============================================================
// dot_split_leveled — 异构粒度多输出拆分点积（按粒度分组）
// ============================================================

std::map<int, std::vector<int64_t>> LeveledSplitDot::dot_split_leveled(
    const std::vector<int64_t>& w, const std::vector<int64_t>& x,
    const std::vector<int64_t>& importance,
    int total_bits,
    const std::vector<int>& options,
    const std::vector<int64_t>& thresholds,
    bool trim_high_diag) {
    if (w.size() != x.size() || w.size() != importance.size()) {
        throw std::invalid_argument("w/x/importance 长度不一致");
    }

    auto levels = select_levels(importance, total_bits, options, thresholds);

    // 分组阶段优化（第 2 轮，见 dot_split_leveled_grouping_heap_overhead_2026_08_14.md §5）：
    // 第 1 轮已改为「预统计容量 + alloc_narrow_parts 连续预分配 + 栈缓冲拆分直写」，
    // 归因对照显示剩余 ~3500us 来自热循环每元素 gw_narrow[b]/gx_narrow[b]/cursor[b]
    // 的 std::map 红黑树查找（K=65536 时每元素 2n+1 ≈ 17 次 × 红黑树指针追逐）。本轮：
    //   1) count_of_b 数组（b 直接下标）替代 group_count 的 std::map 逐元素 ++；
    //   2) slot_of_b 数组（b 直接下标，size=total_bits+1）把组定位降为 O(1) 数组访问；
    //   3) 窄组数据存 vector<NarrowParts>（按槽位连续），热循环提前取出引用复用；
    //   4) cursor 改 per-slot size_t 数组，逐元素 ++ 无 map 查找。
    // b 是 total_bits 的因子（b<=total_bits），两类数组下标均有界；仅窄组建槽位，
    // 标量组仍走 gw_parts/gx_parts（数量极少，保 map 语义）。
    std::vector<size_t> count_of_b(static_cast<size_t>(total_bits) + 1, 0);
    for (int b : levels) {
        ++count_of_b[static_cast<size_t>(b)];
    }

    std::vector<int> slot_of_b(static_cast<size_t>(total_bits) + 1, -1);
    std::vector<int> narrow_bs;             // 槽位对应的 b（结果构造与槽位映射用）
    std::vector<NarrowParts> gw_narrow, gx_narrow;
    std::vector<size_t> cursor_arr;         // per-slot 组内游标
    std::map<int, std::vector<std::vector<int64_t>>> gw_parts, gx_parts;
    for (int b = 1; b <= total_bits; ++b) {
        size_t cnt = count_of_b[static_cast<size_t>(b)];
        if (cnt == 0) continue;
        if (is_narrow_simd(b)) {
            const int n = total_bits / b;
            slot_of_b[static_cast<size_t>(b)] = static_cast<int>(narrow_bs.size());
            narrow_bs.push_back(b);
            gw_narrow.emplace_back(alloc_narrow_parts(b, static_cast<size_t>(n), cnt));
            gx_narrow.emplace_back(alloc_narrow_parts(b, static_cast<size_t>(n), cnt));
            cursor_arr.push_back(0);
        } else {
            // 标量路径仍逐元素 split_parts，但预 reserve 避免 vector 反复 realloc
            gw_parts[b].reserve(cnt);
            gx_parts[b].reserve(cnt);
        }
    }

    // 逐元素拆分直写（栈缓冲 + 槽位下标 + per-slot 游标，无 map 查找）
    int64_t wbuf[32], xbuf[32];  // 窄路径 n<=8（4 位档），32 裕量足够
    for (size_t i = 0; i < w.size(); ++i) {
        int b = levels[i];
        if (is_narrow_simd(b)) {
            const int s = slot_of_b[static_cast<size_t>(b)];
            const int n = total_bits / b;
            size_t idx = cursor_arr[static_cast<size_t>(s)]++;
            NarrowParts& pw = gw_narrow[static_cast<size_t>(s)];
            NarrowParts& px = gx_narrow[static_cast<size_t>(s)];
            split_parts_fixed(w[i], total_bits, b, n, wbuf);
            split_parts_fixed(x[i], total_bits, b, n, xbuf);
            for (int a = 0; a < n; ++a) {
                pack_narrow_value(pw, static_cast<size_t>(a), idx, wbuf[a]);
                pack_narrow_value(px, static_cast<size_t>(a), idx, xbuf[a]);
            }
        } else {
            gw_parts[b].push_back(SplitDot::split_parts(w[i], total_bits, b));
            gx_parts[b].push_back(SplitDot::split_parts(x[i], total_bits, b));
        }
    }

    std::map<int, std::vector<int64_t>> result;
    // 窄路径：直接 narrow_dot（数据已部分主序直写窄缓冲）
    for (size_t s = 0; s < narrow_bs.size(); ++s) {
        result[narrow_bs[s]] = narrow_dot(gw_narrow[s], gx_narrow[s], trim_high_diag);
    }
    // 标量路径：元素主序 → 部分主序转置 + narrow_group_dot 标量回退
    // （split_bits 非窄或非 x86 平台）
    for (auto& [b, wparts] : gw_parts) {
        int n = total_bits / b;
        const auto& xparts = gx_parts[b];
        size_t group_size = wparts.size();
        std::vector<std::vector<int64_t>> parts_w(static_cast<size_t>(n)),
            parts_x(static_cast<size_t>(n));
        for (int a = 0; a < n; ++a) {
            parts_w[static_cast<size_t>(a)].resize(group_size);
            parts_x[static_cast<size_t>(a)].resize(group_size);
        }
        for (size_t i = 0; i < group_size; ++i) {
            for (int a = 0; a < n; ++a) {
                parts_w[static_cast<size_t>(a)][i] = wparts[i][static_cast<size_t>(a)];
                parts_x[static_cast<size_t>(a)][i] = xparts[i][static_cast<size_t>(a)];
            }
        }

        // partial[m] = Σ_{a+c=m} Σ_{i∈group} parts_w[a][i] * parts_x[c][i]
        result[b] = narrow_group_dot(b, parts_w, parts_x, trim_high_diag);
    }
    return result;
}

// ============================================================
// dot_fused_leveled — 异构粒度单融合点积（128 位精确累加）
// ============================================================

int64_t LeveledSplitDot::dot_fused_leveled(
    const std::vector<int64_t>& w, const std::vector<int64_t>& x,
    const std::vector<int64_t>& importance,
    int total_bits,
    const std::vector<int>& options,
    const std::vector<int64_t>& thresholds) {
    auto groups = dot_split_leveled(w, x, importance, total_bits, options, thresholds);

    Int128Acc acc;
    for (auto& [b, partials] : groups) {
        auto [hi, lo] = SplitDot::fuse_128(partials, b);
        acc.add(hi, lo);
    }
    return acc.low64();
}

// ============================================================
// 降档决策 + 降档摊销点积（Level 精度调度语义，见 leveled_split_dot.h）
// ============================================================

std::vector<int> LeveledSplitDot::select_precision(
    const std::vector<int64_t>& importance,
    const std::vector<int>& options,
    const std::vector<int64_t>& thresholds) {
    // 与 select_levels 同一阈值游走（options[0] 给最低重要度桶）；
    // total_bits 参数对阈值游走无影响，仅语义差异（精度位数 vs 拆分粒度）。
    return select_levels(importance, 32, options, thresholds);
}

std::map<int, std::vector<int64_t>> LeveledSplitDot::dot_split_leveled_downcast(
    const std::vector<int64_t>& w, const std::vector<int64_t>& x,
    const std::vector<int64_t>& importance,
    int total_bits,
    const std::vector<int>& options,
    const std::vector<int64_t>& thresholds) {
    if (w.size() != x.size() || w.size() != importance.size()) {
        throw std::invalid_argument("w/x/importance 长度不一致");
    }
    auto levels = select_precision(importance, options, thresholds);
    auto groups = group_by_precision(w, x, levels);

    std::map<int, std::vector<int64_t>> result;
    for (auto& [p, wg] : groups) {
        if (p <= 0 || p > total_bits || total_bits % p != 0) {
            throw std::invalid_argument(
                "dot_split_leveled_downcast: 精度位数必须 >0 且整除 total_bits");
        }
        auto& xg = wg.second;
        const int s = total_bits - p;  // 保留高 p 位 → 右移成 p-bit 值
        std::vector<int64_t> wp(wg.first.size()), xp(xg.size());
        for (size_t i = 0; i < wg.first.size(); ++i) {
            wp[i] = wg.first[i] >> s;  // 算术右移（keep_top 语义的 p-bit 值）
            xp[i] = xg[i] >> s;
        }
        // 4 位摊销路径：n = p/4 组合（低精度组平方级省算）
        NibblePrepared pw = prepare_nibble_from_raw(wp, p);
        NibblePrepared px = prepare_nibble_from_raw(xp, p);
        result[p] = narrow_dot_prepared4(pw, px, false);
    }
    return result;
}

int64_t LeveledSplitDot::dot_fused_leveled_downcast(
    const std::vector<int64_t>& w, const std::vector<int64_t>& x,
    const std::vector<int64_t>& importance,
    int total_bits,
    const std::vector<int>& options,
    const std::vector<int64_t>& thresholds) {
    auto groups = dot_split_leveled_downcast(w, x, importance, total_bits,
                                             options, thresholds);

    Int128Acc acc;
    for (auto& [p, partials] : groups) {
        auto [hi, lo] = SplitDot::fuse_128(partials, 4);  // 组内 p-bit 精确融合
        const int sh = 2 * (total_bits - p);              // 恢复量纲
        auto [n_hi, n_lo] = shl128(hi, lo, sh);
        acc.add(n_hi, n_lo);
    }
    return acc.low64();
}

std::map<int, NibblePrepared> LeveledSplitDot::prepare_downcast(
    const std::vector<int64_t>& w, const std::vector<int64_t>& importance,
    int total_bits,
    const std::vector<int>& options,
    const std::vector<int64_t>& thresholds) {
    if (w.size() != importance.size()) {
        throw std::invalid_argument("w/importance 长度不一致");
    }
    return prepare_downcast_ptr(w.data(), importance.data(), w.size(),
                                total_bits, options, thresholds);
}

std::map<int, NibblePrepared> LeveledSplitDot::prepare_downcast_ptr(
    const int64_t* w, const int64_t* importance, size_t K,
    int total_bits,
    const std::vector<int>& options,
    const std::vector<int64_t>& thresholds) {
    // select_precision 需要 vector<int64_t>；importance 在此暂以指针视图处理——
    // 阈值游走是逐元素读，直接用指针做，避免额外 vector 拷贝（importance 常为
    // 离线共享，但 numpy 零拷贝路径下仍应避免无谓转换）。
    // 先取精度分组（levels 逐元素游走）
    const size_t n_opt = thresholds.size() + 1;
    if (options.size() != n_opt) {
        throw std::invalid_argument("thresholds 长度必须等于 options 长度-1");
    }
    // options 值域校验：count_of_b/slot_of_b 以 total_bits+1 为大小、按 p 直接
    // 下标，p 超界即 OOB 写（安全审计 2026-08-16 M1-1，与 select_levels 对齐）
    if (total_bits <= 0) {
        throw std::invalid_argument("total_bits 必须为正");
    }
    for (int p : options) {
        if (p <= 0 || p > total_bits) {
            throw std::invalid_argument(
                "options 中的精度位数必须在 (0, total_bits] 区间");
        }
        if (total_bits % p != 0) {
            throw std::invalid_argument(
                "options 中的精度位数必须整除 total_bits");
        }
    }
    std::vector<int> levels(K);
    for (size_t i = 0; i < K; ++i) {
        const int64_t imp = importance[i];
        size_t idx = 0;
        for (size_t k = 0; k < thresholds.size(); ++k) {
            if (imp >= thresholds[k]) {
                idx = k + 1;
            } else {
                break;
            }
        }
        levels[i] = options[idx];
    }

    // 阶段 3：count_of_b + slot_of_b 数组分组（替代 std::map 红黑树，复用
    // dot_split_leveled_grouping_heap_overhead_2026_08_14.md §5 经验）。
    // std::map 逐元素插入是 K 次红黑树查找/指针追逐（leveled 分组阶段已证实占
    // ~94%）；这里预统计各 p 组容量 → 连续 vector reserve → 槽位 O(1) 填值。
    // 填值时直接存 keep_top 移位后的 p-bit 值（w >> (total_bits-p)），
    // 省掉原先 wp 中间数组的第二次遍历。p 是 total_bits 的因子，下标有界。
    std::vector<size_t> count_of_b(static_cast<size_t>(total_bits) + 1, 0);
    for (size_t i = 0; i < K; ++i) {
        ++count_of_b[static_cast<size_t>(levels[i])];
    }
    std::vector<int> slot_of_b(static_cast<size_t>(total_bits) + 1, -1);
    std::vector<int> bs;                          // 槽位对应的 p
    std::vector<std::vector<int64_t>> groups;     // 按槽位（已存 p-bit 移位值）
    for (int p = 1; p <= total_bits; ++p) {
        size_t cnt = count_of_b[static_cast<size_t>(p)];
        if (cnt == 0) continue;
        if (total_bits % p != 0) {
            throw std::invalid_argument(
                "prepare_downcast: 精度位数必须整除 total_bits");
        }
        slot_of_b[static_cast<size_t>(p)] = static_cast<int>(bs.size());
        bs.push_back(p);
        groups.emplace_back();
        groups.back().reserve(cnt);  // 连续预分配，避免 push_back 反复 realloc
    }
    for (size_t i = 0; i < K; ++i) {
        const int p = levels[i];
        const int s = slot_of_b[static_cast<size_t>(p)];
        groups[static_cast<size_t>(s)].push_back(w[i] >> (total_bits - p));
    }

    std::map<int, NibblePrepared> result;
    for (size_t s = 0; s < bs.size(); ++s) {
        result[bs[static_cast<size_t>(s)]] =
            prepare_nibble_from_raw(groups[s], bs[static_cast<size_t>(s)]);
    }
    return result;
}

// ============================================================
// 便捷重载（默认 {16,8,4}/{100,1000}）
// ============================================================

std::vector<int> LeveledSplitDot::select_levels_default(
    const std::vector<int64_t>& importance, int total_bits) {
    return select_levels(importance, total_bits, default_options(), default_thresholds());
}

std::map<int, std::vector<int64_t>> LeveledSplitDot::dot_split_leveled_default(
    const std::vector<int64_t>& w, const std::vector<int64_t>& x,
    const std::vector<int64_t>& importance, int total_bits) {
    return dot_split_leveled(w, x, importance, total_bits,
                             default_options(), default_thresholds());
}

int64_t LeveledSplitDot::dot_fused_leveled_default(
    const std::vector<int64_t>& w, const std::vector<int64_t>& x,
    const std::vector<int64_t>& importance, int total_bits) {
    return dot_fused_leveled(w, x, importance, total_bits,
                             default_options(), default_thresholds());
}

// ============================================================
// 降档便捷重载（默认 {8,16,32}/{100,1000}）
// ============================================================

std::vector<int> LeveledSplitDot::select_precision_default(
    const std::vector<int64_t>& importance) {
    return select_precision(importance, default_precision_options(),
                            default_precision_thresholds());
}

std::map<int, std::vector<int64_t>> LeveledSplitDot::dot_split_leveled_downcast_default(
    const std::vector<int64_t>& w, const std::vector<int64_t>& x,
    const std::vector<int64_t>& importance, int total_bits) {
    return dot_split_leveled_downcast(w, x, importance, total_bits,
                                      default_precision_options(),
                                      default_precision_thresholds());
}

int64_t LeveledSplitDot::dot_fused_leveled_downcast_default(
    const std::vector<int64_t>& w, const std::vector<int64_t>& x,
    const std::vector<int64_t>& importance, int total_bits) {
    return dot_fused_leveled_downcast(w, x, importance, total_bits,
                                      default_precision_options(),
                                      default_precision_thresholds());
}

} // namespace sgn
