// leveled_split_dot_bindings.cpp - pybind11 绑定（注册到 sgn 模块）
//
// 暴露的 Python API：
//   - sgn.LeveledSplitDot.select_levels(importance, total_bits, options, thresholds) → list[int]
//   - sgn.LeveledSplitDot.dot_split_leveled(w, x, importance, total_bits, options, thresholds)
//       → dict {b: partials_b}（按粒度分组的 1:N 多输出）
//   - sgn.LeveledSplitDot.dot_fused_leveled(w, x, importance, total_bits, options, thresholds)
//       → int（数值等价原始点积，低 64 位）
//   - sgn.LeveledSplitDot.select_levels_default(importance, total_bits=32)
//   - sgn.LeveledSplitDot.dot_split_leveled_default(w, x, importance, total_bits=32)
//   - sgn.LeveledSplitDot.dot_fused_leveled_default(w, x, importance, total_bits=32)
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "leveled_split_dot.h"

namespace py = pybind11;

void register_leveled_split_dot(py::module_& m) {
    py::class_<sgn::LeveledSplitDot>(m, "LeveledSplitDot",
        "MSInt 逐元素异构粒度拆分点积：1:N 多精度解释 + Level 逐元素精度选择组合")
        .def_static("select_levels", &sgn::LeveledSplitDot::select_levels,
                    py::arg("importance"), py::arg("total_bits"),
                    py::arg("options"), py::arg("thresholds"),
                    "按重要性为每个元素选择拆分粒度（返回 split_bits 列表）")
        .def_static("dot_split_leveled", &sgn::LeveledSplitDot::dot_split_leveled,
                    py::arg("w"), py::arg("x"), py::arg("importance"),
                    py::arg("total_bits"), py::arg("options"), py::arg("thresholds"),
                    py::arg("trim_high_diag") = false,
                    "异构粒度多输出：返回 {b: partials_b}（各组 1:N 多精度解释）；"
                    "trim_high_diag=True 时组内裁剪高位对角（m>=n），partials[m>=n] 保持 0")
        .def_static("dot_fused_leveled", &sgn::LeveledSplitDot::dot_fused_leveled,
                    py::arg("w"), py::arg("x"), py::arg("importance"),
                    py::arg("total_bits"), py::arg("options"), py::arg("thresholds"),
                    "异构粒度单融合：数值等价原始点积（低 64 位）")
        .def_static("select_levels_default", &sgn::LeveledSplitDot::select_levels_default,
                    py::arg("importance"), py::arg("total_bits") = 32,
                    "默认选择器（{16,8,4}/{100,1000}）下逐元素选粒度")
        .def_static("dot_split_leveled_default", &sgn::LeveledSplitDot::dot_split_leveled_default,
                    py::arg("w"), py::arg("x"), py::arg("importance"),
                    py::arg("total_bits") = 32,
                    "默认选择器下异构粒度多输出")
        .def_static("dot_fused_leveled_default", &sgn::LeveledSplitDot::dot_fused_leveled_default,
                    py::arg("w"), py::arg("x"), py::arg("importance"),
                    py::arg("total_bits") = 32,
                    "默认选择器下异构粒度单融合")
        .def_static("select_precision", &sgn::LeveledSplitDot::select_precision,
                    py::arg("importance"), py::arg("options"), py::arg("thresholds"),
                    "降档决策：按重要性选择精度位数 p（低重要度 → 低精度）。\n"
                    "options: 精度位数列表，从低精度 → 高精度排列（如 {8, 16, 32}）\n"
                    "thresholds: 升序阈值，长度 = options.size()-1\n"
                    "返回: p[i] 为第 i 个元素所选精度位数（高重要度 → 高精度位数）")
        .def_static("dot_split_leveled_downcast", &sgn::LeveledSplitDot::dot_split_leveled_downcast,
                    py::arg("w"), py::arg("x"), py::arg("importance"),
                    py::arg("total_bits"), py::arg("options"), py::arg("thresholds"),
                    "降档摊销多输出：按精度分组，每组 keep_top 后走 4 位摊销路径，返回 {p: partials}；\n"
                    "partials[p] 是 p-bit 原始 partials（n=p/4 组合，平方级省算）。\n"
                    "恢复量纲需左移 2(32-p)，见 dot_fused_leveled_downcast。")
        .def_static("dot_fused_leveled_downcast", &sgn::LeveledSplitDot::dot_fused_leveled_downcast,
                    py::arg("w"), py::arg("x"), py::arg("importance"),
                    py::arg("total_bits"), py::arg("options"), py::arg("thresholds"),
                    "降档摊销单融合：跨组恢复量纲 + 128 位累加，返回低 64 位近似结果；\n"
                    "误差 ≈ 2^(1-p)，p 为各元素降档精度（p=32 精确）。")
        .def_static("prepare_downcast", &sgn::LeveledSplitDot::prepare_downcast,
                    py::arg("w"), py::arg("importance"),
                    py::arg("total_bits"), py::arg("options"), py::arg("thresholds"),
                    "降档预解包融合：一次传入原始 w，C++ 内部分组 + keep_top 截断 + prepare，\n"
                    "返回 {p: NibblePrepared}（消除 Python 层分组/截断/重复 pybind 转换）。\n"
                    "权重侧每行离线一次、激活侧每前向一次，供热路径 matmul_prepared4 复用。")
        .def_static("prepare_downcast_np",
                    [](py::array_t<int64_t, py::array::c_style | py::array::forcecast> w,
                       py::array_t<int64_t, py::array::c_style | py::array::forcecast> importance,
                       int total_bits,
                       const std::vector<int>& options,
                       const std::vector<int64_t>& thresholds) {
                        const auto K = w.size();
                        if (K != importance.size()) {
                            throw std::invalid_argument("w/importance 长度不一致");
                        }
                        // 零拷贝：int64 C-contiguous numpy 数组直接读内存指针，
                        // 无 pybind 逐元素 list→vector 转换（阶段 2）。
                        return sgn::LeveledSplitDot::prepare_downcast_ptr(
                            w.data(), importance.data(),
                            static_cast<size_t>(K),
                            total_bits, options, thresholds);
                    },
                    py::arg("w"), py::arg("importance"),
                    py::arg("total_bits"), py::arg("options"), py::arg("thresholds"),
                    "numpy 零拷贝版 prepare_downcast：接受 int64 连续 numpy 数组，\n"
                    "直接读内存指针，消除 pybind list→vector 逐元素转换（K=16384 时该转换\n"
                    "~0.26~0.57ms，是 prepare 剩余主头）。语义与 prepare_downcast 完全一致。")
        .def_static("select_precision_default", &sgn::LeveledSplitDot::select_precision_default,
                    py::arg("importance"),
                    "降档决策便捷版：默认 {8,16,32}/{100,1000}")
        .def_static("dot_split_leveled_downcast_default",
                    &sgn::LeveledSplitDot::dot_split_leveled_downcast_default,
                    py::arg("w"), py::arg("x"), py::arg("importance"),
                    py::arg("total_bits") = 32,
                    "降档摊销多输出便捷版：默认 {8,16,32}/{100,1000}")
        .def_static("dot_fused_leveled_downcast_default",
                    &sgn::LeveledSplitDot::dot_fused_leveled_downcast_default,
                    py::arg("w"), py::arg("x"), py::arg("importance"),
                    py::arg("total_bits") = 32,
                    "降档摊销单融合便捷版：默认 {8,16,32}/{100,1000}");
}
