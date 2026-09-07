// split_dot_bindings.cpp - pybind11 绑定（注册到 sgn 模块）
//
// 暴露的 Python API：
//   - sgn.SplitDot.split_parts(value, total_bits, split_bits) → list[int]
//   - sgn.SplitDot.dot_split(w, x, total_bits=32, split_bits=16, trim_high_diag=False) → list[int]
//       多输出模式：partial[m]，m=0..2n-2
//       n=2（int32→int16）时返回 [fine, cross, coarse]
//       trim_high_diag=True 时跳过高位对角（m>=n），partials[m>=n] 保持 0（4 位档省 44% ALU）
//   - sgn.SplitDot.dot_fused(w, x, total_bits=32, split_bits=16) → int
//       单融合模式：数值等价于原始点积（低 64 位）
//   - sgn.SplitDot.dot_fused_i32(w, x, split_bits=4) → int
//       单融合 int32 截断模式：内部走低位对角裁剪内核（trim_high_diag=true）
//   - sgn.SplitDot.fuse_128(partials, split_bits) → (hi, lo)
//       128 位精确融合，避免 int64 溢出
//   - sgn.SplitDot.prepare_nibble_from_raw(w, total_bits=32) → NibblePrepared
//       4 位摊销接口：拆分+打包+预解包一次（权重离线 / 激活每前向一次），供热路径复用
//   - sgn.SplitDot.dot_prepared4(w, x, trim_high_diag=False) → list[int]
//       4 位摊销热路径：纯 n² dpbusd 点积（无解包），M 输出复用同一预解包结果
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "split_dot.h"

namespace py = pybind11;

void register_split_dot(py::module_& m) {
    py::class_<sgn::NibblePrepared>(m, "NibblePrepared",
        "4 位预解包缓存（权重/激活各一次，供热路径 narrow_dot_prepared4 复用）")
        .def_readonly("n", &sgn::NibblePrepared::n)
        .def_readonly("K", &sgn::NibblePrepared::K);

    py::class_<sgn::SplitDot>(m, "SplitDot",
        "MSInt 前向多精度拆分点积：一次数据搬运，多精度解释")
        .def_static("split_parts", &sgn::SplitDot::split_parts,
                    py::arg("value"), py::arg("total_bits"), py::arg("split_bits"),
                    "把 int64 按 split_bits 拆成 n 部分（低位在前，最高位符号扩展）")
        .def_static("dot_split", &sgn::SplitDot::dot_split,
                    py::arg("w"), py::arg("x"),
                    py::arg("total_bits") = 32, py::arg("split_bits") = 16,
                    py::arg("trim_high_diag") = false,
                    "多输出模式：返回 partial[m]，m=0..2n-2（n=2 时为 [fine, cross, coarse]）；"
                    "trim_high_diag=True 时裁剪高位对角（m>=n），partials[m>=n] 保持 0")
        .def_static("dot_fused", &sgn::SplitDot::dot_fused,
                    py::arg("w"), py::arg("x"),
                    py::arg("total_bits") = 32, py::arg("split_bits") = 16,
                    "单融合模式：数值等价于原始点积（低 64 位）")
        .def_static("dot_fused_i32", &sgn::SplitDot::dot_fused_i32,
                    py::arg("w"), py::arg("x"),
                    py::arg("split_bits") = 4,
                    "单融合 int32 截断模式：内部走低位对角裁剪内核（trim_high_diag=true）")
        .def_static("fuse_128", &sgn::SplitDot::fuse_128,
                    py::arg("partials"), py::arg("split_bits"),
                    "128 位精确融合：返回 (hi, lo)")
        .def_static("prepare_nibble_from_raw", &sgn::prepare_nibble_from_raw,
                    py::arg("w"), py::arg("total_bits") = 32,
                    "4 位摊销：拆分+打包+预解包一次为 NibblePrepared（权重离线/激活每前向一次；"
                    "split_parts_fixed 栈缓冲直写，无逐元素 vector 堆分配）")
        .def_static("matmul_prepared4",
                    [](const py::sequence& W, const sgn::NibblePrepared& x,
                       bool trim_high_diag) {
                        // 引用遍历预解包权重行（py::cast<const&> 无深拷贝）。
                        // 若按值接收 std::vector<NibblePrepared>，pybind 会深拷贝每个
                        // 预解包字节数组（K×8B/行），M=1024、K=16384 时单次深拷贝 128MB，
                        // 完全吞掉摊销收益。
                        std::vector<std::vector<int64_t>> out;
                        out.reserve(py::len(W));
                        for (py::handle item : W) {
                            const auto& pw =
                                py::cast<const sgn::NibblePrepared&>(item);
                            out.push_back(
                                sgn::narrow_dot_prepared4(pw, x, trim_high_diag));
                        }
                        return out;
                    },
                    py::arg("W"), py::arg("x"),
                    py::arg("trim_high_diag") = false,
                    "批量 M 输出摊销（无深拷贝）：引用遍历 M 个预解包权重行，复用同一预解包"
                    "激活 x，M 循环在 C++ 内执行；返回 M×(2n-1) partials；"
                    "trim_high_diag=True 时跳过高位对角（m>=n），partials[m>=n] 保持 0")
        .def_static("dot_prepared4", &sgn::narrow_dot_prepared4,
                    py::arg("w"), py::arg("x"),
                    py::arg("trim_high_diag") = false,
                    "4 位摊销热路径：纯 n² dpbusd 点积（无解包），M 输出复用同一预解包结果；"
                    "trim_high_diag=True 时跳过高位对角（m>=n），partials[m>=n] 保持 0");
}
