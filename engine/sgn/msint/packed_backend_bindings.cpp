// packed_backend_bindings.cpp - pybind11 绑定（注册到 sgn 模块）
//
// 暴露的 Python API：
//   - sgn.SlotSpec：槽位规格（bits, is_signed, offset）
//   - sgn.PackedBackend：打包后端
//       - PackedBackend.from_bits([8, 8, 8], packed=0)
//       - backend.get(index) / backend.set(index, value)
//       - backend.get_all() / backend.get_all_simd()
//       - backend.packed_value / backend.slot_count / backend.total_bits
//   - sgn.batch_get_all(bits_list, packed_array) → 2D numpy array
//   - sgn.batch_decode_to_float(bits_list, packed_array, scale) → 1D float32
//   - sgn.batch_decode_to_float_into(bits_list, packed_array, scale, out) → None
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "mkern/simd/simd_api.h"
#include "packed_backend.h"

namespace py = pybind11;

// ============================================================
// 核心解码逻辑（共享于 batch_decode_to_float 和 _into 变体）
// ============================================================

static void decode_to_float_core(
    const std::vector<int>& bits_list,
    const uint64_t* pv_ptr,
    int n_values,
    float scale,
    float* res_ptr,
    bool signed_concat = true   // A1-1：concat 视角符号语义（默认 true = 现行为）
) {
    int total = 0;
    for (int b : bits_list) {
        if (b <= 0) throw std::invalid_argument("bits 必须 > 0");
        total += b;
    }
    if (total > 64) throw std::invalid_argument("总位数超过 64");

    // 快速路径：backward_int16 schema (8+8 bits, total=16)
    // 解码内核已迁出到 simd::decode_i16_f32（含 SSSE3 PSHUFB+PALIGNR+PMOVSXWD 热路径、
    // AVX2 原始路径回退、标量锚点，见 simd 原语层）；此处仅保留 schema 判定。
    // signed_concat=false 时不可用（decode_i16_f32 为 i16 补码语义）→ 回退通用路径。
    if (signed_concat && total == 16 && bits_list.size() == 2 &&
        bits_list[0] == 8 && bits_list[1] == 8) {
        sgn::simd::decode_i16_f32(pv_ptr, n_values, scale, res_ptr);
    }
    // 快速路径：8-bit 单槽位
    else if (bits_list.size() == 1 && bits_list[0] == 8) {
        for (int i = 0; i < n_values; ++i) {
            if (signed_concat) {
                int8_t val8 = static_cast<int8_t>(pv_ptr[i] & 0xFF);
                res_ptr[i] = static_cast<float>(val8) * scale;
            } else {
                uint8_t val8 = static_cast<uint8_t>(pv_ptr[i] & 0xFF);
                res_ptr[i] = static_cast<float>(val8) * scale;
            }
        }
    }
    // 通用路径：多槽位 concat
    else {
        int n_slots = static_cast<int>(bits_list.size());
        std::vector<int> offsets(n_slots);
        std::vector<uint64_t> masks(n_slots);
        int off = total;
        for (int i = 0; i < n_slots; ++i) {
            off -= bits_list[i];
            offsets[i] = off;
            masks[i] = (bits_list[i] >= 64) ? ~0ULL : ((1ULL << bits_list[i]) - 1);
        }

#ifdef __BMI2__
        // PEXT: 单指令提取所有槽位并紧凑排列
        // 构造 field_mask（标记所有槽位在 packed 中的位置）
        uint64_t field_mask = 0;
        for (int j = 0; j < n_slots; ++j) {
            field_mask |= masks[j] << offsets[j];
        }

        for (int i = 0; i < n_values; ++i) {
            uint64_t concat_val = _pext_u64(pv_ptr[i], field_mask);
            // 符号扩展
            if (total < 64) {
                uint64_t sign_bit = 1ULL << (total - 1);
                if (concat_val & sign_bit) {
                    concat_val |= ~((1ULL << total) - 1);
                }
            }
            int64_t signed_val = static_cast<int64_t>(concat_val);
            res_ptr[i] = static_cast<float>(signed_val) * scale;
        }
#else
        // 回退：逐槽位 shift + mask + OR（兼容无 BMI2 的平台）
        for (int i = 0; i < n_values; ++i) {
            uint64_t pv = pv_ptr[i];
            uint64_t concat_val = 0;
            for (int j = 0; j < n_slots; ++j) {
                uint64_t slot_val = (pv >> offsets[j]) & masks[j];
                concat_val = (concat_val << bits_list[j]) | slot_val;
            }
            // 符号扩展（A1-1：signed_concat=false 时跳过——无符号 concat 视角）
            if (signed_concat && total < 64) {
                uint64_t sign_bit = 1ULL << (total - 1);
                if (concat_val & sign_bit) {
                    concat_val |= ~((1ULL << total) - 1);
                }
            }
            int64_t signed_val = static_cast<int64_t>(concat_val);
            res_ptr[i] = static_cast<float>(signed_val) * scale;
        }
#endif
    }
}

void register_packed_backend(py::module_& m) {
    // ---- SlotSpec ----
    py::class_<sgn::SlotSpec>(m, "SlotSpec")
        .def(py::init<>())
        .def(py::init<int, bool, int>(), py::arg("bits"), py::arg("is_signed"), py::arg("offset"))
        .def_readwrite("bits", &sgn::SlotSpec::bits)
        .def_readwrite("is_signed", &sgn::SlotSpec::is_signed)
        .def_readwrite("offset", &sgn::SlotSpec::offset)
        .def("__repr__", [](const sgn::SlotSpec& s) {
            return "<SlotSpec bits=" + std::to_string(s.bits) +
                   " signed=" + (s.is_signed ? "True" : "False") +
                   " offset=" + std::to_string(s.offset) + ">";
        });

    // ---- PackedBackend ----
    py::class_<sgn::PackedBackend>(m, "PackedBackend")
        .def(py::init<const std::vector<sgn::SlotSpec>&, uint64_t>(),
             py::arg("slots"), py::arg("packed") = 0)
        .def_static("from_bits", &sgn::PackedBackend::from_bits,
                    py::arg("bits_list"),
                    py::arg("signed_flags") = std::vector<bool>(),
                    py::arg("packed") = 0,
                    "从 bits 列表构造（自动计算 offset，第一个槽位在高位）")
        .def("get", &sgn::PackedBackend::get, py::arg("index"),
             "读取第 index 个槽位（含符号位处理）")
        .def("set", &sgn::PackedBackend::set, py::arg("index"), py::arg("value"),
             "写入第 index 个槽位")
        .def("get_all", &sgn::PackedBackend::get_all,
             "批量读取所有槽位（标量实现）")
        .def("get_all_simd", &sgn::PackedBackend::get_all_simd,
             "批量读取所有槽位（AVX2 优化，等宽 ≥8 bit 场景）")
        .def_property_readonly("packed_value", &sgn::PackedBackend::packed_value,
                               "底层打包的整数值")
        .def_property_readonly("slot_count", &sgn::PackedBackend::slot_count,
                               "槽位数量")
        .def_property_readonly("total_bits", &sgn::PackedBackend::total_bits,
                               "总位数")
        .def("slots", &sgn::PackedBackend::slots,
             "槽位规格列表（bits/is_signed/offset）——A1-3：序列化后可独立检查")
        .def("serialize", &sgn::PackedBackend::serialize,
             "序列化为 JSON 字符串（含 slots 数组——A1-3 修复后可独立恢复完整状态）")
        .def_static("deserialize", &sgn::PackedBackend::deserialize,
                    py::arg("json"),
                    "从 serialize() 输出重建（含槽位定义，独立恢复完整状态——A1-3）")
        .def("__repr__", [](const sgn::PackedBackend& b) {
            return "<PackedBackend slots=" + std::to_string(b.slot_count()) +
                   " total_bits=" + std::to_string(b.total_bits()) +
                   " packed=0x" + ([](uint64_t v) -> std::string {
                       char buf[32];
                       snprintf(buf, sizeof(buf), "%llx", static_cast<unsigned long long>(v));
                       return buf;
                   })(b.packed_value()) + ">";
        });

    // ---- batch_get_all：numpy 零拷贝批量读取 ----
    // 接受 numpy uint64 数组，返回 2D int64 数组 [n_values, n_slots]
    // 一次 C++ 调用处理整个数组，消除逐元素 pybind11 开销
    // A1-1：signed_flags（None=全无符号，与 from_bits 约定一致；逐槽补码符号扩展）
    m.def("batch_get_all",
        [](const std::vector<int>& bits_list,
           py::array_t<uint64_t, py::array::c_style> packed_array,
           py::object signed_flags_obj) {
            std::vector<bool> slot_signed;
            bool any_signed = false;
            if (!signed_flags_obj.is_none()) {
                auto flags = signed_flags_obj.cast<std::vector<bool>>();
                slot_signed.resize(bits_list.size(), false);
                for (size_t i = 0; i < flags.size() && i < slot_signed.size(); ++i) {
                    slot_signed[i] = flags[i];
                    any_signed = any_signed || flags[i];
                }
            } else {
                slot_signed.resize(bits_list.size(), false);
            }
            if (packed_array.ndim() != 1) {
                throw std::invalid_argument("packed_array 必须是 1D 数组");
            }
            int n_values = static_cast<int>(packed_array.size());
            int n_slots = static_cast<int>(bits_list.size());
            if (n_values == 0 || n_slots == 0) {
                return py::array_t<int64_t>({0, 0});
            }

            const uint64_t* pv_ptr = packed_array.data(0);

            // 预计算 offsets 和 masks
            int total = 0;
            for (int b : bits_list) {
                if (b <= 0) throw std::invalid_argument("bits 必须 > 0");
                total += b;
            }
            if (total > 64) throw std::invalid_argument("总位数超过 64");

            std::vector<int> offsets(n_slots);
            std::vector<uint64_t> masks(n_slots);
            int off = total;
            for (int i = 0; i < n_slots; ++i) {
                off -= bits_list[i];
                offsets[i] = off;
                masks[i] = (bits_list[i] >= 64) ? ~0ULL : ((1ULL << bits_list[i]) - 1);
            }

            // 检查 8-bit 等宽快速路径（输出为无符号字节值——
            // 任何 signed 槽位存在时回退标量做符号扩展）
            bool use_8bit_fast = (n_slots >= 1 && n_slots <= 8) && !any_signed;
            if (use_8bit_fast) {
                for (int b : bits_list) {
                    if (b != 8) { use_8bit_fast = false; break; }
                }
            }

            auto result = py::array_t<int64_t>({n_values, n_slots});
            int64_t* res_ptr = result.mutable_data(0);

            if (use_8bit_fast) {
                // 8-bit 等宽：直接字节提取 + 反转
                for (int i = 0; i < n_values; ++i) {
                    uint64_t pv = pv_ptr[i];
                    const uint8_t* bytes = reinterpret_cast<const uint8_t*>(&pv);
                    for (int j = 0; j < n_slots; ++j) {
                        res_ptr[i * n_slots + j] =
                            static_cast<int64_t>(bytes[n_slots - 1 - j]);
                    }
                }
            } else {
                // 通用标量路径 + 逐槽符号扩展（与 PackedBackend::get 口径一致：
                // 补码；bits=64 时位型即 int64 值，M1 修复口径）
                for (int i = 0; i < n_values; ++i) {
                    uint64_t pv = pv_ptr[i];
                    for (int j = 0; j < n_slots; ++j) {
                        uint64_t raw = (pv >> offsets[j]) & masks[j];
                        int64_t v = static_cast<int64_t>(raw);
                        if (slot_signed[j]) {
                            int b = bits_list[j];
                            if (b == 64) {
                                v = static_cast<int64_t>(raw);
                            } else if (raw & (1ULL << (b - 1))) {
                                v = static_cast<int64_t>(raw)
                                    - static_cast<int64_t>(1ULL << b);
                            }
                        }
                        res_ptr[i * n_slots + j] = v;
                    }
                }
            }

            return result;
        },
        py::arg("bits_list"),
        py::arg("packed_array"),
        py::arg("signed_flags") = py::none(),
        "批量读取多个 packed 值的所有槽位，返回 2D numpy 数组 [n_values, n_slots]；"
        "signed_flags 为逐槽符号标志列表（None=全无符号）——A1-1"
    );

    // ---- batch_decode_to_float：完整解码流水线（C++ 单次调用）----
    // 接受 numpy uint64 数组 + scale，返回 float32 numpy 数组
    // 一次性完成：槽位提取 → concat → signed 转换 → float32 × scale
    // A1-1：signed 参数显式化（默认 True = 原行为；False = 无符号 concat 视角）
    m.def("batch_decode_to_float",
        [](const std::vector<int>& bits_list,
           py::array_t<uint64_t, py::array::c_style> packed_array,
           float scale,
           bool signed_concat) {
            if (packed_array.ndim() != 1) {
                throw std::invalid_argument("packed_array 必须是 1D 数组");
            }
            int n_values = static_cast<int>(packed_array.size());
            if (n_values == 0 || bits_list.empty()) {
                return py::array_t<float>(0);
            }

            const uint64_t* pv_ptr = packed_array.data(0);
            auto result = py::array_t<float>(n_values);
            float* res_ptr = result.mutable_data(0);

            decode_to_float_core(bits_list, pv_ptr, n_values, scale, res_ptr,
                                 signed_concat);
            return result;
        },
        py::arg("bits_list"),
        py::arg("packed_array"),
        py::arg("scale"),
        py::arg("signed") = true,
        "完整解码流水线：packed → concat → signed → float32 × scale，返回 1D float32 数组；"
        "signed=True（默认）为补码 concat 视角（原行为），False 为无符号视角——A1-1"
    );

    // ---- batch_decode_to_float_into：in-place 变体（预分配输出数组）----
    // 消除输出数组分配开销，适用于训练循环中预分配 buffer 的场景
    m.def("batch_decode_to_float_into",
        [](const std::vector<int>& bits_list,
           py::array_t<uint64_t, py::array::c_style> packed_array,
           float scale,
           py::array_t<float, py::array::c_style> output,
           bool signed_concat) {
            if (packed_array.ndim() != 1) {
                throw std::invalid_argument("packed_array 必须是 1D 数组");
            }
            if (output.ndim() != 1) {
                throw std::invalid_argument("output 必须是 1D 数组");
            }
            int n_values = static_cast<int>(packed_array.size());
            if (static_cast<int>(output.size()) < n_values) {
                throw std::invalid_argument("output 长度不足");
            }
            if (n_values == 0 || bits_list.empty()) {
                return;
            }

            const uint64_t* pv_ptr = packed_array.data(0);
            float* res_ptr = output.mutable_data(0);

            decode_to_float_core(bits_list, pv_ptr, n_values, scale, res_ptr,
                                 signed_concat);
        },
        py::arg("bits_list"),
        py::arg("packed_array"),
        py::arg("scale"),
        py::arg("output"),
        py::arg("signed") = true,
        "in-place 解码：写入预分配的 output 数组，消除分配开销；"
        "signed=True（默认）为补码 concat 视角（原行为），False 为无符号视角——A1-1"
    );
}
