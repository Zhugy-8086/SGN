// level_bindings.cpp - pybind11 绑定（注册到 sgn.level 子模块）
//
// 暴露的 Python API（sgn.level 子模块）：
//   - sgn.level.bits_to_max_range(bits) → int（bits=None 返回 255 旧默认）
//   - sgn.level.max_range_to_bits(max_range, warn_if_non_power) → int
//   - sgn.level.LevelContext（struct）
//   - sgn.level.UpgradedLevelContext（bits 升级版，Stage 3.0.4）
//   - sgn.level.LevelConstants（命名空间常量）
//   - sgn.level.BitsAllocator（class）
//   - sgn.level.LevelOperation（enum）

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <string>

#include "level_math.h"
#include "level_context.h"
#include "level_constants.h"
#include "bits_allocator.h"
#include "level_strategy.h"
#include "neuron_level_stats.h"
#include "level_scheduler.h"

namespace py = pybind11;

namespace {

// DeprecationWarning 发出辅助（PyErr_WarnEx 失败时抛 Python 异常）
inline void emit_deprecation(const std::string& msg) {
    if (PyErr_WarnEx(PyExc_DeprecationWarning, msg.c_str(), 1) < 0) {
        throw py::error_already_set();
    }
}

// py::dict → map<string,string>（value 兼容 str/int/float/bool）
// 旧序列化数据（Stage 2.6）的 value 是 int，新数据是 str
inline std::unordered_map<std::string, std::string> dict_to_map(const py::dict& d) {
    std::unordered_map<std::string, std::string> m;
    for (auto item : d) {
        std::string key = item.first.cast<std::string>();
        py::object v = py::reinterpret_borrow<py::object>(item.second);
        std::string value;
        if (py::isinstance<py::str>(v)) {
            value = v.cast<std::string>();
        } else {
            value = py::str(v).cast<std::string>();  // int/float/bool → str
        }
        m.emplace(std::move(key), std::move(value));
    }
    return m;
}

} // namespace

/* 注册函数：由 placeholder.cpp 的 PYBIND11_MODULE(sgn, m) 调用 */
void register_level(py::module_& m) {
    auto level = m.def_submodule("level", "SGN Level scheduler (precision allocation)");

    // 核心数学函数
    // bits=None → 旧默认 255（test_bits_to_max_range 兼容路径）
    level.def("bits_to_max_range",
              [](py::object bits) -> int64_t {
                  if (bits.is_none()) return sgn::bits_to_max_range(-1);
                  return sgn::bits_to_max_range(bits.cast<int32_t>());
              },
              py::arg("bits"),
              "Convert bits to max_range: (1 << bits) - 1.\n"
              "bits=None or bits < 0 returns 255 (legacy default), "
              "bits >= 32 returns 0xFFFFFFFF.");
    // warn_if_non_power=True 且 max_range 非 2^n-1 时发 DeprecationWarning
    // （仅绑定层发一次，C++ 实现不重复发）
    level.def("max_range_to_bits",
              [](int64_t max_range, bool warn_if_non_power) -> int32_t {
                  int32_t bits = sgn::max_range_to_bits(max_range);
                  if (warn_if_non_power && max_range > 0) {
                      int64_t exact = sgn::bits_to_max_range(bits);
                      if (exact != max_range) {
                          emit_deprecation(
                              "max_range=" + std::to_string(max_range) +
                              " is not 2^n-1, rounding up to bits=" +
                              std::to_string(bits));
                      }
                  }
                  return bits;
              },
              py::arg("max_range"), py::arg("warn_if_non_power") = true,
              "Convert max_range to bits: ceil(log2(max_range + 1)).\n"
              "Returns -1 for max_range <= 0 (unset).\n"
              "warn_if_non_power=True emits DeprecationWarning when max_range "
              "is not exactly 2^n-1.");

    // LevelOperation 枚举
    py::enum_<sgn::LevelOperation>(level, "LevelOperation")
        .value("ADD", sgn::LevelOperation::ADD)
        .value("SUB", sgn::LevelOperation::SUB)
        .value("MUL", sgn::LevelOperation::MUL)
        .value("COMPARE", sgn::LevelOperation::COMPARE)
        .value("ASSIGN", sgn::LevelOperation::ASSIGN)
        .export_values();

    // LevelContext struct
    py::class_<sgn::LevelContext>(level, "LevelContext")
        .def(py::init<>())
        .def(py::init<int32_t, int32_t, int32_t, sgn::LevelOperation, const std::string&>(),
             py::arg("target_level") = 0,
             py::arg("max_range") = 255,
             py::arg("bits") = -1,
             py::arg("operation") = sgn::LevelOperation::ASSIGN,
             py::arg("source") = "")
        .def_readwrite("target_level", &sgn::LevelContext::target_level)
        .def_readwrite("max_range", &sgn::LevelContext::max_range)
        .def_readwrite("bits", &sgn::LevelContext::bits)
        .def_readwrite("operation", &sgn::LevelContext::operation)
        .def_readwrite("source", &sgn::LevelContext::source)
        .def("to_dict", &sgn::LevelContext::to_dict,
             "Serialize to dict (JSON-compatible).")
        .def_static("from_dict", &sgn::LevelContext::from_dict,
                    py::arg("d"),
                    "Deserialize from dict (backward compatible with legacy data).")
        .def("__repr__", [](const sgn::LevelContext& ctx) {
            return "LevelContext(level=" + std::to_string(ctx.target_level) +
                   ", op=" + std::to_string(static_cast<int>(ctx.operation)) +
                   ", range=" + std::to_string(ctx.max_range) +
                   ", bits=" + std::to_string(ctx.bits) + ")";
        });

    // ============================================================
    // UpgradedLevelContext（Stage 3.0.4 Task 4.1/4.2/4.4）
    // ============================================================
    // Python 侧关键字构造：
    //   UpgradedLevelContext()                          # bits=8（从 255 反推）
    //   UpgradedLevelContext(bits=16)                   # bits 入口，同步 max_range
    //   UpgradedLevelContext(max_range=65535)           # max_range 入口，反推 bits
    //   UpgradedLevelContext(bits=16, level_f=..., level_b=...)
    py::class_<sgn::UpgradedLevelContext, sgn::LevelContext>(level, "UpgradedLevelContext")
        .def(py::init([](py::object bits, py::object max_range,
                         py::object level_f, py::object level_b,
                         py::object source) {
                 sgn::UpgradedLevelContext ctx;
                 if (!bits.is_none()) {
                     ctx.set_bits(bits.cast<int32_t>());
                 } else if (!max_range.is_none()) {
                     ctx.max_range = max_range.cast<int32_t>();
                     ctx.sync_from_max_range(false);
                 }
                 if (!level_f.is_none()) {
                     ctx.set_level_f(level_f.cast<sgn::ValueSpec>());
                 }
                 if (!level_b.is_none()) {
                     ctx.set_level_b(level_b.cast<sgn::ValueSpec>());
                 }
                 if (!source.is_none()) {
                     ctx.source = source.cast<std::string>();
                 }
                 return ctx;
             }),
             py::arg("bits") = py::none(),
             py::arg("max_range") = py::none(),
             py::arg("level_f") = py::none(),
             py::arg("level_b") = py::none(),
             py::arg("source") = py::none(),
             "Upgraded LevelContext with bits-first sync and level_f/level_b.")
        // bits property：赋值自动同步 max_range（Task 4.1 双向同步）
        .def_property("bits",
            [](const sgn::UpgradedLevelContext& c) { return c.bits; },
            [](sgn::UpgradedLevelContext& c, int32_t v) { c.set_bits(v); })
        // max_range property：赋值自动反推 bits（旧数据路径）
        .def_property("max_range",
            [](const sgn::UpgradedLevelContext& c) { return c.max_range; },
            [](sgn::UpgradedLevelContext& c, int32_t v) {
                c.max_range = v;
                c.sync_from_max_range(false);
            })
        // level_f / level_b property：None 或 ValueSpec（Task 4.2）
        .def_property("level_f",
            [](const sgn::UpgradedLevelContext& c) -> py::object {
                if (c.has_level_f()) return py::cast(*c.level_f());
                return py::none();
            },
            [](sgn::UpgradedLevelContext& c, py::object v) {
                if (v.is_none()) c.clear_level_f();
                else c.set_level_f(v.cast<sgn::ValueSpec>());
            })
        .def_property("level_b",
            [](const sgn::UpgradedLevelContext& c) -> py::object {
                if (c.has_level_b()) return py::cast(*c.level_b());
                return py::none();
            },
            [](sgn::UpgradedLevelContext& c, py::object v) {
                if (v.is_none()) c.clear_level_b();
                else c.set_level_b(v.cast<sgn::ValueSpec>());
            })
        .def("has_level_f", &sgn::UpgradedLevelContext::has_level_f)
        .def("has_level_b", &sgn::UpgradedLevelContext::has_level_b)
        .def("set_bits", &sgn::UpgradedLevelContext::set_bits,
             py::arg("bits"),
             "Set bits and sync max_range.")
        .def("sync_from_max_range",
             [](sgn::UpgradedLevelContext& c, bool warn) -> bool {
                 bool exact = c.sync_from_max_range(false);
                 if (warn && !exact) {
                     emit_deprecation(
                         "max_range=" + std::to_string(c.max_range) +
                         " is not 2^bits-1, rounding up to bits=" +
                         std::to_string(c.bits));
                 }
                 return exact;
             },
             py::arg("warn") = false,
             "Re-derive bits from max_range. Returns True if exact 2^n-1.\n"
             "warn=True emits DeprecationWarning when non-exact.")
        .def("get_effective_bits",
             &sgn::UpgradedLevelContext::get_effective_bits,
             py::arg("direction"),
             "Effective bits: level_f/level_b (by direction) > bits.")
        .def("get_effective_max_range",
             &sgn::UpgradedLevelContext::get_effective_max_range,
             py::arg("direction"),
             "Effective max_range derived from get_effective_bits.")
        .def("to_dict", &sgn::UpgradedLevelContext::to_dict,
             "Serialize to dict (JSON-compatible, includes level_f/level_b).")
        .def_static("from_dict",
             [](const py::dict& d) {
                 return sgn::UpgradedLevelContext::from_dict(dict_to_map(d));
             },
             py::arg("d"),
             "Deserialize from dict (backward compatible: max_range-only data\n"
             "re-derives bits; values may be str or int).")
        .def("__repr__", [](const sgn::UpgradedLevelContext& c) {
            std::string s = "UpgradedLevelContext(bits=" + std::to_string(c.bits) +
                            ", max_range=" + std::to_string(c.max_range);
            if (c.has_level_f()) s += ", level_f=" + std::to_string(c.level_f()->bits) + "b";
            if (c.has_level_b()) s += ", level_b=" + std::to_string(c.level_b()->bits) + "b";
            return s + ")";
        });

    // LevelConstants（只读属性）
    py::class_<sgn::LevelConstants>(level, "LevelConstants")
        .def_property_readonly_static("DEFAULT_BITS_MIN",
            [](py::object) { return sgn::LevelConstants::DEFAULT_BITS_MIN; })
        .def_property_readonly_static("DEFAULT_BITS_MAX",
            [](py::object) { return sgn::LevelConstants::DEFAULT_BITS_MAX; })
        .def_property_readonly_static("DEFAULT_BITS",
            [](py::object) { return sgn::LevelConstants::DEFAULT_BITS; })
        .def_property_readonly_static("DEFAULT_MAX_RANGE",
            [](py::object) { return sgn::LevelConstants::DEFAULT_MAX_RANGE; })
        .def_property_readonly_static("DEFAULT_TOTAL_BITS",
            [](py::object) { return sgn::LevelConstants::DEFAULT_TOTAL_BITS; })
        .def_property_readonly_static("HYSTERESIS_DELTA",
            [](py::object) { return sgn::LevelConstants::HYSTERESIS_DELTA; })
        .def_property_readonly_static("LEVEL_MIN",
            [](py::object) { return sgn::LevelConstants::LEVEL_MIN; })
        .def_property_readonly_static("LEVEL_MAX",
            [](py::object) { return sgn::LevelConstants::LEVEL_MAX; })
        .def_property_readonly_static("LEVEL_DEFAULT",
            [](py::object) { return sgn::LevelConstants::LEVEL_DEFAULT; })
        .def_property_readonly_static("BITS_UNSET",
            [](py::object) { return sgn::LevelConstants::BITS_UNSET; });

    // BitsAllocator class
    py::class_<sgn::BitsAllocator>(level, "BitsAllocator")
        .def(py::init<uint32_t, uint8_t, uint8_t, uint8_t>(),
             py::arg("total_bits") = sgn::LevelConstants::DEFAULT_TOTAL_BITS,
             py::arg("b_min") = sgn::LevelConstants::DEFAULT_BITS_MIN,
             py::arg("b_max") = sgn::LevelConstants::DEFAULT_BITS_MAX,
             py::arg("hysteresis_delta") = sgn::LevelConstants::HYSTERESIS_DELTA)
        .def("allocate", &sgn::BitsAllocator::allocate,
             py::arg("costs"),
             "Greedy marginal allocation (no hysteresis).\n"
             "costs: list of (grad_l2, in_dim) tuples.\n"
             "Returns list[int] of per-layer bits.")
        .def("allocate_with_hysteresis",
             [](const sgn::BitsAllocator& self,
                const std::vector<std::pair<double, int32_t>>& costs,
                py::object prev_bits) {
                 std::vector<uint8_t> prev;
                 if (!prev_bits.is_none()) {
                     prev = prev_bits.cast<std::vector<uint8_t>>();
                 }
                 return self.allocate_with_hysteresis(costs, prev);
             },
             py::arg("costs"), py::arg("prev_bits"),
             "Greedy allocation with hysteresis (clamp +/- delta, exp23).\n"
             "costs: list of (grad_l2, in_dim) tuples.\n"
             "prev_bits: previous allocation (list[int]) or None (= plain allocate).\n"
             "Returns list[int] of per-layer bits.")
        .def_property_readonly("total_bits", &sgn::BitsAllocator::total_bits)
        .def_property_readonly("b_min", &sgn::BitsAllocator::b_min)
        .def_property_readonly("b_max", &sgn::BitsAllocator::b_max)
        .def_property_readonly("hysteresis_delta", &sgn::BitsAllocator::hysteresis_delta)
        .def("__repr__", [](const sgn::BitsAllocator& ba) {
            return "BitsAllocator(total_bits=" + std::to_string(ba.total_bits()) +
                   ", b_min=" + std::to_string(ba.b_min()) +
                   ", b_max=" + std::to_string(ba.b_max()) + ")";
        });

    // ============================================================
    // LevelStrategy 抽象基类
    // 安全审计 2026-08-16 C1：未实现 pybind11 trampoline——Python 无法子类化
    // （override 不会被 C++ 侧调用）。策略扩展须在 C++ 端实现后经
    // register_strategy 注册；如需 Python 侧策略再评估加 trampoline。
    // ============================================================
    py::class_<sgn::LevelStrategy, std::shared_ptr<sgn::LevelStrategy>>(level, "LevelStrategy")
        .def("name", &sgn::LevelStrategy::name)
        .def("default_level", &sgn::LevelStrategy::default_level)
        .def("get_level_for_operation",
             [](const sgn::LevelStrategy& self, sgn::LevelOperation op,
                int32_t neuron_id, py::object stats_obj) {
                const sgn::NeuronLevelStats* stats_ptr = nullptr;
                if (!stats_obj.is_none()) {
                    stats_ptr = &stats_obj.cast<const sgn::NeuronLevelStats&>();
                }
                return self.get_level_for_operation(op, neuron_id, stats_ptr);
             },
             py::arg("operation"), py::arg("neuron_id") = -1,
             py::arg("stats") = py::none())
        .def("suggest_adaptation",
             [](const sgn::LevelStrategy& self, const sgn::NeuronLevelStats& stats) -> py::object {
                auto result = self.suggest_adaptation(stats);
                if (result.has_value()) {
                    return py::cast(result.value());
                }
                return py::none();
             },
             py::arg("stats"))
        .def("suggest_demotion",
             [](const sgn::LevelStrategy& self, const sgn::NeuronLevelStats& stats) -> py::object {
                auto result = self.suggest_demotion(stats);
                if (result.has_value()) {
                    return py::cast(result.value());
                }
                return py::none();
             },
             py::arg("stats"));

    // StandardStrategy
    py::class_<sgn::StandardStrategy, sgn::LevelStrategy,
               std::shared_ptr<sgn::StandardStrategy>>(level, "StandardStrategy")
        .def(py::init<int32_t>(), py::arg("level") = 0)
        .def("level", &sgn::StandardStrategy::level);

    // AdaptiveStrategy
    py::class_<sgn::AdaptiveStrategy, sgn::LevelStrategy,
               std::shared_ptr<sgn::AdaptiveStrategy>>(level, "AdaptiveStrategy")
        .def(py::init<int32_t, double, int32_t, double, int32_t, double>(),
             py::arg("base_level") = 0,
             py::arg("variance_threshold") = 100.0,
             py::arg("history_window") = 50,
             py::arg("demotion_verification_threshold") = 0.5,
             py::arg("demotion_min_samples") = 30,
             py::arg("grad_variance_threshold") = 0.0)
        .def_property_readonly("variance_threshold",
            &sgn::AdaptiveStrategy::variance_threshold)
        .def_property_readonly("history_window",
            &sgn::AdaptiveStrategy::history_window)
        .def_property_readonly("demotion_verification_threshold",
            &sgn::AdaptiveStrategy::demotion_verification_threshold)
        .def_property_readonly("demotion_min_samples",
            &sgn::AdaptiveStrategy::demotion_min_samples)
        .def_property_readonly("grad_variance_threshold",
            &sgn::AdaptiveStrategy::grad_variance_threshold);

    // NeuronLevelStats
    // 构造支持 level（同时初始化 current/peak）或 current_level/peak_level
    // 分别指定（legacy level.py dataclass 风格，方案 D 测试使用）
    py::class_<sgn::NeuronLevelStats>(level, "NeuronLevelStats")
        .def(py::init([](int32_t neuron_id, py::object level,
                         py::object current_level, py::object peak_level) {
                 sgn::NeuronLevelStats s(neuron_id);
                 if (!level.is_none()) {
                     int32_t lv = level.cast<int32_t>();
                     s.current_level = lv;
                     s.peak_level = lv;
                 }
                 if (!current_level.is_none()) {
                     s.current_level = current_level.cast<int32_t>();
                 }
                 if (!peak_level.is_none()) {
                     s.peak_level = peak_level.cast<int32_t>();
                 }
                 return s;
             }),
             py::arg("neuron_id") = 0,
             py::arg("level") = py::none(),
             py::arg("current_level") = py::none(),
             py::arg("peak_level") = py::none())
        .def_readwrite("neuron_id", &sgn::NeuronLevelStats::neuron_id)
        .def_readwrite("current_level", &sgn::NeuronLevelStats::current_level)
        .def_readwrite("peak_level", &sgn::NeuronLevelStats::peak_level)
        .def_readwrite("verified_count", &sgn::NeuronLevelStats::verified_count)
        .def_readwrite("total_count", &sgn::NeuronLevelStats::total_count)
        .def_readwrite("level_change_count", &sgn::NeuronLevelStats::level_change_count)
        .def_readwrite("last_match", &sgn::NeuronLevelStats::last_match)
        .def_readwrite("last_verification_rate", &sgn::NeuronLevelStats::last_verification_rate)
        // 方案 D 字段直接暴露（测试读取 len(grad_variance_history)）
        .def_readwrite("last_grad_variance", &sgn::NeuronLevelStats::last_grad_variance)
        .def_readwrite("grad_variance_history", &sgn::NeuronLevelStats::grad_variance_history)
        .def("match_variance", &sgn::NeuronLevelStats::match_variance,
             "MAD (Mean Absolute Deviation), pure integer.")
        .def("verification_rate", &sgn::NeuronLevelStats::verification_rate,
             "Verification pass rate.")
        // v1.4-rc20 方案 D: STE 梯度方差（滑动平均，property 与 legacy 对齐）
        .def_property_readonly("grad_variance", &sgn::NeuronLevelStats::grad_variance,
             "Plan-D STE grad variance (moving average of history).\n"
             "Returns 0.0 when fewer than 2 samples.")
        .def("update", &sgn::NeuronLevelStats::update,
             py::arg("match"), py::arg("verified"),
             py::arg("grad_variance") = 0.0,
             "Update statistics with match value and verification flag.\n"
             "grad_variance > 0 accumulates into grad_variance_history (Plan D).")
        .def("set_grad_variance_threshold",
             &sgn::NeuronLevelStats::set_grad_variance_threshold,
             py::arg("threshold"),
             "Plan-D: per-neuron grad variance threshold override.")
        .def("clear_grad_variance_threshold",
             &sgn::NeuronLevelStats::clear_grad_variance_threshold,
             "Plan-D: clear per-neuron override (fall back to strategy default).")
        .def("get_effective_grad_variance_threshold",
             &sgn::NeuronLevelStats::get_effective_grad_variance_threshold,
             py::arg("default_threshold"),
             "Plan-D: effective threshold (override > strategy default).")
        .def("to_dict", &sgn::NeuronLevelStats::to_dict)
        .def_static("from_dict", &sgn::NeuronLevelStats::from_dict, py::arg("d"))
        .def("__repr__", [](const sgn::NeuronLevelStats& s) {
            return "NeuronLevelStats(nid=" + std::to_string(s.neuron_id) +
                   ", level=" + std::to_string(s.current_level) +
                   ", verified=" + std::to_string(s.verified_count) +
                   "/" + std::to_string(s.total_count) + ")";
        });

    // LevelScheduler
    py::class_<sgn::LevelScheduler>(level, "LevelScheduler")
        .def(py::init<int32_t, int32_t, double>(),
             py::arg("cache_size") = 1024,
             py::arg("adapt_interval") = 50,
             py::arg("default_variance_threshold") = 100.0)
        .def("register_strategy", &sgn::LevelScheduler::register_strategy,
             py::arg("name"), py::arg("strategy"))
        .def("bind_neuron", &sgn::LevelScheduler::bind_neuron,
             py::arg("neuron_id"), py::arg("strategy_name"))
        .def("set_default_strategy", &sgn::LevelScheduler::set_default_strategy,
             py::arg("strategy_name"))
        .def("get_strategy", &sgn::LevelScheduler::get_strategy,
             py::arg("name"))
        .def("get_level", &sgn::LevelScheduler::get_level,
             py::arg("neuron_id"), py::arg("operation") = sgn::LevelOperation::ASSIGN)
        .def("get_stats", &sgn::LevelScheduler::get_stats,
             py::arg("neuron_id"),
             // 安全审计 2026-08-16 C2：reference_internal 绑定返回值生命周期
             // 到 scheduler（keeper），避免 Python 侧长期持有裸引用后 scheduler
             // 内部 map rehash 导致的悬挂指针
             py::return_value_policy::reference_internal)
        .def("update_stats", &sgn::LevelScheduler::update_stats,
             py::arg("neuron_id"), py::arg("match"), py::arg("verified"),
             py::arg("grad_variance") = 0.0)
        .def("step", &sgn::LevelScheduler::step)
        .def("to_dict", &sgn::LevelScheduler::to_dict)
        .def_static("from_dict", &sgn::LevelScheduler::from_dict, py::arg("d"))
        .def_property_readonly("adapt_interval", &sgn::LevelScheduler::adapt_interval)
        .def_property_readonly("step_counter", &sgn::LevelScheduler::step_counter)
        .def("__repr__", [](const sgn::LevelScheduler& s) {
            return "LevelScheduler(interval=" + std::to_string(s.adapt_interval()) +
                   ", step=" + std::to_string(s.step_counter()) + ")";
        });
}