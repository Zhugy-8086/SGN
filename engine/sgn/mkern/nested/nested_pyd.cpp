// nested_pyd.cpp - mkern/nested 独立 Python 扩展入口（mkern-only 分发用）
//
// 背景：Phase 2（2026-09-05）——mkern 文件夹独立上传 GitHub 时，Python 绑定
// 的 pyd 一直靠 engine 主构建产出（不在 mkern/ 内）。本入口让 mkern/nested/
// 自足：NESTED_BUILD_PYD=ON 时产出 sgn_nested.cp*.pyd，无需 engine。
//
// 模块名 sgn_nested（避免与 engine 的 sgn 包冲突）。顶层重导出 mkern_nested
// 子模块的全部公开 API，同时也保留 sgn_nested.mkern_nested.* 路径。
//
// 构建（见 CMakeLists.txt NESTED_BUILD_PYD 选项）：
//   cmake -B build -G Ninja -DNESTED_BUILD_PYD=ON \
//         -DCMAKE_CXX_COMPILER=clang++ \
//         -Dpybind11_DIR="$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')"
//   cmake --build build
//   → build/sgn_nested.cp*.pyd

#include <pybind11/pybind11.h>

namespace py = pybind11;

void register_nested(py::module_& m);   // 定义于 nested_bindings.cpp

PYBIND11_MODULE(sgn_nested, m) {
    register_nested(m);
    // 顶层重导出：sgn_nested.nested_quant_i32 等直达（不必经 .mkern_nested.）
    const char* names[] = {"nested_quant_i32", "nested_dequant",
                           "nested_view_codes", "active_backend", "LEVELS"};
    py::object sub = m.attr("mkern_nested");
    for (const char* name : names) {
        m.attr(name) = sub.attr(name);
    }
}
