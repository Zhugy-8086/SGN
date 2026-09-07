# THIRD_PARTY_NOTICES — 第三方组件声明

本项目分发的构建产物（`sgn.cp314-win_amd64.pyd` 及配套 DLL）包含以下第三方组件。
本文件随分发物与仓库一并提供；许可全文见 `LICENSES/LLVM-openmp-LICENSE.txt`（已随仓附带）。

---

## 1. LLVM OpenMP 运行时（libomp.dll）

- **来源**：LLVM Project（openmp runtime），随 Clang/LLVM 工具链分发，本项目经
  `-fopenmp=libomp` 链接、构建后复制至 `.pyd` 同目录（见 COMPILER_TOOLCHAIN.md §8.2）。
- **许可**：**Apache License 2.0 with LLVM Exceptions**（openmp runtime 历史部分另受
  UIUC/BSD-like 与 MIT 双许可覆盖，以 LLVM 官方 LICENSE.TXT 为准）。
- **许可全文**：已随仓附带于 `LICENSES/LLVM-openmp-LICENSE.txt`；官方版本：
  - LLVM 官方：<https://releases.llvm.org/LICENSE.TXT>
  - GitHub（llvm-project/openmp/runtime/LICENSE.txt）：<https://github.com/llvm/llvm-project>
- **对本项目的影响**：Apache 2.0 非 copyleft 许可，且 LLVM Exceptions 明确"与其链接
  编译产生的可执行文件不被该许可覆盖"——本项目代码（Apache-2.0）无需因链接/捆绑
  libomp.dll 而变更许可或开放额外源码。
- **分发义务**（Apache 2.0 §4）：① 随分发物附本声明及许可全文；② 保留原版权与
  NOTICE 声明，不得移除。

> ⚠️ 勿混淆：GCC 工具链的 OpenMP 运行时为 **libgomp**（GPLv3 with GCC Runtime
> Library Exception），与本文件所述 libomp 无关。本项目分发物仅含 Clang/libomp 链
> 构建产物；GCC 构建仅用于本地交叉验证，不进入分发物。

---

## 维护说明

- 新增第三方组件（如未来引入其他捆绑 DLL/静态库）时，在本文件追加一节，
  并在 README License 节保持指向本文件。
