# SGN-Lite

Sparse Graph Network Lite -- 纯整数化竞争/赫布学习识别系统

## 简介

SGN-Lite 是一个**核心识别路径无浮点、无反向传播**的轻量级识别引擎。核心机制为整数化竞争学习 + 模板匹配，适用于嵌入式 MCU 环境的算法验证。

**核心特性**:
- **核心引擎全程整数运算** (DiscreteCoordinate 离散坐标体系)
- 竞争学习 + 赫布增强/削弱
- 双图叠加门控识别 (v4.4)
- 图模式层级记忆 (v5.0)
- 多层神经元架构 (v5.1.3)
- 策略插件化架构

> **注意**: 输入层 (矢量渲染、噪声生成、对比度拉伸) 涉及外部模拟信号，保留浮点运算。核心识别层 (神经元匹配、赫布学习、竞争排序) 严格整数化。

## 安装

### 环境要求
- Python 3.10+
- Windows / Linux / macOS

### 安装依赖

核心功能无需额外安装，Python 标准库即可运行。

```bash
# 可选依赖（图表导出功能）
pip install matplotlib
```

> 未安装 matplotlib 时，脚本会自动降级为 ASCII 曲线显示。

## 快速开始

```bash
cd engine
python main.py
```

按回车进入训练，训练完成后可进行测试/可视化/噪声分析。

### 函数工厂 GUI

基于 pygame 的图形化绘图与训练界面，支持手绘像素图形、实时变换、一键训练。

```bash
# 直接启动 GUI
python gui/sgn_gui_factory.py

# 或从 main.py 菜单进入
python main.py
# 按 f 启动函数工厂
```

**GUI 功能**：
- 网格绘制（4/8/16/32/64）+ 橡皮擦工具
- 内置字符库（全部可打印 ASCII，pygame 覆盖层）
- 实时旋转变换预览（旋转/偏移/缩放）
- 一键训练 + 实时测试
- 撤销/重做（Ctrl+Z / Ctrl+Y，50步历史）
- 图案保存/加载（JSON 文件）
- 画布导出为 PNG 图片
- 自定义训练集管理（多标签共存）
- 导出 CSV/JSON 训练集
- 模型保存/加载（通过桥接层，与 SGN 核心解耦）

详见 [`gui/sgn_gui_factory_documentation.md`](gui/sgn_gui_factory_documentation.md)。

### 输出示例

```
============================================================
  SGN-Lite v5.0 Python PC平台验证脚本 (插件化架构)
============================================================

  模块结构:
    核心层 (Core):
      sgn_core.py      核心引擎(整数化) - 内部逻辑冻结
    扩展层 (Extension):
      sgn_hooks.py     事件总线/钩子系统
      sgn_config.py    配置注册表/动态UI
      sgn_commands.py  命令注册表
    ...



============================================================
  控制面板
============================================================
  当前网络: 256神经元 / 81模板 / 3200步
  [1] 网络架构    [2] 学习参数    [3] 资源限制
  [4] 噪声参数    [5] 界面偏好    [6] 高级选项
  [e] 扩展功能    [s] 保存配置    [l] 加载配置
  [r] 恢复默认    [Enter] 开始训练  [q] 退出
```

训练完成后可使用菜单中的 `[t]` 批量测试、`[c]` 混淆矩阵、`[n]` 噪声测试等功能。

### 命令行参数

```bash
python main.py                          # 交互式训练
python main.py --batch                  # 批量模式
python main.py --auto 50                # 自动模式(50ms/步)
python main.py --mode compact           # 精简模式
python main.py --mode blackbox          # 黑箱模式
python main.py --config config.json     # 加载配置
python main.py --no-color               # 禁用颜色
python main.py --custom-dataset file.json  # 从自定义训练集导入
```

## 架构

```
核心层 (Core)
  sgn_core.py          核心引擎 (竞争/校验/学习/模板合并/多层神经元 v5.1.3)

扩展层 (Extension)
  sgn_hooks.py         事件总线/钩子系统
  sgn_config.py        配置注册表/DiscreteCoordinate
  sgn_commands.py      命令注册表

策略层 (Strategy)
  sgn_input.py         输入管道/噪声模型/矢量渲染
  sgn_layers.py        层提取/边缘提取/分块编码
  sgn_strategies.py    层策略/校验策略/匹配策略
  sgn_metrics.py       评估指标 (准确率/混淆/噪声鲁棒性)
  sgn_storage.py       存储后端 (JSON/SQLite)
  sgn_graph.py         图数据结构 (v5.0)
  sgn_stack.py         图构建与投影 (v5.0)
  sgn_merge.py         跨图合并 (v5.0)
  sgn_graph_match.py   图匹配与推理 (v5.0)

GUI 层 (gui/)
  gui/factory.py       函数工厂主窗口 (GUIFactory)
  gui/canvas.py        绘制画布 (网格/橡皮擦/Bresenham)
  gui/charlib.py       字符库 (全ASCII + pygame字体渲染)
  gui/transform.py     变换引擎 (旋转/偏移/缩放)
  gui/bridge.py        SGN 桥接层 (封装 history/save/load)
  gui/ui.py            UI 组件 (Button/Slider/Label/TextBox)
  gui/dataset_store.py 自定义训练集存储
  gui/utils.py         工具函数

应用层 (Application)
  main.py              入口文件
  sgn_interactive.py   交互菜单
  sgn_training.py      训练循环
  sgn_test.py          推理/批量/混淆/噪声测试
  sgn_visual.py        可视化/统计/仪表盘
  sgn_report.py        图表导出
  sgn_persist.py       模型持久化
  sgn_utils.py         工具函数/颜色输出/日志
```

## 编程接口

除 CLI 外，SGN-Lite 可作为 Python 库直接调用：

```python
from sgn_core import SGNCore
from sgn_input import create_vector_source, create_mixed_vector_source
from sgn_layers import classify_sample

# 1. 创建引擎
core = SGNCore(seed=42)

# 2. 创建输入源（矢量混合模式）
source = create_mixed_vector_source(grid_size=8, samples_per_formula=100)
samples = source.generate_batch(2000)

# 3. 训练
for intensity, label in samples:
    info = core.train(intensity, label)

# 4. 推理
test_intensity, test_label = samples[0]
pred, score = classify_sample(core, test_intensity)
print(f"True: {test_label}, Pred: {pred}, Score: {score}%")

# 5. 保存/加载模型
from sgn_persist import save_model, load_model
save_model(core, "my_model.json")
```

```python
# 使用内置字符模式（4x4）
from sgn_core import SGNCore
from sgn_input import create_default_source

core = SGNCore(seed=42)
source = create_default_source()
samples = source.generate_batch(1000)

for intensity, label in samples:
    core.train(intensity, label)

print(f"Templates: {len(core.templates)}")
```

## 运行模式

| 模式 | 说明 |
|------|------|
| full | 全记载，每步输出 (默认) |
| compact | 精简，仅检查点摘要 |
| blackbox | 黑箱，训练全程零输出 |

## 输入源

| 类型 | 说明 |
|------|------|
| pattern | 内置 4x4 字符 (0-F) |
| vector | 矢量图形 (line/circle/sine/arch/leaf/mixed) |
| file | 从 CSV/JSON 文件加载 |
| 8x8 标准字符 | 0-9 + A-Z (STANDARD_CHARS_8x8) |

矢量图形支持网格大小: 4/8/16/32/64。通过 `--vector-grid` 或控制面板 `[g]` 设置。

## 关键配置项

| 配置 | 默认值 | 可调范围 | 说明 |
|------|--------|---------|------|
| MAX_NEURONS | 256 | 1 ~ 4096 | 神经元数量上限 |
| MAX_TEMPLATES | 500 | 1 ~ 10000 | 模板库容量上限 |
| MAX_ITERATIONS | 100000 | 1 ~ 1000000 | 训练总步数上限 |
| SEED | 42 | 0 ~ 99999 | 随机种子 |
| TOP_K | 6 | 1 ~ 128 | 竞争 Top-K |
| MAX_LOCKOUT | 120 | 1 ~ 1000 | 锁定阈值 |
| ENABLE_GATE_MATCHING | False | -- | 门控匹配 (v4.4) |
| ENABLE_GRAPH_MODE | False | -- | 图模式 (v5.0) |
| ENABLE_MULTI_LAYER_NEURON | False | -- | 多层神经元架构 (v5.1.3) |
| ENABLE_SOFT_GATE | False | -- | 软门控 (v5.1.3) |

> **说明**: 上表"默认值"为系统初始值，"可调范围"为允许的最小/最大值。实际稳态模板数通常在 80 个左右，无需达到上限。

## 图模式 (v5.0)

图模式在神经元之上挂接层级化记忆系统，解决模板识别的三个固有缺陷:
1. 层级缺失 -> 多层图结构
2. 空间结构丢失 -> 连通域 + 位置归一化
3. 过拟合倾向 -> 一致性过滤 + 反馈循环

```bash
# 启用图模式
# 在控制面板 -> 高级选项 中设置 ENABLE_GRAPH_MODE = True
# 或通过配置文件
```

核心机制:
- 神经元竞争 -> 投影为图节点
- 多视图合并 -> 一致性过滤 (高层节点 >=2 投影)
- 反馈迭代 -> 误差图重新扫描
- 层级下压 -> 遗忘机制

## 多层神经元架构 (v5.1.3)

多层神经元架构在神经元之上新增 Layer 1 概念判断层，通过图中间层汇总器连接两层神经元，实现从基础特征到概念判断的层级抽象。

```bash
# 启用多层神经元架构
# 在控制面板 -> 高级选项 中设置 ENABLE_MULTI_LAYER_NEURON = True
# 可选：配合 ENABLE_SOFT_GATE = True 启用软门控
```

数据流:
- Layer 0 神经元竞争 (基础特征专家: 边缘/局部模式)
- 获胜者投影为图节点 -> 图赫布融合
- 提取图结构特征 (10维固定向量) + L0 激活 ID 列表
- Layer 1 神经元竞争 (图特征向量相似度匹配)
- 两层各自独立赫布学习

核心配置:
- `NEURON_LAYER_0_COUNT` / `NEURON_LAYER_1_COUNT`: 两层神经元数量
- `LAYER1_LEARNING_RATE_SCALE`: Layer 1 学习率缩放
- `TOP_K_L1`: Layer 1 竞争 Top-K
- `GATE_SPECIALIZE_THRESHOLD`: 软门控专精化阈值

详见 [`相关修复/v5.1.3_multi_layer_changelog.md`](相关修复/v5.1.3_multi_layer_changelog.md)。

## 评估指标

训练完成后可用的测试:
- `[t]` 批量测试: 识别率
- `[c]` 混淆矩阵: 各类别准确率
- `[n]` 噪声测试: 复合/高斯/椒盐/块遮挡鲁棒性
- `[s]` 统计信息: 神经元/模板/校验通过率
- `[g]` 仪表盘: 综合状态

## 版本历史

- v4.3: 长周期参数重构、Bug修复、8x8标准字符库
- v4.4: 双图叠加门控识别、边缘提取、分块编码
- v5.0: 图模式层级记忆 (sgn_graph/stack/merge/graph_match)
- v5.0-fix: 矢量圆距离计算Bug修复、渲染渐变方向修复
- v5.0-vis: 可视化增强 (10级色阶/8级热力图/模板强度模式/形状标注)
- v5.0-catear: 新增猫耳矢量图形 (4朝向x3间距x3半径x3开角x2形态=216种)
- v5.0-grid: VECTOR_GRID 扩展支持 64x64
- v5.1: catear 语义修正，拆分为 arch(拱弧轮廓) 和 leaf(叶片填充)，原 catear 作为 arch 别名继续兼容
- v5.1: 核心引擎性能优化 (模板索引、门控匹配提前退出、整数化改进)
- v5.1: 输入管线修复 (循环采样、百分位裁剪、BlockNoise 显式 grid_size)
- v5.1.1: 函数工厂 GUI (gui/factory.py) — pygame 绘图界面、全ASCII字符库、撤销/重做、橡皮擦、图案保存/加载、PNG导出
- v5.1.2: GUI 桥接层解耦 — history 访问封装到 bridge、新增模型保存/加载功能、变换状态恢复修复、UI 主题优化
- v5.1.3: 多层神经元架构 — Layer 0/1 双层竞争、图中间层汇总器、软门控、标签专精分片、13个Bug修复

## 当前版本声明 (先验机状态)

> **SGN-Lite v5.0 目前处于先验机状态，而非完整原型机。**
>
> **版本号说明**: 当前版本号 (v5.0) 为先验机阶段版本号，遵循先验机的迭代规则。一旦正式定型为原型机，版本号将清零并作为独立项目重新发布。先验机与原型机分属不同项目阶段，不可混为一谈。
>
> 先验机与原型机的本质区别: 框架与基础设施的成熟度不同。如同胎儿与婴儿 -- 严格意义上均为生命体，但发展阶段完全不同。SGN 当前仍处于"胎儿期"，核心框架已搭建，但关键基础设施尚未完善。
>
> **当前已实现**: 记忆层面的优化代码 (图模式层级记忆、模板合并、竞争学习机制)
>
> **尚未完善**:
> - 时间变量当前仅为计步器，未实现真正的时间维度处理
> - 神经元基础设施 (如动态拓扑、跨层信号传递) 尚未扩展
> - 图模式与神经元的深度衔接尚未完成
>
> **平台定位说明**: SGN 当前为 PC 平台的验证框架代码，**并非 MCU 部署版本**。未来发展方向包含 MCU 适配，但当前代码无法在嵌入式设备上直接运行。下载本代码获取的是算法验证平台，而非可部署的产品固件。
>
> **概念声明**: SGN 不是传统意义上的 SNN (脉冲神经网络) 或 DNN (深度神经网络)，拥有独立的概念体系和术语定义。部分概念可能与现有神经网络范式相似，但存在本质差异。当前优先实现的是记忆架构，而非神经元本体的完整功能。
>
> **使用建议**: 可将 SGN 视为"记忆先验框架"进行实验和扩展，但不应将其作为成熟的神经网络解决方案直接部署。

## 常见问题

### 环境问题

**看到乱码/方块**
使用 PowerShell/Windows Terminal，或执行 `chcp 65001`，或添加 `--no-color` 禁用颜色输出。

**matplotlib 报错**
安装命令: `pip install matplotlib`。脚本会自动降级为 ASCII 曲线显示。

**颜色显示错乱**
添加 `--no-color` 参数强制禁用 ANSI 颜色。脚本会自动检测终端支持情况。

### 训练问题

**训练速度太慢**
使用自动模式: `python main.py --auto 10`，或调大跳步显示（控制面板 `[5]` 界面偏好）。

**想继续之前训练**
启动时使用: `python main.py --resume`，或手动 `[d]` 加载模型文件。

**训练步数设置**
控制面板 `[3]` 资源限制 → MAX_ITERATIONS，建议 3000~5000 收益最大。

**参数修改后没生效**
修改架构参数（神经元数/K值/种子）后返回控制面板再开始。控制面板 `[r]` 是恢复默认配置，不是重置网络！

**切换矢量模式无效**
控制面板 `[6]` 高级选项 → 选择矢量模式 → `[Enter]` 返回。必须返回控制面板再开始训练才会生效。

**按 [a] 又回到控制面板**
这是已修复的 121 结构问题，请更新到最新 main.py。更新后按 `[a]` 会直接开始自动训练。

### 识别问题

**识别率上不去**
尝试调高神经元数 (200~256) 和训练步数 (3000~5000)，并更换随机种子（35 实测比 42 更好）。

**训练后闪退无日志**
检查 `sgn_crash_*.log` 文件，或添加 `--output log.txt`。确保所有 `.py` 文件无缩进错误（Tab/空格混用）。

**Windows 下路径问题**
使用正斜杠或双反斜杠: `sgn_model.json`。脚本已使用 `os.path` 处理路径。

## 依赖

- Python 3.10+
- 可选: matplotlib (图表导出)
- 可选: sqlite3 (SQLite存储后端)
- 可选: pygame-ce (函数工厂 GUI)

## 许可证

Apache License 2.0
