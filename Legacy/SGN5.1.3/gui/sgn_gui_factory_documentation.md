# SGN-Lite v5.1.2 函数工厂 GUI — 设计文档

> **文档日期**: 2025-07-04  
> **版本**: v5.1.2  
> **作者**: zhugy-8086  
> **依赖**: pygame-ce ≥ 2.5, SGN-Lite v5.1 核心

---

## 目录

- [1. 项目概述](#1-项目概述)
- [2. 设计目标](#2-设计目标)
- [3. 架构设计](#3-架构设计)
- [4. 模块拆分](#4-模块拆分)
- [5. 核心模块详解](#5-核心模块详解)
- [6. 增强功能](#6-增强功能)
- [7. 使用指南](#7-使用指南)
- [8. 与现有系统集成](#8-与现有系统集成)
- [9. 文件清单](#9-文件清单)
- [10. 已知限制与后续规划](#10-已知限制与后续规划)

---

## 1. 项目概述

函数工厂 GUI 是 SGN-Lite v5.1 的图形化扩展模块，提供一个基于 `pygame` 的桌面界面，允许用户：

1. **手绘 / 鼠标绘制**像素图形于网格画布上
2. **实时旋转变换**（旋转、偏移、缩放）生成多种变体 —— 移动适配的核心
3. **一键训练**：将当前基础图形及其变换变体送入 SGN 神经网络训练
4. **实时检验**：当前图形直接识别，显示匹配结果、置信度
5. **加载自定义图片**：从外部图片文件加载到画布
6. **导出自定义训练集**：将变体样本导出为 CSV/JSON
7. **内置字符库**：全部可打印 ASCII 字符（32-126），通过 pygame 字体渲染，覆盖所有网格大小
8. **撤销/重做**：最多 50 步操作历史，支持 Ctrl+Z / Ctrl+Y
9. **橡皮擦工具**：独立切换模式，左键擦除
10. **图案保存/加载**：将画布保存为 JSON 文件，下次可直接加载恢复
11. **保存图片**：将画布导出为 PNG 图片
12. **与现有系统无缝集成**：复用 `SGNCore`、`DefaultCompositeNoise`、`classify_sample` 等全部已有能力

---

## 2. 设计目标

| 目标 | 说明 |
|------|------|
| **移动适配** | 用户只需绘制一个基础图形，UI 负责旋转、偏移、缩放生成变体 |
| **窗口自适应** | 支持窗口大小调整，自动重新计算布局 |
| **大网格支持** | 支持 64×64 超大网格，满足高精度需求 |
| **低耦合** | GUI 作为独立模块，不修改 SGN 核心逻辑，通过 `SGNBridge` 桥接 |
| **可复用** | 支持复用已有 `SGNCore` 实例，训练状态不丢失 |
| **实时交互** | 训练在后台逐帧执行，不阻塞 GUI，实时显示进度 |
| **自定义训练集** | 支持图片加载、批量变体生成、CSV/JSON 导出 |
| **无阻塞对话框** | 字符库使用 pygame 覆盖层渲染，文件对话框复用持久 tkinter 根窗口 |
| **操作可逆** | 撤销/重做支持所有画布操作（绘制、清空、填充、加载、切换网格） |

---

## 3. 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                      GUIFactory (主窗口)                     │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐  │
│  │ DrawCanvas   │───→│TransformEngine│───→│  SGNBridge     │  │
│  │ (绘制画布)   │    │ (变换引擎)   │    │ (SGN 桥接)      │  │
│  └─────────────┘    └─────────────┘    └─────────────────┘  │
│         │                   │                      │           │
│         ▼                   ▼                      ▼           │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐      │
│  │ 鼠标事件    │    │ 变换预览    │    │ SGNCore.train() │      │
│  │ (Bresenham) │    │ (pygame)   │    │ SGNBridge.classify│     │
│  └─────────────┘    └─────────────┘    └─────────────────┘      │
│                                                                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐      │
│  │ load_image  │    │ export_csv  │    │ charlib (ASCII) │      │
│  │ (PIL/游戏)   │    │ export_json │    │ pygame overlay  │      │
│  └─────────────┘    └─────────────┘    └─────────────────┘      │
│                                                                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────┐      │
│  │ undo/redo   │    │ save/load   │    │ save_image      │      │
│  │ (50步历史)  │    │ pattern JSON│    │ (PNG export)    │      │
│  └─────────────┘    └─────────────┘    └─────────────────┘      │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
              ┌─────────────────────────────┐
│              │  CommandRegistry [f]       │
│              │  ExtensionMenu [e] → [g]    │
│              └─────────────────────────────┘
```

### 3.1 数据流

```
用户绘制 → DrawCanvas.data → to_intensity() → TransformEngine.apply()
                                                    ↓
                                    ┌──────────────┴──────────────┐
                                    ↓                              ↓
                              SGNCore.train()               SGNBridge.classify()
                              (训练样本)                     (实时测试)
                                    ↓                              ↓
                              history.append()              预测标签 + 匹配度
```

### 3.2 撤销/重做机制

```
操作触发 → _push_undo() → 快照存入 _undo_stack
                         → 清空 _redo_stack

撤销: _undo() → 从 _undo_stack 弹出 → 推入 _redo_stack → 恢复画布
重做: _redo() → 从 _redo_stack 弹出 → 推入 _undo_stack → 恢复画布
```

触发时机：笔画结束（鼠标松开）、清空画布、填充图案、加载图片/字符、切换网格大小、加载图案。

---

## 4. 模块拆分

函数工厂 GUI 从原来的单文件（约 1100 行）拆分为 9 个文件：

| 文件 | 职责 | 行数 |
|------|------|------|
| `gui/theme.py` | 主题、常量、窗口自适应、字体加载 | ~80 |
| `gui/utils.py` | 工具函数：intensity↔surface、图片加载、训练集导出、变换、对比度拉伸 | ~160 |
| `gui/canvas.py` | DrawCanvas：网格绘制、鼠标事件、Bresenham 连线、橡皮擦模式 | ~185 |
| `gui/transform.py` | TransformEngine：旋转、偏移、缩放、随机变体生成 | ~80 |
| `gui/ui.py` | UI 组件：Button、Slider、Label、TextBox | ~240 |
| `gui/bridge.py` | SGNBridge：连接 GUI 与 SGN 核心引擎 | ~90 |
| `gui/charlib.py` | 内置字符库：4×4/8×8 模板 + pygame 字体渲染（全 ASCII） | ~170 |
| `gui/factory.py` | GUIFactory：主窗口、布局自适应、事件循环、所有功能入口 | ~1050 |
| `gui/sgn_gui_factory.py` | 入口文件：sys.path 处理、run_factory() | ~40 |

---

## 5. 核心模块详解

### 5.1 DrawCanvas — 绘制画布

**位置**: `gui/canvas.py:DrawCanvas`

| 功能 | 说明 |
|------|------|
| 网格管理 | `data[y][x]` 二维数组存储像素值（0-255） |
| 鼠标绘制 | 左键绘制（255），右键擦除（0） |
| 橡皮擦模式 | `eraser_mode` 开启时左键变为擦除 |
| 连线绘制 | Bresenham 算法，防止鼠标移动过快跳过网格 |
| 多分辨率 | 支持 4×4 / 8×8 / 16×16 / 32×32 / 64×64，切换时缩放保留内容 |
| 快捷填充 | 十字、圆形、对角线、边框四种测试图案 |
| 图片加载 | `from_image(path, grid_size)` 从外部图片加载 |

**关键接口**:
```python
canvas.to_intensity()      # → List[int] (行优先，供 SGN 使用)
canvas.from_intensity()    # ← List[int] (从 SGN 加载已有图形)
canvas.from_image(path, gs) # ← 从图片文件加载
canvas.eraser_mode         # bool，橡皮擦开关
```

### 5.2 TransformEngine — 变换引擎

**位置**: `gui/transform.py:TransformEngine`

实现 **移动适配** 的核心：用户只画一个图形，引擎负责生成多种变体。

| 变换 | 参数范围 | 说明 |
|------|----------|------|
| 旋转 | -180° ~ 180° | 使用 pygame `transform.rotate()`（浮点运算） |
| X 偏移 | -gs/3 ~ gs/3 | 像素级水平偏移 |
| Y 偏移 | -gs/3 ~ gs/3 | 像素级垂直偏移 |
| 缩放 | 0.5x ~ 1.5x | 使用 pygame `transform.smoothscale()`（浮点运算） |

**关键接口**:
```python
engine.apply(intensity, grid_size)              # 应用当前参数
engine.randomize(grid_size)                     # 随机参数
generate_variants(intensity, gs, count)         # 批量生成变体
```

**渲染管线**（`apply_transform` in `gui/utils.py`）:
```
intensity → Surface → smoothscale(浮点) → rotate(浮点) → blit(居中+偏移) → Surface → intensity
```

> **注意**：输入层（变换、图片加载、对比度拉伸）涉及外部模拟信号，保留浮点运算。核心识别层（SGNCore）仍保持整数化，两者通过 `(intensity, label)` 接口联动，互不侵入。

### 5.3 SGNBridge — SGN 桥接

**位置**: `gui/bridge.py:SGNBridge`

支持两种初始化模式：

```python
# 模式 A：新建网络
bridge = SGNBridge(seed=42)

# 模式 B：复用已有网络（从 main.py 菜单进入）
bridge = SGNBridge(core=existing_core)
```

**关键接口**:

```python
bridge.train_step(intensity, label)    # → Dict (info)
bridge.classify(intensity)               # → (label, score)
bridge.history_length()                  # → int (训练步数)
bridge.model_save(path)                  # → bool (保存模型)
bridge.model_load(path)                  # → bool (加载模型)
bridge.reset(seed)                       # 重置网络
bridge.get_state()                       # → Dict (完整状态)
bridge.state_text()                      # → 状态字符串（供 UI 显示）
```

> **解耦设计**: GUI 层只通过 `SGNBridge` 接口访问 SGN 核心，不直接访问 `core.history`、`core.templates` 等内部属性。SGN 核心变化时只需更新 bridge。

### 5.4 GUIFactory — 主窗口

**位置**: `gui/factory.py:GUIFactory`

| 区域 | 位置 | 内容 |
|------|------|------|
| 左面板 | 0 ~ 45% 窗口宽度 | 画布 + 网格大小选择 + 工具按钮（两行） + 信息面板 |
| 右面板 | 45% ~ 100% 窗口宽度 | 变换滑块 + 预览窗口 + 标签选择 + 训练控制 + 导出 + 状态日志 |

**UI 组件清单 — 左面板工具栏（两行布局）**:

第一行（绘图工具）：

| 组件 | 功能 |
|------|------|
| 4/8/16/32/64 按钮 | 切换网格分辨率（含 64×64） |
| 清空按钮 | 清空画布 |
| 十字/圆形按钮 | 快捷填充测试图案 |
| 橡皮擦按钮 | 切换橡皮擦模式（黄色=关，蓝色=开） |

第二行（文件与操作）：

| 组件 | 功能 |
|------|------|
| 字符库按钮 | pygame 覆盖层选择全部可打印 ASCII 字符 |
| 加载图片按钮 | 文件对话框选择图片文件 |
| 撤销/重做按钮 | 操作历史回退/前进 |
| 保存图片按钮 | 画布导出为 PNG |
| 保存图案按钮 | 画布保存为 JSON 图案文件 |
| 加载图案按钮 | 从 JSON 图案文件恢复画布 |

**右面板组件**：

| 组件 | 功能 |
|------|------|
| 旋转 / X偏移 / Y偏移 / 缩放 滑块 | 实时控制变换参数 |
| 随机变换 / 重置变换 按钮 | 一键随机 / 归零 |
| 生成100变体按钮 | 批量生成 100 个变体到内存（供导出） |
| 变换预览窗口 | 实时显示变换后的图形 |
| 0-9 / A-F 标签按钮 | 选择训练标签 |
| 自定义标签输入框 | 输入任意标签字符 |
| 训练步数滑块 | 10 ~ 5000 步 |
| 开始训练 / 实时测试 / 重置网络 | 核心操作 |
| 导出CSV / 导出JSON | 将生成的变体保存为训练集文件 |
| 保存模型 / 加载模型 | 保存/恢复 SGN 模型（通过桥接层） |
| 训练集管理（添加/保存/加载/清空/多标签训练） | 自定义训练集管理 |
| 状态日志 | 显示训练进度、测试结果、网络状态 |

**布局自适应**：
- 窗口支持 `pygame.RESIZABLE`，大小变化时自动重新计算左/右面板比例
- 画布像素大小根据左面板可用空间自动计算
- 标签按钮大小根据右面板宽度自动调整
- 预览窗口大小根据右面板宽度自适应

### 5.5 字符库（CharLib）

**位置**: `gui/charlib.py`

支持两级字符生成：

1. **预定义模板**：0-9, A-F 的 4×4 位图模板，自动放大到 8×8，最近邻缩放到 16/32/64
2. **pygame 字体渲染**：对所有其他可打印 ASCII 字符（32-126），通过 pygame 字体渲染生成 intensity 列表

字符库 UI 使用 **pygame 覆盖层**（overlay），不创建 tkinter 窗口，不阻塞 pygame 事件循环。

```python
from gui.charlib import get_char, list_chars
intensity = get_char('A', grid_size=64)    # 预定义模板（优先）
intensity = get_char('@', grid_size=16)    # 字体渲染生成
chars = list_chars()  # 95 个可打印 ASCII 字符 (chr(32) ~ chr(126))
```

### 5.6 图片加载与训练集导出

**位置**: `gui/utils.py`

**图片加载**（保留浮点运算）：
```python
# 优先使用 PIL（支持任意格式），回退到 pygame.image
intensity = load_image_to_intensity(path, grid_size=64)
```

**训练集导出**：
```python
# 导出为 CSV 格式
export_training_dataset(samples, path, grid_size=16, format="csv")

# 导出为 JSON 格式
export_training_dataset(samples, path, grid_size=16, format="json")
```

### 5.7 图案保存/加载

**位置**: `gui/factory.py`

将画布保存为独立的 JSON 图案文件，方便下次加载恢复。

**保存格式**：
```json
{
  "name": "A",
  "grid_size": 16,
  "intensity": [0, 255, 255, 0, ...],
  "description": "标签=A 网格=16x16"
}
```

**操作**：
- 点击「保存图案」→ 弹出文件对话框 → 保存为 JSON
- 点击「加载图案」→ 弹出文件对话框 → 加载到画布（自动切换网格大小）

---

## 6. 增强功能

### 6.1 窗口大小自适应

- 窗口支持 `pygame.RESIZABLE` 标志
- `VIDEORESIZE` 事件触发时重新计算布局
- 左面板占 45% 宽度，右面板占 55% 宽度
- 画布像素大小根据可用空间自动计算：`max(4, min(32, avail_w // grid_size, avail_h // grid_size))`

### 6.2 64×64 超大网格

- `GRID_SIZES` 包含 4, 8, 16, 32, 64
- 64×64 时画布像素大小自动缩小到 4~8px，确保完整显示
- 预览窗口自动缩放
- 变换偏移范围自适应：`max_off = grid_size // 3`
- **切换网格时保留画布内容**：自动缩放已有像素数据到新网格大小

### 6.3 自定义图片加载

- 支持格式：PNG, JPG, JPEG, BMP, GIF（通过 PIL）
- 加载流程：
  1. 用户点击「加载图片」按钮
  2. 弹出 tkinter `filedialog.askopenfilename`（复用持久根窗口）
  3. PIL 读取 → 转为灰度 → `resize((gs, gs), LANCZOS)` → 归一化到 0-255
  4. 若未安装 PIL，回退到 `pygame.image.load`（支持 png/bmp）
  5. 加载到画布，自动更新预览
  6. **自动推入撤销栈**，可 Ctrl+Z 恢复

### 6.4 自定义训练集导出

- 用户点击「生成100变体」→ 批量生成 100 个随机变体（含噪声）到内存
- 用户点击「导出CSV」或「导出JSON」→ 弹出 `filedialog.asksaveasfilename`
- 默认文件名：`dataset_{label}_{gs}x{gs}.csv`
- 训练过程中也会自动累积样本到 `train_history`，训练完成后可直接导出

### 6.5 撤销/重做

- 最多保存 **50 步**操作历史
- 操作触发自动记录快照：笔画结束、清空画布、填充图案、加载图片/字符、切换网格大小、加载图案
- 快捷键：**Ctrl+Z** 撤销、**Ctrl+Y** 重做
- 工具栏按钮：「撤销」、「重做」
- 撤销/重做时保持橡皮擦模式状态

### 6.6 橡皮擦工具

- 工具栏「橡皮擦」按钮切换模式
- 快捷键：**E** 切换
- 开启时按钮变为蓝色（accent），关闭时黄色（warning）
- 左键在橡皮擦模式下执行擦除（0），绘制模式下执行绘制（255）
- 右键始终擦除

### 6.7 图案保存/加载

- 「保存图案」将画布保存为 JSON 文件（含标签、网格大小、像素数据）
- 「加载图案」从 JSON 文件恢复画布，自动切换网格大小并设置标签
- 文件格式与训练集不同：保存的是单个画布快照，不是训练样本集

### 6.8 保存图片

- 「保存图片」将画布导出为 PNG 图片
- 每个像素放大为 `max(4, min(32, 512 // gs))` 像素的方块
- 默认文件名：`canvas_{label}_{gs}x{gs}.png`

### 6.9 模型保存/加载

- 「保存模型」将 SGN 核心引擎状态保存为 JSON 文件
- 「加载模型」从 JSON 文件恢复引擎状态（神经元、模板、配置等）
- 默认文件名：`sgn_model_{label}_{steps}steps.json`
- 通过 `SGNBridge.model_save()` / `SGNBridge.model_load()` 实现，不直接访问 SGN 内部
- 保存/加载后状态栏自动更新网络信息

### 6.10 桥接层解耦

- GUI 层（factory.py）不再直接访问 `self.bridge.core.history` 等 SGN 内部属性
- 所有访问通过桥接层接口：`bridge.history_length()`、`bridge.model_save()`、`bridge.model_load()`
- SGN 核心变化时只需更新 `bridge.py`，GUI 层无需修改

---

## 7. 使用指南

### 7.1 启动方式

**方式一：直接运行**
```bash
python gui/sgn_gui_factory.py
```

**方式二：从 main.py 交互菜单启动**
```bash
python main.py
# 进入菜单后按 f 启动函数工厂
# 或进入控制面板 [e] 扩展功能 → [g] 函数工厂GUI
```

**方式三：代码中调用**
```python
from gui.sgn_gui_factory import run_factory
from sgn_core import SGNCore

core = SGNCore(seed=42)
run_factory(core=core)  # 复用已有网络
```

### 7.2 操作流程（单标签）

```
1. 选择网格大小（4/8/16/32/64，默认 16×16）
2. 在画布上绘制基础图形（左键绘制，右键擦除）
   或：点击「字符库」加载内置字符
   或：点击「加载图片」导入外部图片
   或：点击「加载图案」恢复之前保存的画布
3. 画错可按 Ctrl+Z 撤销，或按 E 切换橡皮擦模式
4. 调整变换参数（旋转/偏移/缩放）查看预览
5. 点击「随机变换」生成随机变体
6. 选择训练标签（0-9 / A-F，或自定义输入）
7. 设置训练步数（10 ~ 5000）
8. 点击「开始训练」或按空格键
9. 训练过程中可随时按 T 键实时测试
10. 点击「生成100变体」→「导出CSV/JSON」保存训练集
11. 点击「保存图案」保存画布供下次使用
12. 点击「保存图片」导出为 PNG
13. 按 ESC 退出
```

### 7.3 操作流程（多标签）

```
1. 绘制第一个图形（如字母 A）
2. 设置标签为 A，点击「添加到训练集」
3. 绘制第二个图形（如字母 B）
4. 设置标签为 B，点击「添加到训练集」
5. （可选）重复添加更多标签/图形变体
6. 点击「保存」训练集到 JSON 文件（供命令行导入）
7. 设置训练步数，点击「多标签训练」
8. 系统将循环所有标签，每个图形自动变换+噪声
9. 训练完成后按 T 实时测试任意图形
10. 按 ESC 退出
```

### 7.4 命令行导入训练集（CMD）

```bash
# 从 GUI 生成的训练集文件导入并训练
python main.py --custom-dataset gui/custom_dataset.json --batch --mode compact

# 配合其他参数使用
python main.py --custom-dataset my_shapes.json --batch --mode blackbox
```

### 7.5 快捷键

| 按键 | 功能 |
|------|------|
| `空格` | 开始 / 暂停训练 |
| `T` | 实时测试 |
| `R` | 随机变换 |
| `C` | 清空画布 |
| `E` | 切换橡皮擦模式 |
| `Ctrl+Z` | 撤销 |
| `Ctrl+Y` | 重做 |
| `ESC` | 退出 GUI |

---

## 8. 与现有系统集成

### 8.1 命令注册

在 `sgn_cmd_registry.py` 中保留 `[f] 函数工厂` 命令：

```python
Command("f", "函数工厂", "系统", _cmd_factory, requires_trained=False, order=7)
```

### 8.2 扩展功能菜单

在 `sgn_panel.py` 的 `_extension_menu` 中新增 `[g] 函数工厂GUI`：

```
[o] 保存模型    [d] 加载模型    [x] 导出报告
[w] 成功率查询  [b] 图表后端    [s] 存储后端
[a] 自动保存    [h] 钩子调试    [v] 噪声验证
[g] 函数工厂GUI
[q] 返回主菜单
```

### 8.3 命令行导入（main.py）

在 `main.py` 中新增 `--custom-dataset` 参数：

```bash
python main.py --custom-dataset gui/custom_dataset.json
python main.py --custom-dataset my_shapes.json --batch --mode compact
```

实现方式：
1. `build_parser()` 添加 `--custom-dataset` 参数
2. `apply_cli_args()` 设置 `INPUT_SOURCE_TYPE=custom` 和 `CUSTOM_DATASET_PATH`
3. `_prepare_samples()` 中 `custom` 分支使用 `CustomDatasetInputSource` 加载训练集

### 8.4 向后兼容

- **无侵入性**: GUI 模块不修改任何 SGN 核心文件（sgn_core, sgn_input 等）
- **可选依赖**: 若未安装 `pygame`，GUI 命令给出友好提示，不影响其他功能
- **状态复用**: 从 `main.py` 菜单进入时，当前训练的网络状态完整保留
- **路径自处理**: `gui/sgn_gui_factory.py` 自动将父目录加入 `sys.path`，无需额外配置

### 8.5 依赖模块

```python
from sgn_core import SGNCore              # 核心引擎（不修改）
from sgn_input import DefaultCompositeNoise  # 噪声模型（不修改）
from sgn_test import _classify             # 分类接口（不修改）
from sgn_config import CONFIG              # 配置读取（不修改）
```

---

## 9. 文件清单

| 文件 | 状态 | 说明 |
|------|------|------|
| `gui/theme.py` | **新增** | 主题、常量、窗口自适应、字体 |
| `gui/utils.py` | **新增** | 工具函数：转换、图片加载、训练集导出、变换、对比度拉伸 |
| `gui/canvas.py` | **新增** | 绘制画布：网格、鼠标事件、Bresenham、橡皮擦模式 |
| `gui/transform.py` | **新增** | 变换引擎：旋转、偏移、缩放、随机变体 |
| `gui/ui.py` | **新增** | UI 组件：Button、Slider、Label、TextBox |
| `gui/bridge.py` | **新增** | SGN 桥接：连接 GUI 与 SGN 核心（封装 history/save/load） |
| `gui/charlib.py` | **新增** | 内置字符库：4×4/8×8 模板 + pygame 字体渲染（全 ASCII） |
| `gui/dataset_store.py` | **新增** | 自定义训练集存储：多标签共存、JSON持久化、批量变体生成 |
| `gui/custom_input_source.py` | **新增** | 自定义输入源：供 SGN 核心直接使用 GUI 生成的训练集 |
| `gui/factory.py` | **新增** | 主窗口：布局自适应、多标签训练、训练集管理、撤销重做、图案保存 |
| `gui/sgn_gui_factory.py` | **新增** | 入口文件：sys.path 处理、run_factory() |
| `gui/sgn_gui_factory_documentation.md` | **新增** | 设计文档（本文档） |
| `sgn_panel.py` | **修改** | 在 `[e]` 扩展功能菜单新增 `[g]` 入口 |
| `sgn_cmd_registry.py` | **修改** | 保留 `[f]` 命令入口（导入路径更新） |
| `main.py` | **修改** | 新增 `--custom-dataset` CLI 参数，支持从命令行导入训练集 |

---

## 10. 已知限制与后续规划

### 10.1 已知限制

1. **图片加载依赖 PIL**: 若未安装 Pillow，回退到 pygame.image，仅支持 png/bmp
2. **旋转后锯齿**: 基于像素网格的离散变换，旋转后可能有轻微锯齿
3. **文件对话框仍阻塞**: tkinter 文件对话框（加载图片/导出/保存）在打开期间 pygame 窗口无响应（标准 OS 行为）
4. **撤销历史不持久化**: 关闭 GUI 后操作历史丢失

### 10.2 后续规划

| 功能 | 优先级 | 说明 |
|------|--------|------|
| ~~模型导出/导入~~ | ~~中~~ | ~~图形化保存/加载模型文件~~ ✅ 已实现 |
| 批量测试面板 | 中 | 在 GUI 中直接运行混淆矩阵、噪声测试 |
| 训练曲线可视化 | 中 | 在 GUI 中实时绘制准确率、损失曲线 |
| 多层笔刷/灰度绘制 | 低 | 支持不同灰度值的笔刷 |
| 撤销历史持久化 | 低 | 保存操作历史到文件 |

---

## 附录 A：核心类速查

### A.1 DrawCanvas

```python
class DrawCanvas:
    def __init__(self, x, y, pixel_size, grid_size=16)
    eraser_mode: bool              # 橡皮擦开关
    def resize(self, grid_size: int)
    def clear(self)
    def fill_test_pattern(self, pattern: str)  # "cross" / "circle" / "diagonal" / "border"
    def handle_event(self, event: pygame.event.Event)
    def draw(self, screen: pygame.Surface)
    def to_intensity(self) -> List[int]
    def from_intensity(self, intensity: List[int])
    def from_image(self, path: str, grid_size: int)
```

### A.2 TransformEngine

```python
class TransformEngine:
    angle: float
    offset_x: int
    offset_y: int
    scale: float

    def randomize(self, grid_size: int)
    def apply(self, intensity, grid_size) -> List[int]
    def generate_variants(intensity, grid_size, count) -> List[Tuple]
    def snapshot(self) -> Dict
    def restore(self, state: Dict)
```

### A.3 SGNBridge

```python
class SGNBridge:
    def __init__(self, seed=42, core=None)
    def reset(self, seed=None)
    def train_step(self, intensity, label) -> Dict
    def classify(self, intensity) -> Tuple[str, int]
    def history_length(self) -> int
    def model_save(self, path: str) -> bool
    def model_load(self, path: str) -> bool
    def get_state(self) -> Dict
    def state_text(self) -> str
```

### A.4 GUIFactory

```python
class GUIFactory:
    def __init__(self, core=None)
    def run(self)  # 启动主事件循环

    # 撤销/重做
    def _undo(self)
    def _redo(self)
    def _push_undo(self)

    # 橡皮擦
    def _toggle_eraser(self)

    # 图案保存/加载
    def _save_pattern(self)
    def _load_pattern(self)

    # 保存图片
    def _save_image(self)

    # 字符库覆盖层
    def _show_charlib_dialog(self)
    def _handle_charlib_overlay_event(self, event) -> bool
    def _draw_charlib_overlay(self)
```

### A.5 运行入口

```python
def run_factory(core=None):
    """启动函数工厂 GUI"""
```

### A.6 字符库

```python
from gui.charlib import get_char, list_chars
intensity = get_char('A', grid_size=64)    # 预定义模板
intensity = get_char('@', grid_size=16)    # 字体渲染
chars = list_chars()  # 95 个可打印 ASCII 字符
```

---

> **文档结束**
