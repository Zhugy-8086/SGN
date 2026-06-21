# SGN-Lite

Sparse Graph Network Lite — 纯整数化竞争/赫布学习识别系统

## 简介

SGN-Lite 是一个无浮点、无反向传播的轻量级识别引擎。核心机制为整数化竞争学习 + 模板匹配，适用于嵌入式 MCU 环境的算法验证。

**核心特性**：
- 全程整数运算（DiscreteCoordinate 离散坐标体系）
- 竞争学习 + 赫布增强/削弱
- 双图叠加门控识别（v4.4）
- 图模式层级记忆（v5.0）
- 策略插件化架构

## 快速开始

```bash
cd engine
python main.py
```

按回车进入训练，训练完成后可进行测试/可视化/噪声分析。

### 命令行参数

```bash
python main.py                          # 交互式训练
python main.py --batch                  # 批量模式
python main.py --auto 50                # 自动模式(50ms/步)
python main.py --mode compact           # 精简模式
python main.py --mode blackbox          # 黑箱模式
python main.py --config config.json     # 加载配置
python main.py --no-color               # 禁用颜色
```

## 架构

```
核心层 (Core)
  sgn_core.py          核心引擎（竞争/校验/学习/模板合并）

扩展层 (Extension)
  sgn_hooks.py         事件总线/钩子系统
  sgn_config.py        配置注册表/DiscreteCoordinate
  sgn_commands.py      命令注册表

策略层 (Strategy)
  sgn_input.py         输入管道/噪声模型/矢量渲染
  sgn_layers.py        层提取/边缘提取/分块编码
  sgn_strategies.py    层策略/校验策略/匹配策略
  sgn_metrics.py       评估指标（准确率/混淆/噪声鲁棒性）
  sgn_storage.py       存储后端（JSON/SQLite）
  sgn_graph.py         图数据结构（v5.0）
  sgn_stack.py         图构建与投影（v5.0）
  sgn_merge.py         跨图合并（v5.0）
  sgn_graph_match.py   图匹配与推理（v5.0）

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

## 运行模式

| 模式 | 说明 |
|------|------|
| full | 全记载，每步输出（默认） |
| compact | 精简，仅检查点摘要 |
| blackbox | 黑箱，训练全程零输出 |

## 输入源

| 类型 | 说明 |
|------|------|
| pattern | 内置 4×4 字符（0-F） |
| vector | 矢量图形（line/circle/sine/catear/mixed） |
| file | 从 CSV/JSON 文件加载 |
| 8×8 标准字符 | 0-9 + A-Z（STANDARD_CHARS_8x8） |

矢量图形支持网格大小：4/8/16/32/64。通过 `--vector-grid` 或控制面板 `[g]` 设置。

## 关键配置项

| 配置 | 默认值 | 说明 |
|------|--------|------|
| MAX_NEURONS | 256 | 神经元数量 |
| MAX_TEMPLATES | 500 | 模板库上限 |
| MAX_ITERATIONS | 100000 | 训练总步数 |
| SEED | 42 | 随机种子 |
| TOP_K | 6 | 竞争 Top-K |
| MAX_LOCKOUT | 120 | 锁定阈值 |
| ENABLE_GATE_MATCHING | False | 门控匹配（v4.4） |
| ENABLE_GRAPH_MODE | False | 图模式（v5.0） |

## 图模式（v5.0）

图模式在神经元之上挂接层级化记忆系统，解决模板识别的三个固有缺陷：
1. 层级缺失 → 多层图结构
2. 空间结构丢失 → 连通域 + 位置归一化
3. 过拟合倾向 → 一致性过滤 + 反馈循环

```bash
# 启用图模式
# 在控制面板 → 高级选项 中设置 ENABLE_GRAPH_MODE = True
# 或通过配置文件
```

核心机制：
- 神经元竞争 → 投影为图节点
- 多视图合并 → 一致性过滤（高层节点 ≥2 投影）
- 反馈迭代 → 误差图重新扫描
- 层级下压 → 遗忘机制

## 调参指南

基于实测数据：

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| 神经元 | 200-256 | 64 不够，再往上边际递减 |
| 训练步数 | 3000-5000 | 1万步后收益有限 |
| 模板上限 | 100-150 | 稳态约 80 个 |
| 随机种子 | 需要搜索 | 不同种子 = 完全不同的网络 |

## 评估指标

训练完成后可用的测试：
- `[t]` 批量测试：识别率
- `[c]` 混淆矩阵：各类别准确率
- `[n]` 噪声测试：复合/高斯/椒盐/块遮挡鲁棒性
- `[s]` 统计信息：神经元/模板/校验通过率
- `[g]` 仪表盘：综合状态

## 版本历史

- v4.3：长周期参数重构、Bug修复、8×8标准字符库
- v4.4：双图叠加门控识别、边缘提取、分块编码
- v5.0：图模式层级记忆（sgn_graph/stack/merge/graph_match）
- v5.0-fix：矢量圆距离计算Bug修复、渲染渐变方向修复
- v5.0-vis：可视化增强（10级色阶/8级热力图/模板强度模式/形状标注）
- v5.0-catear：新增猫耳矢量图形（4朝向×3间距×3半径×3开角×2形态=216种）
- v5.0-grid：VECTOR_GRID 扩展支持 64×64

## 依赖

- Python 3.10+
- 可选：matplotlib（图表导出）
- 可选：sqlite3（SQLite存储后端）

## 许可证

Apache License 2.0
