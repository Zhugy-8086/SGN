# MSint demo 集合（engine/sgn/demo）

> 概念层（MSint 是什么）：demo_eight_worlds / demo_cube
> 能力层（MSint 能干嘛）：gallery/（本次主推）

---

## 快速开始

```bash
cd engine/sgn/build

# ── 画廊（能力层，主推）────────────────────────────
python ../demo/gallery/demo_gallery.py                  # 弹窗交互
python ../demo/gallery/demo_gallery.py --save           # 自动流程录 GIF+PNG

# ── 概念层 ──────────────────────────────────────────
python ../demo/demo_cube.py                             # 3D 双魔方（拖拽）
python ../demo/demo_cube.py --save                      # 存 GIF+PNG
python ../demo/demo_eight_worlds.py                     # 2D 视图变焦
python ../demo/demo_eight_worlds.py --save              # 存 PNG

# ── 数据/导出自检 ───────────────────────────────────
python ../demo/gallery/gallery_data.py                  # 母本+内核一致性
python ../demo/gallery/gallery_export.py                # 打包/投影校验
```

## 画廊操作

| 键/控件 | 动作 |
|---|---|
| 滑块 | 位宽 16→1bit，画面连续降级，粒子吞吐飙升 |
| Tab | 切换 [内存操作] ↔ [磁盘存储] |
| 1-5 | 直接切位宽（内存层） |
| 导出按钮 | 5 档 .raw 导出 + 体积柱状图（毫秒级，带进度） |
| 加载校验 | 16bit vs 4bit 投影一致性校验 |

## 目录

```
demo/
├── 画廊演示设计.md / 文件设计.md / 代码实现易错点.md   # 设计文档
├── demo_cube.* / demo_eight_worlds.*                   # 概念层（既有）
├── gallery/
│   ├── demo_gallery.py        # 画廊主程序
│   ├── gallery_data.py        # 母本生成（int16[1024]，固定种子可复现）
│   ├── gallery_export.py      # 导出/加载/投影一致性校验
│   ├── assets/                # 母本缓存 .npy（自动生成）
│   ├── exports/               # 5 档 .raw（运行生成）
│   └── output/                # gallery_demo.gif / .png（--save 产物）
└── logs/                      # 运行日志
```

## 依赖

- matplotlib + numpy + Pillow（--save 用）——全部已有，零新依赖
- sgn（编译内核，`sys.path` 指向 `../build`）