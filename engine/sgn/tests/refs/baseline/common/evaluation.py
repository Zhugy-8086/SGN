"""评估公共模块：各阶段 evaluate.py 共用的工具函数

提供 load_metrics / format_pct / format_delta，避免在三个 evaluate.py 中重复定义。
"""
import json
from pathlib import Path


def load_metrics(path: str, fallback: dict = None) -> dict:
    """加载 metrics JSON 文件

    Args:
        path: JSON 文件路径
        fallback: 文件不存在时的回退数据（可选）

    Returns:
        metrics 字典

    Raises:
        FileNotFoundError: 文件不存在且未提供 fallback
    """
    p = Path(path)
    if not p.is_file():
        if fallback is not None:
            print(f"  [fallback] {path} 不存在，使用历史数据")
            return fallback.copy()
        raise FileNotFoundError(f"metrics 文件不存在：{path}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def format_pct(x: float) -> str:
    """格式化百分比（0.97 → "97.00%"）"""
    return f"{100*x:.2f}%"


def format_delta(val: float, baseline_val: float, higher_better: bool = True) -> str:
    """格式化差异（带符号和百分比）

    Args:
        val: 新值
        baseline_val: 基准值
        higher_better: True=越高越好，False=越低越好

    Returns:
        格式化字符串，如 "+0.0123 (+1.26%) ✓"
    """
    delta = val - baseline_val
    pct = 100 * delta / abs(baseline_val) if baseline_val != 0 else 0.0
    sign = "+" if delta >= 0 else ""
    good = (delta >= 0) == higher_better
    marker = "✓" if good else "✗"
    return f"{sign}{delta:.4f} ({sign}{pct:.2f}%) {marker}"
