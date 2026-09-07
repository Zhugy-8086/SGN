"""checkpoint.py - SGN 训练检查点保存/恢复（基础设施 B2，2026-08-29）

定位（infrastructure_roadmap_2026_08_29.md §5 B2）：
  完整训练状态 save/load/resume——参数 + Optimizer 状态 + step + 随机源状态，
  支持长验证运行（full mode 等）断点续训。替代训练脚本手写的参照 JSON dump。

格式：单 .npz 文件（numpy 原生，快速稳定）。命名约定：
  - 参数：      param.<name>
  - 优化器状态：opt.<field>          （lr / step / optimizer 类名）
                opt.state.<name>.<field>（动量/统计量）
  - 随机源：    rng.<name>.state     （np.random.Generator 的 bit_generator.state，dict）
  - 元数据：    __meta__             （json 字符串：step / seed / extra）
  - 标记：      __sgn_checkpoint__ = 1

使用：
    save_checkpoint(path, params=p, optimizer=opt, step=t, seed=7,
                    rngs={'roam': rng_roam, 'sr': rng_sr}, extra={...})
    ck = load_checkpoint(path)
    p = dict(ck['params']);  opt = SGD(p, **ck['optimizer']['defaults']);  opt.load_state_dict(ck['optimizer'])
    rng_roam = np.random.default_rng(); rng_roam.bit_generator.state = ck['rngs']['roam']
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Optional

import numpy as np

_META_KEY = "__meta__"
_MARK_KEY = "__sgn_checkpoint__"


def save_checkpoint(
    path: str,
    params: Dict[str, np.ndarray],
    optimizer=None,
    step: Optional[int] = None,
    seed: Any = None,
    rngs: Optional[Dict[str, np.random.Generator]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """保存检查点到 path（.npz 单文件）。返回 path。

    params: dict[str, ndarray]（或 nn.Module 的 state_dict）。
    optimizer: 带 state_dict()/load_state_dict() 的 Optimizer（见 optimizer.py）。
    rngs: {name: np.random.Generator}——保存 bit_generator.state，恢复后生成序列不中断。
    """
    arrs: Dict[str, np.ndarray] = {_MARK_KEY: np.array(1)}
    for k, v in params.items():
        arrs[f"param.{k}"] = np.asarray(v)

    meta = {"step": step, "seed": seed, "extra": extra or {}}
    if optimizer is not None:
        sd = optimizer.state_dict()
        meta["optimizer"] = sd["optimizer"]
        meta["lr"] = sd["lr"]
        meta["defaults"] = sd["defaults"]
        meta["opt_step"] = sd["step"]
        for name, st in sd["state"].items():
            for field, arr in st.items():
                arrs[f"opt.state.{name}.{field}"] = np.asarray(arr)
    if rngs:
        meta["rngs"] = {n: g.bit_generator.state for n, g in rngs.items()}
    arrs[_META_KEY] = np.array(json.dumps(meta, default=str))

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **arrs)
    return path


def load_checkpoint(path: str) -> Dict[str, Any]:
    """读取检查点，返回 dict：
      params: dict[str, ndarray]；optimizer: state_dict 或 None；
      step/seed/extra/rngs（rngs 为 {name: bit_generator.state}）。
    """
    z = np.load(path, allow_pickle=True)
    if _MARK_KEY not in z.files or int(z[_MARK_KEY]) != 1:
        raise ValueError(f"不是 SGN checkpoint: {path}")
    meta = json.loads(str(z[_META_KEY]))

    params = {k[len("param."):]: z[k] for k in z.files if k.startswith("param.")}
    out: Dict[str, Any] = {
        "params": params,
        "step": meta.get("step"),
        "seed": meta.get("seed"),
        "extra": meta.get("extra", {}),
        "rngs": meta.get("rngs"),
    }
    if "optimizer" in meta:
        sd = {
            "optimizer": meta["optimizer"],
            "lr": meta["lr"],
            "defaults": meta["defaults"],
            "step": meta["opt_step"],
            "state": {},
        }
        for k in z.files:
            if k.startswith("opt.state."):
                # opt.state.<name>.<field>——参数名本身可含点号（如 Sequential 的
                # 子模块路径 "0.weight"），故只从尾部切出最后一段为 field。
                key = k[len("opt.state."):]
                name, field = key.rsplit(".", 1)
                sd["state"].setdefault(name, {})[field] = z[k]
        out["optimizer"] = sd
    else:
        out["optimizer"] = None
    z.close()
    return out
