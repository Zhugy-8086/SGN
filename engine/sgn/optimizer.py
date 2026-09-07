"""optimizer.py - SGN 优化器封装（基础设施 B1，2026-08-29）

定位（infrastructure_roadmap_2026_08_29.md §5 B1）：
  统一 Optimizer 接口，替代训练脚本各自手写的参数更新循环
  （如 resnet8_free_roam 里 `vel[k] = momentum*vel[k] - lr*grad`）。

接口设计（极简 PyTorch 风格，贴合现有 dict-of-ndarray 训练脚本）：
  - params: dict[str, np.ndarray]（持有引用，不拷贝）；或 nn.Module（自动 state_dict 读写）
  - step(grads): dict[str, np.ndarray] 或 None（跳过）；按名字更新 params
  - state_dict()/load_state_dict(): 优化器状态（momentum/Adam 统计量），供 checkpoint（B2）
  - lr 可外部改（opt.lr = ...），支持学习率调度挂钩

覆盖范围（诚实定位）：
  本类覆盖"标准 float 更新"（FLOAT32 策略全部参数；fixed 策略的 bias/BN 部分）。
  MSint 的 q16 母本 + SR 舍入特殊更新路径（权重）属主攻方向核心，不纳入通用 Optimizer，
  由训练脚本保留（见 resnet8_free_roam 的 sr_update）。
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np


class Optimizer:
    """优化器基类。params 为 dict[str, ndarray] 或 nn.Module。"""

    def __init__(self, params, defaults: Dict[str, float]):
        if hasattr(params, "named_parameters"):
            # nn.Module 模式：经 state_dict/load_state_dict 读写
            self._module = params
            self._params = dict(params.state_dict())
        else:
            self._module = None
            self._params = params
        self._defaults = dict(defaults)
        self._state: Dict[str, dict] = {}
        self._step_count = 0
        self.lr = defaults.get("lr", 0.01)

    # -- 参数/梯度访问 ------------------------------------------------
    def _writeback(self) -> None:
        """Module 模式下把 numpy 参数写回（load_state_dict 含 shape 校验）。"""
        if self._module is not None:
            self._module.load_state_dict(self._params)

    def state_dict(self) -> dict:
        """优化器状态：优化器超参 + 每参数动量/统计量 + step 计数。

        快照语义：返回 numpy 数组的【拷贝】——调用方持有/落盘后，继续 step
        不会污染已保存状态（B2 checkpoint 依赖此保证）。
        """
        return {
            "optimizer": self.__class__.__name__,
            "lr": self.lr,
            "defaults": dict(self._defaults),
            "step": self._step_count,
            "state": {
                k: {kk: np.array(vv) for kk, vv in v.items()}
                for k, v in self._state.items()
            },
        }

    def load_state_dict(self, sd: dict) -> None:
        """从 state_dict 恢复（须与构造时相同参数集，否则抛 KeyError）。

        拷贝语义：恢复的动量/统计量是独立副本，不与源优化器共享缓冲。
        """
        self.lr = sd["lr"]
        self._defaults = dict(sd["defaults"])
        self._step_count = sd["step"]
        self._state = {
            k: {kk: np.array(vv) for kk, vv in v.items()}
            for k, v in sd["state"].items()
        }

    def zero_grad(self) -> None:
        """梯度清零占位：tape 的 clear/start_recording 由训练循环管理。"""
        pass

    def step(self, grads: Optional[Dict[str, np.ndarray]]) -> None:
        """更新一步。grads 为 {name: ndarray}；None 或无梯度名字的项跳过。"""
        raise NotImplementedError


class SGD(Optimizer):
    """随机梯度下降（momentum / weight_decay / dampening）。

    两种动量形式：
      classical=False（默认，PyTorch 式）：
        v = momentum*v + g ; p = p - lr*v
      classical=True（经典/速度式，SGN 训练脚本现状，如 resnet8_free_roam）：
        v = momentum*v - lr*g ; p = p + v
    两者在 momentum=0 时等价；classical=True 用于逐位复现现有训练循环。
    """

    def __init__(self, params, lr: float = 0.02, momentum: float = 0.0,
                 weight_decay: float = 0.0, dampening: float = 0.0,
                 classical: bool = False):
        super().__init__(params, {
            "lr": lr, "momentum": momentum,
            "weight_decay": weight_decay, "dampening": dampening,
            "classical": classical,
        })

    def step(self, grads: Optional[Dict[str, np.ndarray]]) -> None:
        if grads is None:
            return
        self._step_count += 1
        m = self._defaults["momentum"]
        wd = self._defaults["weight_decay"]
        damp = self._defaults["dampening"]
        classical = self._defaults["classical"]
        for name, p in self._params.items():
            g = grads.get(name)
            if g is None:
                continue
            if wd != 0.0:
                g = g + wd * p
            st = self._state.setdefault(name, {})
            if m != 0.0:
                if "v" not in st:
                    st["v"] = np.zeros_like(p)
                    buf = st["v"]
                    buf[:] = -self.lr * g if classical else g   # 首步
                else:
                    buf = st["v"]
                    if damp == 0.0:
                        buf[:] = m * buf + (g if not classical else -self.lr * g)
                    else:
                        buf[:] = m * buf + (1.0 - damp) * (g if not classical else -self.lr * g)
                if classical:
                    p += buf
                else:
                    p -= self.lr * buf
            else:
                p -= self.lr * g
        self._writeback()


class Adam(Optimizer):
    """Adam（betas / eps / weight_decay，含偏差修正）。

    更新（与 PyTorch 同式）：
      g = grad + weight_decay * p
      m = beta1*m + (1-beta1)*g ; v = beta2*v + (1-beta2)*g^2
      m_hat = m/(1-beta1^t) ; v_hat = v/(1-beta2^t)
      p = p - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, params, lr: float = 0.001, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.0):
        super().__init__(params, {
            "lr": lr, "betas": tuple(betas), "eps": eps,
            "weight_decay": weight_decay,
        })

    def step(self, grads: Optional[Dict[str, np.ndarray]]) -> None:
        if grads is None:
            return
        self._step_count += 1
        b1, b2 = self._defaults["betas"]
        eps = self._defaults["eps"]
        wd = self._defaults["weight_decay"]
        t = self._step_count
        for name, p in self._params.items():
            g = grads.get(name)
            if g is None:
                continue
            if wd != 0.0:
                g = g + wd * p
            st = self._state.setdefault(name, {})
            if "m" not in st:
                st["m"] = np.zeros_like(p)
                st["v"] = np.zeros_like(p)
            m, v = st["m"], st["v"]
            m[:] = b1 * m + (1.0 - b1) * g
            v[:] = b2 * v + (1.0 - b2) * (g * g)
            m_hat = m / (1.0 - b1 ** t)
            v_hat = v / (1.0 - b2 ** t)
            p -= self.lr * m_hat / (np.sqrt(v_hat) + eps)
        self._writeback()


class AdamW(Optimizer):
    """AdamW（解耦权重衰减，Loshchilov & Hutter 2019 式）。

    与 Adam 的区别：weight_decay 不混入梯度（不进 m/v 统计），而是
    更新前对参数直接收缩——p *= (1 - lr * weight_decay)：
      p = p * (1 - lr*wd)
      m = beta1*m + (1-beta1)*g ; v = beta2*v + (1-beta2)*g^2
      p = p - lr * m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, params, lr: float = 0.001, betas=(0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 0.01):
        super().__init__(params, {
            "lr": lr, "betas": tuple(betas), "eps": eps,
            "weight_decay": weight_decay,
        })

    def step(self, grads):
        if grads is None:
            return
        self._step_count += 1
        b1, b2 = self._defaults["betas"]
        eps = self._defaults["eps"]
        wd = self._defaults["weight_decay"]
        lr = self.lr
        t = self._step_count
        for name, p in self._params.items():
            g = grads.get(name)
            if g is None:
                continue
            st = self._state.setdefault(name, {})
            if "m" not in st:
                st["m"] = np.zeros_like(p)
                st["v"] = np.zeros_like(p)
            m, v = st["m"], st["v"]
            if wd != 0.0:
                p *= (1.0 - lr * wd)          # 解耦衰减：不污染 m/v
            m[:] = b1 * m + (1.0 - b1) * g
            v[:] = b2 * v + (1.0 - b2) * (g * g)
            m_hat = m / (1.0 - b1 ** t)
            v_hat = v / (1.0 - b2 ** t)
            p -= lr * m_hat / (np.sqrt(v_hat) + eps)
        self._writeback()
