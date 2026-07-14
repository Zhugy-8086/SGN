#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 [zhugy-8086]
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""SGN-Lite v5.0 持久化策略抽象 - StorageBackend / AutosaveStrategy

阶段5重构：解耦 JSON 硬编码，允许更换存储格式和自动保存策略。
"""

from __future__ import annotations

import os
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, Optional


# ============================================================
# StorageBackend - 存储后端抽象
# ============================================================

class StorageBackend(ABC):
    """存储后端抽象基类"""

    extension: str = ".json"
    name: str = ""

    @abstractmethod
    def serialize(self, core) -> str:
        """将引擎状态序列化为字符串（JSON/YAML 等）"""
        pass

    @abstractmethod
    def deserialize(self, data: str, core) -> bool:
        """从字符串恢复引擎状态"""
        pass

    @abstractmethod
    def save(self, core, path: str) -> bool:
        """保存到文件"""
        pass

    @abstractmethod
    def load(self, core, path: str) -> bool:
        """从文件加载"""
        pass


class JSONStorageBackend(StorageBackend):
    """JSON 存储后端（v4.1 默认行为）"""

    extension = ".json"
    name = "json"

    def serialize(self, core) -> str:
        """序列化引擎状态为 JSON 字符串"""
        from engine.config import CONFIG

        data = {
            "version": "5.0",
            "timestamp": datetime.now().isoformat(),
            "config": {k: (v.serialize() if hasattr(v, "serialize") else v) for k, v in CONFIG.items()},
            "input_dim": getattr(core, "D", CONFIG.get("D", 64)),
            "grid_size": getattr(core, "grid_size", 4),
            "seed": CONFIG.get("SEED", 42),
            "graph_mode": getattr(core, "graph_mode", False),
            "storage_backend": CONFIG.get("STORAGE_BACKEND", "json"),
            "neurons": [],
            "templates": [],
            "history_count": len(core.history),
            "label_set": sorted(getattr(core, "label_set", set())),
        }

        # 序列化神经元
        for n in core.N:
            data["neurons"].append({
                "T": n["T"],
                "base": n["base"].serialize() if hasattr(n["base"], "serialize") else n["base"],
                "lock": n["lock"],
                "enc_r": n["enc_r"],
                "enc_b": n["enc_b"].serialize() if hasattr(n["enc_b"], "serialize") else n["enc_b"],
                "L": n["L"],
            })

        # v5.1 多层神经元池序列化
        if hasattr(core, 'neuron_layers') and getattr(core, 'multi_layer_enabled', False):
            data["neuron_layers"] = {}
            for layer_id, pool in core.neuron_layers.items():
                data["neuron_layers"][str(layer_id)] = []
                for n in pool:
                    neuron_state = {
                        "T": n["T"],
                        "base": n["base"].serialize() if hasattr(n["base"], "serialize") else n["base"],
                        "lock": n["lock"],
                        "enc_r": n["enc_r"],
                        "enc_b": n["enc_b"].serialize() if hasattr(n["enc_b"], "serialize") else n["enc_b"],
                        "L": n["L"],
                        # v5.1.6: silenced/T_pending/merge_count 取代旧 gate 字段
                        "silenced": n.get("silenced", False),
                        "specialization": n.get("specialization"),
                        "consecutive_verified": n.get("consecutive_verified", 0),
                        "layer": n.get("layer", 0),
                        # Bug 10 修复：包含所有新字段
                        "T_features": [(dc.level, dc.index) for dc in n.get("T_features", [])],
                        "T_l0_active": n.get("T_l0_active", []),
                        "T_features_initialized": n.get("T_features_initialized", False),
                        "source_layer0": n.get("source_layer0", []),
                        # 补丁 P4：合并累积缓冲字段
                        "T_pending": n.get("T_pending"),
                        "merge_count": n.get("merge_count", 0),
                        # v5.1.7 L1 决策层字段
                        "output_pattern": n.get("output_pattern", []),
                        "verified_count": n.get("verified_count", 0),
                        "cluster_id": n.get("cluster_id"),
                        "cluster_verified_count": n.get("cluster_verified_count", 0),
                        "cluster_total_count": n.get("cluster_total_count", 0),
                    }
                    data["neuron_layers"][str(layer_id)].append(neuron_state)

        if core.graph_mode or getattr(core, 'multi_layer_enabled', False):
            # 图模式 / 多层模式：存储图结构摘要 + 完整图数据
            data["graphs_summary"] = {
                label: {
                    "node_count": g.get_total_nodes(),
                    "max_layer": g.get_max_layer(),
                }
                for label, g in core.graphs.items()
            }
            data["graphs_full"] = {
                label: g.to_dict()
                for label, g in core.graphs.items()
            }
        if not core.graph_mode and not getattr(core, 'multi_layer_enabled', False):
            # 模板模式：序列化模板
            for lb, tm, sc, hc in core.templates:
                data["templates"].append({
                    "label": lb,
                    "mask": tm,
                    "success": sc,
                    "hit_counter": hc,
                })

        # 序列化历史（限制最近1000步）
        data["history"] = core.history[-1000:]

        # 序列化黑箱验证记录
        data["blackbox_log"] = list(core.blackbox_log)

        # v5.1.6: 旧门控字段 atom_dict/high_templates/low_templates 已删除

        return json.dumps(data, indent=2, ensure_ascii=False)

    def deserialize(self, data: str, core) -> bool:
        """从 JSON 字符串恢复引擎状态"""
        from engine.config import CONFIG, DiscreteCoordinate

        try:
            obj = json.loads(data)
        except json.JSONDecodeError:
            return False

        # 恢复配置
        if "config" in obj:
            for k, v in obj["config"].items():
                if k in CONFIG:
                    # 离散坐标反序列化（v4.3）
                    if isinstance(v, dict) and "index" in v and "level" in v:
                        from engine.config import DiscreteCoordinate
                        CONFIG[k] = DiscreteCoordinate.deserialize(v)
                    # 兼容旧 FixedPoint 格式（value + scale）
                    elif isinstance(v, dict) and "value" in v and "scale" in v:
                        from engine.config import DiscreteCoordinate
                        CONFIG[k] = DiscreteCoordinate.from_fixedpoint_legacy(v["value"], v["scale"])
                    else:
                        CONFIG[k] = v

        # 【窗口大小识别】恢复维度锚定
        file_dim = obj.get("input_dim", obj.get("config", {}).get("D", 16))
        file_grid = int(file_dim ** 0.5)
        if file_grid * file_grid != file_dim:
            # 旧模型无 input_dim，从 config.D 回退
            file_dim = 16
            file_grid = 4

        # 若核心引擎已初始化且维度不匹配，强制重建
        if hasattr(core, "D") and core.D != file_dim:
            print(f"  [加载警告] 模型维度({file_dim})与当前网络({core.D})不匹配，重建中...")
            core.D = file_dim
            core.grid_size = file_grid
            core.N = []
            core.templates = []
            core.history = []
        elif not hasattr(core, "D"):
            core.D = file_dim
            core.grid_size = file_grid

        # 恢复神经元
        if "neurons" in obj:
            core.N = []
            mask_max = (1 << core.D) - 1
            for n in obj["neurons"]:
                t_list = n.get("T", [0] * CONFIG.get("LAYER_MAX", 4))
                # 防御：旧模型掩码可能位宽不足，截断/扩展至当前维度
                sanitized = []
                for t in t_list:
                    if isinstance(t, int):
                        sanitized.append(t & mask_max)  # 截断高位或保留低位
                    else:
                        sanitized.append(0)
                # 【v4.3-fix】旧模型可能存的是浮点/整数，需包装为 DiscreteCoordinate
                base_val = n.get("base", 0.5)
                if isinstance(base_val, dict) and "index" in base_val and "level" in base_val:
                    base_dc = DiscreteCoordinate.deserialize(base_val)
                elif isinstance(base_val, (int, float)):
                    base_dc = DiscreteCoordinate.from_float(float(base_val))
                else:
                    base_dc = DiscreteCoordinate(50, 2)
                enc_b_val = n.get("enc_b", 0)
                if isinstance(enc_b_val, dict) and "index" in enc_b_val and "level" in enc_b_val:
                    enc_b_dc = DiscreteCoordinate.deserialize(enc_b_val)
                elif isinstance(enc_b_val, (int, float)):
                    enc_b_dc = DiscreteCoordinate.from_float(float(enc_b_val))
                else:
                    enc_b_dc = DiscreteCoordinate(0, base_dc.level)
                core.N.append({
                    "T": sanitized,
                    "base": base_dc,
                    "lock": n.get("lock", 0),
                    "enc_r": n.get("enc_r", 0),
                    "enc_b": enc_b_dc,
                    "L": n.get("L", False),
                })

        # 恢复模板
        if "templates" in obj:
            core.templates = []
            for t in obj["templates"]:
                core.templates.append((
                    t.get("label", "?"),
                    t.get("mask", 0),
                    t.get("success", 1),
                    t.get("hit_counter", 128),
                ))

        # [v5.1] Rebuild template index
        if hasattr(core, '_template_index'):
            core._template_index = {}
            for i, (tlb, tm, sc, hc) in enumerate(core.templates):
                core._template_index.setdefault(tlb, []).append(i)

        # v5.1 多层神经元池反序列化
        if "neuron_layers" in obj and hasattr(core, 'neuron_layers'):
            core.multi_layer_enabled = True
            core.neuron_layers = {}
            for layer_id_str, pool_data in obj["neuron_layers"].items():
                layer_id = int(layer_id_str)
                pool = []
                for n_data in pool_data:
                    base_val = n_data.get("base", 0)
                    if isinstance(base_val, dict) and "index" in base_val and "level" in base_val:
                        base_dc = DiscreteCoordinate.deserialize(base_val)
                    else:
                        base_dc = DiscreteCoordinate(50, 2)
                    enc_b_val = n_data.get("enc_b", 0)
                    if isinstance(enc_b_val, dict) and "index" in enc_b_val and "level" in enc_b_val:
                        enc_b_dc = DiscreteCoordinate.deserialize(enc_b_val)
                    else:
                        enc_b_dc = DiscreteCoordinate(0, base_dc.level)
                    # v5.1.6: 旧 gate 字段已删除，兼容旧模型文件时忽略
                    n = {
                        "T": n_data.get("T", [0] * CONFIG.get("LAYER_MAX", 4)),
                        "base": base_dc,
                        "lock": n_data.get("lock", 0),
                        "enc_r": n_data.get("enc_r", 0),
                        "enc_b": enc_b_dc,
                        "L": n_data.get("L", False),
                        # v5.1.6: silenced/T_pending/merge_count 取代旧 gate 字段
                        "silenced": n_data.get("silenced", False),
                        "specialization": n_data.get("specialization"),
                        "consecutive_verified": n_data.get("consecutive_verified", 0),
                        "layer": n_data.get("layer", layer_id),
                        # Bug 10 修复：反序列化所有新字段
                        "T_features": [DiscreteCoordinate(idx, lvl) for lvl, idx in n_data.get("T_features", [])],
                        "T_l0_active": n_data.get("T_l0_active", []),
                        "T_features_initialized": n_data.get("T_features_initialized", False),
                        "source_layer0": n_data.get("source_layer0", []),
                        # 补丁 P4：合并累积缓冲字段
                        "T_pending": n_data.get("T_pending"),
                        "merge_count": n_data.get("merge_count", 0),
                        # v5.1.7 L1 决策层字段
                        "output_pattern": n_data.get("output_pattern", []),
                        "verified_count": n_data.get("verified_count", 0),
                        "cluster_id": n_data.get("cluster_id"),
                        "cluster_verified_count": n_data.get("cluster_verified_count", 0),
                        "cluster_total_count": n_data.get("cluster_total_count", 0),
                    }
                    pool.append(n)
                core.neuron_layers[layer_id] = pool
            # 向后兼容：self.N 指向 Layer 0
            core.N = core.neuron_layers.get(0, [])

        # 恢复历史
        if "history" in obj:
            core.history = obj["history"]

        # 恢复黑箱验证记录
        if "blackbox_log" in obj:
            core.blackbox_log = obj["blackbox_log"]
        else:
            core.blackbox_log = []

        # 恢复标签集
        if "label_set" in obj:
            core.label_set = set(obj["label_set"])
        else:
            # 兼容旧模型：从 templates 提取
            core.label_set = set(t[0] for t in core.templates)

        # 【v5.0 图模式 / v5.1 多层模式】恢复图结构
        graph_mode = obj.get("graph_mode", False)
        core.graph_mode = graph_mode
        core._step_counter = 0

        if "graphs_full" in obj and (graph_mode or getattr(core, 'multi_layer_enabled', False)):
            from graph.graph import DynamicGraph
            core.graphs = {}
            for label, g_data in obj["graphs_full"].items():
                core.graphs[label] = DynamicGraph.from_dict(g_data)
            # 确保模板为空（图模式/多层模式冻结模板系统）
            if graph_mode:
                core.templates = []
        elif not graph_mode:
            core.graphs = {}
        else:
            # 兼容：有 graph_mode 标记但无图数据，回退到模板模式
            core.graph_mode = False
            core.graphs = {}

        # v5.1.6: 旧门控字段 atom_dict/high_templates/low_templates 已删除
        # 兼容旧模型文件：忽略这些字段，不报错

        return True

    def save(self, core, path: str) -> bool:
        """原子写入：先写临时文件，成功后再重命名"""
        path = os.path.normpath(path)
        tmp_path = path + ".tmp"
        try:
            data_str = self.serialize(core)
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(data_str)
            os.replace(tmp_path, path)  # 原子重命名
            return True
        except Exception:
            # 清理临时文件
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass
            return False

    def load(self, core, path: str) -> bool:
        path = os.path.normpath(path)
        if not os.path.exists(path):
            return False
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = f.read()
            return self.deserialize(data, core)
        except Exception:
            return False


class SQLiteStorageBackend(StorageBackend):
    """SQLite 存储后端（可选扩展，适合大数据量历史记录）"""

    extension = ".db"
    name = "sqlite"

    def __init__(self):
        self._conn = None
        # 缓存 sqlite3 可用性，避免每次重复导入
        try:
            import sqlite3
            self._sqlite_available = True
        except ImportError:
            self._sqlite_available = False

    def _get_conn(self, path: str):
        """获取或创建数据库连接"""
        import sqlite3
        if self._conn is None:
            self._conn = sqlite3.connect(path)
        return self._conn

    def is_available(self) -> bool:
        return self._sqlite_available

    def serialize(self, core) -> str:
        """SQLite 不使用字符串序列化，返回空"""
        return ""

    def deserialize(self, data: str, core) -> bool:
        """SQLite 不使用字符串反序列化"""
        return False

    def save(self, core, path: str) -> bool:
        if not self.is_available():
            return False
        try:
            conn = self._get_conn(path)
            c = conn.cursor()
            # 创建表
            c.execute('''CREATE TABLE IF NOT EXISTS neurons (
                nid INTEGER PRIMARY KEY, T BLOB, base REAL,
                lock INTEGER, enc_r INTEGER, enc_b REAL, L INTEGER)''')
            c.execute('''CREATE TABLE IF NOT EXISTS templates (
                tid INTEGER PRIMARY KEY, label TEXT, mask INTEGER,
                success INTEGER, hit_counter INTEGER)''')
            c.execute('''CREATE TABLE IF NOT EXISTS history (
                hid INTEGER PRIMARY KEY, step_data TEXT)''')
            c.execute('''CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY, value TEXT)''')
            # v5.0 图结构表
            c.execute('''CREATE TABLE IF NOT EXISTS graph_nodes (
                label TEXT, nid INTEGER,
                layer INTEGER,
                feature_vector BLOB,
                pos_r INTEGER, pos_c INTEGER,
                activation INTEGER, last_updated INTEGER,
                neighbors BLOB,
                layer_history BLOB,
                demotion_count INTEGER,
                source_neurons BLOB,
                PRIMARY KEY (label, nid)
            )''')
            # 清除旧数据
            c.execute("DELETE FROM neurons")
            c.execute("DELETE FROM templates")
            c.execute("DELETE FROM history")
            c.execute("DELETE FROM graph_nodes")
            # 插入神经元
            for i, n in enumerate(core.N):
                import json as _json
                base_json = _json.dumps(n["base"].serialize() if hasattr(n["base"], "serialize") else n["base"])
                enc_b_json = _json.dumps(n["enc_b"].serialize() if hasattr(n["enc_b"], "serialize") else n["enc_b"])
                c.execute("INSERT INTO neurons VALUES (?,?,?,?,?,?,?)",
                         (i, _json.dumps(n["T"]), base_json, n["lock"], n["enc_r"], enc_b_json, int(n["L"])))
            # 插入模板
            for i, (lb, tm, sc, hc) in enumerate(core.templates):
                c.execute("INSERT INTO templates VALUES (?,?,?,?,?)",
                         (i, lb, tm, sc, hc))
            # 插入历史
            import json as _json
            for i, h in enumerate(core.history[-1000:]):
                c.execute("INSERT INTO history VALUES (?,?)",
                         (i, _json.dumps(h)))
            # 保存元数据
            c.execute("DELETE FROM meta WHERE key='label_set'")
            c.execute("INSERT INTO meta VALUES (?,?)",
                     ("label_set", _json.dumps(sorted(getattr(core, "label_set", set())))))
            c.execute("DELETE FROM meta WHERE key='graph_mode'")
            c.execute("INSERT INTO meta VALUES (?,?)",
                     ("graph_mode", str(getattr(core, "graph_mode", False))))
            c.execute("DELETE FROM meta WHERE key='D'")
            c.execute("INSERT INTO meta VALUES (?,?)",
                     ("D", str(getattr(core, "D", 16))))
            # 保存图结构（v5.0新增）
            if getattr(core, "graph_mode", False) and hasattr(core, "graphs") and core.graphs:
                for label, g in core.graphs.items():
                    for nid, node in g.nodes.items():
                        feat_blob = _json.dumps([(dc.level, dc.index) for dc in node.feature_vector])
                        neigh_blob = _json.dumps({str(k): (v.level, v.index) for k, v in node.neighbors.items()})
                        hist_blob = _json.dumps(node.layer_history)
                        src_blob = _json.dumps(node.source_neurons)
                        c.execute(
                            "INSERT INTO graph_nodes VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            (label, nid, node.layer, feat_blob,
                             node.position_norm[0], node.position_norm[1],
                             node.activation, node.last_updated,
                             neigh_blob, hist_blob, node.demotion_count, src_blob)
                        )
            conn.commit()
            return True
        except Exception:
            return False

    def load(self, core, path: str) -> bool:
        if not self.is_available() or not os.path.exists(path):
            return False
        try:
            conn = self._get_conn(path)
            c = conn.cursor()
            # 读取元数据
            meta = {}
            for row in c.execute("SELECT key, value FROM meta"):
                meta[row[0]] = row[1]
            graph_mode = meta.get("graph_mode", "False") == "True"
            core.D = int(meta.get("D", "16"))
            core.grid_size = int(core.D ** 0.5)
            # 恢复神经元
            core.N = []
            for row in c.execute("SELECT T, base, lock, enc_r, enc_b, L FROM neurons ORDER BY nid"):
                import json as _json
                base_raw = _json.loads(row[1]) if isinstance(row[1], str) else row[1]
                enc_b_raw = _json.loads(row[4]) if isinstance(row[4], str) else row[4]
                if isinstance(base_raw, dict) and "index" in base_raw and "level" in base_raw:
                    base_dc = DiscreteCoordinate.deserialize(base_raw)
                elif isinstance(base_raw, (int, float)):
                    base_dc = DiscreteCoordinate.from_float(float(base_raw))
                else:
                    base_dc = DiscreteCoordinate(50, 2)
                if isinstance(enc_b_raw, dict) and "index" in enc_b_raw and "level" in enc_b_raw:
                    enc_b_dc = DiscreteCoordinate.deserialize(enc_b_raw)
                elif isinstance(enc_b_raw, (int, float)):
                    enc_b_dc = DiscreteCoordinate.from_float(float(enc_b_raw))
                else:
                    enc_b_dc = DiscreteCoordinate(0, base_dc.level)
                core.N.append({
                    "T": _json.loads(row[0]),
                    "base": base_dc, "lock": row[2],
                    "enc_r": row[3], "enc_b": enc_b_dc,
                    "L": bool(row[5]),
                })
            # 恢复模板
            core.templates = []
            for row in c.execute("SELECT label, mask, success, hit_counter FROM templates ORDER BY tid"):
                core.templates.append((row[0], row[1], row[2], row[3]))
            # [v5.1] Rebuild template index
            if hasattr(core, '_template_index'):
                core._template_index = {}
                for i, (tlb, tm, sc, hc) in enumerate(core.templates):
                    core._template_index.setdefault(tlb, []).append(i)
            # 恢复历史
            core.history = []
            for row in c.execute("SELECT step_data FROM history ORDER BY hid"):
                import json as _json
                core.history.append(_json.loads(row[0]))
            # 恢复标签集
            try:
                import json as _json
                row = c.execute("SELECT value FROM meta WHERE key='label_set'").fetchone()
                if row:
                    core.label_set = set(_json.loads(row[0]))
                else:
                    core.label_set = set(t[0] for t in core.templates)
            except Exception:
                core.label_set = set(t[0] for t in core.templates)
            # 恢复图结构（v5.0）
            core.graph_mode = graph_mode
            core.graphs = {}
            core._step_counter = 0
            if graph_mode:
                from graph.graph import DynamicGraph, GraphNode
                import json as _json
                rows = c.execute("SELECT * FROM graph_nodes ORDER BY label, nid").fetchall()
                current_label = None
                current_graph = None
                for row in rows:
                    label, nid, layer, feat_blob, pos_r, pos_c, act, l_upd, neigh_blob, hist_blob, dem_count, src_blob = row
                    if label != current_label:
                        if current_graph is not None:
                            core.graphs[current_label] = current_graph
                        current_label = label
                        current_graph = DynamicGraph()
                    feat = [DiscreteCoordinate(idx, lvl) for lvl, idx in _json.loads(feat_blob)]
                    neighbors = {int(k): DiscreteCoordinate(idx, lvl) for k, (lvl, idx) in _json.loads(neigh_blob).items()}
                    layer_history = _json.loads(hist_blob) if hist_blob else [layer]
                    source_neurons = _json.loads(src_blob) if src_blob else []
                    node = GraphNode(
                        node_id=nid,
                        layer=layer,
                        source_image=-1,
                        feature_vector=feat,
                        position_norm=(pos_r, pos_c),
                        activation=act,
                        last_updated=l_upd,
                        neighbors=neighbors,
                        layer_history=layer_history,
                        demotion_count=dem_count,
                        source_neurons=source_neurons
                    )
                    current_graph.nodes[nid] = node
                    for h in node.get_minhash():
                        current_graph._hash_index.setdefault(h, set()).add(nid)
                if current_graph is not None:
                    core.graphs[current_label] = current_graph
                # 图模式下清空模板
                core.templates = []
            conn.close()
            self._conn = None
            return True
        except Exception:
            return False

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __del__(self):
        self.close()


# ============================================================
# AutosaveStrategy - 自动保存策略抽象
# ============================================================

class AutosaveStrategy(ABC):
    """自动保存策略抽象基类

    内置冷却时间：min_interval_seconds = 5
    """

    min_interval_seconds: float = 5.0

    def __init__(self):
        self._last_save_time = 0.0

    def _check_cooldown(self) -> bool:
        """检查冷却时间是否已过"""
        now = time.time()
        if now - self._last_save_time < self.min_interval_seconds:
            return False
        self._last_save_time = now
        return True

    @abstractmethod
    def should_save(self, step: int, core) -> bool:
        """判断当前是否应该保存"""
        pass

    @abstractmethod
    def get_path(self, step: int) -> str:
        """返回保存路径"""
        pass


class IntervalAutosave(AutosaveStrategy):
    """间隔自动保存（v4.1 默认：每50步）"""

    def __init__(self, interval: int = 50, max_backups: int = 3):
        super().__init__()
        self.interval = interval
        self.max_backups = max_backups
        self._base_path = "sgn_autosave"

    def should_save(self, step: int, core) -> bool:
        if step <= 0:
            return False
        if step % self.interval != 0:
            return False
        return self._check_cooldown()

    def get_path(self, step: int) -> str:
        return f"{self._base_path}.json"


class DeltaAutosave(AutosaveStrategy):
    """增量自动保存（当 templates 数量变化时保存）"""

    def __init__(self, min_interval: int = 50):
        super().__init__()
        self._last_template_count = 0
        self._step_counter = 0
        self.min_interval = min_interval

    def should_save(self, step: int, core) -> bool:
        self._step_counter += 1
        current_count = len(core.templates)
        if current_count != self._last_template_count:
            self._last_template_count = current_count
            if self._step_counter >= self.min_interval:
                self._step_counter = 0
                return self._check_cooldown()
        return False

    def get_path(self, step: int) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"sgn_autosave_delta_{ts}.json"


# ============================================================
# StorageRegistry - 存储注册表
# ============================================================

class StorageRegistry:
    """存储注册表"""

    _backends: Dict[str, StorageBackend] = {}
    _default: Optional[str] = None

    @classmethod
    def register(cls, name: str, backend: StorageBackend, default: bool = False) -> None:
        cls._backends[name] = backend
        if default or cls._default is None:
            cls._default = name

    @classmethod
    def get(cls, name: Optional[str] = None) -> StorageBackend:
        if name is None:
            name = cls._default
        if name not in cls._backends:
            raise KeyError(f"未找到存储后端: {name}")
        return cls._backends[name]

    @classmethod
    def list_backends(cls):
        return list(cls._backends.keys())

    @classmethod
    def clear(cls):
        cls._backends.clear()
        cls._default = None


# 自动注册默认后端
StorageRegistry.register("json", JSONStorageBackend(), default=True)

# 注册 SQLite 后端（如果 sqlite3 可用）
_sqlite_backend = SQLiteStorageBackend()
if _sqlite_backend.is_available():
    StorageRegistry.register("sqlite", _sqlite_backend)
