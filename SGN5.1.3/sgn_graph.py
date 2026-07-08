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
# sgn_graph.py - SGN-Lite v5.0 图数据结构
# 完整文件，无外部依赖（仅依赖 sgn_config.DiscreteCoordinate）

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set

from sgn_config import CONFIG, DiscreteCoordinate, SGNConstants


@dataclass
class GraphNode:
    """
    图节点 - 支持层级下压迁移与神经元投影溯源

    Attributes:
        node_id: 全局唯一ID
        layer: 当前所在层级 (0=L0底层, 1=L1, 2=L2, ...)
        source_image: 来源图ID (-1表示合并/投影节点)
        feature_vector: 特征向量，元素为DiscreteCoordinate
        position_norm: 归一化位置 (0~1000)，与grid_size解耦
        neighbors: 邻居节点ID -> 连接强度
        activation: 激活计数（赫布学习累积）
        last_updated: 最后更新步数
        label: 归属标签（仅合并节点有）
        layer_history: 曾驻留的所有层级列表（用于追溯遗忘路径）
        demotion_count: 被下压的次数
        source_neurons: 产生此节点的神经元ID列表（溯源用）
    """
    node_id: int
    layer: int
    source_image: int
    feature_vector: List[DiscreteCoordinate]
    position_norm: Tuple[int, int]
    neighbors: Dict[int, DiscreteCoordinate] = field(default_factory=dict)
    activation: int = 0
    last_updated: int = 0
    label: Optional[str] = None
    layer_history: List[int] = field(default_factory=list)
    demotion_count: int = 0
    source_neurons: List[int] = field(default_factory=list)

    def __post_init__(self):
        """初始化时自动记录当前层级到历史"""
        if not self.layer_history:
            self.layer_history = [self.layer]

    def to_hash_set(self) -> Set[int]:
        """将特征向量转为MinHash可处理的整数集合"""
        hash_set = set()
        for dc in self.feature_vector:
            # 编码: 高8位存level，低24位存index
            code = ((dc.level & 0xFF) << 24) | (dc.index & 0xFFFFFF)
            hash_set.add(code)
        return hash_set

    def get_minhash(self, num_hashes: int = 4) -> List[int]:
        """生成MinHash签名"""
        hash_set = self.to_hash_set()
        if not hash_set:
            return [0] * num_hashes
        primes = SGNConstants.MINHASH_PRIMES
        sigs = []
        for i in range(num_hashes):
            p = primes[i % len(primes)]
            min_h = min(((code * p) & 0xFFFFFFFF) for code in hash_set)
            sigs.append(min_h)
        return sigs

    def demote(self, target_layer: int) -> None:
        """
        将节点下压到指定层级，记录历史
        """
        if target_layer >= self.layer:
            return  # 只能下压到更低层
        self.layer = target_layer
        self.layer_history.append(target_layer)
        self.demotion_count += 1
        self.activation = 0  # 下压后重置激活，给予重新激活的机会

    def promote(self, target_layer: int) -> None:
        """
        将节点提升到更高层级（当它被重新激活时）
        """
        if target_layer <= self.layer:
            return
        self.layer = target_layer
        self.layer_history.append(target_layer)


@dataclass
class DynamicGraph:
    """
    动态图 - 每标签一个实例
    支持层级下压遗忘，无物理剪枝（删除仅发生在L0覆盖）
    """
    nodes: Dict[int, GraphNode] = field(default_factory=dict)
    next_id: int = 0
    _hash_index: Dict[int, Set[int]] = field(default_factory=dict)

    def add_node(self, node: GraphNode) -> int:
        """添加节点，自动分配ID并更新MinHash索引"""
        nid = self.next_id
        node.node_id = nid
        self.nodes[nid] = node
        for h in node.get_minhash():
            self._hash_index.setdefault(h, set()).add(nid)
        self.next_id += 1
        return nid

    def get_node(self, nid: int) -> Optional[GraphNode]:
        return self.nodes.get(nid)

    def remove_node(self, nid: int) -> bool:
        """
        删除节点（仅由L0覆盖机制调用）
        返回是否成功删除
        """
        if nid not in self.nodes:
            return False
        # 清理哈希索引
        for h, ids in list(self._hash_index.items()):
            ids.discard(nid)
            if not ids:
                del self._hash_index[h]
        del self.nodes[nid]
        return True

    def get_layer_nodes(self, layer: int) -> List[GraphNode]:
        """获取指定层级的所有节点"""
        return [n for n in self.nodes.values() if n.layer == layer]

    def get_max_layer(self) -> int:
        """获取当前最高层级"""
        if not self.nodes:
            return 0
        return max(n.layer for n in self.nodes.values())

    def find_candidates(self, query: GraphNode) -> Set[int]:
        """基于MinHash快速召回候选相似节点"""
        candidates = set()
        for h in query.get_minhash():
            candidates.update(self._hash_index.get(h, set()))
        return candidates

    @staticmethod
    def feature_similarity(
        vec1: List[DiscreteCoordinate],
        vec2: List[DiscreteCoordinate]
    ) -> int:
        """
        整数化特征相似度 (0-100)
        支持不同长度的向量比较
        """
        min_len = min(len(vec1), len(vec2))
        if min_len == 0:
            return 0

        # 统一到最高层级
        all_levels = []
        for i in range(min_len):
            all_levels.append(vec1[i].level)
            all_levels.append(vec2[i].level)
        target_level = max(all_levels) if all_levels else 0

        matches = 0
        for i in range(min_len):
            a = vec1[i].to_level(target_level)
            b = vec2[i].to_level(target_level)
            if a.index == b.index:
                matches += 1
        return (matches * 100) // min_len

    def find_best_match(
        self,
        query: GraphNode,
        threshold: int = 80
    ) -> Tuple[Optional[int], int]:
        """
        在图中查找与query最相似的节点
        返回: (匹配节点ID, 相似度) 或 (None, 0)
        """
        candidates = self.find_candidates(query)
        best_id = None
        best_sim = 0

        for cid in candidates:
            other = self.nodes.get(cid)
            if other is None or other.layer != query.layer:
                continue
            sim = self.feature_similarity(query.feature_vector, other.feature_vector)
            if sim > best_sim:
                best_sim = sim
                best_id = cid

        if best_sim < threshold:
            return None, best_sim
        return best_id, best_sim

    def hebbian_merge(
        self,
        source_node: GraphNode,
        step: int,
        lr_alpha_pct: int = 30
    ) -> bool:
        """
        赫布学习融合：将source_node融合到图中
        找到相似节点则进行特征加权平均，否则添加为新节点
        lr_alpha_pct: learning rate percentage (0-100), 30 means 30%
        返回: True=合并成功, False=新建节点
        """
        threshold = CONFIG.get("GRAPH_SIMILARITY_THRESHOLD", 80)
        alpha = lr_alpha_pct  # Direct use, no conversion needed

        # 查找最佳匹配
        best_id, best_sim = self.find_best_match(source_node, threshold)

        if best_id is not None:
            target = self.nodes[best_id]
            # 更新状态
            target.activation += source_node.activation + 1
            target.last_updated = step
            # 如果source有标签而target没有，继承标签
            if source_node.label and not target.label:
                target.label = source_node.label

            # 特征加权平均（整数化）
            min_len = min(len(target.feature_vector), len(source_node.feature_vector))
            for i in range(min_len):
                t_lvl = max(target.feature_vector[i].level, source_node.feature_vector[i].level)
                t_idx = target.feature_vector[i].to_level(t_lvl).index
                s_idx = source_node.feature_vector[i].to_level(t_lvl).index
                new_idx = (t_idx * (100 - alpha) + s_idx * alpha) // 100
                target.feature_vector[i] = DiscreteCoordinate(new_idx, t_lvl)

            # 如果source有更多维度，追加到target
            if len(source_node.feature_vector) > len(target.feature_vector):
                for i in range(len(target.feature_vector), len(source_node.feature_vector)):
                    target.feature_vector.append(source_node.feature_vector[i])

            return True
        else:
            # 添加为新节点
            new_node = GraphNode(
                node_id=-1,
                layer=source_node.layer,
                source_image=-1,
                feature_vector=list(source_node.feature_vector),
                position_norm=source_node.position_norm,
                label=source_node.label or "?",
                activation=source_node.activation + 1,
                last_updated=step,
                layer_history=[source_node.layer],
                source_neurons=source_node.source_neurons
            )
            self.add_node(new_node)
            return False

    def demote_layer(self, layer: int, threshold: int) -> int:
        """
        层级下压：将指定层中 activation < threshold 的节点下压到 layer-1
        返回被下压的节点数量
        """
        if layer <= 0:
            return 0

        to_demote = []
        for nid, node in self.nodes.items():
            if node.layer == layer and node.activation < threshold:
                to_demote.append(nid)

        for nid in to_demote:
            node = self.nodes[nid]
            node.demote(layer - 1)

        return len(to_demote)

    def demote_all(self, threshold: int) -> int:
        """
        从最高层到第1层逐级下压
        返回被下压的节点总数
        """
        max_layer = self.get_max_layer()
        total = 0
        for layer in range(max_layer, 0, -1):
            total += self.demote_layer(layer, threshold)
        return total

    def get_cover_candidates(self, current_step: int, max_age: int = 1000) -> List[int]:
        """
        获取L0层中可被覆盖的节点列表
        条件：在L0层且 last_updated + max_age < current_step
        """
        candidates = []
        for nid, node in self.nodes.items():
            if node.layer == 0 and (node.last_updated + max_age) < current_step:
                candidates.append(nid)
        return candidates

    def get_total_nodes(self) -> int:
        return len(self.nodes)

    def get_all_nodes(self) -> List[GraphNode]:
        return list(self.nodes.values())

    def to_dict(self) -> Dict:
        """序列化为字典（供存储层使用）"""
        return {
            "next_id": self.next_id,
            "nodes": {
                str(nid): {
                    "node_id": node.node_id,
                    "layer": node.layer,
                    "source_image": node.source_image,
                    "feature_vector": [(dc.level, dc.index) for dc in node.feature_vector],
                    "position_norm": node.position_norm,
                    "neighbors": {str(k): (v.level, v.index) for k, v in node.neighbors.items()},
                    "activation": node.activation,
                    "last_updated": node.last_updated,
                    "label": node.label,
                    "layer_history": node.layer_history,
                    "demotion_count": node.demotion_count,
                    "source_neurons": node.source_neurons
                }
                for nid, node in self.nodes.items()
            }
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "DynamicGraph":
        """从字典反序列化"""
        graph = cls()
        graph.next_id = data.get("next_id", 0)
        for nid_str, ndata in data.get("nodes", {}).items():
            node = GraphNode(
                node_id=ndata["node_id"],
                layer=ndata["layer"],
                source_image=ndata["source_image"],
                feature_vector=[
                    DiscreteCoordinate(idx, lvl)
                    for lvl, idx in ndata["feature_vector"]
                ],
                position_norm=tuple(ndata["position_norm"]),
                activation=ndata.get("activation", 0),
                last_updated=ndata.get("last_updated", 0),
                label=ndata.get("label"),
                layer_history=ndata.get("layer_history", [ndata["layer"]]),
                demotion_count=ndata.get("demotion_count", 0),
                source_neurons=ndata.get("source_neurons", [])
            )
            # 恢复邻居
            for k, (lvl, idx) in ndata.get("neighbors", {}).items():
                node.neighbors[int(k)] = DiscreteCoordinate(idx, lvl)
            graph.nodes[node.node_id] = node
            # 重建哈希索引
            for h in node.get_minhash():
                graph._hash_index.setdefault(h, set()).add(node.node_id)
        return graph
