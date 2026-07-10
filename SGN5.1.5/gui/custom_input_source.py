#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gui/custom_input_source.py — 自定义训练集输入源

继承自 sgn_input.InputSource，使 SGN 核心可以直接使用自定义训练集。
不修改任何 SGN 核心文件，纯扩展。
"""
from __future__ import annotations

from typing import List, Tuple, Optional

from engine.input import InputSource
from gui.dataset_store import CustomDatasetStore


class CustomDatasetInputSource(InputSource):
    """自定义训练集输入源

    从 CustomDatasetStore 中读取图形-标签对，
    生成训练样本时自动应用随机变换和噪声。
    """

    def __init__(
        self,
        dataset_store: CustomDatasetStore,
        noise_model=None,
        transform_range: Optional[dict] = None,
    ):
        self.dataset_store = dataset_store
        self.noise = noise_model
        self.transform_range = transform_range or {"angle": 15, "offset": 6, "scale": 0.2}

    def generate_batch(self, count: int, split: str = 'train') -> List[Tuple[List[int], str]]:
        """生成样本批次

        循环使用所有标签，每个图形应用随机变换 + 噪声。
        """
        def noise_fn(intensity):
            if self.noise:
                return self.noise.apply(intensity)
            return intensity

        return self.dataset_store.generate_samples(
            count, noise_fn=noise_fn, transform_range=self.transform_range
        )

    def get_label_distribution(self) -> dict:
        """返回标签分布统计"""
        from collections import Counter
        return dict(Counter(e["label"] for e in self.dataset_store.entries))

    def get_label_list(self) -> list:
        """返回所有标签"""
        return self.dataset_store.get_labels()

    @classmethod
    def from_file(cls, filepath: str, noise_model=None, transform_range: Optional[dict] = None):
        """从文件创建输入源（命令行快捷入口）"""
        store = CustomDatasetStore(filepath)
        store.load()
        return cls(store, noise_model, transform_range)
