#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
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
"""MSInt ABI 第二层：版本引擎

负责版本号的解析、比较和协商。

协商规则：
  - 主版本号（major）不兼容 → 拒绝服务（抛 VersionIncompatibleError）
  - 次版本号（minor）兼容 → 降级到请求方版本（取 min）
  - 修订号（patch）不参与协商（仅用于标识修复版本）

详见: 内部档案
"""

from __future__ import annotations

from typing import Tuple

from .protocol import VersionIncompatibleError


# ============================================================
# 当前 ABI 版本
# ============================================================

CURRENT_ABI_VERSION = "1.0.0"


# ============================================================
# 版本引擎
# ============================================================

class VersionEngine:
    """版本协商引擎

    版本号格式：MAJOR.MINOR.PATCH（语义化版本）

    协商规则：
      - 主版本号不一致 → 不兼容，抛异常
      - 次版本号不一致 → 向下兼容，取 min
      - 修订号不参与协商
    """

    CURRENT = CURRENT_ABI_VERSION

    @staticmethod
    def parse(version: str) -> Tuple[int, int, int]:
        """解析版本号字符串

        Args:
            version: 版本号字符串（如 "1.0.0", "1.2", "2"）

        Returns:
            (major, minor, patch) 元组

        Raises:
            ValueError: 版本号格式非法
        """
        if not isinstance(version, str):
            raise ValueError(
                f"版本号必须是 str，得到 {type(version).__name__}"
            )
        parts = version.split(".")
        if len(parts) < 1 or len(parts) > 3:
            raise ValueError(
                f"版本号格式非法：'{version}'，应为 MAJOR.MINOR.PATCH"
            )
        try:
            major = int(parts[0])
            minor = int(parts[1]) if len(parts) >= 2 else 0
            patch = int(parts[2]) if len(parts) >= 3 else 0
        except ValueError as e:
            raise ValueError(
                f"版本号格式非法：'{version}'，各段必须是整数"
            ) from e

        if major < 0 or minor < 0 or patch < 0:
            raise ValueError(
                f"版本号各段必须 >= 0，得到 {version}"
            )
        return major, minor, patch

    @staticmethod
    def is_compatible(requested: str, supported: str) -> bool:
        """检查版本兼容性

        Args:
            requested: 请求方版本
            supported: 支持方版本

        Returns:
            True 如果主版本号一致（次版本号向下兼容）
        """
        req_major, _, _ = VersionEngine.parse(requested)
        sup_major, _, _ = VersionEngine.parse(supported)
        return req_major == sup_major

    @staticmethod
    def negotiate(requested: str, supported: str) -> str:
        """协商版本号

        Args:
            requested: 请求方版本
            supported: 支持方版本

        Returns:
            协商后的版本号（MAJOR.MINOR.PATCH 格式）

        Raises:
            VersionIncompatibleError: 主版本号不兼容
        """
        req_major, req_minor, req_patch = VersionEngine.parse(requested)
        sup_major, sup_minor, sup_patch = VersionEngine.parse(supported)

        if req_major != sup_major:
            raise VersionIncompatibleError(
                f"主版本号不兼容：请求方 {requested}，支持方 {supported}"
            )

        # 次版本号取 min（向下兼容）
        negotiated_minor = min(req_minor, sup_minor)
        # 修订号取 min（保守策略）
        negotiated_patch = min(req_patch, sup_patch)

        return f"{req_major}.{negotiated_minor}.{negotiated_patch}"

    @staticmethod
    def compare(v1: str, v2: str) -> int:
        """比较两个版本号

        Args:
            v1: 版本号 1
            v2: 版本号 2

        Returns:
            -1 如果 v1 < v2
             0 如果 v1 == v2
             1 如果 v1 > v2
        """
        m1, n1, p1 = VersionEngine.parse(v1)
        m2, n2, p2 = VersionEngine.parse(v2)
        if (m1, n1, p1) < (m2, n2, p2):
            return -1
        if (m1, n1, p1) > (m2, n2, p2):
            return 1
        return 0

    @staticmethod
    def format(major: int, minor: int, patch: int = 0) -> str:
        """格式化版本号

        Args:
            major: 主版本号
            minor: 次版本号
            patch: 修订号

        Returns:
            版本号字符串 "MAJOR.MINOR.PATCH"
        """
        return f"{major}.{minor}.{patch}"
