#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SGN-Lite v5.1.5 启动入口"""

import sys
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent))

from app.main import main

if __name__ == "__main__":
    main()
