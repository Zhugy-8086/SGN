#!/usr/bin/env python3
"""下载测试参考数据（MNIST 原始 IDX 文件）。

源数据已从 legacy/ 迁至 engine/sgn/tests/refs/data/；若该目录缺失或损坏，
可运行本脚本重新下载标准 MNIST 数据集。

MNIST 数据来自 Yann LeCun 官方网站（http://yann.lecun.com/exdb/mnist/），
遵循其原始许可条款。CIFAR-10 等大型数据集未包含在仓库中，需 torchvision 自动下载。
"""

import os
import sys
import gzip
import urllib.request

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJ_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_DATA_DIR = os.path.join(_PROJ_ROOT, "engine", "sgn", "tests", "refs", "data", "MNIST", "raw")

_MNIST_FILES = {
    "train-images-idx3-ubyte.gz": "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz": "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz":  "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz":  "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz",
}


def _download_file(url: str, dest: str) -> None:
    print(f"  下载 {os.path.basename(dest)} ...")
    urllib.request.urlretrieve(url, dest)
    print(f"    完成 ({os.path.getsize(dest) / 1024:.1f} KB)")


def _gunzip(gz_path: str) -> None:
    out_path = gz_path.replace(".gz", "")
    if os.path.isfile(out_path):
        return
    print(f"  解压 {os.path.basename(gz_path)} ...")
    with gzip.open(gz_path, "rb") as f_in:
        with open(out_path, "wb") as f_out:
            f_out.write(f_in.read())
    print(f"    完成 ({os.path.getsize(out_path) / 1024:.1f} KB)")


def main() -> int:
    os.makedirs(_DATA_DIR, exist_ok=True)

    # 检查是否已有完整数据
    all_ok = all(
        os.path.isfile(os.path.join(_DATA_DIR, fname.replace(".gz", "")))
        for fname in _MNIST_FILES
    )
    if all_ok:
        print(f"MNIST 数据已存在: {_DATA_DIR}")
        return 0

    print(f"下载 MNIST 到: {_DATA_DIR}")
    for fname, url in _MNIST_FILES.items():
        gz_path = os.path.join(_DATA_DIR, fname)
        if not os.path.isfile(gz_path):
            _download_file(url, gz_path)
        _gunzip(gz_path)

    # 验证
    expected = [
        "train-images-idx3-ubyte",
        "train-labels-idx1-ubyte",
        "t10k-images-idx3-ubyte",
        "t10k-labels-idx1-ubyte",
    ]
    for fname in expected:
        path = os.path.join(_DATA_DIR, fname)
        if not os.path.isfile(path):
            print(f"  [ERROR] 缺失: {path}")
            return 1
        print(f"  ✓ {fname} ({os.path.getsize(path) / 1024:.0f} KB)")

    print("\n全部完成。")
    return 0


if __name__ == "__main__":
    sys.exit(main())