# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 zhugy-8086
"""画廊演示 · 磁盘层：按位宽导出 / 加载 / 投影一致性校验

职责
----
1. export(master, bits, path) —— 按位宽视图紧密打包写 .raw
   文件大小 = 母本 × bits/16（16bit:2KB / 8bit:1KB / 4bit:0.5KB / 2bit:0.25KB / 1bit:0.128KB）
2. load(path, bits, master) —— 读回并按"量化写回"语义覆盖母本区段
   （母本降精度到该位宽 = 部署降精度，feature 不是 bug，易错点 E4）
3. projection_consistent —— 投影一致性校验：16bit 导出在 view(bits) 下
   与 bits-bit 导出逐位一致（两个不同大小的文件，在公共视图层是同一份数据）
   —— 这是"单拷贝"在存储层的正确证据，易错点 E1/E2

打包格式（MSB-first，与 view() 高 N 位语义一致，易错点 E3）
----
   bits=16: 每值 2 字节（little-endian）
   bits=8 : 每值 1 字节
   bits=4 : 每 2 值 1 字节  [a(高4)|b(低4)]
   bits=2 : 每 4 值 1 字节  [a b c d 各2位]
   bits=1 : 每 8 值 1 字节  [packbits big-endian]

用法
----
   cd engine/sgn/build
   python ../demo/gallery/gallery_export.py        # 自测：roundtrip + 投影校验
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'build'))

# Windows 管道/重定向统一 UTF-8（避免 GBK 混编码，易错点 S1）
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

import gallery_data as G

BITS_VALID = G.BITS_VALID
_BYTES_PER_GROUP = {16: None, 8: 1, 4: 2, 2: 4, 1: 8}


def export_size(n: int, bits: int) -> int:
    """紧密打包后的文件大小（字节）= ceil(n × bits / 8)。"""
    assert bits in BITS_VALID
    return (n * bits + 7) // 8


def pack_view(values: np.ndarray, bits: int) -> bytes:
    """视图值（[0, 2^bits)）紧密打包为字节串，MSB-first。"""
    assert bits in BITS_VALID
    n = len(values)
    if bits == 16:
        return values.astype("<u2").tobytes()
    if bits == 8:
        return values.astype(np.uint8).tobytes()
    g = values.astype(np.uint32).reshape(-1, _BYTES_PER_GROUP[bits])
    acc = np.zeros(g.shape[0], np.uint8)
    for i in range(_BYTES_PER_GROUP[bits]):
        acc = ((acc.astype(np.uint32) << bits) | g[:, i]).astype(np.uint8)
    return acc.tobytes()


def unpack_bytes(raw: bytes, bits: int, n: int) -> np.ndarray:
    """字节串拆回视图值（长度 n），MSB-first。"""
    assert bits in BITS_VALID
    assert len(raw) == export_size(n, bits), \
        f"字节数不符: 期望 {export_size(n, bits)} 实际 {len(raw)}"
    if bits == 16:
        return np.frombuffer(raw, dtype="<u2").astype(np.uint16)
    if bits == 8:
        return np.frombuffer(raw, dtype=np.uint8).astype(np.uint16)
    acc = np.frombuffer(raw, dtype=np.uint8).astype(np.uint16)
    out = np.zeros(n, np.uint16)
    k = _BYTES_PER_GROUP[bits]
    mask = (1 << bits) - 1
    for i in range(k):
        shift = bits * (k - 1 - i)          # MSB-first：第一个值在高位
        out[i::k] = (acc >> shift) & mask
    return out


def export(master: np.ndarray, bits: int, path: str) -> int:
    """把母本的 bits-bit 视图紧密打包写入 path，返回文件字节数。"""
    vals = G.view(master, bits)
    raw = pack_view(vals, bits)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "wb") as f:
        f.write(raw)
    return len(raw)


def load(path: str, bits: int, master: np.ndarray) -> np.ndarray:
    """读回 bits-bit 导出并"量化写回"母本（高 bits 位生效，低位清零）。

    返回新的母本。写回后所有视图共享新的量化母本（单拷贝语义不变）。
    """
    with open(path, "rb") as f:
        raw = f.read()
    vals = unpack_bytes(raw, bits, len(master))
    return (vals.astype(np.uint32) << (16 - bits)).astype(np.uint16)


def projection_consistent(va: np.ndarray, bits_a: int,
                          vb: np.ndarray, bits_b: int) -> tuple[bool, int]:
    """投影一致性校验（存储层"同一份数据"的证据）。

    两份独立导出的视图值，在公共视图层（min 位宽）逐位一致：
        高 bit 导出在 view(lo) 下 == 低 bit 导出本身。
    返回 (是否一致, 首个不一致索引或 -1)。
    """
    lo = min(bits_a, bits_b)
    va_lo = (va.astype(np.uint32) >> (bits_a - lo)).astype(np.uint16)
    vb_lo = (vb.astype(np.uint32) >> (bits_b - lo)).astype(np.uint16)
    diff = np.nonzero(va_lo != vb_lo)[0]
    return (len(diff) == 0, -1 if len(diff) == 0 else int(diff[0]))


def projection_consistent_files(path_a: str, bits_a: int,
                                path_b: str, bits_b: int,
                                n: int) -> tuple[bool, int]:
    """两个磁盘文件（.raw）的投影一致性校验。"""
    with open(path_a, "rb") as f:
        va = unpack_bytes(f.read(), bits_a, n)
    with open(path_b, "rb") as f:
        vb = unpack_bytes(f.read(), bits_b, n)
    return projection_consistent(va, bits_a, vb, bits_b)


def _main() -> int:
    demo, pi = G.load_or_gen_masters()
    n = len(demo)
    print("1) 打包-拆回 roundtrip（view → pack → unpack == view）：")
    ok_rt = True
    for b in BITS_VALID:
        v = G.view(demo, b)
        rt = unpack_bytes(pack_view(v, b), b, n)
        same = np.array_equal(rt, v)
        ok_rt = ok_rt and same
        print(f"   bits={b:>2}: 大小={export_size(n, b):>5}B  值一致={same}")

    print("2) 导出文件大小（理论 = n×bits/8）：")
    tmp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "exports")
    ok_size = True
    for b in BITS_VALID:
        p = os.path.join(tmp, f"{b}bit.raw")
        size = export(demo, b, p)
        ok = size == export_size(n, b)
        ok_size = ok_size and ok
        print(f"   bits={b:>2}: 实际={size}B  理论={export_size(n, b)}B  一致={ok}")

    print("3) 投影一致性（同一母本导出的不同位宽文件，公共视图逐位一致）：")
    ok_proj = True
    paths = {}
    for b in BITS_VALID:
        paths[b] = os.path.join(tmp, f"{b}bit.raw")
        export(demo, b, paths[b])
    for b in BITS_VALID:
        same, idx = projection_consistent_files(paths[16], 16, paths[b], b, n)
        ok_proj = ok_proj and same
        print(f"   {paths[16]} vs {os.path.basename(paths[b])}: 一致={same}"
              + ("" if same else f" 首异@{idx}"))

    print("4) 反例（不同母本的文件应不一致）：")
    p_demo, p_pi = paths[4], os.path.join(tmp, "pi_4bit.raw")
    export(pi, 4, p_pi)
    same, idx = projection_consistent_files(p_demo, 4, p_pi, 4, n)
    ok_proj = ok_proj and (not same)
    print(f"   DEMO-4bit vs PI-4bit: 一致={same}（期望 False）" + ("" if not same else "  FAIL"))

    print("5) 量化写回（load 后母本 = 高 bits 位生效，低位清零）：")
    p4 = os.path.join(tmp, "4bit.raw")
    m_new = load(p4, 4, demo.copy())
    hi4 = (demo.astype(np.uint32) >> 12).astype(np.uint16)
    expected = (hi4.astype(np.uint32) << 12).astype(np.uint16)
    print(f"   load(4bit) 后高4位保留={np.array_equal((m_new>>12).astype(np.uint16), hi4)}"
          f"  低位清零={np.array_equal(m_new, expected)}")

    ok = ok_rt and ok_size and ok_proj and not same
    print("结论:", "全部通过" if ok else "存在失败！")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_main())