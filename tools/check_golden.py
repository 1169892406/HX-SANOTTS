#!/usr/bin/env python3
"""把 C 内核的输出与 numpy 参考实现逐样本比对（改任何一行推理代码都要跑）。

    python3 tools/check_golden.py [--numpy-path DIR]

`--numpy-path` 指向一个能 import numpy 的目录（本仓**不带** numpy；跑校验时才需要）。
不给时依次尝试：环境变量 PYTHONPATH、`~/.cache/hx-sanotts/numpy`。

判定：逐句相关系数 ≥ 0.99，且样本数完全相等。样本数不等说明时长整数化
（exp → clamp(1) → ×scale → round-half-even → clamp(max)）与上游不一致 ——
那是最难从"听起来差不多"里发现的一类错误，所以要单独判。
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import wave
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def load_numpy(paths: list[str]):
    for p in paths:
        if p and Path(p).is_dir():
            sys.path.insert(0, p)
    try:
        import numpy  # noqa: PLC0415
        return numpy
    except ImportError:
        return None


def pearson(a, b) -> float:
    ma = a.mean()
    mb = b.mean()
    xa = a - ma
    xb = b - mb
    denom = (xa @ xa) ** 0.5 * (xb @ xb) ** 0.5
    return float((xa @ xb) / denom) if denom else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--numpy-path", action="append", default=[])
    args = ap.parse_args()
    np = load_numpy(args.numpy_path + [os.environ.get("HX_SANOTTS_NUMPY", ""),
                                      str(Path.home() / ".cache" / "hx-sanotts" / "numpy")])
    if np is None:
        print("跳过：找不到 numpy（用 --numpy-path 指一个装过 numpy 的目录再跑）", file=sys.stderr)
        return 0

    lib_path = HERE / "runtime" / "libsanotts.so"
    if not lib_path.is_file():
        print(f"缺少推理库 {lib_path}，先按 BUILD.md 第 1 步编译", file=sys.stderr)
        return 1
    model = HERE / "runtime" / "models" / "zh"
    lib = ctypes.CDLL(str(lib_path))
    lib.snt_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
    lib.snt_load.restype = ctypes.c_int
    lib.snt_synth.argtypes = [
        ctypes.POINTER(ctypes.c_int32), ctypes.c_int, ctypes.c_float,
        ctypes.POINTER(ctypes.POINTER(ctypes.c_float)),
        ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
    ]
    lib.snt_synth.restype = ctypes.c_int
    lib.snt_free.argtypes = [ctypes.c_void_p]
    lib.snt_error.restype = ctypes.c_char_p
    if lib.snt_load(str(model / "layout.bin").encode(), str(model / "weights.fp16.bin").encode()) != 0:
        print(f"装载失败：{lib.snt_error().decode()}", file=sys.stderr)
        return 1

    rows = json.loads((HERE / "testdata" / "golden" / "golden.json").read_text(encoding="utf-8"))
    ok = True
    for i, row in enumerate(rows):
        ids = (ctypes.c_int32 * len(row["ids"]))(*row["ids"])
        pcm = ctypes.POINTER(ctypes.c_float)()
        n = ctypes.c_int()
        frames = ctypes.c_int()
        if lib.snt_synth(ids, len(row["ids"]), ctypes.c_float(1.0), ctypes.byref(pcm),
                         ctypes.byref(n), ctypes.byref(frames)) != 0:
            print(f"[{i}] 合成失败：{lib.snt_error().decode()}", file=sys.stderr)
            ok = False
            continue
        got = np.ctypeslib.as_array(pcm, shape=(n.value,)).astype(np.float64).copy()
        lib.snt_free(pcm)
        with wave.open(str(HERE / "testdata" / "golden" / f"ref{i}.wav"), "rb") as w:
            ref = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float64) / 32767.0
        if len(got) != len(ref):
            print(f"[{i}] 样本数不符：C={len(got)} 参考={len(ref)}  ← 时长整数化与上游不一致")
            ok = False
            continue
        corr = pearson(got, ref)
        flag = "OK " if corr >= 0.99 else "BAD"
        print(f"[{i}] {flag} corr={corr:.6f} samples={len(got)} frames={frames.value}  {row['text']}")
        if corr < 0.99:
            ok = False
    print("全部通过" if ok else "存在不一致", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())