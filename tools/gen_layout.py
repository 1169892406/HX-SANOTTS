#!/usr/bin/env python3
"""把 sanoTTS 语音包的 manifest.json 压成 C 侧可零依赖解析的 layout.bin。

为什么需要这一步：上游语音包（HF `ampixa/sanoTTS/zh-xiaoya-1p55m`）的权重是
一个 `weights.fp16.bin` 大二进制 + `manifest.json` 里的「张量表」（名字/形状/
偏移）。在 C 里解析 43 KB 的 JSON 需要带一个 JSON 库，而 hOMEX 插件包越小越好
（`.hpkg` 单包 32 MiB、单文件 16 MiB，且每多一个第三方依赖就多一条安全面）。
所以本脚本在**构建期**把那张表转成一个几十 KB 的自描述二进制：

    magic "SNTL" | u32 version | u32 n_tensors
    每个张量：u8 name_len | name | u8 ndim | u32 dim[3] | u32 offset | u32 nbytes
    u32 n_kv；每个键值：u8 key_len | key | f32 value

张量名加上组件前缀（`duration/`、`acoustic/`、`decoder/`），因为三个组件各自
有 `embedding.weight`、`output.weight` —— 不加前缀在 C 侧就是重名。

用法：
    python3 tools/gen_layout.py <voice_dir> <out_layout.bin>
voice_dir 里要有 manifest.json 与 weights.fp16.bin（上游原样，不改字节）。
"""
from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

MAGIC = b"SNTL"
VERSION = 1
MAX_NAME = 255


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    voice_dir = Path(sys.argv[1])
    out_path = Path(sys.argv[2])
    manifest = json.loads((voice_dir / "manifest.json").read_text(encoding="utf-8"))
    weights = (voice_dir / "weights.fp16.bin").read_bytes()
    want = int(manifest["weights_size_bytes"])
    if len(weights) != want:
        raise SystemExit(f"weights 大小不符：{len(weights)} != {want}")

    rows: list[tuple[str, list[int], int, int]] = []
    for comp_name in ("duration", "acoustic", "decoder"):
        comp = manifest["components"][comp_name]
        for t in comp["tensors"]:
            name = f"{comp_name}/{t['name']}"
            if len(name.encode()) > MAX_NAME:
                raise SystemExit(f"张量名过长：{name}")
            shape = list(t["shape"])
            if len(shape) > 3:
                raise SystemExit(f"不支持的维度数：{name} {shape}")
            rows.append((name, shape, int(t["offset_bytes"]), int(t["nbytes"])))

    kv: list[tuple[str, float]] = [
        ("sample_rate", float(manifest["sample_rate"])),
        ("hop_length", float(manifest["hop_length"])),
        ("max_tokens", float(manifest["components"]["duration"]["config"]["max_tokens"])),
        ("max_duration", float(manifest["components"]["duration"]["config"]["max_duration"])),
    ]
    dec_cfg = manifest["components"]["decoder"]["config"]
    for key in ("post_filter_channels", "post_filter_layers", "post_filter_kernel",
                "post_filter_scale"):
        kv.append((key, float(dec_cfg.get(key) or 0)))

    blob = bytearray()
    blob += MAGIC
    blob += struct.pack("<II", VERSION, len(rows))
    for name, shape, offset, nbytes in rows:
        raw = name.encode()
        blob += struct.pack("<B", len(raw)) + raw
        dims = (shape + [0, 0, 0])[:3]
        blob += struct.pack("<BIII", len(shape), dims[0], dims[1], dims[2])
        blob += struct.pack("<II", offset, nbytes)
    blob += struct.pack("<I", len(kv))
    for key, value in kv:
        raw = key.encode()
        blob += struct.pack("<B", len(raw)) + raw + struct.pack("<f", value)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(bytes(blob))
    print(f"layout: {len(rows)} tensors, {len(kv)} kv, {len(blob)} bytes -> {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
