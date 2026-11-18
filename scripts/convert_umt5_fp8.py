"""Convert a ComfyUI fp8 umt5-xxl text encoder into a diffusers/transformers bf16 encoder.

The keys in the ComfyUI ``*_umt5-xxl_fp8_scaled.safetensors`` file are plain HF T5 keys
(``encoder.block.N...``), and when the fp8 tensors are a direct e4m3 cast of the weights
(no per-tensor scale tensor in the file, magnitudes in the normal weight range),
dequantising is ``.to(bf16)``.

Output: ``<model_dir>/text_encoder/model-0000X-of-0000Y.safetensors`` + ``index.json``,
which ``WanPipeline.from_pretrained`` picks up like any other diffusers text encoder.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_DST = ROOT / "models" / "Wan2.1-T2V-1.3B-Diffusers" / "text_encoder"
SKIP = {"spiece_model", "scaled_fp8"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Convert fp8 umt5-xxl safetensors to sharded bf16")
    ap.add_argument(
        "--src",
        default=os.environ.get("UMT5_FP8_SRC", ""),
        help="path to source fp8 safetensors file (or set UMT5_FP8_SRC)",
    )
    ap.add_argument(
        "--dst",
        default=os.environ.get("UMT5_BF16_DST", str(DEFAULT_DST)),
        help="output text_encoder directory",
    )
    ap.add_argument("--shard-gb", type=float, default=2.0, help="target shard size in GiB")
    args = ap.parse_args(argv)

    if not args.src:
        ap.error("--src (or UMT5_FP8_SRC) is required")
    src = pathlib.Path(args.src)
    if not src.is_file():
        raise FileNotFoundError(f"source safetensors not found: {src}")
    dst = pathlib.Path(args.dst)
    shard_bytes = max(1, int(args.shard_gb * (1024 ** 3)))

    dst.mkdir(parents=True, exist_ok=True)
    with safe_open(str(src), framework="pt") as f:
        keys = [k for k in f.keys() if k not in SKIP]
        total_bytes = 0
        for k in keys:
            shape = f.get_slice(k).get_shape()
            n = 1
            for d in shape:
                n *= d
            total_bytes += n * 2
        n_shards = max(1, (total_bytes + shard_bytes - 1) // shard_bytes)
        index = {"metadata": {"total_size": total_bytes}, "weight_map": {}}

        cur: dict = {}
        cur_bytes = 0
        shard_no = 0
        for i, k in enumerate(keys):
            t = f.get_tensor(k)
            if t.dtype != torch.bfloat16:
                t = t.to(torch.bfloat16)
            cur[k] = t
            cur_bytes += t.numel() * 2
            last = i == len(keys) - 1
            # flush early enough to keep peak RAM around one shard
            if cur_bytes >= shard_bytes or last:
                shard_no += 1
                name = f"model-{shard_no:05d}-of-{n_shards:05d}.safetensors"
                save_file(cur, str(dst / name), metadata={"format": "pt"})
                for key in cur:
                    index["weight_map"][key] = name
                print(f"[convert] {name}: {cur_bytes / 1e9:.2f} GB ({len(cur)} tensors)", flush=True)
                cur, cur_bytes = {}, 0

    (dst / "model.safetensors.index.json").write_text(json.dumps(index, indent=2), encoding="utf-8")
    print(f"[convert] done: {len(index['weight_map'])} tensors, {total_bytes / 1e9:.2f} GB in {n_shards} shards")
    return 0


if __name__ == "__main__":
    sys.exit(main())
