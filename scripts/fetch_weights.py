"""Resumable multi-connection downloader for large Hugging Face files.

Single-stream HF downloads stall at ~0.5 MB/s on this network; 16 parallel range
requests give ~1 MB/s and, more importantly, survive per-connection stalls because
every chunk retries independently.

Usage:
    python scripts/fetch_weights.py --repo stdstu123/Yume-5B-720P \
        --dest models/Yume-5B-720P --files diffusion_pytorch_model.safetensors ...
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import shutil
import sys
import threading
import time
import urllib.request

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"
BASE = "https://huggingface.co/{repo}/resolve/main/{path}"


def _headers(extra: dict[str, str] | None = None) -> dict[str, str]:
    hdrs = {"User-Agent": UA}
    token = os.environ.get("HF_TOKEN")
    if token:
        hdrs["Authorization"] = f"Bearer {token}"
    if extra:
        hdrs.update(extra)
    return hdrs


def head_size(url: str) -> int:
    req = urllib.request.Request(url, headers=_headers(), method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            for key in ("X-Linked-Size", "Content-Length"):
                val = r.headers.get(key)
                if val is not None:
                    return int(val)
    except Exception:  # noqa: BLE001 - fall back to a 1-byte range GET
        pass
    req = urllib.request.Request(url, headers=_headers({"Range": "bytes=0-0"}))
    with urllib.request.urlopen(req, timeout=60) as r:
        cr = r.headers.get("Content-Range", "")
        if "/" in cr:
            return int(cr.split("/")[-1])
        return int(r.headers["Content-Length"])


def fetch_range(url: str, start: int, end: int, dst: pathlib.Path, retries: int = 8) -> int:
    """Download [start, end] into dst, resuming from whatever is already there."""
    have = dst.stat().st_size if dst.exists() else 0
    need = end - start + 1
    if have >= need:
        return need
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url, headers=_headers({"Range": f"bytes={start + have}-{end}"})
            )
            with urllib.request.urlopen(req, timeout=120) as r, open(dst, "ab") as f:
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
                    have += len(chunk)
                    if have >= need:
                        break
            if have >= need:
                return need
        except Exception as e:  # noqa: BLE001
            print(f"[warn] {dst.name} retry {attempt + 1}: {e}", file=sys.stderr)
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"failed chunk {dst} {have}/{need}")


def download(url: str, dst: pathlib.Path, n_conn: int = 16, chunk_mb: int = 64) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    size = head_size(url)
    parts_dir = dst.with_suffix(dst.suffix + ".parts")
    parts_dir.mkdir(parents=True, exist_ok=True)
    chunk = chunk_mb << 20
    ranges = [(i, min(i + chunk - 1, size - 1)) for i in range(0, size, chunk)]
    todo = [
        (i, s, e)
        for i, (s, e) in enumerate(ranges)
        if not (parts_dir / f"{i:05d}.bin").exists()
        or (parts_dir / f"{i:05d}.bin").stat().st_size < (e - s + 1)
    ]
    print(f"[dl] {dst.name}: {size / 1e9:.2f} GB, {len(ranges)} chunks, {len(todo)} to fetch", flush=True)
    t0 = time.time()
    done_bytes = [0]
    lock = threading.Lock()

    def work(item):
        i, s, e = item
        p = parts_dir / f"{i:05d}.bin"
        got = fetch_range(url, s, e, p)
        with lock:
            done_bytes[0] += got
        return got

    with cf.ThreadPoolExecutor(max_workers=n_conn) as ex:
        futs = [ex.submit(work, it) for it in todo]
        last = 0.0
        for f in cf.as_completed(futs):
            f.result()
            if time.time() - last > 20:
                el = max(time.time() - t0, 1e-6)
                with lock:
                    d = done_bytes[0]
                rate = d / el / 1e6
                print(f"[dl] {dst.name}: {d / 1e9:.2f} GB, {rate:.2f} MB/s", flush=True)
                last = time.time()

    print(f"[dl] {dst.name}: assembling", flush=True)
    tmp = dst.with_suffix(dst.suffix + ".assembling")
    with open(tmp, "wb") as out:
        for i, _ in enumerate(ranges):
            with open(parts_dir / f"{i:05d}.bin", "rb") as f:
                shutil.copyfileobj(f, out, length=1 << 22)
    tmp.replace(dst)
    shutil.rmtree(parts_dir, ignore_errors=True)
    print(f"[dl] {dst.name}: done {dst.stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--dest", required=True)
    ap.add_argument("--files", nargs="+", required=True)
    ap.add_argument("--conn", type=int, default=16)
    ap.add_argument("--chunk-mb", type=int, default=64)
    args = ap.parse_args()

    dest = pathlib.Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    for rel in args.files:
        url = BASE.format(repo=args.repo, path=rel)
        dst = dest / rel
        if dst.exists() and dst.stat().st_size > 0:
            print(f"[skip] {rel} exists ({dst.stat().st_size / 1e9:.2f} GB)", flush=True)
            continue
        download(url, dst, n_conn=args.conn, chunk_mb=args.chunk_mb)
    print(json.dumps({"ok": True, "dest": str(dest)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
