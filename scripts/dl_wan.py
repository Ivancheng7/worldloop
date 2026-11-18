"""Chunked parallel downloader for the Wan2.1 T2V 1.3B diffusers repo.

Network here is ~3 MB/s aggregate, so we split every file into 8 MB chunks and
download them over many connections (ModelScope + optional mirror), with
per-chunk resume.  Chunks land in ``<file>.chunks/`` and are concatenated in
order once complete.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import pathlib
import shutil
import sys
import threading
import time
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_OUT = os.environ.get("WAN_MODEL_ID", str(ROOT / "models" / "Wan2.1-T2V-1.3B-Diffusers"))

# source bases, tried in order per chunk
BASES = [
    f"https://modelscope.cn/models/{REPO}/resolve/master/",
    f"https://hf-mirror.com/{REPO}/resolve/main/",
]

FILES = [
    # path, size (from ModelScope API), priority
    ("model_index.json", 400, 0),
    ("scheduler/scheduler_config.json", None, 0),
    ("tokenizer/special_tokens_map.json", None, 0),
    ("tokenizer/tokenizer.json", None, 0),
    ("tokenizer/tokenizer_config.json", None, 0),
    ("text_encoder/config.json", None, 0),
    ("text_encoder/model.safetensors.index.json", None, 0),
    ("transformer/config.json", None, 0),
    ("transformer/diffusion_pytorch_model.safetensors.index.json", None, 0),
    ("vae/config.json", None, 0),
    ("transformer/diffusion_pytorch_model-00002-of-00002.safetensors", 677_289_072, 1),
    ("vae/diffusion_pytorch_model.safetensors", 507_591_892, 1),
    ("transformer/diffusion_pytorch_model-00001-of-00002.safetensors", 4_998_781_576, 1),
    ("text_encoder/model-00005-of-00005.safetensors", 2_885_866_152, 2),
    ("text_encoder/model-00001-of-00005.safetensors", 4_972_389_712, 2),
    ("text_encoder/model-00002-of-00005.safetensors", 4_899_225_672, 2),
    ("text_encoder/model-00003-of-00005.safetensors", 4_966_309_504, 2),
    ("text_encoder/model-00004-of-00005.safetensors", 4_999_880_704, 2),
]

CHUNK = 8 * 1024 * 1024
WORKERS = 48

_done_bytes = 0
_total_bytes = 0
_lock = threading.Lock()
_t0 = time.time()


def log(msg: str) -> None:
    with _lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def head_size(url_bases: list[str]) -> int:
    """ModelScope rejects HEAD; ask for a 1-byte range across mirrors and parse total."""
    for attempt in range(4 * len(url_bases)):
        url = url_bases[attempt % len(url_bases)]
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=0-0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                cr = r.headers.get("Content-Range", "")
                if "/" in cr:
                    return int(cr.split("/")[-1])
                n = int(r.headers.get("Content-Length", "0"))
                if n:
                    return n
        except Exception as e:  # noqa: BLE001
            log(f"probe failed {url}: {e}")
            time.sleep(1.0)
    return 0


def fetch_chunk(url_bases: list[str], dest: str, start: int, end: int) -> int:
    """Download byte range [start, end) into dest with retries across mirrors."""
    expected = end - start
    for attempt in range(6):
        base = url_bases[attempt % len(url_bases)]
        try:
            req = urllib.request.Request(base, headers={"Range": f"bytes={start}-{end - 1}"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            if len(data) == expected:
                with open(dest, "wb") as f:
                    f.write(data)
                return len(data)
        except Exception:  # noqa: BLE001
            time.sleep(0.5 + attempt)
    raise RuntimeError(f"chunk failed {start}-{end} of {url_bases[0]}")


def download_file(out_root: str, path: str, size: int, workers: int = WORKERS) -> bool:
    global _done_bytes
    if size <= 0:
        log(f"!! invalid remote size ({size}) for {path}")
        return False
    dst = os.path.join(out_root, path)
    if os.path.exists(dst) and os.path.getsize(dst) >= size:
        log(f"skip {path} (done)")
        return True
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    chunk_dir = dst + ".chunks"
    os.makedirs(chunk_dir, exist_ok=True)

    bases = [b + path for b in BASES]
    n_chunks = (size + CHUNK - 1) // CHUNK
    todo = []
    for i in range(n_chunks):
        cp = os.path.join(chunk_dir, f"{i:05d}")
        want = min(size, (i + 1) * CHUNK) - i * CHUNK
        if not os.path.exists(cp) or os.path.getsize(cp) != want:
            todo.append(i)
    if todo:
        log(f"{path}: {len(todo)}/{n_chunks} chunks to fetch ({size / 1e9:.2f} GB)")

    def job(i: int) -> int:
        global _done_bytes
        dest = os.path.join(chunk_dir, f"{i:05d}")
        s = i * CHUNK
        e = min(size, s + CHUNK)
        got = fetch_chunk(bases, dest, s, e)
        with _lock:
            _done_bytes += got
        return got

    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        list(ex.map(job, todo))

    tmp = dst + ".assembling"
    with open(tmp, "wb") as out:
        for i in range(n_chunks):
            with open(os.path.join(chunk_dir, f"{i:05d}"), "rb") as f:
                shutil.copyfileobj(f, out, length=1 << 20)
    if os.path.getsize(tmp) != size:
        log(f"!! size mismatch {path}: {os.path.getsize(tmp)} != {size}")
        return False
    os.replace(tmp, dst)
    shutil.rmtree(chunk_dir, ignore_errors=True)
    log(f"done {path} ({size / 1e9:.2f} GB)")
    return True


def main(argv: list[str] | None = None) -> int:
    global _total_bytes
    ap = argparse.ArgumentParser(description="Download Wan2.1-T2V-1.3B-Diffusers weights")
    ap.add_argument("--out", default=DEFAULT_OUT, help="destination directory")
    ap.add_argument("--workers", type=int, default=WORKERS, help="parallel chunk workers")
    args = ap.parse_args(argv)

    out_root = str(pathlib.Path(args.out))
    os.makedirs(out_root, exist_ok=True)
    sizes: list[tuple[str, int, int]] = []
    for path, size, prio in FILES:
        if size is None:
            size = head_size([b + path for b in BASES])
            log(f"probe {path} -> {size}")
        sizes.append((path, size, prio))

    order = sorted(sizes, key=lambda x: x[2])
    _total_bytes = sum(s for _, s, _ in order)
    log(f"total {_total_bytes / 1e9:.2f} GB -> {out_root}")

    stop = threading.Event()

    def reporter() -> None:
        last = 0
        while not stop.wait(20):
            with _lock:
                d = _done_bytes
            rate = (d - last) / 20 / 1e6
            last = d
            log(f"progress {d / 1e9:.2f}/{_total_bytes / 1e9:.2f} GB  {rate:.2f} MB/s")

    rt = threading.Thread(target=reporter, daemon=True)
    rt.start()

    ok = True
    for path, size, _ in order:
        try:
            ok &= download_file(out_root, path, size, workers=args.workers)
        except Exception as e:  # noqa: BLE001
            log(f"FAILED {path}: {e}")
            ok = False
    stop.set()
    log(f"ALL DONE ok={ok} in {(time.time() - _t0) / 60:.1f} min")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
