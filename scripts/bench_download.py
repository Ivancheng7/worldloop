"""Measure aggregate throughput of N parallel range requests (HF vs hf-mirror)."""
from __future__ import annotations

import concurrent.futures as cf
import time
import urllib.request

URLS = {
    "hf": "https://huggingface.co/stdstu123/Yume-5B-720P/resolve/main/models_t5_umt5-xxl-enc-bf16.pth",
    "mirror": "https://hf-mirror.com/stdstu123/Yume-5B-720P/resolve/main/models_t5_umt5-xxl-enc-bf16.pth",
}
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/124.0"


def grab(url: str, start: int, size: int) -> int:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Range": f"bytes={start}-{start + size - 1}"})
    got = 0
    with urllib.request.urlopen(req, timeout=30) as r:
        while got < size:
            chunk = r.read(1 << 16)
            if not chunk:
                break
            got += len(chunk)
    return got


def bench(name: str, url: str, n: int = 8, chunk: int = 4 << 20) -> None:
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=n) as ex:
        futs = [ex.submit(grab, url, i * chunk, chunk) for i in range(n)]
        total = sum(f.result() for f in futs)
    dt = max(time.perf_counter() - t0, 1e-6)
    print(f"{name}: {n} conns, {total / 1e6:.1f} MB in {dt:.1f}s -> {total / dt / 1e6:.2f} MB/s")


if __name__ == "__main__":
    for n in (4, 8, 16):
        try:
            bench("hf", URLS["hf"], n)
        except Exception as e:  # noqa: BLE001
            print("hf failed", e)
        try:
            bench("mirror", URLS["mirror"], n)
        except Exception as e:  # noqa: BLE001
            print("mirror failed", e)
