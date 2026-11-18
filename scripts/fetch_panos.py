"""Download CC0 indoor/urban equirectangular panoramas from Poly Haven.

Poly Haven serves every asset as HDRI (.hdr/.exr) plus a tonemapped JPG.
We use the tonemapped JPG (already display-referred, 8-bit) so no tonemapping
code is needed, then downscale to a fixed equirect size for reproducibility.

Usage:
    python scripts/fetch_panos.py --out data/panos --n 10
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import urllib.request

import cv2

API_ASSETS = "https://api.polyhaven.com/assets?t=hdris&c={cat}"
API_FILES = "https://api.polyhaven.com/files/{slug}"

# Curated fallback list: interior / street-level scenes with strong parallax cues.
FALLBACK = [
    "abandoned_bakery",
    "empty_warehouse_01",
    "small_empty_room_1",
    "small_empty_room_2",
    "brown_photostudio_02",
    "studio_small_03",
    "hotel_room",
    "lebombo",
    "venice_sunset",
    "modern_buildings",
]

CATEGORIES = ["indoor", "urban", "outdoor"]


UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
TONEMAPPED_URL = "https://dl.polyhaven.org/file/ph-assets/HDRIs/extra/Tonemapped%20JPG/{slug}.jpg"


def _open(url: str, timeout: int = 120):
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    return urllib.request.urlopen(req, timeout=timeout)


def get_json(url: str):
    with _open(url, timeout=60) as r:
        return json.loads(r.read().decode("utf-8"))


def discover(limit: int = 200):
    """Return candidate slugs ordered by category preference."""
    seen, ordered = set(), []
    for cat in CATEGORIES:
        try:
            assets = get_json(API_ASSETS.format(cat=cat))
        except Exception as e:  # noqa: BLE001
            print(f"[warn] category {cat} failed: {e}", file=sys.stderr)
            continue
        for slug in assets:
            if slug in seen:
                continue
            seen.add(slug)
            ordered.append(slug)
    return ordered[:limit]


def tonemapped_url(slug: str) -> str | None:
    try:
        files = get_json(API_FILES.format(slug=slug))
        tm = files.get("tonemapped") or {}
        if tm.get("url"):
            return tm["url"]
    except Exception:
        pass
    return TONEMAPPED_URL.format(slug=slug)


def download(url: str, dst: pathlib.Path) -> bool:
    if dst.exists() and dst.stat().st_size > 0:
        return True
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        # curl is present on Windows 10+, macOS and most Linux images and is the most
        # reliable path through corporate proxies / TLS interception.
        import shutil
        import subprocess

        if shutil.which("curl"):
            r = subprocess.run(
                ["curl", "-sSL", "--fail", "-m", "600", "-A", UA, "-o", str(tmp), url],
                capture_output=True,
            )
            if r.returncode != 0:
                raise RuntimeError(r.stderr.decode("utf-8", "ignore")[:300])
        else:
            with _open(url, timeout=300) as resp, open(tmp, "wb") as f:
                shutil.copyfileobj(resp, f, length=1 << 20)
        tmp.replace(dst)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"[warn] download failed {url}: {e}", file=sys.stderr)
        if tmp.exists():
            tmp.unlink()
        return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/panos")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--width", type=int, default=4096, help="equirect width after downscale")
    ap.add_argument("--discover", action="store_true", help="query the API for new candidates")
    args = ap.parse_args()

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    candidates = discover() if args.discover else []
    ordered = FALLBACK + [c for c in candidates if c not in set(FALLBACK)]

    manifest = []
    for slug in ordered:
        if len(manifest) >= args.n:
            break
        dst = out / f"{slug}.jpg"
        if dst.exists():
            img = cv2.imread(str(dst))
            if img is not None:
                manifest.append({"slug": slug, "path": dst.as_posix(), "w": img.shape[1], "h": img.shape[0]})
                continue
        url = tonemapped_url(slug)
        if not url:
            continue
        if not download(url, dst):
            continue
        img = cv2.imread(str(dst))
        if img is None:
            print(f"[warn] unreadable {dst}", file=sys.stderr)
            continue
        # 2:1 equirect, downscale to keep the eval set small and deterministic
        h, w = img.shape[:2]
        if w != args.width:
            img = cv2.resize(img, (args.width, args.width // 2), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dst), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        manifest.append({"slug": slug, "path": dst.as_posix(), "w": img.shape[1], "h": img.shape[0], "source": url})
        print(f"[ok] {slug} {img.shape[1]}x{img.shape[0]}")

    meta = out / "panos.json"
    meta.write_text(json.dumps({"panoramas": manifest}, indent=2), encoding="utf-8")
    print(f"[done] {len(manifest)} panoramas -> {meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
