"""One-episode smoke test for the Yume adapter: frames produced, wall time, VRAM."""
from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import cv2

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldloop.adapters.yume import YumeModel  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="abandoned_bakery__loop360")
    ap.add_argument("--steps", type=int, default=6)
    ap.add_argument("--segments", default="turn_right:2.0")
    ap.add_argument("--python", default=None, help="interpreter for yume_cli subprocess")
    ap.add_argument("--weights", default=None, help="Yume-5B-720P checkpoint directory")
    ap.add_argument("--resolution", default="704x1280", choices=["704x1280", "544x960"])
    ap.add_argument("--out", default=str(ROOT / "scratch" / "yume_smoke"))
    args = ap.parse_args()

    start_path = pathlib.Path(f"cases/start/{args.case}.png")
    if not start_path.exists():
        start_path = ROOT / "cases" / "start" / f"{args.case}.png"
    img = cv2.imread(str(start_path))
    if img is None:
        raise SystemExit(f"missing start image: {start_path}")
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    segs = []
    for item in args.segments.split(","):
        action, secs = item.split(":")
        segs.append({"action": action, "seconds": float(secs), "tag": None})

    model = YumeModel(
        steps=args.steps,
        python=args.python,
        weights=args.weights,
        resolution=args.resolution,
    )
    try:
        model.reset(img, prompt="First-person perspective of an abandoned bakery interior.", seed=42)
        t0 = time.perf_counter()
        model.plan(segs, 16.0)
        dt = time.perf_counter() - t0
        frames = model._frames
        print(f"frames={len(frames)} shape={frames[0].shape} wall={dt:.1f}s steps={args.steps}")
        for i, f in enumerate(frames[:: max(1, len(frames) // 8)]):
            cv2.imwrite(str(out / f"f{i:02d}.jpg"), f)
        cv2.imwrite(str(out / "last.jpg"), frames[-1])
        print(json.dumps({"n": len(frames), "wall_s": dt, "cost": model.cost_summary()}, indent=1))
    finally:
        model.close()


if __name__ == "__main__":
    main()
