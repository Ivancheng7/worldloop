"""End-to-end smoke test on the oracle adapter (fast, small resolution)."""
from __future__ import annotations

import json
import math
import pathlib
import sys

import cv2
import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldloop import cases as C  # noqa: E402
from worldloop.adapters.oracle import OracleModel  # noqa: E402
from worldloop.metrics.protocol import (  # noqa: E402
    aggregate_metrics,
    load_frames,
    score_case,
    score_counterfactual,
)
from worldloop.runner import run_case  # noqa: E402

PANOS = sorted((ROOT / "data" / "panos").glob("*.jpg"))
OUT = ROOT / "scratch" / "smoke"
W, H, FPS = 640, 360, 4.0


def main() -> None:
    assert PANOS, "no panoramas yet"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    panos = [{"path": p.as_posix()} for p in PANOS[:2]]
    manifest = C.build_case_set(
        panos,
        OUT / "cases",
        protocols=["loop360", "null_action"],
        counterfactual_scenes=1,
        width=W,
        height=H,
        fps=FPS,
    )
    print(f"cases: {manifest['n_cases']}")

    for case in manifest["cases"]:
        cdir = OUT / "results" / "oracle_sim" / case["case_id"]
        if (cdir / "meta.json").exists():
            continue
        start = cv2.imread(str(OUT / "cases" / "start" / f"{case['case_id']}.png"))
        model = OracleModel(case["pano"], fov_deg=math.degrees(case["fov"]), width=W, height=H, fps=FPS)
        meta = run_case(model, case, start, OUT / "results" / "oracle_sim")
        print(f"ran {case['case_id']}: {meta['n_frames']} frames in {meta['wall_seconds']:.2f}s")

    scores = []
    for case in manifest["cases"]:
        cid = case["case_id"]
        frames = load_frames(OUT / "results" / "oracle_sim" / cid)
        if frames:
            start_img = cv2.imread(str(OUT / "cases" / "start" / f"{cid}.png"))
            scores.append(score_case(case, frames, start_image=start_img, device=device))
    agg = aggregate_metrics(scores)
    by_id = {c["case_id"]: c for c in manifest["cases"]}
    agg["counterfactual"] = score_counterfactual(OUT / "results" / "oracle_sim", by_id, device=device)
    print(json.dumps(agg, indent=2))
    for s in scores:
        print(json.dumps(s))


if __name__ == "__main__":
    main()
