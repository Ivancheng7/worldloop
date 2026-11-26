"""Closed-loop rollout runner."""

from __future__ import annotations

import json
import pathlib
import time

import cv2
import numpy as np

from .adapters.base import ClosedLoopModel


def _resample(frames: list[np.ndarray], n: int) -> list[np.ndarray]:
    if not frames:
        raise RuntimeError("adapter returned no frames")
    if len(frames) == n:
        return frames
    if len(frames) == 1:
        return [frames[0]] * n
    idx = np.linspace(0, len(frames) - 1, n)
    idx = np.clip(np.round(idx).astype(int), 0, len(frames) - 1)
    return [frames[i] for i in idx]


def load_cases(path: str | pathlib.Path) -> dict:
    return json.loads(pathlib.Path(path).read_text(encoding="utf-8"))


def run_case(
    model: ClosedLoopModel,
    case: dict,
    start_image: np.ndarray,
    out_dir: str | pathlib.Path,
    prompt: str = "",
    seed: int = 0,
) -> dict:
    case_id = case["case_id"]
    fps = float(case["fps"])
    cdir = pathlib.Path(out_dir) / case_id
    fdir = cdir / "frames"
    fdir.mkdir(parents=True, exist_ok=True)

    model.records.clear()
    t0 = time.perf_counter()
    model.reset(start_image, prompt=prompt, seed=seed)
    if hasattr(model, "set_pose"):
        model.set_pose(float(case["start_yaw"]), float(case["start_pitch"]))
    if hasattr(model, "set_fov") and "fov" in case:
        model.set_fov(float(case["fov"]))
    # pose-conditioned models want the whole trajectory up front
    if hasattr(model, "plan"):
        model.plan(case["segments"], fps)

    frames: list[np.ndarray] = []
    for seg in case["segments"]:
        seconds = float(seg["seconds"])
        got = model.step(seg["action"], seconds)
        target_n = max(1, int(round(seconds * fps)))
        frames.extend(_resample(got, target_n))

    if not frames:
        raise RuntimeError("rollout produced no frames")

    wall = time.perf_counter() - t0
    for i, fr in enumerate(frames):
        cv2.imwrite(str(fdir / f"{i:04d}.jpg"), fr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(cdir / "rollout.mp4"), fourcc, fps, (w, h))
    for fr in frames:
        vw.write(fr)
    vw.release()
    cv2.imwrite(str(cdir / "start.png"), start_image)

    meta = {
        "case_id": case_id,
        "protocol": case["protocol"],
        "model": model.name,
        "n_frames": len(frames),
        "fps": fps,
        "segments": case["segments"],
        "prompt": prompt,
        "seed": seed,
        "wall_seconds": wall,
        "cost": model.cost_summary(),
        "intentional_freeze": bool(getattr(model, "intentional_freeze", False)),
    }
    if hasattr(model, "meta") and callable(model.meta):
        meta["adapter"] = model.meta()
    (cdir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    model.close()
    return meta
