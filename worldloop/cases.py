"""Evaluation case construction.

A *case* is a controlled world-playback script on a photo-real panorama:

  start pose  ->  conditioned first frame  ->  a sequence of chunk-level actions

Every case ships with its exact pose timeline, so any generated frame can be
compared against the analytically correct camera motion, and any return-to-start
frame can be compared against the frame the model itself produced at that pose
earlier in the rollout.
"""

from __future__ import annotations

import json
import math
import pathlib
from dataclasses import dataclass, asdict, field

import numpy as np

from . import actions as A
from .pano import PanoCamera, load_pano, render, pixel_to_lonlat

DEG = math.pi / 180.0


@dataclass
class Probe:
    """A salient image region that should survive a look-away."""

    lon: float  # world direction of the probe centre (radians)
    lat: float
    size: int  # patch side in pixels at the reference resolution
    ref_px: float
    ref_py: float


@dataclass
class Case:
    case_id: str
    protocol: str
    pano: str
    start_yaw: float
    start_pitch: float
    fov: float
    width: int
    height: int
    fps: float
    segments: list[dict]
    probes: list[dict] = field(default_factory=list)
    tags: dict = field(default_factory=dict)

    @property
    def n_frames(self) -> int:
        return sum(max(1, int(round(float(s["seconds"]) * self.fps))) for s in self.segments)

    def segment_list(self) -> list[A.Segment]:
        return A.from_dicts(self.segments)

    def timeline(self) -> A.PoseTimeline:
        return A.compile_timeline(self.segment_list(), self.fps, self.start_yaw, self.start_pitch)

    def to_dict(self) -> dict:
        return asdict(self)


def _salient_probes(start_img: np.ndarray, cam: PanoCamera, k: int = 3, size: int = 96, seed: int = 0) -> list[Probe]:
    """Pick high-gradient, spatially spread probes in the first frame."""
    import cv2

    gray = cv2.cvtColor(start_img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.GaussianBlur(np.hypot(gx, gy), (0, 0), 6)

    H, W = gray.shape
    eff_size = min(size, max(16, min(H, W) // 3))
    margin = eff_size
    cand = []
    step = max(4, min(16, eff_size // 4))
    for y in range(margin, H - margin, step):
        for x in range(margin, W - margin, step):
            cand.append((mag[y, x], x, y))
    cand.sort(reverse=True)

    chosen: list[tuple[int, int]] = []
    min_dist = eff_size * 1.6
    for _, x, y in cand:
        if all((x - cx) ** 2 + (y - cy) ** 2 > min_dist**2 for cx, cy in chosen):
            chosen.append((x, y))
        if len(chosen) >= k:
            break

    probes = []
    for x, y in chosen:
        lon, lat = pixel_to_lonlat(cam, x, y)
        probes.append(Probe(lon=lon, lat=lat, size=eff_size, ref_px=float(x), ref_py=float(y)))
    return probes


PROTOCOLS: dict[str, list[A.Segment]] = {
    # every segment is a multiple of 2 s: that is the native chunk of the models we
    # evaluate, so no adapter is ever forced to generate a partial chunk
    "loop360": [
        A.Segment("turn_right", 2.0, "loop"),
        A.Segment("turn_right", 2.0, "loop"),
        A.Segment("turn_right", 2.0, "revisit"),
    ],
    # look 90 deg away, hold, come back to the exact start view
    "look_away": [
        A.Segment("turn_right", 2.0, "depart"),
        A.Segment("noop", 2.0, "occluded"),
        A.Segment("turn_left", 2.0, "revisit"),
    ],
    # long occlusion: probes are off-screen for 4 s
    "occlusion": [
        A.Segment("turn_right", 2.0, "depart"),
        A.Segment("noop", 2.0, "occluded"),
        A.Segment("turn_left", 2.0, "revisit"),
    ],
    # nothing happens: a world model should keep the scene stable
    "null_action": [A.Segment("noop", 4.0, "static")],
    # long horizon drift: three full loops, 18 s
    "drift": [A.Segment("turn_right", 18.0, "loop")],
}

# counterfactual pairs share a start frame, a 2 s no-op prefix, and only then diverge
COUNTERFACTUAL = {
    "cf_left": [
        A.Segment("noop", 2.0, "pre"),
        A.Segment("turn_left", 2.0, "intervention"),
        A.Segment("turn_right", 2.0, "return"),
    ],
    "cf_right": [
        A.Segment("noop", 2.0, "pre"),
        A.Segment("turn_right", 2.0, "intervention"),
        A.Segment("turn_left", 2.0, "return"),
    ],
    "cf_null": [
        A.Segment("noop", 2.0, "pre"),
        A.Segment("noop", 2.0, "intervention"),
        A.Segment("noop", 2.0, "return"),
    ],
}


def build_case(
    case_id: str,
    protocol: str,
    pano_path: str,
    segments: list[A.Segment],
    start_yaw: float = 0.0,
    start_pitch: float = 0.0,
    fov_deg: float = 70.0,
    width: int = 1280,
    height: int = 720,
    fps: float = 8.0,
    tags: dict | None = None,
) -> tuple[Case, np.ndarray]:
    pano = load_pano(pano_path)
    cam = PanoCamera(yaw=start_yaw, pitch=start_pitch, fov=fov_deg * DEG, width=width, height=height)
    img = render(pano, cam)
    probes = _salient_probes(img, cam)
    case = Case(
        case_id=case_id,
        protocol=protocol,
        pano=str(pano_path),
        start_yaw=start_yaw,
        start_pitch=start_pitch,
        fov=float(DEG * fov_deg),
        width=width,
        height=height,
        fps=fps,
        segments=A.to_dicts(segments),
        probes=[asdict(p) for p in probes],
        tags=tags or {},
    )
    return case, img


def build_case_set(
    panos: list[dict],
    out_dir: str | pathlib.Path,
    protocols: list[str] | None = None,
    counterfactual_scenes: int = 2,
    drift_scenes: int = 1,
    width: int = 1280,
    height: int = 720,
    fps: float = 8.0,
) -> dict:
    """Render every start frame and write ``cases.json``."""
    out = pathlib.Path(out_dir)
    (out / "start").mkdir(parents=True, exist_ok=True)
    if protocols is None:
        protocols = ["loop360", "occlusion", "null_action"]
    for proto in protocols:
        if proto not in PROTOCOLS:
            raise ValueError(f"unknown protocol {proto!r}; known: {sorted(PROTOCOLS)}")

    cases: list[dict] = []
    meta_panos = []
    for i, p in enumerate(panos):
        path = p["path"] if isinstance(p, dict) else str(p)
        slug = pathlib.Path(path).stem
        # deterministically vary the starting heading per scene, always at eye level
        start_yaw = (i * 47.0 % 360.0) * DEG
        meta_panos.append({"slug": slug, "path": path, "start_yaw": start_yaw})

        for proto in protocols:
            cid = f"{slug}__{proto}"
            case, img = build_case(cid, proto, path, PROTOCOLS[proto], start_yaw=start_yaw, fps=fps,
                                   width=width, height=height, tags={"pano_slug": slug})
            import cv2

            cv2.imwrite(str(out / "start" / f"{cid}.png"), img)
            cases.append(case.to_dict())

    # long-horizon drift is expensive, so only on the first few scenes
    for i, p in enumerate(panos[:drift_scenes]):
        path = p["path"] if isinstance(p, dict) else str(p)
        slug = pathlib.Path(path).stem
        start_yaw = (i * 47.0 % 360.0) * DEG
        cid = f"{slug}__drift"
        case, img = build_case(cid, "drift", path, PROTOCOLS["drift"], start_yaw=start_yaw, fps=fps,
                               width=width, height=height, tags={"pano_slug": slug})
        import cv2

        cv2.imwrite(str(out / "start" / f"{cid}.png"), img)
        cases.append(case.to_dict())

    # counterfactual triplets on the first few scenes: same start, divergent action
    for i, p in enumerate(panos[:counterfactual_scenes]):
        path = p["path"] if isinstance(p, dict) else str(p)
        slug = pathlib.Path(path).stem
        start_yaw = (i * 47.0 % 360.0) * DEG
        for name, segs in COUNTERFACTUAL.items():
            cid = f"{slug}__{name}"
            case, img = build_case(
                cid, "counterfactual", path, segs, start_yaw=start_yaw, fps=fps, width=width, height=height,
                tags={"pano_slug": slug, "group": f"{slug}__cf"},
            )
            import cv2

            cv2.imwrite(str(out / "start" / f"{cid}.png"), img)
            cases.append(case.to_dict())

    manifest = {
        "version": "0.1.0",
        "width": width,
        "height": height,
        "fps": fps,
        "n_cases": len(cases),
        "panos": meta_panos,
        "cases": cases,
    }
    (out / "cases.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
