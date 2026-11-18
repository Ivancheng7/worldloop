"""Oracle adapter: renders the analytically correct view from the panorama.

This is the *upper bound* of the benchmark.  Running it end-to-end proves that the
metric pipeline is correct (a perfect world model must score ~1.0 on every
fidelity axis and ~0 on every drift axis).
"""

from __future__ import annotations

import math

import numpy as np

from .. import actions as A
from ..pano import PanoCamera, load_pano, render
from .base import ClosedLoopModel

DEG = math.pi / 180.0


class OracleModel(ClosedLoopModel):
    name = "oracle_sim"
    note = "analytic panorama renderer; upper bound, not a learned model"

    def __init__(self, pano_path: str, fov_deg: float = 70.0, width: int = 1280, height: int = 720, fps: float = 8.0):
        super().__init__()
        self._pano_path = str(pano_path)
        self.pano = load_pano(self._pano_path)
        self.fov = fov_deg * DEG
        self.width = width
        self.height = height
        self.native_fps = fps
        self.min_chunk_seconds = 1.0 / fps
        self.yaw = 0.0
        self.pitch = 0.0

    def set_pano(self, pano_path: str) -> None:
        p = str(pano_path)
        if p != self._pano_path:
            self.pano = load_pano(p)
            self._pano_path = p

    def reset(self, image: np.ndarray, prompt: str = "", seed: int = 0) -> None:
        # the first frame defines the pose; the runner passes non-zero poses via set_pose
        self.yaw = 0.0
        self.pitch = 0.0
        self.records.clear()

    def set_pose(self, yaw: float, pitch: float) -> None:
        self.yaw, self.pitch = yaw, pitch

    def step(self, action: str, seconds: float) -> list[np.ndarray]:
        t0 = self._time()
        dyaw, dpitch = A.ACTION_RATES[action]
        n = max(1, int(round(seconds * self.native_fps)))
        frames = []
        for _ in range(n):
            self.yaw += dyaw * DEG / self.native_fps
            self.pitch += dpitch * DEG / self.native_fps
            cam = PanoCamera(yaw=self.yaw, pitch=self.pitch, fov=self.fov, width=self.width, height=self.height)
            frames.append(render(self.pano, cam))
        self._record(action, seconds, n, t0)
        return frames
