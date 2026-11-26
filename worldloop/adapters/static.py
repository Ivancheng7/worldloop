"""Degenerate baselines.

These do not model the world at all; they exist to show what the metrics reward and
punish.  A *static* model is the classic degenerate solution to any consistency
benchmark: it never moves, so it never drifts.  It should score ~1.0 on memory
metrics and ~0 on control, which is exactly why the leaderboard reports both.
"""

from __future__ import annotations

import numpy as np

from .base import ClosedLoopModel


class StaticModel(ClosedLoopModel):
    name = "static"
    note = "returns the conditioning frame forever"
    intentional_freeze = True

    def __init__(self, fps: float = 8.0):
        super().__init__()
        self.native_fps = fps
        self.min_chunk_seconds = 1.0 / fps
        self._frame: np.ndarray | None = None

    def reset(self, image: np.ndarray, prompt: str = "", seed: int = 0) -> None:
        self._frame = image.copy()
        self.records.clear()

    def step(self, action: str, seconds: float) -> list[np.ndarray]:
        if self._frame is None:
            raise RuntimeError("reset() must be called before step()")
        t0 = self._time()
        n = max(1, int(round(seconds * self.native_fps)))
        frames = [self._frame.copy() for _ in range(n)]
        self._record(action, seconds, n, t0)
        return frames
