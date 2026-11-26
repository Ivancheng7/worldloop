"""Model adapter interface.

Every interactive world model is wrapped as a *closed-loop* generator:

    model.reset(start_image, prompt, seed)
    for action_segment in script:
        frames = model.step(action, seconds)

The adapter owns its native chunking; the runner resamples the returned frames to
the case frame rate.  Keeping native chunk size inside the adapter is what makes
the benchmark fair across models with 0.5 s and 4 s internal chunks.
"""

from __future__ import annotations

import abc
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class StepRecord:
    action: str
    seconds: float
    n_frames: int
    latency_s: float
    peak_vram_gb: float | None = None


class ClosedLoopModel(abc.ABC):
    """Base class for anything that can be played like a world."""

    name: str = "base"
    native_fps: float = 16.0
    # smallest action duration the model can express; the runner never asks for less
    min_chunk_seconds: float = 0.5
    note: str = ""
    # True when repeating the conditioning frame is the model's actual behaviour
    # (a frozen-frame baseline), so a still rollout must not be read as an
    # exporter failure.  Interactive models leave this False.
    intentional_freeze: bool = False

    def __init__(self, **kwargs):
        self.records: list[StepRecord] = []

    @abc.abstractmethod
    def reset(self, image: np.ndarray, prompt: str = "", seed: int = 0) -> None:
        """Condition the model on a first frame (BGR uint8)."""

    @abc.abstractmethod
    def step(self, action: str, seconds: float) -> list[np.ndarray]:
        """Advance the world; return BGR frames at ``native_fps``."""

    def close(self) -> None:  # pragma: no cover - optional
        pass

    # -- helpers ---------------------------------------------------------
    def _time(self):
        return time.perf_counter()

    def _record(self, action: str, seconds: float, n_frames: int, t0: float, peak_vram_gb=None):
        self.records.append(
            StepRecord(
                action=action,
                seconds=seconds,
                n_frames=n_frames,
                latency_s=time.perf_counter() - t0,
                peak_vram_gb=peak_vram_gb,
            )
        )

    def cost_summary(self) -> dict:
        if not self.records:
            return {}
        lat = np.array([r.latency_s for r in self.records], dtype=np.float64)
        dur = np.array([r.seconds for r in self.records], dtype=np.float64)
        vram = [r.peak_vram_gb for r in self.records if r.peak_vram_gb is not None]
        return {
            "n_steps": len(self.records),
            "latency_s_mean": float(lat.mean()),
            "latency_s_p50": float(np.percentile(lat, 50)),
            "latency_s_p95": float(np.percentile(lat, 95)),
            "realtime_factor": float((dur / np.maximum(lat, 1e-6)).mean()),
            "peak_vram_gb": float(max(vram)) if vram else None,
        }
