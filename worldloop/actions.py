"""Action space and pose timeline compilation.

Interactive video world models disagree about how actions are expressed (keyboard
tokens, camera poses, latent action vectors).  The benchmark therefore defines a
small, model-agnostic *chunk-level* action language:

    each step = one action held for `duration` seconds

and every adapter is responsible for mapping it to its native control.  The
compiler turns a segment script into a per-frame pose timeline, which is what the
metrics compare against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from typing import Iterable

DEG = math.pi / 180.0

# rate in degrees / second
ACTION_RATES: dict[str, tuple[float, float]] = {
    # name: (delta_yaw_deg_per_s, delta_pitch_deg_per_s)
    "turn_right": (60.0, 0.0),
    "turn_left": (-60.0, 0.0),
    "pitch_up": (0.0, 30.0),
    "pitch_down": (0.0, -30.0),
    "noop": (0.0, 0.0),
}

# aliases used by native keyboard interfaces
KEYBOARD_ALIASES = {
    "turn_right": "right",
    "turn_left": "left",
    "pitch_up": "up",
    "pitch_down": "down",
    "noop": ".",
}


@dataclass
class Segment:
    action: str
    seconds: float
    tag: str | None = None  # e.g. "intervention", "revisit"

    def __post_init__(self) -> None:
        if self.action not in ACTION_RATES:
            raise ValueError(f"unknown action {self.action!r}; known: {sorted(ACTION_RATES)}")
        if self.seconds <= 0:
            raise ValueError("segment duration must be > 0")


@dataclass
class PoseTimeline:
    fps: float
    yaw: list[float] = field(default_factory=list)  # radians, absolute
    pitch: list[float] = field(default_factory=list)
    segment_of_frame: list[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.yaw)

    def as_dict(self) -> dict:
        return {"fps": self.fps, "yaw": self.yaw, "pitch": self.pitch, "segment_of_frame": self.segment_of_frame}


def compile_timeline(segments: Iterable[Segment], fps: float, start_yaw: float = 0.0, start_pitch: float = 0.0) -> PoseTimeline:
    """Expand segments into a per-frame absolute pose timeline.

    Frame 0 is the first *generated* frame (the conditioning image is pose -1).
    """
    if fps <= 0:
        raise ValueError("fps must be > 0")
    segments = list(segments)
    tl = PoseTimeline(fps=fps)
    yaw, pitch = start_yaw, start_pitch
    for si, seg in enumerate(segments):
        n = max(1, int(round(seg.seconds * fps)))
        dyaw, dpitch = ACTION_RATES[seg.action]
        for _ in range(n):
            yaw += dyaw * DEG / fps
            pitch += dpitch * DEG / fps
            tl.yaw.append(yaw)
            tl.pitch.append(pitch)
            tl.segment_of_frame.append(si)
    return tl


def total_seconds(segments: Iterable[Segment]) -> float:
    return sum(s.seconds for s in segments)


def script_string(segments: Iterable[Segment]) -> str:
    return " > ".join(f"{s.action}:{s.seconds:g}s" for s in segments)


def to_dicts(segments: Iterable[Segment]) -> list[dict]:
    return [asdict(s) for s in segments]


def from_dicts(items: Iterable[dict]) -> list[Segment]:
    return [Segment(**d) for d in items]
