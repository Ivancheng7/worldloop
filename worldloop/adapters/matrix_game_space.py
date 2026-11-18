"""Matrix-Game 3.5 adapter (Hugging Face Space ``hugging-apps/matrix-game-3-5``).

Why this adapter looks different from ``yume``/``wan_t2v``
--------------------------------------------------------
Matrix-Game 3.5 is *trajectory-conditioned*: one call consumes an anchor frame plus
a full ``extrinsics_c2w``/``intrinsics`` camera path and returns a whole video
(80 frames per block, up to 2 blocks per call on the Space).  It cannot be stepped
frame-by-frame the way a keyboard-driven model can, so the adapter splits the
work in two:

    plan(segments, fps)      one Space call -> decode -> cache every frame
    step(action, seconds)    pop ``seconds * native_fps`` cached frames

The cache is what makes the closed-loop interface work: the runner still drives
the rollout segment by segment, and this adapter simply replays the frames whose
pose interval matches.

Coordinate convention
---------------------
``worldloop.pano`` uses a world frame with x right, **y up**, z forward, and builds
its camera basis as ``f = (cos p sin y, sin p, cos p cos y)``, ``r = cross(up, f)``,
``u = cross(f, r)``.  OpenCV (and therefore the Matrix-Game 3.5 Space) uses x right,
**y down**, z forward for *both* camera and world coordinates.  Mapping the OpenCV
camera axes ``(right, down, forward) = (r, -u, f)`` from ``pano``'s y-up world frame
into OpenCV's y-down world frame via ``S = diag(1, -1, 1)`` gives

    R = S @ [r, -u, f]  ==  R_y(yaw) @ R_x(pitch)

which is a proper ``SO(3)`` rotation (``det(R) == +1`` everywhere, ``R(0, 0) == I_3``)
matching ``app.py``'s ``preset_trajectory``.  Rendering an equirectangular panorama
with an OpenCV pinhole camera ``(R, K = [fx, fy, cx, cy])`` (where OpenCV world rays
``dw = R @ d_cam`` have ``lon = atan2(dw.x, dw.z)`` and ``lat = arcsin(-dw.y)``)
reproduces ``worldloop.pano.render`` pixel-for-pixel.

Two consequences worth knowing:

* ``fov`` in ``cases.json`` is the **vertical** field of view of the renderer
  (its ray grid uses ``tan(fov/2)`` vertically and ``tan(fov/2)*aspect``
  horizontally).  The true intrinsics of the 1280x720 start frames are therefore
  ``fx = fy = (720/2)/tan(fov/2) = 514.13``, ``cx = 640``, ``cy = 360`` (352 after
  the Space's 1280x704 centre crop).  ``fov_axis="h"`` is also implemented for
  comparison, but it is *not* the geometry the start frames were rendered with.
* With this right-handed basis, a benchmark ``turn_right`` (``yaw > 0``) makes image
  content move **left** (``dx < 0``), matching *both* ``worldloop.pano.render`` /
  ``rotation_flow_forward`` and the Space's own ``camera_preset="Turn right (in
  place)"`` (``R_y(+0.714 deg)`` per pose).  Likewise ``pitch_up`` (``pitch > 0``)
  tilts the camera upward (``dy > 0`` in image pixels), matching ``"Look up"``.

Status of this module
---------------------
The pose/intrinsics geometry is verified offline.  The end-to-end path through
the Space could **not** be exercised from an anonymous session: ZeroGPU rejects
the request before any GPU is allocated (see ``scratch/mg35_val/probe*.json``).
The class therefore refuses to fabricate frames: ``plan`` raises
``MatrixGameSpaceError`` unless a client is available and the call succeeds.
"""

from __future__ import annotations

import json
import math
import os
import pathlib
import shutil
import tempfile
import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from .base import ClosedLoopModel

# the Space centre-crops whatever it is given to this size
SPACE_WIDTH = 1280
SPACE_HEIGHT = 704
# from the Space's app.py: need = 1 + POSES_PER_BLOCK * num_blocks, 80 new frames/block
POSES_PER_BLOCK = 84
FRAMES_PER_BLOCK = 80
MAX_BLOCKS = 2  # the Space's slider stops at 2


class MatrixGameSpaceError(RuntimeError):
    """The Space did not produce a usable rollout."""


@dataclass
class PlanInfo:
    """Everything worth recording about the single Space call behind a rollout."""

    num_blocks: int = 0
    n_poses: int = 0
    n_poses_needed: int = 0
    poses_padded: int = 0
    n_frames_cached: int = 0
    n_frames_needed: int = 0
    frames_truncated: int = 0
    video_fps: float | None = None
    video_size: list[int] = field(default_factory=list)
    frame_drop: int = 0
    wall_seconds: float | None = None
    seed_used: int | None = None
    num_blocks_request: int = 0
    notes: list[str] = field(default_factory=list)
    space_args: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "num_blocks": self.num_blocks,
            "num_blocks_request": self.num_blocks_request,
            "n_poses": self.n_poses,
            "n_poses_needed": self.n_poses_needed,
            "poses_padded": self.poses_padded,
            "n_frames_cached": self.n_frames_cached,
            "n_frames_needed": self.n_frames_needed,
            "frames_truncated": self.frames_truncated,
            "video_fps": self.video_fps,
            "video_size": self.video_size,
            "frame_drop": self.frame_drop,
            "wall_seconds": self.wall_seconds,
            "seed_used": self.seed_used,
            "notes": self.notes,
            "space_args": self.space_args,
        }


def camera_basis(yaw: float, pitch: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Benchmark camera basis: forward, right, up (world: x right, y up, z forward).

    Mirrors ``worldloop.pano._camera_basis`` so the adapter cannot drift from the
    renderer that produced the conditioning frames.
    """
    from ..pano import _camera_basis  # single source of truth for the convention

    return _camera_basis(float(yaw), float(pitch))


def c2w_matrix(yaw: float, pitch: float, translation: np.ndarray | None = None) -> np.ndarray:
    """4x4 camera-to-world matrix in the OpenCV convention the Space consumes.

    ``worldloop.pano`` uses a world frame with x right, **y up**, z forward;
    OpenCV (and Matrix-Game 3.5) uses x right, **y down**, z forward.  Mapping
    the OpenCV camera axes ``(right, down, forward) = (r, -u, f)`` into the
    OpenCV world frame via ``diag(1, -1, 1)`` gives ``R = diag(1, -1, 1) @ [r, -u, f]``
    (equivalent to ``R_y(yaw) @ R_x(pitch)``), so ``det(R) == +1`` everywhere
    and ``c2w_matrix(0, 0) == I_4``.
    """
    f, r, u = camera_basis(yaw, pitch)
    R = np.column_stack([r, -u, f])
    R[1, :] *= -1.0
    R = np.where(R == 0.0, 0.0, R)
    M = np.eye(4, dtype=np.float32)
    M[:3, :3] = R.astype(np.float32)
    if translation is not None:
        M[:3, 3] = np.asarray(translation, dtype=np.float32)
    return M


def intrinsics(fov: float, width: int = SPACE_WIDTH, height: int = SPACE_HEIGHT,
               axis: str = "v") -> np.ndarray:
    """``[fx, fy, cx, cy]`` of the conditioning camera, in *cropped* image pixels.

    ``axis="v"`` matches what ``worldloop.pano.render`` actually does (``fov`` spans
    the vertical axis); ``axis="h"`` is the docstring's claim, kept only for
    experiments.
    """
    if axis not in ("v", "h"):
        raise ValueError("fov_axis must be 'v' or 'h'")
    # the crop only removes rows, so the full-frame height is what defines the fov
    full_h = 720 if height == SPACE_HEIGHT else height
    half = (full_h if axis == "v" else width) / 2.0
    fx = fy = half / math.tan(fov / 2.0)
    return np.array([fx, fy, width / 2.0, height / 2.0], dtype=np.float32)


def plan_poses(segments, fps: float, native_fps: float, start_yaw: float, start_pitch: float,
               total_seconds: float) -> tuple[np.ndarray, list[float], list[float]]:
    """Sample the case timeline at ``native_fps`` and turn it into c2w matrices.

    Pose 0 is the conditioning frame (the anchor, identity orientation); pose ``j``
    (``j >= 1``) is the camera at ``t = j / native_fps``.  Pure rotation, so the
    translation column stays zero.  Returns ``(extrinsics, yaw, pitch)``.

    ``worldloop.actions.compile_timeline`` is 1-based: ``tl.yaw[i]`` is the pose at
    ``t = (i + 1) / fps`` (frame 0 is the first *generated* frame, the conditioning
    image sits at ``t = 0``).  The resampling below therefore evaluates the
    piecewise-linear curve at ``t * fps`` case-frames, whose value at integer
    ``m >= 1`` is ``tl.yaw[m - 1]``.
    """
    from .. import actions as A

    if isinstance(segments[0], dict):
        segs = A.from_dicts(segments)
    else:
        segs = list(segments)
    tl = A.compile_timeline(segs, fps, start_yaw, start_pitch)
    n = len(tl)

    def value_at(m: int, which: str) -> float:
        """Timeline value at integer case-frame index m (m = 0 -> t = 0, anchor)."""
        if m <= 0:
            return float(start_yaw if which == "yaw" else start_pitch)
        arr = tl.yaw if which == "yaw" else tl.pitch
        return float(arr[min(m, n) - 1])

    n_frames = max(1, int(round(total_seconds * native_fps)))
    yaw = [float(start_yaw)]
    pitch = [float(start_pitch)]
    for j in range(1, n_frames + 1):
        u = (j / native_fps) * fps     # position on the case timeline, in case frames
        m0 = int(math.floor(u))
        frac = u - m0
        for dst, which in ((yaw, "yaw"), (pitch, "pitch")):
            v0, v1 = value_at(m0, which), value_at(m0 + 1, which)
            dst.append(float(v0 * (1.0 - frac) + v1 * frac))
    extr = np.stack([c2w_matrix(y, p) for y, p in zip(yaw, pitch)], axis=0)
    return extr, yaw, pitch


class MatrixGameSpace(ClosedLoopModel):
    """One-shot trajectory-conditioned rollout through the Matrix-Game 3.5 Space."""

    name = "matrix_game_3_5"
    native_fps = 16.0
    min_chunk_seconds = 5.0

    def __init__(
        self,
        space: str = "hugging-apps/matrix-game-3-5",
        prompt: str = "",
        steps: int = 8,
        cfg_scale: float = 5.0,
        seed: int = 42,
        num_blocks: int | None = None,
        poses_per_block: int = POSES_PER_BLOCK,
        frames_per_block: int = FRAMES_PER_BLOCK,
        fov_axis: str = "v",
        hf_token: str | None = None,
        timeout_s: float = 2400.0,
        client=None,
        workdir: str | pathlib.Path | None = None,
        keep_video: bool = False,
        out_width: int = 1280,
        out_height: int = 720,
        native_fps_override: float | None = None,
        frame_pose_offset: int = 1,
    ):
        super().__init__()
        self.space = space
        self.prompt = prompt or (
            "First-person walk-through of a detailed indoor scene, photorealistic, "
            "stable camera, consistent geometry."
        )
        # the Space's sliders are steps in [8, 30] (25 max for 2 blocks); a lower
        # value is rejected by Gradio before any GPU is requested
        self.steps = int(max(8, min(30, int(steps))))
        self.steps_clamped = self.steps != int(steps)
        self.cfg_scale = float(cfg_scale)
        self.seed = int(seed)
        self.num_blocks = None if num_blocks is None else int(num_blocks)
        self.poses_per_block = int(poses_per_block)
        self.frames_per_block = int(frames_per_block)
        self.fov_axis = fov_axis
        self.hf_token = hf_token if hf_token is not None else os.environ.get("HF_TOKEN")
        self.timeout_s = float(timeout_s)
        self._client = client
        self._ext_client = client is not None
        self.workdir = pathlib.Path(workdir) if workdir is not None else None
        self.keep_video = keep_video
        self.out_width = int(out_width)
        self.out_height = int(out_height)
        # decoded frame k shows the camera at pose k + frame_pose_offset (the video
        # starts one frame after the anchor, mirroring infer.py's pose slicing
        # ``[1 + POSES_PER_BLOCK * k, ...]``).  Unverified against a live call.
        self.frame_pose_offset = int(frame_pose_offset)
        if native_fps_override:
            self.native_fps = float(native_fps_override)

        self._tmp: pathlib.Path | None = None
        self._anchor: np.ndarray | None = None
        self._anchor_path: pathlib.Path | None = None
        self._cache: list[np.ndarray] = []
        self._cursor = 0
        self._start_yaw = 0.0
        self._start_pitch = 0.0
        self._case_fov: float | None = None
        self._planned = False
        self.info = PlanInfo()

    # ------------------------------------------------------------------ setup
    def reset(self, image: np.ndarray, prompt: str = "", seed: int | None = None) -> None:
        self.records.clear()
        self._cache = []
        self._cursor = 0
        self._start_yaw = 0.0
        self._start_pitch = 0.0
        self._case_fov = None
        self._planned = False
        self.info = PlanInfo()
        if self._tmp is not None:
            shutil.rmtree(self._tmp, ignore_errors=True)
        base = self.workdir if self.workdir is not None else pathlib.Path(tempfile.mkdtemp(prefix="worldloop_mg35_"))
        base.mkdir(parents=True, exist_ok=True)
        self._tmp = base
        if prompt:
            self.prompt = prompt
        if seed is not None:
            self.seed = int(seed)
        self._anchor = self._crop(image)
        self._anchor_path = self._tmp / "anchor_1280x704.png"
        cv2.imwrite(str(self._anchor_path), self._anchor)

    def set_pose(self, yaw: float, pitch: float = 0.0) -> None:
        """Only the deltas matter (pose 0 is the anchor), but keep the pose for QA."""
        self._start_yaw = float(yaw)
        self._start_pitch = float(pitch)

    def set_fov(self, fov: float) -> None:
        """Optional: the case fov; otherwise ``fov`` from plan() defaults to 70 deg."""
        self._case_fov = float(fov)

    @staticmethod
    def _crop(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        if (w, h) == (SPACE_WIDTH, SPACE_HEIGHT):
            return image
        if w != SPACE_WIDTH:
            image = cv2.resize(image, (SPACE_WIDTH, int(round(h * SPACE_WIDTH / w))),
                               interpolation=cv2.INTER_AREA)
            h, w = image.shape[:2]
        top = max(0, (h - SPACE_HEIGHT) // 2)
        return image[top:top + SPACE_HEIGHT]

    # ------------------------------------------------------------------- plan
    def plan(self, segments, fps: float, fov: float = math.radians(70.0),
             total_seconds: float | None = None) -> dict:
        """Compile the whole case into one camera path, call the Space, cache frames."""
        if self._anchor is None:
            raise MatrixGameSpaceError("call reset() before plan()")
        t0 = self._time()
        fov = self._case_fov if self._case_fov is not None else float(fov)
        if total_seconds is None:
            total_seconds = float(sum(s["seconds"] if isinstance(s, dict) else s.seconds for s in segments))

        n_frames_needed = max(1, int(round(total_seconds * self.native_fps)))
        n_poses_needed = n_frames_needed + 1
        blocks_needed = max(1, math.ceil(n_frames_needed / self.frames_per_block))
        blocks = self.num_blocks if self.num_blocks is not None else min(MAX_BLOCKS, blocks_needed)
        blocks = int(max(1, min(MAX_BLOCKS, blocks)))
        need = 1 + self.poses_per_block * blocks

        extr, yaw, pitch = plan_poses(segments, fps, self.native_fps, self._start_yaw,
                                      self._start_pitch, total_seconds)
        intr = np.tile(intrinsics(fov, SPACE_WIDTH, SPACE_HEIGHT, self.fov_axis), (len(extr), 1))

        info = PlanInfo(num_blocks=blocks, num_blocks_request=blocks,
                        n_poses=len(extr), n_poses_needed=need,
                        n_frames_needed=n_frames_needed,
                        poses_padded=max(0, need - len(extr)),
                        frames_truncated=max(0, n_frames_needed - self.frames_per_block * blocks))
        if blocks_needed > MAX_BLOCKS:
            info.notes.append(
                f"trajectory needs {n_frames_needed} frames ({total_seconds:g}s at {self.native_fps:g} fps) "
                f"but {MAX_BLOCKS} blocks only cover {self.frames_per_block * MAX_BLOCKS} frames; "
                f"the rollout is truncated and the tail is not evaluated"
            )
        if len(extr) > need:
            # the Space truncates to `need`; mirror that here so the recorded path is
            # exactly the path that was sent
            extr, intr = extr[:need], intr[:need]
            yaw, pitch = yaw[:need], pitch[:need]
            info.notes.append(f"camera path truncated to {need} poses (2-block Space limit)")
        info.n_poses = len(extr)

        cam_path = self._tmp / "camera.npz"
        np.savez_compressed(cam_path, extrinsics_c2w=extr.astype(np.float32),
                            intrinsics=intr.astype(np.float32))
        (self._tmp / "camera_poses.json").write_text(
            json.dumps({"n_poses": len(extr), "yaw": yaw, "pitch": pitch,
                        "fov": fov, "fov_axis": self.fov_axis,
                        "intrinsics": intr[0].tolist()}, indent=2), encoding="utf-8")

        info.space_args = {
            "space": self.space, "prompt": self.prompt, "camera_file": str(cam_path),
            "camera_preset": "", "mode": "First-person", "num_blocks": blocks,
            "steps": self.steps, "cfg_scale": self.cfg_scale, "seed": self.seed,
        }
        video_path, seed_used = self._call_space(cam_path, blocks)
        info.seed_used = seed_used

        frames, vid_fps, vid_size, dropped = self._decode(video_path, info, fps)
        self._cache = frames
        self.info = info
        self.info.frame_drop = dropped
        self._planned = True
        self._record("plan", total_seconds, len(frames), t0)
        self.info.wall_seconds = time.perf_counter() - t0
        return info.as_dict()

    # -------------------------------------------------------------- space call
    def _call_space(self, camera_file: pathlib.Path, blocks: int):
        client = self._client
        if client is None:
            try:
                from gradio_client import Client, handle_file
            except ImportError as exc:  # pragma: no cover - depends on the venv
                raise MatrixGameSpaceError(
                    "gradio_client is not installed in this interpreter; run "
                    "`pip install gradio_client` in the benchmark venv"
                ) from exc
            self._handle_file = handle_file
            try:
                client = Client(self.space, hf_token=self.hf_token, verbose=False)
            except Exception as exc:  # noqa: BLE001
                raise MatrixGameSpaceError(f"cannot connect to {self.space}: {exc}") from exc
            self._client = client
        if not hasattr(self, "_handle_file"):
            try:
                from gradio_client import handle_file
            except ImportError:  # pragma: no cover
                handle_file = lambda p: p  # noqa: E731
            self._handle_file = handle_file

        try:
            result = client.predict(
                self._handle_file(str(self._anchor_path)),
                self.prompt,
                self._handle_file(str(camera_file)),
                "",                      # camera_preset: ignored when camera_file is set
                "First-person",
                [],
                float(blocks),
                float(self.steps),
                float(self.cfg_scale),
                int(self.seed),
                False,
                api_name="/generate",
            )
        except Exception as exc:  # noqa: BLE001 - surface the Space's own message
            raise MatrixGameSpaceError(f"{type(exc).__name__}: {exc}") from exc

        if isinstance(result, (list, tuple)):
            video = result[0]
            seed_used = result[1] if len(result) > 1 else None
        else:
            video, seed_used = result, None
        if not video or not pathlib.Path(str(video)).is_file():
            raise MatrixGameSpaceError(f"Space returned no video file: {result!r}")
        if not self.keep_video:
            dst = self._tmp / "space_video.mp4"
            shutil.copyfile(str(video), dst)
            return dst, seed_used
        return pathlib.Path(str(video)), seed_used

    # ------------------------------------------------------------------ decode
    def _decode(self, video_path: pathlib.Path, info: PlanInfo, case_fps: float):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise MatrixGameSpaceError(f"cannot open returned video {video_path}")
        vid_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()
        if not frames:
            raise MatrixGameSpaceError(f"returned video has no frames: {video_path}")
        info.video_fps = vid_fps or None
        info.video_size = [w, h]
        if vid_fps and abs(vid_fps - self.native_fps) > 0.5:
            info.notes.append(
                f"returned video is {vid_fps:g} fps but native_fps={self.native_fps:g}; "
                f"frame-to-pose mapping may be off by {vid_fps / self.native_fps:.3f}x"
            )

        # Align the frame grid with the case frame grid.  Cache index i must show the
        # camera at pose i * r + r with r = native_fps / case_fps, because the runner
        # then keeps case frame i = cache[i * r] and a case frame is the pose one
        # case-frame *after* the anchor.  decoded[k] is pose k + frame_pose_offset,
        # hence cache[i] = decoded[i + r - offset] = pose i + r.
        dropped = 0
        r = self.native_fps / float(case_fps) if case_fps else 0.0
        if r > 1e-9 and abs(r - round(r)) < 1e-6:
            dropped = int(max(0, int(round(r)) - self.frame_pose_offset))
            dropped = min(dropped, max(0, len(frames) - 1))
        elif r:
            info.notes.append(
                f"native_fps/case_fps = {r:.3f} is not an integer; frame/pose alignment "
                f"left at 0 (sub-frame pose offset up to {1.0 / self.native_fps:.4f}s)"
            )
        if dropped:
            frames = frames[dropped:]

        out = [cv2.resize(f, (self.out_width, self.out_height), interpolation=cv2.INTER_AREA)
               for f in frames]
        info.n_frames_cached = len(out)
        return out, vid_fps, [w, h], dropped

    # -------------------------------------------------------------------- step
    def step(self, action: str, seconds: float) -> list[np.ndarray]:
        if not self._planned:
            raise MatrixGameSpaceError("plan() must be called before step()")
        n = max(1, int(round(seconds * self.native_fps)))
        frames = self._cache[self._cursor:self._cursor + n]
        self._cursor += len(frames)
        if not frames:
            raise MatrixGameSpaceError("cached rollout exhausted")
        # deliberately no StepRecord here: the GPU work all happens in plan(), and
        # per-step ~0 s latencies would make cost_summary() look like realtime
        return frames

    # -------------------------------------------------------------------- meta
    def meta(self) -> dict:
        return {
            "space": self.space,
            "steps": self.steps,
            "cfg_scale": self.cfg_scale,
            "seed": self.seed,
            "native_fps": self.native_fps,
            "poses_per_block": self.poses_per_block,
            "frames_per_block": self.frames_per_block,
            "fov_axis": self.fov_axis,
            "plan": self.info.as_dict(),
        }

    def close(self) -> None:
        if self._tmp is not None and self.workdir is None:
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None
