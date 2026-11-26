"""Unit and geometry tests for WorldLoop model adapters and figure utilities."""

from __future__ import annotations

import io
import json
import math
import os
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldloop.adapters import (  # noqa: E402
    MatrixGameSpace,
    MatrixGameSpaceError,
    OracleModel,
    StaticModel,
    WanT2VBaseline,
    YumeModel,
)
from worldloop.adapters import matrix_game_space as mgs  # noqa: E402
from worldloop.adapters import wan_t2v as wan_mod  # noqa: E402
from worldloop.adapters import yume as yume_mod  # noqa: E402
from worldloop.pano import PanoCamera, load_pano, render, rotation_flow_forward  # noqa: E402
from worldloop.report import group_scores  # noqa: E402


def _make_synthetic_pano(width: int = 1024, height: int = 512) -> np.ndarray:
    """Deterministic multi-frequency textured equirectangular BGR panorama."""
    u = np.linspace(-math.pi, math.pi, width, endpoint=False, dtype=np.float32)
    v = np.linspace(math.pi / 2, -math.pi / 2, height, endpoint=False, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    b = 127.5 + 90.0 * np.sin(3.0 * uu) * np.cos(2.0 * vv) + 30.0 * np.cos(11.0 * uu)
    g = 127.5 + 90.0 * np.cos(5.0 * uu + 0.7) * np.cos(3.0 * vv) + 30.0 * np.sin(7.0 * vv)
    r = 127.5 + 90.0 * np.sin(2.0 * uu - 1.1) * np.sin(4.0 * vv) + 30.0 * np.sin(13.0 * uu * vv)
    return np.clip(np.stack([b, g, r], axis=-1), 0, 255).astype(np.uint8)


def _opencv_pinhole_render(
    pano: np.ndarray, M_c2w: np.ndarray, K: np.ndarray, width: int, height: int
) -> np.ndarray:
    """Independent OpenCV pinhole renderer (x right, y down, z forward in camera & world)."""
    fx, fy, cx, cy = [float(x) for x in K]
    ph, pw = pano.shape[:2]
    us, vs = np.meshgrid(np.arange(width, dtype=np.float64), np.arange(height, dtype=np.float64))
    x = (us + 0.5 - cx) / fx
    y = (vs + 0.5 - cy) / fy
    d_cam = np.stack([x, y, np.ones_like(x)], axis=-1)
    d_cam /= np.linalg.norm(d_cam, axis=-1, keepdims=True)
    # Transform camera ray to OpenCV world (x right, y down, z forward)
    dw = d_cam @ M_c2w[:3, :3].astype(np.float64).T
    # Convert OpenCV world ray to equirectangular (lon from +z toward +x, lat toward -y_cv == up)
    lon = np.arctan2(dw[..., 0], dw[..., 2])
    lat = np.arcsin(np.clip(-dw[..., 1], -1.0, 1.0))
    mu = ((lon / (2.0 * math.pi) + 0.5) * pw - 0.5).astype(np.float32)
    mv = ((0.5 - lat / math.pi) * ph - 0.5).astype(np.float32)
    map_x = (mu + 0.5) % pw - 0.5
    map_y = np.clip(mv, 0.0, ph - 1.0)
    return cv2.remap(pano, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)


def _rot_y_cv(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=np.float64)


def _rot_x_cv(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]], dtype=np.float64)


def _phase_shift(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    ga = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gb = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32)
    win = cv2.createHanningWindow((ga.shape[1], ga.shape[0]), cv2.CV_32F)
    (dx, dy), _ = cv2.phaseCorrelate(ga, gb, win)
    return float(dx), float(dy)


def test_oracle_and_static_contracts() -> None:
    with tempfile.TemporaryDirectory(prefix="wl_test_oracle_") as tmp:
        pano = _make_synthetic_pano()
        pano_path = pathlib.Path(tmp) / "synth.png"
        cv2.imwrite(str(pano_path), pano)

        fov_deg = 70.0
        w, h, fps = 320, 180, 4.0
        start_img = render(pano, PanoCamera(yaw=0.0, pitch=0.0, fov=math.radians(fov_deg), width=w, height=h))

        # --- OracleModel ---
        oracle = OracleModel(str(pano_path), fov_deg=fov_deg, width=w, height=h, fps=fps)
        assert oracle.cost_summary() == {}
        oracle.reset(start_img, prompt="test", seed=7)
        f1 = oracle.step("turn_right", 2.0)
        f2 = oracle.step("turn_right", 4.0)  # total 6s * 60 deg/s = 360 deg revisit
        assert len(f1) == 8 and len(f2) == 16
        assert f2[-1].shape == (h, w, 3) and f2[-1].dtype == np.uint8
        # Full 360 deg turn must return to start_img
        revisit_err = float(np.abs(f2[-1].astype(np.float32) - start_img.astype(np.float32)).mean())
        assert revisit_err < 1e-3, f"360 revisit error too high: {revisit_err}"

        cost = oracle.cost_summary()
        assert cost["n_steps"] == 2
        assert cost["latency_s_mean"] >= 0.0
        assert cost["realtime_factor"] > 0.0
        assert cost["peak_vram_gb"] is None

        # Verify reset() clears records and resets pose (no state leak across episodes)
        oracle.step("pitch_up", 1.0)
        oracle.reset(start_img)
        assert oracle.records == []
        assert oracle.yaw == 0.0 and oracle.pitch == 0.0
        noop_frames = oracle.step("noop", 0.5)
        assert np.array_equal(noop_frames[0], start_img)

        # Verify set_pano() switches panoramas and reuses identical path cleanly
        pano2 = np.full_like(pano, 210)
        pano2_path = pathlib.Path(tmp) / "synth2.png"
        cv2.imwrite(str(pano2_path), pano2)
        oracle.set_pano(str(pano2_path))
        oracle.reset(start_img)
        f_new = oracle.step("noop", 0.25)
        assert int(f_new[0].mean()) == 210
        oracle.set_pano(str(pano2_path))  # cached no-op reload
        assert int(oracle.pano.mean()) == 210

        # --- StaticModel ---
        static = StaticModel(fps=fps)
        try:
            static.step("turn_right", 1.0)
            raise AssertionError("StaticModel.step() before reset() must fail")
        except RuntimeError:
            pass

        static.reset(start_img, prompt="test", seed=0)
        sf = static.step("turn_right", 1.5)
        assert len(sf) == 6
        assert all(np.array_equal(fr, start_img) for fr in sf)
        # Mutating returned frame must not corrupt internal state
        sf[0][:] = 0
        sf2 = static.step("noop", 0.5)
        assert np.array_equal(sf2[0], start_img)
        scost = static.cost_summary()
        assert scost["n_steps"] == 2 and scost["peak_vram_gb"] is None


def test_matrix_game_space_geometry_and_rollout() -> None:
    # 1. Identity at (0, 0)
    I0 = mgs.c2w_matrix(0.0, 0.0)
    assert I0.shape == (4, 4) and I0.dtype == np.float32
    assert np.array_equal(I0, np.eye(4, dtype=np.float32)), f"c2w(0,0) != I_4:\n{I0}"

    # 2. Proper SO(3) rotation (det == +1, R.T @ R == I_3) + agreement with MG3.5 R_y @ R_x
    test_angles = [
        (0.0, 0.0),
        (0.35, 0.0),
        (-0.8, 0.0),
        (0.0, 0.25),
        (0.0, -0.4),
        (1.23, 0.31),
        (-2.4, -0.52),
        (math.pi, 0.15),
        (3.7, -0.1),
        (0.5, math.pi / 2),    # north pole gimbal edge case
        (-1.1, -math.pi / 2),  # south pole gimbal edge case
    ]
    for yaw, pitch in test_angles:
        M = mgs.c2w_matrix(yaw, pitch)
        R = M[:3, :3].astype(np.float64)
        det = float(np.linalg.det(R))
        assert abs(det - 1.0) < 1e-6, f"det(R)={det} != +1 for yaw={yaw}, pitch={pitch}"
        ortho_err = float(np.max(np.abs(R.T @ R - np.eye(3))))
        assert ortho_err < 1e-6, f"orthogonality error {ortho_err} for yaw={yaw}, pitch={pitch}"
        if abs(pitch) < math.pi / 2 - 1e-4:
            R_mg35 = _rot_y_cv(yaw) @ _rot_x_cv(pitch)
            assert np.allclose(R, R_mg35, atol=1e-6), (
                f"c2w mismatch vs R_y(yaw)@R_x(pitch) at ({yaw}, {pitch}):\n{R}\nvs\n{R_mg35}"
            )

    # 3. Pixel-for-pixel agreement between OpenCV pinhole projection (c2w + intrinsics) and pano.render
    real_panos = sorted((ROOT / "data" / "panos").glob("*.jpg"))
    pano = load_pano(str(real_panos[0])) if real_panos else _make_synthetic_pano(2048, 1024)
    fov = math.radians(70.0)
    w, h = 640, 360
    K = mgs.intrinsics(fov, width=w, height=h, axis="v")

    for yaw_deg, pitch_deg in [(0.0, 0.0), (37.5, 0.0), (-125.0, 0.0), (0.0, 15.0), (62.0, -18.0)]:
        yaw, pitch = math.radians(yaw_deg), math.radians(pitch_deg)
        ref = render(pano, PanoCamera(yaw=yaw, pitch=pitch, fov=fov, width=w, height=h))
        got = _opencv_pinhole_render(pano, mgs.c2w_matrix(yaw, pitch), K, w, h)
        mean_diff = float(np.abs(ref.astype(np.float32) - got.astype(np.float32)).mean())
        max_diff = int(np.abs(ref.astype(np.int16) - got.astype(np.int16)).max())
        assert mean_diff < 0.05 and max_diff <= 1, (
            f"render mismatch at ({yaw_deg}, {pitch_deg}): mean={mean_diff}, max={max_diff}"
        )

    #Also check cropped 1280x704 intrinsics match the centre 704 rows of a 1280x720 pano.render
    K_crop = mgs.intrinsics(fov, width=1280, height=704, axis="v")
    ref_full = render(pano, PanoCamera(yaw=0.4, pitch=-0.1, fov=fov, width=1280, height=720))
    got_crop = _opencv_pinhole_render(pano, mgs.c2w_matrix(0.4, -0.1), K_crop, 1280, 704)
    crop_diff = float(np.abs(ref_full[8:712].astype(np.float32) - got_crop.astype(np.float32)).mean())
    assert crop_diff < 0.05, f"cropped 1280x704 render mismatch: {crop_diff}"

    # 4. Sign check: turn_right (yaw > 0) moves image content LEFT (dx < 0), matching GT & MG3.5 preset
    dyaw = math.radians(2.0)
    im0 = _opencv_pinhole_render(pano, mgs.c2w_matrix(0.0, 0.0), K, w, h)
    im_right = _opencv_pinhole_render(pano, mgs.c2w_matrix(+dyaw, 0.0), K, w, h)
    dx_meas, dy_meas = _phase_shift(im0, im_right)
    gt_flow = rotation_flow_forward(
        PanoCamera(yaw=0.0, pitch=0.0, fov=fov, width=w, height=h),
        PanoCamera(yaw=+dyaw, pitch=0.0, fov=fov, width=w, height=h),
    )
    gt_dx = float(np.nanmean(gt_flow[..., 0]))
    assert gt_dx < -5.0 and dx_meas < -5.0, f"expected leftward flow (dx < 0), got gt={gt_dx}, meas={dx_meas}"
    assert abs(dy_meas) < 1.0

    # pitch_up (pitch > 0) moves image content DOWN (dy > 0)
    dpitch = math.radians(2.0)
    im_up = _opencv_pinhole_render(pano, mgs.c2w_matrix(0.0, +dpitch), K, w, h)
    _, dy_up = _phase_shift(im0, im_up)
    gt_up = rotation_flow_forward(
        PanoCamera(yaw=0.0, pitch=0.0, fov=fov, width=w, height=h),
        PanoCamera(yaw=0.0, pitch=+dpitch, fov=fov, width=w, height=h),
    )
    gt_dy = float(np.nanmean(gt_up[..., 1]))
    assert gt_dy > 5.0 and dy_up > 5.0, f"expected downward pixel flow on pitch_up, got gt={gt_dy}, meas={dy_up}"

    # 5. Offline rollout contract via fake Space client
    class _FakeSpaceClient:
        def __init__(self, out_dir: pathlib.Path):
            self.out_dir = out_dir
            self.last_seed = None

        def predict(self, image, prompt, camera_file, camera_preset, mode, refs,
                    num_blocks, steps, cfg_scale, seed, randomize_seed, api_name=None):
            self.last_seed = seed
            cam_p = camera_file["path"] if isinstance(camera_file, dict) else camera_file
            z = np.load(cam_p)
            extr = z["extrinsics_c2w"]
            assert extr.ndim == 3 and extr.shape[1:] == (4, 4)
            dets = np.linalg.det(extr[:, :3, :3])
            assert np.allclose(dets, 1.0, atol=1e-5)
            mp4 = self.out_dir / "fake.mp4"
            vw = cv2.VideoWriter(str(mp4), cv2.VideoWriter_fourcc(*"mp4v"), 16.0, (320, 176))
            for idx in range(int(num_blocks) * 80):
                fr = np.full((176, 320, 3), idx % 255, dtype=np.uint8)
                vw.write(fr)
            vw.release()
            return str(mp4), int(seed)

    with tempfile.TemporaryDirectory(prefix="wl_test_mg35_") as tmp:
        tmp_p = pathlib.Path(tmp)
        fake = _FakeSpaceClient(tmp_p)
        adapter = MatrixGameSpace(client=fake, steps=8, out_width=320, out_height=180)
        start = np.zeros((720, 1280, 3), dtype=np.uint8)
        adapter.reset(start, prompt="corridor", seed=0)
        assert adapter.seed == 0, "seed=0 must not be ignored by reset()"
        adapter.set_pose(0.25, 0.0)
        adapter.set_fov(fov)
        info = adapter.plan([{"action": "turn_right", "seconds": 2.0}], fps=8.0)
        assert info["num_blocks"] == 1
        out_frames = adapter.step("turn_right", 2.0)
        assert len(out_frames) == 32 and out_frames[0].shape == (180, 320, 3)
        assert adapter.cost_summary()["n_steps"] == 1
        adapter.close()


def test_wan_t2v_cpu_and_helpers_without_weights() -> None:
    import inspect

    # 1. Portable DEFAULT_MODEL_ID (no hardcoded user path in source; resolves relative to repo root)
    wan_src = inspect.getsource(wan_mod)
    assert "27236" not in wan_src and "C:\\Users" not in wan_src
    assert pathlib.Path(wan_mod.DEFAULT_MODEL_ID).name == "Wan2.1-T2V-1.3B-Diffusers"

    # 2. Fake pipeline returning deterministic float RGB clips in [0, 1]
    class _FakeWanPipe:
        def __init__(self):
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            nf = kwargs["num_frames"]
            h = kwargs["height"]
            w = kwargs["width"]
            arr = np.linspace(0.2, 0.8, nf * h * w * 3, dtype=np.float32).reshape(1, nf, h, w, 3)
            return SimpleNamespace(frames=arr)

    fake_pipe = _FakeWanPipe()
    wan = WanT2VBaseline(
        model_id="fake/wan",
        steps=2,
        width=160,
        height=96,
        out_width=320,
        out_height=180,
        device="cpu",
        stitch_frames=4,
        color_anchor=True,
        max_frames=17,  # force multi-clip split on a 2s (33-frame) step
        keep_history=True,
        verbose=True,
        _pipe=fake_pipe,
    )

    # _n_frames must always satisfy 4k + 1 with k >= 1
    for secs in (0.05, 0.25, 0.5, 1.0, 2.0, 6.0):
        nf = wan._n_frames(secs)
        assert nf >= 5 and (nf - 1) % 4 == 0, f"invalid Wan frame count {nf} for {secs}s"

    # _anchor check: frame 0 equals resized prev, cross-dissolve + color anchor produce uint8 BGR
    prev = np.full((180, 320, 3), 40, dtype=np.uint8)
    raw_clip = [np.full((180, 320, 3), 200, dtype=np.uint8) for _ in range(9)]
    anchored = wan._anchor(raw_clip, prev)
    assert np.array_equal(anchored[0], prev)
    assert anchored[-1].dtype == np.uint8 and anchored[-1].shape == (180, 320, 3)
    # Color anchor shifts the overall clip mean (originally 200) down toward prev's mean (40)
    clip_mean = float(np.mean([f.mean() for f in anchored[1:]]))
    assert abs(clip_mean - 40.0) < 10.0, f"expected clip mean near 40, got {clip_mean}"

    # Full step() on CPU with verbose=True (regression check for peak=None formatting crash)
    buf = io.StringIO()
    start_bgr = np.full((180, 320, 3), 90, dtype=np.uint8)
    wan.reset(start_bgr, prompt="test room", seed=123)
    with redirect_stdout(buf):
        frames = wan.step("turn_right", 2.0)
    log_out = buf.getvalue()
    assert "peak=n/a" in log_out
    assert wan.ignored_actions == ["turn_right"]
    assert len(fake_pipe.calls) == 2  # 33 frames (k=8) split across max_frames=17 (k_max=4) -> 2 clips
    assert len(frames) > 0 and frames[0].shape == (180, 320, 3)
    cost = wan.cost_summary()
    assert cost["n_steps"] == 1 and cost["peak_vram_gb"] is None
    wan.close()


def test_yume_docstring_and_vram_recording() -> None:
    # 1. Docstring mojibake check
    assert "(→)." in (yume_mod.__doc__ or "")
    assert "鈫" not in (yume_mod.__doc__ or "")

    # 2. One-shot subprocess fallback (use_server=False) verifying peak_vram_gb propagation
    stub_code = """
import argparse, json, pathlib, numpy as np, cv2
ap = argparse.ArgumentParser()
ap.add_argument("--outdir", required=True)
ap.add_argument("--frames", type=int, default=8)
args, _ = ap.parse_known_args()
out = pathlib.Path(args.outdir)
fdir = out / "stub_frames"
fdir.mkdir(parents=True, exist_ok=True)
for i in range(args.frames):
    cv2.imwrite(str(fdir / f"{i:04d}.png"), np.full((180, 320, 3), i * 5 % 255, dtype=np.uint8))
meta = {
    "memory": {
        "peak_allocated_gb": 14.375,
        "load_peak_allocated_gb": 10.125,
        "peak_reserved_gb": 15.5,
    }
}
(out / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
"""
    with tempfile.TemporaryDirectory(prefix="wl_test_yume_") as tmp:
        stub = pathlib.Path(tmp) / "fake_yume_cli.py"
        stub.write_text(stub_code, encoding="utf-8")
        model = YumeModel(
            cli=stub,
            python=sys.executable,
            out_width=320,
            out_height=180,
            use_server=False,
        )
        try:
            start = np.zeros((180, 320, 3), dtype=np.uint8)
            model.reset(start, prompt="bakery", seed=42)
            model.plan([{"action": "turn_right", "seconds": 0.5}], fps=8.0)
            got = model.step("turn_right", 0.5)
            assert len(got) == 8 and got[0].shape == (180, 320, 3)
            cost = model.cost_summary()
            assert cost["peak_vram_gb"] == 14.375, f"expected 14.375 GB, got {cost}"
        finally:
            model.close()


def test_yume_cli_hook_tail_slice_and_t5_cpu() -> None:
    import importlib.util
    import torch

    cli_path = ROOT / "third_party" / "YUME" / "yume_cli.py"
    spec = importlib.util.spec_from_file_location("yume_cli_mod", cli_path)
    assert spec is not None and spec.loader is not None
    yume_cli = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = yume_cli
    spec.loader.exec_module(yume_cli)

    # 1. Verify make_capture_hook tail-slicing on [head16 + prev33 + tail33] (82-frame) continuations
    captured: list[torch.Tensor] = []
    hook = yume_cli.make_capture_hook(captured, chunk_frames=33, resume_from=None)

    # Chunk 0: 33 newly generated frames filled with 1.0
    c0 = torch.full((3, 33, 8, 8), 1.0)
    hook(c0, 16, "out0.mp4")

    # Chunk 1 (continue_from_last=True): [head16 (0.0) + prev33 (1.0) + tail33 (2.0)] = 82 frames
    c1 = torch.cat(
        [
            torch.full((3, 16, 8, 8), 0.0),
            torch.full((3, 33, 8, 8), 1.0),
            torch.full((3, 33, 8, 8), 2.0),
        ],
        dim=1,
    )
    hook(c1, 16, "out1.mp4")

    # Chunk 2 (continue_from_last=True): [head16 (0.0) + prev33 (2.0) + tail33 (3.0)] = 82 frames
    c2 = torch.cat(
        [
            torch.full((3, 16, 8, 8), 0.0),
            torch.full((3, 33, 8, 8), 2.0),
            torch.full((3, 33, 8, 8), 3.0),
        ],
        dim=1,
    )
    hook(c2, 16, "out2.mp4")

    assert [t.shape[1] for t in captured] == [33, 33, 33]
    video = torch.cat(captured, dim=1)[:, :96]
    assert video.shape == (3, 96, 8, 8)
    assert torch.all(video[:, :33] == 1.0), "chunk 0 corrupted"
    assert torch.all(video[:, 33:66] == 2.0), "chunk 1 repeated chunk 0 instead of slicing tail"
    assert torch.all(video[:, 66:96] == 3.0), "chunk 2 repeated previous frames instead of slicing tail"

    # Also check resume_from slices even on the first call
    cap_resumed: list[torch.Tensor] = []
    hook_resumed = yume_cli.make_capture_hook(cap_resumed, chunk_frames=33, resume_from="prev.json")
    hook_resumed(c1, 16, "resumed.mp4")
    assert cap_resumed[0].shape[1] == 33 and torch.all(cap_resumed[0] == 2.0)

    # 2. Verify pin_t5_to_cpu keeps T5 on CPU even if .to("cuda") is requested
    class _DummyT5:
        def __init__(self):
            self.device = torch.device("cpu")
            self.model = torch.nn.Linear(4, 4)

        def __call__(self, texts, device=None):
            x = torch.ones(len(texts), 4, device=device or self.device)
            return [self.model(x)]

    dummy_wan = SimpleNamespace(t5_cpu=False, text_encoder=_DummyT5())
    yume_cli.pin_t5_to_cpu(dummy_wan)
    assert dummy_wan.t5_cpu is True
    dummy_wan.text_encoder.model = dummy_wan.text_encoder.model.to("cuda")
    assert next(dummy_wan.text_encoder.model.parameters()).device.type == "cpu"
    ctx = dummy_wan.text_encoder(["pan right"], torch.device("cuda:0"))
    assert ctx[0].device.type == "cpu" and ctx[0].shape == (1, 4)

    # 3. Verify yume_cli --dry_run with --omit_numeric_params
    with tempfile.TemporaryDirectory(prefix="wl_test_dryrun_") as tmp:
        with redirect_stdout(io.StringIO()):
            rc = yume_cli.main(
                [
                    "--dry_run",
                    "--actions",
                    "None|right:33",
                    "--frames",
                    "33",
                    "--omit_numeric_params",
                    "--outdir",
                    tmp,
                ]
            )
        assert rc == 0
        dry = json.loads((pathlib.Path(tmp) / "meta_dryrun.json").read_text(encoding="utf-8"))
        assert dry["action_mapping"]["numeric_params"]["included"] is False


def test_yume_serve_daemon_and_chunk_records() -> None:
    serve_stub = """
import argparse, json, pathlib, sys, numpy as np, cv2

PREFIX = "__YUME_JSON__:"
ap = argparse.ArgumentParser()
ap.add_argument("--serve", action="store_true")
ap.add_argument("--ckpt_dir", default="")
ap.add_argument("--resolution", default="544x960")
ap.add_argument("--t5_cpu", action="store_true")
ap.add_argument("--boot_counter", required=True)
args, _ = ap.parse_known_args()

cnt_path = pathlib.Path(args.boot_counter)
n_boots = int(cnt_path.read_text()) + 1 if cnt_path.exists() else 1
cnt_path.write_text(str(n_boots))

print("[mock_yume] loading fake weights ...", flush=True)
sys.stdout.write(PREFIX + json.dumps({
    "status": "ready",
    "model_load_s": 148.5,
    "load_peak_allocated_gb": 12.25,
    "t5_cpu": args.t5_cpu,
}) + "\\n")
sys.stdout.flush()

ep_idx = 0
for raw in sys.stdin:
    line = raw.strip()
    if not line:
        continue
    req = json.loads(line)
    cmd = req.get("cmd")
    if cmd == "quit":
        sys.stdout.write(PREFIX + json.dumps({"status": "bye"}) + "\\n")
        sys.stdout.flush()
        break
    if cmd == "reset":
        sys.stdout.write(PREFIX + json.dumps({"status": "ok"}) + "\\n")
        sys.stdout.flush()
        continue
    if cmd == "episode":
        if req.get("prompt") == "__TRIGGER_ERROR__":
            sys.stdout.write(PREFIX + json.dumps({"status": "error", "error": "bad prompt"}) + "\\n")
            sys.stdout.flush()
            continue
        outdir = pathlib.Path(req["outdir"])
        fdir = outdir / "ep_frames"
        fdir.mkdir(parents=True, exist_ok=True)
        n_frames = int(req["frames"])
        for i in range(n_frames):
            cv2.imwrite(str(fdir / f"{i:04d}.png"), np.full((90, 160, 3), (ep_idx * 40 + i) % 255, dtype=np.uint8))
        chunk_times = [42.0, 44.0, 43.0]
        meta = {
            "action_mapping": {
                "numeric_params": {"included": not req.get("omit_numeric_params", False)},
                "chunks": [
                    {"chunk": 0, "frames": 33, "keys": "None", "mouse": "right"},
                    {"chunk": 1, "frames": 33, "keys": "None", "mouse": "right"},
                    {"chunk": 2, "frames": 30, "keys": "None", "mouse": "right"},
                ],
            },
            "timing": {
                "model_load_s": 148.5 if ep_idx == 0 else 0.0,
                "server_boot_load_s": 148.5,
                "generation_s": 129.0,
                "chunk_times_s": chunk_times,
            },
            "memory": {
                "peak_allocated_gb": 14.75,
                "load_peak_allocated_gb": 12.25,
            },
        }
        ep_idx += 1
        meta_path = outdir / "meta.json"
        meta_path.write_text(json.dumps(meta), encoding="utf-8")
        sys.stdout.write(PREFIX + json.dumps({
            "status": "ok",
            "meta_path": str(meta_path),
            "frames_n": n_frames,
            "generation_s": 129.0,
            "chunk_times_s": chunk_times,
            "peak_allocated_gb": 14.75,
        }) + "\\n")
        sys.stdout.flush()
"""
    YumeModel.unload()
    # the stub loads no weights, so the VRAM guard would only trip on an unrelated busy GPU
    os.environ["WORLDLOOP_MIN_FREE_VRAM_GB"] = "0"
    with tempfile.TemporaryDirectory(prefix="wl_test_yume_serve_") as tmp:
        tmp_p = pathlib.Path(tmp)
        stub = tmp_p / "mock_yume_serve.py"
        boot_counter = tmp_p / "boots.txt"
        stub.write_text(serve_stub, encoding="utf-8")

        try:
            start = np.zeros((180, 320, 3), dtype=np.uint8)
            segs = [
                {"action": "turn_right", "seconds": 2.0},
                {"action": "turn_right", "seconds": 2.0},
                {"action": "turn_right", "seconds": 2.0},
            ]

            # Episode 1 on adapter instance 1
            m1 = YumeModel(
                cli=stub,
                python=sys.executable,
                weights=tmp_p,
                resolution="544x960",
                out_width=320,
                out_height=180,
                use_server=True,
                t5_cpu=True,
                omit_numeric_params=True,
                extra_args=["--boot_counter", str(boot_counter)],
            )
            m1.reset(start, prompt="bakery 1", seed=1)
            m1.plan(segs, fps=8.0)
            f_ep1 = m1.step("turn_right", 6.0)
            assert len(f_ep1) == 96
            assert m1.episode_meta["action_mapping"]["numeric_params"]["included"] is False
            assert m1.episode_meta["model_load_s"] == 148.5

            # Per-chunk StepRecord cost accounting: mean latency must be mean([42, 44, 43]) = 43.0s
            # (NOT 148.5 + 129 = 277.5s!)
            cost1 = m1.cost_summary()
            assert cost1["n_steps"] == 3, f"expected 3 chunk StepRecords, got {cost1}"
            assert abs(cost1["latency_s_mean"] - 43.0) < 1e-6, f"inflated chunk latency: {cost1}"
            assert cost1["peak_vram_gb"] == 14.75
            m1.close()

            # Error resilience: bad request raises RuntimeError without killing the cached daemon
            m_err = YumeModel(
                cli=stub,
                python=sys.executable,
                weights=tmp_p,
                resolution="544x960",
                out_width=320,
                out_height=180,
                use_server=True,
                t5_cpu=True,
                extra_args=["--boot_counter", str(boot_counter)],
            )
            m_err.reset(start, prompt="__TRIGGER_ERROR__", seed=2)
            try:
                m_err.plan([{"action": "noop", "seconds": 2.0}], fps=8.0)
                raise AssertionError("expected RuntimeError from bad request")
            except RuntimeError as exc:
                assert "bad prompt" in str(exc)
            m_err.close()

            # Episode 2 on a brand-new YumeModel instance -> must reuse the SAME daemon (boots == 1)
            m2 = YumeModel(
                cli=stub,
                python=sys.executable,
                weights=tmp_p,
                resolution="544x960",
                out_width=320,
                out_height=180,
                use_server=True,
                t5_cpu=True,
                extra_args=["--boot_counter", str(boot_counter)],
            )
            srv = m2._get_server()
            assert srv.request({"cmd": "reset"}, timeout=10.0)["status"] == "ok"
            m2.reset(start, prompt="bakery 2", seed=3)
            m2.plan(segs, fps=8.0)
            f_ep2 = m2.step("turn_right", 6.0)
            assert len(f_ep2) == 96
            assert int(boot_counter.read_text()) == 1, "daemon reloaded instead of reusing _SERVER_CACHE!"
            m2.close()
        finally:
            YumeModel.unload()
            assert YumeModel._SERVER_CACHE == {}


def test_calibrate_gain_oracle_check() -> None:
    import time as _time
    import scripts.calibrate_gain as cg

    with tempfile.TemporaryDirectory(prefix="wl_test_cal_") as tmp:
        tmp_p = pathlib.Path(tmp)
        t0 = _time.perf_counter()
        with redirect_stdout(io.StringIO()):
            rc = cg.main(
                [
                    "--oracle-check",
                    "--include-null",
                    "--configs",
                    "omit",
                    "3,3",
                    "--out",
                    str(tmp_p),
                ]
            )
        dt = _time.perf_counter() - t0
        assert rc == 0
        assert dt < 5.0, f"--oracle-check took too long: {dt:.2f}s"
        cal_json = tmp_p / "calibration.json"
        cal_md = tmp_p / "calibration.md"
        cal_fig = tmp_p / "yume_gain_curve.png"
        assert cal_json.is_file() and cal_md.is_file() and cal_fig.is_file()
        data = json.loads(cal_json.read_text(encoding="utf-8"))
        assert len(data["results"]) == 2
        r0 = data["results"][0]
        assert r0["flow_cos"] > 0.95
        assert abs(r0["est_deg_per_s"] - 60.0) < 10.0
        assert r0["null_motion_px"] is not None and r0["null_motion_px"] < 0.2


def test_make_figures_helpers() -> None:
    import scripts.make_figures as mf

    with tempfile.TemporaryDirectory(prefix="wl_test_figs_") as tmp:
        tmp_p = pathlib.Path(tmp)

        # 1. leaderboard_bars with lowercase report.group_scores keys + None causality
        s_yume = group_scores(
            {
                "flow_cos": 0.7,
                "axis_purity": 0.9,
                "switch_latency_frames": 1.0,
                "revisit_dino": 0.4,
                "revisit_ssim": 0.5,
                "permanence_dino": 0.3,
                "null_motion_px": 2.0,
                "drift_slope_per_s": 0.01,
                "asr_mean": None,  # counterfactual not run -> causality is None
            }
        )
        lb_out = tmp_p / "leaderboard.png"
        mf.leaderboard_bars({"oracle_sim": s_yume, "yume_5b": s_yume}, lb_out)
        assert lb_out.is_file() and lb_out.stat().st_size > 1000

        # 2. contact_sheet with 1-frame rollout (regression for IndexError) and 10-frame rollout
        rdir = tmp_p / "res_m"
        for cid, nf in (("case_one", 1), ("case_ten", 10)):
            fdir = rdir / cid / "frames"
            fdir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(rdir / cid / "start.png"), np.full((90, 160, 3), 200, dtype=np.uint8))
            for i in range(nf):
                cv2.imwrite(str(fdir / f"{i:04d}.jpg"), np.full((90, 160, 3), i * 20, dtype=np.uint8))
        cs_out = tmp_p / "sheet.png"
        mf.contact_sheet({"m": rdir}, "case_one", cs_out)
        assert cs_out.is_file()

        # 3. drift_curve with unequal rollout lengths + conditioning start.png reference
        for i in range(6):
            fdir2 = rdir / "case_six" / "frames"
            fdir2.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(fdir2 / f"{i:04d}.jpg"), np.full((90, 160, 3), 50 + i * 10, dtype=np.uint8))

        orig_dino = mf.dino_embed
        seen_first_pixels: list[int] = []

        def _fake_dino(images, device="cpu", img_size=518):
            seen_first_pixels.append(int(images[0][0, 0, 0]))
            vecs = []
            for idx_im, im in enumerate(images):
                v = np.array([float(im.mean()) + 1.0, float(idx_im) + 1.0], dtype=np.float32)
                v /= np.linalg.norm(v)
                vecs.append(v)
            return np.stack(vecs, axis=0)

        mf.dino_embed = _fake_dino
        try:
            dc_out = tmp_p / "drift.png"
            mf.drift_curve({"m": rdir}, ["case_ten", "case_six"], dc_out, device="cpu")
            assert dc_out.is_file() and dc_out.stat().st_size > 1000
            # First call ("case_ten") has start.png with pixel value ~200 (not frame 0's 0)
            assert seen_first_pixels[0] > 150, f"drift_curve did not use start.png: {seen_first_pixels}"
        finally:
            mf.dino_embed = orig_dino


class AdapterTests(unittest.TestCase):
    def test_oracle_and_static(self) -> None:
        test_oracle_and_static_contracts()

    def test_matrix_game_space(self) -> None:
        test_matrix_game_space_geometry_and_rollout()

    def test_wan_t2v(self) -> None:
        test_wan_t2v_cpu_and_helpers_without_weights()

    def test_yume(self) -> None:
        test_yume_docstring_and_vram_recording()

    def test_yume_cli_hook_and_t5_cpu(self) -> None:
        test_yume_cli_hook_tail_slice_and_t5_cpu()

    def test_yume_serve_and_chunk_records(self) -> None:
        test_yume_serve_daemon_and_chunk_records()

    def test_calibrate_gain_smoke(self) -> None:
        test_calibrate_gain_oracle_check()

    def test_make_figures(self) -> None:
        test_make_figures_helpers()


def main() -> None:
    tests = [
        ("OracleModel & StaticModel contracts (+ set_pano)", test_oracle_and_static_contracts),
        ("MatrixGameSpace SO(3) geometry & rollout", test_matrix_game_space_geometry_and_rollout),
        ("WanT2VBaseline helpers & CPU safety", test_wan_t2v_cpu_and_helpers_without_weights),
        ("YumeModel docstring & one-shot peak_vram_gb", test_yume_docstring_and_vram_recording),
        ("yume_cli _hook tail-slicing & --t5_cpu pinning", test_yume_cli_hook_tail_slice_and_t5_cpu),
        ("YumeModel persistent --serve daemon & chunk StepRecords", test_yume_serve_daemon_and_chunk_records),
        ("calibrate_gain.py --oracle-check smoke test", test_calibrate_gain_oracle_check),
        ("make_figures contact_sheet / drift_curve / leaderboard_bars", test_make_figures_helpers),
    ]
    for label, fn in tests:
        fn()
        print(f"[PASS] {label}")
    print("ALL ADAPTER TESTS PASSED")


if __name__ == "__main__":
    main()
