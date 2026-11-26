"""Unit and integration tests for WorldLoop actions, cases, runner, metrics, CLI, and report."""

from __future__ import annotations

import inspect
import json
import math
import pathlib
import sys
import tempfile
import unittest
from unittest.mock import patch as mock_patch

import cv2
import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from worldloop import actions as A  # noqa: E402
from worldloop import cases as C  # noqa: E402
from worldloop import cli  # noqa: E402
from worldloop import report as R  # noqa: E402
from worldloop.adapters.base import ClosedLoopModel  # noqa: E402
from worldloop.adapters.oracle import OracleModel  # noqa: E402
from worldloop.adapters.static import StaticModel  # noqa: E402
from worldloop.metrics import features as F  # noqa: E402
from worldloop.metrics import protocol as P  # noqa: E402
from worldloop.pano import PanoCamera, render, rotation_flow_forward  # noqa: E402
from worldloop.runner import run_case  # noqa: E402


def _make_synthetic_pano(width: int = 512, height: int = 256, seed: int = 7) -> np.ndarray:
    """Deterministic textured equirectangular panorama with distinct spatial landmarks."""
    rng = np.random.default_rng(seed)
    ph = float(seed) * 1.7
    xs = np.linspace(0, 2 * math.pi, width, endpoint=False, dtype=np.float32)
    ys = np.linspace(0, math.pi, height, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    r = 127.5 + 80.0 * np.sin(3.0 * gx + ph) * np.sin(2.0 * gy)
    g = 127.5 + 80.0 * np.cos(5.0 * gx + 0.7 - ph) * np.sin(3.0 * gy)
    b = 127.5 + 80.0 * np.sin(7.0 * gx - 1.1 + 2.0 * ph) * np.cos(2.0 * gy)
    pano = np.stack([b, g, r], axis=-1).clip(0, 255).astype(np.uint8)
    for _ in range(40):
        cx = int(rng.integers(16, width - 16))
        cy = int(rng.integers(24, height - 24))
        rad = int(rng.integers(5, 16))
        color = tuple(int(v) for v in rng.integers(20, 245, size=3))
        cv2.circle(pano, (cx, cy), rad, color, -1)
    return pano


def _fast_embed(images: list[np.ndarray], device: str = "cuda", img_size: int = 518) -> np.ndarray:
    """Fast deterministic spatial-color descriptor (L2-normalised) for mocked protocol tests."""
    if not images:
        return np.zeros((0, 192), dtype=np.float32)
    vecs = []
    for im in images:
        small = cv2.resize(im, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32)
        v = small.reshape(-1) - float(small.mean())
        n = float(np.linalg.norm(v))
        vecs.append(v / (n if n > 1e-6 else 1.0))
    return np.stack(vecs, axis=0)


class _HookSpyModel(ClosedLoopModel):
    name = "hook_spy"
    native_fps = 8.0

    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple] = []
        self._frame: np.ndarray | None = None

    def reset(self, image: np.ndarray, prompt: str = "", seed: int = 0) -> None:
        self.calls.append(("reset", prompt, seed))
        self._frame = image.copy()

    def set_pose(self, yaw: float, pitch: float) -> None:
        self.calls.append(("set_pose", yaw, pitch))

    def set_fov(self, fov: float) -> None:
        self.calls.append(("set_fov", fov))

    def plan(self, segments: list[dict], fps: float) -> None:
        self.calls.append(("plan", len(segments), fps))

    def step(self, action: str, seconds: float) -> list[np.ndarray]:
        t0 = self._time()
        n = max(1, int(round(seconds * self.native_fps)))
        self.calls.append(("step", action, seconds, n))
        out = [self._frame.copy() for _ in range(n)]
        self._record(action, seconds, n, t0)
        return out

    def meta(self) -> dict:
        self.calls.append(("meta",))
        return {"spy": True}

    def close(self) -> None:
        self.calls.append(("close",))


class TestWorldLoopPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory(prefix="worldloop_test_")
        cls.root = pathlib.Path(cls._tmpdir.name)
        cls.pano_paths = []
        for idx in range(6):
            p = cls.root / f"scene_{idx:02d}.jpg"
            cv2.imwrite(str(p), _make_synthetic_pano(seed=100 + idx))
            cls.pano_paths.append(p)
        cls.panos_json = cls.root / "panos.json"
        cls.panos_json.write_text(
            json.dumps({"panoramas": [{"slug": p.stem, "path": str(p)} for p in cls.pano_paths]}),
            encoding="utf-8",
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    # ------------------------------------------------------------------ 1. actions & timeline
    def test_actions_and_timeline_compilation(self) -> None:
        with self.assertRaises(ValueError):
            A.Segment("fly_up", 2.0)
        with self.assertRaises(ValueError):
            A.Segment("noop", 0.0)
        with self.assertRaises(ValueError):
            A.compile_timeline([A.Segment("noop", 1.0)], fps=0.0)

        segs = C.PROTOCOLS["occlusion"]
        tl = A.compile_timeline(segs, fps=8.0, start_yaw=0.25, start_pitch=-0.1)
        self.assertEqual(len(tl), 48)
        self.assertEqual(tl.segment_of_frame[:16], [0] * 16)
        self.assertEqual(tl.segment_of_frame[16:32], [1] * 16)
        self.assertEqual(tl.segment_of_frame[32:48], [2] * 16)
        # +120 deg after 2s turn_right, held for 2s noop, back to start_yaw after 2s turn_left
        self.assertAlmostEqual(tl.yaw[15], 0.25 + math.radians(120.0), places=6)
        self.assertAlmostEqual(tl.yaw[31], 0.25 + math.radians(120.0), places=6)
        self.assertAlmostEqual(tl.yaw[47], 0.25, places=6)
        self.assertAlmostEqual(tl.pitch[47], -0.1, places=6)

        roundtrip = A.from_dicts(A.to_dicts(segs))
        self.assertEqual(roundtrip, segs)
        self.assertAlmostEqual(A.total_seconds(segs), 6.0)
        self.assertEqual(A.script_string(segs), "turn_right:2s > noop:2s > turn_left:2s")

    # ------------------------------------------------------------------ 2. cases & CLI defaults
    def test_gen_cases_and_cli_defaults(self) -> None:
        out_dir = self.root / "gen_cases_25"
        rc = cli.main(
            [
                "gen-cases",
                "--panos",
                str(self.panos_json),
                "--n-panos",
                "6",
                "--out",
                str(out_dir),
                "--width",
                "160",
                "--height",
                "120",
                "--fps",
                "4.0",
            ]
        )
        self.assertEqual(rc, 0)
        manifest = json.loads((out_dir / "cases.json").read_text(encoding="utf-8"))
        # 6 scenes * 3 default protocols (loop360, occlusion, null_action) + 1 drift + 2*3 cf = 25
        self.assertEqual(manifest["n_cases"], 25)
        protos = {c["protocol"] for c in manifest["cases"]}
        self.assertNotIn("look_away", protos)
        self.assertEqual(protos, {"loop360", "occlusion", "null_action", "drift", "counterfactual"})
        # Small resolution (160x120) must still extract valid salient probes
        for c in manifest["cases"]:
            self.assertGreaterEqual(len(c["probes"]), 1)
            case_obj = C.Case(**c)
            self.assertEqual(case_obj.n_frames, len(case_obj.timeline()))

        # Explicit empty protocols=[] must not fall back to default protocols
        cf_only_dir = self.root / "cf_only"
        m2 = C.build_case_set(
            [{"path": str(self.pano_paths[0])}],
            cf_only_dir,
            protocols=[],
            counterfactual_scenes=1,
            drift_scenes=0,
            width=160,
            height=120,
            fps=4.0,
        )
        self.assertEqual(m2["n_cases"], 3)

    # ------------------------------------------------------------------ 3. runner hooks & models
    def test_runner_set_fov_and_rollouts(self) -> None:
        case, start = C.build_case(
            "scene_00__loop",
            "loop360",
            str(self.pano_paths[0]),
            [A.Segment("turn_right", 1.0, "loop")],
            start_yaw=0.5,
            start_pitch=-0.05,
            fov_deg=65.0,
            width=160,
            height=120,
            fps=4.0,
        )
        cdict = case.to_dict()
        spy = _HookSpyModel()
        out_dir = self.root / "runner_spy"
        # Pass str out_dir to verify string path handling
        meta = run_case(spy, cdict, start, str(out_dir), prompt="test prompt", seed=123)
        self.assertEqual(meta["n_frames"], 4)
        self.assertEqual(meta["adapter"], {"spy": True})

        call_names = [c[0] for c in spy.calls]
        self.assertEqual(
            call_names,
            ["reset", "set_pose", "set_fov", "plan", "step", "meta", "close"],
        )
        fov_call = dict((c[0], c[1:]) for c in spy.calls)["set_fov"]
        self.assertAlmostEqual(fov_call[0], math.radians(65.0), places=6)

        # Also verify OracleModel and StaticModel produce valid artifacts
        for model_name, model in [
            ("oracle", OracleModel(cdict["pano"], fov_deg=65.0, width=160, height=120, fps=4.0)),
            ("static", StaticModel(fps=4.0)),
        ]:
            mdir = self.root / f"runner_{model_name}"
            m = run_case(model, cdict, start, mdir)
            self.assertEqual(m["n_frames"], 4)
            self.assertEqual(len(P.load_frames(mdir / cdict["case_id"])), 4)
            self.assertTrue((mdir / cdict["case_id"] / "rollout.mp4").is_file())
            self.assertTrue((mdir / cdict["case_id"] / "start.png").is_file())

    # ------------------------------------------------------------------ 4. CPU device consistency
    def test_features_device_resolution_cpu_on_cuda_host(self) -> None:
        self.assertEqual(F._resolve_device("cpu"), "cpu")
        with mock_patch.object(torch.cuda, "is_available", return_value=False):
            self.assertEqual(F._resolve_device("cuda"), "cpu")
            self.assertEqual(F._resolve_device("cuda:0"), "cpu")

        pano = cv2.imread(str(self.pano_paths[0]))
        cam0 = PanoCamera(yaw=0.0, pitch=0.0, fov=math.radians(70.0), width=160, height=128)
        cam1 = cam0.with_pose(yaw=math.radians(7.5))
        f0 = render(pano, cam0)
        f1 = render(pano, cam1)

        # Previously crashed on CUDA hosts when device="cpu" due to model/input device mismatch
        flow = F.raft_flow(f0, f1, device="cpu", max_side=160)
        self.assertEqual(flow.shape, (128, 160, 2))
        self.assertTrue(np.isfinite(flow).all())
        # Right pan moves content left -> negative mean horizontal flow
        self.assertLess(float(np.mean(flow[..., 0])), -2.0)

        emb = F.dino_embed([f0, f1], device="cpu", img_size=224)
        self.assertEqual(emb.shape[0], 2)
        norms = np.linalg.norm(emb, axis=-1)
        np.testing.assert_allclose(norms, np.ones(2), atol=1e-4)
        self.assertEqual(F.dino_embed([], device="cpu").shape, (0, 384))

    # ------------------------------------------------------------------ 5. switch_latency_frames
    def test_switch_latency_stride_invariance_and_transitions(self) -> None:
        case, start = C.build_case(
            "scene_00__cf_left",
            "counterfactual",
            str(self.pano_paths[0]),
            C.COUNTERFACTUAL["cf_left"],  # noop 2s [0..16], turn_left 2s [16..32], turn_right 2s [32..48]
            width=160,
            height=120,
            fps=8.0,
        )
        cdict = case.to_dict()
        n = case.n_frames  # 48 -> score_case uses stride=2 when flow_stride=1!
        self.assertEqual(n, 48)
        # Tag each frame with its index in pixel [0, 0] so synthetic flow knows exact frame index i
        frames = []
        for idx in range(n):
            fr = start.copy()
            fr[0, 0, 0] = idx
            frames.append(fr)

        def make_flow_fn(lag_frames: int):
            # Boundary between turn_left (seg 1, dx > 0) and turn_right (seg 2, dx < 0) is bi = 32.
            # Step bi - 1 = 31 is frames[31] -> frames[32] (1st step of seg 2).
            # With lag_frames = d, steps 31 .. 31 + d - 1 stay positive (+12 px) and step 31 + d flips to -12 px.
            flip_fi = 31 + lag_frames

            def _fake_flow(prev: np.ndarray, nxt: np.ndarray, device: str = "cuda") -> np.ndarray:
                fi = int(prev[0, 0, 0])
                h, w = prev.shape[:2]
                out = np.zeros((h, w, 2), dtype=np.float32)
                if fi < 15:
                    dx = 0.0
                elif fi < flip_fi:
                    dx = 12.0  # turn_left -> image moves right (+x)
                else:
                    dx = -12.0  # turn_right -> image moves left (-x)
                out[..., 0] = dx
                return out

            return _fake_flow

        for lag in (0, 1, 2, 3, 5):
            with mock_patch.object(P, "raft_flow", side_effect=make_flow_fn(lag)), mock_patch.object(
                P, "dino_embed", side_effect=_fast_embed
            ), mock_patch.object(P, "lpips_dist", return_value=0.01):
                s_stride1 = P.score_case(cdict, frames, start_image=start, device="cpu", flow_stride=1)
                s_stride2 = P.score_case(cdict, frames, start_image=start, device="cpu", flow_stride=2)
                s_stride4 = P.score_case(cdict, frames, start_image=start, device="cpu", flow_stride=4)
                expected = float(lag + 1)
                self.assertEqual(s_stride1.get("switch_latency_frames"), expected)
                self.assertEqual(s_stride2.get("switch_latency_frames"), expected)
                self.assertEqual(s_stride4.get("switch_latency_frames"), expected)

        # Non-switching protocols (cf_null and loop360) must NOT emit spurious switch_latency_frames
        # even if a noisy generator wobbles horizontally across chunk boundaries.
        def _wobble_flow(prev: np.ndarray, nxt: np.ndarray, device: str = "cuda") -> np.ndarray:
            fi = int(prev[0, 0, 0])
            h, w = prev.shape[:2]
            out = np.zeros((h, w, 2), dtype=np.float32)
            out[..., 0] = 8.0 if fi < 31 else -8.0
            return out

        for proto_name, segs in [("counterfactual", C.COUNTERFACTUAL["cf_null"]), ("loop360", C.PROTOCOLS["loop360"])]:
            ns_case, ns_start = C.build_case(
                f"scene_00__{proto_name}_wobble", proto_name, str(self.pano_paths[0]), segs, width=160, height=120, fps=8.0
            )
            with mock_patch.object(P, "raft_flow", side_effect=_wobble_flow), mock_patch.object(
                P, "dino_embed", side_effect=_fast_embed
            ), mock_patch.object(P, "lpips_dist", return_value=0.01):
                res = P.score_case(ns_case.to_dict(), frames, start_image=ns_start, device="cpu")
                self.assertNotIn("switch_latency_frames", res)

    # ------------------------------------------------------------------ 6. score_case revisit gating & oracle vs static
    def test_score_case_oracle_vs_static_and_revisit_gating(self) -> None:
        def _analytic_gt_flow_from_frames(case_dict: dict, rollout_frames: list[np.ndarray]):
            cams, _ = P._cams(case_dict)
            tagged = []
            for idx, fr in enumerate(rollout_frames):
                cp = fr.copy()
                cp[0, 0, 1] = idx
                tagged.append(cp)

            def _flow(prev: np.ndarray, nxt: np.ndarray, device: str = "cuda") -> np.ndarray:
                if np.array_equal(prev, nxt):
                    return np.zeros((prev.shape[0], prev.shape[1], 2), dtype=np.float32)
                fi = int(prev[0, 0, 1])
                gt = rotation_flow_forward(cams[fi], cams[fi + 1])
                return np.nan_to_num(gt, nan=0.0).astype(np.float32)

            return tagged, _flow

        specs = [
            ("loop360", C.PROTOCOLS["loop360"]),
            ("occlusion", C.PROTOCOLS["occlusion"]),
            ("null_action", C.PROTOCOLS["null_action"]),
            ("counterfactual_left", C.COUNTERFACTUAL["cf_left"]),
            ("counterfactual_null", C.COUNTERFACTUAL["cf_null"]),
        ]
        for label, segs in specs:
            proto = "counterfactual" if label.startswith("counterfactual") else label
            case, start = C.build_case(
                f"scene_00__{label}", proto, str(self.pano_paths[0]), segs, width=160, height=120, fps=8.0
            )
            cdict = case.to_dict()
            oracle = OracleModel(cdict["pano"], fov_deg=70.0, width=160, height=120, fps=8.0)
            out_dir = self.root / "score_check"
            run_case(oracle, cdict, start, out_dir)
            frames = P.load_frames(out_dir / cdict["case_id"])
            tagged, flow_fn = _analytic_gt_flow_from_frames(cdict, frames)

            with mock_patch.object(P, "raft_flow", side_effect=flow_fn), mock_patch.object(
                P, "dino_embed", side_effect=_fast_embed
            ), mock_patch.object(P, "lpips_dist", return_value=0.005):
                sc = P.score_case(cdict, tagged, start_image=start, device="cpu")

            # Strict RFC 8259 JSON check (no bare NaN / Infinity)
            json.dumps(sc, allow_nan=False)

            if label in ("null_action", "counterfactual_null"):
                self.assertNotIn("revisit_frame", sc)
                self.assertNotIn("revisit_dino", sc)
                self.assertNotIn("permanence_dino", sc)
                self.assertNotIn("axis_purity", sc)
                if label == "null_action":
                    self.assertAlmostEqual(sc["null_motion_px"], 0.0, places=5)
            else:
                # loop360, occlusion, AND cf_left must all evaluate revisit at the final frame (47), NOT frame 1!
                self.assertEqual(sc["revisit_frame"], 47)
                self.assertGreater(sc["flow_cos"], 0.99)
                self.assertGreater(sc["revisit_dino"], 0.98)
                self.assertGreater(sc["permanence_dino"], 0.95)
                if label == "counterfactual_left":
                    self.assertEqual(sc["switch_latency_frames"], 1.0)

    # ------------------------------------------------------------------ 7. score_counterfactual pre window
    def test_score_counterfactual_pre_window_and_indexing(self) -> None:
        sig = inspect.signature(P.score_counterfactual)
        self.assertEqual(sig.parameters["prefix_seconds"].default, 2.0)

        cf_root = self.root / "cf_window_test"
        cases_by_id = {}
        for name, segs in [("cf_left", C.COUNTERFACTUAL["cf_left"]), ("cf_right", C.COUNTERFACTUAL["cf_right"])]:
            cid = f"scene_00__{name}"
            case, _ = C.build_case(
                cid,
                "counterfactual",
                str(self.pano_paths[0]),
                segs,
                width=64,
                height=64,
                fps=8.0,
                tags={"pano_slug": "scene_00", "group": "scene_00__cf"},
            )
            cases_by_id[cid] = case.to_dict()
            fdir = cf_root / cid / "frames"
            fdir.mkdir(parents=True, exist_ok=True)
            for k in range(48):
                im = np.zeros((64, 64, 3), dtype=np.uint8)
                im[0, 0, 0] = k
                cv2.imwrite(str(fdir / f"{k:04d}.png"), im)

        # Controlled embeddings:
        # Pre-intervention frames 0..15 (2.0s @ 8fps): cos = 0.98 -> dist = 0.02 (including frames 0 and 8..15!)
        # Post-intervention frames 16..47: cos = 0.40 -> dist = 0.60
        # Under old prefix_seconds=1.0 bug (cut=8), frames 8..15 diluted post to (8*0.02 + 32*0.60)/40 = 0.484!
        call_idx = {"n": 0}

        def _controlled_embed(images: list[np.ndarray], device: str = "cuda", img_size: int = 518) -> np.ndarray:
            branch = call_idx["n"]
            call_idx["n"] += 1
            n = len(images)
            out = np.zeros((n, 2), dtype=np.float32)
            if branch == 0:
                out[:, 0] = 1.0
            else:
                for k in range(n):
                    c = 0.98 if k < 16 else 0.40
                    out[k, 0] = c
                    out[k, 1] = math.sqrt(max(0.0, 1.0 - c * c))
            return out

        with mock_patch.object(P, "dino_embed", side_effect=_controlled_embed):
            res = P.score_counterfactual(cf_root, cases_by_id, device="cpu")

        pair = res["scene_00__cf"]["pairs"]["scene_00__cf_left__vs__scene_00__cf_right"]
        self.assertAlmostEqual(pair["pre"], 0.02, places=5)
        self.assertAlmostEqual(pair["post"], 0.60, places=5)
        self.assertAlmostEqual(pair["asr"], 30.0, places=3)

        # Dynamic "pre" tag with 1.5s (12 frames @ 8fps) must cut at frame 12 even when prefix_seconds=2.0
        for cid in cases_by_id:
            cases_by_id[cid]["segments"][0]["seconds"] = 1.5
        call_idx["n"] = 0
        with mock_patch.object(P, "dino_embed", side_effect=_controlled_embed):
            res_15 = P.score_counterfactual(cf_root, cases_by_id, device="cpu")
        pair_15 = res_15["scene_00__cf"]["pairs"]["scene_00__cf_left__vs__scene_00__cf_right"]
        # pre is frames 0..11 (all 0.02); post is frames 12..47 (4 frames of 0.02 + 32 frames of 0.60)
        self.assertAlmostEqual(pair_15["pre"], 0.02, places=5)
        self.assertAlmostEqual(pair_15["post"], (4 * 0.02 + 32 * 0.60) / 36.0, places=5)

    # ------------------------------------------------------------------ 7b. causal attribution metric
    def test_causality_requires_both_separation_and_a_still_null_branch(self) -> None:
        """The scored quantity must be 0 for both failure modes, not an average."""
        base = np.zeros(4, dtype=np.float32)
        base[0] = 1.0
        other = np.zeros(4, dtype=np.float32)
        other[1] = 1.0
        segs = [{"tag": "pre", "seconds": 2.0},
                {"tag": "intervention", "action": "turn_left", "seconds": 2.0},
                {"tag": "return", "action": "turn_right", "seconds": 2.0}]
        null_segs = [{"tag": "pre", "seconds": 2.0},
                     {"tag": "intervention", "action": "noop", "seconds": 2.0},
                     {"tag": "return", "action": "noop", "seconds": 2.0}]
        cases = {"g__null": {"fps": 8.0, "segments": null_segs},
                 "g__left": {"fps": 8.0, "segments": segs}}

        # frozen frame: nothing separated, nothing drifted
        frozen = np.stack([base] * 48)
        g = P.cf_group_metrics({"g__null": frozen, "g__left": frozen.copy()}, cases, null_id="g__null")
        self.assertAlmostEqual(g["cf_action_sep"], 0.0, places=6)
        self.assertAlmostEqual(g["cf_null_drift"], 0.0, places=6)

        # faithful: null holds still, the commanded branch moves only while commanded
        acted = np.stack([base] * 16 + [other] * 16 + [base] * 16)
        g = P.cf_group_metrics({"g__null": frozen, "g__left": acted}, cases, null_id="g__null")
        self.assertGreater(g["cf_action_sep"], 0.9)
        self.assertAlmostEqual(g["cf_null_drift"], 0.0, places=6)
        self.assertAlmostEqual(g["cf_return_gap"], 0.0, places=6)

        # a generator that wanders everywhere: "separation" is just drift
        wander = np.stack([base] + [other] * 47)
        g = P.cf_group_metrics({"g__null": wander, "g__left": wander.copy()}, cases, null_id="g__null")
        self.assertAlmostEqual(g["cf_action_sep"], 0.0, places=6)
        self.assertGreater(g["cf_null_drift"], 0.9)

        # and the fold into the Causality axis
        self.assertGreater(R.group_scores({"cf_action_sep": 0.30, "cf_null_drift": 0.0})["causality"], 99.0)
        self.assertGreater(R.group_scores({"cf_action_sep": 0.30, "cf_null_drift": 0.9})["causality"], 0.0)
        self.assertLess(R.group_scores({"cf_action_sep": 0.30, "cf_null_drift": 0.9})["causality"], 25.0)
        self.assertEqual(R.group_scores({"cf_action_sep": 0.0, "cf_null_drift": 0.0})["causality"], 0.0)
        # pre-rewrite results fall back to the legacy ASR so old metrics still render
        self.assertAlmostEqual(R.group_scores({"asr_mean": 3.0})["causality"], 50.0, places=3)

    # ------------------------------------------------------------------ 8. report & group_scores
    def test_report_and_group_scores(self) -> None:
        empty_report = self.root / "nonexistent_subdir" / "EMPTY.md"
        R.build_report([], str(empty_report))
        self.assertIn("_no results yet_", empty_report.read_text(encoding="utf-8"))

        oracle_metrics = {
            "aggregate": {
                "per_protocol": {
                    "loop360": {
                        "flow_cos": 0.999,
                        "flow_mag_ratio": 1.0,
                        "axis_purity": 1.0,
                        "revisit_dino": 0.999,
                        "revisit_ssim": 0.99,
                        "revisit_lpips": 0.005,
                        "permanence_dino": 0.99,
                        "drift_slope_per_s": -0.002,
                    },
                    "look_away": {
                        "flow_cos": 0.998,
                        "flow_mag_ratio": 1.0,
                        "axis_purity": 1.0,
                        "revisit_dino": 0.998,
                        "revisit_ssim": 0.99,
                        "revisit_lpips": 0.005,
                        "permanence_dino": 0.99,
                        "drift_slope_per_s": -0.001,
                    },
                    "null_action": {"null_motion_px": 0.02},
                    "counterfactual": {"switch_latency_frames": 1.0},
                },
                "counterfactual": {"_summary": {"asr_mean": 100.0}},
                "cost": {"latency_s_mean": 1.5, "realtime_factor": 1.3},
            }
        }
        static_metrics = {
            "aggregate": {
                "per_protocol": {
                    "loop360": {
                        "flow_cos": 0.0,
                        "flow_mag_ratio": 0.0,
                        "revisit_dino": 1.0,
                        "revisit_ssim": 0.99,
                        "permanence_dino": 0.99,
                        "drift_slope_per_s": 0.0,
                    },
                    "null_action": {"null_motion_px": 0.0},
                }
            }
        }

        r_oracle = self.root / "rep_results" / "oracle_sim"
        r_static = self.root / "rep_results" / "static"
        r_oracle.mkdir(parents=True, exist_ok=True)
        r_static.mkdir(parents=True, exist_ok=True)
        (r_oracle / "metrics.json").write_text(json.dumps(oracle_metrics), encoding="utf-8")
        (r_static / "metrics.json").write_text(json.dumps(static_metrics), encoding="utf-8")

        row_o = R._extract(oracle_metrics)
        self.assertEqual(row_o["switch_latency_frames"], 1.0)
        self.assertAlmostEqual(row_o["flow_cos"], 0.9985, places=5)
        gs_o = R.group_scores(row_o)
        self.assertGreater(gs_o["control"], 95.0)
        self.assertGreater(gs_o["memory"], 99.0)
        self.assertGreater(gs_o["stability"], 95.0)
        self.assertGreater(gs_o["causality"], 96.0)
        self.assertGreater(gs_o["world_score"], 97.0)

        row_s = R._extract(static_metrics)
        gs_s = R.group_scores(row_s)
        self.assertEqual(gs_s["control"], 0.0)
        self.assertGreater(gs_s["memory"], 99.0)
        self.assertIsNone(gs_s["causality"])
        self.assertEqual(gs_s["world_score"], 0.0)

        out_md = self.root / "rep_results" / "LEADERBOARD.md"
        R.build_report([str(r_oracle), str(r_static)], str(out_md))
        md_text = out_md.read_text(encoding="utf-8")
        # Columns: Flow dir. cos (0.9985 -> 0.998), Flow mag. ratio (1.00), Axis purity (1.00), Switch latency (1.00)
        self.assertIn("| `oracle_sim` | 0.998 | 1.00 | 1.00 | 1.00 |", md_text)
        self.assertIn("look_away", md_text)

    # ------------------------------------------------------------------ 9. CLI run + score + report
    def test_cli_end_to_end_pipeline(self) -> None:
        cdir = self.root / "cli_e2e" / "cases"
        rdir = self.root / "cli_e2e" / "results" / "oracle_sim"
        md_out = self.root / "cli_e2e" / "RESULTS.md"

        C.build_case_set(
            [{"path": str(self.pano_paths[0])}],
            cdir,
            protocols=["occlusion", "null_action"],
            counterfactual_scenes=0,
            drift_scenes=0,
            width=160,
            height=120,
            fps=4.0,
        )
        # Omit --cases-dir to verify cmd_run resolves cases_dir from --cases parent
        self.assertEqual(
            cli.main(["run", "--model", "oracle", "--cases", str(cdir / "cases.json"), "--out", str(rdir)]),
            0,
        )
        with mock_patch.object(
            P, "raft_flow", return_value=np.zeros((120, 160, 2), dtype=np.float32)
        ), mock_patch.object(P, "dino_embed", side_effect=_fast_embed), mock_patch.object(
            P, "lpips_dist", return_value=0.01
        ):
            self.assertEqual(
                cli.main(["score", "--results", str(rdir), "--cases", str(cdir / "cases.json"), "--device", "cpu"]),
                0,
            )

        raw = (rdir / "metrics.json").read_text(encoding="utf-8")
        # Must be strict valid JSON (no NaN)
        parsed = json.loads(raw, parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"invalid JSON constant {c}")))
        self.assertEqual(len(parsed["per_case"]), 2)
        self.assertIn("cost", parsed["aggregate"])

        self.assertEqual(cli.main(["report", "--results", str(rdir), "--out", str(md_out)]), 0)
        self.assertTrue(md_out.is_file())

    # ------------------------------------------------------------------ 10. trajectory-aware revisit
    def test_trajectory_aware_revisit_disentangles_gain_vs_hallucination(self) -> None:
        pano0 = cv2.imread(str(self.pano_paths[0]))
        pano1 = cv2.imread(str(self.pano_paths[4]))
        case, start = C.build_case(
            "scene_00__underrot_loop",
            "loop360",
            str(self.pano_paths[0]),
            C.PROTOCOLS["loop360"],  # commanded 360 deg over 48 frames @ 8fps (7.5 deg/frame)
            width=160,
            height=120,
            fps=8.0,
        )
        cdict = case.to_dict()
        gain = 2.0 / 3.0  # 40 deg/s -> 240 deg total instead of 360 deg
        step_rad = math.radians(60.0 * gain / 8.0)

        cams_under = [
            PanoCamera(yaw=float(cdict["start_yaw"]) + (k + 1) * step_rad, pitch=0.0, fov=cdict["fov"], width=160, height=120)
            for k in range(48)
        ]
        # Scenario A: faithful 3D world memory, just 0.67x angular gain (ends at 240 deg)
        frames_faithful = []
        for idx, cam in enumerate(cams_under):
            fr = render(pano0, cam)
            fr[0, 0, 2] = idx
            frames_faithful.append(fr)

        # Scenario B: same 0.67x angular motion, but hallucinates a different room (pano1) by the end
        frames_halluc = []
        for idx, cam in enumerate(cams_under):
            src = pano0 if idx < 16 else pano1
            fr = render(src, cam)
            fr[0, 0, 2] = idx
            frames_halluc.append(fr)

        start_cam = PanoCamera(yaw=float(cdict["start_yaw"]), pitch=0.0, fov=cdict["fov"], width=160, height=120)
        start_tagged = start.copy()
        start_tagged[0, 0, 2] = 255

        def _underrot_flow(prev: np.ndarray, nxt: np.ndarray, device: str = "cuda") -> np.ndarray:
            tag = int(prev[0, 0, 2])
            c0 = start_cam if tag == 255 else cams_under[tag]
            c1 = cams_under[0] if tag == 255 else cams_under[tag + 1]
            return np.nan_to_num(rotation_flow_forward(c0, c1), nan=0.0).astype(np.float32)

        with mock_patch.object(P, "raft_flow", side_effect=_underrot_flow), mock_patch.object(
            P, "dino_embed", side_effect=_fast_embed
        ), mock_patch.object(P, "lpips_dist", return_value=0.05):
            sc_s1 = P.score_case(cdict, frames_faithful, start_image=start_tagged, device="cpu", flow_stride=1)
            sc_s2 = P.score_case(cdict, frames_faithful, start_image=start_tagged, device="cpu", flow_stride=2)
            sc_hal = P.score_case(cdict, frames_halluc, start_image=start_tagged, device="cpu", flow_stride=2)

        self.assertAlmostEqual(sc_s1["commanded_yaw_deg"], 360.0, places=3)
        self.assertAlmostEqual(sc_s2["commanded_yaw_deg"], 360.0, places=3)
        self.assertAlmostEqual(sc_s1["integrated_yaw_deg"], 240.0, delta=1.0)
        self.assertAlmostEqual(sc_s2["integrated_yaw_deg"], 240.0, delta=1.0)

        # Standard revisit_dino is low because 240 deg != 0 deg, whereas revisit_dino_est_pose recovers ~1.0!
        self.assertLess(sc_s2["revisit_dino"], 0.5)
        self.assertGreater(sc_s2["revisit_dino_est_pose"], 0.99)
        self.assertGreater(sc_s2["revisit_ssim_est_pose"], 0.95)
        # Hallucinating a different room fails BOTH revisit_dino and revisit_dino_est_pose
        self.assertLess(sc_hal["revisit_dino_est_pose"], sc_s2["revisit_dino_est_pose"] - 0.3)

    # ------------------------------------------------------------------ 11. multi-loop drift
    def test_multi_loop_drift_metrics(self) -> None:
        pano0 = cv2.imread(str(self.pano_paths[0]))
        case, start = C.build_case(
            "scene_00__drift",
            "drift",
            str(self.pano_paths[0]),
            C.PROTOCOLS["drift"],  # 18s -> 3 loops of 360 deg (72 frames @ 4fps: loops at 23, 47, 71)
            width=160,
            height=120,
            fps=4.0,
        )
        cdict = case.to_dict()
        cams, _ = P._cams(cdict)
        self.assertEqual(len(cams), 72)

        rng = np.random.default_rng(99)
        frames = []
        for idx, cam in enumerate(cams):
            fr = render(pano0, cam).astype(np.float32)
            # Inject increasing noise per loop so loop1 > loop2 > loop3
            loop_no = idx // 24
            if loop_no > 0:
                fr = np.clip(fr + rng.normal(0.0, 25.0 * loop_no, size=fr.shape), 0, 255)
            frames.append(fr.astype(np.uint8))

        with mock_patch.object(
            P, "raft_flow", return_value=np.zeros((120, 160, 2), dtype=np.float32)
        ), mock_patch.object(P, "dino_embed", side_effect=_fast_embed), mock_patch.object(
            P, "lpips_dist", return_value=0.02
        ):
            full = P.score_case(cdict, frames, start_image=start, device="cpu", flow_stride=4)
            trunc = P.score_case(cdict, frames[:50], start_image=start, device="cpu", flow_stride=4)

        self.assertIn("drift_loop1_dino", full)
        self.assertIn("drift_loop2_dino", full)
        self.assertIn("drift_loop3_dino", full)
        self.assertGreater(full["drift_loop1_dino"], full["drift_loop2_dino"])
        self.assertGreater(full["drift_loop2_dino"], full["drift_loop3_dino"])

        # Truncated rollout (50 frames @ 4fps = 12.5s) covers loop 1 & 2, omits loop 3
        self.assertIn("drift_loop1_dino", trunc)
        self.assertIn("drift_loop2_dino", trunc)
        self.assertNotIn("drift_loop3_dino", trunc)

    # ------------------------------------------------------------------ 12. --pano-slugs & model reuse/shutdown
    def test_pano_slugs_and_model_reuse_shutdown(self) -> None:
        ood_dir = self.root / "ood_slugs"
        rc = cli.main(
            [
                "gen-cases",
                "--panos",
                str(self.panos_json),
                "--pano-slugs",
                "scene_05,scene_02",
                "--protocols",
                "loop360",
                "null_action",
                "--cf-scenes",
                "1",
                "--drift-scenes",
                "0",
                "--width",
                "160",
                "--height",
                "120",
                "--fps",
                "4.0",
                "--out",
                str(ood_dir),
            ]
        )
        self.assertEqual(rc, 0)
        manifest = json.loads((ood_dir / "cases.json").read_text(encoding="utf-8"))
        # 2 scenes * 2 protocols + 1 * 3 cf = 7 cases, preserving requested slug order
        self.assertEqual(manifest["n_cases"], 7)
        self.assertEqual([p["slug"] for p in manifest["panos"]], ["scene_05", "scene_02"])
        cf_ids = [c["case_id"] for c in manifest["cases"] if c["protocol"] == "counterfactual"]
        self.assertTrue(all(cid.startswith("scene_05__") for cid in cf_ids))

        # Verify cmd_run builds a heavy/non-oracle model only ONCE across multiple cases and calls shutdown()
        created_instances: list[ClosedLoopModel] = []

        class _ReusableDummy(_HookSpyModel):
            def __init__(self) -> None:
                super().__init__()
                self.shutdown_called = 0
                created_instances.append(self)

            def shutdown(self) -> None:
                self.shutdown_called += 1

        run_out = self.root / "reuse_out"
        with mock_patch.object(cli, "_build_model", side_effect=lambda name, case, args: _ReusableDummy()):
            rc_run = cli.main(
                [
                    "run",
                    "--model",
                    "yume",
                    "--cases",
                    str(ood_dir / "cases.json"),
                    "--max-cases",
                    "3",
                    "--out",
                    str(run_out),
                ]
            )
        self.assertEqual(rc_run, 0)
        self.assertEqual(len(created_instances), 1)
        self.assertEqual(created_instances[0].shutdown_called, 1)

    # ------------------------------------------------------------------ 13. exporter integrity
    def test_frame_integrity_flags_rewound_chunks(self) -> None:
        """A chunked exporter that replays a stale buffer must be detected.

        The v0.1 Yume results were corrupted this way: a swallowed chunk boundary
        repeated the previous frame and then rewound to frame 0, so the rollout
        still looked plausible while half its commanded motion was missing.  A
        frozen-frame baseline is a legitimate still, so ``static`` must not flag.
        """
        pano0 = cv2.imread(str(self.pano_paths[0]))
        case, start = C.build_case(
            "scene_00__loop360",
            "loop360",
            str(self.pano_paths[0]),
            C.PROTOCOLS["loop360"],
            width=160,
            height=120,
            fps=8.0,
        )
        cdict = case.to_dict()
        cams, _ = P._cams(cdict)
        self.assertGreater(len(cams), 40)
        clean = [render(pano0, cam) for cam in cams]

        good = P.frame_integrity(clean, cams)
        self.assertEqual(good["rewound_frames"], 0)
        self.assertEqual(good["dup_frames"], 0)

        # the v0.1 bug: freeze at 17..23, then replay 0..7 in place of 24..31
        broken = list(clean)
        for i in range(17, 24):
            broken[i] = clean[16]
        for i in range(8):
            broken[24 + i] = clean[i]
        bad = P.frame_integrity(broken, cams)
        self.assertGreaterEqual(bad["rewound_frames"], 8)
        self.assertGreater(bad["rewound_frame_ratio"], 0.15)
        self.assertGreater(bad["rewound_motion_deg"], 30.0)

        # holding a still frame while the camera is commanded to stay put is legal
        pano_still = cv2.imread(str(self.pano_paths[0]))
        still_case, _ = C.build_case(
            "scene_00__null_action",
            "null_action",
            str(self.pano_paths[0]),
            C.PROTOCOLS["null_action"],
            width=160,
            height=120,
            fps=8.0,
        )
        still_cams, _ = P._cams(still_case.to_dict())
        holding = [render(pano_still, still_cams[0]) for _ in still_cams]
        still = P.frame_integrity(holding, still_cams)
        self.assertEqual(still["dup_frames"], len(still_cams) - 1)
        self.assertEqual(still["rewound_frames"], 0)

        self.assertTrue(StaticModel.intentional_freeze)
        self.assertFalse(OracleModel.intentional_freeze)


if __name__ == "__main__":
    unittest.main()
