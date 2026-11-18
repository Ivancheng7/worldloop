"""Sweep Yume-5B numeric camera-gain prompt parameters on a single 2 s turn_right chunk.

Measures optical-flow direction agreement (`flow_cos`), gain ratio (`flow_mag_ratio`),
estimated angular rate (`est_deg_per_s = flow_mag_ratio * 60`), and optional null-action
drift (`null_motion_px`), then writes `calibration.json`, `calibration.md`, and
`docs/figures/yume_gain_curve.png`.

Smoke test (<2 s, no GPU weights needed):
    python scripts/calibrate_gain.py --oracle-check --out scratch/cal_smoke
"""

from __future__ import annotations

import argparse
import dataclasses
import functools
import json
import math
import pathlib
import sys
import time

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldloop import actions as A  # noqa: E402
from worldloop.adapters.oracle import OracleModel  # noqa: E402
from worldloop.adapters.yume import YumeModel  # noqa: E402
from worldloop.pano import PanoCamera, load_pano, render, rotation_flow_forward  # noqa: E402
from worldloop.runner import _resample  # noqa: E402

DEFAULT_CONFIGS = ["omit", "1,1", "3,3", "5,5", "8,8"]
DEFAULT_OUT_REL = pathlib.Path("results") / "yume_gain_calibration"
DEFAULT_FIG_REL = pathlib.Path("docs") / "figures" / "yume_gain_curve.png"
TARGET_DEG_PER_S = 60.0


@dataclasses.dataclass(frozen=True)
class GainConfig:
    label: str
    omit_numeric_params: bool
    distance: float = 4.0
    angular: float = 0.0
    rotation: float = 0.0


def parse_gain_configs(items: list[str]) -> list[GainConfig]:
    tokens: list[str] = []
    for raw in items:
        for part in raw.replace(";", " ").split():
            if part.strip():
                tokens.append(part.strip())
    out: list[GainConfig] = []
    for tok in tokens:
        if tok.lower() in ("omit", "none"):
            out.append(GainConfig(label="omit", omit_numeric_params=True))
            continue
        nums = [float(x) for x in tok.split(",")]
        if len(nums) == 2:
            ang, rot = nums
            dist = 4.0
            label = f"a={ang:g},r={rot:g}"
        elif len(nums) == 3:
            dist, ang, rot = nums
            label = f"d={dist:g},a={ang:g},r={rot:g}"
        else:
            raise ValueError(
                f"invalid gain config {tok!r}; expected 'omit', '<angular>,<rotation>', "
                f"or '<distance>,<angular>,<rotation>'"
            )
        out.append(
            GainConfig(
                label=label,
                omit_numeric_params=False,
                distance=dist,
                angular=ang,
                rotation=rot,
            )
        )
    if not out:
        raise ValueError("at least one gain config is required")
    return out


def _synthetic_pano(width: int = 1024, height: int = 512) -> np.ndarray:
    u = np.linspace(-math.pi, math.pi, width, endpoint=False, dtype=np.float32)
    v = np.linspace(math.pi / 2, -math.pi / 2, height, endpoint=False, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)
    b = 127.5 + 90.0 * np.sin(3.0 * uu) * np.cos(2.0 * vv) + 30.0 * np.cos(11.0 * uu)
    g = 127.5 + 90.0 * np.cos(5.0 * uu + 0.7) * np.cos(3.0 * vv) + 30.0 * np.sin(7.0 * vv)
    r = 127.5 + 90.0 * np.sin(2.0 * uu - 1.1) * np.sin(4.0 * vv) + 30.0 * np.sin(13.0 * uu * vv)
    return np.clip(np.stack([b, g, r], axis=-1), 0, 255).astype(np.uint8)


@functools.lru_cache(maxsize=1)
def _dis_instance():
    if hasattr(cv2, "DISOpticalFlow_create"):
        return cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)
    return None


def _dis_flow(prev: np.ndarray, nxt: np.ndarray, max_side: int = 480) -> np.ndarray:
    """Fast CPU variational Dense Inverse Search optical flow (no PyTorch weights needed)."""
    h0, w0 = prev.shape[:2]
    scale = min(1.0, max_side / max(h0, w0))
    ww = max(16, int(round(w0 * scale)))
    hh = max(16, int(round(h0 * scale)))
    g0 = cv2.cvtColor(cv2.resize(prev, (ww, hh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    g1 = cv2.cvtColor(cv2.resize(nxt, (ww, hh), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
    dis = _dis_instance()
    if dis is not None:
        fl = dis.calc(g0, g1, None)
    else:
        fl = cv2.calcOpticalFlowFarneback(g0, g1, None, 0.5, 5, 31, 4, 7, 1.5, 0)
    fl = fl * np.array([w0 / ww, h0 / hh], dtype=np.float32)
    return cv2.resize(fl, (w0, h0), interpolation=cv2.INTER_LINEAR)


def _predict_flow(prev: np.ndarray, nxt: np.ndarray, backend: str, device: str) -> np.ndarray:
    if backend in ("dis", "farneback"):
        return _dis_flow(prev, nxt)
    from worldloop.metrics.features import raft_flow

    return raft_flow(prev, nxt, device=device)


def score_turn_chunk(
    frames: list[np.ndarray],
    *,
    fov: float,
    fps: float = 8.0,
    start_yaw: float = 0.0,
    start_pitch: float = 0.0,
    flow_backend: str = "raft",
    device: str = "cuda",
) -> dict[str, float]:
    h, w = frames[0].shape[:2]
    seconds = len(frames) / fps
    tl = A.compile_timeline([A.Segment("turn_right", seconds)], fps, start_yaw, start_pitch)
    n = min(len(frames), len(tl))
    cams = [PanoCamera(yaw=tl.yaw[i], pitch=tl.pitch[i], fov=fov, width=w, height=h) for i in range(n)]

    rs: list[float] = []
    ratios: list[float] = []
    for i in range(n - 1):
        gt = rotation_flow_forward(cams[i], cams[i + 1])
        pred = _predict_flow(frames[i], frames[i + 1], backend=flow_backend, device=device)
        valid = np.isfinite(gt).all(-1)
        gmag = np.linalg.norm(np.nan_to_num(gt), axis=-1)
        valid &= gmag > 1.5
        if int(valid.sum()) < 50:
            continue
        p = pred[valid]
        g = gt[valid]
        pn = np.linalg.norm(p, axis=-1)
        gn = np.linalg.norm(g, axis=-1)
        den = float(np.sum(pn * gn))
        if den > 1e-6:
            rs.append(float(np.sum(p * g) / den))
        if pn.size:
            ratios.append(float(np.median(pn / (gn + 1e-9))))

    signed_cos = float(np.mean(rs)) if rs else 0.0
    flow_cos = max(0.0, signed_cos)
    mag_ratio = float(np.median(ratios)) if ratios else 0.0
    return {
        "flow_cos": round(flow_cos, 4),
        "flow_cos_signed": round(signed_cos, 4),
        "flow_mag_ratio": round(mag_ratio, 4),
        "est_deg_per_s": round(mag_ratio * TARGET_DEG_PER_S, 2),
    }


def score_null_chunk(
    frames: list[np.ndarray], *, flow_backend: str = "raft", device: str = "cuda"
) -> float:
    mags = []
    for i in range(len(frames) - 1):
        pred = _predict_flow(frames[i], frames[i + 1], backend=flow_backend, device=device)
        dx = float(np.mean(pred[..., 0]))
        dy = float(np.mean(pred[..., 1]))
        mags.append(math.hypot(dx, dy))
    return round(float(np.mean(mags)) if mags else 0.0, 4)


def build_markdown_table(rows: list[dict], recommended: str | None, include_null: bool) -> str:
    lines = [
        "# Yume-5B Camera Gain Calibration",
        "",
        f"Target angular speed: **{TARGET_DEG_PER_S:.0f} deg/s** (`turn_right`, 2.0 s / 1 chunk).",
        "",
    ]
    if include_null:
        lines.append("| config | omit | angular | rotation | flow_cos | flow_mag_ratio | est deg/s | null px | wall s |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
    else:
        lines.append("| config | omit | angular | rotation | flow_cos | flow_mag_ratio | est deg/s | wall s |")
        lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        mark = " **(rec)**" if r["label"] == recommended else ""
        base = (
            f"| `{r['label']}`{mark} | {r['omit_numeric_params']} | {r['angular']:g} | "
            f"{r['rotation']:g} | {r['flow_cos']:.3f} | {r['flow_mag_ratio']:.3f} | "
            f"{r['est_deg_per_s']:.1f} |"
        )
        if include_null:
            null_s = f"{r['null_motion_px']:.2f}" if r.get("null_motion_px") is not None else "-"
            lines.append(f"{base} {null_s} | {r['wall_s']:.1f} |")
        else:
            lines.append(f"{base} {r['wall_s']:.1f} |")
    if recommended:
        lines += ["", f"Recommended config: **`{recommended}`**."]
    return "\n".join(lines) + "\n"


def plot_gain_curve(rows: list[dict], out_fig: pathlib.Path, include_null: bool) -> None:
    labels = [r["label"] for r in rows]
    x = np.arange(len(labels))
    deg_s = [r["est_deg_per_s"] for r in rows]
    fcos = [r["flow_cos"] for r in rows]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.4, 3.3), dpi=160)
    ax1.plot(x, deg_s, marker="o", lw=1.8, color="#1f77b4", label="estimated deg/s")
    ax1.axhline(TARGET_DEG_PER_S, ls="--", lw=1.2, color="#d62728", label="commanded 60 deg/s")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, fontsize=8)
    ax1.set_ylabel("estimated yaw rate (deg/s)")
    ax1.set_title("Effective Turn Gain")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=7)

    ax2.bar(x, fcos, width=0.55, color="#2ca02c", alpha=0.85, label="flow_cos")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=20, fontsize=8)
    ax2.set_ylim(0.0, 1.08)
    ax2.set_ylabel("direction agreement (flow_cos)")
    ax2.set_title("Direction Fidelity" + (" & Null Motion" if include_null else ""))
    ax2.grid(axis="y", alpha=0.3)
    if include_null and any(r.get("null_motion_px") is not None for r in rows):
        ax2b = ax2.twinx()
        nulls = [r.get("null_motion_px") or 0.0 for r in rows]
        ax2b.plot(x, nulls, marker="s", lw=1.4, color="#ff7f0e", label="null px/frame")
        ax2b.set_ylabel("null motion (px/frame)")
        lines1, labels1 = ax2.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="lower right")
    else:
        ax2.legend(fontsize=7, loc="lower right")

    fig.tight_layout()
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Calibrate Yume-5B camera turn gain on abandoned_bakery")
    ap.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                    help="gain configs: 'omit' or '<angular>,<rotation>'")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--resolution", default="544x960", choices=["704x1280", "544x960"])
    ap.add_argument("--python", default=str(ROOT / ".venv_yume" / "Scripts" / "python.exe"))
    ap.add_argument("--weights", default=str(ROOT / "models" / "Yume-5B-720P"))
    ap.add_argument("--start-image", default=str(ROOT / "cases" / "start" / "abandoned_bakery__loop360.png"))
    ap.add_argument("--pano", default=str(ROOT / "data" / "panos" / "abandoned_bakery.jpg"))
    ap.add_argument("--prompt", default="First-person perspective of an abandoned bakery.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--t5-cpu", action="store_true", help="keep Yume's 10.6 GB T5 encoder on CPU")
    ap.add_argument("--no-server", action="store_true", help="disable persistent --serve daemon")
    ap.add_argument("--include-null", action="store_true", help="also evaluate a 2 s noop chunk per config")
    ap.add_argument("--oracle-check", "--mock", dest="oracle_check", action="store_true",
                    help="smoke-test calibration pipeline with OracleModel (<2 s, no GPU weights)")
    ap.add_argument("--flow-backend", choices=["auto", "raft", "dis", "farneback"], default="auto")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--out", default=None, help="output directory (default: results/yume_gain_calibration)")
    ap.add_argument("--fig", default=None, help="output figure path (default: docs/figures/yume_gain_curve.png)")
    args = ap.parse_args(argv)

    configs = parse_gain_configs(args.configs)
    custom_out = args.out is not None
    out_dir = pathlib.Path(args.out) if custom_out else (ROOT / DEFAULT_OUT_REL)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.fig is not None:
        fig_path = pathlib.Path(args.fig)
    elif args.oracle_check or custom_out:
        fig_path = out_dir / "yume_gain_curve.png"
    else:
        fig_path = ROOT / DEFAULT_FIG_REL

    fov = math.radians(70.0)
    target_fps = 8.0
    target_frames = 16  # 2.0 s at 8 fps
    chunk_seconds = 33.0 / 16.0  # 1 native 33-frame chunk at 16 fps

    if args.flow_backend == "auto":
        flow_backend = "dis" if args.oracle_check else "raft"
    else:
        flow_backend = args.flow_backend

    if args.device == "auto":
        if flow_backend in ("dis", "farneback"):
            device = "cpu"
        else:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    rows: list[dict] = []
    if args.oracle_check:
        w, h = 480, 270
        pano_p = pathlib.Path(args.pano)
        if pano_p.is_file():
            pano_tmp = pano_p
        else:
            pano_tmp = out_dir / "_synth_pano.png"
            cv2.imwrite(str(pano_tmp), _synthetic_pano())
        oracle = OracleModel(str(pano_tmp), fov_deg=70.0, width=w, height=h, fps=target_fps)
        if oracle.pano.shape[1] > 1024:
            oracle.pano = cv2.resize(oracle.pano, (1024, 512), interpolation=cv2.INTER_AREA)
        start_img = render(
            oracle.pano,
            PanoCamera(yaw=0.0, pitch=0.0, fov=fov, width=w, height=h),
        )
        for cfg in configs:
            t0 = time.perf_counter()
            oracle.reset(start_img, prompt=args.prompt, seed=args.seed)
            got = oracle.step("turn_right", 2.0)
            frames_8fps = _resample(got, target_frames)
            metrics = score_turn_chunk(
                frames_8fps, fov=fov, fps=target_fps, flow_backend=flow_backend, device=device
            )
            null_px = None
            if args.include_null:
                oracle.reset(start_img, prompt=args.prompt, seed=args.seed)
                got_null = oracle.step("noop", 2.0)
                null_px = score_null_chunk(
                    _resample(got_null, target_frames), flow_backend=flow_backend, device=device
                )
            wall_s = time.perf_counter() - t0
            rows.append(
                {
                    "label": cfg.label,
                    "omit_numeric_params": cfg.omit_numeric_params,
                    "distance": cfg.distance,
                    "angular": cfg.angular,
                    "rotation": cfg.rotation,
                    **metrics,
                    "null_motion_px": null_px,
                    "wall_s": round(wall_s, 3),
                }
            )
    else:
        start_p = pathlib.Path(args.start_image)
        start_img = cv2.imread(str(start_p))
        if start_img is None:
            raise FileNotFoundError(f"missing start image: {start_p}")
        h, w = start_img.shape[:2]
        model = YumeModel(
            python=args.python,
            weights=args.weights,
            steps=args.steps,
            resolution=args.resolution,
            out_width=w,
            out_height=h,
            chunk_frames=33,
            use_server=not args.no_server,
            t5_cpu=args.t5_cpu,
        )
        try:
            for idx_c, cfg in enumerate(configs):
                model.distance = cfg.distance
                model.angular = cfg.angular
                model.rotation = cfg.rotation
                model.omit_numeric_params = cfg.omit_numeric_params

                t0 = time.perf_counter()
                model.reset(start_img, prompt=args.prompt, seed=args.seed)
                model.plan([{"action": "turn_right", "seconds": chunk_seconds}], fps=target_fps)
                got = model.step("turn_right", chunk_seconds)
                frames_8fps = _resample(got, target_frames)
                cdir = out_dir / f"cfg_{idx_c:02d}_{cfg.label.replace(',', '_').replace('=', '')}"
                fdir = cdir / "frames"
                fdir.mkdir(parents=True, exist_ok=True)
                for fi, fr in enumerate(frames_8fps):
                    cv2.imwrite(str(fdir / f"{fi:04d}.jpg"), fr, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

                metrics = score_turn_chunk(
                    frames_8fps, fov=fov, fps=target_fps, flow_backend=flow_backend, device=device
                )
                null_px = None
                if args.include_null:
                    model.reset(start_img, prompt=args.prompt, seed=args.seed)
                    model.plan([{"action": "noop", "seconds": chunk_seconds}], fps=target_fps)
                    got_null = model.step("noop", chunk_seconds)
                    null_px = score_null_chunk(
                        _resample(got_null, target_frames), flow_backend=flow_backend, device=device
                    )
                wall_s = time.perf_counter() - t0
                cost = model.cost_summary()
                model.close()
                rows.append(
                    {
                        "label": cfg.label,
                        "omit_numeric_params": cfg.omit_numeric_params,
                        "distance": cfg.distance,
                        "angular": cfg.angular,
                        "rotation": cfg.rotation,
                        **metrics,
                        "null_motion_px": null_px,
                        "wall_s": round(wall_s, 2),
                        "cost": cost,
                    }
                )
        finally:
            model.close()
            YumeModel.shutdown()

    viable = [r for r in rows if r["flow_cos"] >= 0.4] or rows
    best = min(viable, key=lambda r: abs(r["est_deg_per_s"] - TARGET_DEG_PER_S))
    recommended = best["label"] if best else None

    summary = {
        "mode": "oracle_check" if args.oracle_check else "yume_5b",
        "target_deg_per_s": TARGET_DEG_PER_S,
        "steps": args.steps,
        "resolution": args.resolution,
        "flow_backend": flow_backend,
        "recommended": recommended,
        "results": rows,
    }
    json_path = out_dir / "calibration.json"
    md_path = out_dir / "calibration.md"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    md_text = build_markdown_table(rows, recommended=recommended, include_null=args.include_null)
    md_path.write_text(md_text, encoding="utf-8")
    plot_gain_curve(rows, fig_path, include_null=args.include_null)

    print(md_text, end="")
    print(f"[calibrate] json -> {json_path}")
    print(f"[calibrate] fig  -> {fig_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
