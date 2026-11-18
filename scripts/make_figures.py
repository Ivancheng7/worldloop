"""Figures for the report: contact sheets, drift curves, leaderboard bars."""

from __future__ import annotations

import json
import os
import pathlib
import sys

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from worldloop.metrics.features import dino_embed  # noqa: E402
from worldloop.metrics.protocol import load_frames  # noqa: E402
from worldloop.report import _extract, group_scores  # noqa: E402

FIG = ROOT / "docs" / "figures"

GROUP_SPEC = [
    ("control", "Control"),
    ("memory", "Memory"),
    ("stability", "Stability"),
    ("causality", "Causality"),
]


def _default_device() -> str:
    override = os.environ.get("WORLDLOOP_DEVICE")
    if override:
        return override
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _case_fps(case_dir: pathlib.Path, default: float = 8.0) -> float:
    meta_path = case_dir / "meta.json"
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(meta.get("fps"), (int, float)) and meta["fps"] > 0:
                return float(meta["fps"])
        except Exception:  # noqa: BLE001
            pass
    return default


def _load_start_image(
    rdir: pathlib.Path, cid: str, cases_dir: pathlib.Path | None = None
) -> np.ndarray | None:
    cands = [rdir / cid / "start.png"]
    if cases_dir is not None:
        cands.append(cases_dir / "start" / f"{cid}.png")
    cands.append(ROOT / "cases" / "start" / f"{cid}.png")
    for p in cands:
        if p.exists():
            img = cv2.imread(str(p))
            if img is not None:
                return img
    return None


def _label(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 26), (0, 0, 0), -1)
    cv2.putText(out, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def contact_sheet(
    results: dict[str, pathlib.Path], case_id: str, out: pathlib.Path, n_cols: int = 4
) -> None:
    rows = []
    for model, rdir in results.items():
        case_dir = rdir / case_id
        frames = load_frames(case_dir)
        if not frames:
            continue
        fps = _case_fps(case_dir)
        idx = np.round(np.linspace(0, len(frames) - 1, n_cols)).astype(int).tolist()
        tiles = [
            _label(cv2.resize(frames[i], (320, 180)), f"{model} t={(i + 1) / fps:.1f}s")
            for i in idx
        ]
        rows.append(np.hstack(tiles))
    if not rows:
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))


def drift_curve(
    results: dict[str, pathlib.Path],
    case_ids: list[str],
    out: pathlib.Path,
    cases_dir: pathlib.Path | None = None,
    device: str | None = None,
) -> None:
    dev = device or _default_device()
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=160)
    plotted = False
    for model, rdir in results.items():
        curves: list[np.ndarray] = []
        fps_vals: list[float] = []
        for cid in case_ids:
            case_dir = rdir / cid
            frames = load_frames(case_dir)
            if len(frames) < 4:
                continue
            ref_img = _load_start_image(rdir, cid, cases_dir=cases_dir)
            if ref_img is not None:
                all_emb = dino_embed([ref_img, *frames], device=dev)
                ref, emb = all_emb[0], all_emb[1:]
            else:
                emb = dino_embed(frames, device=dev)
                ref = emb[0]
            curves.append(np.asarray(1.0 - emb @ ref, dtype=np.float64))
            fps_vals.append(_case_fps(case_dir))
        if not curves:
            continue
        max_len = max(len(c) for c in curves)
        padded = np.full((len(curves), max_len), np.nan, dtype=np.float64)
        for idx_c, c in enumerate(curves):
            padded[idx_c, : len(c)] = c
        mean_curve = np.nanmean(padded, axis=0)
        fps = float(np.median(fps_vals)) if fps_vals else 8.0
        t = (np.arange(max_len) + 1) / fps
        ax.plot(t, mean_curve, label=model, lw=1.6)
        plotted = True
    ax.set_xlabel("seconds")
    ax.set_ylabel("feature drift vs conditioning frame\n(1 - DINOv2 cosine)")
    ax.set_title("Long-horizon drift under one continuous full turn")
    ax.grid(alpha=0.3)
    if plotted:
        ax.legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def leaderboard_bars(metrics: dict[str, dict], out: pathlib.Path) -> None:
    models = [m for m in metrics if m != "oracle_sim"]
    if not models:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=160)
    width = 0.8 / max(len(models), 1)
    x = np.arange(len(GROUP_SPEC))
    for i, m in enumerate(models):
        m_scores = metrics[m] or {}
        vals = []
        for key, label in GROUP_SPEC:
            v = m_scores.get(key, m_scores.get(label))
            vals.append(float(v) if isinstance(v, (int, float)) and not np.isnan(v) else np.nan)
        ax.bar(x + i * width, vals, width, label=m)
    ax.set_xticks(x + width * (len(models) - 1) / 2)
    ax.set_xticklabels([label for _, label in GROUP_SPEC])
    ax.set_ylim(0, 110)
    ax.set_ylabel("group score (oracle = 100)")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    results = {}
    for name in sys.argv[1:]:
        p = pathlib.Path(name)
        if p.exists():
            results[p.name] = p
    cases_dir = ROOT / "cases"
    manifest = json.loads((cases_dir / "cases.json").read_text(encoding="utf-8"))
    drift_cases = [c["case_id"] for c in manifest["cases"] if c["protocol"] == "loop360"]
    FIG.mkdir(parents=True, exist_ok=True)

    for proto in ("loop360", "occlusion", "null_action"):
        cid = next((c["case_id"] for c in manifest["cases"] if c["protocol"] == proto), None)
        if cid:
            contact_sheet(results, cid, FIG / f"sheet_{proto}.png")

    scores_by_model: dict[str, dict] = {}
    for model, rdir in results.items():
        mp = rdir / "metrics.json"
        if mp.exists():
            raw = json.loads(mp.read_text(encoding="utf-8"))
            scores_by_model[model] = group_scores(_extract(raw))
    if scores_by_model:
        leaderboard_bars(scores_by_model, FIG / "leaderboard.png")

    if results and os.environ.get("WORLDLOOP_NO_GPU") != "1":
        drift_curve(results, drift_cases, FIG / "drift.png", cases_dir=cases_dir)

    print(f"figures -> {FIG}")


if __name__ == "__main__":
    main()
