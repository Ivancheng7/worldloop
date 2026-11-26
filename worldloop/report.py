"""Build a Markdown leaderboard from one or more ``metrics.json`` files.

Group scores use *absolute* scales (documented below) rather than ratios against the
oracle, because a perfect renderer makes some ratios meaningless (oracle LPIPS is
0.006, so a perfectly reasonable model with LPIPS 0.2 would score 3/100).  The
oracle is still reported as the calibration row and as the sanity check that every
metric can reach its ceiling.
"""

from __future__ import annotations

import json
import pathlib
from typing import Iterable

import numpy as np

# metric -> (label, group, better) ; better in {+1, -1, 0}
METRICS = {
    "flow_cos": ("Flow dir. cos", "control", +1),
    "flow_mag_ratio": ("Flow mag. ratio", "control", 0),
    "axis_purity": ("Axis purity", "control", +1),
    "switch_latency_frames": ("Switch latency (f)", "control", -1),
    "revisit_dino": ("Revisit DINO", "memory", +1),
    "revisit_dino_floor": ("Revisit DINO floor", "memory", -1),
    "revisit_dino_norm": ("Revisit DINO (floor-rel.)", "memory", +1),
    "revisit_ssim": ("Revisit SSIM", "memory", +1),
    "revisit_ssim_floor": ("Revisit SSIM floor", "memory", -1),
    "revisit_ssim_norm": ("Revisit SSIM (floor-rel.)", "memory", +1),
    "revisit_lpips": ("Revisit LPIPS", "memory", 0),
    "permanence_dino": ("Permanence DINO", "memory", +1),
    "permanence_dino_floor": ("Permanence floor", "memory", -1),
    "permanence_dino_norm": ("Permanence (floor-rel.)", "memory", +1),
    "drift_slope_per_s": ("Drift / s", "stability", -1),
    "null_motion_px": ("Null motion px", "stability", -1),
    "cf_action_sep": ("CF action sep", "causality", +1),
    "cf_null_drift": ("CF null drift", "causality", -1),
    "asr_mean": ("ASR (legacy)", "causality", 0),
}

GROUPS = ["control", "memory", "stability", "causality"]

# a drift of 0.05 feature-distance per second means "total failure by ~20 s"
DRIFT_FAIL = 0.05
# a switch latency beyond 8 frames (1 s at 8 fps) is a failure
LATENCY_FAIL = 8.0
# motion under a null action beyond 8 px/frame is a failure
NULL_MOTION_FAIL = 8.0
# A commanded 2 s turn separates the analytic renderer from the no-op branch by
# 0.17-0.36 DINO distance; 0.25 is "the command moved the view as much as the
# geometry says it should".
CF_SEP_REF = 0.25
# Half-credit drift for the no-op branch: 0.25 DINO distance is roughly a third of the
# 0.79 measured between two *different* panoramas.  Kept soft (1/(1+x)) so the axis
# still ranks models between "perfectly still" and "regenerates the world".
CF_NULL_DRIFT_HALF = 0.25
# Legacy ASR saturation, retained only to display the old number.
ASR_HALF = 3.0


def _fmt(v: float) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "-"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def _clip01(x: float | None) -> float | None:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    return float(np.clip(x, 0.0, 1.0))


def _extract(metrics: dict) -> dict:
    row: dict[str, float | None] = {}
    proto = metrics.get("aggregate", {}).get("per_protocol", {})
    control = [proto.get(k, {}) for k in ("loop360", "look_away", "occlusion", "drift")]
    mem = [proto.get(k, {}) for k in ("occlusion", "look_away", "loop360")]
    switch_srcs = control + [proto.get("counterfactual", {})]

    def mean(key, srcs):
        vals = [s[key] for s in srcs if isinstance(s.get(key), (int, float)) and not np.isnan(s[key])]
        return float(np.mean(vals)) if vals else None

    row["flow_cos"] = mean("flow_cos", control)
    row["flow_mag_ratio"] = mean("flow_mag_ratio", control)
    row["axis_purity"] = mean("axis_purity", control)
    row["switch_latency_frames"] = mean("switch_latency_frames", switch_srcs)
    row["revisit_dino"] = mean("revisit_dino", control)
    row["revisit_dino_floor"] = mean("revisit_dino_floor", control)
    row["revisit_dino_norm"] = mean("revisit_dino_norm", control)
    row["revisit_ssim"] = mean("revisit_ssim", control)
    row["revisit_ssim_floor"] = mean("revisit_ssim_floor", control)
    row["revisit_ssim_norm"] = mean("revisit_ssim_norm", control)
    row["revisit_lpips"] = mean("revisit_lpips", control)
    row["revisit_dino_est_pose"] = mean("revisit_dino_est_pose", control)
    row["revisit_ssim_est_pose"] = mean("revisit_ssim_est_pose", control)
    row["permanence_dino"] = mean("permanence_dino", mem)
    row["permanence_dino_floor"] = mean("permanence_dino_floor", mem)
    row["permanence_dino_norm"] = mean("permanence_dino_norm", mem)
    row["drift_slope_per_s"] = mean("drift_slope_per_s", control)
    row["null_motion_px"] = mean("null_motion_px", [proto.get("null_action", {})])
    cf = metrics.get("aggregate", {}).get("counterfactual", {})
    cf_sum = cf.get("_summary", {})
    row["cf_action_sep"] = mean("cf_action_sep", [cf_sum])
    row["cf_null_drift"] = mean("cf_null_drift", [cf_sum])
    row["asr_mean"] = mean("asr_mean", [cf_sum])
    cost = metrics.get("aggregate", {}).get("cost", {})
    row["latency_s_mean"] = mean("latency_s_mean", [cost])
    row["realtime_factor"] = mean("realtime_factor", [cost])
    return row


def group_scores(row: dict) -> dict[str, float | None]:
    """Absolute 0-100 scores. Each component is defined so that 100 = oracle-like."""
    parts: dict[str, list[float]] = {g: [] for g in GROUPS}

    def add(group: str, value: float | None) -> None:
        if value is not None and not np.isnan(value):
            parts[group].append(float(np.clip(value, 0.0, 1.0)) * 100.0)

    add("control", _clip01(row.get("flow_cos")))
    add("control", _clip01(row.get("axis_purity")))
    lat = row.get("switch_latency_frames")
    if lat is not None and not np.isnan(lat):
        add("control", max(0.0, 1.0 - lat / LATENCY_FAIL))

    # Memory components are similarity scores, so they are scored *relative to the
    # negative-pair floor* measured with the same instrument on pairs that are definitely
    # not revisits (see protocol.py).  Without that, a model whose imagery is smooth scores
    # high for being smooth: Yume read 0.437 revisit SSIM indoors against a 0.497 floor,
    # i.e. below "a completely different direction", and its out-of-distribution SSIM of
    # 0.622 was tracking a 0.636 floor rather than remembering more.  Raw values stay in
    # the raw table for audit; the score uses the floor-relative ones when present.
    for raw_key, norm_key in (
        ("revisit_dino", "revisit_dino_norm"),
        ("revisit_ssim", "revisit_ssim_norm"),
        ("permanence_dino", "permanence_dino_norm"),
    ):
        val = row.get(norm_key)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            val = row.get(raw_key)  # legacy metrics.json without floors
        add("memory", _clip01(val))

    null = row.get("null_motion_px")
    if null is not None and not np.isnan(null):
        add("stability", max(0.0, 1.0 - null / NULL_MOTION_FAIL))
    drift = row.get("drift_slope_per_s")
    if drift is not None and not np.isnan(drift):
        add("stability", max(0.0, 1.0 - abs(drift) / DRIFT_FAIL))

    # Causality is an AND of two independent failures, not an average:
    #   sensitivity - the action branches must separate from the no-op branch
    #   specificity - the no-op branch must stay where it started
    # A frozen frame has perfect specificity and no sensitivity; a drifting model gets
    # separation only because it drifts everywhere, and its specificity collapses.  The
    # product is 0 if either half fails, which is the honest reading of the protocol.
    sep = row.get("cf_action_sep")
    drift = row.get("cf_null_drift")
    if sep is not None and drift is not None and not np.isnan(sep) and not np.isnan(drift):
        sens = _clip01(float(sep) / CF_SEP_REF)
        spec = 1.0 / (1.0 + max(0.0, float(drift)) / CF_NULL_DRIFT_HALF)
        add("causality", float(sens * spec))
    else:
        # results predating the counterfactual rewrite only carry the legacy ASR
        asr = row.get("asr_mean")
        if asr is not None and not np.isnan(asr):
            add("causality", max(0.0, float(asr)) / (max(0.0, float(asr)) + ASR_HALF))

    # a group with no measurable component scores zero: e.g. a model that never moves
    # cannot demonstrate control at all, and silently dropping the group would flatter it.
    # Causality is different: if the counterfactual protocol was not run at all, the axis
    # is *not evaluated* (None), not zero.
    out = {g: (float(np.mean(v)) if v else 0.0) for g, v in parts.items()}
    if not parts["causality"]:
        out["causality"] = None
    control, memory = out.get("control"), out.get("memory")
    out["world_score"] = (
        float(np.sqrt(max(control, 0.0) * max(memory, 0.0))) if control is not None and memory is not None else None
    )
    return out


def coverage(metrics: dict) -> tuple[int, dict[str, int]]:
    """Cases actually scored, and how many per protocol.

    Read from ``per_case`` rather than from the aggregate so a model that silently
    dropped a case (a crashed run, an empty ``frames/`` directory) cannot claim the
    coverage it was asked for.
    """
    per_case = metrics.get("per_case") or []
    if isinstance(per_case, dict):
        per_case = list(per_case.values())
    counts: dict[str, int] = {}
    for row in per_case:
        proto = row.get("protocol") if isinstance(row, dict) else None
        if proto:
            counts[proto] = counts.get(proto, 0) + 1
    return len(per_case), dict(sorted(counts.items()))


def build_report(results: Iterable[str], out_path: str, title: str = "WorldLoop leaderboard") -> str:
    rows = {}
    covs = {}
    for r in results:
        p = pathlib.Path(r) / "metrics.json"
        if not p.exists():
            continue
        raw = json.loads(p.read_text(encoding="utf-8"))
        name = pathlib.Path(r).name
        rows[name] = _extract(raw)
        covs[name] = coverage(raw)

    lines = [f"# {title}", ""]
    if not rows:
        lines.append("_no results yet_")
        pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(out_path).write_text("\n".join(lines), encoding="utf-8")
        return out_path

    lines += ["## Coverage", ""]
    lines.append("| model | cases | per protocol |")
    lines.append("|---|---|---|")
    for name, (n, counts) in covs.items():
        detail = ", ".join(f"{k} {v}" for k, v in counts.items()) or "-"
        lines.append(f"| `{name}` | {n} | {detail} |")

    header = ["model"] + [f"{METRICS[k][0]}" for k in METRICS]
    lines += ["", "## Raw metrics", ""]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for name, row in rows.items():
        cells = [f"`{name}`"] + [_fmt(row.get(k)) for k in METRICS]
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## Group scores (0-100, higher is better)", ""]
    g_header = ["model", "Control", "Memory", "Stability", "Causality", "**World Score**"]
    lines.append("| " + " | ".join(g_header) + " |")
    lines.append("|" + "|".join(["---"] * len(g_header)) + "|")
    for name, row in rows.items():
        s = group_scores(row)
        cells = [f"`{name}`", _fmt(s["control"]), _fmt(s["memory"]), _fmt(s["stability"]),
                 _fmt(s["causality"]), _fmt(s["world_score"])]
        lines.append("| " + " | ".join(cells) + " |")

    lines += ["", "## Cost (same GPU unless noted)", ""]
    lines.append("| model | s/chunk | realtime factor |")
    lines.append("|---|---|---|")
    for name, row in rows.items():
        lines.append(f"| `{name}` | {_fmt(row.get('latency_s_mean'))} | {_fmt(row.get('realtime_factor'))} |")

    # per-protocol raw numbers, the part a reviewer will actually re-derive
    lines += ["", "## Per-protocol detail", ""]
    lines.append(
        "| model | protocol | flow_cos | flow_mag | int_yaw_deg | revisit_ssim | revisit_ssim_norm | "
        "revisit_dino | revisit_dino_norm | permanence | permanence_norm | drift/s | null px |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in results:
        p = pathlib.Path(r) / "metrics.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text(encoding="utf-8"))
        name = pathlib.Path(r).name
        for proto, vals in sorted(m.get("aggregate", {}).get("per_protocol", {}).items()):
            lines.append(
                f"| `{name}` | {proto} | {_fmt(vals.get('flow_cos'))} | {_fmt(vals.get('flow_mag_ratio'))} | "
                f"{_fmt(vals.get('integrated_yaw_deg'))} | {_fmt(vals.get('revisit_ssim'))} | "
                f"{_fmt(vals.get('revisit_ssim_norm'))} | "
                f"{_fmt(vals.get('revisit_dino'))} | {_fmt(vals.get('revisit_dino_norm'))} | "
                f"{_fmt(vals.get('permanence_dino'))} | {_fmt(vals.get('permanence_dino_norm'))} | "
                f"{_fmt(vals.get('drift_slope_per_s'))} | "
                f"{_fmt(vals.get('null_motion_px'))} |"
            )

    # counterfactual groups, listed per scene so the separation/drift pair is auditable
    lines += ["", "## Counterfactual detail (per group)", ""]
    has_cf = False
    for r in results:
        p = pathlib.Path(r) / "metrics.json"
        if not p.exists():
            continue
        m = json.loads(p.read_text(encoding="utf-8"))
        cf = m.get("aggregate", {}).get("counterfactual", {})
        groups = {k: v for k, v in cf.items() if k != "_summary" and isinstance(v, dict)}
        if groups and not has_cf:
            lines.append("| model | group | action sep | null drift | return gap | legacy ASR |")
            lines.append("|---|---|---|---|---|---|")
            has_cf = True
        for g, vals in sorted(groups.items()):
            asrs = [pp.get("asr_capped", 0.0) for pp in (vals.get("pairs") or {}).values()]
            lines.append(
                f"| `{pathlib.Path(r).name}` | {g} | {_fmt(vals.get('cf_action_sep'))} | "
                f"{_fmt(vals.get('cf_null_drift'))} | {_fmt(vals.get('cf_return_gap'))} | "
                f"{_fmt(float(np.mean(asrs)) if asrs else None)} |"
            )
    if not has_cf:
        lines.append("_no counterfactual runs_")

    lines += [
        "",
        "_Component definitions (all mapped to 0-100, 100 = a perfect analytic renderer):_",
        "_Control_ = mean(flow_cos, axis_purity, max(0, 1 − switch_latency/8)). ",
        "_Memory_ = mean(revisit_dino_norm, revisit_ssim_norm, permanence_dino_norm), each "
        "a similarity scored *relative to a negative-pair floor*: the same DINO/SSIM "
        "instrument applied to pairs that are definitely not revisits (the reference against "
        "frames with the commanded camera ≥120° away; for permanence, probe A's patch against "
        "probe B's direction). Similarity without that subtraction measures image smoothness "
        "as much as memory — Yume's revisit DINO was 0.352 indoors against its own 0.350 "
        "floor, i.e. no more like the start view than like a random other direction, and its "
        "out-of-distribution SSIM of 0.622 was tracking a 0.572 floor. Raw values are kept in "
        "the per-protocol table for audit. ",
        "_Stability_ = mean(max(0, 1 − null_motion/8), max(0, 1 − |drift/s|/0.05)). ",
        "_Causality_ = clip01(mean CF action sep / 0.25) × 1/(1 + mean CF null drift / 0.25), "
        "both means taken over counterfactual groups: the action branches must separate from "
        "the no-op branch *and* the no-op branch must hold still. "
        "`ASR (legacy)` = post/pre with a ×100 cap; it is kept for audit only and is no "
        "longer scored, because `pre` is legitimately 0 for a deterministic model, which "
        "saturates the ratio for every model that renders a stable prefix. "
        "`-` means the counterfactual protocol was not run for that model. ",
        "_**World Score = sqrt(Control × Memory)** — a world model must both move when told "
        "and keep the world it moved through. This is deliberately a geometric mean: doing "
        "nothing scores Control 0, and regenerating a new world every chunk scores Memory ~0. ",
        "_The raw metrics above are the primary record; group scores exist to expose the "
        "trade-off (a static model wins Memory and loses Control)._",
    ]
    pathlib.Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_path
