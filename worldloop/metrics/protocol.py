"""Protocol scoring: turn a rollout into numbers.

All comparisons are *intra-rollout* (the model against itself) or against the
analytic ground-truth flow of the commanded camera motion.  No reference video is
needed, which is what lets the same protocol run on any model, on any hardware,
on any panorama.
"""

from __future__ import annotations

import functools
import math
from pathlib import Path

import cv2
import numpy as np

from .. import actions as A
from ..pano import PanoCamera, load_pano, project_dir, render, rotation_flow_forward
from .features import cosine, dino_embed, lpips_dist, raft_flow, ssim, patch


@functools.lru_cache(maxsize=8)
def _cached_pano(pano_path: str) -> np.ndarray:
    return load_pano(pano_path)


def _flow_step_angles(pred: np.ndarray, fx: float, fy: float) -> tuple[float, float]:
    """Estimate (delta_yaw, delta_pitch) in radians from a dense forward flow field.

    Evaluating exact pinhole ray angles ``arctan(x_ndc) - arctan(x_ndc + u / fx)`` over
    the central image region avoids both the wide-angle ``1 + x_ndc^2`` inflation (~1.54x
    at 102 deg HFOV) and border disocclusion artefacts, while matching ``(-dx/fx, dy/fy)``
    for small or spatially uniform flows.
    """
    h, w = pred.shape[:2]
    y0, y1 = h // 4, max(h // 4 + 1, (3 * h) // 4)
    x0, x1 = w // 4, max(w // 4 + 1, (3 * w) // 4)
    gx = (np.arange(x0, x1, dtype=np.float64) + 0.5 - w / 2.0) / fx
    gy = (h / 2.0 - (np.arange(y0, y1, dtype=np.float64) + 0.5)) / fy
    u = pred[y0:y1, x0:x1, 0].astype(np.float64)
    v = pred[y0:y1, x0:x1, 1].astype(np.float64)
    dyaw = float(np.nanmedian(np.arctan(gx)[None, :] - np.arctan(gx[None, :] + u / fx)))
    dpitch = float(np.nanmedian(np.arctan(gy)[:, None] - np.arctan(gy[:, None] - v / fy)))
    return dyaw, dpitch


def load_frames(case_dir: str | Path) -> list[np.ndarray]:
    fdir = Path(case_dir) / "frames"
    files = sorted(fdir.glob("*.jpg")) or sorted(fdir.glob("*.png"))
    imgs = [cv2.imread(str(f)) for f in files]
    return [im for im in imgs if im is not None]


def _cams(case: dict) -> tuple[list[PanoCamera], A.PoseTimeline]:
    tl = A.compile_timeline(A.from_dicts(case["segments"]), case["fps"], case["start_yaw"], case["start_pitch"])
    return [
        PanoCamera(yaw=y, pitch=p, fov=case["fov"], width=case["width"], height=case["height"])
        for y, p in zip(tl.yaw, tl.pitch)
    ], tl


def _flow_scores(frames: list[np.ndarray], cams: list[PanoCamera], seg_ids: list[int], stride: int = 1, device: str = "cuda"):
    """Per-frame flow agreement with the analytic ground truth.

    ``r = <p, g> / |g|^2`` is a projection coefficient: +1 = exactly the commanded
    motion, 0 = no motion, -1 = the opposite motion.  Unlike an average of per-pixel
    cosines it is not biased by noise (random flow gives r ~ 0, while the mean |cos|
    of random 2-D vectors is ~0.64 — which would make a video generator look like it
    follows instructions).

    Agreement is averaged *within* a segment, so a protocol that turns right and then
    left does not cancel itself out, and the mean over segments is reported.
    """
    rows = []
    fx = fy = (cams[0].height / 2.0) / math.tan(cams[0].fov / 2.0) if cams else 1.0
    for i in range(0, len(frames) - 1, stride):
        gt = rotation_flow_forward(cams[i], cams[i + 1])
        pred = raft_flow(frames[i], frames[i + 1], device=device)
        dx = float(np.mean(pred[..., 0]))
        dy = float(np.mean(pred[..., 1]))
        dyaw, dpitch = _flow_step_angles(pred, fx, fy)
        valid = np.isfinite(gt).all(-1)
        gmag = np.linalg.norm(np.nan_to_num(gt), axis=-1)
        valid &= gmag > 1.5
        r = None
        ratio = None
        if valid.sum() >= 200:
            p = pred[valid]
            g = gt[valid]
            pn = np.linalg.norm(p, axis=-1)
            gn = np.linalg.norm(g, axis=-1)
            # magnitude-weighted mean cosine: gain-invariant (a model moving at half the
            # commanded speed still gets 1.0 if the direction is right), sign-aware
            # (backwards -> -1) and unbiased under noise (random flow -> 0)
            den = float(np.sum(pn * gn))
            if den > 1e-6:
                r = float(np.sum(np.sum(p * g, axis=-1)) / den)
            if pn.size:
                ratio = float(np.median(pn / (gn + 1e-9)))
        seg_idx = seg_ids[i + 1] if i + 1 < len(seg_ids) else (seg_ids[i] if i < len(seg_ids) else -1)
        rows.append(
            {
                "frame": i,
                "seg": seg_idx,
                "r": r,
                "ratio": ratio,
                "dx": dx,
                "dy": dy,
                "dyaw": dyaw,
                "dpitch": dpitch,
            }
        )
    return rows


def _segment_bounds(tl: A.PoseTimeline) -> list[tuple[int, int]]:
    bounds = []
    start = 0
    for k in range(1, len(tl.segment_of_frame) + 1):
        if k == len(tl.segment_of_frame) or tl.segment_of_frame[k] != tl.segment_of_frame[start]:
            bounds.append((start, k))
            start = k
    return bounds


def score_case(
    case: dict,
    frames: list[np.ndarray],
    start_image: np.ndarray | None = None,
    device: str = "cuda",
    flow_stride: int = 1,
) -> dict:
    """Score one rollout.

    ``start_image`` is the conditioning frame the model was given.  It is the correct
    reference for every revisit metric: a revisit happens *at the pose of the
    conditioning frame*, so a perfect world model must reproduce that image.
    Comparing against the model's own frame 0 instead would silently shift every
    reference by one frame of camera motion.
    """
    n = len(frames)
    cams, tl = _cams(case)
    n = min(n, len(cams))
    frames = frames[:n]
    cams = cams[:n]

    out: dict = {"case_id": case["case_id"], "protocol": case["protocol"], "n_frames": n}
    if n == 0:
        return out

    # ---- action fidelity -------------------------------------------------
    stride = max(1, flow_stride if n <= 24 else max(flow_stride, 2))
    rows = _flow_scores(frames, cams, tl.segment_of_frame[:n], stride=stride, device=device)
    per_seg: dict[int, list[float]] = {}
    ratios = []
    for row in rows:
        if row["r"] is not None:
            per_seg.setdefault(row["seg"], []).append(row["r"])
        if row["ratio"] is not None:
            ratios.append(row["ratio"])
    if per_seg:
        seg_means = [float(np.mean(v)) for v in per_seg.values()]
        # a segment that moves the wrong way must not be rewarded by an absolute value
        out["flow_cos"] = float(np.mean([max(0.0, s) for s in seg_means]))
        out["flow_cos_signed"] = float(np.mean(seg_means))
    if ratios:
        out["flow_mag_ratio"] = float(np.median(ratios))

    dx: list[float] = []
    dy: list[float] = []
    if rows:
        dx = [r["dx"] for r in rows]
        dy = [r["dy"] for r in rows]
        dx_np, dy_np = np.array(dx), np.array(dy)
        moving = np.hypot(dx_np, dy_np) > 1.0
        energy = dx_np**2 + dy_np**2 + 1e-9
        # axis purity is only meaningful while the camera is actually moving
        if moving.any():
            out["axis_purity"] = float(np.sum(dx_np[moving] ** 2) / np.sum(energy[moving]))
        out["motion_px_per_frame"] = float(np.mean(np.hypot(dx_np, dy_np)))

        # switch latency: frames until the dominant motion flips sign at a commanded
        # opposite-direction transition between two moving segments.
        # Measured at single-frame resolution regardless of flow_stride.
        segs = A.from_dicts(case["segments"])
        bounds = _segment_bounds(tl)
        flow_cache: dict[int, tuple[float, float]] = {r["frame"]: (r["dx"], r["dy"]) for r in rows}

        def _mean_flow_at(fi: int) -> tuple[float, float]:
            if fi not in flow_cache:
                pred = raft_flow(frames[fi], frames[fi + 1], device=device)
                flow_cache[fi] = (float(np.mean(pred[..., 0])), float(np.mean(pred[..., 1])))
            return flow_cache[fi]

        lat = []
        for k in range(1, len(bounds)):
            bi = bounds[k][0]
            if bi < 2 or bi >= n:
                continue
            s0 = tl.segment_of_frame[bounds[k - 1][0]]
            s1 = tl.segment_of_frame[bi]
            dyaw0, dpitch0 = A.ACTION_RATES[segs[s0].action]
            dyaw1, dpitch1 = A.ACTION_RATES[segs[s1].action]
            if dyaw0 * dyaw1 < 0:
                axis = 0
                # panning right (dyaw > 0) moves image content left (dx < 0)
                exp_prev_sign = -math.copysign(1.0, dyaw0)
                exp_new_sign = -math.copysign(1.0, dyaw1)
            elif dpitch0 * dpitch1 < 0:
                axis = 1
                # pitching up (dpitch > 0) moves image content down (dy > 0)
                exp_prev_sign = math.copysign(1.0, dpitch0)
                exp_new_sign = math.copysign(1.0, dpitch1)
            else:
                continue

            # bi - 2 is frames[bi - 2] -> frames[bi - 1] (last step of segment s0);
            # bi - 1 is frames[bi - 1] -> frames[bi] (1st step of segment s1).
            prev_val = _mean_flow_at(bi - 2)[axis]
            if abs(prev_val) <= 2.0 or np.sign(prev_val) != exp_prev_sign:
                continue
            for offset in range(min(12, n - bi)):
                cur_val = _mean_flow_at(bi - 1 + offset)[axis]
                if abs(cur_val) > 2.0 and np.sign(cur_val) == exp_new_sign:
                    lat.append(offset + 1)
                    break
        if lat:
            out["switch_latency_frames"] = float(np.mean(lat))

    # ---- appearance features --------------------------------------------
    ref_img = start_image if start_image is not None else frames[0]
    emb = dino_embed(frames, device=device)
    ref = dino_embed([ref_img], device=device)[0]

    # revisit index: frame whose camera returns closest to the start pose *after* departing
    start_cam = PanoCamera(yaw=case["start_yaw"], pitch=case["start_pitch"], fov=case["fov"],
                           width=case["width"], height=case["height"])
    yaw_diff = np.array([abs(_wrap(c.yaw - start_cam.yaw)) + abs(c.pitch - start_cam.pitch) for c in cams])
    depart_frames = np.where(yaw_diff >= math.radians(20.0) - 1e-6)[0]
    revisit_idx: int | None = None
    if len(depart_frames) > 0:
        first_depart = int(depart_frames[0])
        if first_depart + 1 < n:
            cand_idx = int(np.argmin(yaw_diff[first_depart + 1 :]) + first_depart + 1)
            if yaw_diff[cand_idx] <= math.radians(15.0) + 1e-6:
                revisit_idx = cand_idx

    if revisit_idx is not None:
        out["revisit_frame"] = revisit_idx
        out["revisit_pose_error_deg"] = float(np.degrees(yaw_diff[revisit_idx]))
        out["revisit_dino"] = cosine(ref, emb[revisit_idx])
        out["revisit_ssim"] = ssim(ref_img, frames[revisit_idx])
        try:
            out["revisit_lpips"] = lpips_dist(ref_img, frames[revisit_idx], device=device)
        except Exception:  # noqa: BLE001 - lpips is optional
            pass

    # ---- trajectory-aware yaw integration & estimated-pose revisit ------
    fx = fy = (float(case["height"]) / 2.0) / math.tan(float(case["fov"]) / 2.0)
    target_idx = revisit_idx if revisit_idx is not None else (n - 1)

    def _integrate_angles(upto_idx: int) -> tuple[float, float]:
        total_steps = upto_idx + 1
        if stride == 1:
            sub = [r for r in rows if r["frame"] < upto_idx]
            dyaw_sum = sum(r["dyaw"] for r in sub)
            dpitch_sum = sum(r["dpitch"] for r in sub)
            if start_image is not None:
                p0 = raft_flow(start_image, frames[0], device=device)
                dy0, dp0 = _flow_step_angles(p0, fx, fy)
            elif rows:
                dy0, dp0 = rows[0]["dyaw"], rows[0]["dpitch"]
            else:
                dy0, dp0 = 0.0, 0.0
            return dy0 + dyaw_sum, dp0 + dpitch_sum
        sub = [r for r in rows if r["frame"] <= upto_idx]
        if not sub:
            if start_image is not None:
                p0 = raft_flow(start_image, frames[0], device=device)
                return _flow_step_angles(p0, fx, fy)
            return 0.0, 0.0
        weights = [min(stride, total_steps - r["frame"]) for r in sub]
        w_sum = sum(weights)
        if w_sum < total_steps:
            weights[-1] += total_steps - w_sum
        dyaw_sum = sum(w * r["dyaw"] for w, r in zip(weights, sub))
        dpitch_sum = sum(w * r["dpitch"] for w, r in zip(weights, sub))
        return float(dyaw_sum), float(dpitch_sum)

    est_dyaw, est_dpitch = _integrate_angles(target_idx)
    out["integrated_yaw_deg"] = float(np.degrees(est_dyaw))
    out["commanded_yaw_deg"] = float(np.degrees(tl.yaw[target_idx] - case["start_yaw"]))

    pano_str = case.get("pano")
    if revisit_idx is not None and revisit_idx > 0 and pano_str:
        pano_path = Path(pano_str)
        if not pano_path.is_file():
            alt = Path(__file__).resolve().parents[2] / pano_str
            if alt.is_file():
                pano_path = alt
        if pano_path.is_file():
            est_yaw = float(case["start_yaw"]) + est_dyaw
            est_pitch = float(
                np.clip(float(case["start_pitch"]) + est_dpitch, -math.radians(85.0), math.radians(85.0))
            )
            est_cam = PanoCamera(
                yaw=est_yaw,
                pitch=est_pitch,
                fov=case["fov"],
                width=case["width"],
                height=case["height"],
            )
            est_ref = render(_cached_pano(str(pano_path.resolve())), est_cam)
            est_emb = dino_embed([est_ref], device=device)[0]
            out["revisit_dino_est_pose"] = cosine(est_emb, emb[revisit_idx])
            out["revisit_ssim_est_pose"] = ssim(est_ref, frames[revisit_idx])

    # ---- multi-loop revisit on long-horizon drift -----------------------
    if case["protocol"] == "drift":
        cum_yaw = np.array([abs(y - case["start_yaw"]) for y in tl.yaw[:n]])
        for loop_idx in (1, 2, 3):
            err = np.abs(cum_yaw - loop_idx * 2.0 * math.pi)
            if len(err) > 0 and float(np.min(err)) <= math.radians(15.0) + 1e-6:
                li = int(np.argmin(err))
                out[f"drift_loop{loop_idx}_dino"] = cosine(ref, emb[li])

    # drift: how fast the model forgets the conditioning frame (dino distance per second)
    if n >= 4:
        dist = np.maximum(0.0, 1.0 - emb @ ref)
        t = np.arange(n, dtype=np.float64) / case["fps"]
        slope = float(np.polyfit(t, dist, 1)[0])
        out["drift_slope_per_s"] = slope
        out["drift_final"] = float(dist[-1])

    # reconstruction: the very first generated frame should still look like the input
    out["recon_dino"] = cosine(ref, emb[0])
    out["recon_ssim"] = ssim(ref_img, frames[0])

    # ---- probe permanence (look_away / occlusion / loop360) -------------
    probes = case.get("probes") or []
    if probes and revisit_idx is not None:
        sims = []
        for pr in probes:
            ref_patch = patch(ref_img, pr["ref_px"], pr["ref_py"], pr["size"])
            px, py, vis = project_dir(cams[revisit_idx], pr["lon"], pr["lat"])
            if ref_patch is None or not vis:
                continue
            cur = patch(frames[revisit_idx], px, py, pr["size"])
            if cur is None:
                continue
            pe = dino_embed([ref_patch, cur], device=device)
            sims.append(cosine(pe[0], pe[1]))
        if sims:
            out["permanence_dino"] = float(np.mean(sims))
            out["permanence_min"] = float(np.min(sims))

    # ---- null action -----------------------------------------------------
    if case["protocol"] == "null_action":
        if dx:
            out["null_motion_px"] = float(np.mean(np.hypot(np.array(dx), np.array(dy))))
        out["null_dino_drift"] = float(np.mean(np.maximum(0.0, 1.0 - emb @ ref))) if n > 0 else 0.0

    return out


def _wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


def score_paired_divergence(results_root: str | Path, cases: dict[str, dict], device: str = "cuda") -> dict:
    """Cheap causal probe: same start, one branch turns, the other does nothing.

    The counterfactual protocol is the clean version of this, but it costs three extra
    rollouts per scene.  When those are not available we can still ask whether the
    action had *any* effect: compare a turning rollout (`loop360`/`occlusion`) with the
    static one (`null_action`) from the same panorama at t = 2 s, normalised by their
    distance at t = 0.

        divergence = D(emb_turn(2s), emb_static(2s)) / D(emb_turn(0), emb_static(0))

    It is not ASR (there is no shared no-op prefix), but a model that ignores actions
    scores ~1 and a model that acts scores > 1.
    """
    root = Path(results_root)
    by_pano: dict[str, list[str]] = {}
    for cid, c in cases.items():
        if (root / cid / "frames").is_dir():
            by_pano.setdefault((c.get("tags") or {}).get("pano_slug", ""), []).append(cid)

    out: dict = {}
    for slug, cids in by_pano.items():
        actions = [c for c in cids if cases[c]["protocol"] in ("loop360", "occlusion", "look_away")]
        statics = [c for c in cids if cases[c]["protocol"] == "null_action"]
        if not actions or not statics:
            continue
        a, s = actions[0], statics[0]
        fa, fs = load_frames(root / a), load_frames(root / s)
        if len(fa) < 17 or len(fs) < 17:
            continue
        ea = dino_embed(fa[:17:2], device=device)  # ~2 s at 8 fps
        es = dino_embed(fs[:17:2], device=device)
        d0 = max(0.0, 1.0 - float(ea[0] @ es[0]))
        dt = max(0.0, 1.0 - float(np.mean(np.sum(ea[1:] * es[1:], axis=-1))))
        out[slug] = {"turn_case": a, "static_case": s, "d0": d0, "d_2s": dt,
                     "divergence_ratio": dt / max(d0, 1e-3)}
    if out:
        vals = [v["d_2s"] for k, v in out.items() if k != "_summary"]
        out["_summary"] = {"n_scenes": len(out), "d_2s_mean": float(np.mean(vals))}
    return out


def aggregate_metrics(case_scores: list[dict]) -> dict:
    """Group per-case scores into protocol-level numbers."""
    by_proto: dict[str, list[dict]] = {}
    for s in case_scores:
        by_proto.setdefault(s["protocol"], []).append(s)

    agg: dict = {"per_protocol": {}}
    for proto, rows in by_proto.items():
        keys = set().union(*[set(r) for r in rows]) - {"case_id", "protocol"}
        entry = {}
        for k in keys:
            vals = [r[k] for r in rows if isinstance(r.get(k), (int, float)) and not (isinstance(r[k], float) and np.isnan(r[k]))]
            if not vals:
                continue
            entry[k] = float(np.mean(vals))
            entry[f"{k}__std"] = float(np.std(vals))
        entry["n_cases"] = len(rows)
        agg["per_protocol"][proto] = entry
    return agg


def _pre_cutoff_frames(case: dict, prefix_seconds: float) -> int:
    fps = float(case["fps"])
    segs = case.get("segments") or []
    pre_frames = 0
    for s in segs:
        tag = s.get("tag") if isinstance(s, dict) else getattr(s, "tag", None)
        sec = float(s["seconds"] if isinstance(s, dict) else s.seconds)
        if tag == "pre":
            pre_frames += max(1, int(round(sec * fps)))
        else:
            break
    if pre_frames > 0:
        return pre_frames
    return max(0, int(round(prefix_seconds * fps)))


def score_counterfactual(
    results_root: str | Path,
    cases: dict[str, dict],
    device: str = "cuda",
    prefix_seconds: float = 2.0,
) -> dict:
    """Action Sufficiency Ratio on paired rollouts.

    All rollouts in a group start from the same frame and share a ``"pre"``-tagged
    (or ``prefix_seconds``) no-op prefix, then diverge:

        ASR(A, B) = mean feature distance after divergence / mean before divergence

    A model that ignores actions gets ASR ~= 0 (both branches look the same).  A model
    that is pure noise gets ASR ~ 1, because the pre-divergence distance is already
    large.  The ratio, not the raw divergence, is the reported number.
    """
    root = Path(results_root)
    groups: dict[str, dict[str, Path]] = {}
    for cid, c in cases.items():
        g = (c.get("tags") or {}).get("group")
        if not g:
            continue
        d = root / cid
        if (d / "frames").is_dir():
            groups.setdefault(g, {})[cid] = d

    out: dict = {}
    for g, members in groups.items():
        if len(members) < 2:
            continue
        embs = {}
        for cid, d in members.items():
            frames = load_frames(d)
            if not frames:
                continue
            embs[cid] = dino_embed(frames, device=device)
        if len(embs) < 2:
            continue
        n = min(len(e) for e in embs.values())
        cut = min(n, min(_pre_cutoff_frames(cases[cid], prefix_seconds) for cid in embs))

        def pair_stats(a: str, b: str) -> dict:
            ea, eb = embs[a][:n], embs[b][:n]
            dist = np.maximum(0.0, 1.0 - np.sum(ea * eb, axis=-1))
            pre = float(np.mean(dist[:cut])) if cut > 0 else 0.0
            post = float(np.mean(dist[cut:])) if n > cut else 0.0
            # a fully deterministic model can have pre == 0; clamp instead of blowing up
            asr = post / max(pre, 1e-3)
            return {"pre": pre, "post": post, "asr": asr, "asr_capped": float(min(asr, 100.0))}

        ids = sorted(embs)
        stats = {}
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                stats[f"{ids[i]}__vs__{ids[j]}"] = pair_stats(ids[i], ids[j])
        out[g] = {"members": ids, "pairs": stats}

    if out:
        asrs = [p["asr_capped"] for g in out.values() for p in g["pairs"].values()]
        out["_summary"] = {"n_groups": len(out), "asr_mean": float(np.mean(asrs)), "asr_min": float(np.min(asrs))}
    return out
