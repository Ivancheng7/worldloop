"""Command line interface: generate cases, run a model, score, report."""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import cv2
import numpy as np

from . import cases as C
from .metrics.protocol import aggregate_metrics, load_frames, score_case, score_counterfactual, score_paired_divergence
from .runner import run_case


def cmd_gen_cases(args) -> int:
    pano_meta = json.loads(pathlib.Path(args.panos).read_text(encoding="utf-8"))
    panos = pano_meta["panoramas"]
    pano_slugs = getattr(args, "pano_slugs", None)
    if pano_slugs:
        wanted = [s.strip() for s in pano_slugs.split(",") if s.strip()]
        by_slug = {(p.get("slug") or pathlib.Path(p["path"]).stem): p for p in panos}
        missing = [s for s in wanted if s not in by_slug]
        if missing:
            raise SystemExit(f"unknown panorama slug(s): {missing}; available: {sorted(by_slug)}")
        panos = [by_slug[s] for s in wanted]
    else:
        panos = panos[: args.n_panos]
    manifest = C.build_case_set(
        panos,
        args.out,
        protocols=args.protocols,
        counterfactual_scenes=args.cf_scenes,
        drift_scenes=getattr(args, "drift_scenes", 1),
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    print(f"[gen-cases] {manifest['n_cases']} cases -> {args.out}")
    return 0


def _build_model(name: str, case: dict, args):
    if name == "oracle":
        from .adapters.oracle import OracleModel

        return OracleModel(case["pano"], fov_deg=np.degrees(case["fov"]), width=case["width"],
                           height=case["height"], fps=case["fps"])
    if name == "yume":
        from .adapters.yume import YumeModel

        return YumeModel(weights=args.weights, steps=args.steps, python=args.model_python,
                         resolution=args.resolution)
    if name == "static":
        from .adapters.static import StaticModel

        return StaticModel(fps=case["fps"])
    if name == "wan":
        from .adapters.wan_t2v import WanT2VBaseline

        # weights=None -> the adapter's local default (models/Wan2.1-T2V-1.3B-Diffusers)
        return WanT2VBaseline(model_id=args.weights, steps=args.steps)
    if name in ("matrix_game_3_5", "matrix_game"):
        from .adapters.matrix_game_space import MatrixGameSpace

        return MatrixGameSpace(steps=args.steps)
    raise SystemExit(f"unknown model {name!r}")


def cmd_run(args) -> int:
    manifest = json.loads(pathlib.Path(args.cases).read_text(encoding="utf-8"))
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    case_list = manifest["cases"]
    if args.panos:
        wanted = {p.strip() for p in args.panos.split(",") if p.strip()}
        case_list = [c for c in case_list if (c.get("tags") or {}).get("pano_slug") in wanted]
    if args.cases_filter:
        case_list = [c for c in case_list if args.cases_filter in c["case_id"]]
    if args.max_cases:
        case_list = case_list[: args.max_cases]
    print(f"[run] {len(case_list)} cases")

    cases_dir = pathlib.Path(args.cases_dir) if args.cases_dir else pathlib.Path(args.cases).parent
    model = None
    last_pano = None
    try:
        for case in case_list:
            cdir = out / case["case_id"]
            if (cdir / "meta.json").exists() and not args.force:
                continue
            start = cv2.imread(str(cases_dir / "start" / f"{case['case_id']}.png"))
            if start is None:
                raise FileNotFoundError(f"missing start image for {case['case_id']}")
            if model is None:
                model = _build_model(args.model, case, args)
                last_pano = case.get("pano")
            elif args.model == "oracle":
                if case.get("pano") != last_pano:
                    if hasattr(model, "set_pano"):
                        model.set_pano(case["pano"])
                    else:
                        model = _build_model(args.model, case, args)
                    last_pano = case.get("pano")
            elif args.model == "static":
                model.native_fps = float(case["fps"])
            prompt = args.prompt
            if not prompt:
                slug = (case.get("tags") or {}).get("pano_slug", "indoor scene")
                prompt = f"First-person perspective of a {slug.replace('_', ' ')}."
            meta = run_case(model, case, start, out, prompt=prompt, seed=args.seed)
            print(f"[run] {case['case_id']} {meta['n_frames']}f {meta['wall_seconds']:.1f}s")
    finally:
        if model is not None and hasattr(model, "shutdown"):
            model.shutdown()
    return 0


def cmd_score(args) -> int:
    manifest = json.loads(pathlib.Path(args.cases).read_text(encoding="utf-8"))
    by_id = {c["case_id"]: c for c in manifest["cases"]}
    cases_dir = pathlib.Path(args.cases_dir) if args.cases_dir else pathlib.Path(args.cases).parent
    root = pathlib.Path(args.results)
    scores = []
    for cid, case in by_id.items():
        cdir = root / cid
        if not (cdir / "frames").is_dir():
            continue
        frames = load_frames(cdir)
        if not frames:
            continue
        start_img = cv2.imread(str(cases_dir / "start" / f"{cid}.png"))
        scores.append(score_case(case, frames, start_image=start_img, device=args.device))
        print(f"[score] {cid} ok")

    agg = aggregate_metrics(scores)
    cf = score_counterfactual(root, by_id, device=args.device)
    if cf:
        agg["counterfactual"] = cf
    pd = score_paired_divergence(root, by_id, device=args.device)
    if pd:
        agg["paired_divergence"] = pd

    cost = {}
    for cid in by_id:
        m = root / cid / "meta.json"
        if m.exists():
            cost[cid] = json.loads(m.read_text(encoding="utf-8")).get("cost", {})
    if cost:
        keys = sorted({k for c in cost.values() for k in c} - {"n_steps"})
        if keys:
            agg["cost"] = {
                k: float(np.mean([c[k] for c in cost.values() if c.get(k) is not None]))
                for k in keys
                if any(c.get(k) is not None for c in cost.values())
            }

    out = root / "metrics.json"
    out.write_text(json.dumps({"per_case": scores, "aggregate": agg}, indent=2), encoding="utf-8")
    print(f"[score] -> {out}")
    return 0


def cmd_report(args) -> int:
    from .report import build_report

    path = build_report(args.results, args.out)
    print(f"[report] -> {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser("worldloop")
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gen-cases")
    g.add_argument("--panos", default="data/panos/panos.json")
    g.add_argument("--n-panos", type=int, default=8)
    g.add_argument("--pano-slugs", default=None, help="comma separated panorama slugs to include")
    g.add_argument("--out", default="cases")
    g.add_argument("--protocols", nargs="*", default=["loop360", "occlusion", "null_action"])
    g.add_argument("--cf-scenes", type=int, default=2)
    g.add_argument("--drift-scenes", type=int, default=1)
    g.add_argument("--width", type=int, default=1280)
    g.add_argument("--height", type=int, default=720)
    g.add_argument("--fps", type=float, default=8.0)
    g.set_defaults(func=cmd_gen_cases)

    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--cases", default="cases/cases.json")
    r.add_argument("--cases-dir", default=None)
    r.add_argument("--out", required=True)
    r.add_argument("--cases-filter", default=None)
    r.add_argument("--panos", default=None, help="comma separated pano slugs")
    r.add_argument("--max-cases", type=int, default=None)
    r.add_argument("--weights", default=None)
    r.add_argument("--steps", type=int, default=6)
    r.add_argument("--device", default="cuda")
    r.add_argument("--model-python", default=None, help="interpreter that runs the model subprocess")
    r.add_argument("--resolution", default="704x1280", help="native generation resolution (HxW)")
    r.add_argument("--prompt", default="")
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--force", action="store_true")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("score")
    s.add_argument("--results", required=True)
    s.add_argument("--cases", default="cases/cases.json")
    s.add_argument("--cases-dir", default=None)
    s.add_argument("--device", default="cuda")
    s.set_defaults(func=cmd_score)

    p = sub.add_parser("report")
    p.add_argument("--results", nargs="*", required=True)
    p.add_argument("--out", default="docs/RESULTS.md")
    p.set_defaults(func=cmd_report)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

