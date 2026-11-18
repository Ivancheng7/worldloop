# WorldLoop

**A closed-loop, simulator-grounded benchmark for interactive world models.**

Most world-model evaluations score a *video*: they feed a trajectory, render the
result, and compare it with a reference clip. That answers "is this a good video?",
not "is this a world?". WorldLoop asks the three questions that separate a world
model from a video generator:

1. **Does the world change because of my action?** (counterfactual sensitivity)
2. **Does it stay changed while I am not looking?** (state persistence)
3. **Is it the same world when I come back?** (revisit / loop closure)

It is *closed loop* — every generated frame is fed back as the next condition,
exactly how these models are used — and *simulator-grounded*: cases are built on
photo-real 360° panoramas, so the camera trajectory, the occlusion schedule and the
ground-truth optical flow are analytic, not annotated.

```
panorama (CC0)  ──►  start frame + pose script  ──►  model plays the script
      │                                              (frame → action → frame …)
      └────────────────►  analytic GT flow  ◄──────  scores
```

## Why

The 2026 survey of 130 world-model benchmarks
([world-model-benchmarks.github.io](https://world-model-benchmarks.github.io/))
counts 74 control-oriented benchmarks; only 6 of them close the loop, and none of
them tests returning to a previously visited state. Yet "walk away, come back, and
the room is still the room" is the cheapest behavioural test of a world model
there is — and the one that fails first in practice.

## Protocols

| Protocol | Script | Duration | Question |
|---|---|---|---|
| `loop360` | 3 × `turn_right` 2 s (60°/s) | 6 s | Does the world survive a full turn? |
| `occlusion` | `turn_right` 2 s → `noop` 2 s → `turn_left` 2 s | 6 s | Do objects survive being off-screen for 4 s? |
| `null_action` | `noop` 4 s | 4 s | Does a world at rest stay at rest? |
| `drift` | `turn_right` 18 s | 18 s | How fast does it fall apart? |
| `counterfactual` | `noop` 2 s → left / right / `noop` 2 s → return 2 s | 6 s | Does the branch actually depend on the action? |

Metrics per protocol: analytic-flow agreement (`flow_cos`, `flow_mag_ratio`,
`axis_purity`, `switch_latency_frames`), revisit consistency (`revisit_dino`,
`revisit_ssim`, `revisit_lpips`), probe permanence (`permanence_dino`), stability
and drift (`null_motion_px`, `drift_slope_per_s`), and `ASR` (action sufficiency
ratio) for the counterfactual triplets. Latency, realtime factor and peak VRAM are
measured on the same GPU for every model.

## Quick start

```bash
pip install -e .
python scripts/fetch_panos.py --out data/panos --n 10     # CC0 panoramas
python -m worldloop.cli gen-cases --n-panos 6 --out cases \
    --protocols loop360 occlusion null_action --cf-scenes 2

# calibration: the analytic renderer is the upper bound of the benchmark
python -m worldloop.cli run   --model oracle --cases cases/cases.json --out results/oracle_sim
python -m worldloop.cli score --results results/oracle_sim --cases cases/cases.json

# a real model (--model: oracle | static | yume | wan | matrix_game_3_5)
python -m worldloop.cli run   --model yume --cases cases/cases.json --out results/yume_5b
python -m worldloop.cli score --results results/yume_5b --cases cases/cases.json
python -m worldloop.cli report --results results/oracle_sim results/yume_5b \
    --out docs/RESULTS_leaderboard.md
```

## Adding a model

Any model with a `step(action) -> frames` loop can be evaluated:

```python
from worldloop.adapters.base import ClosedLoopModel

class MyModel(ClosedLoopModel):
    name = "my_model"
    native_fps = 16.0
    min_chunk_seconds = 2.0

    def reset(self, image, prompt="", seed=0): ...
    def step(self, action, seconds): ...   # returns BGR frames at native_fps
```

Chunk size stays inside the adapter, so a 0.5 s-chunk streaming model and a 4 s-chunk
offline model are compared on the same timeline. See `docs/PROTOCOL.md` for the full
spec and `worldloop/adapters/` for reference implementations (`oracle` analytic
renderer, `static` frozen baseline, `yume` keyboard-driven world model,
`matrix_game_space` trajectory-conditioned Space adapter, and `wan_t2v` negative
control).

## Results

See [`docs/RESULTS.md`](docs/RESULTS.md) for the full table and analysis, and
[`docs/PORTFOLIO_zh.md`](docs/PORTFOLIO_zh.md) for a Chinese write-up.

The figure below is the same `loop360` case played by four systems: four frames out
of a 6 s, 360° turn. A world model has to end where it started — the analytic
renderer does (`revisit_ssim = 0.99`), the 5 B interactive model does not, the
text-to-video model leaves the scene entirely, and the frozen baseline never moves.

![oracle loop360](docs/figures/sheet_loop360.png)

Headline findings from the v0.1 run (4 systems, 1× RTX 5060 Ti 16 GB, 15 cases for
the interactive model, 12 for the control):

| model | Control | Memory | Stability | Causality | **World Score** |
|---|---|---|---|---|---|
| `oracle_sim` (analytic) | 99.9 | 99.2 | 94.1 | 97.1 | **99.6** |
| `yume_5b` (interactive WM) | 81.4 | 34.9 | 0.0 | 92.5 | **53.3** |
| `wan2.1_t2v_1.3b` (no actions) | 27.5 | 26.1 | 74.5 | 0.0 | **26.8** |
| `static` (frozen frame) | 5.7 | 99.2 | 99.9 | 0.0 | **23.8** |

- an interactive model follows the *direction* of a command (0.76) at roughly
  two-thirds of the commanded *gain* (0.65–0.73) — the two must be scored separately;
- one 360° turn is enough to lose the world: revisit SSIM 0.46 / DINO 0.33 vs
  0.99 / 1.00 for the analytic renderer, and a 4 s look-away is even worse
  (permanence 0.25);
- told to hold the camera still, the same model still moves the image
  **11.8 px/frame** (oracle: 0.02) — it cannot represent "no change";
- causality is the cleanest separator: counterfactual branches give **ASR 37.1**
  for the interactive model (deterministic, pre-divergence distance 0.002) and
  exactly **0.000** for the text-to-video model, whose branches are byte-identical;
- a frozen frame wins Memory and loses Control, so the geometric-mean World Score
  keeps both degenerate solutions at the bottom.

## Repository layout

```
worldloop/
  pano.py          equirect → perspective rendering + analytic GT flow
  actions.py       chunk-level action language → per-frame pose timeline
  cases.py         protocol definitions and case construction
  runner.py        closed-loop rollout
  metrics/         flow, DINOv2 features, revisit, permanence, counterfactual ASR
  adapters/        oracle, static, yume (keyboard WM), matrix_game_space (pose), wan_t2v
  report.py        leaderboard generation
docs/PROTOCOL.md   the evaluation spec
docs/DATASET.md    case-set card
docs/RESULTS.md    v0.1 analysis (raw table in docs/RESULTS_leaderboard.md)
```

## Limitations

Rotation-only camera motion (no translation ⇒ no physics/contact evaluation), 720p
at 8 fps, one panorama per case, probes are image regions rather than segmented
object instances. Details and rationale in `docs/PROTOCOL.md` §7.

## License

Code: Apache-2.0. Panoramas: CC0 (Poly Haven).
