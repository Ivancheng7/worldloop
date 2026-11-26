# WorldLoop Protocol v0.2

> A closed-loop, simulator-grounded evaluation for **interactive** world models.

## 1. Why another world-model benchmark

Existing suites answer "is the video good?" (VBench, WorldScore) or "does it follow
a given trajectory?" (WorldMark, WBench, WorldRoamBench). Almost none of them asks
the question that separates a *world model* from a *video generator*:

> When I act, does the world change **because of my action**, does it **stay
> changed** while I am not looking, and is it **the same world** when I come back?

The 2026 survey of 130 world-model benchmarks reports that only 6 of 74
control-oriented benchmarks evaluate a closed loop at all, and none evaluates
revisit consistency — the ability to return to a previously seen state. That is
the gap WorldLoop targets.

WorldLoop has three design constraints:

1. **No reference video.** Generative world models are stochastic; pixel-exact GT
   is not attainable and not the point. Every score is either intra-rollout
   (self-consistency) or against *analytic* camera-motion ground truth.
2. **Closed loop.** The model's own frame is fed back as the next condition, which
   is how these models are actually used.
3. **Runs on one 16 GB consumer GPU.** Cases are 4–18 s, 1280×720 at 8 fps.

## 2. How cases are built

A case is a script played on a **photo-real 360° panorama** (Poly Haven, CC0).

For a camera that only *rotates* about the panorama centre, the perspective view is
an exact analytic function of `(yaw, pitch, fov)`:

```
d(lon, lat) = (cos(lat)sin(lon), sin(lat), cos(lat)cos(lon))
u = (lon / 2π + 0.5) W,  v = (0.5 - lat/π) H
```

Two consequences, and they are the reason this benchmark is cheap and exact:

* **Ground-truth optical flow is free.** For any two camera orientations we can
  compute dense optical flow analytically — both backward flow on `cam1`'s grid for
  image warping (`worldloop.pano.rotation_flow`) and forward flow on `cam0`'s grid
  (`worldloop.pano.rotation_flow_forward`), matching RAFT pixel-for-pixel — with no
  network and no reference video.
* **Revisits are exact.** "Return to the start heading" is a pose, so the frame
  produced on return can be compared against the frame the model itself produced
  at that pose earlier — the classic loop-closure test, now with a known answer.

Each case also carries **probes**: three high-gradient, spatially separated image
regions of the first frame, converted to world directions. Probes make object
permanence measurable: when the camera comes back, a probe should project back to
(approximately) where it started and look the same.

**Camera convention.** `fov` is the *vertical* field of view (the ray grid uses
`tan(fov/2)` along y), so the default 70° at 16:9 corresponds to ≈102° horizontal —
a wide-angle look that keeps translation-free panoramas from feeling like a zoom.
The camera basis is right-handed (`r = up × f`), and `tests/test_pano.py` asserts
that panning right moves image content *left*: with the mirrored basis an otherwise
correct model would score `flow_cos ≈ −1`.

## 3. Protocols

| id | action script | duration | what it measures |
|---|---|---|---|
| `loop360` | 3 × `turn_right` 2 s (60°/s) | 6 s | full 360° loop: does the world come back? |
| `occlusion` | `turn_right` 2 s → `noop` 2 s → `turn_left` 2 s | 6 s | 120° look-away, probes off-screen for 4 s |
| `null_action` | `noop` 4 s | 4 s | a world at rest should stay at rest |
| `drift` | `turn_right` 18 s | 18 s | long-horizon error accumulation (3 loops) |
| `counterfactual` | `noop` 2 s → (`turn_left` \| `turn_right` \| `noop`) 2 s → return 2 s | 6 s | does the branch actually depend on the action? |

Every segment is a multiple of **2 s**, the native chunk of the evaluated models, so
no adapter is ever forced to generate a partial chunk. Action primitives are
model-agnostic chunk-level commands at 60°/s yaw and 30°/s pitch; adapters map them
to keyboard tokens, camera poses, or latent actions. (`look_away`, a 4 s variant, is
defined in `worldloop/cases.py` but not part of the active set — `occlusion` subsumes
it.)

## 4. Metrics

**Control** (needs the analytic GT flow of the commanded motion, `rotation_flow_forward`)

| metric | definition | better |
|---|---|---|
| `flow_cos` | segment-averaged, magnitude-weighted cosine between predicted RAFT flow and analytic GT flow over pixels with \|GT\| > 1.5 px (each segment mean clamped at 0) | ↑ |
| `flow_mag_ratio` | median \|pred\| / \|GT\| over pixels with \|GT\| > 1.5 px | → 1 |
| `axis_purity` | horizontal motion energy / total motion energy while moving (\|flow\| > 1 px) | ↑ |
| `switch_latency_frames` | frames after an action flip between moving segments until the dominant horizontal flow flips sign | ↓ |

For each frame pair, directional agreement uses the magnitude-weighted cosine
`∑⟨p, g⟩ / ∑(|p|·|g|)` (gain-invariant, sign-aware, and unbiased under random noise
where unweighted `|cos|` averages ~0.64). Frame scores are averaged *within* each
segment and clamped at `0` (`max(0, seg_mean)`) before averaging across segments, so
opposite-direction motion scores `0` rather than cancelling out across right/left
turns or being rewarded by `|cos|` (the unclamped mean is also recorded as
`flow_cos_signed`).

**Memory / persistence** (no GT needed)

| metric | definition | better |
|---|---|---|
| `revisit_dino` | DINOv2 cosine between the first frame and the revisit frame at the same pose | ↑ |
| `revisit_ssim` / `revisit_lpips` | pixel-level agreement of the two visits | ↑ / ↓ |
| `permanence_dino` | mean DINOv2 cosine of probe patches, reference vs projected-back patch on revisit | ↑ |
| `drift_slope_per_s` | slope of `1 − cos(emb_t, emb_0)` over time | ↓ |

**Stability**

| metric | definition | better |
|---|---|---|
| `null_motion_px` | mean flow magnitude under `null_action` | ↓ (0 ideal) |
| `null_dino_drift` | feature drift under `null_action` | ↓ |

**Causality**

The protocol runs each scene as a triplet that shares a start frame and a 2 s no-op
prefix, then diverges for 2 s (`turn_left` | `turn_right` | `noop`) and returns for
2 s. Two quantities are measured on the DINOv2 embeddings, both on the intervention
window (`16 < t ≤ 32` at 8 fps):

| metric | definition | better |
|---|---|---|
| `cf_action_sep` | mean `1 − cos(emb^action_t, emb^null_t)` over the intervention window | ↑ |
| `cf_null_drift` | mean `1 − cos(emb^null_t, emb^null_0)` over the intervention window | ↓ (0 ideal) |
| `cf_return_gap` | how much of the separation survives the return segment | ↓ (0 ideal) |
| `asr` (legacy) | `mean_t D(t > cut) / max(mean_t D(0 < t < cut), 1e-3)`, capped at 100 | — |

`cf_action_sep` answers *does the branch depend on the action*, and `cf_null_drift`
answers *is the no-op branch actually a control*. The pair is a 2×2: an
action-sensitive, stable model separates and holds still; a frozen frame separates
nothing; a model that regenerates the world every chunk separates only because it
drifts everywhere, which `cf_null_drift` catches.

`ASR` is retained for audit but **is no longer scored**. Its denominator is
legitimately `0` for a deterministic renderer, so the ratio saturates at the `×100`
cap for the oracle itself — on the four-model board, oracle (97) and a Yume-5B run
(95) were indistinguishable on this axis purely as an artifact of the cap. The raw
`pre`/`post` distances are still printed so the old number can be re-derived, and
`cf_return_gap` keeps the "does the branch come back" signal that the old
single-window `post` average mixed together with the intervention.

**Cost** is measured, not inferred: chunks/s, mean and p95 latency per chunk,
realtime factor (generated seconds per wall-clock second) and peak VRAM, all on the
same GPU.

## 5. Scoring

Raw metrics are the primary record. In `worldloop/report.py`, protocol-level means
are aggregated across active protocols (`loop360`, `occlusion`, `drift` for control,
revisit and drift metrics; `occlusion` and `loop360` for `permanence_dino`;
`null_action` for `null_motion_px`; `counterfactual` for `cf_action_sep` and
`cf_null_drift`) and folded into
four group scores on absolute 0–100 scales (each component clipped to `[0, 1]` and
multiplied by `100`, where `100` ≈ what the analytic renderer achieves), plus the
headline **World Score**:

```
Control     = 100 × mean( clip01(flow_cos), clip01(axis_purity), max(0, 1 − switch_latency_frames / 8) )
Memory      = 100 × mean( clip01(revisit_dino), clip01(revisit_ssim), clip01(permanence_dino) )
Stability   = 100 × mean( max(0, 1 − null_motion_px / 8), max(0, 1 − |drift_slope_per_s| / 0.05) )
Causality   = 100 × clip01( sep / 0.25 ) × 1 / (1 + max(0, drift) / 0.25)
              # sep, drift = mean over counterfactual groups of cf_action_sep / cf_null_drift
World Score = sqrt( Control × Memory )               # Causality is None ("-") if counterfactual was not run
```

`Causality` is a product rather than a mean because the two halves fail
independently: a frozen frame has perfect specificity and no sensitivity, and a
drifting generator buys sensitivity with drift, which collapses its specificity. A
product is `0` when either half fails, which is the honest reading of the protocol —
an average would let each failure be half-credited by the other working component.
The `0.25` scale is calibrated against the measured DINOv2 distance between two
*different* panoramas on this dataset (0.79 mean over 15 scene pairs), so a separation
of 0.25 is roughly a third of a scene change; the same number as the drift
half-credit means "the no-op branch has wandered a third of the way to another room".

Within each group, defined components are averaged; if a group other than
`Causality` has no measurable component (for example, a frozen frame produces no
moving segments), it scores `0.0` rather than being omitted. If `counterfactual` was
not run for a model, `Causality` is reported as `-` (not evaluated).

The thresholds encode what counts as failure: a direction switch taking longer than
1 s (`8` frames at 8 fps), feature drift of `0.05/s` (total forgetting within ~20 s),
motion of `8 px/frame` under a null action, and a counterfactual separation or no-op
drift of `0.25` DINOv2 cosine distance (~a third of the 0.79 measured between two
*different* scenes in this dataset).

Group scores are *not* oracle ratios. A ratio against a perfect renderer is
uninformative for pixel metrics — the oracle's LPIPS is 0.006, so a perfectly
reasonable model with LPIPS 0.2 would score 3/100 on that column. The oracle is kept
as the calibration row: it demonstrates that every metric can reach its ceiling.

**Why `World Score = sqrt(Control × Memory)` rather than an arithmetic mean.** The
axes are deliberately in tension, and an arithmetic average of the four groups can be
gamed by degenerate systems: `static` (a frozen frame) scores ~99 on Memory and ~100
on Stability while scoring ~0 on Control, and an action-blind video generator can
still score high on Stability (~75) while failing Control and Memory. Taking the
geometric mean of `Control` and `Memory` requires a world model to *both* move when
commanded and preserve the world it moved through (regenerating a new scene every
chunk collapses Memory toward 0; standing still collapses Control toward 0), while
`Stability` and `Causality` remain separate diagnostic axes alongside it.

## 6. Adapter contract

```python
class ClosedLoopModel:
    name: str
    native_fps: float
    min_chunk_seconds: float

    def reset(self, image, prompt, seed) -> None: ...
    def step(self, action: str, seconds: float) -> list[np.ndarray]: ...
```

The adapter owns its native chunking and is allowed to refuse durations shorter
than `min_chunk_seconds`; the runner resamples returned frames onto the case
timeline. Anything with a `step(action) -> frames` interface can be evaluated:
keyboard-driven game models, camera-pose models, or a text-to-video model used as a
negative control.

## 7. Known limitations (v0.1)

* **Rotation only.** Translation would require depth; the panorama gives an exact
  rotating camera, not a moving one. Physics/contact is therefore out of scope —
  use Physics-IQ / PhyGenBench for that.
* **8 fps, 720p.** Chosen so a 5 B model runs on a 16 GB card; models trained at
  16 fps are subsampled.
* **One scene per case.** No multi-object interaction, no agent embodiment.
* **Probes are image regions, not object instances.** No instance segmentation or
  amodal reasoning.
* **Stochasticity.** Rollouts use a fixed seed, but seeds are model-specific. The
  counterfactual pair shares the seed and the start frame, so the no-op branch's drift
  (`cf_null_drift`) is a direct measure of that model's own noise floor, which is why
  it is scored rather than assumed away.
