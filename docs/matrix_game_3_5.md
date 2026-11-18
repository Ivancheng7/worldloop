# Matrix-Game 3.5 Space (`hugging-apps/matrix-game-3-5`) — probe report

Date: 2026-11-13 (Asia/Shanghai). Everything below is measured, not assumed;
raw records live in `scratch/mg35_val/`.

## TL;DR

* The Space is public and **RUNNING** (`zero-a10g`, 1 replica, sha `6fafe07`), and
  the `/generate` schema matches the documented one.
* **Anonymous calls do not work.** ZeroGPU rejects the request *before* allocating
  a GPU: `The requested GPU duration (278s) is larger than the maximum allowed`.
  Anonymous accounts have a **2 min/day** ZeroGPU budget
  (<https://huggingface.co/docs/hub/en/spaces-zerogpu>), while the smallest
  request this app can emit is 278 s. No rollout can be produced anonymously — not
  via the API and not via the web UI at these settings.
* No `results/matrix_game_3_5/` was produced. Nothing was fabricated into
  `results/`.
* `worldloop/adapters/matrix_game_space.py` was written anyway, and its camera
  geometry was verified offline to be **bit-exact** against `worldloop.pano.render`
  (the renderer that produced the case start frames). The network path is the only
  unverified part.

## 1. Probe log (2 attempts, then stopped)

Setup: Python 3.10 venv `.venv_mg35` (`gradio_client==2.7.2`, numpy 2.2.6,
opencv 5.0.0). No HF token anywhere on the machine (`~/.cache/huggingface/token`
absent, `HF_TOKEN` unset). Input image: `cases/start/abandoned_bakery__loop360.png`
centre-cropped 1280x720 → 1280x704.

### Attempt 1 — the prescribed minimal call

`/generate(image, prompt, camera_file=None, camera_preset="Turn right (in place)",
mode="First-person", refs=[], num_blocks=1, steps=8, cfg_scale=5.0, seed=42,
randomize_seed=False)`

| item | value |
| --- | --- |
| client connect | 4.63 s |
| submit → error | 6.03 s (total wall 10.7 s) |
| queue | **none** — `Status.STARTING`, then the app error; no GPU time, no wait |
| result | failure |

Error text, verbatim:

```
gradio_client.exceptions.AppError: The requested GPU duration (278s) is larger
than the maximum allowedSubscribe to Hugging Face PRO to allow GPU tasks up to
40 min - https://huggingface.co/subscribe/pro?from=ZeroGPU
```

Record: `scratch/mg35_val/probe_summary.json`.

### Attempt 2 — smallest request the Space can emit

`steps=1` (to shrink the dynamic `@spaces.GPU` duration) is rejected by the
Gradio slider before the GPU is even considered:

```
gradio_client.exceptions.AppError: Value 1 is less than minimum value 8.
```

(wall 9.0 s). Record: `scratch/mg35_val/probe2_summary.json`.

### Why 278 s is the floor, and why anonymous can never pay for it

`app.py` (`mg35_app.py:325`, the Space source snapshot already in the repo) computes

```
_duration = int(1.15 * (30 + 45*blocks + 5.8*steps*(1.0 or 2.2)))
```

`num_blocks` ≥ 1 and `steps` ≥ 8 are slider-bounded, so the smallest possible
value is `int(1.15*(30+45+46.4)) = 139`, and the platform doubles it for
`size="xlarge"` (2× quota per the ZeroGPU docs) → **278 s**.
Unauthenticated accounts get 2 min/day, i.e. the request can never fit — not even
as the only request of the day. A 2-block/8-step case requests `int(1.15*(30+90+
102.08)) = 255` → 510 s, which needs a PRO (40 min/day) or Team account.

## 2. What unblocks it

Put a PRO/Team token in `HF_TOKEN` (or pass `hf_token=` to `MatrixGameSpace`); the
adapter picks it up and `gradio_client.Client(..., hf_token=...)` authenticates the
ZeroGPU quota check. Then:

```powershell
$env:HF_TOKEN = "hf_..."
.\.venv_mg35\Scripts\python.exe -m worldloop.cli run --model matrix_game_3_5 `
    --out results/matrix_game_3_5 --cases-filter abandoned_bakery__loop360 --steps 8 --seed 42
.\.venv_mg35\Scripts\python.exe -m worldloop.cli run --model matrix_game_3_5 `
    --out results/matrix_game_3_5 --cases-filter empty_warehouse_01__loop360 --steps 8 --seed 42
```

Scoring still has to run in the venv that carries RAFT/DINO (`.venv`), the
Space-facing run in `.venv_mg35` (it has `gradio_client`).

## 3. Verified coordinate conventions (no GPU needed)

> [!NOTE]
> **Post-probe geometry update:** Sections 3.1–3.4 below preserve the original
> 2026-11-13 coordinate audit (`scratch/mg35_val/coordinate_check.json`), which
> uncovered that the initial `worldloop/pano.py` used a horizontally mirrored camera
> basis (`r = f × up`). Following this audit:
> - `worldloop/pano.py` (`_camera_basis`) was updated to the un-mirrored right-handed
>   basis (`r = up × f`, `u = f × r`, with `fov` also explicitly documented as
>   vertical), and `tests/test_pano.py` now asserts `dx < -10` px on a `+7.5°` right
>   pan. Consequently, `turn_right` in `worldloop` now moves image content **left**
>   (`dx < 0`), matching standard pinhole geometry and Matrix-Game 3.5's
>   `Turn right (in place)` preset (`R_y(+θ)`).
> - `worldloop/adapters/matrix_game_space.py`'s `c2w_matrix` has been updated
>   accordingly so that `R` remains a proper rotation with `det(R) = +1` in `SO(3)`
>   under the un-mirrored basis.

All of the following is reproduced by `scratch/mg35_coord_check.py` on the real
panorama; numbers are in `scratch/mg35_val/coordinate_check.json`.

### 3.1 The c2w convention is right, exactly

`R = [r, -u, f]` (columns right/down/forward) with
`f = (cos p sin y, sin p, cos p cos y)`, `r = cross(f, up)`, `u = cross(r, f)` —
i.e. exactly `worldloop.pano._camera_basis` — plus a pinhole `K` renders the
panorama **identically** to `worldloop.pano.render`:

| pose | mean abs pixel diff (independent renderer vs `pano.render`) |
| --- | --- |
| yaw 0°, pitch 0° | **0.000** |
| yaw 37.5° | **0.000** |
| yaw 200° | **0.000** |
| pitch +15° | **0.000** |

So the pose list the adapter writes into `camera.npz` is the same camera the
benchmark used to render the conditioning frame.

### 3.2 `fov` is the **vertical** fov — the suggested intrinsics formula is 1.78× off

`pano.py`'s docstring says `fov` is horizontal, but its ray grid uses
`tan(fov/2)` vertically and `tan(fov/2)*aspect` horizontally. The intrinsics of
the 1280x720 start frames are therefore

```
fx = fy = (720/2) / tan(fov/2) = 514.13 px,  cx = 640,  cy = 360   (352 after the 704 crop)
```

Using `(1280/2)/tan(fov/2) = 914.01` (the "horizontal" reading) instead renders a
visibly different image: mean abs diff **39.2**, p99 135. The adapter defaults to
the vertical reading (`fov_axis="v"`) and keeps `"h"` only as an experiment knob.

### 3.3 Direction of `turn_right`, measured on pixels

| camera motion | Farneback dx | phase-corr dx | analytic GT mean dx |
| --- | --- | --- | --- |
| yaw **+0.5°** (`turn_right`) | **+5.99** | **+5.33** | **+6.80** |
| yaw **−0.5°** (`turn_left`) | −5.98 | −5.34 | −6.80 |
| pitch +0.5° (`pitch_up`) | dy **+4.88** | dy +4.77 | dy +4.9 |

So in this benchmark `turn_right` makes image content move **right** (the camera
pans toward the image's left). The ground-truth flow the scorer uses has the same
sign.

### 3.4 The Space's own preset is the *opposite* direction — do not compare to it

`app.py`'s `preset_trajectory("Turn right (in place)")` is `R_y(+0.714°)` per pose
over a block (60° / 84 poses), with the documented OpenCV basis (x right, y down).
Rendered with the same panorama + K:

| motion | Farneback dx | phase-corr dx |
| --- | --- | --- |
| preset `Turn right (in place)`, 1 pose (0.714°) | **−28.75** | **−22.77** |
| our `turn_right`, 0.5° | **+5.99** | **+5.33** |

Opposite sign. Reason: `worldloop.pano.render` is **mirrored** relative to a
standard equirect viewer (image right == negative longitude, `r = cross(f, up) =
−x` at yaw 0), and the scorer's analytic flow is generated with that same mirrored
basis. A model driven so that it looks like the preset would score
`flow_cos ≈ −1`. **The adapter must match the GT, not the preset** — that is why
the pose list is built from `pano._camera_basis` rather than from a textbook
"+x is right" basis.

Caveat: 3.4 is computed from `app.py`'s source math, not from an actual model
output (no call succeeded). It establishes the preset's geometry, not the model's
compliance with it.

## 4. Adapter: `worldloop/adapters/matrix_game_space.py`

`MatrixGameSpace(ClosedLoopModel)`, `name="matrix_game_3_5"`,
`native_fps=16.0`, `min_chunk_seconds=5.0`, registered in `worldloop/cli.py` as
`--model matrix_game_3_5` (additive branch, no existing model touched).

* `reset(image, prompt, seed)` — crops to 1280x704 and writes the anchor.
* `set_pose(yaw, pitch)` / `set_fov(fov)` — optional; only deltas matter.
* `plan(segments, fps)` — one `/generate` call for the whole case:
  * resamples `compile_timeline` at `native_fps` (piecewise-linear; the timeline is
    1-based, `tl.yaw[i]` is the pose at `t=(i+1)/fps`), builds pose 0 = anchor and
    pose `j` = `t=j/native_fps`, writes `extrinsics_c2w` + `intrinsics`;
  * `num_blocks = ceil(needed_frames/80)` capped at the Space's 2;
  * decodes the returned mp4 with cv2, aligns the frame grid, resizes to 1280x720
    (BGR uint8) and caches.
* `step(action, seconds)` — returns `seconds*native_fps` cached frames, advancing a
  cursor.

Bookkeeping for a 6 s / 8 fps loop360 case: 96 model frames needed → 97 poses are
written into `camera.npz`; the Space's `_load_camera` pads to `1 + 84*2 = 169` poses
by repeating the last one (the padded tail is never evaluated). 2 blocks are
requested because one block is only 80 frames. `frame_pose_offset=1` (decoded
frame *k* shows pose *k+1*, mirroring `infer.py`'s `[1 + 84k, ...]` pose slicing)
and `frame_drop = native_fps/case_fps − offset = 1` put case frame *i* on pose
`2i+2`, i.e. exactly `t=(i+1)/8`. The returned video's declared fps is checked and
a mismatch is recorded in `meta()["plan"]["notes"]` instead of silently skewing the
mapping.

## 5. Offline end-to-end test (fake Space client)

`scratch/mg35_e2e_fake.py` replaces the network with a client that reads the very
`camera.npz` the adapter uploads and renders the panorama along that path (same
arguments, same 1280x704 output, mp4-encoded). This exercises the whole adapter
path without a GPU.

| check | result |
| --- | --- |
| frames produced for the 6 s case | 48 (720, 1280, 3) |
| adapter pose at case frame *i* vs `compile_timeline` | max err **0.000°** (all 48) |
| frame *i* vs benchmark render at that pose | mean abs diff **3.2–3.6** (mp4+resize only) |
| same frame vs the pose one case-frame earlier | **25–42** (check has teeth) |
| flow direction | matches GT sign |
| 24 fps video vs `native_fps=16` | warning recorded, as designed |

Record: `scratch/mg35_val/offline_e2e.json`. Note: the flow *magnitude* cannot be
cross-checked with a global-translation estimator — for the 7.5°/frame case
(`gt_mean_dx = 103.8`) Farneback/phase-correlation return ≈68 because they report
the centre of a wide (102° hfov) rotation field where the per-pixel flow ranges
from 68 px to >200 px; only a dense predictor (RAFT, which the scorer uses) is
comparable.

## 6. Open risks (unverifiable without a real call)

1. **`frame_pose_offset`**: assumed the returned video starts at pose 1 (the anchor
   is not repeated as frame 0). If the Space returns the anchor as frame 0, set
   `frame_pose_offset=0`; the constant one-frame (62 ms) pose offset would
   otherwise cost ≈3.75° of yaw on the revisit frame.
2. **Intrinsics normalisation**: `app.py` notes "only fx/cx matters after the
   dataset's intrinsics normalisation". The adapter sends the *true* K of the
   anchor (514.13) rather than the Space's default 0.62·W ≈ 794 (which assumes a
   77.5° hfov; our frames are 102.6° hfov / 70° vertical). If the pipeline
   disagrees, `fov_axis="h"` (914.0) is available as a knob.
3. **Possible history strip**: standalone `infer.py` strips a top row from the
   validation video (`writer.write(frame[row_h:])`), while `app.py` copies the mp4
   whole. If the returned video carries that strip, the first ~30 px of every frame
   must be cropped — not handled automatically.
4. **`native_fps=16`** is the brief's value; the Space writes its anchor video at
   24 fps. Whatever the returned mp4 declares is recorded in the plan info, and a
   mismatch is surfaced rather than hidden.
5. **Model compliance** with a commanded trajectory is exactly what the benchmark
   measures — unknown until a real (PRO) run happens.
