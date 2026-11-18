# WorldLoop Case Set v0.1 — dataset card

## What it is

25 reproducible evaluation cases for interactive world models. Each case is

* one `1280×720` start frame rendered from a CC0 panorama,
* a chunk-level action script (yaw/pitch commands in 2 s multiples; case durations 4 s–18 s),
* the analytic pose timeline and per-frame ground-truth flow,
* three probe regions (world directions) for object permanence.

No model output is included: the case set is the *input* to the benchmark and is
identical for every model.

## Composition

| protocol | n | duration | purpose |
|---|---|---|---|
| `loop360` | 6 | 6 s | full turn, revisit the start view |
| `occlusion` | 6 | 6 s | 120° look-away (probes off-screen 4 s) |
| `null_action` | 6 | 4 s | static world, motion floor |
| `counterfactual` | 6 (2 groups × 3 branches) | 6 s | left / right / no-op from one start |
| `drift` | 1 | 18 s | three consecutive loops |
| **total** | **25** | 150 s of video | |

## Scenes

Ten CC0 panoramas from [Poly Haven](https://polyhaven.com), downscaled to
`4096×2048` equirect: `abandoned_bakery`, `empty_warehouse_01`,
`small_empty_room_1`, `small_empty_room_2`, `brown_photostudio_02`,
`studio_small_03`, `hotel_room`, `lebombo`, `venice_sunset`,
`modern_buildings` (10 downloaded, 6 used for the core protocols; the rest are
reserved for the OOD split).

The start heading of each scene is `i × 47° mod 360°`, so cases differ in what is
in front of the camera without any cherry-picking.

## Files

```
cases/
  cases.json            # everything: segments, poses, probes, tags
  start/<case_id>.png   # the conditioned first frame (1280x720)
```

`cases.json` entry:

```json
{
  "case_id": "abandoned_bakery__loop360",
  "protocol": "loop360",
  "pano": "data/panos/abandoned_bakery.jpg",
  "start_yaw": 0.0, "start_pitch": 0.0, "fov": 1.2217,
  "width": 1280, "height": 720, "fps": 8.0,
  "segments": [{"action": "turn_right", "seconds": 2.0, "tag": "loop"}, ...],
  "probes": [{"lon": ..., "lat": ..., "size": 96, "ref_px": 880.0, "ref_py": 400.0}, ...]
}
```

## Regenerating

```bash
python scripts/fetch_panos.py --out data/panos --n 10
python -m worldloop.cli gen-cases --n-panos 6 --out cases \
    --protocols loop360 occlusion null_action --cf-scenes 2
```

Generation is deterministic: same panoramas + same code ⇒ byte-identical
`cases.json` and start frames (JPEG encoding aside).

## Splits and leakage

There is no train/test split because nothing is trained. The un-used panoramas
(`hotel_room`, `lebombo`, `venice_sunset`, `modern_buildings`) form an optional
**OOD split** — scenes with different lighting (night sky, outdoor sunset) that
stress the model's prior rather than its memory.

## Licence

Panoramas are CC0 (Poly Haven). Start frames and scripts are derived works
released under CC0 as well. Code: Apache-2.0.

## Known issues

* `small_empty_room_1` resolves to a near-featureless wall in some headings; probe
  selection still finds gradient edges but the case is visually plain.
* Rotation-only motion: no translation, no physics.
* 8 fps timeline; models trained at 16 fps are temporally subsampled by the runner.
