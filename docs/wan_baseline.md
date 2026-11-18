# Wan2.1 T2V baseline (negative control) — measured

Adapter: `worldloop/adapters/wan_t2v.py` (`WanT2VBaseline`, `name = "wan2.1_t2v_1.3b"`).

Purpose: a model that *generates video but cannot be controlled*. `step()` receives `action`,
stores it in `self.ignored_actions`, and never uses it. Every chunk is the IID sample
`(prompt, seed + k*1000)`; chunk boundaries are hidden with a cross-dissolve onto the
previous chunk's last frame. So action-control metrics should land at ~0 and drift metrics
high — that is the whole point of the baseline.

## Environment (isolated from `.venv`)

```
py -3.10 -m venv C:\Users\27236\Desktop\vla\.venv_wan
.venv_wan\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv_wan\Scripts\python.exe -m pip install diffusers transformers accelerate safetensors imageio imageio-ffmpeg opencv-python-headless
```

Installed: torch 2.11.0+cu128, torchvision 0.26.0+cu128, diffusers 0.41.0,
transformers 5.20.2, accelerate 1.15.0, safetensors 0.8.0, opencv-python-headless 5.0.0.93.

## Weights

`models/Wan2.1-T2V-1.3B-Diffusers/` (16.4 GB on disk):

| part | source | notes |
|---|---|---|
| `transformer/` | ModelScope `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` (5.6 GB, fp32) | chunked parallel download, `scripts/dl_wan.py` |
| `vae/` | same repo (0.51 GB) | |
| `tokenizer/`, `scheduler/`, `model_index.json` | same repo | |
| `text_encoder/` | **local conversion** of `Desktop\wan2.2\...\nsfw_wan_umt5-xxl_fp8_scaled.safetensors` (6.3 GB fp8) → 11.36 GB bf16, `scripts/convert_umt5_fp8.py` | the official fp32 encoder is 21.7 GB and would have needed ~70 min at this box's 5 MB/s; the fp8 file has HF T5 keys and direct-cast e4m3 weights, so `to(bf16)` is a faithful dequant |

Downloaded the official fp32 encoder? No — only the DiT + VAE (6.2 GB). The text encoder is a
converted ComfyUI fp8 umt5-xxl; it is *not* byte-identical to `Wan-AI`'s fp32 encoder
(fp8 e4m3 quantization noise + the "nsfw" fine-tune it was saved from). If quality matters
more than time, delete `text_encoder/` and run `scripts/dl_wan.py` (it will fetch the
official 21.7 GB fp32 shards; ~70 min at 5 MB/s).

## Measured (RTX 5060 Ti 16 GB, 832x480 -> 1280x720, 33 frames per chunk)

| | |
|---|---|
| sampling steps | 20 (`UniPCMultistepScheduler`, flow matching, guidance 5.0) |
| wall time per 2 s chunk | **193.9 / 192.4 s** (run 1), **180.3 / 326.6 s** (run 2; second chunk slowed by another process using the GPU). Uncontended ≈ 180–195 s, i.e. 6.6–8.4 s per DiT step |
| peak VRAM (`torch.cuda.max_memory_allocated`) | **4.67 GB** (identical in every run) |
| peak VRAM (whole process, nvidia-smi) | ~6.4 GB (desktop baseline ~2.5 GB) |
| pipeline load | 15–27 s warm, 24 s CLI construction |
| realtime factor | ~0.010 s wall per second of video (~100x slower than real time) |
| determinism | same seed + same step index -> byte-identical frames across runs |
| cost model | ~95 s per 1 s of output at 20 steps; 81-frame (5 s) clips are ~2.2x a 33-frame clip |

Measured continuity inside the smoke rollout (mean abs pixel diff): 3.37 between frames 16
and 32 inside a chunk, 0.79 across the chunk boundary (the cross-dissolve), 56.2 between
frame 0 and frame 16 -- the T2V model immediately leaves the conditioning scene, which is
exactly the failure mode a negative control should exhibit.

Long actions are split into sub-clips of ≤81 frames: the 18 s `drift` case becomes
4 clips x 73 frames ≈ 4 x 7 min ≈ 28 min. With `--steps 6` the same case is ~3x faster.

## Reproduce

```powershell
# one smoke rollout (2 chunks, 2 s each), from a benchmark panorama
.venv_wan\Scripts\python.exe scratch\wan_smoke\run_smoke.py

# action-blindness proof, no GPU minutes
.venv_wan\Scripts\python.exe scratch\wan_smoke\check_action_blind.py

# benchmark run (parent process, one case at a time)
.venv_wan\Scripts\python.exe -m worldloop.cli run --model wan --cases cases\cases.json --out results\wan_t2v --steps 20 --seed 42
```

Artifacts: `scratch/wan_smoke/rollout_2chunks.mp4`, `frame_000.jpg`, `frame_033.jpg`,
`first_frame.png`, `smoke_meta.json`.

`WanT2VBaseline(keep_history=True)` additionally accumulates every emitted frame in
`model.last_frames` (the smoke script uses it). It is **off by default**: 300 frames of
720p is ~800 MB, and the runner collects the frames it needs anyway.

## Integration

`runner.run_case` was exercised end-to-end with a stubbed `_generate` (no GPU minutes):
2 segments x 2 s at `fps=8` -> 32 resampled frames written, `rollout.mp4` + `meta.json`
produced, and both `StepRecord`s recorded. The real generation path is the smoke run above.
The adapter also resizes an off-size start image before anchoring, so a case rendered at
another resolution cannot break the stitching.

## Memory strategy

`pipe.text_encoder` stays on the CPU (a T5 forward is one prompt per chunk, ~10-20 s); the
DiT (2.6 GB bf16) and the fp32 Wan VAE run on the GPU with `vae.enable_slicing()` +
`enable_tiling()`. `enable_model_cpu_offload()` is available (`offload=True`) but not
needed: moving the 11 GB encoder to the GPU each call would peak at ~15.7 GB on this card.
