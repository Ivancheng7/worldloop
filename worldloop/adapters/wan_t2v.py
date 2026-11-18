"""Negative-control baseline: Wan2.1 text-to-video, *without* action conditioning.

Why this exists
---------------
``WorldLoop`` asks whether a model can be *played*: does it respond to actions in a
causally correct way, keep the scene stable, and remember what it looked away from?
A plain text-to-video model is the natural floor for that: it produces convincing
video, but its output is a function of (prompt, seed) only.

This adapter therefore **ignores the ``action`` argument completely** -- the string is
received, counted, and thrown away.  Each chunk is a fresh IID sample.

How the (fake) closed loop is kept
----------------------------------
There is no action signal, but we still want the rollout to look like one continuous
video so the benchmark is measuring *control*, not *cuts*:

* ``reset()`` stores the first frame and the prompt.
* ``step()`` generates a clip with Wan2.1-T2V-1.3B and then *appearance-anchors* it to
  the previous chunk's last frame: frame 0 is replaced by that frame and the first
  ``stitch_frames`` frames are cross-dissolved out of it (plus a soft per-channel
  mean match).  The next chunk starts from this new last frame.

So consecutive chunks are seamless at the boundary while the *motion* inside them is
unrelated to the requested action -- which is exactly the behaviour the benchmark's
action-control metrics should punish with ~0.

``Wan2.1-T2V-1.3B`` has no image conditioning at all (``transformer.config.image_dim`` is
``None``; only the 14B I2V checkpoints have it), so "previous frame as image condition" can
only be faked, and the cross-dissolve above is that fake.  This is deliberate: the adapter
must not be *better* than a text-to-video model actually is.
"""

from __future__ import annotations

import os
import pathlib
import time

import numpy as np

from .base import ClosedLoopModel

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_MODEL_ID = os.environ.get(
    "WAN_MODEL_ID", str(_REPO_ROOT / "models" / "Wan2.1-T2V-1.3B-Diffusers")
)

# A "generic photoreal video" prompt: an action-blind model gets one fixed idea of the
# scene and nothing else.  Kept short because umt5 truncates at 512 tokens.
DEFAULT_PROMPT = (
    "a photorealistic first person walkthrough view of a real indoor place, "
    "wide angle lens, natural light, sharp details, smooth camera motion"
)

# Wan's own recommended negative prompt for the 1.3B model.
NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, "
    "many people in the background, walking backwards"
)


class WanT2VBaseline(ClosedLoopModel):
    """Wan2.1-T2V-1.3B wrapped as a closed-loop model that ignores actions."""

    name = "wan2.1_t2v_1.3b"
    native_fps = 16.0  # Wan2.1: 16 fps, num_frames must be 4k+1
    min_chunk_seconds = 5.0 / 16.0  # 5 frames
    note = "action-blind text-to-video baseline; step() discards `action` and re-samples each chunk"

    # One pipeline per (model_id, dtype, device, placement): the benchmark builds a new
    # adapter for every case, and re-loading 16 GB of weights per case would be absurd.
    _PIPE_CACHE: dict = {}

    def __init__(
        self,
        model_id: str | None = None,
        steps: int = 20,
        width: int = 832,
        height: int = 480,
        out_width: int = 1280,
        out_height: int = 720,
        device: str = "cuda",
        dtype: str = "bfloat16",
        guidance_scale: float = 5.0,
        offload: bool = False,
        text_encoder_on_cpu: bool = True,
        prompt: str | None = None,
        stitch_frames: int = 4,
        color_anchor: bool = True,
        max_frames: int = 81,
        seed_stride: int = 1000,
        keep_history: bool = False,
        verbose: bool = True,
        _pipe=None,
    ) -> None:
        super().__init__()
        self.model_id = model_id or DEFAULT_MODEL_ID
        self.steps = int(steps)
        self.width = int(width) - int(width) % 16
        self.height = int(height) - int(height) % 16
        self.out_width = int(out_width)
        self.out_height = int(out_height)
        self.device = device
        self.dtype_name = dtype
        self.guidance_scale = float(guidance_scale)
        self.offload = bool(offload)
        self.text_encoder_on_cpu = bool(text_encoder_on_cpu)
        self.user_prompt = prompt
        self.stitch_frames = int(stitch_frames)
        self.color_anchor = bool(color_anchor)
        self.max_frames = 4 * ((int(max_frames) - 1) // 4) + 1
        self.seed_stride = int(seed_stride)
        # Off by default: the runner already collects frames, and 300 frames of 720p is
        # ~800 MB of RAM on a box that is already paging.
        self.keep_history = bool(keep_history)
        self.verbose = bool(verbose)

        self._pipe = _pipe if _pipe is not None else self._load_pipeline()
        self._gen = None  # torch.Generator, lazily created
        self._torch = None
        self._cv2 = None
        self._pil = None

        self._prompt = DEFAULT_PROMPT
        self._seed = 0
        self._step_index = 0
        self._last_frame: np.ndarray | None = None
        self.last_frames: list[np.ndarray] = []  # full history, BGR uint8
        self.ignored_actions: list[str] = []  # evidence that action never reaches the model

    # ------------------------------------------------------------------ loading
    def _load_pipeline(self):
        key = (os.path.abspath(self.model_id) if os.path.isdir(self.model_id) else self.model_id,
               self.dtype_name, self.device, self.offload, self.text_encoder_on_cpu)
        if key in self._PIPE_CACHE:
            return self._PIPE_CACHE[key]

        import torch
        from diffusers import AutoencoderKLWan, WanPipeline

        if str(self.device).startswith("cuda") and not torch.cuda.is_available():
            self.device = "cpu"
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(
            self.dtype_name, torch.bfloat16
        )
        if self.verbose:
            print(f"[wan] loading {self.model_id} dtype={self.dtype_name} device={self.device} ...", flush=True)
        t0 = time.perf_counter()
        vae = AutoencoderKLWan.from_pretrained(self.model_id, subfolder="vae", torch_dtype=torch.float32)
        pipe = WanPipeline.from_pretrained(self.model_id, vae=vae, torch_dtype=dtype)

        pipe.vae.enable_slicing()
        try:
            pipe.vae.enable_tiling()
        except Exception:  # noqa: BLE001 - tiling is optional
            pass
        if self.offload and torch.cuda.is_available():
            pipe.enable_model_cpu_offload()
        else:
            # Explicit placement: keep the 11 GB umt5 encoder on the CPU and the DiT/VAE
            # on the GPU.  This is what keeps the measured peak under ~8 GB on a 16 GB card
            # (the encoder alone would need 11 GB of VRAM); the cost is one T5 forward on
            # CPU per chunk (~10-20 s), which is small next to 20 DiT steps.
            if self.text_encoder_on_cpu:
                pipe.text_encoder.to("cpu")
            else:
                pipe.text_encoder.to(self.device)
            pipe.transformer.to(self.device)
            pipe.vae.to(self.device)

        self._PIPE_CACHE[key] = pipe
        if self.verbose:
            print(f"[wan] pipeline ready in {time.perf_counter() - t0:.1f}s", flush=True)
        return pipe

    # ------------------------------------------------------------------ helpers
    def _lazy(self):
        if self._torch is None:
            import cv2
            import torch
            from PIL import Image

            self._torch, self._cv2, self._pil = torch, cv2, Image
        return self._torch, self._cv2, self._pil

    def _n_frames(self, seconds: float) -> int:
        """Wan needs num_frames = 4k+1; round the requested duration to the nearest."""
        raw = seconds * self.native_fps
        k = max(1, int(round((raw - 1.0) / 4.0)))
        return 4 * k + 1

    def _resize_bgr(self, rgb: np.ndarray) -> np.ndarray:
        _, cv2, _ = self._lazy()
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if (bgr.shape[1], bgr.shape[0]) != (self.out_width, self.out_height):
            bgr = cv2.resize(bgr, (self.out_width, self.out_height), interpolation=cv2.INTER_LANCZOS4)
        return bgr

    # ------------------------------------------------------------------ API
    def reset(self, image: np.ndarray, prompt: str = "", seed: int = 0) -> None:
        assert image is not None and image.ndim == 3 and image.shape[2] == 3
        self._prompt = (prompt or self.user_prompt or DEFAULT_PROMPT).strip()
        self._seed = int(seed)
        self._step_index = 0
        self._last_frame = np.ascontiguousarray(image.astype(np.uint8)).copy()
        self.last_frames = [self._last_frame.copy()] if self.keep_history else []
        self.ignored_actions = []
        self.records.clear()

    def _generate(self, n_frames: int) -> list[np.ndarray]:
        """One text-to-video chunk.  No action, no image, only (prompt, seed)."""
        torch, _, _ = self._lazy()
        # A fresh generator per chunk: chunk k is the IID sample seed + k*stride.
        self._gen = torch.Generator(device="cpu").manual_seed(self._seed + self._step_index * self.seed_stride)
        pipe = self._pipe

        kwargs = dict(
            prompt=self._prompt,
            negative_prompt=NEGATIVE_PROMPT,
            height=self.height,
            width=self.width,
            num_frames=int(n_frames),
            num_inference_steps=self.steps,
            guidance_scale=self.guidance_scale,
            generator=self._gen,
            output_type="np",
        )
        out = pipe(**kwargs)
        arr = out.frames[0]  # (F, H, W, 3) float in [0, 1]
        arr = np.clip(arr, 0.0, 1.0)
        rgb = (arr * 255.0 + 0.5).astype(np.uint8)
        return [self._resize_bgr(f) for f in rgb]

    def _anchor(self, frames: list[np.ndarray], prev: np.ndarray) -> list[np.ndarray]:
        """Seamlessly attach a fresh clip to the previous chunk's last frame.

        Purely cosmetic continuity: frame 0 := previous last frame, then the next
        ``stitch_frames`` frames are cross-dissolved out of it.  It does not make the
        motion action-dependent; it only hides the cut.
        """
        if not frames:
            return frames
        out = [f.copy() for f in frames]
        # the start image may arrive at any resolution; anchor in the generated one
        h, w = out[0].shape[:2]
        if prev.shape[:2] != (h, w):
            _, cv2, _ = self._lazy()
            prev = cv2.resize(prev, (w, h), interpolation=cv2.INTER_AREA)
        n = min(self.stitch_frames, len(out))
        anchor = prev.astype(np.float32)
        for i in range(n):
            alpha = (i + 1.0) / (n + 1.0)  # 0 -> anchor, 1 -> generated
            if i == 0:
                out[i] = prev.copy()
            else:
                out[i] = np.clip(alpha * out[i].astype(np.float32) + (1.0 - alpha) * anchor, 0, 255).astype(np.uint8)

        if self.color_anchor:
            # One global per-channel offset (clip mean -> anchor mean): cheap, keeps the
            # clip's own brightness variation, no per-frame flicker injected.
            ref = prev.reshape(-1, 3).mean(axis=0)
            cur = np.stack([out[i].reshape(-1, 3).mean(axis=0) for i in range(1, len(out))]).mean(axis=0) \
                if len(out) > 1 else ref
            shift = (ref - cur).astype(np.float32)
            for i in range(1, len(out)):
                out[i] = np.clip(out[i].astype(np.float32) + shift, 0, 255).astype(np.uint8)
        return out

    def step(self, action: str, seconds: float) -> list[np.ndarray]:
        if self._last_frame is None:
            raise RuntimeError("reset() must be called before step()")
        t0 = self._time()
        torch, _, _ = self._lazy()

        # ------------------------------------------------------------------
        # The negative-control bit: `action` is recorded and then discarded.
        # Nothing below this line can see it -- the chunk depends only on
        # (prompt, seed, previous frame).
        # ------------------------------------------------------------------
        self.ignored_actions.append(action)

        n_frames = self._n_frames(seconds)
        use_cuda = torch.cuda.is_available() and (self.offload or str(self.device).startswith("cuda"))
        if use_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        # Long actions are split into sub-clips of at most max_frames (81 = 5 s, Wan's
        # native training length) so cost/memory stay linear instead of quadratic.
        # Work in "k" units (frames = 4k+1) and split k evenly: no degenerate tail clip.
        k_total = max(1, (n_frames - 1) // 4)
        k_max = max(1, (self.max_frames - 1) // 4)
        n_clips = -(-k_total // k_max)  # ceil
        base_k, rem = divmod(k_total, n_clips)

        frames: list[np.ndarray] = []
        prev = self._last_frame
        for i in range(n_clips):
            k = 4 * (base_k + (1 if i < rem else 0)) + 1
            clip = self._anchor(self._generate(k), prev)
            frames.extend(clip)
            prev = clip[-1]
            self._step_index += 1

        if use_cuda:
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / (1024 ** 3)
        else:
            peak = None

        self._last_frame = frames[-1].copy()
        if self.keep_history:
            self.last_frames.extend(f.copy() for f in frames)
        self._record(action, seconds, len(frames), t0, peak)
        if self.verbose:
            r = self.records[-1]
            peak_str = f"{peak:.2f}GB" if peak is not None else "n/a"
            print(
                f"[wan] step {self._step_index} action={action!r} (ignored) "
                f"{len(frames)}f {self.width}x{self.height} -> {self.out_width}x{self.out_height} "
                f"{r.latency_s:.1f}s peak={peak_str}",
                flush=True,
            )
        return frames

    def close(self) -> None:
        # Keep the pipeline cached: the runner builds one adapter per case.
        self._last_frame = None

    @classmethod
    def unload(cls) -> None:
        """Drop the cached pipeline (frees VRAM/RAM)."""
        cls._PIPE_CACHE.clear()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass
