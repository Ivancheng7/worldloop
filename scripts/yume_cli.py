#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
yume_cli.py -- programmatic inference entry point for Yume-5B (Yume-1.5) 720P.

Wraps the repository's *single-GPU* sampler (webapp_single_gpu.long_generate),
which is the path the author validated on a 16 GB RTX 4090 Laptop, and adds:

  * a discrete action timeline parser (keyboard WASD / mouse arrows)
  * chunked generation (33 frames = 2.06 s @ 16 fps per chunk), each chunk gets
    one (keys, mouse) pair because the model has a single discrete camera token
    per generated chunk
  * closed loop continuation (chunk N conditions on chunk N-1: continue_from_last)
  * mp4 + optional PNG frame dump + meta.json with timings / peak VRAM / the
    exact action -> prompt mapping that was used

Usage:
  python yume_cli.py --image <path> --prompt "<text>" \
      --actions "W:10,A:10" --frames 33 --steps 6 --outdir <dir> --seed 42

Action grammar (each comma separated item is  <keys>|<mouse>:<frames> ):
  keys   : W A S D W+A W+D S+A S+D None          (camera translation; discrete)
  mouse  : up down left right none               (ASCII aliases: u d l r -)
  frames : how long that state is held (frames, 16 fps)
  "none"/"-"        = no key / mouse stationary
  a bare "W" (single item, no :N) means "hold W for the whole clip"
  the last state is held until the end if the timeline is shorter than --frames
Each generated chunk (default 33 frames) is assigned the majority action of the
frames it covers; the per-chunk vote is recorded in meta.json.

The hint that the camera params are only text tokens (not tensors) is confirmed
by the repo: scripts/inference/sample_5b.sh feeds caption_re.txt whose lines are
  "First-person perspective. The camera pushes forward (W). The rotation
   direction of the camera remains stationary (DOT). Actual distance moved:4 at
   100 meters per second. Angular change rate (turn speed):0. View rotation speed:0."
followed by the scene description.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# --------------------------------------------------------------------------
# paths / env: must be set before importing torch / the repo code
# --------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", os.path.join(ROOT, ".cache", "huggingface"))

DEFAULT_CKPT = os.path.abspath(
    os.path.join(ROOT, "..", "..", "models", "Yume-5B-720P")
)
if not os.path.isdir(DEFAULT_CKPT):  # fall back to the layout the README assumes
    DEFAULT_CKPT = os.path.join(ROOT, "Yume-5B-720P")

# --------------------------------------------------------------------------
# action vocabulary (taken verbatim from webapp_single_gpu.long_generate).
# All non-ASCII is expressed with \u escapes so the file survives any editor /
# codepage round trip:  RIGHT=\u2192  LEFT=\u2190  UP=\u2191  DOWN=\u2193  DOT=\u00b7
# --------------------------------------------------------------------------
ARROW_R, ARROW_L, ARROW_U, ARROW_D, DOT = (
    "\u2192", "\u2190", "\u2191", "\u2193", "\u00b7")

KEY_VOCAB = {
    "W": "The camera pushes forward (W).",
    "A": "The camera moves to the left (A).",
    "S": "The camera pulls back (S).",
    "D": "The camera moves to the right (D).",
    "W+A": "The camera pushes forward and moves to the left (W+A).",
    "W+D": "The camera pushes forward and moves to the right (W+D).",
    "S+D": "The camera pulls back and moves to the right (S+D).",
    "S+A": "The camera pulls back and moves to the left (S+A).",
    "None": "The camera's movement direction remains stationary (%s)." % DOT,
}
MOUSE_VOCAB = {
    "right": "The camera pans to the right (%s)." % ARROW_R,
    "left": "The camera pans to the left (%s)." % ARROW_L,
    "up": "The camera tilts up (%s)." % ARROW_U,
    "down": "The camera tilts down (%s)." % ARROW_D,
    "up+right": "The camera tilts up and pans to the right (%s)." % (ARROW_U + ARROW_R),
    "up+left": "The camera tilts up and pans to the left (%s)." % (ARROW_U + ARROW_L),
    "down+right": "The camera tilts down and pans to the right (%s)." % (ARROW_D + ARROW_R),
    "down+left": "The camera tilts down and pans to the left (%s)." % (ARROW_D + ARROW_L),
    "none": "The rotation direction of the camera remains stationary (%s)." % DOT,
}
# webapp_single_gpu.LongGenArgs wants the *repo's own* glyphs for the mouse
MOUSE_TO_REPO = {
    "right": ARROW_R, "left": ARROW_L, "up": ARROW_U, "down": ARROW_D,
    "up+right": ARROW_U + ARROW_R, "up+left": ARROW_U + ARROW_L,
    "down+right": ARROW_D + ARROW_R, "down+left": ARROW_D + ARROW_L,
    "none": DOT,
}
MOUSE_ALIASES = {
    ARROW_R: "right", "->": "right", "r": "right",
    ARROW_L: "left", "<-": "left", "l": "left",
    ARROW_U: "up", "^": "up", "u": "up",
    ARROW_D: "down", "v": "down", "d": "down",
    DOT: "none", "-": "none", "": "none", "0": "none", "stationary": "none",
    "rightup": "up+right", "up+right": "up+right", ARROW_R + ARROW_U: "up+right",
    "upleft": "up+left", "up+left": "up+left", ARROW_L + ARROW_U: "up+left",
    "downright": "down+right", "down+right": "down+right", ARROW_R + ARROW_D: "down+right",
    "downleft": "down+left", "down+left": "down+left", ARROW_L + ARROW_D: "down+left",
}


def norm_keys(raw: str) -> str:
    """Normalise a keyboard token to one of KEY_VOCAB."""
    s = (raw or "").strip().upper().replace(" ", "").replace("_", "+")
    if s in ("", "NONE", DOT, "-", "N", "0", "STILL", "NO"):
        return "None"
    if s in KEY_VOCAB:
        return s
    parts = sorted(set(tok for tok in s.replace("+", "").replace(",", "") if tok))
    cand = "+".join(parts)
    if cand in KEY_VOCAB:
        return cand
    raise ValueError(
        f"unsupported keyboard action {raw!r}; expected one of "
        f"{sorted(KEY_VOCAB)} (e.g. W, A, W+A, None) or 'W|right'"
    )


def norm_mouse(raw: str) -> str:
    s = (raw or "").strip().lower().replace(" ", "")
    if s in MOUSE_ALIASES:
        return MOUSE_ALIASES[s]
    if s in (MOUSE_VOCAB.keys()):
        return s
    raise ValueError(
        f"unsupported mouse action {raw!r}; expected one of "
        f"up/down/left/right/up+right/up+left/down+right/down+left/none"
    )


@dataclass
class Action:
    keys: str
    mouse: str
    frames: int
    start: int = 0

    @property
    def end(self) -> int:
        return self.start + self.frames

    def as_dict(self) -> Dict[str, Any]:
        return {
            "frame_start": self.start,
            "frame_end": self.end - 1,
            "frames": self.frames,
            "keys": self.keys,
            "mouse": self.mouse,
        }


def parse_actions(spec: str, total_frames: int) -> Tuple[List[Action], List[str]]:
    """Parse --actions into a per-frame timeline of at least `total_frames`."""
    spec = (spec or "").strip()
    if not spec:
        spec = "None|none"
    raw_items = [it.strip() for it in spec.split(",") if it.strip()]
    warns: List[str] = []
    acts: List[Action] = []
    cur = 0
    for i, item in enumerate(raw_items):
        frames: Optional[int] = None
        if ":" in item:
            head, tail = item.rsplit(":", 1)
            try:
                frames = int(tail)
            except ValueError:
                raise ValueError(f"bad frame count in action item {item!r}")
            item = head
        if "|" in item:
            k, m = item.split("|", 1)
        else:
            k, m = item, "none"
        keys, mouse = norm_keys(k), norm_mouse(m)
        if frames is None:
            if len(raw_items) != 1 and i != len(raw_items) - 1:
                raise ValueError(
                    f"action item {item!r} is missing ':N' and is not the last item"
                )
            frames = max(total_frames - cur, 0)
        if frames < 0:
            raise ValueError(f"negative frame count in {item!r}")
        if frames == 0:
            continue
        acts.append(Action(keys, mouse, frames, start=cur))
        cur += frames
    if cur < total_frames:
        last = acts[-1]
        acts[-1] = Action(last.keys, last.mouse, last.frames + (total_frames - cur), last.start)
        warns.append(
            f"action timeline covered {cur}/{total_frames} frames; "
            f"held {acts[-1].keys}|{acts[-1].mouse} for the remaining "
            f"{total_frames - cur} frames"
        )
    elif cur > total_frames:
        warns.append(
            f"action timeline covers {cur} > {total_frames} frames; the tail is ignored"
        )
    return acts, warns


def actions_for_chunk(acts: List[Action], start: int, end: int) -> Tuple[str, str, Dict[str, int]]:
    """Majority vote of the per-frame actions inside [start, end)."""
    votes: Dict[str, int] = {}
    for a in acts:
        lo, hi = max(a.start, start), min(a.end, end)
        if hi <= lo:
            continue
        tag = f"{a.keys}|{a.mouse}"
        votes[tag] = votes.get(tag, 0) + (hi - lo)
    if not votes:
        return "None", "none", {}
    # deterministic: highest vote count, tie -> the one that starts earliest
    order = {}
    for a in acts:
        order.setdefault(f"{a.keys}|{a.mouse}", a.start)
    best = max(votes.items(), key=lambda kv: (kv[1], -order[kv[0]]))[0]
    keys, mouse = best.split("|", 1)
    return keys, mouse, votes


def build_numeric_sentence(distance: float, angular: float, rotation: float) -> str:
    def n(v):
        f = float(v)
        return str(int(f)) if f.is_integer() else repr(f)
    return (
        f"Actual distance moved:{n(distance)} at 100 meters per second. "
        f"Angular change rate (turn speed):{n(angular)}. "
        f"View rotation speed:{n(rotation)}."
    )


def camera_sentence(keys: str, mouse: str) -> str:
    s = "First-person perspective."
    if keys != "None":
        s += " " + KEY_VOCAB[keys]
    if mouse != "none":
        s += " " + MOUSE_VOCAB[mouse]
    return s


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
SERVE_PREFIX = "__YUME_JSON__:"


def gb(n: float) -> float:
    return round(float(n) / (1024 ** 3), 3)


def _emit_serve_json(payload: Dict[str, Any]) -> None:
    sys.stdout.write(SERVE_PREFIX + json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def make_capture_hook(
    captured: List[Any], chunk_frames: int, resume_from: Optional[str] = None
):
    """Build a ``webapp_single_gpu._postprocess_video`` hook that slices continuation tails.

    When ``continue_from_last=True`` (``c > 0`` or ``resume_from``),
    ``webapp_single_gpu.long_generate`` prepends ``model_input_de`` (16 static
    head frames + the previous chunk) onto ``video_tail``; slicing the trailing
    ``chunk_frames`` keeps only the newly generated frames.
    """

    def _hook(video, fps, out_path):
        tail = video[:, -chunk_frames:] if (captured or resume_from) else video
        captured.append(tail.detach().float().cpu().clone())
        return out_path

    return _hook


def pin_t5_to_cpu(wan) -> None:
    """Keep the 10.6 GB T5 encoder permanently on CPU across chunks."""
    import torch

    wan.t5_cpu = True
    te = getattr(wan, "text_encoder", None)
    if te is None:
        return
    te.device = torch.device("cpu")
    if hasattr(te, "model") and te.model is not None:
        te.model = te.model.cpu()
        orig_to = te.model.to

        def _cpu_only_to(*args, **kwargs):
            new_args = []
            for a in args:
                if isinstance(a, str) and a.startswith("cuda"):
                    new_args.append("cpu")
                elif isinstance(a, torch.device) and a.type == "cuda":
                    new_args.append(torch.device("cpu"))
                else:
                    new_args.append(a)
            if "device" in kwargs:
                d = kwargs["device"]
                if (isinstance(d, str) and d.startswith("cuda")) or (
                    isinstance(d, torch.device) and d.type == "cuda"
                ):
                    kwargs = dict(kwargs, device=torch.device("cpu"))
            return orig_to(*new_args, **kwargs)

        te.model.to = _cpu_only_to

    orig_cls = type(te)
    if not getattr(orig_cls, "_pinned_cpu_t5", False):

        class _PinnedCPUT5(orig_cls):
            _pinned_cpu_t5 = True

            def __call__(self, texts, device=None):
                with torch.amp.autocast("cuda", enabled=False):
                    return orig_cls.__call__(self, texts, torch.device("cpu"))

        te.__class__ = _PinnedCPUT5


def _reset_last_state(W, torch_mod) -> None:
    import gc as _gc

    W.LAST["last_model_input_latent"] = None
    W.LAST["last_model_input_de"] = None
    W.LAST["frame_total"] = 0
    W.LAST["last_video_path"] = None
    W.LAST["last_prompt"] = ""
    _gc.collect()
    if torch_mod.cuda.is_available():
        torch_mod.cuda.empty_cache()


def patch_frugal_yume_init():
    """Replace Yume.__init__ with a memory-frugal equivalent (31.6 GB RAM box).

    The upstream __init__ builds the DiT twice (from_config + from_pretrained,
    both fp32 => ~2 x 20.8 GB) and then load_file()s the 19.8 GB fp32
    safetensors a third time. On a 32 GB machine that thrashes/OOMs, so here we
    build in bf16 and stream the safetensors tensor-by-tensor, moving the DiT to
    the GPU before the 10.8 GB T5 is allocated.
    """
    import torch
    import wan23.textimage2video as T
    from safetensors import safe_open
    from wan23.modules.model import WanModel
    from wan23.modules.t5 import T5EncoderModel
    from wan23.modules.vae2_2 import Wan2_2_VAE

    def _init(self, config, checkpoint_dir, device_id=0, rank=0, t5_fsdp=False,
              dit_fsdp=False, use_sp=False, t5_cpu=False, init_on_cpu=True,
              convert_model_dtype=False):
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu
        self.init_on_cpu = init_on_cpu
        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype
        self.sp_size = 1
        self.sample_neg_prompt = config.sample_neg_prompt

        t0 = time.perf_counter()
        cfg = WanModel.load_config(checkpoint_dir)
        old = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            model = WanModel.from_config(cfg)
        finally:
            torch.set_default_dtype(old)

        sd = model.state_dict()
        path = os.path.join(checkpoint_dir, "diffusion_pytorch_model.safetensors")
        missing, unexpected = [], []
        with safe_open(path, framework="pt", device="cpu") as f:
            fk = set(f.keys())
            missing = [k for k in sd if k not in fk]
            unexpected = sorted(fk - set(sd))
            with torch.no_grad():
                for k in sd:
                    if k in fk:
                        sd[k].copy_(f.get_tensor(k))
        n_missing = len(missing)
        del sd
        if missing:
            raise RuntimeError(f"checkpoint is missing {n_missing} tensors, e.g. {missing[:5]}")
        print(f"[yume_cli] DiT: {len(fk)} tensors loaded from safetensors "
              f"(unexpected in file: {len(unexpected)}) in {time.perf_counter() - t0:.1f}s",
              flush=True)

        # free the CPU copy immediately: 10.4 GB back to the OS before T5 loads
        self.model = model.to(self.device, dtype=torch.bfloat16)
        del model
        import gc as _gc
        _gc.collect()
        torch.cuda.empty_cache()

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_2_VAE(vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
                              device=self.device)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device("cpu"),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=None,
        )

    T.Yume.__init__ = _init


def preflight(args) -> int:
    """Validate everything except the 19.8 GB DiT (VAE + T5 + tokenizer + env)."""
    import json as _json
    import numpy as np
    from PIL import Image
    import torch

    rep: Dict[str, Any] = {"checkpoint_dir": os.path.abspath(args.ckpt_dir), "files": {},
                           "env": {}, "steps": []}
    print(f"[preflight] checkpoint dir: {rep['checkpoint_dir']}")
    expect = {
        "diffusion_pytorch_model.safetensors": 19.373,
        "models_t5_umt5-xxl-enc-bf16.pth": 10.582,
        "Wan2.2_VAE.pth": 2.625,
        "config.json": 0.0,
        "google/umt5-xxl/tokenizer.json": 0.016,
    }
    for k, want in expect.items():
        p = os.path.join(rep["checkpoint_dir"], k)
        ok = os.path.isfile(p)
        gb_ = round(os.path.getsize(p) / 1024 ** 3, 3) if ok else 0.0
        rep["files"][k] = {"present": ok, "gb": gb_, "expected_gb": want,
                           "complete": ok and gb_ >= max(want * 0.93, want - 0.1)}
        print(f"  {'OK  ' if rep['files'][k]['complete'] else 'MISS'} {k:48s} {gb_:7.3f} GB")

    ok = torch.cuda.is_available()
    rep["env"] = {
        "python": sys.version.split()[0], "torch": torch.__version__,
        "cuda": torch.version.cuda, "cuda_available": ok,
        "device": torch.cuda.get_device_name(0) if ok else None,
        "capability": list(torch.cuda.get_device_capability(0)) if ok else None,
        "total_vram_gb": gb(torch.cuda.get_device_properties(0).total_memory) if ok else 0,
        "bf16_supported": torch.cuda.is_bf16_supported() if ok else False,
    }
    print(f"[preflight] {_json.dumps(rep['env'])}")

    # ---- 1) tokenizer + T5 (prompt -> text embedding) ----
    t5p = os.path.join(rep["checkpoint_dir"], "models_t5_umt5-xxl-enc-bf16.pth")
    if rep["files"].get("models_t5_umt5-xxl-enc-bf16.pth", {}).get("complete"):
        try:
            from wan23.configs import WAN_CONFIGS
            from wan23.modules.t5 import T5EncoderModel
            cfg = WAN_CONFIGS["ti2v-5B"]
            t0 = time.perf_counter()
            te = T5EncoderModel(
                text_len=cfg.text_len, dtype=cfg.t5_dtype, device=torch.device("cpu"),
                checkpoint_path=t5p,
                tokenizer_path=os.path.join(rep["checkpoint_dir"], cfg.t5_tokenizer),
                shard_fn=None)
            t_load = time.perf_counter() - t0
            t0 = time.perf_counter()
            ctx = te([args.prompt or "A robot walking down a street."], torch.device("cpu"))
            t_enc = time.perf_counter() - t0
            rep["steps"].append({"step": "t5_encode", "ok": True,
                                 "load_s": round(t_load, 2), "encode_cpu_s": round(t_enc, 2),
                                 "context_shape": list(ctx[0].shape),
                                 "context_dtype": str(ctx[0].dtype)})
            print(f"[preflight] T5 OK load {t_load:.1f}s encode(CPU) {t_enc:.1f}s "
                  f"shape {tuple(ctx[0].shape)}")
            del te, ctx
            import gc as _gc; _gc.collect()
        except Exception as e:
            rep["steps"].append({"step": "t5_encode", "ok": False,
                                 "error": f"{type(e).__name__}: {e}"})
            print(f"[preflight] T5 FAILED: {type(e).__name__}: {e}")
    else:
        rep["steps"].append({"step": "t5_encode", "ok": None, "skipped": "weights missing"})
        print("[preflight] T5 skipped (weights missing)")

    # ---- 2) VAE encode/decode round trip on a real 49-frame sequence ----
    vaep = os.path.join(rep["checkpoint_dir"], "Wan2.2_VAE.pth")
    if rep["files"].get("Wan2.2_VAE.pth", {}).get("complete"):
        try:
            from wan23.configs import WAN_CONFIGS
            from wan23.modules.vae2_2 import Wan2_2_VAE
            cfg = WAN_CONFIGS["ti2v-5B"]
            dev = torch.device("cuda:0")
            t0 = time.perf_counter()
            vae = Wan2_2_VAE(vae_pth=vaep, device=dev)
            t_load = time.perf_counter() - t0
            torch.cuda.reset_peak_memory_stats()
            H1, W1 = (704, 1280) if args.resolution == "704x1280" else (544, 960)
            F = args.chunk_frames
            # exactly the tensor webapp_single_gpu.create_video_from_image builds
            img = None
            if args.image and os.path.isfile(args.image):
                img = Image.open(args.image).convert("RGB")
                arr = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
            else:
                arr = torch.rand(3, 720, 1280)
            resized = torch.nn.functional.interpolate(
                arr.unsqueeze(0), size=(H1, W1), mode="bilinear", align_corners=False)[0]
            vid = torch.zeros(3, F, H1, W1)
            vid[:, 0] = (resized - 0.5) * 2
            vid = torch.cat([vid[:, 0:1].repeat(1, 16, 1, 1), vid], dim=1).to(dev)  # 16+F frames
            t0 = time.perf_counter()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                lat_a = vae.encode([vid[:, :-F]])[0]
                lat_b = vae.encode([vid[:, -F:]])[0]
            lat = torch.cat([lat_a, lat_b], dim=1)
            t_enc = time.perf_counter() - t0
            t0 = time.perf_counter()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = vae.decode([lat[:, -(F - 1) // 4 - 1:]])[0]
            t_dec = time.perf_counter() - t0
            peak = torch.cuda.max_memory_allocated()
            # save frame 0 as a visual sanity check
            v = out.clamp(-1, 1).add(1).div(2).mul(255).byte().cpu().numpy()
            os.makedirs(args.outdir, exist_ok=True)
            png = os.path.join(args.outdir, "preflight_vae_roundtrip_frame0.png")
            Image.fromarray(np.transpose(v[:, 0], (1, 2, 0))).save(png)
            rep["steps"].append({
                "step": "vae_roundtrip", "ok": True, "load_s": round(t_load, 2),
                "encode_2x_s": round(t_enc, 2), "decode_s": round(t_dec, 2),
                "in_frames": int(vid.shape[1]), "latent_shape": list(lat.shape),
                "out_frames": int(out.shape[1]), "out_hw": list(out.shape[2:]),
                "peak_vram_gb": gb(peak), "png": png})
            print(f"[preflight] VAE OK load {t_load:.1f}s encode {t_enc:.1f}s "
                  f"decode {t_dec:.1f}s | latent {tuple(lat.shape)} -> "
                  f"{tuple(out.shape)} | peak {gb(peak):.2f} GB -> {png}")
        except Exception as e:
            import traceback as _tb
            rep["steps"].append({"step": "vae_roundtrip", "ok": False,
                                 "error": f"{type(e).__name__}: {e}",
                                 "trace": _tb.format_exc()[-2000:]})
            print(f"[preflight] VAE FAILED: {type(e).__name__}: {e}")
    else:
        rep["steps"].append({"step": "vae_roundtrip", "ok": None, "skipped": "weights missing"})
        print("[preflight] VAE skipped (weights missing)")

    # ---- 3) DiT VRAM budget prediction ----
    dit_ok = rep["files"].get("diffusion_pytorch_model.safetensors", {}).get("complete")
    rep["predicted_dit"] = {
        "weights_bf16_gb": 9.7,
        "tokens_first_chunk_13lat": 13 * 22 * 40,
        "tokens_continuation_22lat": 22 * 22 * 40,
        "notes": "DiT 5.20B params -> 9.7 GiB in bf16 + ~1.5-2.5 GiB activations. "
                 "T5 (10.8 GB) must be off the GPU while the DiT samples.",
        "dit_weights_ready": bool(dit_ok),
    }
    os.makedirs(args.outdir, exist_ok=True)
    p = os.path.join(args.outdir, "preflight.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=2, ensure_ascii=False)
    print(f"[preflight] wrote {p}")
    ok_all = all((s.get("ok") is not False) for s in rep["steps"])
    print(f"[preflight] {'PASS' if ok_all else 'FAIL'} "
          f"({sum(1 for s in rep['steps'] if s.get('ok'))}/{len(rep['steps'])} steps ran)")
    return 0 if ok_all else 1


def _build_chunk_plan(
    *,
    actions: str,
    frames: int,
    chunk_frames: int,
    prompt: str,
    distance: float,
    angular: float,
    rotation: float,
    omit_numeric_params: bool,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    if chunk_frames < 5 or (chunk_frames - 1) % 4 != 0:
        raise ValueError("--chunk_frames must be 4n+1 (e.g. 33, 37, 41)")
    if frames < 1:
        raise ValueError("--frames must be >= 1")
    acts, warns = parse_actions(actions, frames)
    n_chunks = max(1, math.ceil(frames / chunk_frames))
    chunks: List[Dict[str, Any]] = []
    for c in range(n_chunks):
        lo, hi = c * chunk_frames, min((c + 1) * chunk_frames, frames)
        keys, mouse, votes = actions_for_chunk(acts, lo, hi)
        cam = camera_sentence(keys, mouse)
        num = "" if omit_numeric_params else (
            build_numeric_sentence(distance, angular, rotation) + " "
        )
        # webapp_single_gpu prefixes "First-person perspective.<key>.<mouse> " to
        # whatever we pass as prompt, so the numeric sentence goes first here.
        full_prompt = f"{num}{prompt}".strip()
        chunks.append({
            "chunk": c, "frame_start": lo, "frame_end": hi - 1, "frames": hi - lo,
            "keys": keys, "mouse": mouse, "votes": votes,
            "camera_sentence": cam,
            "prompt_passed_to_model": full_prompt,
            "final_prompt": (cam + " " + full_prompt).strip(),
        })

    mapping = {
        "input": actions,
        "grammar": "<keys>|<mouse>:<frames>, comma separated; keys in "
                   "{W,A,S,D,W+A,W+D,S+A,S+D,None}; mouse in "
                   "{up,down,left,right,up+right,up+left,down+right,down+left,none}",
        "timeline": [a.as_dict() for a in acts],
        "warnings": warns,
        "key_vocab": KEY_VOCAB,
        "mouse_vocab": MOUSE_VOCAB,
        "mouse_aliases_ascii": {"up": "^ / u", "down": "v / d",
                                "left": "<- / l", "right": "-> / r",
                                "none": "- / dot (u00b7)"},
        "numeric_params": {
            "distance": distance, "angular": angular,
            "rotation": rotation,
            "sentence": build_numeric_sentence(distance, angular, rotation),
            "included": not omit_numeric_params,
        },
        "chunking": {
            "chunk_frames": chunk_frames,
            "chunks": n_chunks,
            "rule": "one discrete (keys,mouse) pair per chunk, chosen by majority "
                    "vote of the per-frame timeline; ties go to the earliest state",
        },
        "chunks": chunks,
    }
    return chunks, mapping


def _run_episode(
    W,
    torch_mod,
    *,
    image: str,
    prompt: str,
    actions: str,
    frames: int,
    chunk_frames: int,
    steps: int,
    seed: int,
    fps: int,
    shift: float,
    resolution: str,
    outdir: str,
    distance: float,
    angular: float,
    rotation: float,
    omit_numeric_params: bool,
    save_frames: bool,
    resume_from: Optional[str],
    t5_cpu: bool,
    load_s: float,
    load_peak_alloc: int,
    t_episode_start: float,
    server_boot_load_s: Optional[float] = None,
) -> Dict[str, Any]:
    if not image or not os.path.isfile(image):
        raise FileNotFoundError(f"input image not found: {image!r}")
    if resolution not in ("704x1280", "544x960"):
        raise ValueError(f"unsupported resolution {resolution!r}")

    os.makedirs(outdir, exist_ok=True)
    chunks, mapping = _build_chunk_plan(
        actions=actions,
        frames=frames,
        chunk_frames=chunk_frames,
        prompt=prompt,
        distance=distance,
        angular=angular,
        rotation=rotation,
        omit_numeric_params=omit_numeric_params,
    )
    n_chunks = len(chunks)
    W.OUTPUT_DIR = os.path.abspath(outdir)

    captured: List[Any] = []
    # intercept the repo's exporter so we keep the raw (C,F,H,W) tensor.
    # When `continue_from_last=True`, `webapp_single_gpu.long_generate` prepends
    # `model_input_de` (16 static head frames + the previous chunk) onto `video_tail`;
    # slice the trailing `chunk_frames` so only newly generated frames are appended.
    W._postprocess_video = make_capture_hook(captured, chunk_frames, resume_from=resume_from)

    # Clear previous episode's latent state first to prevent cross-episode leakage
    _reset_last_state(W, torch_mod)

    if resume_from:
        import numpy as _np

        with open(resume_from, "r", encoding="utf-8") as f:
            prev = json.load(f)
        state = prev.get("closed_loop_state") or {}
        npz = state.get("npz")
        if npz and os.path.isfile(npz):
            print(f"[yume_cli] resuming latent state from {npz}", flush=True)
            z = _np.load(npz)
            W.LAST["last_model_input_latent"] = torch_mod.from_numpy(
                z["latent"].astype("float32")
            ).to(W.MODELS.device)
            W.LAST["last_model_input_de"] = torch_mod.from_numpy(
                z["de"].astype("float32")
            ).to(W.MODELS.device)
            W.LAST["frame_total"] = int(z["frame_total"])
            W.LAST["last_prompt"] = state.get("prompt", "")
        else:
            print(
                f"[yume_cli][warn] {resume_from} has no closed_loop_state.npz; "
                f"starting a fresh clip",
                flush=True,
            )

    torch_mod.cuda.reset_peak_memory_stats()
    t_gen = time.perf_counter()
    chunk_times: List[float] = []
    for c, ch in enumerate(chunks):
        random.seed(seed + c)
        torch_mod.manual_seed(seed + c)
        g = W.LongGenArgs(
            prompt=ch["prompt_passed_to_model"],
            jpg_path=os.path.abspath(image),
            output_dir=os.path.abspath(outdir),
            fps=fps,
            sample_steps=steps,
            sample_num=1,
            frame_zero=chunk_frames,
            shift=shift,
            seed=seed + c,
            continue_from_last=bool(c > 0 or resume_from),
            refine_from_image=False,
            caption_path=None,
            mode="I2V",
            resolution=resolution,
            memory_optimization=True,      # required to stay under 16 GB
            vae_memory_optimization=True,   # tiled VAE decode: mandatory on 16 GB
            camera_movement1=ch["keys"],   # repo glyphs handled below
            camera_movement2=MOUSE_TO_REPO[ch["mouse"]],
        )
        t0 = time.perf_counter()
        _, final_prompt = W.long_generate(g)
        torch_mod.cuda.synchronize()
        dt = time.perf_counter() - t0
        chunk_times.append(dt)
        ch["final_prompt"] = final_prompt
        ch["seconds"] = round(dt, 2)
        print(
            f"[yume_cli] chunk {c + 1}/{n_chunks} -> {ch['frames']} frames in {dt:.1f}s "
            f"| keys={ch['keys']} mouse={ch['mouse']}",
            flush=True,
        )

    gen_s = time.perf_counter() - t_gen
    peak_alloc = torch_mod.cuda.max_memory_allocated()
    peak_resv = torch_mod.cuda.max_memory_reserved()

    if not captured:
        raise RuntimeError("no video tensor captured from long_generate (hook not called)")
    video = torch_mod.cat(captured, dim=1)  # (C, F, H, W) in [-1,1]
    total_gen = video.shape[1]
    if total_gen > frames:
        video = video[:, :frames]
    frames_n = video.shape[1]

    # ---- export ----
    from diffusers.utils import export_to_video
    from PIL import Image
    import numpy as _np

    v = video.clamp(-1, 1).add(1).div(2)
    v = (v * 255).byte().numpy()               # (C,F,H,W)
    v = _np.transpose(v, (1, 2, 3, 0))         # (F,H,W,C)
    pil = [Image.fromarray(f) for f in v]

    stem = f"yume_{int(time.time())}_s{seed}"
    mp4_path = os.path.join(outdir, stem + ".mp4")
    export_to_video(pil, mp4_path, fps=fps)
    frames_dir = None
    if save_frames:
        frames_dir = os.path.join(outdir, stem + "_frames")
        os.makedirs(frames_dir, exist_ok=True)
        for i, im in enumerate(pil):
            im.save(os.path.join(frames_dir, f"{i:04d}.png"))

    wall = time.perf_counter() - t_episode_start
    timing_dict: Dict[str, Any] = {
        "model_load_s": round(load_s, 2),
        "generation_s": round(gen_s, 2),
        "chunk_times_s": [round(x, 2) for x in chunk_times],
        "total_wall_s": round(wall, 2),
        "s_per_frame": round(gen_s / max(frames_n, 1), 3),
    }
    if server_boot_load_s is not None:
        timing_dict["server_boot_load_s"] = round(server_boot_load_s, 2)

    meta = {
        "tool": "yume_cli.py",
        "model": {
            "name": "Yume-5B-720P (Yume-1.5)",
            "checkpoint_dir": W.CKPT_DIR,
            "dtype": "bfloat16",
            "resolution": resolution,
            "t5_cpu": bool(t5_cpu),
        },
        "inputs": {
            "image": os.path.abspath(image),
            "prompt": prompt,
            "actions": actions,
            "seed": seed,
        },
        "action_interface": {
            "type": "discrete keyboard + mouse camera tokens injected into the text prompt",
            "keys": "W A S D (+ combos), one token per generated chunk",
            "mouse": "up/down/left/right/none (one discrete token per chunk)",
            "numeric_text_params": "Actual distance moved / Angular change rate / "
                                   "View rotation speed are appended as plain text",
            "frames_per_call": chunk_frames,
            "latent_frames_per_call": (chunk_frames - 1) // 4 + 1,
            "closed_loop": "yes: chunk N>0 uses continue_from_last=True, conditioning "
                           "on the 49-frame latent context produced by chunk N-1",
            "meta_latent_state": "written to meta.json as _latent_state (float16) so a "
                                 "later run can continue this exact world state",
        },
        "action_mapping": mapping,
        "generation": {
            "frames_requested": frames,
            "frames_generated": int(frames_n),
            "fps": fps,
            "duration_s": round(frames_n / fps, 3),
            "steps": steps,
            "shift": shift,
            "chunks": n_chunks,
            "chunk_frames": chunk_frames,
            "latent_frame_zero": (chunk_frames - 1) // 4 + 1,
        },
        "timing": timing_dict,
        "memory": {
            "peak_allocated_gb": gb(peak_alloc),
            "peak_reserved_gb": gb(peak_resv),
            "load_peak_allocated_gb": gb(load_peak_alloc),
            "device": torch_mod.cuda.get_device_name(0),
            "total_vram_gb": gb(torch_mod.cuda.get_device_properties(0).total_memory),
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch_mod.__version__,
            "cuda": torch_mod.version.cuda,
            "platform": sys.platform,
        },
        "outputs": {
            "video": mp4_path,
            "frames_dir": frames_dir,
            "meta": None,
        },
        "closed_loop_state": {
            "npz": None,
            "frame_total": int(W.LAST.get("frame_total") or 0),
            "prompt": W.LAST.get("last_prompt", ""),
        },
    }
    npz_path = os.path.join(outdir, stem + ".latent.npz")
    _np.savez_compressed(
        npz_path,
        latent=W.LAST["last_model_input_latent"].detach().to(torch_mod.float16).cpu().numpy(),
        de=W.LAST["last_model_input_de"].detach().to(torch_mod.float16).cpu().numpy(),
        frame_total=_np.int64(W.LAST.get("frame_total") or 0),
    )
    meta["closed_loop_state"]["npz"] = npz_path
    meta["outputs"]["latent_npz"] = npz_path
    meta_path = os.path.join(outdir, stem + ".meta.json")
    meta["outputs"]["meta"] = meta_path
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    alias_meta_path = os.path.join(outdir, "meta.json")
    with open(alias_meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print(f"[yume_cli] video: {mp4_path}", flush=True)
    print(f"[yume_cli] meta : {meta_path}", flush=True)
    print(
        f"[yume_cli] peak VRAM {gb(peak_resv):.2f} GB reserved / "
        f"{gb(peak_alloc):.2f} GB allocated; "
        f"gen {gen_s:.1f}s total wall {wall:.1f}s",
        flush=True,
    )
    return {
        "meta_path": alias_meta_path,
        "video_path": mp4_path,
        "frames_dir": frames_dir,
        "frames_n": int(frames_n),
        "generation_s": round(gen_s, 2),
        "chunk_times_s": [round(x, 2) for x in chunk_times],
        "peak_allocated_gb": gb(peak_alloc),
        "peak_reserved_gb": gb(peak_resv),
    }


def main(argv: Optional[List[str]] = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass

    ap = argparse.ArgumentParser(
        description="Yume-5B (Yume-1.5) 720P programmatic inference",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--image", required=False, help="input image (1280x704 or any aspect)")
    ap.add_argument("--prompt", default="", help="scene / event description")
    ap.add_argument("--actions", default="None|none",
                    help='discrete camera timeline, e.g. "W:10,A:10" or "W+A|right:33"')
    ap.add_argument("--frames", type=int, default=33, help="frames to generate (16 fps)")
    ap.add_argument("--steps", type=int, default=6, help="diffusion sampling steps")
    ap.add_argument("--outdir", required=False, default="./outputs_cli")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--chunk_frames", type=int, default=33,
                    help="frames per autoregressive chunk (must be 4n+1)")
    ap.add_argument("--resolution", default="704x1280", choices=["704x1280", "544x960"])
    ap.add_argument("--shift", type=float, default=5.0)
    ap.add_argument("--distance", type=float, default=4.0,
                    help="'Actual distance moved' text parameter")
    ap.add_argument("--angular", type=float, default=0.0,
                    help="'Angular change rate' text parameter")
    ap.add_argument("--rotation", type=float, default=0.0,
                    help="'View rotation speed' text parameter")
    ap.add_argument("--omit_numeric_params", action="store_true",
                    help="match webapp_single_gpu.py exactly (no distance/angular/rotation sentence)")
    ap.add_argument("--ckpt_dir", default=DEFAULT_CKPT)
    ap.add_argument("--save_frames", dest="save_frames", action="store_true", default=True,
                    help="dump the mp4 as a PNG frame sequence too (default: on)")
    ap.add_argument("--no_frames", dest="save_frames", action="store_false",
                    help="only write the mp4")
    ap.add_argument("--dry_run", action="store_true",
                    help="only resolve actions/prompts, do not load the model")
    ap.add_argument("--preflight", action="store_true",
                    help="validate env + VAE + T5 without the 19.8 GB DiT, then exit")
    ap.add_argument("--resume_from", default=None,
                    help="meta.json of a previous run: continue that latent state (closed loop)")
    ap.add_argument("--serve", action="store_true",
                    help="keep model loaded and serve JSON requests over stdin")
    ap.add_argument("--t5_cpu", action="store_true",
                    help="keep the 10.6 GB T5 text encoder permanently on CPU across chunks")
    args = ap.parse_args(argv)

    t_start = time.perf_counter()
    if args.chunk_frames < 5 or (args.chunk_frames - 1) % 4 != 0:
        ap.error("--chunk_frames must be 4n+1 (e.g. 33, 37, 41)")
    if args.frames < 1:
        ap.error("--frames must be >= 1")
    if not (args.dry_run or args.preflight or args.serve) and not args.image:
        ap.error("--image is required unless --dry_run, --preflight, or --serve")

    if args.dry_run:
        os.makedirs(args.outdir, exist_ok=True)
        _, mapping = _build_chunk_plan(
            actions=args.actions,
            frames=args.frames,
            chunk_frames=args.chunk_frames,
            prompt=args.prompt,
            distance=args.distance,
            angular=args.angular,
            rotation=args.rotation,
            omit_numeric_params=args.omit_numeric_params,
        )
        out = {
            "dry_run": True,
            "args": vars(args),
            "action_mapping": mapping,
            "outputs": {},
        }
        p = os.path.join(args.outdir, "meta_dryrun.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(json.dumps(mapping, indent=2, ensure_ascii=False))
        print(f"\n[dry-run] wrote {p}")
        return 0

    if args.preflight:
        if ROOT not in sys.path:
            sys.path.insert(0, ROOT)
        return preflight(args)

    # ---------------- load model once ----------------
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)

    import torch

    import import_shim  # noqa: F401  (puts wan23/... on sys.path)
    import webapp_single_gpu as W

    W.CKPT_DIR = os.path.abspath(args.ckpt_dir)
    if not os.path.isdir(W.CKPT_DIR):
        raise SystemExit(f"checkpoint dir not found: {W.CKPT_DIR}")
    os.makedirs(args.outdir, exist_ok=True)
    W.OUTPUT_DIR = os.path.abspath(args.outdir)
    patch_frugal_yume_init()

    print(f"[yume_cli] loading model from {W.CKPT_DIR} ...", flush=True)
    torch.cuda.init()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    t_load0 = time.perf_counter()
    msg = W.load_wan()
    if args.t5_cpu:
        pin_t5_to_cpu(W.MODELS.wan_i2v)
        print("[yume_cli] T5 encoder pinned to CPU (--t5_cpu)", flush=True)
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t_load0
    load_peak_alloc = torch.cuda.max_memory_allocated()
    print(f"[yume_cli] {msg}  ({load_s:.1f}s, load peak {gb(load_peak_alloc):.2f} GB)", flush=True)

    if args.serve:
        _emit_serve_json({
            "status": "ready",
            "model_load_s": round(load_s, 2),
            "load_peak_allocated_gb": gb(load_peak_alloc),
            "t5_cpu": bool(args.t5_cpu),
        })
        episodes_served = 0
        for raw_line in sys.stdin:
            line = raw_line.strip()
            if not line:
                continue
            try:
                req = json.loads(line)
                if not isinstance(req, dict):
                    raise ValueError("request must be a JSON object")
            except Exception as exc:
                _emit_serve_json({"status": "error", "error": f"invalid request JSON: {exc}"})
                continue

            cmd = req.get("cmd")
            if cmd == "quit":
                _emit_serve_json({"status": "bye"})
                break
            if cmd == "reset":
                _reset_last_state(W, torch)
                _emit_serve_json({"status": "ok"})
                continue
            if cmd == "episode":
                try:
                    t_ep = time.perf_counter()
                    ep_res = _run_episode(
                        W,
                        torch,
                        image=str(req.get("image") or ""),
                        prompt=str(req.get("prompt", args.prompt)),
                        actions=str(req.get("actions", args.actions)),
                        frames=int(req.get("frames", args.frames)),
                        chunk_frames=int(req.get("chunk_frames", args.chunk_frames)),
                        steps=int(req.get("steps", args.steps)),
                        seed=int(req.get("seed", args.seed)),
                        fps=int(req.get("fps", args.fps)),
                        shift=float(req.get("shift", args.shift)),
                        resolution=str(req.get("resolution", args.resolution)),
                        outdir=str(req.get("outdir", args.outdir)),
                        distance=float(req.get("distance", args.distance)),
                        angular=float(req.get("angular", args.angular)),
                        rotation=float(req.get("rotation", args.rotation)),
                        omit_numeric_params=bool(
                            req.get("omit_numeric_params", args.omit_numeric_params)
                        ),
                        save_frames=bool(req.get("save_frames", args.save_frames)),
                        resume_from=req.get("resume_from"),
                        t5_cpu=bool(args.t5_cpu),
                        load_s=load_s if episodes_served == 0 else 0.0,
                        load_peak_alloc=load_peak_alloc,
                        t_episode_start=t_ep,
                        server_boot_load_s=load_s,
                    )
                    episodes_served += 1
                    _reset_last_state(W, torch)
                    _emit_serve_json({"status": "ok", **ep_res})
                except Exception as exc:
                    _emit_serve_json({
                        "status": "error",
                        "error": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback.format_exc()[-2000:],
                    })
                continue
            _emit_serve_json({"status": "error", "error": f"unknown cmd {cmd!r}"})
        return 0

    # ---------------- single one-shot run ----------------
    _run_episode(
        W,
        torch,
        image=args.image,
        prompt=args.prompt,
        actions=args.actions,
        frames=args.frames,
        chunk_frames=args.chunk_frames,
        steps=args.steps,
        seed=args.seed,
        fps=args.fps,
        shift=args.shift,
        resolution=args.resolution,
        outdir=args.outdir,
        distance=args.distance,
        angular=args.angular,
        rotation=args.rotation,
        omit_numeric_params=args.omit_numeric_params,
        save_frames=args.save_frames,
        resume_from=args.resume_from,
        t5_cpu=bool(args.t5_cpu),
        load_s=load_s,
        load_peak_alloc=load_peak_alloc,
        t_episode_start=t_start,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(1)

