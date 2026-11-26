"""Yume-5B adapter (keyboard-driven interactive world model).

Yume-1.5 exposes camera control as *text*: the camera sentence and three numeric
camera parameters are spliced into the prompt, e.g.

    "First-person perspective. The camera pans to the right (→).
     Actual distance moved:4 at 100 meters per second.
     Angular change rate (turn speed):3. View rotation speed:3."

Because the model is autoregressive over 33-frame chunks and loading 33 GB of
weights per chunk or per case would dominate the runtime, the adapter keeps a
persistent ``yume_cli.py --serve`` daemon alive across episodes (``_SERVER_CACHE``),
generates each episode in one session (``plan``), and serves the frames chunk by
chunk (``step``).
"""

from __future__ import annotations

import atexit
import collections
import json
import os
import pathlib
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time

import cv2
import numpy as np

from .base import ClosedLoopModel, StepRecord

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
SERVE_PREFIX = "__YUME_JSON__:"

# Yume-5B needs ~13.6 GB of weights plus activations; refusing to start below this
# keeps a leaked daemon from turning the next run into an access violation.  Set
# ``WORLDLOOP_MIN_FREE_VRAM_GB=0`` to disable the check (multi-GPU hosts, stubs).
MIN_FREE_VRAM_GB = 14.0


def _gpu_free_gb() -> float | None:
    """Free VRAM in GB, or ``None`` when it cannot be determined."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=20,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    best = None
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            total, used = float(parts[0]), float(parts[1])
        except ValueError:
            continue
        free = (total - used) / 1024.0
        best = free if best is None else max(best, free)
    return best


def _stray_serve_pids() -> list[int]:
    """PIDs of leftover ``yume_cli.py --serve`` daemons other than our parent chain."""
    try:
        import psutil  # local dev convenience; absent in a minimal install
    except ImportError:
        return []
    pids = []
    keep = {os.getpid(), os.getppid()}
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            pid = proc.info["pid"]
            cmd = proc.info.get("cmdline") or []
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
        if pid in keep:
            continue
        joined = " ".join(cmd)
        if "yume_cli.py" in joined and "--serve" in joined:
            pids.append(pid)
    return pids


def _preflight_gpu_free() -> None:
    """Fail with a useful message instead of crashing inside the model process.

    A ``yume_cli.py --serve`` daemon that outlived its run keeps ~10 GB resident.
    Starting a second one then dies with a Windows access violation, which names
    neither the daemon nor the memory pressure, so check before spawning.
    """
    try:
        floor = float(os.environ.get("WORLDLOOP_MIN_FREE_VRAM_GB", MIN_FREE_VRAM_GB))
    except ValueError:
        floor = MIN_FREE_VRAM_GB
    if floor <= 0.0:
        return
    free = _gpu_free_gb()
    if free is None or free >= floor:
        return
    strays = _stray_serve_pids()
    hint = ""
    if strays:
        listed = ", ".join(str(p) for p in strays)
        hint = (
            f" Leftover yume_cli --serve daemon(s) hold the memory: pid {listed}. "
            "Kill them first, e.g. in PowerShell:\n"
            "  Get-CimInstance Win32_Process -Filter \"name='python.exe'\" | "
            "Where-Object { $_.CommandLine -like '*yume_cli.py*--serve*' } | Stop-Process -Force"
        )
    else:
        hint = " Another process is using the GPU; free it before running Yume."
    raise RuntimeError(
        f"only {free:.1f} GB of GPU memory free, but Yume-5B needs about "
        f"{floor:.0f} GB.{hint}\n"
        "Set WORLDLOOP_MIN_FREE_VRAM_GB=0 to skip this check."
    )

# (keys, mouse): mouse pans the camera (yaw/pitch); W/A/S/D translate
ACTION_CLI = {
    "turn_right": ("None", "right"),
    "turn_left": ("None", "left"),
    "pitch_up": ("None", "up"),
    "pitch_down": ("None", "down"),
    "noop": ("None", "none"),
}


def _resolve_repo_path(p: str | pathlib.Path) -> pathlib.Path:
    cand = pathlib.Path(p)
    if cand.is_absolute() or cand.exists():
        return cand.resolve()
    return (_REPO_ROOT / cand).resolve()


def _extract_peak_vram_gb(episode_meta: dict) -> float | None:
    if not isinstance(episode_meta, dict):
        return None
    mem = episode_meta.get("memory")
    if isinstance(mem, dict):
        alloc = [
            float(mem[k])
            for k in ("peak_allocated_gb", "load_peak_allocated_gb")
            if isinstance(mem.get(k), (int, float))
        ]
        if alloc:
            return max(alloc)
        if isinstance(mem.get("peak_reserved_gb"), (int, float)):
            return float(mem["peak_reserved_gb"])
    if isinstance(episode_meta.get("peak_vram_gb"), (int, float)):
        return float(episode_meta["peak_vram_gb"])
    return None


class _YumeServer:
    """Persistent ``yume_cli.py --serve`` subprocess speaking line-delimited JSON."""

    def __init__(
        self,
        *,
        python: str,
        cli: pathlib.Path,
        weights: str,
        resolution: str,
        t5_cpu: bool,
        extra_args: list[str],
        startup_timeout: float = 1800.0,
    ) -> None:
        self.python = python
        self.cli = cli
        self.weights = weights
        self.resolution = resolution
        self.t5_cpu = bool(t5_cpu)
        self.extra_args = list(extra_args)
        self._lock = threading.Lock()
        self._queue: queue.Queue[tuple[str, str | None]] = queue.Queue()
        self._recent_logs: collections.deque[str] = collections.deque(maxlen=120)
        self.model_load_s: float | None = None
        self.load_peak_allocated_gb: float | None = None

        _preflight_gpu_free()

        cmd = [
            self.python,
            str(self.cli),
            "--serve",
            "--ckpt_dir",
            str(self.weights),
            "--resolution",
            self.resolution,
            *(["--t5_cpu"] if self.t5_cpu else []),
            *self.extra_args,
        ]
        env = dict(os.environ)
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(self.cli.parent),
            env=env,
            encoding="utf-8",
            errors="replace",
        )
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()

        try:
            ready = self._wait_response(startup_timeout)
        except Exception:
            self.shutdown()
            raise
        if ready.get("status") != "ready":
            self.shutdown()
            raise RuntimeError(f"yume_cli --serve failed to initialize: {ready!r}")
        if isinstance(ready.get("model_load_s"), (int, float)):
            self.model_load_s = float(ready["model_load_s"])
        if isinstance(ready.get("load_peak_allocated_gb"), (int, float)):
            self.load_peak_allocated_gb = float(ready["load_peak_allocated_gb"])

    def _read_stdout(self) -> None:
        assert self.proc.stdout is not None
        try:
            for line in self.proc.stdout:
                if line.startswith(SERVE_PREFIX):
                    self._queue.put(("json", line[len(SERVE_PREFIX) :].strip()))
                else:
                    stripped = line.rstrip("\r\n")
                    if stripped:
                        self._recent_logs.append(stripped)
        finally:
            self._queue.put(("eof", None))

    def is_alive(self) -> bool:
        return self.proc.poll() is None

    def _wait_response(self, timeout: float) -> dict:
        deadline = time.perf_counter() + max(float(timeout), 1.0)
        while True:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(f"yume_cli --serve timed out after {timeout}s")
            try:
                kind, payload = self._queue.get(timeout=remaining)
            except queue.Empty as exc:
                raise TimeoutError(f"yume_cli --serve timed out after {timeout}s") from exc
            if kind == "eof":
                tail = "\n".join(self._recent_logs)[-2000:]
                raise RuntimeError(
                    f"yume_cli --serve exited unexpectedly (code={self.proc.poll()}): {tail}"
                )
            if kind == "json" and payload is not None:
                return json.loads(payload)

    def request(self, req: dict, timeout: float) -> dict:
        with self._lock:
            if not self.is_alive() or self.proc.stdin is None:
                tail = "\n".join(self._recent_logs)[-2000:]
                raise RuntimeError(f"yume_cli --serve is not running: {tail}")
            self.proc.stdin.write(json.dumps(req, ensure_ascii=False) + "\n")
            self.proc.stdin.flush()
            resp = self._wait_response(timeout)
            if resp.get("status") == "error":
                tb = resp.get("traceback", "")
                detail = f"{resp.get('error')}\n{tb}".strip()
                raise RuntimeError(f"yume_cli --serve error: {detail}")
            return resp

    def shutdown(self) -> None:
        with self._lock:
            if self.proc.poll() is None:
                try:
                    if self.proc.stdin is not None:
                        self.proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                        self.proc.stdin.flush()
                    self.proc.wait(timeout=5.0)
                except Exception:  # noqa: BLE001
                    self.proc.kill()
                    try:
                        self.proc.wait(timeout=2.0)
                    except Exception:  # noqa: BLE001
                        pass
            for pipe in (self.proc.stdin, self.proc.stdout):
                if pipe is not None:
                    try:
                        pipe.close()
                    except Exception:  # noqa: BLE001
                        pass


class YumeModel(ClosedLoopModel):
    name = "yume_5b"
    native_fps = 16.0
    min_chunk_seconds = 2.0

    # Keyed by (python, cli, weights, resolution, t5_cpu[, extra_args])
    _SERVER_CACHE: dict[tuple, _YumeServer] = {}
    _CACHE_LOCK = threading.Lock()

    def __init__(
        self,
        cli: str | pathlib.Path = "third_party/YUME/yume_cli.py",
        python: str | pathlib.Path | None = None,
        weights: str | pathlib.Path | None = "models/Yume-5B-720P",
        steps: int = 6,
        distance: float = 4.0,
        angular: float = 3.0,
        rotation: float = 3.0,
        resolution: str = "704x1280",
        out_width: int = 1280,
        out_height: int = 720,
        chunk_frames: int = 33,
        timeout: int = 7200,
        extra_args: list[str] | None = None,
        use_server: bool = True,
        t5_cpu: bool = False,
        omit_numeric_params: bool = False,
    ):
        super().__init__()
        self.cli = _resolve_repo_path(cli)
        self.python = str(_resolve_repo_path(python)) if python else sys.executable
        self.weights = str(_resolve_repo_path(weights or "models/Yume-5B-720P"))
        self.steps = steps
        self.distance = distance
        self.angular = angular
        self.rotation = rotation
        self.resolution = resolution
        self.out_width = out_width
        self.out_height = out_height
        self.chunk_frames = chunk_frames
        self.timeout = timeout
        self.extra_args = list(extra_args) if extra_args else []
        self.use_server = bool(use_server)
        self.t5_cpu = bool(t5_cpu)
        self.omit_numeric_params = bool(omit_numeric_params)
        self._tmp: pathlib.Path | None = None
        self._last: np.ndarray | None = None
        self._prompt = ""
        self._seed = 0
        self._frames: list[np.ndarray] = []
        self._cursor = 0
        self.episode_meta: dict = {}

    def _server_key(self) -> tuple:
        base = (self.python, str(self.cli), str(self.weights), self.resolution, self.t5_cpu)
        return (*base, tuple(self.extra_args)) if self.extra_args else base

    def _get_server(self) -> _YumeServer:
        key = self._server_key()
        with self._CACHE_LOCK:
            srv = self._SERVER_CACHE.get(key)
            if srv is not None and srv.is_alive():
                return srv
            if srv is not None:
                srv.shutdown()
                self._SERVER_CACHE.pop(key, None)
            srv = _YumeServer(
                python=self.python,
                cli=self.cli,
                weights=self.weights,
                resolution=self.resolution,
                t5_cpu=self.t5_cpu,
                extra_args=self.extra_args,
                startup_timeout=self.timeout,
            )
            self._SERVER_CACHE[key] = srv
            return srv

    @classmethod
    def shutdown(cls) -> None:
        """Terminate all cached persistent ``yume_cli.py --serve`` daemons."""
        with cls._CACHE_LOCK:
            servers = list(cls._SERVER_CACHE.values())
            cls._SERVER_CACHE.clear()
        for srv in servers:
            srv.shutdown()

    @classmethod
    def unload(cls) -> None:
        """Alias for ``shutdown()`` matching ``WanT2VBaseline.unload()``."""
        cls.shutdown()

    # ------------------------------------------------------------------
    def reset(self, image: np.ndarray, prompt: str = "", seed: int = 0) -> None:
        self.records.clear()
        if self._tmp is not None:
            shutil.rmtree(self._tmp, ignore_errors=True)
        self._tmp = pathlib.Path(tempfile.mkdtemp(prefix="worldloop_yume_"))
        self._prompt = prompt or "First-person perspective of a realistic scene."
        self._seed = seed
        self._last = image.copy()
        self._frames = []
        self._cursor = 0
        self.episode_meta = {}

    # -- whole-episode generation ---------------------------------------
    def plan(self, segments: list, fps: float) -> None:
        """Generate the full action script in one model session."""
        if self._tmp is None or self._last is None:
            raise RuntimeError("reset() must be called before plan()")
        t0 = self._time()
        cond = self._tmp / "cond.jpg"
        cv2.imwrite(str(cond), self._last, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        counts, total = [], 0
        for seg in segments:
            sec = seg["seconds"] if isinstance(seg, dict) else seg.seconds
            act = seg["action"] if isinstance(seg, dict) else seg.action
            n = max(1, int(round(float(sec) * self.native_fps)))
            keys, mouse = ACTION_CLI[act]
            counts.append(f"{keys}|{mouse}:{n}")
            total += n

        outdir = self._tmp / "episode"
        outdir.mkdir(parents=True, exist_ok=True)
        server_load_s: float | None = None

        if self.use_server:
            srv = self._get_server()
            server_load_s = srv.model_load_s
            req = {
                "cmd": "episode",
                "image": str(cond),
                "prompt": self._prompt,
                "actions": ",".join(counts),
                "frames": int(total),
                "chunk_frames": int(self.chunk_frames),
                "steps": int(self.steps),
                "seed": int(self._seed),
                "resolution": self.resolution,
                "outdir": str(outdir),
                "distance": float(self.distance),
                "angular": float(self.angular),
                "rotation": float(self.rotation),
                "omit_numeric_params": bool(self.omit_numeric_params),
                "save_frames": True,
            }
            srv.request(req, timeout=self.timeout)
            stdout_tail = ""
        else:
            cmd = [
                self.python,
                str(self.cli),
                "--image",
                str(cond),
                "--prompt",
                self._prompt,
                "--actions",
                ",".join(counts),
                "--frames",
                str(total),
                "--chunk_frames",
                str(self.chunk_frames),
                "--steps",
                str(self.steps),
                "--seed",
                str(self._seed),
                "--ckpt_dir",
                str(self.weights),
                "--resolution",
                self.resolution,
                "--outdir",
                str(outdir),
                "--distance",
                str(self.distance),
                "--angular",
                str(self.angular),
                "--rotation",
                str(self.rotation),
                *(["--t5_cpu"] if self.t5_cpu else []),
                *(["--omit_numeric_params"] if self.omit_numeric_params else []),
                *self.extra_args,
            ]
            env = dict(os.environ)
            env.setdefault("PYTHONIOENCODING", "utf-8")
            env.setdefault("PYTHONUTF8", "1")
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=str(self.cli.parent),
                env=env,
                encoding="utf-8",
                errors="replace",
            )
            if proc.returncode != 0:
                tail = (proc.stderr or proc.stdout or "")[-2000:]
                raise RuntimeError(f"yume_cli failed ({proc.returncode}): {tail}")
            stdout_tail = proc.stdout[-800:]

        frames = self._read_frames(outdir)
        if not frames:
            raise RuntimeError(f"yume_cli produced no frames: {stdout_tail}")
        self._frames = [self._fit(f) for f in frames]
        self._cursor = 0
        meta = outdir / "meta.json"
        if meta.exists():
            try:
                self.episode_meta = json.loads(meta.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                self.episode_meta = {}

        peak_vram = _extract_peak_vram_gb(self.episode_meta)
        timing = self.episode_meta.get("timing") if isinstance(self.episode_meta, dict) else None
        chunk_times = (timing or {}).get("chunk_times_s") if isinstance(timing, dict) else None
        load_s = None
        if isinstance(timing, dict):
            if isinstance(timing.get("model_load_s"), (int, float)):
                load_s = float(timing["model_load_s"])
            elif isinstance(timing.get("server_boot_load_s"), (int, float)):
                load_s = float(timing["server_boot_load_s"])
        if load_s is None and server_load_s is not None:
            load_s = float(server_load_s)
        if isinstance(self.episode_meta, dict) and load_s is not None:
            self.episode_meta["model_load_s"] = load_s

        if isinstance(chunk_times, list) and chunk_times:
            chunk_dur = self.chunk_frames / self.native_fps
            chunks_meta = (
                (self.episode_meta.get("action_mapping") or {}).get("chunks")
                if isinstance(self.episode_meta, dict)
                else None
            ) or []
            for idx, dt in enumerate(chunk_times):
                n_cf = self.chunk_frames
                label = f"chunk:{idx}"
                if idx < len(chunks_meta) and isinstance(chunks_meta[idx], dict):
                    cm = chunks_meta[idx]
                    if isinstance(cm.get("frames"), int) and cm["frames"] > 0:
                        n_cf = int(cm["frames"])
                    if cm.get("keys") and cm.get("mouse"):
                        label = f"chunk:{idx}:{cm['keys']}|{cm['mouse']}"
                self.records.append(
                    StepRecord(
                        action=label,
                        seconds=chunk_dur,
                        n_frames=n_cf,
                        latency_s=float(dt),
                        peak_vram_gb=peak_vram,
                    )
                )
        else:
            self._record(
                f"plan:{len(counts)}seg",
                total / self.native_fps,
                len(self._frames),
                t0,
                peak_vram_gb=peak_vram,
            )
        vram_str = f", peak={peak_vram:.2f}GB" if peak_vram is not None else ""
        mode_str = "serve" if self.use_server else "oneshot"
        print(
            f"[yume:{mode_str}] episode: {len(self._frames)} frames in {self._time() - t0:.1f}s{vram_str}",
            flush=True,
        )

    def step(self, action: str, seconds: float) -> list[np.ndarray]:
        if not self._frames:
            raise RuntimeError("plan() must be called before step() for this adapter")
        n = max(1, int(round(seconds * self.native_fps)))
        out = self._frames[self._cursor : self._cursor + n]
        self._cursor += len(out)
        if not out:
            raise RuntimeError("episode exhausted")
        self._last = out[-1]
        return out

    def _fit(self, frame: np.ndarray) -> np.ndarray:
        if frame.shape[1] == self.out_width and frame.shape[0] == self.out_height:
            return frame
        return cv2.resize(frame, (self.out_width, self.out_height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _read_frames(d: pathlib.Path) -> list[np.ndarray]:
        cands = sorted(p for p in d.glob("*_frames") if p.is_dir())
        fdir = cands[0] if cands else d
        files = sorted(list(fdir.glob("*.png")) + list(fdir.glob("*.jpg")))
        imgs = [cv2.imread(str(f)) for f in files]
        imgs = [im for im in imgs if im is not None]
        if imgs:
            return imgs
        for mp4 in sorted(d.glob("*.mp4")):
            cap = cv2.VideoCapture(str(mp4))
            out = []
            while True:
                ok, fr = cap.read()
                if not ok:
                    break
                out.append(fr)
            cap.release()
            if out:
                return out
        return []

    def meta(self) -> dict:
        return {
            "steps": self.steps,
            "distance": self.distance,
            "angular": self.angular,
            "rotation": self.rotation,
            "omit_numeric_params": self.omit_numeric_params,
            "resolution": self.resolution,
            "chunk_frames": self.chunk_frames,
            "weights": self.weights,
            "use_server": self.use_server,
            "t5_cpu": self.t5_cpu,
            "model_load_s": self.episode_meta.get("model_load_s"),
            "episode": {k: v for k, v in self.episode_meta.items() if k != "_latent_state"},
        }

    def close(self) -> None:
        if self._tmp is not None:
            shutil.rmtree(self._tmp, ignore_errors=True)
            self._tmp = None


atexit.register(YumeModel.shutdown)
