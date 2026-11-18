"""Exact perspective rendering from equirectangular panoramas.

The whole benchmark rests on one property: for a *rotating* camera located at the
panorama's capture point, the perspective view is an exact function of
(yaw, pitch, fov).  That gives us, for free:

* photo-real first frames for any scene, and
* dense ground-truth optical flow between any two camera orientations,

with no learned components and no reference video.  Everything is deterministic
and reproducible from a CC0 panorama + a pose script.

Conventions
-----------
World frame: x right, y up, z forward (OpenGL-ish).
* ``lon``   azimuth, radians, measured from +z towards +x.
* ``lat``   elevation, radians, from the xz-plane towards +y.
* ``yaw``   == lon of the camera's optical axis.
* ``pitch`` == lat of the camera's optical axis.

Equirect panorama pixel ``(u, v)`` with ``u in [0, W)`` maps to
``lon = (u / W - 0.5) * 2*pi`` and ``lat = (0.5 - v / H) * pi``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PanoCamera:
    """A pinhole camera that rotates about the panorama centre."""

    yaw: float = 0.0  # radians
    pitch: float = 0.0  # radians
    fov: float = math.radians(70.0)  # VERTICAL field of view (the ray grid uses tan(fov/2) along y)
    width: int = 1280
    height: int = 720

    def with_pose(self, yaw: float | None = None, pitch: float | None = None) -> "PanoCamera":
        return PanoCamera(
            yaw=self.yaw if yaw is None else yaw,
            pitch=self.pitch if pitch is None else pitch,
            fov=self.fov,
            width=self.width,
            height=self.height,
        )


def _camera_basis(yaw: float, pitch: float):
    """Right-handed camera basis (x right, y up, z forward).

    ``r = up x f`` (not ``f x up``) is what makes the rendered image *not* mirrored:
    with the wrong order, image-right maps to world -x, the panorama comes out
    horizontally flipped, and a model that correctly pans right produces image
    motion opposite to the analytic ground truth.
    """
    f = np.array(
        [math.cos(pitch) * math.sin(yaw), math.sin(pitch), math.cos(pitch) * math.cos(yaw)],
        dtype=np.float64,
    )
    up_hint = np.array([0.0, 1.0, 0.0])
    r = np.cross(up_hint, f)
    n = np.linalg.norm(r)
    if n < 1e-9:  # looking straight up/down
        r = np.array([1.0, 0.0, 0.0])
    else:
        r = r / n
    u = np.cross(f, r)
    return f, r, u


def _ray_grid(cam: PanoCamera) -> np.ndarray:
    """Unit world directions for every output pixel, shape (H, W, 3)."""
    f, r, u = _camera_basis(cam.yaw, cam.pitch)
    aspect = cam.width / cam.height
    tan_half = math.tan(cam.fov / 2.0)
    xs = ((np.arange(cam.width, dtype=np.float64) + 0.5) / cam.width * 2.0 - 1.0) * tan_half * aspect
    ys = (1.0 - (np.arange(cam.height, dtype=np.float64) + 0.5) / cam.height * 2.0) * tan_half
    gx, gy = np.meshgrid(xs, ys)
    d = f[None, None, :] + gx[..., None] * r[None, None, :] + gy[..., None] * u[None, None, :]
    d /= np.linalg.norm(d, axis=-1, keepdims=True)
    return d


def dirs_to_pano_pixels(dirs: np.ndarray, width: int, height: int):
    """Map unit world directions to continuous equirect pixel coords (u, v)."""
    lon = np.arctan2(dirs[..., 0], dirs[..., 2])
    lat = np.arcsin(np.clip(dirs[..., 1], -1.0, 1.0))
    u = (lon / (2.0 * math.pi) + 0.5) * width - 0.5
    v = (0.5 - lat / math.pi) * height - 0.5
    return u, v


def render(pano: np.ndarray, cam: PanoCamera) -> np.ndarray:
    """Render a perspective view from an equirect panorama."""
    ph, pw = pano.shape[:2]
    dirs = _ray_grid(cam)
    u, v = dirs_to_pano_pixels(dirs, pw, ph)
    # wrap horizontally, clamp vertically
    map_x = (u.astype(np.float32) + 0.5) % pw - 0.5
    map_y = np.clip(v.astype(np.float32), 0.0, ph - 1.0)
    out = cv2.remap(
        pano,
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )
    return out


def rotation_flow(cam0: PanoCamera, cam1: PanoCamera) -> np.ndarray:
    """Dense backward flow (H, W, 2) such that frame1(p) == frame0(p + flow(p))."""
    if (cam0.width, cam0.height) != (cam1.width, cam1.height):
        raise ValueError("flow needs identical resolution")
    dirs1 = _ray_grid(cam1)
    # express dirs1 in cam0's basis
    f0, r0, u0 = _camera_basis(cam0.yaw, cam0.pitch)
    x = dirs1 @ r0
    y = dirs1 @ u0
    z = dirs1 @ f0
    tan_half = math.tan(cam0.fov / 2.0)
    aspect = cam0.width / cam0.height
    ndc_x = (x / z) / (tan_half * aspect)
    ndc_y = (y / z) / tan_half
    valid = z > 1e-6
    ndc_x = np.where(valid, ndc_x, np.nan)
    ndc_y = np.where(valid, ndc_y, np.nan)
    px = (ndc_x + 1.0) * 0.5 * cam0.width - 0.5
    py = (1.0 - ndc_y) * 0.5 * cam0.height - 0.5
    gx, gy = np.meshgrid(
        np.arange(cam1.width, dtype=np.float64), np.arange(cam1.height, dtype=np.float64)
    )
    flow = np.stack([px - gx, py - gy], axis=-1)
    flow[~valid] = np.nan
    return flow


def pixel_to_lonlat(cam: PanoCamera, px: float, py: float) -> tuple[float, float]:
    """Inverse projection: image pixel -> world direction (lon, lat)."""
    aspect = cam.width / cam.height
    tan_half = math.tan(cam.fov / 2.0)
    ndc_x = ((px + 0.5) / cam.width * 2.0 - 1.0) * tan_half * aspect
    ndc_y = (1.0 - (py + 0.5) / cam.height * 2.0) * tan_half
    f, r, u = _camera_basis(cam.yaw, cam.pitch)
    d = f + ndc_x * r + ndc_y * u
    d /= np.linalg.norm(d)
    return math.atan2(d[0], d[2]), math.asin(float(np.clip(d[1], -1.0, 1.0)))


def rotation_flow_forward(cam0: PanoCamera, cam1: PanoCamera) -> np.ndarray:
    """Forward flow on cam0's grid: flow(p0) = p1 - p0.

    This is the convention used by RAFT (and by every optical-flow benchmark), so
    predicted and analytic flow can be compared pixel by pixel.
    """
    if (cam0.width, cam0.height) != (cam1.width, cam1.height):
        raise ValueError("flow needs identical resolution")
    dirs0 = _ray_grid(cam0)
    f1, r1, u1 = _camera_basis(cam1.yaw, cam1.pitch)
    x = dirs0 @ r1
    y = dirs0 @ u1
    z = dirs0 @ f1
    tan_half = math.tan(cam1.fov / 2.0)
    aspect = cam1.width / cam1.height
    ndc_x = (x / z) / (tan_half * aspect)
    ndc_y = (y / z) / tan_half
    valid = z > 1e-6
    px = (np.where(valid, ndc_x, np.nan) + 1.0) * 0.5 * cam1.width - 0.5
    py = (1.0 - np.where(valid, ndc_y, np.nan)) * 0.5 * cam1.height - 0.5
    gx, gy = np.meshgrid(
        np.arange(cam0.width, dtype=np.float64), np.arange(cam0.height, dtype=np.float64)
    )
    flow = np.stack([px - gx, py - gy], axis=-1)
    flow[~valid] = np.nan
    return flow


def project_dir(cam: PanoCamera, lon: float, lat: float) -> tuple[float, float, bool]:
    """Pixel coords of a world direction in a camera, plus a visibility flag."""
    d = np.array([math.cos(lat) * math.sin(lon), math.sin(lat), math.cos(lat) * math.cos(lon)])
    f, r, u = _camera_basis(cam.yaw, cam.pitch)
    x, y, z = d @ r, d @ u, d @ f
    if z <= 1e-6:
        return float("nan"), float("nan"), False
    tan_half = math.tan(cam.fov / 2.0)
    aspect = cam.width / cam.height
    ndc_x = (x / z) / (tan_half * aspect)
    ndc_y = (y / z) / tan_half
    px = (ndc_x + 1.0) * 0.5 * cam.width - 0.5
    py = (1.0 - ndc_y) * 0.5 * cam.height - 0.5
    inside = 0 <= px < cam.width and 0 <= py < cam.height
    return px, py, bool(inside)


def load_pano(path: str) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def pano_feature_points(pano: np.ndarray, n: int = 200, seed: int = 0) -> np.ndarray:
    """Pick high-gradient panorama pixels; used to anchor revisit checks."""
    gray = cv2.cvtColor(pano, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.hypot(gx, gy)
    mag = cv2.GaussianBlur(mag, (0, 0), 8)
    flat = mag.reshape(-1)
    idx = np.argsort(flat)[-max(n * 20, 4000):]
    rng = np.random.default_rng(seed)
    idx = rng.choice(idx, size=min(n, len(idx)), replace=False)
    ys, xs = np.unravel_index(idx, mag.shape)
    ph, pw = pano.shape[:2]
    lon = (xs / pw - 0.5) * 2.0 * math.pi
    lat = (0.5 - ys / ph) * math.pi
    return np.stack([lon, lat], axis=-1)
