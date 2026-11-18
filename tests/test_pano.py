"""Geometry self-checks for the panorama renderer and the analytic flow."""
from __future__ import annotations

import math
import pathlib
import sys
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from worldloop.pano import PanoCamera, load_pano, render, rotation_flow, rotation_flow_forward, project_dir, pixel_to_lonlat  # noqa: E402


def check_warp_identity(pano_path: str) -> float:
    """frame1(p) == frame0(p + flow(p)) must hold for a rotating camera."""
    pano = load_pano(pano_path)
    cam0 = PanoCamera(yaw=0.3, pitch=0.05, fov=math.radians(70), width=640, height=360)
    cam1 = cam0.with_pose(yaw=0.55, pitch=0.02)
    f0 = render(pano, cam0).astype(np.float32)
    f1 = render(pano, cam1).astype(np.float32)
    flow = rotation_flow(cam0, cam1)
    h, w = f0.shape[:2]
    gx, gy = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (gx + flow[..., 0]).astype(np.float32)
    map_y = (gy + flow[..., 1]).astype(np.float32)
    valid = np.isfinite(flow).all(-1) & (map_x >= 0) & (map_x < w) & (map_y >= 0) & (map_y < h)
    warped = cv2.remap(f0, map_x, map_y, interpolation=cv2.INTER_LINEAR)
    err = np.abs(warped[valid] - f1[valid]).mean()
    return float(err)


def check_inverse_projection(pano_path: str) -> float:
    pano = load_pano(pano_path)
    cam = PanoCamera(yaw=1.2, pitch=0.1, fov=math.radians(70), width=640, height=360)
    errs = []
    for px, py in [(10, 10), (320, 180), (600, 40), (100, 340)]:
        lon, lat = pixel_to_lonlat(cam, px, py)
        rx, ry, vis = project_dir(cam, lon, lat)
        if vis:
            errs.append(math.hypot(rx - px, ry - py))
    return float(max(errs))


def check_direction(pano_path: str) -> float:
    """Panning the camera right must move image content LEFT (standard pinhole)."""
    pano = load_pano(pano_path)
    cam0 = PanoCamera(yaw=0.0, pitch=0.0, fov=math.radians(70), width=640, height=360)
    cam1 = cam0.with_pose(yaw=math.radians(7.5))
    flow = rotation_flow_forward(cam0, cam1)
    valid = np.isfinite(flow).all(-1)
    return float(np.mean(flow[..., 0][valid]))


def check_pole_unproject() -> None:
    """Looking straight up/down must not fail with math.asin domain error."""
    for pitch in (math.pi / 2, -math.pi / 2):
        cam = PanoCamera(yaw=0.4, pitch=pitch, fov=math.radians(70), width=64, height=64)
        lon, lat = pixel_to_lonlat(cam, 31.5, 31.5)
        assert math.isfinite(lon) and math.isfinite(lat)


class TestPano(unittest.TestCase):
    def test_pano_geometry(self) -> None:
        panos = sorted(pathlib.Path("data/panos").glob("*.jpg"))
        self.assertTrue(panos, "run scripts/fetch_panos.py first")
        p = str(panos[0])
        self.assertLess(check_warp_identity(p), 8.0)
        self.assertLess(check_inverse_projection(p), 0.5)
        self.assertLess(check_direction(p), -10.0)
        check_pole_unproject()


def main() -> None:
    panos = sorted(pathlib.Path("data/panos").glob("*.jpg"))
    assert panos, "run scripts/fetch_panos.py first"
    p = str(panos[0])
    err = check_warp_identity(p)
    print(f"warp identity mean abs error: {err:.3f} (0-255 scale, should be < 8)")
    inv = check_inverse_projection(p)
    print(f"projection round-trip max error: {inv:.3f} px (should be < 0.5)")
    dx = check_direction(p)
    print(f"mean dx for a right pan: {dx:.2f} px (must be < 0: content moves left)")
    assert err < 8, "analytic flow does not warp frame0 onto frame1"
    assert inv < 0.5, "project/unproject are not inverse"
    assert dx < -10, "right pan must move content left; the camera basis is mirrored"
    check_pole_unproject()
    print("OK")


if __name__ == "__main__":
    main()


