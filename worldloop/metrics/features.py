"""Feature extractors and similarity metrics shared by all protocols."""

from __future__ import annotations

import functools

import cv2
import numpy as np
import torch


def _resolve_device(device: str | torch.device = "cuda") -> str:
    dev = str(device)
    if dev.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    return dev


@functools.lru_cache(maxsize=4)
def _dino(device: str = "cuda"):
    import timm

    dev = _resolve_device(device)
    model = timm.create_model(
        "vit_small_patch14_dinov2.lvd142m",
        pretrained=True,
        num_classes=0,
        dynamic_img_size=True,
    )
    model.eval().to(dev)
    return model


_DINO_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_DINO_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@torch.no_grad()
def dino_embed(images: list[np.ndarray], device: str = "cuda", img_size: int = 518) -> np.ndarray:
    """L2-normalised global DINOv2 descriptors for RGB (or BGR) uint8 images."""
    if not images:
        return np.zeros((0, 384), dtype=np.float32)
    dev = _resolve_device(device)
    model = _dino(dev)
    batch = []
    for im in images:
        if im.ndim == 2:
            im = np.stack([im] * 3, -1)
        rgb = cv2.cvtColor(im, cv2.COLOR_BGR2RGB) if im.shape[-1] == 3 else im
        rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        rgb = (rgb - _DINO_MEAN) / _DINO_STD
        batch.append(rgb.transpose(2, 0, 1))
    chunks = []
    for s in range(0, len(batch), 32):
        x = torch.from_numpy(np.stack(batch[s : s + 32])).to(dev)
        feats = model(x)
        feats = torch.nn.functional.normalize(feats.float(), dim=-1)
        chunks.append(feats.cpu().numpy())
    return np.concatenate(chunks, axis=0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    from skimage.metrics import structural_similarity

    if a.shape != b.shape:
        b = cv2.resize(b, (a.shape[1], a.shape[0]), interpolation=cv2.INTER_AREA)
    return float(structural_similarity(a, b, channel_axis=2, data_range=255))


@functools.lru_cache(maxsize=4)
def _lpips(device: str = "cuda"):
    import lpips

    dev = _resolve_device(device)
    model = lpips.LPIPS(net="alex").to(dev).eval()
    return model


@torch.no_grad()
def lpips_dist(a: np.ndarray, b: np.ndarray, device: str = "cuda") -> float:
    dev = _resolve_device(device)
    model = _lpips(dev)
    tw, th = max(64, b.shape[1]), max(64, b.shape[0])

    def prep(x):
        x = cv2.resize(x, (tw, th))
        t = torch.from_numpy(x[:, :, ::-1].copy()).float().permute(2, 0, 1) / 127.5 - 1.0
        return t.unsqueeze(0).to(dev)

    return float(model(prep(a), prep(b)).item())


def patch(img: np.ndarray, px: float, py: float, size: int) -> np.ndarray | None:
    h, w = img.shape[:2]
    x0, y0 = int(px - size / 2), int(py - size / 2)
    x1, y1 = x0 + size, y0 + size
    if x0 < 0 or y0 < 0 or x1 > w or y1 > h:
        return None
    return img[y0:y1, x0:x1]


@functools.lru_cache(maxsize=4)
def _raft(device: str = "cuda"):
    from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

    dev = _resolve_device(device)
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(dev).eval()
    return model


@functools.lru_cache(maxsize=1)
def _raft_tf():
    from torchvision.models.optical_flow import Raft_Large_Weights

    return Raft_Large_Weights.DEFAULT.transforms()


@torch.no_grad()
def raft_flow(prev: np.ndarray, nxt: np.ndarray, device: str = "cuda", max_side: int = 640) -> np.ndarray:
    """Dense flow (H, W, 2) from prev to nxt (pixel units), in the resolution of prev."""
    dev = _resolve_device(device)
    h0, w0 = prev.shape[:2]
    scale = min(1.0, max_side / max(h0, w0))
    hh = max(128, int(round(h0 * scale / 8) * 8))
    ww = max(128, int(round(w0 * scale / 8) * 8))
    a = cv2.resize(prev, (ww, hh), interpolation=cv2.INTER_AREA)
    b = cv2.resize(nxt, (ww, hh), interpolation=cv2.INTER_AREA)
    tf = _raft_tf()
    ta, tb = tf(
        torch.from_numpy(a[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0,
        torch.from_numpy(b[:, :, ::-1].copy()).permute(2, 0, 1).unsqueeze(0).float() / 255.0,
    )
    model = _raft(dev)
    flows = model(ta.to(dev), tb.to(dev))
    flow = flows[-1][0].permute(1, 2, 0).cpu().numpy()
    # rescale to original resolution
    flow = flow * np.array([w0 / ww, h0 / hh], dtype=np.float32)
    return cv2.resize(flow, (w0, h0), interpolation=cv2.INTER_LINEAR)


def flow_polar(flow: np.ndarray, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    mag = np.linalg.norm(flow, axis=-1)
    ang = np.arctan2(flow[..., 1], flow[..., 0])
    if mask is not None:
        mag = mag[mask]
        ang = ang[mask]
    return mag, ang


def circular_mean_deg(angles: np.ndarray, weights: np.ndarray | None = None) -> float:
    if weights is None:
        weights = np.ones_like(angles)
    s = float(np.sum(weights * np.sin(angles)))
    c = float(np.sum(weights * np.cos(angles)))
    return float(np.degrees(np.arctan2(s, c)))
