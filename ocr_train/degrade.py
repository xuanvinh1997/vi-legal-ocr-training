"""Scan / photocopy / phone-photo degradations on PIL images (NumPy + Pillow only, no OpenCV)."""
from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageFilter

_PAPER = [(255, 255, 255), (250, 248, 240), (245, 240, 225), (238, 238, 238), (252, 250, 245), (240, 236, 218)]


def _gradient(h: int, w: int, rng: random.Random, strength: float) -> np.ndarray:
    """Uneven illumination: a linear ramp plus a soft vignette, multiplicative around 1.0."""
    angle = rng.uniform(0, np.pi * 2)
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    ramp = ((xs / max(w, 1) - 0.5) * np.cos(angle) + (ys / max(h, 1) - 0.5) * np.sin(angle)) * rng.uniform(0.0, strength)
    cx, cy = rng.uniform(0.3, 0.7), rng.uniform(0.3, 0.7)
    vign = -(((xs / max(w, 1) - cx) ** 2 + (ys / max(h, 1) - cy) ** 2)) * rng.uniform(0.0, strength)
    return 1.0 + ramp + vign


def degrade(img: Image.Image, rng: random.Random, level: float = 1.0) -> Image.Image:
    """`level` scales how often and how hard each effect applies (0 = untouched, 1 = typical scan, 2 = poor photocopy)."""
    img = img.convert("RGB")
    w, h = img.size

    if rng.random() < 0.35 * level:  # low-resolution scan: lose detail, then upsample
        f = rng.uniform(0.45, 0.85)
        img = img.resize((max(8, int(w * f)), max(8, int(h * f))), Image.BILINEAR).resize((w, h), Image.BICUBIC)
    if rng.random() < 0.5 * level:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.5)))
    if rng.random() < 0.15 * level:  # ink spread (bolder) or fading (thinner)
        img = img.filter(ImageFilter.MinFilter(3) if rng.random() < 0.5 else ImageFilter.MaxFilter(3))

    arr = np.asarray(img, dtype=np.float32)
    tint = np.array(rng.choice(_PAPER), dtype=np.float32) / 255.0
    arr = arr * tint  # paper colour
    if rng.random() < 0.7 * level:
        arr = arr * _gradient(*arr.shape[:2], rng, 0.5 * level)[..., None]
    contrast = rng.uniform(0.75, 1.15) if rng.random() < 0.6 else 1.0
    arr = (arr - 128.0) * contrast + 128.0 + rng.uniform(-20, 15) * (rng.random() < 0.5)
    if rng.random() < 0.4 * level:
        arr = arr + np.random.default_rng(rng.getrandbits(32)).normal(0, rng.uniform(2, 12), arr.shape[:2])[..., None]
    if rng.random() < 0.12 * level:  # speckle
        gen = np.random.default_rng(rng.getrandbits(32))
        mask = gen.random(arr.shape[:2]) < rng.uniform(0.0005, 0.004)
        arr[mask] = gen.uniform(0, 120, (int(mask.sum()), 1))
    arr = np.clip(arr, 0, 255)
    if rng.random() < 0.08 * level:  # hard binarisation of a bad photocopy
        gray = arr.mean(axis=2, keepdims=True)
        arr = np.where(gray < rng.uniform(110, 170), 20.0, 250.0) * np.ones_like(arr)
    img = Image.fromarray(arr.astype(np.uint8), "RGB")

    if rng.random() < 0.7:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(30, 92))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return img
