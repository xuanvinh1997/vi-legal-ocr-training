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


def degrade_scan(img: Image.Image, rng: random.Random, level: float = 1.0) -> Image.Image:
    """A line as a detector crop of a ~150 dpi scan. The caller has rendered it large and shrunk it to scan resolution, so the
    strokes are anti-aliased like a scanner's; this adds what such a crop has on top: the margin a detector box leaves, slivers of
    the neighbouring lines' descenders and ascenders at its top and bottom edges, a slight skew, a colour cast, blur, noise and
    JPEG compression at that resolution. The image keeps that small size (30 px high or so): recognition resizes it to 48 itself,
    as it does at inference, so a model trained on these sees the upsampled, soft tone marks it meets on real scans."""
    img = img.convert("RGB")
    arr0 = np.asarray(img)
    paper = np.median(arr0.reshape(-1, 3), axis=0)
    luma = arr0.mean(axis=2)
    ink = arr0[luma <= np.percentile(luma, 3)].mean(axis=0) if luma.size else np.zeros(3)

    pad_t, pad_b = rng.randint(1, 5), rng.randint(1, 5)
    h, w = arr0.shape[:2]
    canvas = np.empty((h + pad_t + pad_b, w, 3), dtype=np.float32)
    canvas[:] = paper
    canvas[pad_t:pad_t + h] = arr0
    gen = np.random.default_rng(rng.getrandbits(32))
    for edge, pad in (("top", pad_t), ("bottom", pad_b)):
        if rng.random() < 0.3 * min(level, 1.5):  # a few grey flecks, not a fringe
            cols = np.flatnonzero(gen.random(w) < rng.uniform(0.01, 0.05))
            for x in cols:
                length = rng.randint(1, min(pad + 1, 3))
                width = rng.randint(1, 2)
                rows = slice(0, length) if edge == "top" else slice(canvas.shape[0] - length, canvas.shape[0])
                canvas[rows, x:x + width] = ink + (paper - ink) * rng.uniform(0.3, 0.75)
    img = Image.fromarray(np.clip(canvas, 0, 255).astype(np.uint8), "RGB")

    if rng.random() < 0.3:  # a page is never perfectly straight in the scanner
        img = img.rotate(rng.uniform(-1.2, 1.2), resample=Image.BICUBIC, expand=True, fillcolor=tuple(int(v) for v in paper))
    if rng.random() < 0.7:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 0.9)))

    arr = np.asarray(img, dtype=np.float32)
    if rng.random() < 0.3 * level:  # the colour cast along one side of a scan (a blue-violet shade at the margin)
        ramp = np.clip((np.linspace(0, 1, arr.shape[1]) - rng.uniform(0.5, 0.9)) / 0.3, 0, 1) * rng.uniform(0.04, 0.14)
        ramp = ramp[::-1] if rng.random() < 0.5 else ramp
        arr = arr * np.stack([1 - ramp, 1 - ramp, np.ones_like(ramp)], axis=-1)[None]
    if rng.random() < 0.6 * level:
        arr = arr * _gradient(*arr.shape[:2], rng, 0.3 * level)[..., None]
    contrast = rng.uniform(0.8, 1.15)
    arr = (arr - 128.0) * contrast + 128.0 + rng.uniform(-15, 10)
    if rng.random() < 0.5 * level:
        arr = arr + np.random.default_rng(rng.getrandbits(32)).normal(0, rng.uniform(2, 9), arr.shape[:2])[..., None]
    img = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

    if rng.random() < 0.9:  # compression happens at scan resolution, before anything is enlarged
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=rng.randint(35, 88))
        img = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    return img
