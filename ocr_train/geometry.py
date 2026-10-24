"""Quadrilateral geometry: rotating labels together with the page, and rectifying a text quad like the OCR pipeline."""
from __future__ import annotations

import math

import numpy as np
from PIL import Image

Point = tuple[float, float]
Quad = list[Point]  # clockwise from top-left


def box_to_quad(x0: float, y0: float, x1: float, y1: float) -> Quad:
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def quad_bounds(quad: Quad) -> tuple[float, float, float, float]:
    xs, ys = zip(*quad)
    return min(xs), min(ys), max(xs), max(ys)


def rotate_points(points: Quad, angle_deg: float, old_size: tuple[int, int], new_size: tuple[int, int]) -> Quad:
    """Where points land after `Image.rotate(angle_deg, expand=True)` (counter-clockwise, about the image centre)."""
    t = math.radians(angle_deg)
    c, s = math.cos(t), math.sin(t)
    ox, oy = old_size[0] / 2, old_size[1] / 2
    nx, ny = new_size[0] / 2, new_size[1] / 2
    return [((x - ox) * c + (y - oy) * s + nx, -(x - ox) * s + (y - oy) * c + ny) for x, y in points]


def scale_points(points: Quad, sx: float, sy: float) -> Quad:
    return [(x * sx, y * sy) for x, y in points]


def jitter_quad(quad: Quad, rng, amount: float) -> Quad:
    """Grow the quad by a random margin per side, the way a detector's unclip step over- or under-shoots."""
    x0, y0, x1, y1 = quad_bounds(quad)
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    out = []
    for x, y in quad:
        out.append((x + math.copysign(rng.uniform(-0.3, 1.0) * amount, x - cx), y + math.copysign(rng.uniform(-0.3, 1.0) * amount, y - cy)))
    return out


def crop_quad(img: Image.Image, quad: Quad) -> Image.Image | None:
    """Warp the quad (TL, TR, BR, BL) to an upright rectangle, as PaddleOCR's `get_rotate_crop_image` does."""
    tl, tr, br, bl = quad
    w = int(round(max(math.dist(tl, tr), math.dist(bl, br))))
    h = int(round(max(math.dist(tl, bl), math.dist(tr, br))))
    if w < 4 or h < 4:
        return None
    # PIL's QUAD takes the source corners in the order NW, SW, SE, NE.
    data = [*tl, *bl, *br, *tr]
    return img.transform((w, h), Image.QUAD, data, resample=Image.BICUBIC, fillcolor=(255, 255, 255))


def clip_quad(quad: Quad, width: int, height: int) -> Quad:
    return [(min(max(x, 0), width - 1), min(max(y, 0), height - 1)) for x, y in quad]


def poly_area(quad: Quad) -> float:
    pts = np.asarray(quad)
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2)
