"""Text rendering primitives shared by the line and page generators."""
from __future__ import annotations

import random
from functools import lru_cache

from PIL import Image, ImageDraw, ImageFont

Box = tuple[int, int, int, int]


@lru_cache(maxsize=512)
def load_font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size)


def ink_box(font: ImageFont.FreeTypeFont, text: str) -> Box:
    """Glyph extents relative to the baseline origin (x right, y down): (left, top, right, bottom)."""
    l, t, r, b = font.getbbox(text, anchor="ls")
    return int(l), int(t), int(r + 0.999), int(b + 0.999)


def draw_run(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, font: ImageFont.FreeTypeFont, fill) -> None:
    """Draw with the baseline at xy[1]."""
    draw.text(xy, text, font=font, fill=fill, anchor="ls")


def random_ink(rng: random.Random) -> tuple[int, int, int]:
    g = rng.randint(0, 70)
    return (g, g, min(255, g + rng.randint(0, 25))) if rng.random() < 0.25 else (g, g, g)


def random_paper(rng: random.Random) -> tuple[int, int, int]:
    base = rng.randint(238, 255)
    return (base, base - rng.randint(0, 8), base - rng.randint(0, 20))


def render_line(text: str, font: ImageFont.FreeTypeFont, rng: random.Random, ink=None, paper=None) -> Image.Image:
    """One text line, cropped like a detector crop would be: tight to the ink plus a little random margin."""
    ink = ink or random_ink(rng)
    paper = paper or random_paper(rng)
    l, t, r, b = ink_box(font, text)
    pad_l, pad_r = rng.randint(2, 14), rng.randint(2, 14)
    pad_t, pad_b = rng.randint(2, 10), rng.randint(2, 10)
    img = Image.new("RGB", (r - l + pad_l + pad_r, b - t + pad_t + pad_b), paper)
    draw_run(ImageDraw.Draw(img), (pad_l - l, pad_t - t), text, font, ink)
    return img
