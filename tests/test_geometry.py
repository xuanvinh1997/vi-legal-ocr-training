import random

import numpy as np
from PIL import Image, ImageDraw

from ocr_train import geometry as g


def test_rotate_points_matches_pillow():
    img = Image.new("L", (400, 300), 255)
    ImageDraw.Draw(img).rectangle([300, 40, 303, 43], fill=0)  # marker near the top right
    for angle in (-7.0, 3.5, 20.0):
        rot = img.rotate(angle, expand=True, fillcolor=255)
        ys, xs = np.nonzero(np.asarray(rot) < 128)
        got = (xs.mean(), ys.mean())
        (x, y), = g.rotate_points([(301.5, 41.5)], angle, img.size, rot.size)
        assert abs(got[0] - x) < 2.0 and abs(got[1] - y) < 2.0, (angle, got, (x, y))


def test_crop_quad_rectifies_rotated_text():
    img = Image.new("L", (400, 300), 255)
    d = ImageDraw.Draw(img)
    d.rectangle([100, 100, 250, 130], fill=0)
    rot = img.rotate(12, expand=True, fillcolor=255)
    quad = g.rotate_points(g.box_to_quad(100, 100, 251, 131), 12, img.size, rot.size)
    crop = g.crop_quad(rot.convert("RGB"), quad)
    arr = np.asarray(crop.convert("L"))
    assert abs(crop.width - 151) <= 2 and abs(crop.height - 31) <= 2
    assert (arr < 128).mean() > 0.9  # the crop is (almost) all ink: upright, no background wedges


def test_jitter_keeps_orientation():
    q = g.box_to_quad(10, 10, 100, 40)
    j = g.jitter_quad(q, random.Random(1), 5)
    x0, y0, x1, y1 = g.quad_bounds(j)
    assert 10 - 5 <= x0 <= 10 + 1.6 and 10 - 5 <= y0 <= 10 + 1.6
    assert 100 - 1.6 <= x1 <= 100 + 5 and 40 - 1.6 <= y1 <= 40 + 5
    assert g.poly_area(j) > 0
