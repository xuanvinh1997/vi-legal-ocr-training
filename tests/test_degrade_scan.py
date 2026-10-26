import random

import numpy as np
from PIL import Image, ImageDraw

from ocr_train.degrade import degrade_scan


def _line(w=400, h=30):
    img = Image.new("RGB", (w, h), (250, 248, 240))
    ImageDraw.Draw(img).text((6, 6), "Căn cứ Luật Tổ chức", fill=(20, 20, 20))
    return img


def test_degrade_scan_keeps_scan_resolution_and_adds_a_margin():
    for seed in range(20):
        out = degrade_scan(_line(), random.Random(seed), 1.0)
        assert out.mode == "RGB"
        # 30 px of line plus 2-10 px of margin (a slight skew can add a few more); never upsampled to the 48 px recognition height
        assert 31 <= out.height <= 48 and 380 <= out.width <= 420


def test_degrade_scan_is_reproducible_and_changes_the_pixels():
    a = degrade_scan(_line(), random.Random(3), 1.0)
    b = degrade_scan(_line(), random.Random(3), 1.0)
    c = degrade_scan(_line(), random.Random(4), 1.0)
    assert np.array_equal(np.asarray(a), np.asarray(b)) and not np.array_equal(np.asarray(a), np.asarray(c))


def test_degrade_scan_never_wipes_out_the_text():
    out = np.asarray(degrade_scan(_line(), random.Random(9), 2.0)).mean(axis=2)
    assert out.min() < 140 and out.max() > 200  # dark strokes on light paper are still there
