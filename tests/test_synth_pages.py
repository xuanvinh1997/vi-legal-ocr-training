import json
import os

import pytest
from PIL import Image

from ocr_train import synth_pages

FONTS = os.environ.get("OCR_TEST_FONTS")
CORPUS = os.environ.get("OCR_TEST_CORPUS")
pytestmark = pytest.mark.skipif(not (FONTS and CORPUS), reason="set OCR_TEST_FONTS and OCR_TEST_CORPUS")


def test_build_pages(tmp_path):
    stats = synth_pages.build(tmp_path, CORPUS, FONTS, 6, workers=1, val_ratio=0.3, rec_per_page=5)
    assert stats["det_train"] + stats["det_val"] >= 5
    det_line = (tmp_path / "det/synth_pages/train.txt").read_text(encoding="utf-8").splitlines()[0]
    rel, payload = det_line.split("\t")
    items = json.loads(payload)
    img = Image.open(tmp_path / "det" / rel)
    assert len(items) >= 3
    for item in items:
        assert len(item["points"]) == 4 and item["transcription"]
        for x, y in item["points"]:
            assert -50 <= x <= img.width + 50 and -50 <= y <= img.height + 50
    coco = json.loads((tmp_path / "layout/annotations/instance_train.json").read_text())
    assert coco["categories"][0]["name"] == "_background_" and len(coco["categories"]) == len(synth_pages.CATEGORIES) + 1
    assert coco["annotations"] and all(a["category_id"] >= 1 for a in coco["annotations"])
    assert (tmp_path / "layout/images/JPEGImages" / coco["images"][0]["file_name"].split("/")[1]).exists()
    rec = (tmp_path / "rec/synth_pages/train.txt").read_text(encoding="utf-8").splitlines()
    assert rec and Image.open(tmp_path / "rec" / rec[0].split("\t")[0]).height > 8
