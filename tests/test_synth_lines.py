import json
import os
from pathlib import Path

import pytest
from PIL import Image

from ocr_train import charset, synth_lines

FONTS = os.environ.get("OCR_TEST_FONTS")
pytestmark = pytest.mark.skipif(not FONTS, reason="set OCR_TEST_FONTS to a directory for the font cache (downloads once)")


def _corpus(tmp_path):
    blocks = [{"kind": "text", "text": t, "center": False, "bold": False, "cells": None} for t in [
        "Căn cứ Luật Tổ chức chính quyền địa phương ngày 19 tháng 6 năm 2015;",
        "Điều 1. Phạm vi điều chỉnh và đối tượng áp dụng của Quyết định này",
        "Ủy ban nhân dân tỉnh chịu trách nhiệm thi hành Quyết định này kể từ ngày ký",
        "Người có thẩm quyền xử phạt vi phạm hành chính được quy định tại khoản 2 Điều 5"]]
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps({"id": "1", "blocks": blocks}, ensure_ascii=False) + "\n", encoding="utf-8")
    return p


def test_build_lines(tmp_path):
    stats = synth_lines.build(tmp_path / "rec", _corpus(tmp_path), FONTS, 40, workers=1, val_ratio=0.2)
    assert stats["train"] + stats["val"] >= 30
    allowed = set(charset.build_charset()) | {" "}
    for line in (tmp_path / "rec/synth_lines/train.txt").read_text(encoding="utf-8").splitlines():
        rel, text = line.split("\t")
        assert set(text) <= allowed and text == charset.normalize(text)
        img = Image.open(tmp_path / "rec" / rel)
        assert img.height >= 12 and img.width >= 12
