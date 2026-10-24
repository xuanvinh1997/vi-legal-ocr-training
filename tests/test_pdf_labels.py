import os

import pytest
from PIL import Image

from ocr_train import pdf_labels

FONTS = os.environ.get("OCR_TEST_FONTS")
pytestmark = pytest.mark.skipif(not FONTS, reason="set OCR_TEST_FONTS")


def _make_pdf(path, fonts_dir):
    import fitz

    doc = fitz.open()
    good = doc.new_page()
    font = os.path.join(fonts_dir, "Tinos-Regular.ttf")
    for i, text in enumerate(["CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "Điều 1. Phạm vi điều chỉnh của Quyết định này", "Người đứng đầu chịu trách nhiệm thi hành ngày 05 tháng 3 năm 2023"]):
        good.insert_text((72, 100 + 30 * i), text, fontsize=13, fontfile=font, fontname="tinos")
    bad = doc.new_page()  # a text layer that decodes to legacy-encoding soup
    for i in range(4):
        bad.insert_text((72, 100 + 30 * i), "C¨n cø LuËt Tæ chøc chÝnh quyÒn ®Þa ph­¬ng", fontsize=13)
    scan = doc.new_page()
    pix = Image.new("RGB", (600, 800), "white")
    pix.save(path.with_suffix(".png"))
    scan.insert_image(scan.rect, filename=str(path.with_suffix(".png")))
    scan.insert_text((72, 100), "Văn bản ocr cũ", fontsize=13, fontfile=font, fontname="tinos", render_mode=3)
    doc.save(path)


def test_pdf_pages_labelled_exactly(tmp_path):
    pdfs = tmp_path / "pdfs"
    pdfs.mkdir()
    _make_pdf(pdfs / "a.pdf", FONTS)
    stats = pdf_labels.build(tmp_path / "data", pdfs, workers=1, val_ratio=0.0, level=0.0, rec_per_page=10)
    assert stats["pages"] == 1 and stats["scan"] == 1 and stats["bad_text"] == 1, stats
    rec = (tmp_path / "data/rec/pdf/train.txt").read_text(encoding="utf-8").splitlines()
    texts = sorted(l.split("\t")[1] for l in rec)
    assert "Điều 1. Phạm vi điều chỉnh của Quyết định này" in texts and len(texts) == 3
    first = tmp_path / "data/rec" / rec[0].split("\t")[0]
    assert Image.open(first).height > 10
