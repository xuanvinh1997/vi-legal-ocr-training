"""What run_pdf.py and eval_real.py share: loading the service's paddle-onnx method, rendering a PDF page.

The pipeline itself is `services/ocr` of the private-ai monorepo (this repo is nested in it as training/vi-legal-ocr-training); set
`OCR_SERVICE_DIR` when the two are somewhere else."""
from __future__ import annotations

import functools
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SERVICE = Path(os.environ.get("OCR_SERVICE_DIR") or REPO.parents[1] / "services" / "ocr")
if not (SERVICE / "ocr_service").is_dir():
    raise SystemExit(f"ocr_service not found in {SERVICE}: set OCR_SERVICE_DIR to the services/ocr folder of private-ai")
sys.path.insert(0, str(SERVICE))  # so `ocr_service` imports without installing the package

#: where the test PDFs live
TESTS_DIR = REPO / "tests"
DPI = 150  # the resolution the service gets scanned pages at; the page images of the real-page evaluation use it too


def load_method(models: Path, layout: str = "auto", unclip: float | None = None, device: str = "cpu"):
    """The paddle-onnx method on `models` (`det/`, `rec/`, `layout/`, `doc_ori/`). `unclip` widens (or narrows) the text boxes
    the detector finds; None keeps the service's default."""
    import ocr_service.methods.paddle_onnx as paddle_onnx
    from ocr_service.settings import Settings

    if unclip is not None:
        paddle_onnx.det_boxes = functools.partial(paddle_onnx.det_boxes, unclip_ratio=unclip)
    return paddle_onnx.PaddleOnnx(Settings(models_dir=models, device=device, layout=layout))


def render(pdf: Path, page: int, dpi: int = DPI):
    """Page `page` (1-based) of `pdf` as an RGB PIL image."""
    import pymupdf
    from PIL import Image

    with pymupdf.open(pdf) as doc:
        pix = doc[page - 1].get_pixmap(dpi=dpi)
    return Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
