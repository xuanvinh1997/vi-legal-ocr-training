"""Real pages with exact labels: born-digital PDFs (official gazette, Word exports) carry a text layer, so every line's
text and position is known without any OCR. We render the page, tighten each box to the ink, then scan-degrade it.

Pages that are scans (a page-sized image, possibly with an old OCR text layer) are skipped and counted: their text layer
is somebody else's OCR output and must not be treated as ground truth. Pages whose text decodes to garbage (legacy
VNI/TCVN3 fonts) are skipped too.

Output shards: `<root>/det/pdf`, `<root>/rec/pdf` (same layout as the synthetic ones).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from . import charset, geometry as geo
from .common import is_val, rng_for, run_parallel, write_lines
from .synth_pages import finalize_labels

SCAN_COVERAGE = 0.8  # a page-sized picture covering this much of the page marks it as a scan


@dataclass
class PdfJob:
    pdfs: list[str]
    root: str
    seed: int
    val_ratio: float
    dpi: int
    level: float
    rec_per_page: int
    max_chars: int
    max_pages: int


def _is_scan(page) -> bool:
    area = page.rect.width * page.rect.height
    for info in page.get_images(full=True):
        for rect in page.get_image_rects(info[0]):
            if rect.width * rect.height >= SCAN_COVERAGE * area:
                return True
    return False


def page_segments(page, allowed: set[str], gap_factor: float = 2.5) -> tuple[list[tuple[str, tuple[float, float, float, float]]], int]:
    """(text, bbox in points) per horizontal run, plus how many runs were rejected. Runs separated by a wide gap (columns
    sharing a baseline, e.g. "Số: …" and the date) become separate lines, as a detector would see them."""
    out: list[tuple[str, tuple[float, float, float, float]]] = []
    rejected = 0
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block["lines"]:
            if abs(line["dir"][1]) > 0.01 or line["dir"][0] < 0:
                continue
            runs: list[list[dict]] = []
            for span in line["spans"]:
                if not span["text"].strip():
                    continue
                if runs and span["bbox"][0] - runs[-1][-1]["bbox"][2] > gap_factor * span["size"]:
                    runs.append([])
                if not runs:
                    runs.append([])
                runs[-1].append(span)
            for run in runs:
                raw = "".join(s["text"] for s in run)
                text = charset.restrict(charset.normalize(raw), allowed, 1.0)
                if text is None or not charset.looks_vietnamese(text):
                    rejected += 1
                    continue
                out.append((text, (min(s["bbox"][0] for s in run), min(s["bbox"][1] for s in run), max(s["bbox"][2] for s in run), max(s["bbox"][3] for s in run))))
    return out, rejected


def tighten(gray: np.ndarray, box: tuple[float, float, float, float], pad: int = 2) -> tuple[float, float, float, float] | None:
    """Shrink a font-metric box to the ink inside it; None when it holds no ink (invisible or white-on-white text)."""
    h, w = gray.shape
    x0, y0, x1, y1 = (int(max(0, box[0] - pad)), int(max(0, box[1] - pad)), int(min(w, box[2] + pad)), int(min(h, box[3] + pad)))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return None
    crop = gray[y0:y1, x0:x1].astype(np.float32)
    lo, hi = np.percentile(crop, 2), np.percentile(crop, 98)
    if hi - lo < 60:
        return None
    ink = crop < (lo + hi) / 2
    rows, cols = np.nonzero(ink.any(axis=1))[0], np.nonzero(ink.any(axis=0))[0]
    if len(rows) < 5 or len(cols) < 3:
        return None
    return x0 + cols[0] - pad, y0 + rows[0] - pad, x0 + cols[-1] + 1 + pad, y0 + rows[-1] + 1 + pad


def _process(job: PdfJob) -> dict:
    import fitz  # PyMuPDF

    allowed = set(charset.build_charset())
    root = Path(job.root)
    for kind in ("det", "rec"):
        (root / kind / "pdf" / "images").mkdir(parents=True, exist_ok=True)
    out = {"det": {"train": [], "val": []}, "rec": {"train": [], "val": []}, "pages": 0, "scan": 0, "bad_text": 0, "empty": 0}
    zoom = job.dpi / 72
    for pdf in job.pdfs:
        split = "val" if is_val(Path(pdf).name, job.val_ratio) else "train"
        rng = rng_for(job.seed, "pdf", Path(pdf).name)
        try:
            doc = fitz.open(pdf)
        except Exception:
            continue
        with doc:
            for number, page in enumerate(doc):
                if number >= job.max_pages:
                    break
                if _is_scan(page):
                    out["scan"] += 1
                    continue
                segments, rejected = page_segments(page, allowed)
                if rejected > 0.3 * (rejected + len(segments)):
                    out["bad_text"] += 1
                    continue
                if len(segments) < 3:
                    out["empty"] += 1
                    continue
                pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), colorspace=fitz.csRGB, alpha=False)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                gray = np.asarray(img.convert("L"))
                lines = []
                for text, box in segments:
                    tight = tighten(gray, tuple(v * zoom for v in box))
                    if tight:
                        lines.append((text, geo.box_to_quad(*tight)))
                if len(lines) < 3:
                    out["empty"] += 1
                    continue
                img, lines, _, _ = finalize_labels(img, lines, [], [], rng, job.level)
                stem = f"{Path(pdf).stem[:40]}_{number:04d}".replace(" ", "_")
                rel = f"pdf/images/{stem}.jpg"
                img.save(root / "det" / rel, quality=rng.randint(80, 95))
                det = [{"transcription": t, "points": [[round(x, 1), round(y, 1)] for x, y in q]} for t, q in lines]
                out["det"][split].append(f"{rel}\t{json.dumps(det, ensure_ascii=False)}")
                for k, (text, quad) in enumerate(rng.sample(lines, min(job.rec_per_page, len(lines)))):
                    if len(text) > job.max_chars:
                        continue
                    crop = geo.crop_quad(img, geo.clip_quad(geo.jitter_quad(quad, rng, 3), *img.size))
                    if crop is not None:
                        crel = f"pdf/images/{stem}_{k:02d}.jpg"
                        crop.save(root / "rec" / crel, quality=92)
                        out["rec"][split].append(f"{crel}\t{text}")
                out["pages"] += 1
    return out


def build(root: str | Path, pdf_dir: str | Path, *, workers: int = 2, seed: int = 0, val_ratio: float = 0.1, dpi: int = 150,
          level: float = 1.0, rec_per_page: int = 40, max_chars: int = 100, max_pages: int = 40) -> dict[str, int]:
    pdfs = sorted(str(p) for p in Path(pdf_dir).rglob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"no PDF files under {pdf_dir}")
    chunks = [pdfs[i::max(1, workers * 3)] for i in range(max(1, workers * 3))]
    jobs = [PdfJob(c, str(root), seed, val_ratio, dpi, level, rec_per_page, max_chars, max_pages) for c in chunks if c]
    results = run_parallel(_process, jobs, workers)
    stats = {k: sum(r[k] for r in results) for k in ("pages", "scan", "bad_text", "empty")}
    for kind in ("det", "rec"):
        for split in ("train", "val"):
            lines = [l for r in results for l in r[kind][split]]
            write_lines(Path(root) / kind / "pdf" / f"{split}.txt", lines)
            stats[f"{kind}_{split}"] = len(lines)
    return stats
