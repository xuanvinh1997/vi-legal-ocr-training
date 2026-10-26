"""Every page of a PDF through the paddle-onnx pipeline (orientation, layout, detection, recognition).

    python scripts/run_pdf.py <pdf> --models <models dir> --out <result dir> [--unclip 2.0] [--no-layout] [--pages 1-5]

Writes `<out>/document.md` (all pages), `<out>/pages/pNN.md`, `<out>/pages/pNN_layout.png` (the page with the layout blocks drawn
on it, upright) and `<out>/summary.json` (seconds, rotation, blocks, tables, mean confidence per page).
Needs the service's dependencies (onnxruntime, opencv, pyclipper; see services/ocr) and PyMuPDF."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from _common import DPI, load_method, render

COLORS = {"title": (0, 0, 220), "heading": (0, 120, 255), "text": (60, 160, 60), "list": (160, 160, 0), "table": (220, 0, 220),
          "figure": (200, 120, 0), "caption": (0, 170, 170), "footer": (120, 120, 120), "header": (120, 120, 120), "footnote": (120, 120, 120)}


def pages_in(spec: str | None, total: int) -> list[int]:
    """`3`, `2-5` or `1,4,7-9` (1-based); everything when empty."""
    if not spec:
        return list(range(1, total + 1))
    out: list[int] = []
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        out += range(int(lo), int(hi or lo) + 1)
    return [p for p in out if 1 <= p <= total]


def overlay(image, blocks: list[dict], kinds: dict) -> np.ndarray:
    """The upright page in BGR with each block tinted, outlined and numbered by its label."""
    vis = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    color = lambda b: COLORS.get(kinds.get(b["label"], "text"), COLORS["text"])
    tint = vis.copy()
    for b in blocks:
        x0, y0, x1, y1 = b["bbox"]
        cv2.rectangle(tint, (x0, y0), (x1, y1), color(b), -1)
    vis = cv2.addWeighted(tint, 0.18, vis, 0.82, 0)
    for n, b in enumerate(blocks, 1):
        x0, y0, x1, y1 = b["bbox"]
        cv2.rectangle(vis, (x0, y0), (x1, y1), color(b), 2)
        tag = f"{n} {b['label']}"
        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis, (x0, max(0, y0 - th - 8)), (x0 + tw + 8, max(th + 8, y0)), color(b), -1)
        cv2.putText(vis, tag, (x0 + 4, max(th + 2, y0 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return vis


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("pdf", type=Path)
    p.add_argument("--models", required=True, type=Path, help="folder with det/, rec/ and, for layout and rotation, layout/ and doc_ori/")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--unclip", type=float, help="DB unclip ratio (service default 1.5); 2.0 keeps the dot-below marks on tight scans")
    p.add_argument("--no-layout", action="store_true"), p.add_argument("--pages", help="e.g. 1-3,7 (default: all)")
    p.add_argument("--dpi", type=int, default=DPI)
    args = p.parse_args(argv)

    import pymupdf
    from ocr_service import layout as L, orient
    from ocr_service.methods import Request

    method = load_method(args.models, "off" if args.no_layout else "auto", args.unclip)
    kinds = {k: v for k, v in L.KINDS.items() if v}
    with pymupdf.open(args.pdf) as doc:
        total = len(doc)
    (args.out / "pages").mkdir(parents=True, exist_ok=True)
    summary, parts = [], []
    for n in pages_in(args.pages, total):
        image = render(args.pdf, n, args.dpi)
        start = time.time()
        result = method.read(Request(image=image))
        seconds = time.time() - start
        detail, blocks = result.detail, result.detail.get("blocks", [])
        (args.out / "pages" / f"p{n:02d}.md").write_text(result.text, encoding="utf-8")
        parts.append(f"<!-- page {n} -->\n\n{result.text}\n")
        upright, _ = orient.upright(image, method.settings.orient_path)
        cv2.imwrite(str(args.out / "pages" / f"p{n:02d}_layout.png"), overlay(upright, blocks, kinds))
        row = {"page": n, "sec": round(seconds, 1), "rotated": detail.get("rotated"), "boxes": detail.get("boxes"),
               "blocks": len(blocks), "tables": sum(kinds.get(b["label"]) == "table" for b in blocks),
               "chars": len(result.text), "confidence": detail.get("meanConfidence")}
        summary.append(row)
        print(json.dumps(row), flush=True)
    (args.out / "document.md").write_text("\n".join(parts), encoding="utf-8")
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"{args.pdf.name}: {len(summary)} pages, {sum(r['sec'] for r in summary):.0f} s -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
