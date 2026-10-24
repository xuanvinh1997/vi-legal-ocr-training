"""Public data: what exists, under which licence, and loaders that bring it into the shard layout of `dataset.py`.

Licences below were read from each dataset card on 2026-10-24. Re-check before shipping a model: a licence on a mirror
does not override the licence of the data it was copied from.
"""
from __future__ import annotations

import time
import urllib.request
import urllib.robotparser
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from PIL import Image

from . import charset
from .common import is_val, write_lines


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    licence: str
    use: str
    note: str


SOURCES = [
    Source("th1nhng0/vietnamese-legal-documents", "https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents", "CC-BY-4.0",
           "text for synthetic pages and lines (`corpus`)", "170k documents as HTML in config `content`, collected from vbpl.vn. Attribution required."),
    Source("Born-digital PDFs from official gazettes (congbao.chinhphu.vn) and vbpl.vn", "https://vbpl.vn",
           "Legal normative documents are outside copyright protection (Luật Sở hữu trí tuệ, Điều 15); check the site terms for bulk download",
           "real pages with exact labels (`pdf-pages`)", "You choose the URLs (`download-pdfs`). Scanned PDFs and PDFs in legacy VNI/TCVN3 fonts are detected and skipped."),
    Source("niits/vietnamese-legal-ocr", "https://huggingface.co/datasets/niits/vietnamese-legal-ocr", "MIT",
           "9.6k synthetic legal text lines (images + labels.txt); mix in as a recognition shard (`import-rec`)", "Synthetic, so it adds little beyond `synth-lines`; useful as an independent sanity check."),
    Source("iAmHieu2012/vietnamese-ocr-dataset-aggregated", "https://huggingface.co/datasets/iAmHieu2012/vietnamese-ocr-dataset-aggregated", "MIT",
           "columns image,label; 3.5k rows (`import-hf-rec`)", "Aggregated from unnamed sources; inspect samples and label noise before mixing."),
    Source("SEACrowd/vintext (VinText scene text)", "https://huggingface.co/datasets/SEACrowd/vintext", "AGPL-3.0 on the loader; original data is research-oriented",
           "robustness of detection on photos, NOT legal-domain", "Not wired in: licence is restrictive. Use only after reading the VinAI terms."),
]


def print_sources() -> str:
    return "\n".join(f"- {s.name}\n    licence: {s.licence}\n    use: {s.use}\n    {s.note}\n    {s.url}" for s in SOURCES)


def _save_rec_shard(root: Path, name: str, items: Iterable[tuple[str, Image.Image, str]], val_ratio: float, max_chars: int, min_ratio: float) -> dict[str, int]:
    allowed = set(charset.build_charset())
    out_dir = root / "rec" / name / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    train, val, skipped = [], [], 0
    for key, image, raw in items:
        text = charset.restrict(charset.normalize(raw), allowed, 0.98)
        if not text or len(text) > max_chars or "\n" in raw or image.height < 8 or image.width / image.height < min_ratio:
            skipped += 1
            continue
        rel = f"{name}/images/{len(train) + len(val):08d}.jpg"
        image.convert("RGB").save(root / "rec" / rel, quality=93)
        (val if is_val(key, val_ratio) else train).append(f"{rel}\t{text}")
    write_lines(root / "rec" / name / "train.txt", train)
    write_lines(root / "rec" / name / "val.txt", val)
    return {"train": len(train), "val": len(val), "skipped": skipped}


def import_rec_tsv(root: str | Path, name: str, labels: str | Path, images_dir: str | Path | None = None, *, val_ratio: float = 0.02,
                   max_chars: int = 100, min_ratio: float = 1.0) -> dict[str, int]:
    """A folder of line images plus a `image<TAB>text` file (e.g. niits/vietnamese-legal-ocr's labels.txt)."""
    labels = Path(labels)
    base = Path(images_dir) if images_dir else labels.parent

    def items():
        for line in labels.read_text(encoding="utf-8").splitlines():
            if "\t" not in line:
                continue
            rel, text = line.split("\t", 1)
            try:
                with Image.open(base / rel) as im:
                    yield rel, im.copy(), text
            except OSError:
                continue

    return _save_rec_shard(Path(root), name, items(), val_ratio, max_chars, min_ratio)


def import_hf_rec(root: str | Path, name: str, dataset: str, *, image_col: str = "image", text_col: str = "label", split: str = "train",
                  config: str | None = None, limit: int = 20000, val_ratio: float = 0.02, max_chars: int = 100, min_ratio: float = 1.5) -> dict[str, int]:
    from datasets import load_dataset

    stream = load_dataset(dataset, config, split=split, streaming=True)

    def items():
        for i, row in enumerate(stream):
            if i >= limit:
                return
            yield f"{dataset}:{i}", row[image_col], str(row[text_col])

    return _save_rec_shard(Path(root), name, items(), val_ratio, max_chars, min_ratio)


def download_pdfs(urls_file: str | Path, out_dir: str | Path, *, delay: float = 2.0, respect_robots: bool = True, user_agent: str = "private-ai-ocr-training/0.1") -> dict[str, int]:
    """Polite, sequential fetch of PDFs you listed (one URL per line, `#` comments). One request per `delay` seconds."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    robots: dict[str, urllib.robotparser.RobotFileParser | None] = {}
    stats = {"downloaded": 0, "skipped": 0, "failed": 0, "blocked": 0}
    for raw in Path(urls_file).read_text(encoding="utf-8").splitlines():
        url = raw.split("#", 1)[0].strip()
        if not url:
            continue
        parts = urlparse(url)
        if parts.scheme not in ("http", "https"):
            stats["failed"] += 1
            continue
        target = out / (Path(parts.path).name or f"doc_{stats['downloaded']}.pdf")
        if target.exists():
            stats["skipped"] += 1
            continue
        if respect_robots:
            host = f"{parts.scheme}://{parts.netloc}"
            if host not in robots:
                parser = urllib.robotparser.RobotFileParser(f"{host}/robots.txt")
                try:
                    parser.read()
                    robots[host] = parser
                except OSError:
                    robots[host] = None
            parser = robots[host]
            if parser is not None and not parser.can_fetch(user_agent, url):
                stats["blocked"] += 1
                continue
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": user_agent}), timeout=60) as response:
                data = response.read()
        except OSError:
            stats["failed"] += 1
            continue
        if not data.startswith(b"%PDF"):
            stats["failed"] += 1
            continue
        target.write_bytes(data)
        stats["downloaded"] += 1
        time.sleep(delay)
    return stats
