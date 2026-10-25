"""Synthetic recognition data: text lines in legal-document fonts, degraded like scans.

Output (a "shard" under the recognition root):
    <root>/synth_lines/images/<n>.jpg
    <root>/synth_lines/train.txt, val.txt     `synth_lines/images/<n>.jpg<TAB>text`   (paths relative to <root>)
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

from . import charset, corpus, legal_text
from .common import is_val, rng_for, run_parallel, write_lines
from .degrade import degrade
from .fonts import Family, ensure_fonts, pick
from .render import load_font, render_line

SHARD = "synth_lines"


@dataclass
class LineJob:
    index: int
    count: int
    root: str
    corpus_path: str
    fonts_dir: str
    extra_fonts: list[str]
    seed: int
    max_chars: int
    val_ratio: float
    level: float
    shard: str


def sample_text(snippets, rng: random.Random, max_chars: int) -> str:
    roll = rng.random()
    if roll < 0.15:
        text = legal_text.random_fragment(rng)
    elif roll < 0.22:  # label + value style fields
        text = f"{legal_text.random_fragment(rng)}; {next(snippets)}"
    else:
        text = next(snippets)
    if rng.random() < 0.10:
        text = text.upper()
    return text[:max_chars].rstrip() or "0"


def _work(job: LineJob) -> tuple[list[str], list[str]]:
    rng = rng_for(job.seed, "lines", job.index)
    families = ensure_fonts(job.fonts_dir, job.extra_fonts)
    docs = corpus.load_corpus(job.corpus_path)
    snippets = corpus.sentences(docs, rng, 3, job.max_chars)
    allowed = set(charset.build_charset())
    out_dir = Path(job.root) / job.shard / "images"
    out_dir.mkdir(parents=True, exist_ok=True)
    train, val = [], []
    for k in range(job.count):
        n = job.index * job.count + k
        text = charset.restrict(sample_text(snippets, rng, job.max_chars), allowed, 1.0)
        if not text:
            continue
        family: Family = pick(families, rng)
        bold = rng.random() < 0.15
        italic = rng.random() < 0.08
        font = load_font(str(family.path(bold, italic)), rng.randint(24, 56))
        img = degrade(render_line(text, font, rng), rng, job.level)
        rel = f"{job.shard}/images/{n:08d}.jpg"
        img.save(Path(job.root) / rel, quality=rng.randint(85, 95))
        (val if is_val(str(n), job.val_ratio) else train).append(f"{rel}\t{text}")
    return train, val


def build(root: str | Path, corpus_path: str | Path, fonts_dir: str | Path, count: int, *, workers: int = 2,
          seed: int = 0, max_chars: int = 100, val_ratio: float = 0.01, level: float = 1.0,
          extra_fonts: list[str] = (), shard: str = SHARD) -> dict[str, int]:
    """`shard` + a new `seed` add data next to an earlier run instead of overwriting it; `val_ratio=0` keeps the validation set unchanged."""
    ensure_fonts(fonts_dir, list(extra_fonts))  # download once, before the workers race for it
    shards = max(1, workers * 4)
    per = -(-count // shards)
    jobs = [LineJob(i, per, str(root), str(corpus_path), str(fonts_dir), list(extra_fonts), seed, max_chars, val_ratio, level, shard) for i in range(shards)]
    results = run_parallel(_work, jobs, workers)
    train = [l for t, _ in results for l in t]
    val = [l for _, v in results for l in v]
    base = Path(root) / shard
    base.mkdir(parents=True, exist_ok=True)
    write_lines(base / "train.txt", train)
    write_lines(base / "val.txt", val)
    return {"train": len(train), "val": len(val)}
