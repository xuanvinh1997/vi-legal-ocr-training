"""Combine shards into the files PaddleOCR trains from, and catch data problems before an expensive run.

On-disk contract (everything under one data directory):
    rec/<shard>/{images/,train.txt,val.txt}  ->  rec/train_list.txt, rec/val_list.txt      `<shard>/images/x.jpg<TAB>text`
    det/<shard>/{images/,train.txt,val.txt}  ->  det/train_list.txt, det/val_list.txt      `<shard>/images/x.jpg<TAB>[json]`
    layout/{images/JPEGImages, annotations/instance_{train,val}.json}                       PaddleX COCO
Label paths are relative to rec/ or det/, which is what the config's `data_dir` points at.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

from PIL import Image

from . import charset
from .common import merge_label_files

REC_H = 48


def shards(root: str | Path, kind: str) -> list[str]:
    base = Path(root) / kind
    return sorted(p.name for p in base.iterdir() if p.is_dir() and ((p / "train.txt").exists() or (p / "val.txt").exists())) if base.exists() else []


def merge(root: str | Path, kind: str, only: list[str] | None = None, seed: int = 0, repeat: dict[str, int] | None = None) -> dict[str, int]:
    """Training lists may repeat a shard (`repeat={"pdf": 3}`); validation lists never do."""
    names = only or shards(root, kind)
    if not names:
        raise FileNotFoundError(f"no shards under {Path(root) / kind}")
    return {
        "train": merge_label_files(Path(root) / kind, names, "train.txt", "train_list.txt", seed, repeat),
        "val": merge_label_files(Path(root) / kind, names, "val.txt", "val_list.txt", None),
        "shards": len(names),
    }


def write_wh_list(root: str | Path, list_name: str = "train_list.txt") -> Path:
    """`<name>_wh.txt`: the same lines with image width and height appended. PaddleOCR's width-aware sampler (`ds_width`) needs
    them to put lines of similar width in one batch instead of padding every line to the widest."""
    base = Path(root) / "rec"
    out = base / list_name.replace(".txt", "_wh.txt")
    n = 0
    with out.open("w", encoding="utf-8") as handle:
        for line in (base / list_name).read_text(encoding="utf-8").split("\n"):
            if not line:
                continue
            rel, text = line.split("\t", 1)
            try:
                with Image.open(base / rel) as im:
                    width, height = im.size
            except OSError:
                continue
            handle.write(f"{rel}\t{text}\t{width}\t{height}\n")
            n += 1
    if not n:
        raise ValueError(f"no readable images listed in {base / list_name}")
    return out


def rec_stats(root: str | Path, sample: int = 20000, seed: int = 0) -> dict[str, float]:
    """Sizes that decide the recognition config: how wide a line is at height 48 and how many characters it has."""
    lines = [l for l in (Path(root) / "rec" / "train_list.txt").read_text(encoding="utf-8").split("\n") if l]
    random.Random(seed).shuffle(lines)
    widths, chars = [], []
    for line in lines[:sample]:
        rel, text = line.split("\t", 1)
        try:
            with Image.open(Path(root) / "rec" / rel) as im:
                widths.append(im.width * REC_H / im.height)
        except OSError:
            continue
        chars.append(len(text))
    if not widths:
        raise ValueError("no readable recognition images")
    widths.sort(), chars.sort()
    q = lambda xs, p: xs[min(len(xs) - 1, int(p * len(xs)))]
    p99_w, max_c = q(widths, 0.99), chars[-1]
    return {"lines": len(lines), "sampled": len(widths), "p50_width": q(widths, 0.5), "p99_width": p99_w, "max_chars": max_c,
            "p99_chars": q(chars, 0.99), "img_w": int(min(1600, max(320, math.ceil(p99_w / 32) * 32))), "max_text_length": max_c + 4}


def check_rec(root: str | Path, dict_path: str | Path, limit: int = 50000) -> dict[str, int]:
    """Every label char must be in the dictionary (else PaddleOCR drops the sample silently) and every image must open."""
    chars = set(charset.read_dict(dict_path)) | {" "}
    problems = {"missing_image": 0, "unknown_char": 0, "empty": 0, "checked": 0}
    unknown: dict[str, int] = {}
    for name in ("train_list.txt", "val_list.txt"):
        f = Path(root) / "rec" / name
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").split("\n")[:limit]:
            if not line:
                continue
            rel, text = line.split("\t", 1)
            problems["checked"] += 1
            if not text.strip():
                problems["empty"] += 1
            bad = set(text) - chars
            if bad:
                problems["unknown_char"] += 1
                for c in bad:
                    unknown[c] = unknown.get(c, 0) + 1
            if not (Path(root) / "rec" / rel).exists():
                problems["missing_image"] += 1
    problems["unknown_chars_seen"] = "".join(sorted(unknown)) or ""  # type: ignore[assignment]
    return problems


def check_det(root: str | Path, limit: int = 2000) -> dict[str, int]:
    out = {"checked": 0, "missing_image": 0, "bad_json": 0, "boxes": 0, "ignored": 0}
    for name in ("train_list.txt", "val_list.txt"):
        f = Path(root) / "det" / name
        if not f.exists():
            continue
        for line in f.read_text(encoding="utf-8").split("\n")[:limit]:
            if not line:
                continue
            rel, payload = line.split("\t", 1)
            out["checked"] += 1
            if not (Path(root) / "det" / rel).exists():
                out["missing_image"] += 1
            try:
                items = json.loads(payload)
                out["boxes"] += len(items)
                out["ignored"] += sum(i["transcription"] == "###" for i in items)
                assert all(len(i["points"]) >= 4 for i in items)
            except (ValueError, KeyError, AssertionError):
                out["bad_json"] += 1
    return out
