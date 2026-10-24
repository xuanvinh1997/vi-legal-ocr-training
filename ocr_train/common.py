"""Small helpers shared by the dataset builders."""
from __future__ import annotations

import hashlib
import multiprocessing as mp
import os
import random
from pathlib import Path
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")


def is_val(key: str, ratio: float) -> bool:
    """Stable split by a group key (a document id, never a single line), so a document is entirely train or val."""
    return int(hashlib.sha1(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < ratio


def rng_for(seed: int, *parts: object) -> random.Random:
    return random.Random(f"{seed}:" + ":".join(map(str, parts)))


def run_parallel(fn: Callable[[T], object], items: Iterable[T], workers: int) -> list:
    """`fn` must be a module-level function. Falls back to a plain loop for one worker (debuggable, test friendly)."""
    items = list(items)
    if workers <= 1 or len(items) <= 1:
        return [fn(i) for i in items]
    with mp.get_context("spawn" if os.name == "nt" else "fork").Pool(min(workers, len(items))) as pool:
        return pool.map(fn, items, chunksize=1)


def write_lines(path: str | Path, lines: Iterable[str]) -> int:
    n = 0
    with Path(path).open("w", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line + "\n")
            n += 1
    return n


def merge_label_files(root: str | Path, shards: list[str], name: str, out_name: str, shuffle_seed: int | None = 0,
                      repeat: dict[str, int] | None = None) -> int:
    """Concatenate `<root>/<shard>/<name>` into `<root>/<out_name>`. Label lines already hold paths relative to `root`.
    `repeat` lists a shard that many times, which is how scarce real data is up-weighted against abundant synthetic data."""
    root = Path(root)
    lines: list[str] = []
    for shard in shards:
        f = root / shard / name
        if f.exists():
            shard_lines = [l for l in f.read_text(encoding="utf-8").split("\n") if l]
            lines += shard_lines * max(1, (repeat or {}).get(shard, 1))
    if shuffle_seed is not None:
        random.Random(shuffle_seed).shuffle(lines)
    return write_lines(root / out_name, lines)
