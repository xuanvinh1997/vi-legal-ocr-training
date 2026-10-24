"""Legal text for the synthetic generators.

Source: the Hugging Face dataset `th1nhng0/vietnamese-legal-documents` (CC-BY-4.0, built from vbpl.vn), config
`content`, where each row is the HTML body of one legal document. We keep the structure the page composer needs
(paragraph kind, alignment, emphasis) and discard everything else."""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, asdict
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable, Iterator

from . import charset

DATASET = "th1nhng0/vietnamese-legal-documents"
_BLOCK_TAGS = {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "table", "br", "ul", "ol"}
_CENTER = re.compile(r"text-align\s*:\s*center", re.I)
_HEADING = re.compile(r"^(Điều|Chương|Mục|Phần|Phụ lục|Tiểu mục)\s+[0-9IVXLC]+", re.I)


@dataclass
class Block:
    kind: str  # "text" | "heading" | "title" | "table_row"
    text: str
    center: bool = False
    bold: bool = False
    cells: list[str] | None = None


class _Extractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[Block] = []
        self._buf: list[str] = []
        self._bold_chars = 0
        self._chars = 0
        self._center_stack: list[bool] = []
        self._bold_depth = 0
        self._center_now = False
        self._cells: list[str] | None = None
        self._depth = 0
        self._skip = 0

    def _flush(self) -> None:
        text = charset.normalize(" ".join("".join(self._buf).split()))
        if text:
            bold = self._chars > 0 and self._bold_chars / self._chars > 0.7
            self.blocks.append(Block("text", text, self._center_now, bold))
        self._buf, self._bold_chars, self._chars = [], 0, 0

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in ("script", "style", "head"):
            self._skip += 1
            return
        centered = a.get("align") == "center" or bool(_CENTER.search(a.get("style") or ""))
        if tag == "tr":
            self._depth += 1
            if self._depth == 1:
                self._flush()
                self._cells = []
        elif tag in ("td", "th"):
            if self._depth == 1:
                self._buf, self._bold_chars, self._chars = [], 0, 0
        elif tag in _BLOCK_TAGS:
            if self._depth:
                self._buf.append(" ")  # blocks inside a table cell stay in the cell
            else:
                self._flush()
        if tag in ("b", "strong"):
            self._bold_depth += 1
        if tag in _BLOCK_TAGS or tag in ("td", "th", "span", "font"):
            self._center_stack.append(centered)
            self._center_now = any(self._center_stack)

    def handle_endtag(self, tag):
        if tag in ("script", "style", "head"):
            self._skip = max(0, self._skip - 1)
            return
        if tag in ("b", "strong"):
            self._bold_depth = max(0, self._bold_depth - 1)
        if tag in ("td", "th"):
            if self._depth == 1 and self._cells is not None:
                self._cells.append(charset.normalize(" ".join("".join(self._buf).split())))
                self._buf, self._bold_chars, self._chars = [], 0, 0
        elif tag == "tr":
            self._depth = max(0, self._depth - 1)
            if self._depth == 0 and self._cells is not None:
                if any(self._cells):
                    self.blocks.append(Block("table_row", " | ".join(self._cells), cells=list(self._cells)))
                self._cells = None
        elif tag in _BLOCK_TAGS:
            if self._depth:
                self._buf.append(" ")
            else:
                self._flush()
        if (tag in _BLOCK_TAGS or tag in ("td", "th", "span", "font")) and self._center_stack:
            self._center_stack.pop()
            self._center_now = any(self._center_stack)

    def handle_data(self, data):
        if self._skip:
            return
        self._buf.append(data)
        n = len(data.strip())
        self._chars += n
        if self._bold_depth:
            self._bold_chars += n


def html_to_blocks(html: str) -> list[Block]:
    """Structured blocks of one document, already normalised and with table cells grouped per row."""
    parser = _Extractor()
    parser.feed(html)
    parser.close()
    parser._flush()
    blocks: list[Block] = []
    for block in parser.blocks:
        if block.kind == "table_row":
            blocks.append(block)
            continue
        if block.center and block.text.isupper() and len(block.text) < 200:
            block.kind = "title"
        elif _HEADING.match(block.text) and len(block.text) < 160:
            block.kind = "heading"
        blocks.append(block)
    return blocks


def clean_blocks(blocks: Iterable[Block], allowed: set[str], min_len: int = 4) -> list[Block]:
    """Keep blocks that are fully expressible in the charset and are really Vietnamese."""
    out: list[Block] = []
    for block in blocks:
        text = charset.restrict(block.text, allowed, 0.99)
        if text is None or len(text) < min_len or not charset.looks_vietnamese(text):
            continue
        if sum(c.isalnum() for c in text) < max(2, 0.4 * len(text)):  # rule lines such as "______" or "......"
            continue
        cells = None
        if block.cells is not None:
            cells = [charset.restrict(c, allowed, 0.99) or "" for c in block.cells]
        out.append(Block(block.kind, text, block.center, block.bold, cells))
    return out


def stream_documents(limit: int, seed: int = 0, dataset: str = DATASET, config: str = "content") -> Iterator[tuple[str, str]]:
    """(id, html) pairs streamed from the Hub, so only the first shards are downloaded."""
    from datasets import load_dataset

    stream = load_dataset(dataset, config, split="data", streaming=True).shuffle(seed=seed, buffer_size=2000)
    for i, row in enumerate(stream):
        if i >= limit:
            return
        yield str(row["id"]), row["content_html"]


def build_corpus(out_path: str | Path, documents: Iterable[tuple[str, str]], max_blocks: int = 400) -> dict[str, int]:
    allowed = set(charset.build_charset())
    stats = {"documents": 0, "blocks": 0}
    with Path(out_path).open("w", encoding="utf-8") as handle:
        for doc_id, html in documents:
            blocks = clean_blocks(html_to_blocks(html), allowed)[:max_blocks]
            if len(blocks) < 5:
                continue
            handle.write(json.dumps({"id": doc_id, "blocks": [asdict(b) for b in blocks]}, ensure_ascii=False) + "\n")
            stats["documents"] += 1
            stats["blocks"] += len(blocks)
    return stats


def load_corpus(path: str | Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sentences(corpus: list[dict], rng: random.Random, min_chars: int = 3, max_chars: int = 120) -> Iterator[str]:
    """Endless stream of text snippets cut at word boundaries, favouring short and medium lengths like real lines."""
    paragraphs = [b["text"] for doc in corpus for b in doc["blocks"] if b["kind"] in ("text", "heading", "title")]
    if not paragraphs:
        raise ValueError("empty corpus")
    while True:
        words = rng.choice(paragraphs).split(" ")
        if len(words) < 2:
            continue
        start = rng.randrange(len(words)) if rng.random() < 0.7 else 0
        # a quarter short (table cells, labels), the rest skewed towards full-width lines
        target = rng.randint(min_chars, min(30, max_chars)) if rng.random() < 0.25 else int(min_chars + (max_chars - min_chars) * rng.random() ** 0.6)
        out: list[str] = []
        for word in words[start:]:
            if len(" ".join(out + [word])) > target and out:
                break
            out.append(word)
        text = " ".join(out)
        if len(text) >= min_chars:
            yield text
