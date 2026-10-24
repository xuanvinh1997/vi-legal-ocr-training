"""Vietnamese character set, text normalisation and a cheap "is this really Vietnamese" test.

Why this matters: a CTC recogniser can only emit characters that are in its dictionary. The dictionaries shipped with
PP-OCRv5-latin and PP-OCRv6 lack the Vietnamese letters that carry two marks (ạ ả ấ ầ ế ề ợ ữ ...), so no amount of
training data fixes them until the dictionary itself is extended. `missing_from` measures that gap.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path

VI_BASES = "aăâeêioôơuưy"
VI_TONES = "̣̀́̉̃"  # huyền sắc hỏi ngã nặng
_TONE_SET = set(VI_TONES)

# Typographic characters that occur in Vietnamese legal text. Anything outside the charset is dropped, not guessed.
EXTRA = "–—‘’“”…•·°²³¹½¼¾§№€₫£¥±×÷≥≤≠→←‰′″«»µ"

_CHAR_MAP = {
    " ": " ", " ": " ", " ": " ", " ": " ", "\t": " ",
    "‐": "-", "‑": "-", "‒": "-", "−": "-",
    "­": "", "​": "", "‌": "", "‍": "", "﻿": "",
    "ʼ": "’", "`": "'", "´": "'",
}
_TRANS = str.maketrans(_CHAR_MAP)


def vietnamese_letters() -> list[str]:
    """Every precomposed (NFC) Vietnamese letter, lower then upper case, plus đ/Đ."""
    out: list[str] = []
    for base in VI_BASES:
        out.append(base)
        out.extend(unicodedata.normalize("NFC", base + tone) for tone in VI_TONES)
    out.append("đ")
    out += [c.upper() for c in out]
    seen: set[str] = set()
    return [c for c in out if not (c in seen or seen.add(c))]


def _latin1_letters() -> list[str]:
    return [chr(c) for c in range(0xC0, 0x100) if chr(c) not in "×÷"]


def build_charset() -> list[str]:
    """The compact charset: ASCII, Vietnamese, Latin-1 letters (foreign names) and legal typography. No space; the
    PaddleOCR config appends it (`use_space_char`)."""
    ordered = [chr(c) for c in range(33, 127)] + vietnamese_letters() + _latin1_letters() + list(EXTRA)
    seen: set[str] = set()
    return [c for c in ordered if not (c in seen or seen.add(c))]


def read_dict(path: str | Path) -> list[str]:
    """Dictionary lines exactly as PaddleOCR reads them (one entry per line, in order)."""
    text = Path(path).read_text(encoding="utf-8")
    return [line.rstrip("\r") for line in text.split("\n") if line.rstrip("\r") != ""]


def write_dict(path: str | Path, chars: list[str]) -> None:
    if any(c == " " or len(c) != 1 for c in chars):
        raise ValueError("dictionary entries must be single non-space characters")
    if len(set(chars)) != len(chars):
        raise ValueError("duplicate dictionary entries")
    Path(path).write_text("\n".join(chars) + "\n", encoding="utf-8")


def missing_from(dictionary: list[str] | set[str], wanted: list[str] | None = None) -> list[str]:
    """Characters of `wanted` (default: all Vietnamese letters) that a dictionary cannot produce."""
    have = set(dictionary)
    return [c for c in (wanted if wanted is not None else vietnamese_letters()) if c not in have]


def extend_dict(upstream: list[str], wanted: list[str]) -> list[str]:
    """Upstream dictionary with the missing characters appended, so every existing class index is unchanged."""
    have = set(upstream)
    return list(upstream) + [c for c in wanted if c not in have and c != " "]


def normalize(text: str) -> str:
    """NFC, unify look-alike punctuation, drop zero-width characters and collapse whitespace."""
    text = unicodedata.normalize("NFC", text).translate(_TRANS)
    return re.sub(r" {2,}", " ", text).strip()


def coverage(text: str, charset: set[str]) -> float:
    letters = [c for c in text if c != " "]
    return 1.0 if not letters else sum(c in charset for c in letters) / len(letters)


def restrict(text: str, charset: set[str], min_coverage: float = 0.98) -> str | None:
    """Drop characters outside the charset; refuse the line when too much of it would be lost."""
    if coverage(text, charset) < min_coverage:
        return None
    return normalize("".join(c for c in text if c == " " or c in charset)) or None


_ONSET = r"(?:ngh|ng|nh|ph|th|tr|ch|kh|gh|gi|qu|[bcdđghklmnpqrstvx])?"
_SYLLABLE = re.compile(rf"^{_ONSET}[aăâeêioôơuưy]{{1,4}}(?:ng|nh|ch|c|m|n|p|t)?$")


def _strip_tone(word: str) -> str:
    return unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFD", word) if c not in _TONE_SET))


def is_syllable(word: str) -> bool:
    return bool(_SYLLABLE.match(_strip_tone(word.lower())))


def looks_vietnamese(text: str, min_ratio: float = 0.8) -> bool:
    """Guards against text layers in legacy encodings (VNI, TCVN3) that decode to consonant soup. Numbers, symbols and
    short upper-case abbreviations (UBND, NĐ-CP) are neutral; every other word must be a valid Vietnamese syllable."""
    good = bad = 0
    for token in re.split(r"[\s,.;:()\[\]“”\"'/\-–—]+", text):
        if not token or not any(c.isalpha() for c in token):
            continue
        if any(c.isdigit() for c in token) or (token.isupper() and len(token) <= 6) or token[0].isascii() and token.isupper():
            continue
        if is_syllable(token):
            good += 1
        else:
            bad += 1
    total = good + bad
    return total == 0 or good / total >= min_ratio
