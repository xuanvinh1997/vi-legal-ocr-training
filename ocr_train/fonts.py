"""Fonts for synthetic pages. Only open-licence static fonts from google/fonts are fetched; a font is kept only if it
really covers every Vietnamese letter (a missing glyph renders as a box and would poison the labels)."""
from __future__ import annotations

import random
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .charset import vietnamese_letters

_BASE = "https://github.com/google/fonts/raw/main"
# family -> (weight of use, {style: path in google/fonts}). Tinos is metric-compatible with Times New Roman, which is
# what Vietnamese administrative documents are set in (Nghị định 30/2020/NĐ-CP).
FAMILIES: dict[str, tuple[float, dict[str, str]]] = {
    "Tinos": (6.0, {
        "regular": "ofl/tinos/Tinos-Regular.ttf", "bold": "ofl/tinos/Tinos-Bold.ttf",
        "italic": "ofl/tinos/Tinos-Italic.ttf", "bolditalic": "ofl/tinos/Tinos-BoldItalic.ttf"}),
    "BeVietnamPro": (1.0, {
        "regular": "ofl/bevietnampro/BeVietnamPro-Regular.ttf", "bold": "ofl/bevietnampro/BeVietnamPro-Bold.ttf",
        "italic": "ofl/bevietnampro/BeVietnamPro-Italic.ttf", "bolditalic": "ofl/bevietnampro/BeVietnamPro-BoldItalic.ttf"}),
    "CrimsonText": (1.0, {
        "regular": "ofl/crimsontext/CrimsonText-Regular.ttf", "bold": "ofl/crimsontext/CrimsonText-Bold.ttf",
        "italic": "ofl/crimsontext/CrimsonText-Italic.ttf", "bolditalic": "ofl/crimsontext/CrimsonText-BoldItalic.ttf"}),
    # Typewriter look for old photocopied decisions.
    "Cousine": (0.7, {
        "regular": "ofl/cousine/Cousine-Regular.ttf", "bold": "ofl/cousine/Cousine-Bold.ttf",
        "italic": "ofl/cousine/Cousine-Italic.ttf", "bolditalic": "ofl/cousine/Cousine-BoldItalic.ttf"}),
}


@dataclass(frozen=True)
class Family:
    name: str
    weight: float
    styles: dict[str, Path]

    def path(self, bold: bool = False, italic: bool = False) -> Path:
        key = ("bold" if bold else "") + ("italic" if italic else "") or "regular"
        return self.styles.get(key) or self.styles["regular"]


def covers_vietnamese(path: str | Path) -> bool:
    from fontTools.ttLib import TTFont

    cmap = TTFont(str(path), lazy=True).getBestCmap() or {}
    return all(ord(c) in cmap for c in vietnamese_letters())


def _download(url: str, dest: Path) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read()
    except OSError:
        return False
    dest.write_bytes(data)
    return True


def ensure_fonts(directory: str | Path, extra_dirs: list[str | Path] = ()) -> list[Family]:
    """Download (once) and validate the font families; `extra_dirs` may hold the user's own `*.ttf` families named
    `<Family>-Regular.ttf`, `<Family>-Bold.ttf`, `-Italic`, `-BoldItalic` (e.g. a licensed Times New Roman)."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    families: list[Family] = []
    for name, (weight, styles) in FAMILIES.items():
        found: dict[str, Path] = {}
        for style, remote in styles.items():
            local = directory / Path(remote).name
            if not local.exists() and not _download(f"{_BASE}/{remote}", local):
                continue
            if covers_vietnamese(local):
                found[style] = local
        if "regular" in found:
            families.append(Family(name, weight, found))
    for extra in extra_dirs:
        for regular in sorted(Path(extra).glob("*-Regular.ttf")):
            stem = regular.name[: -len("-Regular.ttf")]
            styles = {"regular": regular}
            for style, suffix in (("bold", "Bold"), ("italic", "Italic"), ("bolditalic", "BoldItalic")):
                candidate = regular.with_name(f"{stem}-{suffix}.ttf")
                if candidate.exists():
                    styles[style] = candidate
            if all(covers_vietnamese(p) for p in styles.values()):
                families.append(Family(stem, 6.0, styles))
    if not families:
        raise RuntimeError(f"no usable Vietnamese font in {directory}; check network access or pass --fonts-dir")
    return families


def pick(families: list[Family], rng: random.Random) -> Family:
    return rng.choices(families, weights=[f.weight for f in families])[0]
