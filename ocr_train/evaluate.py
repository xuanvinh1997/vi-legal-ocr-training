"""Recognition metrics that expose what matters for Vietnamese: not just character errors but *tone* errors."""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from .charset import VI_TONES, normalize


def edit_distance(a: str, b: str) -> int:
    try:
        from rapidfuzz.distance import Levenshtein

        return Levenshtein.distance(a, b)
    except ImportError:
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]


def strip_tones(text: str) -> str:
    return unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFD", text) if c not in VI_TONES))


@dataclass
class RecReport:
    lines: int = 0
    exact: int = 0
    chars: int = 0
    errors: int = 0
    errors_ignoring_tones: int = 0
    worst: list[tuple[int, str, str]] = field(default_factory=list)

    @property
    def cer(self) -> float:
        return self.errors / max(1, self.chars)

    @property
    def cer_ignoring_tones(self) -> float:
        return self.errors_ignoring_tones / max(1, self.chars)

    @property
    def tone_cer(self) -> float:
        """Share of the character error that disappears once tone marks are ignored, i.e. pure tone/diacritic mistakes."""
        return max(0.0, self.cer - self.cer_ignoring_tones)

    @property
    def accuracy(self) -> float:
        return self.exact / max(1, self.lines)

    def summary(self) -> str:
        return (f"lines={self.lines}  line-accuracy={self.accuracy:.1%}  CER={self.cer:.2%}  "
                f"CER-ignoring-tones={self.cer_ignoring_tones:.2%}  tone-error-share={self.tone_cer:.2%}")


def score(pairs: list[tuple[str, str]], keep_worst: int = 15) -> RecReport:
    """pairs of (reference, prediction)."""
    report = RecReport()
    for ref, hyp in pairs:
        ref, hyp = normalize(ref), normalize(hyp)
        d = edit_distance(ref, hyp)
        report.lines += 1
        report.exact += ref == hyp
        report.chars += len(ref)
        report.errors += d
        report.errors_ignoring_tones += edit_distance(strip_tones(ref), strip_tones(hyp))
        if d:
            report.worst.append((d, ref, hyp))
    report.worst = sorted(report.worst, reverse=True)[:keep_worst]
    return report


def predict_rec(images: list[str], model_name: str, model_dir: str | None = None, batch_size: int = 32) -> list[str]:
    """Run a PaddleOCR recognition model (a stock one by name, or a fine-tuned inference directory)."""
    from paddleocr import TextRecognition

    model = TextRecognition(model_name=model_name, model_dir=model_dir) if model_dir else TextRecognition(model_name=model_name)
    out: list[str] = []
    for i in range(0, len(images), batch_size):
        for res in model.predict(images[i:i + batch_size], batch_size=batch_size):
            out.append(str(res["rec_text"]))
    return out


def evaluate_list(root: str | Path, list_name: str, model_name: str, model_dir: str | None, limit: int = 2000, seed: int = 0) -> RecReport:
    import random

    lines = [l for l in (Path(root) / "rec" / list_name).read_text(encoding="utf-8").split("\n") if l]
    random.Random(seed).shuffle(lines)
    chosen = [l.split("\t", 1) for l in lines[:limit]]
    preds = predict_rec([str(Path(root) / "rec" / rel) for rel, _ in chosen], model_name, model_dir)
    return score([(text, pred) for (_, text), pred in zip(chosen, preds)])
