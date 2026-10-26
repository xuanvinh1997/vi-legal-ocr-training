"""Character error rate of the paddle-onnx pipeline (detection + recognition, no layout) on real scanned pages that have
hand-typed ground truth (`real_pages/`), the measure that decided the r3 training run: PaddleOCR's own validation accuracy mixes
synthetic and real lines and did not follow the scans.

    python scripts/eval_real.py --models <models dir> [--unclip 2.0] [--json result.json]

The pages are cut from the PDFs in `tests/` (`--pdf-dir` to move them). Besides the CER it counts how
many letters with circumflex + tone mark (ố ề ệ ấ ...) were misread: the family Times New Roman drew differently from the training
font and that the first model got wrong 47% of the time.
Needs the service's dependencies (onnxruntime, opencv, pyclipper; see services/ocr), PyMuPDF and RapidFuzz."""
from __future__ import annotations

import argparse
import collections
import json
import unicodedata
from pathlib import Path

from _common import TESTS_DIR, load_method, render

HERE = Path(__file__).resolve().parent
HARD = set("ốồổỗộếềểễệấầẩẫậ")  # circumflex + tone: the letters a font with stacked marks teaches wrongly


def norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFC", text).split())


def base(char: str) -> str:
    """The letter without its marks (đ is its own base letter)."""
    return "d" if char in "đĐ" else unicodedata.normalize("NFD", char)[0].lower()


def score(truth: str, text: str) -> dict:
    """Edit operations between the ground truth and the matching stretch of the recognised text. The page is read whole, so the
    text may hold more than the truth (a stamp, a footer): the truth is aligned to its best-fitting window of the text first."""
    from rapidfuzz import fuzz
    from rapidfuzz.distance import Levenshtein

    gt, hyp = norm(truth), norm(text)
    if len(hyp) >= len(gt):
        a = fuzz.partial_ratio_alignment(gt, hyp)
        hyp = hyp[a.dest_start:a.dest_end]
    ops = Levenshtein.editops(gt, hyp)
    subs = [(gt[o.src_pos], hyp[o.dest_pos]) for o in ops if o.tag == "replace"]
    deletes = [gt[o.src_pos] for o in ops if o.tag == "delete"]
    return {"chars": len(gt), "errors": len(ops), "subs": len(subs), "deletes": len(deletes), "inserts": sum(o.tag == "insert" for o in ops),
            "tone_only": sum(base(a) == base(b) for a, b in subs),
            "hard_seen": sum(c in HARD for c in gt), "hard_missed": sum(a in HARD for a, _ in subs) + sum(c in HARD for c in deletes),
            "confusions": collections.Counter(subs)}


def total(rows: list[dict]) -> dict:
    out = {k: sum(r[k] for r in rows) for k in ("chars", "errors", "subs", "deletes", "inserts", "tone_only", "hard_seen", "hard_missed")}
    out["confusions"] = sum((r["confusions"] for r in rows), collections.Counter())
    return out


def line(name: str, r: dict) -> str:
    return (f"{name:<8} chars {r['chars']:>5}  CER {r['errors'] / r['chars']:.4f}  errors {r['errors']:>3} (sub {r['subs']} del {r['deletes']} "
            f"ins {r['inserts']}; tone-only {r['tone_only']})  circumflex+tone letters misread {r['hard_missed']}/{r['hard_seen']}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--models", required=True, type=Path, help="folder with det/ and rec/")
    p.add_argument("--unclip", type=float, help="DB unclip ratio (service default 1.5)")
    p.add_argument("--pdf-dir", type=Path, default=TESTS_DIR), p.add_argument("--pages", type=Path, default=HERE / "real_pages" / "pages.json")
    p.add_argument("--only", nargs="*", help="page names from pages.json (default: all)")
    p.add_argument("--json", type=Path, help="write the numbers here")
    args = p.parse_args(argv)

    from ocr_service.methods import Request

    method = load_method(args.models, layout="off", unclip=args.unclip)
    pages = json.loads(args.pages.read_text(encoding="utf-8"))
    rows, per_page = [], {}
    for name, page in pages.items():
        if args.only and name not in args.only:
            continue
        text = method.read(Request(image=render(args.pdf_dir / page["pdf"], page["page"]))).text
        row = score((args.pages.parent / page["truth"]).read_text(encoding="utf-8"), text)
        rows.append(row), per_page.update({name: row})
        print(line(name, row), flush=True)
    t = total(rows)
    print(line("ALL", t))
    print("top confusions:", ", ".join(f"{a}->{b} x{n}" for (a, b), n in t["confusions"].most_common(10)))
    if args.json:
        dump = lambda r: {k: v for k, v in r.items() if k != "confusions"} | {"cer": r["errors"] / r["chars"]}
        args.json.write_text(json.dumps({"pages": {k: dump(v) for k, v in per_page.items()}, "all": dump(t)}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
