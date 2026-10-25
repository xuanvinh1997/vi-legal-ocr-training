"""Chạy Unlimited-OCR (baidu/Unlimited-OCR chính thức, đã vá sang MPS bằng setup_mps.py) trên PDF tiếng Việt.

    .venv/bin/python run.py ../tests/*.pdf --pages 1-4               # multi: mỗi lần 1 trang (--chunk N để gộp N trang)
    .venv/bin/python run.py ../tests/foo.pdf --mode single --pages 2   # từng trang, gundam

multi: `infer_multi`, base 1024, ngram_window 1024. Tổng prompt + output bị chặn bởi --max-length (32768), nên PDF dài
được chia chunk (--chunk trang / lần). single: `infer`, gundam 1024/640, ngram_window 128.
Kết quả: results/<pdf>/page_NNN.md, merged.md, report.json. Nếu PDF có text layer thì tính CER từng trang so với text layer
(chỉ tham khảo, text layer không phải nhãn chuẩn).
"""
import argparse
import json
import re
import time
from pathlib import Path

import pymupdf
import torch
from PIL import Image, ImageDraw
from rapidfuzz.distance import Levenshtein
from transformers import AutoModel, AutoTokenizer

from orient import upright

HERE = Path(__file__).resolve().parent
MARK = re.compile(r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>|<\|det\|>.*?<\|/det\|>", re.S)


def parse_pages(spec, n):
    if not spec:
        return list(range(n))
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += range(int(a) - 1, int(b or a))
    return [p for p in out if 0 <= p < n]


def clean(text):
    return re.sub(r"\n{3,}", "\n\n", MARK.sub("", text)).strip()


BLOCK = re.compile(r"(?:<\|ref\|>([\w-]+)<\|/ref\|>)?<\|det\|>\s*(?:([\w-]+)\s*)?(\[\[?[\d,\s\[\]]+\]?\])\s*<\|/det\|>(.*?)(?=<\|ref\|>|<\|det\|>|\Z)", re.S)
COLORS = {"title": (214, 39, 40), "text": (31, 119, 180), "list": (44, 160, 44), "page_number": (148, 103, 189),
          "equation": (255, 127, 14), "table": (23, 190, 207), "image": (227, 119, 194), "header": (140, 86, 75), "footer": (127, 127, 127)}


def parse_layout(raw, w, h):
    """Output thô -> [{label, bbox (pixel), text}]. Tọa độ model là 0-999 theo chiều rộng/cao ảnh."""
    blocks = []
    for ref, det, box, text in BLOCK.findall(raw):
        try:
            nums = json.loads(box)
        except ValueError:
            continue
        for b in [nums] if isinstance(nums[0], (int, float)) else nums:
            x1, y1, x2, y2 = b
            blocks.append(dict(label=ref or det or "?", bbox=[round(x1 / 999 * w), round(y1 / 999 * h), round(x2 / 999 * w), round(y2 / 999 * h)],
                               text=text.strip()))
    return blocks


def draw_layout(img_path, blocks, out_path):
    img = Image.open(img_path).convert("RGB")
    over = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(over)
    for i, b in enumerate(blocks):
        col = COLORS.get(b["label"], (0, 0, 0))
        d.rectangle(b["bbox"], outline=col + (255,), fill=col + (40,), width=3)
        tag = f"{i}:{b['label']}"
        x, y = b["bbox"][:2]
        d.rectangle([x, y - 18, x + 8 * len(tag) + 6, y], fill=col + (255,))
        d.text((x + 3, y - 15), tag, fill=(255, 255, 255, 255))
    Image.alpha_composite(img.convert("RGBA"), over).convert("RGB").save(out_path)


def norm(s):
    return re.sub(r"\s+", " ", re.sub(r"<[^>]*>|[#*|`>\-]", " ", s)).strip().lower()


def looped(text, n=40, reps=6):
    """Cụm ký tự lặp liền nhau nhiều lần: dấu hiệu vòng lặp khi sinh."""
    return re.search(r"(.{%d,}?)\1{%d,}" % (n, reps - 1), text, re.S) is not None


def suspect(raw):
    """Output suy biến: rỗng, vòng lặp, một từ lặp liên tục, hoặc nhiều khối trùng hệt tọa độ."""
    if not raw.strip():
        return "rỗng"
    if looped(raw):
        return "vòng lặp"
    if re.search(r"\b(\w+)\b(?:\s+\1\b){7,}", raw):
        return "từ lặp"
    boxes = re.findall(r"\[\s*\d+,\s*\d+,\s*\d+,\s*\d+\s*\]", raw)
    if boxes and max(boxes.count(b) for b in set(boxes)) >= 4:
        return "khối trùng tọa độ"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pdfs", nargs="+")
    ap.add_argument("--model-dir", default=str(HERE / "mps_model"))
    ap.add_argument("--out", default=str(HERE / "results"))
    ap.add_argument("--pages", help="vd 1-3,7 (đếm từ 1)")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--mode", choices=["multi", "single"], default="multi")
    ap.add_argument("--chunk", type=int, default=1, help="số trang mỗi lần gọi ở chế độ multi (>1 từng làm trộn nội dung giữa các trang)")
    ap.add_argument("--max-length", type=int, default=32768)
    ap.add_argument("--no-auto-rotate", dest="auto_rotate", action="store_false", help="tắt tự xoay trang")
    ap.add_argument("--force", action="store_true", help="chạy lại cả trang đã có kết quả")
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(a.model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(a.model_dir, trust_remote_code=True, use_safetensors=True, torch_dtype=torch.bfloat16)
    model = model.eval().to(dev)
    print(f"model loaded on {dev} in {time.time() - t0:.0f}s", flush=True)

    for pdf in map(Path, a.pdfs):
        doc = pymupdf.open(pdf)
        work = Path(a.out) / re.sub(r"[^\w.-]+", "_", pdf.stem)[:60]
        (work / "img").mkdir(parents=True, exist_ok=True)
        pages = parse_pages(a.pages, len(doc))
        print(f"\n### {pdf.name}: {len(doc)} trang, chạy {len(pages)}, mode={a.mode}", flush=True)

        imgs, rot = {}, {}
        for p in pages:
            imgs[p] = work / "img" / f"page_{p + 1:03d}.png"
            pm = doc[p].get_pixmap(dpi=a.dpi)
            im = Image.frombytes("RGB", (pm.width, pm.height), pm.samples)
            im, rot[p] = upright(im) if a.auto_rotate else (im, 0)
            im.save(imgs[p])
            if rot[p]:
                print(f"  trang {p + 1}: xoay {rot[p]}° cho thẳng", flush=True)

        def ocr(grp, mode):
            if mode == "multi":
                out, _ = model.infer_multi(tok, prompt="<image>Multi page parsing.", image_files=[str(imgs[p]) for p in grp],
                                           output_path=str(work / "tmp"), image_size=1024, max_length=a.max_length,
                                           no_repeat_ngram_size=35, ngram_window=1024, save_results=False)
                raws = out.split("<PAGE>")[1:]
                if len(raws) != len(grp):
                    print(f"  !! chunk {[p + 1 for p in grp]}: model trả {len(raws)} trang thay vì {len(grp)}", flush=True)
                return (raws + [""] * len(grp))[:len(grp)]
            out = model.infer(tok, prompt="<image>document parsing.", image_file=str(imgs[grp[0]]), output_path=str(work / "tmp"),
                              base_size=1024, image_size=640, crop_mode=True, max_length=a.max_length,
                              no_repeat_ngram_size=35, ngram_window=128, save_results=False)
            return [(out[0] if isinstance(out, tuple) else out) or ""]

        def save_page(p, raw):
            (work / f"page_{p + 1:03d}.md").write_text(clean(raw), encoding="utf-8")
            (work / f"page_{p + 1:03d}.raw.txt").write_text(raw, encoding="utf-8")
            w, h = Image.open(imgs[p]).size
            blocks = parse_layout(raw, w, h)
            (work / f"page_{p + 1:03d}.layout.json").write_text(json.dumps(blocks, ensure_ascii=False, indent=1), encoding="utf-8")
            (work / "vis").mkdir(exist_ok=True)
            draw_layout(imgs[p], blocks, work / "vis" / f"page_{p + 1:03d}.png")

        todo = [p for p in pages if a.force or not (work / f"page_{p + 1:03d}.md").exists()]
        step = a.chunk if a.mode == "multi" else 1
        secs, tries = {}, {}
        for i in range(0, len(todo), step):
            grp = todo[i:i + step]
            t1 = time.time()
            raws = ocr(grp, a.mode)
            print(f"  trang {[p + 1 for p in grp]}: {time.time() - t1:.0f}s", flush=True)
            for p, raw in zip(grp, raws):
                tries[p], secs[p] = [f"{a.mode}x{len(grp)}"], (time.time() - t1) / len(grp)
                # chuỗi chạy lại cho trang nghi lỗi: riêng trang (multi) -> riêng trang (gundam)
                for mode in (["multi", "single"] if a.mode == "multi" else []):
                    if not suspect(raw):
                        break
                    if mode == "multi" and len(grp) == 1:
                        continue  # đã là chạy riêng rồi
                    print(f"  !! trang {p + 1} nghi lỗi ({suspect(raw)}), chạy lại riêng ({mode})", flush=True)
                    t2 = time.time()
                    raw = ocr([p], mode)[0]
                    tries[p].append(mode)
                    secs[p] += time.time() - t2
                save_page(p, raw)

        rows = []
        for p in pages:
            text = (work / f"page_{p + 1:03d}.md").read_text(encoding="utf-8")
            raw = (work / f"page_{p + 1:03d}.raw.txt").read_text(encoding="utf-8")
            ref = norm(doc[p].get_text())
            cer = Levenshtein.normalized_distance(norm(text), ref) if len(ref) > 50 else None
            rows.append(dict(page=p + 1, rotated=rot[p], chars=len(text), seconds=round(secs[p], 1) if p in secs else None,
                             tries=tries.get(p), suspect=suspect(raw), cer_vs_text_layer=None if cer is None else round(cer, 4)))
        (work / "merged.md").write_text("\n\n".join(f"<!-- trang {r['page']} -->\n" + (work / f"page_{r['page']:03d}.md").read_text(encoding="utf-8") for r in rows), encoding="utf-8")
        (work / "report.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
        for r in rows:
            print(r)
        print(f"-> {work}/merged.md")


if __name__ == "__main__":
    main()
