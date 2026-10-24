"""Synthetic Vietnamese administrative pages with exact labels for all three models.

One composed page yields:
  * detection  – line polygons (`###` marks the seal and signature, which must not be read as text),
  * layout     – COCO boxes in legal-document classes (PaddleX layout detection format),
  * recognition – line crops rectified from the final, degraded page (what the recogniser sees in production).

The page follows Nghị định 30/2020/NĐ-CP: national motto top right, issuing agency top left, number and date, title,
justified body in Times-like type, tables, "Nơi nhận" bottom left, signature block with seal bottom right.
"""
from __future__ import annotations

import json
import math
import os
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

from . import charset, corpus, geometry as geo, legal_text
from .common import is_val, rng_for, run_parallel, write_lines
from .degrade import degrade
from .fonts import Family, ensure_fonts, pick
from .render import ink_box, load_font, random_ink, random_paper

DPI = 150
PX_PER_PT = DPI / 72
PX_PER_MM = DPI / 25.4

# Legal-document layout classes. Ids start at 1: PaddleX/COCO reserve 0 for `_background_`.
CATEGORIES = ["national_header", "issuer", "doc_number_date", "doc_title", "para_title", "text", "table",
              "signature", "recipients", "seal", "page_number"]
CAT_ID = {name: i + 1 for i, name in enumerate(CATEGORIES)}

_PARENTS = ["ỦY BAN NHÂN DÂN", "HỘI ĐỒNG NHÂN DÂN", "CHÍNH PHỦ", "BỘ TÀI CHÍNH", "BỘ Y TẾ", "BỘ GIÁO DỤC VÀ ĐÀO TẠO",
            "SỞ XÂY DỰNG", "SỞ TÀI NGUYÊN VÀ MÔI TRƯỜNG", "TÒA ÁN NHÂN DÂN TỐI CAO", "NGÂN HÀNG NHÀ NƯỚC VIỆT NAM"]
_PLACES = ["TỈNH BÌNH PHƯỚC", "THÀNH PHỐ HÀ NỘI", "TỈNH NGHỆ AN", "TỈNH KON TUM", "THÀNH PHỐ ĐÀ NẴNG", "TỈNH LÂM ĐỒNG", "HUYỆN CỦ CHI"]
_DOC_TYPES = ["QUYẾT ĐỊNH", "NGHỊ ĐỊNH", "THÔNG TƯ", "NGHỊ QUYẾT", "CHỈ THỊ", "KẾ HOẠCH", "THÔNG BÁO", "BÁO CÁO", "LUẬT"]
_POSITIONS = ["CHỦ TỊCH", "PHÓ CHỦ TỊCH", "BỘ TRƯỞNG", "THỨ TRƯỞNG", "GIÁM ĐỐC", "CHÁNH VĂN PHÒNG", "TỔNG GIÁM ĐỐC", "CHÁNH ÁN"]
_ON_BEHALF = ["TM. ỦY BAN NHÂN DÂN", "KT. BỘ TRƯỞNG", "TM. CHÍNH PHỦ", "TL. BỘ TRƯỞNG", "TM. HỘI ĐỒNG NHÂN DÂN", ""]
_SURNAMES = ["Nguyễn", "Trần", "Lê", "Phạm", "Hoàng", "Huỳnh", "Phan", "Vũ", "Võ", "Đặng", "Bùi", "Đỗ", "Hồ", "Ngô", "Dương", "Lý"]
_MIDDLE = ["Văn", "Thị", "Hữu", "Đức", "Minh", "Quốc", "Thanh", "Ngọc", "Xuân", "Hồng", "Thế", "Công"]
_GIVEN = ["Anh", "Bình", "Cường", "Dũng", "Hà", "Hải", "Hùng", "Khánh", "Lan", "Linh", "Nam", "Phong", "Quang", "Sơn", "Tâm", "Thảo", "Tuấn", "Việt", "Yến", "Ánh", "Đạt", "Ngân"]
_RECIPIENT_ITEMS = ["Như Điều 3", "Thủ tướng Chính phủ", "Văn phòng Chính phủ", "Bộ Tư pháp (Cục Kiểm tra văn bản)", "Thường trực Tỉnh ủy",
                    "Các Sở, ban, ngành", "Lưu: VT, TH", "Lưu: VT, NC", "Cổng thông tin điện tử", "Công báo", "UBND các huyện, thị xã"]


@dataclass
class Canvas:
    width: int
    height: int
    paper: tuple[int, int, int]
    left: float
    right: float
    top: float
    bottom: float
    img: Image.Image = field(init=False)
    draw: ImageDraw.ImageDraw = field(init=False)
    lines: list[tuple[str, geo.Quad]] = field(default_factory=list)
    regions: list[tuple[str, geo.Quad]] = field(default_factory=list)
    ignores: list[geo.Quad] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.img = Image.new("RGB", (self.width, self.height), self.paper)
        self.draw = ImageDraw.Draw(self.img)

    def line(self, text: str, font, x: float, baseline: float, ink, *, x_end: float | None = None) -> geo.Quad:
        """Draw one run with its baseline at `baseline`; record the ink-tight label. `x_end` overrides the right edge
        for justified lines, which are drawn word by word."""
        l, t, r, b = ink_box(font, text)
        self.draw.text((x, baseline), text, font=font, fill=ink, anchor="ls")
        quad = geo.box_to_quad(x + l - 1, baseline + t - 1, (x_end if x_end is not None else x + r) + 1, baseline + b + 1)
        self.lines.append((text, quad))
        return quad

    def region(self, category: str, quads: list[geo.Quad]) -> None:
        if quads:
            xs0, ys0, xs1, ys1 = zip(*(geo.quad_bounds(q) for q in quads))
            self.regions.append((category, geo.box_to_quad(min(xs0), min(ys0), max(xs1), max(ys1))))


def _wrap(words: list[str], font, width: float, first_indent: float = 0.0) -> list[list[str]]:
    space = font.getlength(" ")
    lines: list[list[str]] = [[]]
    used = first_indent
    for word in words:
        w = font.getlength(word)
        if lines[-1] and used + space + w > width:
            lines.append([])
            used = 0.0
        used += (space if lines[-1] else 0) + w
        lines[-1].append(word)
    return lines


def paragraph(c: Canvas, text: str, font, x0: float, x1: float, y: float, ink, rng: random.Random, *, category: str,
              align: str = "justify", indent: float = 0.0, spacing: float = 1.25, record: bool = True) -> float:
    """Lay a paragraph out from `y` (top). Stops at the page bottom. Returns the y after the paragraph."""
    ascent, descent = font.getmetrics()
    step = (ascent + descent) * spacing
    wrapped = _wrap(text.split(" "), font, x1 - x0, indent)
    quads: list[geo.Quad] = []
    space = font.getlength(" ")
    for i, words in enumerate(wrapped):
        baseline = y + ascent
        if baseline + descent > c.bottom:
            break
        start = x0 + (indent if i == 0 else 0)
        width = sum(font.getlength(w) for w in words) + space * (len(words) - 1)
        last = i == len(wrapped) - 1
        if align == "center":
            quads.append(c.line(" ".join(words), font, (x0 + x1) / 2 - width / 2, baseline, ink))
        elif align == "justify" and not last and len(words) > 1:
            gap = (x1 - start - sum(font.getlength(w) for w in words)) / (len(words) - 1)
            x = start
            for word in words:
                c.draw.text((x, baseline), word, font=font, fill=ink, anchor="ls")
                x += font.getlength(word) + gap
            l, t, r, b = ink_box(font, " ".join(words))
            quad = geo.box_to_quad(start + l - 1, baseline + t - 1, x1 + 1, baseline + b + 1)
            c.lines.append((" ".join(words), quad))
            quads.append(quad)
        else:
            quads.append(c.line(" ".join(words), font, start, baseline, ink))
        y += step
    if record:
        c.region(category, quads)
    return y


def _table(c: Canvas, rows: list[list[str]], font, bold_font, x0: float, x1: float, y: float, ink, rng: random.Random) -> float:
    ncols = len(rows[0])
    weights = [min(40, max(4, max(len(r[i]) for r in rows))) for i in range(ncols)]
    widths = [(x1 - x0) * w / sum(weights) for w in weights]
    xs = [x0]
    for w in widths:
        xs.append(xs[-1] + w)
    pad = 6
    ascent, descent = font.getmetrics()
    step = (ascent + descent) * 1.15
    top = y
    line_w = rng.choice([1, 1, 2, 3])
    quads: list[geo.Quad] = []
    for ri, row in enumerate(rows):
        cell_lines = [_wrap(cell.split(" "), bold_font if ri == 0 else font, widths[i] - 2 * pad)[:6] if cell else [[]] for i, cell in enumerate(row)]
        height = max(len(cl) for cl in cell_lines) * step + 2 * pad
        if y + height > c.bottom:
            break
        c.draw.line([(x0, y), (x1, y)], fill=ink, width=line_w)
        for i, lines in enumerate(cell_lines):
            f = bold_font if ri == 0 else font
            for k, words in enumerate(lines):
                if not words:
                    continue
                text = " ".join(words)
                w = f.getlength(text)
                x = xs[i] + (widths[i] - w) / 2 if ri == 0 else xs[i] + pad
                quads.append(c.line(text, f, x, y + pad + ascent + k * step, ink))
        y += height
    c.draw.line([(x0, y), (x1, y)], fill=ink, width=line_w)
    for x in xs:
        c.draw.line([(x, top), (x, y)], fill=ink, width=line_w)
    c.region("table", [geo.box_to_quad(x0, top, x1, y)])
    return y


def _seal(c: Canvas, cx: float, cy: float, radius: float, rng: random.Random) -> None:
    """Red round stamp, composited translucent so text below stays partly visible."""
    layer = Image.new("RGBA", c.img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    red = (rng.randint(170, 215), rng.randint(20, 60), rng.randint(20, 60), rng.randint(150, 215))
    w = rng.randint(3, 6)
    d.ellipse([cx - radius, cy - radius, cx + radius, cy + radius], outline=red, width=w)
    inner = radius * rng.uniform(0.62, 0.74)
    d.ellipse([cx - inner, cy - inner, cx + inner, cy + inner], outline=red, width=max(2, w - 2))
    for k in range(rng.randint(26, 40)):  # dashes standing in for the ring text
        a = 2 * math.pi * k / 40 + rng.uniform(-0.03, 0.03)
        r0, r1 = inner + 6, radius - w - 5
        d.line([(cx + r0 * math.cos(a), cy + r0 * math.sin(a)), (cx + r1 * math.cos(a), cy + r1 * math.sin(a))], fill=red, width=rng.randint(3, 6))
    star = [(cx + inner * 0.6 * math.cos(-math.pi / 2 + k * 2 * math.pi / 5 * 2), cy + inner * 0.6 * math.sin(-math.pi / 2 + k * 2 * math.pi / 5 * 2)) for k in range(5)]
    d.polygon(star, fill=red)
    layer = layer.filter(ImageFilter.GaussianBlur(rng.uniform(0.3, 1.0)))
    c.img.paste(layer, (0, 0), layer)
    c.draw = ImageDraw.Draw(c.img)
    box = geo.box_to_quad(cx - radius, cy - radius, cx + radius, cy + radius)
    c.ignores.append(box)
    c.regions.append(("seal", box))


def _scribble(c: Canvas, x0: float, y0: float, x1: float, y1: float, rng: random.Random) -> None:
    ink = rng.choice([(20, 30, 120), (10, 10, 10), (30, 40, 150)])
    pts = []
    n = rng.randint(10, 18)
    for k in range(n):
        t = k / (n - 1)
        pts.append((x0 + (x1 - x0) * t + rng.uniform(-8, 8), (y0 + y1) / 2 + math.sin(t * rng.uniform(5, 11)) * (y1 - y0) / 2 * rng.uniform(0.5, 1)))
    c.draw.line(pts, fill=ink, width=rng.randint(2, 4), joint="curve")
    c.ignores.append(geo.box_to_quad(x0 - 6, y0 - 6, x1 + 6, y1 + 6))


def _person(rng: random.Random) -> str:
    return f"{rng.choice(_SURNAMES)} {rng.choice(_MIDDLE)} {rng.choice(_GIVEN)}" if rng.random() < 0.8 else f"{rng.choice(_SURNAMES)} {rng.choice(_GIVEN)}"


class Style:
    """Per-page typographic choices."""

    def __init__(self, family: Family, rng: random.Random) -> None:
        self.family = family
        self.body_pt = rng.choice([12, 13, 13, 14, 14])
        self.spacing = rng.uniform(1.1, 1.45)
        self.indent = rng.choice([0, 0.8, 1.0, 1.27]) * 10 * PX_PER_MM
        self.align = "justify" if rng.random() < 0.75 else "left"
        self.ink = random_ink(rng)

    def font(self, pt: float, bold: bool = False, italic: bool = False):
        return load_font(str(self.family.path(bold, italic)), max(10, round(pt * PX_PER_PT)))


def _header(c: Canvas, st: Style, rng: random.Random, y: float) -> float:
    width = c.right - c.left
    left_col = (c.left, c.left + width * 0.42)
    right_col = (c.left + width * 0.44, c.right)
    mid = lambda col: (col[0] + col[1]) / 2
    # national motto (right)
    f1, f2 = st.font(rng.choice([12, 13]), bold=True), st.font(rng.choice([13, 14]), bold=True)
    asc1, asc2 = f1.getmetrics()[0], f2.getmetrics()[0]
    q = []
    t1, t2 = "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM", "Độc lập - Tự do - Hạnh phúc"
    q.append(c.line(t1, f1, mid(right_col) - f1.getlength(t1) / 2, y + asc1, st.ink))
    y2 = y + (asc1 + f1.getmetrics()[1]) * 1.2
    q.append(c.line(t2, f2, mid(right_col) - f2.getlength(t2) / 2, y2 + asc2, st.ink))
    rule_y = y2 + asc2 + f2.getmetrics()[1] + 3
    half = f2.getlength(t2) / 2
    c.draw.line([(mid(right_col) - half, rule_y), (mid(right_col) + half, rule_y)], fill=st.ink, width=2)
    c.region("national_header", q)
    # issuing agency (left)
    fa, fb = st.font(rng.choice([12, 13]), bold=False), st.font(rng.choice([12, 13]), bold=True)
    parent, child = rng.choice(_PARENTS), rng.choice(_PLACES)
    qa = [c.line(parent, fa, mid(left_col) - fa.getlength(parent) / 2, y + fa.getmetrics()[0], st.ink)]
    ya = y + (fa.getmetrics()[0] + fa.getmetrics()[1]) * 1.2
    qa.append(c.line(child, fb, mid(left_col) - fb.getlength(child) / 2, ya + fb.getmetrics()[0], st.ink))
    ry = ya + fb.getmetrics()[0] + fb.getmetrics()[1] + 3
    c.draw.line([(mid(left_col) - fb.getlength(child) / 4, ry), (mid(left_col) + fb.getlength(child) / 4, ry)], fill=st.ink, width=2)
    c.region("issuer", qa)
    y = max(rule_y, ry) + 14 * PX_PER_PT / 2
    # number and date
    fn, fd = st.font(13), st.font(rng.choice([13, 14]), italic=True)
    num = f"Số: {legal_text.doc_number(rng)}"
    date = legal_text.date_line(rng)
    qn = c.line(num, fn, mid(left_col) - fn.getlength(num) / 2, y + fn.getmetrics()[0], st.ink)
    qd = c.line(date, fd, mid(right_col) - fd.getlength(date) / 2, y + fd.getmetrics()[0], st.ink)
    c.region("doc_number_date", [qn])
    c.region("doc_number_date", [qd])
    return y + max(sum(fn.getmetrics()), sum(fd.getmetrics())) * 1.3 + 10 * PX_PER_PT


def _title(c: Canvas, st: Style, rng: random.Random, y: float, subject: str, kind: str) -> float:
    ft = st.font(rng.choice([14, 15, 16]), bold=True)
    asc, desc = ft.getmetrics()
    first_line = len(c.lines)
    c.line(kind, ft, (c.left + c.right) / 2 - ft.getlength(kind) / 2, y + asc, st.ink)
    y += (asc + desc) * 1.2
    fs = st.font(rng.choice([13, 14]), bold=True)
    inner = (c.right - c.left) * rng.uniform(0.55, 0.8)
    mid = (c.left + c.right) / 2
    y = paragraph(c, subject, fs, mid - inner / 2, mid + inner / 2, y, st.ink, rng, category="doc_title", align="center", spacing=1.15, record=False)
    c.region("doc_title", [q for _, q in c.lines[first_line:]])  # one region over the type and the subject
    return y + 10 * PX_PER_PT


def _signature(c: Canvas, st: Style, rng: random.Random, y: float, with_recipients: bool) -> float:
    """Recipients bottom-left and the signature block bottom-right, from `y` (top). Returns the y after both."""
    width = c.right - c.left
    sig_x0, sig_x1 = c.left + width * 0.5, c.right
    mid = (sig_x0 + sig_x1) / 2
    fb = st.font(rng.choice([13, 14]), bold=True)
    asc, desc = fb.getmetrics()
    step = (asc + desc) * 1.2
    quads, yy = [], y
    on_behalf = rng.choice(_ON_BEHALF)
    for text in [t for t in (on_behalf, rng.choice(_POSITIONS)) if t]:
        quads.append(c.line(text, fb, mid - fb.getlength(text) / 2, yy + asc, st.ink))
        yy += step
    gap = rng.uniform(2.5, 4.0) * 10 * PX_PER_MM
    if yy + gap + step > c.bottom + 1:
        return y
    name = _person(rng)
    name_y = yy + gap
    quads.append(c.line(name, fb, mid - fb.getlength(name) / 2, name_y + asc, st.ink))
    c.region("signature", quads)
    if rng.random() < 0.8:
        _scribble(c, mid - width * 0.1, yy + 4, mid + width * 0.08, name_y - 4, rng)
    if rng.random() < 0.85:
        radius = rng.uniform(0.34, 0.45) * 10 * PX_PER_MM * 3
        _seal(c, sig_x0 + (sig_x1 - sig_x0) * rng.uniform(0.12, 0.4), yy + gap * 0.5, radius, rng)
    end = name_y + step
    if with_recipients:
        fr = st.font(11, bold=True, italic=True)
        fi = st.font(11)
        ra, rd = fr.getmetrics()
        rq = [c.line("Nơi nhận:", fr, c.left, y + ra, st.ink)]
        ry = y + (ra + rd) * 1.15
        items = rng.sample(_RECIPIENT_ITEMS, rng.randint(2, 6))
        for k, item in enumerate(items):
            text = "- " + item + (";" if k < len(items) - 1 else ".")
            ia, idsc = fi.getmetrics()
            if ry + ia + idsc > c.bottom:
                break
            rq.append(c.line(text, fi, c.left, ry + ia, st.ink))
            ry += (ia + idsc) * 1.1
        c.region("recipients", rq)
        end = max(end, ry)
    return end


def _page_number(c: Canvas, st: Style, rng: random.Random, number: int) -> None:
    f = st.font(rng.choice([12, 13]))
    text = str(number)
    asc = f.getmetrics()[0]
    at_top = rng.random() < 0.6
    baseline = (c.top - 12 * PX_PER_PT / 2) if at_top else (c.bottom + 14 * PX_PER_PT + asc)
    baseline = max(baseline, asc + 6)
    quad = c.line(text, f, (c.left + c.right) / 2 - f.getlength(text) / 2, baseline, st.ink)
    c.region("page_number", [quad])


_TABLE_HEADS = [["STT", "Nội dung", "Đơn vị tính", "Số lượng"], ["STT", "Họ và tên", "Chức vụ", "Ghi chú"], ["Số TT", "Tên đơn vị", "Số tiền", "Thời hạn"],
                ["STT", "Chỉ tiêu", "Kế hoạch", "Thực hiện", "Tỷ lệ (%)"], ["Mã số", "Danh mục", "Mức phí"]]


def _random_table(rng: random.Random, blocks: list[dict]) -> list[list[str]]:
    """A data table with the cell types that matter in legal annexes: running numbers, names, amounts, dates."""
    head = list(rng.choice(_TABLE_HEADS))
    pool = [b["text"][:70] for b in blocks if b["kind"] == "text" and len(b["text"]) > 12] or ["Nội dung"]
    fillers = [lambda: rng.choice(pool), lambda: legal_text.amount(rng), lambda: _person(rng), lambda: legal_text.date_line(rng).split(", ", 1)[1], lambda: str(rng.randint(1, 500))]
    kinds = [rng.choice(fillers) for _ in head[2:]]
    rows = [head]
    for n in range(1, rng.randint(3, 9) + 1):
        rows.append([str(n), rng.choice(pool)] + [f() for f in kinds])
    return rows


def split_document(rng: random.Random, blocks: list[dict]) -> tuple[str, str, int]:
    """(document type, subject, index of the first body block). The source HTML opens with its own letterhead, which the
    composer redraws, so the body starts at the first "Căn cứ"/"Điều 1" block when there is one."""
    head = blocks[:14]
    kind = next((b["text"] for b in head if b["kind"] == "title" and len(b["text"]) <= 25), None) or rng.choice(_DOC_TYPES)
    body = next((i for i, b in enumerate(blocks) if b["text"].startswith(("Căn cứ", "Điều 1"))), min(len(blocks) - 1, 4))
    subject = next((b["text"] for b in blocks[:max(body, 1)] if 20 <= len(b["text"]) <= 200 and b["kind"] != "table_row"
                    and b["text"] != kind and not b["text"].isupper()), None) or "Về việc ban hành quy chế làm việc"
    return kind, subject, body


def compose_page(rng: random.Random, blocks: list[dict], families: list[Family], *, first: bool, last: bool,
                 page_no: int, start: int, kind: str = "QUYẾT ĐỊNH", subject: str = "Về việc ban hành quy chế") -> tuple[Canvas, int]:
    """One A4 page at 150 dpi. Returns the canvas and the index of the next unused block."""
    width, height = round(210 * PX_PER_MM), round(297 * PX_PER_MM)
    c = Canvas(width, height, random_paper(rng), left=rng.uniform(27, 33) * PX_PER_MM, right=width - rng.uniform(13, 18) * PX_PER_MM,
               top=rng.uniform(18, 24) * PX_PER_MM, bottom=height - rng.uniform(18, 24) * PX_PER_MM)
    st = Style(pick(families, rng), rng)
    y = c.top
    if first:
        y = _header(c, st, rng, y)
        y = _title(c, st, rng, y, subject, kind)
    elif page_no > 1:
        _page_number(c, st, rng, page_no)
    if last:
        c.bottom -= rng.uniform(45, 70) * PX_PER_MM  # room for the signature block
    body = st.font(st.body_pt)
    bold = st.font(st.body_pt, bold=True)
    i = start
    n = len(blocks)
    table_at = rng.randint(1, 5) if rng.random() < 0.22 else -1
    emitted = 0
    while i < n and y < c.bottom - body.size * 2:
        if emitted == table_at:
            y = _table(c, _random_table(rng, blocks), body, bold, c.left, c.right, y + 6, st.ink, rng) + 8 * PX_PER_PT
        emitted += 1
        b = blocks[i]
        i += 1
        if b["kind"] == "table_row":
            rows = [b["cells"]]
            while i < n and blocks[i]["kind"] == "table_row" and len(blocks[i]["cells"]) == len(rows[0]) and len(rows) < 12:
                rows.append(blocks[i]["cells"])
                i += 1
            cells = [cell for r in rows for cell in r]
            # letterhead and signature tables (few rows, empty or long cells) are layout, not data tables
            if 2 <= len(rows[0]) <= 6 and len(rows) >= 2 and max(map(len, cells)) <= 150 and sum(map(bool, cells)) >= 0.7 * len(cells):
                y = _table(c, rows, body, bold, c.left, c.right, y + 6, st.ink, rng) + 8 * PX_PER_PT
            continue
        if b["kind"] == "title" or (b["kind"] == "heading" and b.get("center")):
            y = paragraph(c, b["text"], bold, c.left, c.right, y + 4 * PX_PER_PT, st.ink, rng, category="para_title", align="center", spacing=st.spacing)
        elif b["kind"] == "heading":
            y = paragraph(c, b["text"], bold, c.left, c.right, y + 4 * PX_PER_PT, st.ink, rng, category="para_title", align="left", spacing=st.spacing)
        else:
            align = "center" if b.get("center") and len(b["text"]) < 90 else st.align
            y = paragraph(c, b["text"], bold if b.get("bold") and align == "center" else body, c.left, c.right, y, st.ink, rng,
                          category="text", align=align, indent=st.indent if align != "center" else 0, spacing=st.spacing)
        y += 3 * PX_PER_PT
    if last:
        c.bottom += 70 * PX_PER_MM
        _signature(c, st, rng, y + 6 * PX_PER_PT, with_recipients=True)
    return c, i


def finalize(c: Canvas, rng: random.Random, level: float):
    return finalize_labels(c.img, c.lines, c.regions, c.ignores, rng, level)


def finalize_labels(img: Image.Image, lines: list[tuple[str, geo.Quad]], regions: list[tuple[str, geo.Quad]], ignores: list[geo.Quad],
                    rng: random.Random, level: float) -> tuple[Image.Image, list[tuple[str, geo.Quad]], list[tuple[str, geo.Quad]], list[geo.Quad]]:
    """Skew, rescale (different scan resolutions) and degrade the page, carrying every label along."""
    angle = rng.uniform(-2.0, 2.0) if rng.random() < 0.75 else 0.0
    if abs(angle) > 0.05:
        fill = random_paper(rng) if rng.random() < 0.8 else (rng.randint(90, 200),) * 3
        rotated = img.rotate(angle, expand=True, resample=Image.BICUBIC, fillcolor=fill)
        mp = lambda q: geo.rotate_points(q, angle, img.size, rotated.size)
        lines = [(t, mp(q)) for t, q in lines]
        regions = [(k, geo.box_to_quad(*geo.quad_bounds(mp(q)))) for k, q in regions]
        ignores = [mp(q) for q in ignores]
        img = rotated
    scale = rng.choice([0.6, 0.8, 1.0, 1.0, 1.0, 1.3])
    if scale != 1.0:
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.LANCZOS)
        lines = [(t, geo.scale_points(q, scale, scale)) for t, q in lines]
        regions = [(k, geo.scale_points(q, scale, scale)) for k, q in regions]
        ignores = [geo.scale_points(q, scale, scale) for q in ignores]
    return degrade(img, rng, level), lines, regions, ignores


@dataclass
class PageJob:
    index: int
    count: int
    root: str
    corpus_path: str
    fonts_dir: str
    extra_fonts: list[str]
    seed: int
    val_ratio: float
    level: float
    rec_per_page: int
    max_chars: int


def _doc_pages(rng: random.Random, doc: dict, families: list[Family]):
    blocks = doc["blocks"]
    kind, subject, cursor = split_document(rng, blocks)
    n_pages = rng.choices([1, 2, 3], [5, 3, 2])[0]
    for p in range(1, n_pages + 1):
        c, cursor = compose_page(rng, blocks, families, first=p == 1, last=p == n_pages, page_no=p, start=cursor, kind=kind, subject=subject)
        yield p, c


def _work(job: PageJob) -> dict:
    rng = rng_for(job.seed, "pages", job.index)
    families = ensure_fonts(job.fonts_dir, job.extra_fonts)
    docs = [d for d in corpus.load_corpus(job.corpus_path) if len(d["blocks"]) >= 8]
    allowed = set(charset.build_charset())
    root = Path(job.root)
    (root / "det/synth_pages/images").mkdir(parents=True, exist_ok=True)
    (root / "rec/synth_pages/images").mkdir(parents=True, exist_ok=True)
    out = {"det": {"train": [], "val": []}, "rec": {"train": [], "val": []}, "coco": {"train": [], "val": []}}
    made = 0
    while made < job.count:
        doc = rng.choice(docs)
        split = "val" if is_val(doc["id"], job.val_ratio) else "train"
        for p, canvas in _doc_pages(rng, doc, families):
            if made >= job.count:
                break
            if len(canvas.lines) < 3:
                continue
            page_id = f"p{job.index:03d}_{made:06d}"
            img, lines, regions, ignores = finalize(canvas, rng, job.level)
            rel = f"synth_pages/images/{page_id}.jpg"
            img.save(root / "det" / rel, quality=rng.randint(80, 95))
            det = [{"transcription": t, "points": [[round(x, 1), round(y, 1)] for x, y in q]} for t, q in lines]
            det += [{"transcription": "###", "points": [[round(x, 1), round(y, 1)] for x, y in q]} for q in ignores]
            out["det"][split].append(f"{rel}\t{json.dumps(det, ensure_ascii=False)}")
            out["coco"][split].append({"file": f"{page_id}.jpg", "size": img.size, "regions": [(CAT_ID[k], geo.quad_bounds(q)) for k, q in regions]})
            chosen = rng.sample(lines, min(job.rec_per_page, len(lines)))
            for k, (text, quad) in enumerate(chosen):
                text = charset.restrict(charset.normalize(text), allowed, 1.0)
                if not text or len(text) > job.max_chars:
                    continue
                crop = geo.crop_quad(img, geo.clip_quad(geo.jitter_quad(quad, rng, 4), *img.size))
                if crop is None:
                    continue
                crel = f"synth_pages/images/{page_id}_{k:02d}.jpg"
                crop.save(root / "rec" / crel, quality=92)
                out["rec"][split].append(f"{crel}\t{text}")
            made += 1
    return out


def build(root: str | Path, corpus_path: str | Path, fonts_dir: str | Path, count: int, *, workers: int = 2, seed: int = 0,
          val_ratio: float = 0.04, level: float = 1.0, rec_per_page: int = 20, max_chars: int = 100,
          extra_fonts: list[str] = ()) -> dict[str, int]:
    """Writes `<root>/det/synth_pages`, `<root>/rec/synth_pages` and `<root>/layout` (PaddleX COCO)."""
    ensure_fonts(fonts_dir, list(extra_fonts))
    root = Path(root)
    shards = max(1, workers * 3)
    per = -(-count // shards)
    jobs = [PageJob(i, per, str(root), str(corpus_path), str(fonts_dir), list(extra_fonts), seed, val_ratio, level, rec_per_page, max_chars) for i in range(shards)]
    results = run_parallel(_work, jobs, workers)
    stats: dict[str, int] = {}
    for kind in ("det", "rec"):
        for split in ("train", "val"):
            lines = [l for r in results for l in r[kind][split]]
            write_lines(root / kind / "synth_pages" / f"{split}.txt", lines)
            stats[f"{kind}_{split}"] = len(lines)
    _write_coco(root / "layout", root / "det/synth_pages/images", [r["coco"] for r in results], stats)
    return stats


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(os.path.relpath(src, dst.parent), dst)
    except OSError:  # file systems without symlinks (Drive mounts)
        shutil.copytree(src, dst)


def _write_coco(layout_root: Path, images_dir: Path, parts: list[dict], stats: dict[str, int]) -> None:
    (layout_root / "annotations").mkdir(parents=True, exist_ok=True)
    (layout_root / "images").mkdir(parents=True, exist_ok=True)
    _link_or_copy(images_dir, layout_root / "images" / "JPEGImages")
    cats = [{"supercategory": None, "id": 0, "name": "_background_"}] + [{"supercategory": None, "id": CAT_ID[n], "name": n} for n in CATEGORIES]
    for split in ("train", "val"):
        images, anns = [], []
        for part in parts:
            for item in part[split]:
                image_id = len(images)
                w, h = item["size"]
                images.append({"license": 0, "url": None, "file_name": f"JPEGImages/{item['file']}", "height": h, "width": w, "date_captured": None, "id": image_id})
                for cat, (x0, y0, x1, y1) in item["regions"]:
                    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(w, x1), min(h, y1)
                    if x1 - x0 < 4 or y1 - y0 < 4:
                        continue
                    anns.append({"id": len(anns), "image_id": image_id, "category_id": cat, "segmentation": [[x0, y0, x0, y1, x1, y1, x1, y0]],
                                 "area": (x1 - x0) * (y1 - y0), "bbox": [x0, y0, x1 - x0, y1 - y0], "iscrowd": 0})
        data = {"info": {"description": "synthetic Vietnamese administrative pages"}, "licenses": [], "type": "instances", "annotations": anns, "images": images, "categories": cats}
        (layout_root / "annotations" / f"instance_{split}.json").write_text(json.dumps(data), encoding="utf-8")
        stats[f"layout_{split}_images"] = len(images)
