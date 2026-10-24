import random

from ocr_train import charset, corpus

HTML = """<html><head><style>p{}</style></head><body>
<p align="center"><strong>QUYẾT ĐỊNH</strong></p>
<p align="center">Về việc ban hành quy chế</p>
<p><strong>Điều 1.</strong> Ban hành kèm theo Quyết định này Quy chế làm việc của Ủy ban nhân dân tỉnh.</p>
<p>Căn cứ Luật Tổ chức chính quyền địa phương ngày 19 tháng 6 năm 2015;</p>
<table><tr><td>STT</td><td>Họ và tên</td></tr><tr><td>1</td><td>Nguyễn Văn A</td></tr></table>
<p>C¨n cø LuËt Tæ chøc chÝnh quyÒn ®Þa ph­¬ng</p>
<p>&nbsp;</p></body></html>"""


def test_html_to_blocks_kinds():
    blocks = corpus.html_to_blocks(HTML)
    kinds = [(b.kind, b.text) for b in blocks]
    assert ("title", "QUYẾT ĐỊNH") in kinds
    assert any(k == "heading" for k, _ in kinds)
    rows = [b for b in blocks if b.kind == "table_row"]
    assert rows[1].cells == ["1", "Nguyễn Văn A"]
    assert not any("style" in t for _, t in kinds)


def test_clean_blocks_drops_legacy_encoding():
    allowed = set(charset.build_charset())
    cleaned = corpus.clean_blocks(corpus.html_to_blocks(HTML), allowed)
    assert not any("¨" in b.text for b in cleaned)
    assert any(b.text.startswith("Căn cứ") for b in cleaned)


def test_sentences_in_range():
    allowed = set(charset.build_charset())
    docs = [{"id": "1", "blocks": [{"kind": "text", "text": "Căn cứ Luật Tổ chức chính quyền địa phương ngày 19 tháng 6 năm 2015 và các văn bản liên quan"}]}]
    gen = corpus.sentences(docs, random.Random(0), 3, 40)
    lines = [next(gen) for _ in range(200)]
    assert all(3 <= len(l) <= 60 for l in lines) and len(set(lines)) > 20
    assert all(set(l) <= allowed | {" "} for l in lines)


def test_table_cells_with_inner_paragraphs():
    html = ("<table><tr><td><p>STT</p></td><td><p><strong>Tên</strong></p><p>đơn vị</p></td></tr>"
            "<tr><td><p>1</p></td><td><div>Sở Tài chính</div></td></tr>"
            "<tr><td><table><tr><td>lồng</td></tr></table></td><td>x</td></tr></table><p>Sau bảng</p>")
    blocks = corpus.html_to_blocks(html)
    rows = [b.cells for b in blocks if b.kind == "table_row"]
    assert rows[0] == ["STT", "Tên đơn vị"] and rows[1] == ["1", "Sở Tài chính"]
    assert len(rows) == 3 and blocks[-1].text == "Sau bảng"
