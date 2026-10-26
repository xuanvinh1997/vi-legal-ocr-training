"""The helper scripts' own logic (scoring against ground truth, page ranges); the pipeline they drive is tested elsewhere."""
import importlib.util
import sys
from pathlib import Path

import pytest

pytest.importorskip("rapidfuzz")
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


def load(name: str):
    sys.path.insert(0, str(SCRIPTS))
    try:
        spec = importlib.util.spec_from_file_location(f"scripts_{name}", SCRIPTS / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(SCRIPTS))
    return module


def test_a_missing_dot_below_is_a_tone_only_substitution_of_a_hard_letter():
    ev = load("eval_real")
    got = ev.score("Nghị quyết số 57 ban hành ngày hôm nay", "Nghi quyết số 57 ban hành ngày hôm nay")
    assert got["errors"] == 1 and got["subs"] == 1 and got["tone_only"] == 1 and got["hard_missed"] == 0
    hard = ev.score("Nghị quyết tiếp tục được thực hiện đầy đủ", "Nghị quyết tiêp tục được thực hiện đầy đủ")  # ế -> ê
    assert hard["hard_seen"] == 4 and hard["hard_missed"] == 1 and hard["confusions"][("ế", "ê")] == 1


def test_extra_text_around_the_page_is_not_counted_against_the_reading():
    ev = load("eval_real")
    truth = "Điều 1. Phạm vi điều chỉnh của nghị quyết này"
    got = ev.score(truth, "SAO Y Văn phòng " + truth + " 12 Giờ ký 24/06/2026")
    assert got["errors"] == 0 and got["chars"] == len(truth)


def test_totals_add_the_pages_and_their_confusions():
    ev = load("eval_real")
    a = ev.score("Cán bộ tạo điều kiện, tạo cơ hội cho người học", "Cán bộ tao điều kiện, tao cơ hội cho người học")
    b = ev.score("Người học buộc phải đền bù kinh phí", "Người học buôc phải đền bù kinh phí")
    t = ev.total([a, b])
    assert t["chars"] == a["chars"] + b["chars"] and t["errors"] == 3 and t["confusions"][("ạ", "a")] == 2 and t["confusions"][("ộ", "ô")] == 1


def test_the_page_ranges_of_run_pdf():
    run = load("run_pdf")
    assert run.pages_in(None, 4) == [1, 2, 3, 4] and run.pages_in("2-3,7", 5) == [2, 3] and run.pages_in("4", 9) == [4]


def test_every_ground_truth_page_is_listed_with_its_text_and_pdf():
    import json

    pages = json.loads((SCRIPTS / "real_pages" / "pages.json").read_text(encoding="utf-8"))
    assert set(pages) == {"qd_p1", "qd_p2", "nq_p2"}
    for page in pages.values():
        assert (SCRIPTS / "real_pages" / page["truth"]).read_text(encoding="utf-8").strip() and page["page"] >= 1
