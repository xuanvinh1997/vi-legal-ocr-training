import json
from pathlib import Path

from PIL import Image

from ocr_train import charset, dataset, evaluate, public_sets


def _shard(root, kind, name, lines, images):
    d = Path(root) / kind / name / "images"
    d.mkdir(parents=True)
    for rel in images:
        Image.new("RGB", (300, 48), "white").save(Path(root) / kind / rel)
    (Path(root) / kind / name / "train.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_merge_stats_check(tmp_path):
    _shard(tmp_path, "rec", "a", ["a/images/1.jpg\tĐiều 1", "a/images/2.jpg\tkhoản 2 ạ"], ["a/images/1.jpg", "a/images/2.jpg"])
    _shard(tmp_path, "rec", "b", ["b/images/1.jpg\tCộng hòa 中"], ["b/images/1.jpg"])
    counts = dataset.merge(tmp_path, "rec")
    assert counts == {"train": 3, "val": 0, "shards": 2}
    stats = dataset.rec_stats(tmp_path)
    assert stats["img_w"] == 320 and stats["max_text_length"] == stats["max_chars"] + 4
    d = tmp_path / "vi.txt"
    charset.write_dict(d, charset.build_charset())
    problems = dataset.check_rec(tmp_path, d)
    assert problems["unknown_char"] == 1 and problems["unknown_chars_seen"] == "中" and problems["missing_image"] == 0


def test_check_det(tmp_path):
    _shard(tmp_path, "det", "a", ["a/images/1.jpg\t" + json.dumps([{"transcription": "x", "points": [[0, 0], [5, 0], [5, 5], [0, 5]]}, {"transcription": "###", "points": [[0, 0], [5, 0], [5, 5], [0, 5]]}])], ["a/images/1.jpg"])
    dataset.merge(tmp_path, "det")
    assert dataset.check_det(tmp_path) == {"checked": 1, "missing_image": 0, "bad_json": 0, "boxes": 2, "ignored": 1}


def test_import_rec_tsv_filters(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    for i in range(4):
        Image.new("RGB", (240, 40), "white").save(src / f"{i}.png")
    (src / "labels.txt").write_text("0.png\tCăn cứ Luật số 12\n1.png\t中中中中\n2.png\t" + "x" * 300 + "\n3.png\tĐiều 5\n", encoding="utf-8")
    stats = public_sets.import_rec_tsv(tmp_path / "data", "ext", src / "labels.txt", val_ratio=0.0)
    assert stats == {"train": 2, "val": 0, "skipped": 2}


def test_score_separates_tone_errors():
    r = evaluate.score([("Điều 12 quy định", "Điều 12 quy định"), ("Điều 12 quy định", "Điều 12 quy đinh"), ("Công ty", "Cong ty")])
    assert r.lines == 3 and r.exact == 1
    assert r.errors > r.errors_ignoring_tones == 1  # "Công"->"Cong" is a circumflex change (not a tone); định->đinh is a tone
    assert 0 < r.tone_cer < r.cer


def test_merge_repeat_only_train(tmp_path):
    _shard(tmp_path, "rec", "a", ["a/images/1.jpg\tx"], ["a/images/1.jpg"])
    (tmp_path / "rec/a/val.txt").write_text("a/images/1.jpg\tx\n", encoding="utf-8")
    _shard(tmp_path, "rec", "b", ["b/images/1.jpg\ty"], ["b/images/1.jpg"])
    counts = dataset.merge(tmp_path, "rec", repeat={"b": 3})
    assert counts["train"] == 4 and counts["val"] == 1


def test_write_wh_list(tmp_path):
    _shard(tmp_path, "rec", "a", ["a/images/1.jpg\tĐiều 1", "a/images/2.jpg\tkhoản 2"], ["a/images/1.jpg", "a/images/2.jpg"])
    dataset.merge(tmp_path, "rec")
    out = dataset.write_wh_list(tmp_path)
    rows = [l.split("\t") for l in out.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 and all(r[2:] == ["300", "48"] for r in rows) and {r[1] for r in rows} == {"Điều 1", "khoản 2"}
