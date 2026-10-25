import os
from pathlib import Path

import numpy as np
import pytest
import yaml

from ocr_train import charset, head_surgery, paddle_cfg

UPSTREAM = os.environ.get("OCR_TEST_PADDLEOCR")  # a PaddleOCR checkout
PDPARAMS = os.environ.get("OCR_TEST_PDPARAMS")  # PP-OCRv6_small_rec_pretrained.pdparams
needs_repo = pytest.mark.skipif(not UPSTREAM, reason="set OCR_TEST_PADDLEOCR to a PaddleOCR checkout")


@needs_repo
@pytest.mark.parametrize("size", ["small", "tiny"])
def test_rec_config(tmp_path, size):
    src = Path(UPSTREAM) / f"configs/rec/PP-OCRv6/PP-OCRv6_{size}_rec.yml"
    out = paddle_cfg.build_rec_config(src, tmp_path / "rec.yml", data_root="/d", train_list="/d/t.txt", val_list="/d/v.txt", dict_path="/d/vi.txt",
                                      max_text_length=110, img_w=1280, epochs=7, batch_size=48, save_dir="/o", pretrained="/p.pdparams")
    cfg = yaml.safe_load(out.read_text())
    assert cfg["Global"]["max_text_length"] == 110 and cfg["Global"]["character_dict_path"] == "/d/vi.txt"
    nrtr = [h["NRTRHead"] for h in cfg["Architecture"]["Head"]["head_list"] if "NRTRHead" in h]
    assert nrtr and nrtr[0]["max_text_length"] == 110
    assert not any("RecConAug" in t for t in cfg["Train"]["dataset"]["transforms"])
    assert cfg["Train"]["sampler"]["scales"][1] == [1280, 48] and cfg["Train"]["loader"]["batch_size_per_card"] == 48
    assert cfg["Eval"]["dataset"]["transforms"][2]["RecResizeImg"]["image_shape"] == [3, 48, 1280]
    assert cfg["Optimizer"]["lr"]["learning_rate"] < 0.0005


@needs_repo
@pytest.mark.parametrize("size", ["small", "tiny"])
def test_det_config(tmp_path, size):
    src = Path(UPSTREAM) / f"configs/det/PP-OCRv6/PP-OCRv6_{size}_det.yml"
    out = paddle_cfg.build_det_config(src, tmp_path / "det.yml", data_root="/d", train_list="/d/t.txt", val_list="/d/v.txt", epochs=9, batch_size=6, save_dir="/o")
    cfg = yaml.safe_load(out.read_text())
    assert cfg["Global"]["epoch_num"] == 9 and cfg["Train"]["dataset"]["label_file_list"] == ["/d/t.txt"]
    tf = {k: v for t in cfg["Train"]["dataset"]["transforms"] if t for k, v in (t.items() if isinstance(t, dict) else [])}
    assert tf["MakeShrinkMap"]["total_epoch"] == 9 and tf["RandomCrop"]["size"] == [640, 640]


def test_remap_head_synthetic():
    old = ["a", "b", "ê", "c"]
    new = ["a", "ê", "ế", "ạ", "đ", "c"]  # b dropped; ế, ạ come from ê / a; đ is new
    rng = np.random.default_rng(0)
    old_c = len(old) + 2
    params = {"backbone.w": rng.normal(size=(3, 3)).astype("float32"),
              "head.ctc_head.fc.weight": rng.normal(size=(8, old_c)).astype("float32"),
              "head.ctc_head.fc.bias": rng.normal(size=(old_c,)).astype("float32")}
    out, stats = head_surgery.remap_ctc_head(params, old, new)
    w_old, w_new = params["head.ctc_head.fc.weight"], out["head.ctc_head.fc.weight"]
    assert w_new.shape == (8, len(new) + 2)
    assert np.array_equal(w_new[:, 0], w_old[:, 0])  # blank
    assert np.array_equal(w_new[:, 1], w_old[:, 1])  # a
    assert np.array_equal(w_new[:, 2], w_old[:, 3])  # ê
    assert np.array_equal(w_new[:, -2], w_old[:, 4])  # c
    assert np.array_equal(w_new[:, -1], w_old[:, -1])  # space stays last
    assert np.abs(w_new[:, 3] - w_old[:, 3]).max() < 1.0 and not np.array_equal(w_new[:, 3], w_old[:, 3])  # ế ~ ê
    assert stats["copied"] == 5 and stats["from_base"] == 2 and stats["random"] == 1
    assert np.array_equal(out["backbone.w"], params["backbone.w"])
    with pytest.raises(ValueError):
        head_surgery.remap_ctc_head(params, old + ["x"], new)


@pytest.mark.skipif(not PDPARAMS or not UPSTREAM, reason="needs real pretrained weights and checkout")
def test_remap_head_real_checkpoint(tmp_path):
    old = charset.read_dict(Path(UPSTREAM) / "ppocr/utils/dict/ppocrv6_dict.txt")
    new = charset.build_charset()
    params = head_surgery.load_pdparams(PDPARAMS)
    out, stats = head_surgery.remap_ctc_head(params, old, new)
    assert out["head.ctc_head.fc.weight"].shape == (120, len(new) + 2)
    assert stats["copied"] > 200 and stats["from_base"] >= 80
    head_surgery.save_pdparams(out, tmp_path / "x.pdparams")
    again = head_surgery.load_pdparams(tmp_path / "x.pdparams")
    assert set(again) == set(params)


@needs_repo
def test_rec_config_width_aware(tmp_path):
    src = Path(UPSTREAM) / "configs/rec/PP-OCRv6/PP-OCRv6_small_rec.yml"
    out = paddle_cfg.build_rec_config(src, tmp_path / "rec.yml", data_root="/d", train_list="/d/t_wh.txt", val_list="/d/v.txt", dict_path="/d/vi.txt",
                                      max_text_length=110, img_w=1600, epochs=2, batch_size=32, save_dir="/o", ds_width=True)
    cfg = yaml.safe_load(out.read_text())
    assert not any("RecAug" in x for x in cfg["Train"]["dataset"]["transforms"])
    assert cfg["Train"]["dataset"]["ds_width"] is True and cfg["Train"]["sampler"]["max_w"] == 1600
    assert cfg["Train"]["dataset"]["label_file_list"] == ["/d/t_wh.txt"]


def test_check_init_rejects_another_dictionary():
    dictionary = ["a", "ê", "ế", "c"]
    n = len(dictionary) + 2
    params = {"head.ctc_head.fc.weight": np.zeros((8, n), "float32"), "head.ctc_head.fc.bias": np.zeros((n,), "float32")}
    assert head_surgery.check_init(params, dictionary) == n
    with pytest.raises(ValueError):
        head_surgery.check_init(params, dictionary + ["ạ"])


def _config_rec(tmp_path, monkeypatch, meta_len, like_len, extra=()):
    from ocr_train import cli
    data = tmp_path / "data"
    (data / "rec").mkdir(parents=True)
    (data / "rec" / "meta.json").write_text(f'{{"max_text_length": {meta_len}, "img_w": 1600}}')
    (tmp_path / "prev.yml").write_text(yaml.safe_dump({"Global": {"max_text_length": like_len}, "Train": {"sampler": {"scales": [[1280, 32], [1280, 48], [1280, 64]]}}}))
    seen = {}
    monkeypatch.setattr(paddle_cfg, "build_rec_config", lambda *a, **kw: seen.update(kw) or tmp_path / "o.yml")
    cli.main(["--data", str(data), "config-rec", "--paddleocr", str(tmp_path), "--dict", str(tmp_path / "d.txt"), "--out", str(tmp_path / "o.yml"),
              "--save-dir", str(tmp_path / "s"), "--like", str(tmp_path / "prev.yml"), *extra])
    return seen


def test_config_rec_like_keeps_the_checkpoints_shapes(tmp_path, monkeypatch):
    seen = _config_rec(tmp_path, monkeypatch, meta_len=104, like_len=104)
    assert seen["max_text_length"] == 104 and seen["img_w"] == 1280  # width from the earlier config, not the data's 1600


def test_config_rec_like_refuses_labels_that_would_be_dropped(tmp_path, monkeypatch):
    with pytest.raises(SystemExit, match="max_text_length=104"):
        _config_rec(tmp_path, monkeypatch, meta_len=130, like_len=104)
