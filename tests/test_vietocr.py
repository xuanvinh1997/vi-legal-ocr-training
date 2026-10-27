import os
from collections import Counter

import pytest

from ocr_train import charset, vietocr_cfg as vc


def test_default_vocab_is_vietocrs():
    assert len(vc.DEFAULT_VOCAB) == 229 and len(set(vc.DEFAULT_VOCAB)) == 229
    assert vc.DEFAULT_VOCAB.endswith(" ") and "ế" in vc.DEFAULT_VOCAB and "§" not in vc.DEFAULT_VOCAB


def test_vocab_keeps_old_ids_and_adds_only_what_is_missing():
    chars = charset.build_charset()
    vocab = vc.build_vocab(chars)
    assert vocab.startswith(vc.DEFAULT_VOCAB) and len(set(vocab)) == len(vocab)
    assert set(chars) <= set(vocab)
    assert vc.build_vocab(list(vc.DEFAULT_VOCAB)) == vc.DEFAULT_VOCAB  # nothing missing, nothing added


def test_expansion_plan():
    old = "aêeAÂ "
    new = old + "ếạ§Ä"
    plan = vc.expansion_plan(old, new)
    assert plan[:vc.SPECIAL] == [(i, "copied") for i in range(vc.SPECIAL)]
    assert plan[vc.SPECIAL:vc.SPECIAL + len(old)] == [(vc.SPECIAL + i, "copied") for i in range(len(old))]
    added = dict(zip(new[len(old):], plan[vc.SPECIAL + len(old):]))
    assert added["ế"] == (vc.SPECIAL + old.index("ê"), "from_base")  # tone off: ê, not e
    assert added["ạ"] == (vc.SPECIAL + old.index("a"), "from_base")
    assert added["Ä"] == (vc.SPECIAL + old.index("A"), "from_base")  # any mark off
    assert added["§"] == (None, "random")


def test_expansion_plan_on_the_real_vocabulary():
    plan = vc.expansion_plan(vc.DEFAULT_VOCAB, vc.build_vocab(charset.build_charset()))
    assert Counter(how for _, how in plan)["copied"] == len(vc.DEFAULT_VOCAB) + vc.SPECIAL


def test_bucket_width_matches_vietocr():
    resize = pytest.importorskip("vietocr.tool.translate").resize
    for w, h in [(10, 48), (500, 48), (1600, 48), (333, 31), (97, 30)]:
        for height, lo, hi in [(32, 32, 512), (32, 32, 1100), (48, 32, 2048)]:
            assert vc.bucket_width(w, h, height, lo, hi) == resize(w, h, height, lo, hi)[0]


def test_encoder_positions_and_width_limit():
    assert vc.encoder_positions(32, 1100) == 2 * 275
    assert vc.encoder_positions(32, vc.max_width_for(32)) == vc.MAX_SEQ_LENGTH
    assert vc.encoder_positions(32, vc.max_width_for(32) + 4) > vc.MAX_SEQ_LENGTH
    assert vc.max_width_for(48) == 341 * 4


def test_suggest_max_width_is_the_widest_line_unless_it_cannot_fit():
    samples = [("a.png", "x", 480, 48), ("b.png", "y", 1500, 48), ("c.png", "z", 100, 48)]
    assert vc.suggest_max_width(samples, 32) == 1000
    assert vc.suggest_max_width([("w.png", "x", 90000, 48)], 32) == vc.max_width_for(32)
    assert vc.suggest_max_width([("n.png", "x", None, None)], 32) == vc.max_width_for(32)


def test_read_list_with_and_without_sizes(tmp_path):
    f = tmp_path / "l.txt"
    f.write_text("a/1.png\tHà Nội\nb/2.png\tĐiều 5\t640\t48\n\n", encoding="utf-8")
    assert vc.read_list(f) == [("a/1.png", "Hà Nội", None, None), ("b/2.png", "Điều 5", 640, 48)]
    f.write_text("a.png\t2024\n", encoding="utf-8")  # a label that ends in digits is not a size
    assert vc.read_list(f) == [("a.png", "2024", None, None)]


def test_filter_samples_counts_what_it_drops():
    samples = [("a", "Hà Nội", 1, 1), ("b", "  ", 1, 1), ("c", "x" * 200, 1, 1), ("d", "日本", 1, 1), ("e", "§ 5", 1, 1)]
    kept, dropped = vc.filter_samples(samples, vc.build_vocab(charset.build_charset()), 126)
    assert [s[0] for s in kept] == ["a", "e"] and dropped == {"empty": 1, "too_long": 1, "unknown_char": 1}


def test_batches_share_a_width_and_cover_every_line_once():
    widths = [100, 100, 100, 100, 100, 250, 250, 900] * 5
    batches = vc.make_batches(widths, 4, seed=3)
    assert sorted(i for b in batches for i in b) == list(range(len(widths)))
    assert all(len({widths[i] for i in b}) == 1 and len(b) <= 4 for b in batches)
    assert any(len(b) < 4 for b in batches), "short batches must be kept"
    assert batches == vc.make_batches(widths, 4, seed=3) and batches != vc.make_batches(widths, 4, seed=4)


def _cfg(**kw):
    args = dict(vocab=vc.DEFAULT_VOCAB, data_root="/d/rec", train_list="train_list_wh.txt", val_list="val_list.txt", save_dir="/o/vietocr")
    return vc.build_config(**{**args, **kw})


def test_config_round_trip_and_values(tmp_path):
    cfg = _cfg(epochs=3, batch_size=16, max_lr=1e-4, image_max_width=1100)
    assert vc.read_config(vc.write_config(cfg, tmp_path / "sub" / "c.yml")) == cfg
    assert cfg["trainer"]["epochs"] == 3 and cfg["optimizer"]["max_lr"] == 1e-4 and cfg["cnn"]["pretrained"] is False
    assert cfg["weights"].replace("\\", "/") == "/o/vietocr/best.pth" and cfg["pretrain_vocab"] == vc.DEFAULT_VOCAB


@pytest.mark.parametrize("kw", [dict(image_height=30), dict(image_height=8), dict(image_max_width=2100), dict(max_chars=1100)])
def test_config_rejects_what_the_model_cannot_take(kw):
    with pytest.raises(ValueError):
        _cfg(**kw)


# ---- the parts that need torch + vietocr -------------------------------------------------------------------------------

torch = pytest.importorskip("torch")
pytest.importorskip("vietocr")
from ocr_train import vietocr_train as vt  # noqa: E402


def _state(vocab_len, d=8):
    g = torch.Generator().manual_seed(0)
    n = vocab_len + vc.SPECIAL
    return {"transformer.embed_tgt.weight": torch.randn(n, d, generator=g), "transformer.fc.weight": torch.randn(n, d, generator=g),
            "transformer.fc.bias": torch.randn(n, generator=g), "other.weight": torch.randn(3, 3, generator=g)}


def test_expand_state_copies_known_rows_and_seeds_new_ones_from_their_base():
    old, new = "aêe ", "aêe ếz"
    state = _state(len(old))
    out = vt.expand_state(state, old, new)
    for key in vt.VOCAB_KEYS:
        assert out[key].shape[0] == len(new) + vc.SPECIAL
        assert torch.equal(out[key][:len(old) + vc.SPECIAL], state[key])  # nothing that existed moved
    e_hat, e_acute = vc.SPECIAL + old.index("ê"), vc.SPECIAL + new.index("ế")
    assert torch.allclose(out["transformer.fc.bias"][e_acute], state["transformer.fc.bias"][e_hat] - 1.0)
    assert (out["transformer.fc.weight"][e_acute] - state["transformer.fc.weight"][e_hat]).abs().max() < 1.0  # close, not equal
    assert not torch.equal(out["transformer.fc.weight"][e_acute], state["transformer.fc.weight"][e_hat])
    assert out["other.weight"] is state["other.weight"]
    assert torch.equal(vt.expand_state(state, old, new)["transformer.fc.weight"], out["transformer.fc.weight"])  # deterministic


def test_expand_state_refuses_weights_of_another_vocabulary():
    with pytest.raises(ValueError, match="another vocabulary"):
        vt.expand_state(_state(10), "abc", "abcd")


def test_collate_follows_vietocr_conventions():
    from vietocr.model.vocab import Vocab

    vocab = Vocab(vc.DEFAULT_VOCAB)
    batch = vt.collate([{"img": __import__("numpy").zeros((3, 32, 40)), "word": vocab.encode(t)} for t in ("ab", "abcd")])
    tgt_in, tgt_out, pad = batch["tgt_input"], batch["tgt_output"], batch["tgt_padding_mask"]
    assert tgt_in.shape == (6, 2) and tgt_out.shape == (2, 6) and pad.shape == (2, 6) and batch["img"].shape == (2, 3, 32, 40)
    assert tgt_in[0].tolist() == [vocab.go, vocab.go] and tgt_in[3, 0] == vocab.eos
    assert tgt_out[0].tolist() == [vocab.c2i["a"], vocab.c2i["b"], vocab.eos, 0, 0, 0]  # input shifted left, <eos> predicted last
    assert pad[0].tolist() == [False, False, False, True, True, True]  # <sos> a b are attended; everything from <eos> on is masked
    masked = vt.collate([{"img": __import__("numpy").zeros((3, 32, 40)), "word": vocab.encode("abcdefgh" * 4)}], mask_prob=1.0)["tgt_input"]
    assert masked[0, 0] == vocab.go and masked[-1, 0] == vocab.eos and (masked[1:-1] == vocab.mask_token).all()


@pytest.mark.skipif(not os.environ.get("OCR_TEST_VIETOCR"), reason="trains a small VietOCR on the CPU; set OCR_TEST_VIETOCR=1")
def test_train_resume_evaluate_export(tmp_path):
    """A pretrained checkpoint on VietOCR's vocabulary is widened, trained for a few steps, interrupted, resumed, scored and exported."""
    import numpy as np
    from PIL import Image, ImageDraw
    from vietocr.tool.translate import build_model

    rec = tmp_path / "rec"
    (rec / "s").mkdir(parents=True)
    words = ["điều", "khoản", "§", "số", "ế", "2024"]
    lines = []
    for i in range(24):
        text = " ".join(words[(i + k) % len(words)] for k in range(1 + i % 3))
        im = Image.new("L", (40 + 60 * len(text), 48), 255)
        ImageDraw.Draw(im).text((4, 12), text, fill=0)
        im.save(rec / "s" / f"{i}.png")
        lines.append(f"s/{i}.png\t{text}")
    (rec / "train_list.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (rec / "val_list.txt").write_text("\n".join(lines[:8]) + "\n", encoding="utf-8")

    cfg = vc.build_config(vocab=vc.build_vocab(charset.build_charset()), data_root=str(rec), train_list="train_list.txt", val_list="val_list.txt",
                          save_dir=str(tmp_path / "run"), pretrained=str(tmp_path / "stock.pth"), epochs=2, batch_size=4, workers=0,
                          valid_every=3, save_every=2, print_every=1, metrics=8, device="cpu", amp=False, image_max_width=600)
    cfg["transformer"].update(num_encoder_layers=1, num_decoder_layers=1, dim_feedforward=64)  # keep the CPU run short
    stock = dict(cfg, vocab=vc.DEFAULT_VOCAB)
    torch.save(build_model(stock)[0].state_dict(), tmp_path / "stock.pth")

    first = vt.train(cfg)
    assert first["steps"] > 0 and (tmp_path / "run" / "best.pth").exists() and (tmp_path / "run" / "latest.pt").exists()
    with pytest.raises(SystemExit, match="already holds a run"):
        vt.train(cfg)
    again = vt.train(cfg, resume=True)  # finished run: resumes at the last step, nothing left to do
    assert again["steps"] == first["steps"]

    report = vt.evaluate_weights(cfg, str(tmp_path / "stock.pth"), "val_list.txt", limit=8)  # the stock vocabulary scores lines it cannot spell
    assert report.lines == 8
    assert vt.evaluate_weights(cfg, cfg["weights"], "val_list.txt", limit=8).lines == 8
    served = vt.export(cfg, tmp_path / "x")
    assert (tmp_path / "x" / "weights.pth").exists() and vc.read_config(served)["weights"] == "weights.pth"
    assert np.isfinite(first["best_cer"])
