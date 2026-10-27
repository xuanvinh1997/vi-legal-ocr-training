"""VietOCR (VGG19-BN + Transformer) fine-tuning, the part that needs no torch: vocabulary, config, the plan for carrying the
pretrained weights over to a bigger vocabulary, and the width-bucketed batches. `vietocr_train` has the model code.

Same problem as the PaddleOCR recogniser, same fix. VietOCR's vocabulary (229 characters) has no `§ № ₫ ‘ ’ “ ” – — …` or the Latin-1 letters
(foreign names) that legal text contains, and the decoder can only emit what it has an id for. So the vocabulary is VietOCR's plus every character
of our dictionary that it lacks, appended: the ids of the old characters stay the same, and `expansion_plan` says where each new id starts from.

VietOCR feeds the Transformer a flat sequence of (feature-map height x width/4) positions, so a batch must have a single width (there is no
padding mask on the encoder). Hence the bucketing: lines of equal resized width share a batch, exactly as VietOCR's own sampler does.
"""
from __future__ import annotations

import random
import unicodedata
from collections import defaultdict
from pathlib import Path

import yaml

from .charset import VI_TONES

#: VietOCR's `<pad> <sos> <eos> <mask>` take ids 0-3; the characters of the vocabulary follow
SPECIAL = 4
PRETRAIN_URL = "https://vocr.vn/data/vietocr/vgg_transformer.pth"
#: base.yml of VietOCR 0.3.13, the vocabulary `vgg_transformer.pth` was trained with
DEFAULT_VOCAB = 'aAàÀảẢãÃáÁạẠăĂằẰẳẲẵẴắẮặẶâÂầẦẩẨẫẪấẤậẬbBcCdDđĐeEèÈẻẺẽẼéÉẹẸêÊềỀểỂễỄếẾệỆfFgGhHiIìÌỉỈĩĨíÍịỊjJkKlLmMnNoOòÒỏỎõÕóÓọỌôÔồỒổỔỗỖốỐộỘơƠờỜởỞỡỠớỚợỢpPqQrRsStTuUùÙủỦũŨúÚụỤưƯừỪửỬữỮứỨựỰvVwWxXyYỳỲỷỶỹỸýÝỵỴzZ0123456789!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
#: the VGG19-BN backbone shrinks the height by 16 and the width by 4; the encoder sees height/16 * width/4 positions
HEIGHT_STRIDE, WIDTH_STRIDE = 16, 4
MAX_SEQ_LENGTH = 1024  # the positional encoding's size, for the encoder (positions) and the decoder (characters) alike

Sample = tuple[str, str, "int | None", "int | None"]  # image path (relative), text, width, height


def build_vocab(dictionary: list[str], base: str = DEFAULT_VOCAB) -> str:
    """`base` plus the dictionary's characters that it lacks, appended (so no existing id moves)."""
    have = set(base)
    return base + "".join(c for c in dictionary if c not in have and c != " ")


def strip_tones(ch: str) -> str:
    return unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFD", ch) if c not in VI_TONES))


def strip_marks(ch: str) -> str:
    return unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFD", ch) if not unicodedata.combining(c)))


def expansion_plan(old_vocab: str, new_vocab: str) -> list[tuple[int | None, str]]:
    """Per id of the new vocabulary: the old id to copy its embedding and output row from, and how.
    `copied`: the character was there; `from_base`: its tone-stripped letter was (ế from ê); `random`: neither."""
    old_ids = {c: i + SPECIAL for i, c in enumerate(old_vocab)}
    plan: list[tuple[int | None, str]] = [(i, "copied") for i in range(SPECIAL)]
    for ch in new_vocab:
        if ch in old_ids:
            plan.append((old_ids[ch], "copied"))
            continue
        # ế -> ê (tone off), then Ä -> A (any mark off)
        base = next((b for b in (strip_tones(ch), strip_marks(ch)) if b != ch and b in old_ids), None)
        plan.append((old_ids[base], "from_base") if base else (None, "random"))
    return plan


def bucket_width(w: int, h: int, height: int, min_w: int, max_w: int) -> int:
    """The width VietOCR resizes a w x h line to (`vietocr.tool.translate.resize`): scaled to `height`, rounded up to 10, clamped."""
    new_w = -(-int(height * float(w) / float(h)) // 10) * 10
    return min(max(new_w, min_w), max_w)


def encoder_positions(height: int, width: int) -> int:
    return max(1, height // HEIGHT_STRIDE) * (width // WIDTH_STRIDE)


def max_width_for(height: int) -> int:
    """The widest image whose positions still fit the positional encoding."""
    return (MAX_SEQ_LENGTH // max(1, height // HEIGHT_STRIDE)) * WIDTH_STRIDE


def suggest_max_width(samples: list[Sample], height: int) -> int:
    """Widest line of the data at `height` (rounded up to 10), so no line is squeezed; never beyond what the positional encoding fits.
    A line wider than the limit is squashed horizontally, which the model reads badly."""
    widths = [bucket_width(w, h, height, 0, 10 ** 9) for _, _, w, h in samples if w and h]
    limit = max_width_for(height)
    return min(limit, max(widths)) if widths else limit


def read_list(path: str | Path) -> list[Sample]:
    """`rel<TAB>text`, or `rel<TAB>text<TAB>width<TAB>height` (the `*_wh.txt` PaddleOCR's width-aware sampler wants)."""
    out: list[Sample] = []
    for line in Path(path).read_text(encoding="utf-8").split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 4 and parts[-1].isdigit() and parts[-2].isdigit():
            out.append((parts[0], "\t".join(parts[1:-2]), int(parts[-2]), int(parts[-1])))
        else:
            out.append((parts[0], "\t".join(parts[1:]), None, None))
    return out


def filter_samples(samples: list[Sample], vocab: str, max_chars: int) -> tuple[list[Sample], dict[str, int]]:
    """Drop what VietOCR cannot learn from: an unknown character (a KeyError mid-epoch otherwise), a label longer than the decoder
    runs for, an empty label. Counted, never silent."""
    have = set(vocab)
    kept: list[Sample] = []
    dropped = {"empty": 0, "too_long": 0, "unknown_char": 0}
    for sample in samples:
        text = sample[1]
        if not text.strip():
            dropped["empty"] += 1
        elif len(text) > max_chars:
            dropped["too_long"] += 1
        elif not set(text) <= have:
            dropped["unknown_char"] += 1
        else:
            kept.append(sample)
    return kept, dropped


def make_batches(widths: list[int], batch_size: int, seed: int = 0, shuffle: bool = True) -> list[list[int]]:
    """Index batches in which every line has the same resized width. The last batch of a width is kept (shorter), not dropped: those
    are the rare long lines, which the model must not stop seeing."""
    groups: dict[int, list[int]] = defaultdict(list)
    for i, w in enumerate(widths):
        groups[w].append(i)
    rng = random.Random(seed)
    batches: list[list[int]] = []
    for _, indices in sorted(groups.items()):
        if shuffle:
            rng.shuffle(indices)
        batches += [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]
    if shuffle:
        rng.shuffle(batches)
    return batches


def build_config(*, vocab: str, data_root: str, train_list: str, val_list: str, save_dir: str, pretrained: str = PRETRAIN_URL,
                 pretrained_vocab: str = DEFAULT_VOCAB, epochs: int = 6, batch_size: int = 32, image_height: int = 32,
                 image_max_width: int = 1100, image_min_width: int = 32, max_lr: float = 3e-4 * 0.4, max_chars: int = 126,
                 workers: int = 4, valid_every: int = 2000, save_every: int = 2000, print_every: int = 100, metrics: int = 1500,
                 mask_prob: float = 0.05, device: str = "cuda:0", amp: bool = True) -> dict:
    """A VietOCR config (`vgg_transformer` + base.yml, written out so no download is needed) plus the keys the loop here reads.
    The VietOCR keys keep their names, so `Predictor(Cfg.load_config_from_file(...))` works on the exported copy."""
    if image_height < HEIGHT_STRIDE or image_height % HEIGHT_STRIDE:
        raise ValueError(f"image_height must be a multiple of {HEIGHT_STRIDE}, got {image_height}")
    if encoder_positions(image_height, image_max_width) > MAX_SEQ_LENGTH:
        raise ValueError(f"{image_height}x{image_max_width} gives {encoder_positions(image_height, image_max_width)} encoder positions, "
                         f"the positional encoding holds {MAX_SEQ_LENGTH} (max width at this height: {max_width_for(image_height)})")
    if max_chars + 2 > MAX_SEQ_LENGTH:
        raise ValueError("max_chars does not fit the decoder")
    pooling = [[2, 2], [2, 2], [2, 1], [2, 1], [1, 1]]
    return {
        "vocab": vocab, "device": device, "seq_modeling": "transformer", "backbone": "vgg19_bn",
        # the ImageNet weights are replaced by the checkpoint right after, so do not download 550 MB for nothing
        "cnn": {"pretrained": False, "ss": pooling, "ks": pooling, "hidden": 256},
        "transformer": {"d_model": 256, "nhead": 8, "num_encoder_layers": 6, "num_decoder_layers": 6, "dim_feedforward": 2048,
                        "max_seq_length": MAX_SEQ_LENGTH, "pos_dropout": 0.1, "trans_dropout": 0.1},
        "optimizer": {"max_lr": max_lr, "pct_start": 0.1},
        "trainer": {"epochs": epochs, "batch_size": batch_size, "print_every": print_every, "valid_every": valid_every, "save_every": save_every,
                    "metrics": metrics, "save_dir": save_dir, "amp": amp, "max_chars": max_chars},
        "dataset": {"data_root": data_root, "train_list": train_list, "val_list": val_list, "image_height": image_height,
                    "image_min_width": image_min_width, "image_max_width": image_max_width},
        "dataloader": {"num_workers": workers, "pin_memory": True},
        "aug": {"mask_prob": mask_prob},
        "predictor": {"beamsearch": False},
        "pretrain": pretrained, "pretrain_vocab": pretrained_vocab,
        "weights": str(Path(save_dir) / "best.pth"),
        "quiet": False,
    }


def write_config(cfg: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False, default_flow_style=None, width=10 ** 9), encoding="utf-8")
    return path


def read_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))
