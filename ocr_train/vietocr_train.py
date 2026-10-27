"""VietOCR fine-tuning loop. It uses VietOCR's model, vocabulary, resize and greedy decoder, so the exported weights load into
`vietocr.tool.predictor.Predictor` unchanged, but not `vietocr.model.trainer.Trainer`, which
  * imports `imgaug` at module level (broken on the numpy 2 that Colab ships),
  * never writes a checkpoint (`load_checkpoint` exists, `save_checkpoint` is never called) and its resume constructs an optimiser that raises,
  * rebuilds an LMDB cache only when its directory is missing, so a longer annotation file is silently ignored,
  * keeps the best model by whole-line accuracy, which says little about a Vietnamese line that is one tone mark off.
Here: images are read from the PaddleOCR-layout lists directly, the best model is the lowest CER, and `latest.pt` makes a Colab
disconnect harmless.
"""
from __future__ import annotations

import contextlib
import copy
import functools
import os
import random
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from vietocr.model.vocab import Vocab
from vietocr.optim.labelsmoothingloss import LabelSmoothingLoss
from vietocr.tool.translate import build_model, process_image, translate
from vietocr.tool.utils import download_weights

from . import vietocr_cfg as vc
from .evaluate import RecReport, score

#: the tensors whose first dimension is the vocabulary
VOCAB_KEYS = ("transformer.embed_tgt.weight", "transformer.fc.weight", "transformer.fc.bias")


def expand_state(state: dict, old_vocab: str, new_vocab: str, seed: int = 0) -> dict:
    """Pretrained weights for a larger vocabulary: rows of known characters are copied, a new letter starts from its base letter plus
    noise (and a lower output bias, so it does not outvote the letter it came from), anything else is random. Everything else is untouched."""
    old_n = len(old_vocab) + vc.SPECIAL
    for key in VOCAB_KEYS:
        if key not in state or state[key].shape[0] != old_n:
            have = state[key].shape[0] if key in state else None
            raise ValueError(f"{key} has {have} rows, a vocabulary of {len(old_vocab)} characters needs {old_n}: "
                             f"the weights were trained with another vocabulary")
    if old_vocab == new_vocab:
        return state
    plan = vc.expansion_plan(old_vocab, new_vocab)
    gen = torch.Generator().manual_seed(seed)
    out = dict(state)
    for key in VOCAB_KEYS:
        old = state[key].float()
        scale = float(old.std()) if old.numel() > 1 else 0.02
        new = torch.empty((len(plan),) + tuple(old.shape[1:]))
        for j, (src, how) in enumerate(plan):
            if key.endswith("bias"):
                new[j] = old[src] - 1.0 if how == "from_base" else old[src] if how == "copied" else old.min()
            elif how == "copied":
                new[j] = old[src]
            else:
                noise = torch.randn(old.shape[1:], generator=gen)
                new[j] = old[src] + noise * 0.1 * scale if how == "from_base" else noise * scale
        out[key] = new.to(state[key].dtype)
    return out


def load_state(uri: str) -> dict:
    """A weights file or URL as a state dict. VietOCR caches a download in the temp directory under its file name and never checks it, so a
    download cut short by a disconnect would be reused for ever: an unreadable cached file is deleted and the run stops with the way out."""
    path = download_weights(uri, quiet=True)
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        if uri.startswith("http"):
            Path(path).unlink(missing_ok=True)
        raise SystemExit(f"{path} is not a readable weights file ({error}); "
                         f"{'it was deleted, run again to download it afresh' if uri.startswith('http') else 'check the path'}") from error


def load_pretrained(model: torch.nn.Module, cfg: dict) -> None:
    """Weights from `cfg["pretrain"]` (URL or file), widened to `cfg["vocab"]`. Missing or unexpected tensors are an error: loading nothing
    and training a random network under a pretrained name is the failure to rule out."""
    state = expand_state(load_state(cfg["pretrain"]), cfg["pretrain_vocab"], cfg["vocab"])
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys or result.unexpected_keys:
        raise SystemExit(f"{cfg['pretrain']} does not fit the model: missing {result.missing_keys[:5]}, unexpected {result.unexpected_keys[:5]}")


class Lines(Dataset):
    """Line crops from a PaddleOCR-style list, resized the way VietOCR does. `vocab=None` skips the labels (evaluating a model whose
    vocabulary does not cover them all)."""

    def __init__(self, root: str | Path, samples: list[vc.Sample], vocab: Vocab | None, ds: dict):
        self.root, self.samples, self.vocab = Path(root), samples, vocab
        self.height, self.min_w, self.max_w = ds["image_height"], ds["image_min_width"], ds["image_max_width"]
        self.widths = [vc.bucket_width(w, h, self.height, self.min_w, self.max_w) for _, _, w, h in samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        rel, text, _, _ = self.samples[i]
        with Image.open(self.root / rel) as image:
            img = process_image(image, self.height, self.min_w, self.max_w)
        return {"img": img, "word": self.vocab.encode(text) if self.vocab else None}


def collate(batch: list[dict], mask_prob: float = 0.0) -> dict:
    """VietOCR's `Collator`: `tgt_input` is (T, N) = <sos> text <eos> <pad>..., `tgt_output` the same shifted left, and the padding mask
    covers everything from <eos> on. `mask_prob` replaces that share of the input characters by the mask token (VietOCR's masked-LM trick)."""
    img = np.stack([b["img"] for b in batch]).astype(np.float32)
    lens = np.array([len(b["word"]) for b in batch])
    tgt = np.zeros((len(batch), lens.max()), dtype=np.int64)
    for i, b in enumerate(batch):
        tgt[i, :lens[i]] = b["word"]
    tgt_input = np.ascontiguousarray(tgt.T)
    tgt_output = np.ascontiguousarray(np.roll(tgt_input, -1, 0).T)
    tgt_output[:, -1] = 0
    padding = np.arange(tgt.shape[1])[None, :] >= (lens - 1)[:, None]
    tgt_input = torch.from_numpy(tgt_input)
    if mask_prob > 0:  # torch's RNG, which DataLoader seeds per worker (numpy's would repeat across workers)
        hide = (torch.rand(tgt_input.shape) < mask_prob) & (tgt_input > 2)
        tgt_input = tgt_input.masked_fill(hide, 3)
    return {"img": torch.from_numpy(img), "tgt_input": tgt_input, "tgt_output": torch.from_numpy(tgt_output),
            "tgt_padding_mask": torch.from_numpy(padding)}


def load_samples(cfg: dict, list_name: str, vocab: str | None = None) -> tuple[list[vc.Sample], dict[str, int]]:
    """Samples of a list with their image sizes (read from the file when the list carries them, else from the image headers).
    With `vocab`, lines it cannot represent are dropped and counted."""
    root = Path(cfg["dataset"]["data_root"])
    samples = vc.read_list(root / list_name)
    if vocab is not None:
        samples, dropped = vc.filter_samples(samples, vocab, cfg["trainer"]["max_chars"])
    else:
        dropped = {"empty": sum(not s[1].strip() for s in samples)}
        samples = [s for s in samples if s[1].strip()]
    sized: list[vc.Sample] = []
    dropped["unreadable"] = 0
    for rel, text, w, h in samples:
        if not (w and h):
            try:
                with Image.open(root / rel) as image:
                    w, h = image.size
            except OSError:
                dropped["unreadable"] += 1
                continue
        sized.append((rel, text, w, h))
    if not sized:
        raise SystemExit(f"{root / list_name}: no usable line ({dropped})")
    return sized, dropped


def pick_device(cfg: dict) -> torch.device:
    want = cfg.get("device", "cuda:0")
    if str(want).startswith("cuda") and not torch.cuda.is_available():
        print(f"no CUDA device: running {want!r} on the CPU")
        want = "cpu"
    return torch.device(want)


def amp_dtype(device: torch.device, enabled: bool) -> torch.dtype | None:
    """bfloat16 where the GPU has it (L4, A100), else float16 with loss scaling (T4). The CPU trains in float32."""
    if not enabled or device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast(device: torch.device, dtype: torch.dtype | None):
    return torch.autocast(device.type, dtype=dtype) if dtype else contextlib.nullcontext()


@torch.no_grad()
def evaluate_model(model, vocab: Vocab, dataset: Lines, device: torch.device, dtype: torch.dtype | None, max_len: int,
                   batch_size: int = 64) -> tuple[RecReport, float | None]:
    """CER / tone report of greedy decoding over `dataset`, and the validation loss when it has labels."""
    model.eval()
    criterion = LabelSmoothingLoss(len(vocab), padding_idx=vocab.pad, smoothing=0.1)
    pairs: list[tuple[str, str]] = []
    losses: list[float] = []
    for indices in vc.make_batches(dataset.widths, batch_size, shuffle=False):
        items = [dataset[i] for i in indices]
        img = torch.from_numpy(np.stack([it["img"] for it in items]).astype(np.float32)).to(device)
        if dataset.vocab is not None:
            batch = collate(items)
            with autocast(device, dtype):
                out = model(img, batch["tgt_input"].to(device), tgt_key_padding_mask=batch["tgt_padding_mask"].to(device))
            losses.append(criterion(out.float().reshape(-1, out.size(-1)), batch["tgt_output"].to(device).reshape(-1)).item())
        with autocast(device, dtype):
            sentences, _ = translate(img, model, max_seq_length=max_len)
        pairs += [(dataset.samples[i][1], hyp) for i, hyp in zip(indices, vocab.batch_decode(sentences.tolist()))]
    return score(pairs), (float(np.mean(losses)) if losses else None)


def subset(samples: list[vc.Sample], n: int, seed: int = 0) -> list[vc.Sample]:
    """The same `n` validation lines every time, so two checkpoints (or two models) are scored on identical text."""
    picked = list(samples)
    random.Random(seed).shuffle(picked)
    return picked[:n] if n else picked


def _save(state: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, tmp)
    os.replace(tmp, path)  # a disconnect mid-write must not leave a half checkpoint where the good one was


def train(cfg: dict, resume: bool = False) -> dict:
    t, d = cfg["trainer"], cfg["dataset"]
    save_dir = Path(t["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)
    latest = save_dir / "latest.pt"
    if latest.exists() and not resume:
        raise SystemExit(f"{save_dir} already holds a run: resume it (RESUME=1 / --resume) or pick another save directory")
    cfg = copy.deepcopy(cfg)
    device = pick_device(cfg)
    cfg["device"] = str(device)
    dtype = amp_dtype(device, t["amp"])
    torch.manual_seed(0)
    log_file = (save_dir / "train.log").open("a", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    model, vocab = build_model(cfg)
    train_samples, dropped = load_samples(cfg, d["train_list"], cfg["vocab"])
    val_samples, val_dropped = load_samples(cfg, d["val_list"], cfg["vocab"])
    train_ds = Lines(d["data_root"], train_samples, vocab, d)
    val_ds = Lines(d["data_root"], subset(val_samples, t["metrics"]), vocab, d)
    steps_per_epoch = len(vc.make_batches(train_ds.widths, t["batch_size"], 0))
    total = t["epochs"] * steps_per_epoch
    log(f"train {len(train_ds)} lines (dropped {dropped}), validation {len(val_ds)} of {len(val_samples)} (dropped {val_dropped}); "
        f"{steps_per_epoch} steps/epoch x {t['epochs']} = {total}; device {device}, amp {dtype}; vocab {len(vocab)} ids")

    optimizer = AdamW(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    step, best = 0, float("inf")
    if resume and latest.exists():
        # our own file (pickled scheduler state), written by this function; never point this at a download
        ck = torch.load(latest, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scaler.load_state_dict(ck["scaler"])
        step, best, total = ck["step"], ck["best_cer"], ck["total"]  # the schedule of the interrupted run, not a new one
        scheduler = OneCycleLR(optimizer, total_steps=total, **cfg["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        log(f"resumed {latest} at step {step}/{total}, best CER {best:.2%}")
    else:
        load_pretrained(model, cfg)
        scheduler = OneCycleLR(optimizer, total_steps=total, **cfg["optimizer"])
    criterion = LabelSmoothingLoss(len(vocab), padding_idx=vocab.pad, smoothing=0.1)
    max_len = t["max_chars"] + 2

    def checkpoint() -> None:
        _save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
               "scaler": scaler.state_dict(), "step": step, "best_cer": best, "total": total}, latest)

    def validate() -> None:
        nonlocal best
        report, loss = evaluate_model(model, vocab, val_ds, device, dtype, max_len)
        model.train()
        log(f"step {step:07d} valid loss {loss:.3f}  {report.summary()}")
        if report.cer < best:
            best = report.cer
            _save(model.state_dict(), save_dir / "best.pth")
            log(f"  best CER {best:.2%} -> {save_dir / 'best.pth'}")

    model.train()
    epoch, skip, seen, running, tick = step // steps_per_epoch, step % steps_per_epoch, 0, 0.0, time.time()
    loader_args = dict(num_workers=cfg["dataloader"]["num_workers"], pin_memory=cfg["dataloader"]["pin_memory"] and device.type == "cuda",
                       collate_fn=functools.partial(collate, mask_prob=cfg["aug"]["mask_prob"]))
    try:
        while step < total:
            batches = vc.make_batches(train_ds.widths, t["batch_size"], seed=epoch)[skip:]  # a new order each epoch; after a resume, the rest of this one
            skip = 0
            for batch in DataLoader(train_ds, batch_sampler=batches, **loader_args):
                img, tgt_in, tgt_out, mask = (batch[k].to(device, non_blocking=True) for k in ("img", "tgt_input", "tgt_output", "tgt_padding_mask"))
                optimizer.zero_grad(set_to_none=True)
                with autocast(device, dtype):
                    out = model(img, tgt_in, tgt_key_padding_mask=mask)
                loss = criterion(out.float().reshape(-1, out.size(-1)), tgt_out.reshape(-1))
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                step += 1
                seen += img.size(0)
                running += loss.item()
                if step % t["print_every"] == 0:
                    log(f"step {step:07d}/{total} epoch {epoch} loss {running / t['print_every']:.3f} lr {scheduler.get_last_lr()[0]:.2e} "
                        f"{seen / (time.time() - tick):.0f} lines/s")
                    running, seen, tick = 0.0, 0, time.time()
                if step % t["valid_every"] == 0 or step == total:
                    validate()
                if step % t["save_every"] == 0 or step == total:
                    checkpoint()
                if step >= total:
                    break
            epoch += 1
    except KeyboardInterrupt:
        checkpoint()
        log(f"interrupted at step {step}; resume with --resume")
        raise
    log_file.close()
    return {"steps": step, "best_cer": best, "best": str(save_dir / "best.pth")}


def read_vocab_for(weights_state: dict, cfg: dict) -> str:
    """Which vocabulary a weights file belongs to, from its output layer: VietOCR's own, or the one in the config."""
    rows = weights_state["transformer.fc.bias"].shape[0]
    for vocab in (cfg["vocab"], vc.DEFAULT_VOCAB):
        if len(vocab) + vc.SPECIAL == rows:
            return vocab
    raise SystemExit(f"the weights have {rows} output rows; the config's vocabulary has {len(cfg['vocab']) + vc.SPECIAL}, VietOCR's {len(vc.DEFAULT_VOCAB) + vc.SPECIAL}")


def evaluate_weights(cfg: dict, weights: str, list_name: str, limit: int = 1500, batch_size: int = 64) -> RecReport:
    """CER / tone report of a weights file (a path or a URL: the stock `vgg_transformer.pth` works) on a list of the data directory."""
    cfg = copy.deepcopy(cfg)
    device = pick_device(cfg)
    cfg["device"] = str(device)
    state = load_state(weights)
    cfg["vocab"] = read_vocab_for(state, cfg)
    model, vocab = build_model(cfg)
    model.load_state_dict(state)
    samples, _ = load_samples(cfg, list_name)
    dataset = Lines(cfg["dataset"]["data_root"], subset(samples, limit), None, cfg["dataset"])
    report, _ = evaluate_model(model, vocab, dataset, device, amp_dtype(device, cfg["trainer"]["amp"]), cfg["trainer"]["max_chars"] + 2, batch_size)
    return report


def export(cfg: dict, out_dir: str | Path) -> Path:
    """`weights.pth` + `config.yml` next to each other: `Predictor(Cfg.load_config_from_file(config.yml))` after pointing `weights` at the file."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    best = Path(cfg["weights"])
    if not best.exists():
        raise SystemExit(f"{best} not found: no validation ran yet, nothing to export")
    shutil.copy(best, out / "weights.pth")
    served = copy.deepcopy(cfg)
    served["weights"], served["device"] = "weights.pth", "cpu"
    return vc.write_config(served, out / "config.yml")
