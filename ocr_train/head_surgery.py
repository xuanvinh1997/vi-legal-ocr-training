"""Carry a pretrained recogniser over to a new dictionary instead of re-learning the classifier from scratch.

The CTC output layer has one column per class: [blank, dictionary..., space]. For every character of the new dictionary
that the old one also had, the old column is copied; new characters start from their tone-stripped letter (ế from ê,
ợ from ơ, ạ from a) plus noise, which is a far better start than random. Everything else (backbone, neck) is untouched, and the
auxiliary NRTR head is left alone: its shape no longer matches, so PaddleOCR skips it, and it is dropped at export.

`.pdparams` files are pickles. We only ever unpickle NumPy arrays and an OrderedDict, never arbitrary classes.
"""
from __future__ import annotations

import pickle
import unicodedata
from collections import OrderedDict
from pathlib import Path

import numpy as np

from .charset import VI_TONES


class _ArraysOnly(pickle.Unpickler):
    _ALLOWED = {("collections", "OrderedDict"), ("numpy", "ndarray"), ("numpy", "dtype"),
                ("numpy._core.multiarray", "_reconstruct"), ("numpy.core.multiarray", "_reconstruct"),
                ("numpy._core.multiarray", "scalar"), ("numpy.core.multiarray", "scalar"),
                ("numpy._core.numeric", "_frombuffer"), ("numpy.core.numeric", "_frombuffer")}

    def find_class(self, module, name):
        if (module, name) in self._ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f"refusing to unpickle {module}.{name}")


def load_pdparams(path: str | Path) -> OrderedDict:
    with Path(path).open("rb") as handle:
        return _ArraysOnly(handle).load()


def save_pdparams(params: dict, path: str | Path) -> None:
    with Path(path).open("wb") as handle:
        pickle.dump(OrderedDict(params), handle, protocol=4)


def class_list(dictionary: list[str], use_space_char: bool = True) -> list[str]:
    """Class order used by PaddleOCR's CTCLabelDecode: blank first, space (if enabled) last."""
    return ["blank"] + list(dictionary) + ([" "] if use_space_char else [])


def remap_ctc_head(params: dict, old_dict: list[str], new_dict: list[str], *, use_space_char: bool = True, seed: int = 0) -> tuple[OrderedDict, dict]:
    old_classes, new_classes = class_list(old_dict, use_space_char), class_list(new_dict, use_space_char)
    old_index: dict[str, int] = {}
    for i, ch in enumerate(old_classes):
        old_index.setdefault(ch, i)
    targets = [k for k, v in params.items() if "ctc_head" in k and getattr(v, "shape", None) and v.shape[-1] == len(old_classes)]
    if not any(k.endswith("weight") for k in targets) or not any(k.endswith("bias") for k in targets):
        raise ValueError(f"could not find the CTC output layer for {len(old_classes)} classes; "
                         f"is the old dictionary the one this checkpoint was trained with?")
    rng = np.random.default_rng(seed)
    out = OrderedDict(params)
    stats = {"copied": 0, "from_base": 0, "random": 0, "tensors": targets}
    plan: list[tuple[int | None, str]] = []  # per new class: (source column or None, how)
    for ch in new_classes:
        if ch in old_index:
            plan.append((old_index[ch], "copied"))
            continue
        base = unicodedata.normalize("NFC", "".join(c for c in unicodedata.normalize("NFD", ch) if c not in VI_TONES))  # ế -> ê, ạ -> a
        plan.append((old_index[base], "from_base") if base != ch and base in old_index else (None, "random"))
    for _, how in plan:
        stats[how] += 1
    for key in targets:
        old = np.asarray(params[key])
        scale = float(old.std()) if old.size else 0.02
        new = np.empty(old.shape[:-1] + (len(new_classes),), dtype=old.dtype)
        for j, (src, how) in enumerate(plan):
            if how == "copied":
                new[..., j] = old[..., src]
            elif how == "from_base":
                new[..., j] = old[..., src] + rng.normal(0, 0.1 * scale, old.shape[:-1]).astype(old.dtype)
            else:
                new[..., j] = rng.normal(0, scale, old.shape[:-1]).astype(old.dtype)
            if key.endswith("bias") and how != "copied":
                new[..., j] = old[..., src] - 1.0 if src is not None else float(old.min())
        out[key] = new
    return out, stats
