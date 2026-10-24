"""Derive training configs from the upstream PaddleOCR YAML instead of keeping forked copies that rot.

Why not just `-o key=value`: the recognition recipe repeats `max_text_length` through YAML anchors and nests it in
lists (NRTR head, RecConAug), which `-o` cannot address. Rewriting the parsed YAML is the only reliable way.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REC_HEIGHT = 48


def _load(path: str | Path) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _dump(cfg: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False, width=120), encoding="utf-8")
    return path


def _drop(transforms: list[dict], name: str) -> list[dict]:
    return [t for t in transforms if name not in t]


def build_rec_config(upstream: str | Path, out: str | Path, *, data_root: str, train_list: str, val_list: str, dict_path: str,
                     max_text_length: int, img_w: int, epochs: int, batch_size: int, save_dir: str, pretrained: str | None = None,
                     lr_scale: float = 0.4, workers: int = 4, eval_every: int = 2000, eval_start: int = 2000, save_every: int = 1,
                     use_gpu: bool = True, keep_con_aug: bool = False, ds_width: bool = False, rec_aug: bool = False) -> Path:
    cfg = _load(upstream)
    g = cfg["Global"]
    g.update(character_dict_path=dict_path, max_text_length=max_text_length, epoch_num=epochs, save_model_dir=save_dir,
             pretrained_model=pretrained, eval_batch_step=[eval_start, eval_every], save_epoch_step=save_every, use_gpu=use_gpu,
             print_batch_step=20, d2s_train_image_shape=[3, REC_HEIGHT, img_w], use_space_char=True)
    g.pop("infer_img", None)

    for head in cfg["Architecture"]["Head"]["head_list"]:
        if "NRTRHead" in head:  # positional table length of the auxiliary decoder
            head["NRTRHead"]["max_text_length"] = max_text_length

    opt = cfg["Optimizer"]["lr"]
    opt["learning_rate"] = float(opt["learning_rate"]) * lr_scale
    opt["warmup_epoch"] = 1

    train = cfg["Train"]
    train["dataset"].update(data_dir=data_root, label_file_list=[train_list])
    tf = train["dataset"]["transforms"]
    if keep_con_aug:
        for t in tf:
            if "RecConAug" in t:
                t["RecConAug"].update(image_shape=[REC_HEIGHT, img_w, 3], max_text_length=max_text_length)
    else:  # concatenating samples is meant for 25-character crops; our lines are long already
        train["dataset"]["transforms"] = _drop(tf, "RecConAug")
        train["dataset"]["ext_op_transform_idx"] = 1
    if not rec_aug:  # our lines are already degraded like scans; RecAug's CPU cost halves throughput on a 2-core Colab VM (45 -> ~100 lines/s)
        train["dataset"]["transforms"] = _drop(train["dataset"]["transforms"], "RecAug")
    sampler = train["sampler"]
    sampler.update(scales=[[img_w, 32], [img_w, REC_HEIGHT], [img_w, 64]], first_bs=batch_size)
    if ds_width:  # `train_list` must then carry width and height columns (dataset.write_wh_list)
        train["dataset"]["ds_width"] = True
        sampler["max_w"] = img_w
    train["loader"].update(batch_size_per_card=batch_size, num_workers=workers)

    ev = cfg["Eval"]
    ev["dataset"].update(data_dir=data_root, label_file_list=[val_list])
    for t in ev["dataset"]["transforms"]:
        if "RecResizeImg" in t:
            t["RecResizeImg"]["image_shape"] = [3, REC_HEIGHT, img_w]
    ev["loader"].update(batch_size_per_card=batch_size, num_workers=max(1, workers // 2))
    return _dump(cfg, out)


def build_det_config(upstream: str | Path, out: str | Path, *, data_root: str, train_list: str, val_list: str, epochs: int,
                     batch_size: int, save_dir: str, pretrained: str | None = None, lr_scale: float = 0.5, workers: int = 4,
                     eval_every: int = 500, eval_start: int = 500, save_every: int = 5, use_gpu: bool = True, crop: int = 640,
                     max_rotate: float = 10.0, scale_range: tuple[float, float] = (0.4, 2.0)) -> Path:
    cfg = _load(upstream)
    g = cfg["Global"]
    g.update(epoch_num=epochs, save_model_dir=save_dir, pretrained_model=pretrained, eval_batch_step=[eval_start, eval_every],
             save_epoch_step=save_every, use_gpu=use_gpu, print_batch_step=20)
    g.pop("infer_img", None)
    # `epoch_num` is also used by MakeBorderMap / MakeShrinkMap (total_epoch) through an anchor.
    opt = cfg["Optimizer"]["lr"]
    opt["learning_rate"] = float(opt["learning_rate"]) * lr_scale
    opt["warmup_epoch"] = 1

    train = cfg["Train"]
    train["dataset"].update(data_dir=data_root, label_file_list=[train_list], ratio_list=[1.0])
    for t in train["dataset"]["transforms"]:
        if "MakeBorderMap" in t:
            t["MakeBorderMap"]["total_epoch"] = epochs
        if "MakeShrinkMap" in t:
            t["MakeShrinkMap"]["total_epoch"] = epochs
        if "RandomCrop" in t:
            t["RandomCrop"]["size"] = [crop, crop]
        if "IaaAugment" in t:
            for aug in t["IaaAugment"]["augmenter_args"]:
                if aug["type"] == "Affine":  # documents are near-horizontal; sideways pages are fixed by orientation models
                    aug["args"]["rotate"] = [-max_rotate, max_rotate]
                if aug["type"] == "Resize":
                    aug["args"]["size"] = list(scale_range)
    train["loader"].update(batch_size_per_card=batch_size, num_workers=workers)

    ev = cfg["Eval"]
    ev["dataset"].update(data_dir=data_root, label_file_list=[val_list])
    ev["loader"]["num_workers"] = max(1, workers // 2)
    return _dump(cfg, out)
