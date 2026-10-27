"""`python -m ocr_train <command>` – every step of the pipeline, in the order the notebook runs them."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

from . import charset, corpus, dataset, english, evaluate, export, head_surgery, paddle_cfg, pdf_labels, public_sets, synth_lines, synth_pages, vietocr_cfg


def _data(args) -> Path:
    return Path(args.data)


def cmd_sources(args):
    print(public_sets.print_sources())


def cmd_corpus(args):
    out = _data(args) / "corpus.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    docs = list(english.template_documents(args.en_docs, seed=args.seed))
    if args.en_dataset:
        docs += english.stream_documents(args.en_dataset, args.en_dataset_docs, config=args.en_config, column=args.en_column, seed=args.seed)
    print(corpus.build_corpus(out, corpus.stream_documents(args.docs, seed=args.seed), english=docs), "->", out)


def cmd_charset(args):
    chars = charset.build_charset()
    if args.upstream_dict:
        upstream = charset.read_dict(args.upstream_dict)
        missing = charset.missing_from(upstream)
        print(f"{args.upstream_dict}: {len(upstream)} classes, missing {len(missing)} Vietnamese letters: {''.join(missing)}")
        if args.mode == "extend":
            chars = charset.extend_dict(upstream, chars)
    out = _data(args) / "dict" / f"vi_{args.mode}.txt"
    out.parent.mkdir(parents=True, exist_ok=True)
    charset.write_dict(out, chars)
    print(f"{args.mode}: {len(chars)} characters -> {out}")


def cmd_synth_lines(args):
    print(synth_lines.build(_data(args) / "rec", _data(args) / "corpus.jsonl", args.fonts_dir, args.count, workers=args.workers, seed=args.seed,
                            max_chars=args.max_chars, level=args.level, extra_fonts=args.extra_fonts, shard=args.shard or synth_lines.SHARD, val_ratio=args.val_ratio, profile=args.profile,
                            en_ratio=args.en_ratio))


def cmd_synth_pages(args):
    print(synth_pages.build(_data(args), _data(args) / "corpus.jsonl", args.fonts_dir, args.count, workers=args.workers, seed=args.seed,
                            level=args.level, max_chars=args.max_chars, extra_fonts=args.extra_fonts, shard=args.shard or synth_pages.SHARD, val_ratio=args.val_ratio, en_ratio=args.en_ratio))


def cmd_pdf_pages(args):
    print(pdf_labels.build(_data(args), args.pdfs, workers=args.workers, seed=args.seed, dpi=args.dpi, level=args.level, max_pages=args.max_pages))


def cmd_download_pdfs(args):
    print(public_sets.download_pdfs(args.urls, args.out, delay=args.delay))


def cmd_import_rec(args):
    print(public_sets.import_rec_tsv(_data(args), args.name, args.labels, args.images_dir, max_chars=args.max_chars))


def cmd_import_hf_rec(args):
    print(public_sets.import_hf_rec(_data(args), args.name, args.dataset, image_col=args.image_col, text_col=args.text_col, split=args.split, limit=args.limit, max_chars=args.max_chars))


def cmd_merge(args):
    data = _data(args)
    repeat = {name: int(n) for name, n in (item.split("=") for item in args.repeat)}
    result = {}
    for kind in ("rec", "det"):
        if dataset.shards(data, kind):
            result[kind] = dataset.merge(data, kind, seed=args.seed, repeat=repeat)
    if "rec" in result:
        dataset.write_wh_list(data)
        stats = dataset.rec_stats(data)
        (data / "rec" / "meta.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
        result["rec_stats"] = stats
    print(json.dumps(result, indent=2))


def cmd_check(args):
    print("rec:", dataset.check_rec(_data(args), args.dict))
    if dataset.shards(_data(args), "det"):
        print("det:", dataset.check_det(_data(args)))


def _upstream(args, kind: str) -> Path:
    family = "PP-OCRv6" if args.family == "v6" else "PP-OCRv5"
    name = f"{family}_{args.size}_{kind}.yml"
    return Path(args.paddleocr) / "configs" / kind / family / name


def cmd_config_rec(args):
    data = _data(args).resolve()
    meta = json.loads((data / "rec" / "meta.json").read_text())
    wh_list = data / "rec" / "train_list_wh.txt"
    max_len, img_w = args.max_text_length or meta["max_text_length"], args.img_w or meta["img_w"]
    if args.like:  # continue a run: same input width and label length as the config the checkpoint was trained with
        import yaml
        prev = yaml.safe_load(Path(args.like).read_text(encoding="utf-8"))
        prev_len, prev_w = prev["Global"]["max_text_length"], prev["Train"]["sampler"]["scales"][1][0]
        if meta["max_text_length"] > prev_len:  # PaddleOCR would drop these labels without a word
            raise SystemExit(f"the merged data has labels up to {meta['max_text_length']} (incl. margin) but {args.like} was trained with max_text_length={prev_len}")
        max_len, img_w = args.max_text_length or prev_len, args.img_w or prev_w
        print(f"like {args.like}: max_text_length={max_len}, img_w={img_w} (data now suggests {meta['max_text_length']}, {meta['img_w']})")
    ds_width = not args.fixed_width and wh_list.exists()
    out = paddle_cfg.build_rec_config(
        _upstream(args, "rec"), args.out, data_root=str(data / "rec"), train_list=str(wh_list if ds_width else data / "rec" / "train_list.txt"), ds_width=ds_width, rec_aug=args.rec_aug, val_list=str(data / "rec" / "val_list.txt"),
        dict_path=str(Path(args.dict).resolve()), max_text_length=max_len, img_w=img_w,
        epochs=args.epochs, batch_size=args.batch_size, save_dir=str(Path(args.save_dir).resolve()), pretrained=args.pretrained, workers=args.workers,
        use_gpu=not args.cpu, eval_every=args.eval_every, eval_start=args.eval_every, lr_scale=args.lr_scale)
    print(out)


def cmd_config_det(args):
    data = _data(args).resolve()
    out = paddle_cfg.build_det_config(
        _upstream(args, "det"), args.out, data_root=str(data / "det"), train_list=str(data / "det" / "train_list.txt"), val_list=str(data / "det" / "val_list.txt"),
        epochs=args.epochs, batch_size=args.batch_size, save_dir=str(Path(args.save_dir).resolve()), pretrained=args.pretrained, workers=args.workers,
        use_gpu=not args.cpu, eval_every=args.eval_every, eval_start=args.eval_every, lr_scale=args.lr_scale)
    print(out)


def cmd_surgery(args):
    params = head_surgery.load_pdparams(args.pretrained)
    new, stats = head_surgery.remap_ctc_head(params, charset.read_dict(args.old_dict), charset.read_dict(args.new_dict))
    head_surgery.save_pdparams(new, args.out)
    print({k: v for k, v in stats.items() if k != "tensors"}, "->", args.out)


def cmd_check_init(args):
    print("ok,", head_surgery.check_init(head_surgery.load_pdparams(args.pretrained), charset.read_dict(args.dict)), "classes")


def cmd_eval_rec(args):
    report = evaluate.evaluate_list(_data(args), args.list, args.model_name, args.model_dir, limit=args.limit)
    print(report.summary())
    for d, ref, hyp in report.worst[:8]:
        print(f"  [{d}] {ref!r}\n       {hyp!r}")


def cmd_config_vietocr(args):
    """The VietOCR counterpart of config-rec: reads the same merged `rec/` lists, writes one YAML for `train-vietocr`."""
    rec = _data(args).resolve() / "rec"
    train_list = args.train_list or ("train_list_wh.txt" if (rec / "train_list_wh.txt").exists() else "train_list.txt")
    base_vocab, pretrained, height, max_chars = vietocr_cfg.DEFAULT_VOCAB, args.pretrained or vietocr_cfg.PRETRAIN_URL, args.height, args.max_chars
    init_vocab = base_vocab
    if args.like:  # continue from an earlier run: its vocabulary (extended if the dictionary grew), its weights, its image size
        prev = vietocr_cfg.read_config(args.like)
        base_vocab = init_vocab = prev["vocab"]
        pretrained = args.pretrained or prev["weights"]
        height = args.height or prev["dataset"]["image_height"]
        max_chars = args.max_chars or prev["trainer"]["max_chars"]
    height, max_chars = height or 32, max_chars or 126
    vocab = vietocr_cfg.build_vocab(charset.read_dict(args.dict), base_vocab)
    max_width = args.max_width
    if not max_width:
        samples = vietocr_cfg.read_list(rec / train_list)
        if not any(s[2] for s in samples):
            raise SystemExit(f"{rec / train_list} has no image sizes: run `merge` (it writes train_list_wh.txt) or pass --max-width")
        max_width = vietocr_cfg.suggest_max_width(samples, height)
        print(f"widest line at height {height}: {max_width}px (the positional encoding fits {vietocr_cfg.max_width_for(height)})")
    cfg = vietocr_cfg.build_config(
        vocab=vocab, data_root=str(rec), train_list=train_list, val_list=args.val_list, save_dir=str(Path(args.save_dir).resolve()), pretrained=pretrained,
        pretrained_vocab=init_vocab, epochs=args.epochs, batch_size=args.batch_size, image_height=height, image_max_width=max_width,
        max_lr=3e-4 * args.lr_scale, max_chars=max_chars, workers=args.workers, valid_every=args.valid_every, save_every=args.save_every,
        metrics=args.metrics, print_every=args.print_every, mask_prob=args.mask_prob, device="cpu" if args.cpu else "cuda:0", amp=not args.no_amp)
    print(vietocr_cfg.write_config(cfg, args.out), f"(vocabulary {len(vocab)} characters, {len(vocab) - len(base_vocab)} added)")


def cmd_train_vietocr(args):
    from . import vietocr_train  # torch + vietocr, imported only here so the data commands run without them

    print(vietocr_train.train(vietocr_cfg.read_config(args.config), resume=args.resume))


def cmd_eval_vietocr(args):
    from . import vietocr_train

    cfg = vietocr_cfg.read_config(args.config)
    report = vietocr_train.evaluate_weights(cfg, args.weights or cfg["weights"], args.list, limit=args.limit)
    print(report.summary())
    for d, ref, hyp in report.worst[:8]:
        print(f"  [{d}] {ref!r}\n       {hyp!r}")


def cmd_export_vietocr(args):
    from . import vietocr_train

    print(vietocr_train.export(vietocr_cfg.read_config(args.config), args.out))


def cmd_export_all(args):
    """Paddle inference model + ONNX for everything that was trained. Run it with a CPU build of Paddle (see export.export_inference)."""
    out, exp, cfg = Path(args.out), Path(args.out) / "export", Path(args.out) / "configs"

    def checkpoint(directory: Path) -> Path:  # best_accuracy exists once an evaluation ran; a very short run only has `latest`
        return directory / "best_accuracy" if (directory / "best_accuracy.pdparams").exists() else directory / "latest"

    wanted = set(args.only or ("rec", "det", "layout"))
    for name, shape in (("rec", (2, 3, 48, 640)), ("det", (1, 3, 736, 544))):
        run = args.rec_tag if name == "rec" else name  # a continued recognition run lives in its own directory and exports next to the first
        if name not in wanted or not (out / run).exists():
            continue
        target = export.export_inference(args.paddleocr, cfg / f"{run}.yml", checkpoint(out / run), exp / run)
        if name == "rec":
            export.patch_rec_inference_yml(target)
        export.to_onnx(target, target / "model.onnx")
        print(f"{run} onnx vs paddle, max |diff|:", export.verify_onnx(target, target / "model.onnx", shape))
    if "layout" in wanted and (out / "layout/best_model/inference").exists():
        shutil.copytree(out / "layout/best_model/inference", exp / "layout", dirs_exist_ok=True)
        export.to_onnx(exp / "layout", exp / "layout/model.onnx")  # multi-input model: conversion is checked, not numerics
    print("exported:", sorted(str(p.relative_to(exp)) for p in exp.rglob("*") if p.is_file()))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="ocr_train", description=__doc__)
    p.add_argument("--data", default="data", help="working directory for corpus, datasets and dictionaries")
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, fn, help_, hard_exit=False):
        s = sub.add_parser(name, help=help_)
        s.set_defaults(fn=fn, hard_exit=hard_exit)
        return s

    add("sources", cmd_sources, "list public data sources and licences")
    s = add("corpus", cmd_corpus, "stream legal documents from the Hub into corpus.jsonl", hard_exit=True)
    s.add_argument("--docs", type=int, default=3000), s.add_argument("--seed", type=int, default=0)
    s.add_argument("--en-docs", type=int, default=600, help="English contracts built from templates (no download); 0 for a Vietnamese-only corpus")
    s.add_argument("--en-dataset", help="optional Hub dataset with English plain text to add (check its licence)"), s.add_argument("--en-config"), s.add_argument("--en-column", default="text")
    s.add_argument("--en-dataset-docs", type=int, default=1000)
    s = add("charset", cmd_charset, "write the Vietnamese dictionary (compact, or extend an upstream one)")
    s.add_argument("--mode", choices=["compact", "extend"], default="compact"), s.add_argument("--upstream-dict")
    for name, fn, help_ in [("synth-lines", cmd_synth_lines, "synthetic recognition lines"), ("synth-pages", cmd_synth_pages, "synthetic pages (det + layout + rec crops)")]:
        s = add(name, fn, help_)
        s.add_argument("--count", type=int, required=True), s.add_argument("--fonts-dir", default="fonts"), s.add_argument("--extra-fonts", nargs="*", default=[])
        s.add_argument("--workers", type=int, default=2), s.add_argument("--seed", type=int, default=0), s.add_argument("--level", type=float, default=1.0)
        s.add_argument("--max-chars", type=int, default=100)
        s.add_argument("--en-ratio", type=float, default=0.25 if name == "synth-lines" else 0.2, help="share of English lines/pages; 0 gives Vietnamese only")
        s.add_argument("--shard", help="write to a new shard with this name (and use a new --seed) to add data without touching an earlier run")
        if name == "synth-lines":
            s.add_argument("--profile", choices=["standard", "scan"], default="standard", help="`scan`: lines at scan resolution (~30 px high) with a detector crop's margin, neighbouring-line slivers, more italics, JPEG at that resolution")
        s.add_argument("--val-ratio", type=float, default=0.01 if name == "synth-lines" else 0.04, help="0 keeps the validation list unchanged, so accuracy stays comparable between rounds")
    s = add("pdf-pages", cmd_pdf_pages, "real born-digital PDFs -> labelled pages")
    s.add_argument("--pdfs", required=True), s.add_argument("--workers", type=int, default=2), s.add_argument("--seed", type=int, default=0)
    s.add_argument("--dpi", type=int, default=150), s.add_argument("--level", type=float, default=1.0), s.add_argument("--max-pages", type=int, default=40)
    s = add("download-pdfs", cmd_download_pdfs, "polite download of the PDF URLs you list")
    s.add_argument("--urls", required=True), s.add_argument("--out", required=True), s.add_argument("--delay", type=float, default=2.0)
    s = add("import-rec", cmd_import_rec, "import image<TAB>text line data")
    s.add_argument("--name", required=True), s.add_argument("--labels", required=True), s.add_argument("--images-dir"), s.add_argument("--max-chars", type=int, default=100)
    s = add("import-hf-rec", cmd_import_hf_rec, "import a Hugging Face image/text dataset", hard_exit=True)
    s.add_argument("--name", required=True), s.add_argument("--dataset", required=True), s.add_argument("--image-col", default="image")
    s.add_argument("--text-col", default="label"), s.add_argument("--split", default="train"), s.add_argument("--limit", type=int, default=20000), s.add_argument("--max-chars", type=int, default=100)
    s = add("merge", cmd_merge, "merge shards into train_list/val_list and measure sizes")
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--repeat", nargs="*", default=[], metavar="SHARD=N", help="list a shard N times in the training list, e.g. pdf=3")
    s = add("check", cmd_check, "validate labels against the dictionary")
    s.add_argument("--dict", required=True)
    for name, fn in [("config-rec", cmd_config_rec), ("config-det", cmd_config_det)]:
        s = add(name, fn, f"write a {name[-3:]} training config from the upstream recipe")
        s.add_argument("--paddleocr", required=True, help="PaddleOCR checkout"), s.add_argument("--size", choices=["tiny", "small"], default="small")
        s.add_argument("--family", choices=["v6", "v5"], default="v6"), s.add_argument("--out", required=True), s.add_argument("--save-dir", required=True)
        s.add_argument("--pretrained"), s.add_argument("--epochs", type=int, default=20), s.add_argument("--batch-size", type=int, default=64 if name == "config-rec" else 16)
        s.add_argument("--workers", type=int, default=4), s.add_argument("--cpu", action="store_true"), s.add_argument("--lr-scale", type=float, default=0.4)
        s.add_argument("--eval-every", type=int, default=2000 if name == "config-rec" else 500)
        if name == "config-rec":
            s.add_argument("--dict", required=True), s.add_argument("--img-w", type=int), s.add_argument("--max-text-length", type=int)
            s.add_argument("--like", help="an earlier rec config: reuse its input width and max_text_length (needed to continue its checkpoint)")
            s.add_argument("--rec-aug", action="store_true", help="keep PaddleOCR's RecAug (slower on few CPU cores; the data is already degraded)")
            s.add_argument("--fixed-width", action="store_true", help="pad every line to --img-w instead of grouping batches by width")
    s = add("config-vietocr", cmd_config_vietocr, "write a VietOCR fine-tuning config (VGG19-BN + Transformer) for the merged rec/ data")
    s.add_argument("--dict", required=True, help="the dictionary the data was checked against; its characters VietOCR lacks are appended to the vocabulary")
    s.add_argument("--out", required=True), s.add_argument("--save-dir", required=True)
    s.add_argument("--pretrained", help="weights to start from: a file or URL (default: VietOCR's vgg_transformer.pth)")
    s.add_argument("--like", help="an earlier VietOCR config: start from its best weights and keep its vocabulary and image size")
    s.add_argument("--train-list"), s.add_argument("--val-list", default="val_list.txt")
    s.add_argument("--epochs", type=int, default=6), s.add_argument("--batch-size", type=int, default=32), s.add_argument("--workers", type=int, default=4)
    s.add_argument("--height", type=int, help="image height, a multiple of 16 (default 32 as VietOCR; taller shows tone marks better but the pretrained weights saw 32)")
    s.add_argument("--max-width", type=int, help="default: the widest line of the data, so none is squeezed")
    s.add_argument("--max-chars", type=int, help="longest label kept (default 126, the decoder runs 128 steps)")
    s.add_argument("--lr-scale", type=float, default=0.4, help="times VietOCR's 3e-4 peak rate (not tuned)")
    s.add_argument("--valid-every", type=int, default=2000), s.add_argument("--save-every", type=int, default=2000), s.add_argument("--print-every", type=int, default=100)
    s.add_argument("--metrics", type=int, default=1500, help="validation lines decoded at each check (the same ones every time)")
    s.add_argument("--mask-prob", type=float, default=0.05, help="share of input characters hidden during training (VietOCR's masked-LM trick)")
    s.add_argument("--cpu", action="store_true"), s.add_argument("--no-amp", action="store_true", help="train in float32 on the GPU too")
    s = add("train-vietocr", cmd_train_vietocr, "fine-tune VietOCR from a config-vietocr file")
    s.add_argument("--config", required=True), s.add_argument("--resume", action="store_true", help="continue `latest.pt` of an interrupted run (same schedule)")
    s = add("eval-vietocr", cmd_eval_vietocr, "CER / tone-error report of VietOCR weights on a list file (the stock model works: --weights <url>)")
    s.add_argument("--config", required=True), s.add_argument("--weights", help="default: the config's best.pth")
    s.add_argument("--list", default="val_list.txt"), s.add_argument("--limit", type=int, default=1500)
    s = add("export-vietocr", cmd_export_vietocr, "copy the best weights and a serving config to a directory")
    s.add_argument("--config", required=True), s.add_argument("--out", required=True)
    s = add("surgery", cmd_surgery, "carry pretrained recogniser weights over to a new dictionary")
    s.add_argument("--pretrained", required=True), s.add_argument("--old-dict", required=True), s.add_argument("--new-dict", required=True), s.add_argument("--out", required=True)
    s = add("export-all", cmd_export_all, "export trained models to Paddle inference + ONNX")
    s.add_argument("--paddleocr", required=True), s.add_argument("--out", required=True, help="the training output directory")
    s.add_argument("--rec-tag", default="rec", help="recognition run to export (its directory under --out); exported to export/<tag> unless it is `rec`")
    s.add_argument("--only", nargs="*", choices=["rec", "det", "layout"], help="export just these models (default: everything that was trained)")
    s = add("check-init", cmd_check_init, "fail unless a checkpoint was trained with this dictionary")
    s.add_argument("--pretrained", required=True), s.add_argument("--dict", required=True)
    s = add("eval-rec", cmd_eval_rec, "CER / tone-error report of a recognition model on a list file")
    s.add_argument("--list", default="val_list.txt"), s.add_argument("--model-name", required=True), s.add_argument("--model-dir"), s.add_argument("--limit", type=int, default=2000)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    args.fn(args)
    if args.hard_exit:
        # A half-read Hugging Face stream can leave a download thread that blocks interpreter shutdown (seen on Colab: the
        # file was complete but the process never exited). The work is done and flushed, so leave without waiting for it.
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
