#!/bin/bash
# Fine-tune VietOCR (VGG19-BN + Transformer) on the same data as the PaddleOCR recogniser. The notebook calls it phase by phase;
# `colab exec` or any Linux GPU box can run it too.
#
#   PHASES="setup data config baseline train eval export" bash run_vietocr.sh      (all knobs below are environment variables)
#
# Unlike run_pipeline.sh there is no virtualenv: VietOCR needs only PyTorch, which Colab ships with its GPU build. The data directory is
# the one run_pipeline.sh builds (rec/ lists + dict/vi_compact.txt), so a data.tar made for Paddle is restored and reused as it is;
# the `data` phase only builds it when it is missing.
#
# A second round never overwrites the first: INIT_FROM=<config of the first run> starts from that run's best.pth with its vocabulary and
# image size, into a new VOCR_TAG. RESUME=1 instead continues an interrupted run (same schedule, from latest.pt).
#   PHASES="config train eval export" VOCR_TAG=vietocr_r2 INIT_FROM=$OUT/configs/vietocr.yml VOCR_EPOCHS=2 bash run_vietocr.sh
set -uo pipefail
: "${PHASES:=setup data config baseline train eval export}"
: "${WORK:=/content}" "${CODE:=$WORK/ocr}" "${DATA:=$WORK/data}" "${OUT:=$WORK/out}" "${FONTS:=$WORK/fonts}"
: "${CORPUS_DOCS:=3000}" "${SYNTH_LINES:=300000}" "${SYNTH_PAGES:=4000}" "${PDF_DIR:=}" "${PDF_REPEAT:=3}" "${WORKERS:=$(nproc)}"
: "${VOCR_TAG:=vietocr}" "${VOCR_EPOCHS:=3}" "${VOCR_BATCH:=32}" "${VOCR_HEIGHT:=}" "${VOCR_LR_SCALE:=0.4}" "${VOCR_EVAL:=2000}" "${VOCR_SAVE:=2000}"
: "${VOCR_METRICS:=1500}" "${VOCR_WORKERS:=4}" "${VOCR_AMP:=1}" "${EVAL_LIST:=val_list.txt}" "${INIT_FROM:=}" "${RESUME:=0}"
export PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 MPLBACKEND=Agg
CFG=$OUT/configs/$VOCR_TAG.yml; RUN=$OUT/$VOCR_TAG; DICT=$DATA/dict/vi_compact.txt
STOCK=https://vocr.vn/data/vietocr/vgg_transformer.pth
mkdir -p "$WORK" "$OUT/configs"

step() { echo; echo "=== [$(date +%T)] $*"; echo "$*" > "$WORK/status"; }
die()  { echo "FAILED: $*"; echo "FAILED: $*" > "$WORK/status"; exit 1; }
has()  { [[ " $PHASES " == *" $1 "* ]]; }
ocr()  { python -m ocr_train --data "$DATA" "$@"; }

if has setup; then
  step "setup"
  # torch/torchvision come with Colab. vietocr is installed without its dependencies: they pin imgaug, which does not import on numpy 2,
  # and this code does not use the part of vietocr that needs it (see ocr_train/vietocr_train.py).
  python -m pip install -q -e "$CODE" || die pip-code
  python -m pip install -q --no-deps vietocr==0.3.13 || die pip-vietocr
  python -m pip install -q einops gdown requests tqdm pyyaml || die pip-deps
  python -c "import torch, vietocr; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || die import
fi
if has data; then
  REPEAT=(); [ -n "$PDF_DIR" ] && REPEAT=(--repeat "pdf=$PDF_REPEAT")
  if [ -s "$DATA/rec/train_list.txt" ] && [ -s "$DICT" ]; then
    step "data: reusing $DATA"
    # an archive made before the width lists existed: rebuild just those (merge is deterministic and only rewrites the lists)
    [ -s "$DATA/rec/train_list_wh.txt" ] || ocr merge "${REPEAT[@]}" || die merge
  else
    step "data: corpus";       ocr corpus --docs "$CORPUS_DOCS" || die corpus
    ocr charset --mode compact || die charset
    step "data: synth-lines";  ocr synth-lines --count "$SYNTH_LINES" --workers "$WORKERS" --fonts-dir "$FONTS" || die synth-lines
    step "data: synth-pages";  ocr synth-pages --count "$SYNTH_PAGES" --workers "$WORKERS" --fonts-dir "$FONTS" || die synth-pages
    [ -n "$PDF_DIR" ] && { ocr pdf-pages --pdfs "$PDF_DIR" --workers "$WORKERS" || die pdf-pages; }
    ocr merge "${REPEAT[@]}" || die merge
    ocr check --dict "$DICT" || die check
  fi
  cat "$DATA/rec/meta.json"
fi
if has config; then
  step "config ($VOCR_TAG)"
  [ -e "$RUN/latest.pt" ] && [ "$RESUME" != 1 ] && die "$RUN already holds a run: set RESUME=1 to continue it or pick another VOCR_TAG"
  if [ "$RESUME" = 1 ] && [ -f "$CFG" ]; then
    echo "resuming: keeping $CFG"
  else
    EXTRA=(); [ -n "$INIT_FROM" ] && EXTRA+=(--like "$INIT_FROM"); [ -n "$VOCR_HEIGHT" ] && EXTRA+=(--height "$VOCR_HEIGHT"); [ "$VOCR_AMP" = 0 ] && EXTRA+=(--no-amp)
    ocr config-vietocr --dict "$DICT" --out "$CFG" --save-dir "$RUN" --epochs "$VOCR_EPOCHS" --batch-size "$VOCR_BATCH" --workers "$VOCR_WORKERS" \
      --lr-scale "$VOCR_LR_SCALE" --valid-every "$VOCR_EVAL" --save-every "$VOCR_SAVE" --metrics "$VOCR_METRICS" "${EXTRA[@]}" || die config
  fi
fi
if has baseline; then
  step "baseline: VietOCR's own vgg_transformer.pth on $EVAL_LIST"
  ocr eval-vietocr --config "$CFG" --weights "$STOCK" --list "$EVAL_LIST" --limit "$VOCR_METRICS" || die baseline
fi
if has train; then
  step "train ($VOCR_TAG)"
  RES=(); [ "$RESUME" = 1 ] && RES=(--resume)
  ocr train-vietocr --config "$CFG" "${RES[@]}" || die train
fi
if has eval; then
  step "eval: $VOCR_TAG on $EVAL_LIST"
  ocr eval-vietocr --config "$CFG" --list "$EVAL_LIST" --limit "$VOCR_METRICS" || die eval
fi
if has export; then
  step "export"
  ocr export-vietocr --config "$CFG" --out "$OUT/export/$VOCR_TAG" || die export
fi
step "DONE ($PHASES)"
