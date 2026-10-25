#!/bin/bash
# The whole pipeline, headless. The notebook calls it phase by phase; `colab exec` or any Linux GPU box can run it too.
#
#   PHASES="setup data install rec det layout export" bash run_pipeline.sh      (all knobs below are environment variables)
#
# A second round (more data, a bigger GPU) never overwrites the first: `more` writes new shards, and a recognition run with
# REC_TAG=rec_r2 INIT_FROM=rec/best_accuracy starts from the first run's weights and trains into its own directory.
#   PHASES="setup install more rec" REC_TAG=rec_r2 INIT_FROM=rec/best_accuracy MORE_LINES=400000 REC_BATCH=128 bash run_pipeline.sh
#
# Why venvs: Colab's system Python (3.13) has no wheel for paddle2onnx 2.0.2rc3, and GPU builds of Paddle write `linear_v2`
# operators that paddle2onnx cannot convert. So training runs in a Python 3.12 venv with the GPU build, and export + ONNX run
# in a second 3.12 venv with the CPU build.
set -uo pipefail
: "${PHASES:=setup data install rec det layout export}"
: "${WORK:=/content}" "${CODE:=$WORK/ocr}" "${DATA:=$WORK/data}" "${OUT:=$WORK/out}"
: "${SIZE:=small}" "${CORPUS_DOCS:=3000}" "${SYNTH_LINES:=300000}" "${SYNTH_PAGES:=4000}" "${PDF_DIR:=}" "${PDF_REPEAT:=3}" "${WORKERS:=$(nproc)}"
: "${REC_EPOCHS:=6}" "${REC_BATCH:=32}" "${REC_EVAL:=2000}" "${DET_EPOCHS:=30}" "${DET_BATCH:=8}" "${DET_EVAL:=500}"
: "${LAYOUT_EPOCHS:=40}" "${LAYOUT_BATCH:=8}" "${RESUME:=0}"
: "${REC_TAG:=rec}" "${REC_LR_SCALE:=0.4}" "${INIT_FROM:=}"          # INIT_FROM: a checkpoint (relative paths are under $OUT) to continue from instead of the pretrained weights
: "${MORE_TAG:=r2}" "${MORE_SEED:=1}" "${MORE_LINES:=0}" "${MORE_PAGES:=0}"   # extra data: shards synth_lines_$MORE_TAG / synth_pages_$MORE_TAG
export PYTHONUNBUFFERED=1 PIP_DISABLE_PIP_VERSION_CHECK=1 MPLBACKEND=Agg  # Colab's kernel sets an inline backend that plain processes cannot load
PO=$WORK/PaddleOCR; PX=$WORK/PaddleX; W=$WORK/pretrained; CFG=$OUT/configs; FONTS=$WORK/fonts
HF=https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model
mkdir -p "$OUT" "$W" "$CFG"
UP=$PO/ppocr/utils/dict/$([ "$SIZE" = small ] && echo ppocrv6_dict.txt || echo ppocrv6_tiny_dict.txt)
DICT=$DATA/dict/vi_compact.txt

step() { echo; echo "=== [$(date +%T)] $*"; echo "$*" > "$WORK/status"; }
die()  { echo "FAILED: $*"; echo "FAILED: $*" > "$WORK/status"; exit 1; }
has()  { [[ " $PHASES " == *" $1 "* ]]; }

[ -d "$WORK/venv" ] || uv venv --seed --python 3.12 "$WORK/venv" || die venv
unset UV_SYSTEM_PYTHON UV_BREAK_SYSTEM_PACKAGES  # Colab sets UV_SYSTEM_PYTHON=true, which makes `uv pip` ignore the venv
source "$WORK/venv/bin/activate"
pip() { uv pip "$@"; }

if has setup; then
  step "setup"
  [ -d "$PO" ] || git clone -q --depth 1 --branch v3.7.0 https://github.com/PaddlePaddle/PaddleOCR.git "$PO" || die clone
  pip install -q -e "$CODE" || die pip-code
fi
if has data; then
  step "data: corpus";       python -m ocr_train --data "$DATA" corpus --docs "$CORPUS_DOCS" || die corpus
  python -m ocr_train --data "$DATA" charset --mode compact --upstream-dict "$UP" || die charset
  step "data: synth-lines";  python -m ocr_train --data "$DATA" synth-lines --count "$SYNTH_LINES" --workers "$WORKERS" --fonts-dir "$FONTS" || die synth-lines
  step "data: synth-pages";  python -m ocr_train --data "$DATA" synth-pages --count "$SYNTH_PAGES" --workers "$WORKERS" --fonts-dir "$FONTS" || die synth-pages
  REPEAT=(); if [ -n "$PDF_DIR" ]; then
    python -m ocr_train --data "$DATA" pdf-pages --pdfs "$PDF_DIR" --workers "$WORKERS" || die pdf-pages
    REPEAT=(--repeat "pdf=$PDF_REPEAT")
  fi
  python -m ocr_train --data "$DATA" merge "${REPEAT[@]}" || die merge
  python -m ocr_train --data "$DATA" check --dict "$DICT" || die check
  cat "$DATA/rec/meta.json"
fi
if has more; then
  [ -s "$DATA/corpus.jsonl" ] || die "more: $DATA has no corpus.jsonl (restore data.tar first)"
  # New seed + new shard name = new files next to the old ones. val ratio 0 keeps the validation lists, so accuracy stays comparable with round 1.
  if [ "$MORE_LINES" -gt 0 ]; then
    step "data: more synth-lines ($MORE_TAG)"
    python -m ocr_train --data "$DATA" synth-lines --count "$MORE_LINES" --workers "$WORKERS" --fonts-dir "$FONTS" --seed "$MORE_SEED" --shard "synth_lines_$MORE_TAG" --val-ratio 0 || die more-lines
  fi
  if [ "$MORE_PAGES" -gt 0 ]; then
    step "data: more synth-pages ($MORE_TAG)"
    python -m ocr_train --data "$DATA" synth-pages --count "$MORE_PAGES" --workers "$WORKERS" --fonts-dir "$FONTS" --seed "$MORE_SEED" --shard "synth_pages_$MORE_TAG" --val-ratio 0 || die more-pages
  fi
  REPEAT=(); [ -n "$PDF_DIR" ] && REPEAT=(--repeat "pdf=$PDF_REPEAT")
  python -m ocr_train --data "$DATA" merge "${REPEAT[@]}" || die merge
  python -m ocr_train --data "$DATA" check --dict "$DICT" || die check
  cat "$DATA/rec/meta.json"
fi
if has install; then
  step "install paddle (GPU build)"
  MAJOR=$(nvidia-smi | grep -oE 'CUDA Version: [0-9]+' | grep -oE '[0-9]+$'); TAG=$([ "${MAJOR:-12}" -ge 13 ] && echo cu130 || echo cu126)
  pip install -q paddlepaddle-gpu==3.3.1 --index-url "https://www.paddlepaddle.org.cn/packages/stable/$TAG/" --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match || die paddle
  pip install -q "paddleocr==3.7.0" "paddlex[ocr-core]==3.7.2" pycocotools "setuptools<81" -r "$PO/requirements.txt" || die pip-paddleocr
  python -c "import paddle; paddle.utils.run_check()" 2>&1 | tail -2
fi
if has rec; then
  step "rec: train ($REC_TAG)"
  [ -n "$INIT_FROM" ] && [ "$REC_TAG" = rec ] && die "INIT_FROM needs its own REC_TAG (e.g. rec_r2): training into rec/ would overwrite the run you continue from"
  LIKE=()
  if [ -n "$INIT_FROM" ]; then
    case "$INIT_FROM" in /*) PRE=$INIT_FROM ;; *) PRE=$OUT/$INIT_FROM ;; esac
    [ -f "$PRE.pdparams" ] || [ -f "$PRE" ] || die "INIT_FROM $PRE not found"
    # warm start: weights only (fresh optimiser and learning-rate schedule), valid only for the dictionary it was trained with
    python -m ocr_train check-init --pretrained "${PRE%.pdparams}.pdparams" --dict "$DICT" || die init-dict
    PREV=$CFG/$(basename "$(dirname "$PRE")").yml; [ -f "$PREV" ] && LIKE=(--like "$PREV") || echo "note: $PREV not found, input width and max_text_length come from the merged data"
  else
    [ -s "$W/rec.pdparams" ] || wget -q -O "$W/rec.pdparams" "$HF/PP-OCRv6_${SIZE}_rec_pretrained.pdparams" || die wget-rec
    python -m ocr_train surgery --pretrained "$W/rec.pdparams" --old-dict "$UP" --new-dict "$DICT" --out "$W/rec_vi.pdparams" || die surgery
    PRE=$W/rec_vi.pdparams
  fi
  [ "$REC_TAG" != rec ] && [ -e "$OUT/$REC_TAG/latest.pdparams" ] && [ "$RESUME" != 1 ] && die "$OUT/$REC_TAG already holds a run: set RESUME=1 to continue it or pick another REC_TAG"
  python -m ocr_train --data "$DATA" config-rec --paddleocr "$PO" --size "$SIZE" --dict "$DICT" --out "$CFG/$REC_TAG.yml" --save-dir "$OUT/$REC_TAG" --pretrained "$PRE" --epochs "$REC_EPOCHS" --batch-size "$REC_BATCH" --workers "$WORKERS" --eval-every "$REC_EVAL" --lr-scale "$REC_LR_SCALE" "${LIKE[@]}" || die config-rec
  RES=(); [ "$RESUME" = 1 ] && [ -f "$OUT/$REC_TAG/latest.pdparams" ] && RES=(-o "Global.checkpoints=$OUT/$REC_TAG/latest")
  (cd "$PO" && python tools/train.py -c "$CFG/$REC_TAG.yml" "${RES[@]}") > "$OUT/${REC_TAG}_train.log" 2>&1 || { tail -30 "$OUT/${REC_TAG}_train.log"; die rec-train; }
fi
if has det; then
  step "det: train"
  [ -s "$W/det.pdparams" ] || wget -q -O "$W/det.pdparams" "$HF/PP-OCRv6_${SIZE}_det_pretrained.pdparams" || die wget-det
  python -m ocr_train --data "$DATA" config-det --paddleocr "$PO" --size "$SIZE" --out "$CFG/det.yml" --save-dir "$OUT/det" --pretrained "$W/det.pdparams" --epochs "$DET_EPOCHS" --batch-size "$DET_BATCH" --workers "$WORKERS" --eval-every "$DET_EVAL" || die config-det
  RES=(); [ "$RESUME" = 1 ] && [ -f "$OUT/det/latest.pdparams" ] && RES=(-o "Global.checkpoints=$OUT/det/latest")
  (cd "$PO" && python tools/train.py -c "$CFG/det.yml" "${RES[@]}") > "$OUT/det_train.log" 2>&1 || { tail -30 "$OUT/det_train.log"; die det-train; }
fi
if has layout; then
  step "layout: train"
  [ -d "$PX" ] || git clone -q --depth 1 --branch v3.7.2 https://github.com/PaddlePaddle/PaddleX.git "$PX" || die clone-px
  # `paddlex --install` prompts when the plugin already exists, so only run it once
  (cd "$PX" && pip install -q -e . && { [ -d paddlex/repo_manager/repos/PaddleDetection ] || paddlex --install PaddleDetection; }) > "$OUT/layout_install.log" 2>&1 || { tail -20 "$OUT/layout_install.log"; die layout-install; }
  NC=$(python -c "from ocr_train.synth_pages import CATEGORIES; print(len(CATEGORIES)+1)")
  LC=paddlex/configs/modules/layout_detection/PP-DocLayout-S.yaml
  (cd "$PX" && python main.py -c $LC -o Global.mode=check_dataset -o Global.dataset_dir="$DATA/layout" && \
   python main.py -c $LC -o Global.mode=train -o Global.dataset_dir="$DATA/layout" -o Global.device=gpu:0 -o Global.output="$OUT/layout" \
     -o Train.num_classes="$NC" -o Train.epochs_iters="$LAYOUT_EPOCHS" -o Train.batch_size="$LAYOUT_BATCH") > "$OUT/layout_train.log" 2>&1 || { tail -30 "$OUT/layout_train.log"; die layout-train; }
fi
if has export; then
  step "export (CPU build of Paddle)"
  if [ ! -x "$WORK/venv_cpu/bin/python" ]; then
    uv venv --seed --python 3.12 "$WORK/venv_cpu" || die venv-cpu
    uv pip install -q --python "$WORK/venv_cpu/bin/python" paddlepaddle==3.3.1 "paddleocr==3.7.0" "paddlex[ocr-core]==3.7.2" paddle2onnx==2.0.2rc3 onnxruntime "setuptools<81" -r "$PO/requirements.txt" -e "$CODE" || die pip-cpu
  fi
  "$WORK/venv_cpu/bin/python" -m ocr_train export-all --paddleocr "$PO" --out "$OUT" --rec-tag "$REC_TAG" ${EXPORT_ONLY:+--only $EXPORT_ONLY} || die export
fi
step "DONE ($PHASES)"
