"""Feed an MLflow tracking server from PaddleOCR's training log.

PaddleOCR logs to a text file (and VisualDL/W&B if asked), not to MLflow. This follows that file while a run is going and records
each step's loss, accuracy, learning rate, throughput and GPU memory, each evaluation's accuracy, the run's settings (from its
config) and, at the end, the config and the log itself as artifacts. It only reads: the training container is not touched, so it
can be started late (it reads the log from the top), stopped and restarted (it keeps its place in a state file).

    MLFLOW_TRACKING_URI=http://127.0.0.1:5055 python -m ocr_train.mlflow_sync --tag rec_r3 --log out/rec_r3_train.log \\
        --config out/configs/rec_r3.yml --status out/status_rec_r3 --state mlflow/data --param lines_scan=400000

Needs `mlflow` and `pyyaml` only (no Paddle)."""
from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

LINE = re.compile(r"^\[(\d{4}/\d\d/\d\d \d\d:\d\d:\d\d)\] ppocr INFO: (.*)$")
TRAIN = re.compile(r"epoch: \[(\d+)/(\d+)\], global_step: (\d+), (.*)")
PAIR = re.compile(r"([A-Za-z_]+): ([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")
#: what PaddleOCR calls a training-step number -> the metric name used here
TRAIN_METRICS = {"lr": "lr", "acc": "train_acc", "norm_edit_dis": "train_norm_edit_dis", "CTCLoss": "ctc_loss", "NRTRLoss": "nrtr_loss",
                 "loss": "loss", "avg_batch_cost": "batch_cost_s", "avg_samples": "batch_samples", "ips": "samples_per_s",
                 "max_mem_reserved": "gpu_mem_reserved_mb", "max_mem_allocated": "gpu_mem_allocated_mb"}
EVAL_METRICS = {"acc": "eval_acc", "norm_edit_dis": "eval_norm_edit_dis", "fps": "eval_fps"}


def _ms(stamp: str) -> int:
    """The log's clock is the container's, which is UTC."""
    return int(datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)


def parse_line(line: str, step: int = 0) -> tuple[list[tuple[str, float, int, int]], int]:
    """(metrics as (name, value, step, timestamp ms), the step to carry on with). A line that is not a step or an evaluation gives none.
    Evaluations carry no step of their own and are put at the last training step seen."""
    found = LINE.match(line.strip())
    if not found:
        return [], step
    ts, text = _ms(found.group(1)), found.group(2)
    train = TRAIN.match(text)
    if train:
        step = int(train.group(3))
        values = {k: float(v) for k, v in PAIR.findall(train.group(4))}
        out = [(TRAIN_METRICS[k], v, step, ts) for k, v in values.items() if k in TRAIN_METRICS]
        return out + [("epoch", float(train.group(1)), step, ts)], step
    if text.startswith("cur metric,"):
        return [(EVAL_METRICS[k], float(v), step, ts) for k, v in PAIR.findall(text) if k in EVAL_METRICS], step
    if text.startswith("best metric,"):
        values = {k: float(v) for k, v in PAIR.findall(text)}
        return [(name, values[k], step, ts) for k, name in (("acc", "best_eval_acc"), ("best_epoch", "best_epoch")) if k in values], step
    return [], step


def read_new(path: Path, offset: int) -> tuple[list[str], int]:
    """The complete lines written since `offset`, and the offset after them. Progress bars end in a carriage return, so those split too;
    a last line without its newline yet is left for the next time."""
    if not path.exists():
        return [], offset
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read()
    cut = max(data.rfind(b"\n"), data.rfind(b"\r"))
    if cut < 0:
        return [], offset
    text = data[: cut + 1].decode("utf-8", errors="replace")
    return [l for l in re.split(r"[\r\n]+", text) if l.strip()], offset + cut + 1


def settings(config: Path | None, extra: dict[str, str]) -> dict[str, str]:
    """The settings worth comparing between runs, from the training config, plus whatever the caller names."""
    out: dict[str, str] = {}
    if config and config.exists():
        import yaml

        cfg = yaml.safe_load(config.read_text(encoding="utf-8")) or {}
        g, opt = cfg.get("Global", {}), (cfg.get("Optimizer", {}) or {}).get("lr", {})
        train = cfg.get("Train", {})
        pick = {"epochs": g.get("epoch_num"), "max_text_length": g.get("max_text_length"), "init_from": g.get("pretrained_model"),
                "dictionary": g.get("character_dict_path"), "eval_every_steps": (g.get("eval_batch_step") or [None, None])[-1],
                "peak_lr": opt.get("learning_rate"), "lr_schedule": opt.get("name"), "warmup_epochs": opt.get("warmup_epoch"),
                "batch_size": (train.get("loader") or {}).get("batch_size_per_card"),
                "input_width": ((train.get("sampler") or {}).get("scales") or [[None]])[-1][0],
                "architecture": ((cfg.get("Architecture") or {}).get("Backbone") or {}).get("name")}
        out.update({k: str(v) for k, v in pick.items() if v is not None})
    out.update(extra)
    return out


def _client():
    from mlflow import MlflowClient

    return MlflowClient()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tag", required=True, help="the run's name")
    p.add_argument("--log", required=True, type=Path), p.add_argument("--config", type=Path), p.add_argument("--status", type=Path)
    p.add_argument("--state", required=True, type=Path, help="a writable folder where the run id and the position in the log are kept")
    p.add_argument("--experiment", default="ocr-rec-vi-legal"), p.add_argument("--poll", type=float, default=10.0)
    p.add_argument("--param", action="append", default=[], metavar="KEY=VALUE"), p.add_argument("--artifact", action="append", default=[], type=Path)
    args = p.parse_args(argv)

    from mlflow.entities import Metric, Param, RunTag

    client = _client()
    args.state.mkdir(parents=True, exist_ok=True)
    state_file = args.state / f"{args.tag}.json"
    state = json.loads(state_file.read_text()) if state_file.exists() else {}
    if "run_id" not in state:
        exp = client.get_experiment_by_name(args.experiment)
        exp_id = exp.experiment_id if exp else client.create_experiment(args.experiment)
        run = client.create_run(exp_id, run_name=args.tag, tags={"source": "ocr_train.mlflow_sync", "log": str(args.log)})
        extra = dict(item.split("=", 1) for item in args.param)
        client.log_batch(run.info.run_id, params=[Param(k, v[:500]) for k, v in settings(args.config, extra).items()])
        state = {"run_id": run.info.run_id, "offset": 0, "step": 0}
        state_file.write_text(json.dumps(state))
        print(f"run {args.tag}: {run.info.run_id}", flush=True)
    run_id, offset, step = state["run_id"], state["offset"], state["step"]

    while True:
        lines, offset = read_new(args.log, offset)
        batch = []
        for line in lines:
            metrics, step = parse_line(line, step)
            batch += [Metric(k, v, ts, st) for k, v, st, ts in metrics]
        for i in range(0, len(batch), 900):
            client.log_batch(run_id, metrics=batch[i:i + 900])
        state_file.write_text(json.dumps({"run_id": run_id, "offset": offset, "step": step}))
        status = args.status.read_text().strip() if args.status and args.status.exists() else ""
        if not lines and (status.startswith("done") or status.startswith("FAILED")):
            for path in [args.config, args.log, *args.artifact]:
                if path and path.exists():
                    client.log_artifact(run_id, str(path))
            client.set_terminated(run_id, "FAILED" if status.startswith("FAILED") else "FINISHED")
            print(f"run {args.tag}: {status}", flush=True)
            return 0
        time.sleep(args.poll)


if __name__ == "__main__":
    raise SystemExit(main())
