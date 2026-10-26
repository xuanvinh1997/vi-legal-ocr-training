from pathlib import Path

from ocr_train import mlflow_sync

# Lines as PaddleOCR 3.7 writes them (a training step, an evaluation, the "best" line after it, and the config echo at the start).
STEP = ("[2026/10/26 03:45:41] ppocr INFO: epoch: [1/10], global_step: 5820, lr: 0.000059, acc: 0.906250, norm_edit_dis: 0.988460, "
        "CTCLoss: 0.105549, NRTRLoss: 0.911482, loss: 1.020283, avg_reader_cost: 0.00075 s, avg_batch_cost: 0.05297 s, avg_samples: 40.0, "
        "ips: 755.21048 samples/s, eta: 3:22:36, max_mem_reserved: 14479 MB, max_mem_allocated: 9564 MB")
CUR = "[2026/10/26 03:43:43] ppocr INFO: cur metric, acc: 0.6687969495808982, norm_edit_dis: 0.9250635413598673, fps: 561.4017022377927"
BEST = ("[2026/10/26 03:43:43] ppocr INFO: best metric, acc: 0.6687969495808982, is_float16: False, "
        "norm_edit_dis: 0.9250635413598673, fps: 561.4017022377927, best_epoch: 1")


def test_a_training_step_gives_its_numbers_at_its_step():
    metrics, step = mlflow_sync.parse_line(STEP)
    got = {k: (v, st) for k, v, st, _ in metrics}
    assert step == 5820 and got["loss"] == (1.020283, 5820) and got["ctc_loss"][0] == 0.105549 and got["nrtr_loss"][0] == 0.911482
    assert got["lr"][0] == 0.000059 and got["train_acc"][0] == 0.90625 and got["samples_per_s"][0] == 755.21048
    assert got["gpu_mem_reserved_mb"][0] == 14479 and got["gpu_mem_allocated_mb"][0] == 9564 and got["epoch"][0] == 1
    assert "eta" not in got  # a time, not a number
    assert metrics[0][3] == 1792986341000  # 2026-10-26 03:45:41 UTC, in ms


def test_an_evaluation_goes_to_the_last_step_and_the_best_line_to_best_metrics():
    metrics, step = mlflow_sync.parse_line(CUR, 2000)
    assert step == 2000 and {k: (round(v, 4), st) for k, v, st, _ in metrics} == {
        "eval_acc": (0.6688, 2000), "eval_norm_edit_dis": (0.9251, 2000), "eval_fps": (561.4017, 2000)}
    best, _ = mlflow_sync.parse_line(BEST, 2000)
    assert {k: v for k, v, _, _ in best} == {"best_eval_acc": 0.6687969495808982, "best_epoch": 1.0}


def test_other_lines_give_nothing_and_keep_the_step():
    for line in ["[2026/10/26 03:39:13] ppocr INFO:     model_size : small", "eval model::  25%|██▌ | 1/4", "", "Traceback (most recent call last):"]:
        assert mlflow_sync.parse_line(line, 77) == ([], 77)


def test_only_complete_lines_are_read_and_the_offset_moves_past_them(tmp_path: Path):
    log = tmp_path / "t.log"
    log.write_bytes((STEP + "\n").encode() + CUR.encode()[:30])  # the second line is still being written
    lines, offset = mlflow_sync.read_new(log, 0)
    assert lines == [STEP] and offset == len(STEP) + 1
    log.write_bytes((STEP + "\n" + CUR + "\reval model::  25%|\r").encode())  # progress bars end in a carriage return
    lines, offset2 = mlflow_sync.read_new(log, offset)
    assert lines == [CUR, "eval model::  25%|"] and mlflow_sync.read_new(log, offset2) == ([], offset2)
    assert mlflow_sync.read_new(tmp_path / "missing.log", 0) == ([], 0)


def test_settings_come_from_the_config_and_the_caller(tmp_path: Path):
    cfg = tmp_path / "c.yml"
    cfg.write_text("Global: {epoch_num: 10, max_text_length: 104, pretrained_model: /w/ckpt/model, eval_batch_step: [2000, 2000]}\n"
                   "Optimizer: {lr: {name: Cosine, learning_rate: 0.0002, warmup_epoch: 1}}\n"
                   "Train: {loader: {batch_size_per_card: 48}, sampler: {scales: [[1600, 32], [1600, 48], [1600, 64]]}}\n"
                   "Architecture: {Backbone: {name: PPLCNetV3}}\n", encoding="utf-8")
    got = mlflow_sync.settings(cfg, {"fonts": "Tinos,Termes"})
    assert got["epochs"] == "10" and got["peak_lr"] == "0.0002" and got["batch_size"] == "48" and got["input_width"] == "1600"
    assert got["init_from"] == "/w/ckpt/model" and got["fonts"] == "Tinos,Termes" and got["lr_schedule"] == "Cosine"
    assert mlflow_sync.settings(None, {"a": "b"}) == {"a": "b"}
