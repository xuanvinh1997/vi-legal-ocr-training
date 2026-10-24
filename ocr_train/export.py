"""Trained weights -> Paddle inference model -> ONNX, with the two fixes the stock path does not make for you."""
from __future__ import annotations

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REC_BASE_WIDTH = 320


def export_inference(paddleocr_dir: str | Path, config: str | Path, weights: str | Path, out_dir: str | Path, python: str = sys.executable) -> Path:
    """`tools/export_model.py` of the PaddleOCR checkout. `weights` is the checkpoint prefix, without `.pdparams`.

    Run it with a **CPU build** of Paddle: GPU builds write `linear_v2` operators into `inference.json`, which Paddle2ONNX cannot
    convert ("unsupported operators", seen on Colab). Export needs no GPU, so `use_gpu` is forced off."""
    out_dir = Path(out_dir)
    subprocess.run([python, "tools/export_model.py", "-c", str(config), "-o", f"Global.pretrained_model={weights}", f"Global.save_inference_dir={out_dir}", "Global.use_gpu=False"],
                   cwd=paddleocr_dir, check=True)
    if not (out_dir / "inference.pdiparams").exists():
        raise RuntimeError(f"export produced no model in {out_dir}")
    return out_dir


def patch_rec_inference_yml(model_dir: str | Path, width: int = REC_BASE_WIDTH) -> None:
    """Training uses one wide `RecResizeImg` shape (to fit whole legal lines). PaddleX reads that shape at inference as the
    minimum padded width of *every* image, so a short line would be padded to ~1600 px and run several times slower. Put the
    upstream value back: width is then taken from the widest line in each batch."""
    path = Path(model_dir) / "inference.yml"
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for op in cfg["PreProcess"]["transform_ops"]:
        if "RecResizeImg" in op:
            op["RecResizeImg"]["image_shape"] = [3, 48, width]
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")


_F64_SCALE = re.compile(r'(\{"AT":\{"#":"0\.a_array","D":\[)((?:\{"#":"0\.a_f64","D":[-0-9.eE+]+\},?)+)(\]\},"N":"scale"\})')


def _float32_scale_copy(model_dir: Path, dst: Path) -> int:
    """Paddle 3.3 writes the `scale` attribute of interpolation ops as float64, which Paddle2ONNX (built for float32) rejects with
    "the 0th elementwise MUST be ir::FloatAttribute". Every DB text detector hits it: their FPN upsamples with nearest_interp.
    Rewriting the attribute type changes no value, but Paddle itself then refuses the file, so it is done on a throw-away copy
    that only feeds the converter."""
    text = (model_dir / "inference.json").read_text(encoding="utf-8")
    text, count = _F64_SCALE.subn(lambda m: m.group(1) + m.group(2).replace("0.a_f64", "0.a_f32") + m.group(3), text)
    (dst / "inference.json").write_text(text, encoding="utf-8")
    shutil.copy(model_dir / "inference.pdiparams", dst / "inference.pdiparams")
    return count


def to_onnx(model_dir: str | Path, onnx_path: str | Path, opset: int = 14, python: str = sys.executable) -> Path:
    """paddle2onnx (pin 2.0.2rc3, the version PaddleX ships) reads the PIR `inference.json`. Batch and width stay dynamic."""
    model_dir = Path(model_dir)
    code = "import sys; from paddle2onnx.command import main; sys.exit(main())"
    with tempfile.TemporaryDirectory() as tmp:
        _float32_scale_copy(model_dir, Path(tmp))
        subprocess.run([python, "-c", code, "--model_dir", tmp, "--model_filename", "inference.json", "--params_filename", "inference.pdiparams",
                        "--save_file", str(onnx_path), "--opset_version", str(opset)], check=True)
    return Path(onnx_path)


def verify_onnx(model_dir: str | Path, onnx_path: str | Path, shape: tuple[int, ...], atol: float = 1e-3, seed: int = 0) -> float:
    """Largest absolute difference between Paddle Inference and ONNX Runtime on the same random input. Raises past `atol`."""
    import numpy as np
    import onnxruntime as ort
    from paddle import inference

    x = np.random.default_rng(seed).random(shape, dtype=np.float32)
    config = inference.Config(str(Path(model_dir) / "inference.json"), str(Path(model_dir) / "inference.pdiparams"))
    config.disable_gpu()
    config.disable_mkldnn()  # oneDNN (x86 CPUs) cannot run the float64 interpolation attributes of DB detectors
    predictor = inference.create_predictor(config)
    handle = predictor.get_input_handle(predictor.get_input_names()[0])
    handle.reshape(list(shape))
    handle.copy_from_cpu(x)
    predictor.run()
    expected = predictor.get_output_handle(predictor.get_output_names()[0]).copy_to_cpu()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = session.run(None, {session.get_inputs()[0].name: x})[0]
    if got.shape != expected.shape:
        raise AssertionError(f"shape mismatch: paddle {expected.shape} vs onnx {got.shape}")
    worst = float(np.abs(got - expected).max())
    if worst > atol:
        raise AssertionError(f"ONNX differs from Paddle by {worst:.2e} (> {atol})")
    return worst
