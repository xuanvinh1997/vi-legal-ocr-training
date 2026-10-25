"""Phát hiện hướng trang bằng PP-LCNet_x1_0_doc_ori (ONNX, ~6,8 MB, CPU). Tạo doc_ori.onnx: xem README."""
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

LABELS = (0, 90, 180, 270)
_sess = None


def _prep(img):
    img = img.convert("RGB")
    w, h = img.size
    s = 256 / min(w, h)
    img = img.resize((max(224, round(w * s)), max(224, round(h * s))), Image.BILINEAR)
    w, h = img.size
    l, t = (w - 224) // 2, (h - 224) // 2
    x = np.asarray(img.crop((l, t, l + 224, t + 224)), dtype=np.float32) / 255
    x = (x - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    return x.transpose(2, 0, 1)[None].astype(np.float32)


def classify(img, model=Path(__file__).with_name("doc_ori.onnx")):
    """(nhãn, độ tin cậy). Nhãn = góc mà ảnh đang bị xoay theo quy ước của model; dùng `upright()` để sửa."""
    global _sess
    if _sess is None:
        _sess = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    p = _sess.run(None, {_sess.get_inputs()[0].name: _prep(img)})[0][0]
    return LABELS[int(p.argmax())], float(p.max())


def upright(img, min_conf=0.5):
    """(ảnh đã xoay thẳng, góc đã sửa). Nhãn L nghĩa là ảnh lệch L° nên sửa bằng rotate(L) (PIL, ngược chiều kim đồng hồ)."""
    label, conf = classify(img)
    if label == 0 or conf < min_conf:
        return img, 0
    return img.rotate(label, expand=True), label
