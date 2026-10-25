"""Tạo mps_model/ từ official/ (baidu/Unlimited-OCR): đổi mọi chỗ cứng CUDA sang thiết bị cấu hình được.

    .venv/bin/python setup_mps.py

Code gốc gọi `.cuda()` và `torch.autocast("cuda")`. Ở đây `.cuda()` -> `.to(_DEV)` với _DEV lấy từ biến môi trường
UOCR_DEVICE (mặc định mps nếu có, không thì cpu), và autocast bị bỏ vì model đã load sẵn bfloat16.
"""
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC, DST = HERE / "official", HERE / "mps_model"

PRELUDE = '''import os as _os, torch as _torch
_DEV = _os.environ.get("UOCR_DEVICE") or ("mps" if _torch.backends.mps.is_available() else "cpu")
'''


def patch(text):
    text, n1 = re.subn(r"\.cuda\(\)", ".to(_DEV)", text)
    text, n2 = re.subn(r'torch\.autocast\("cuda", dtype=torch\.bfloat16\)', "__import__('contextlib').nullcontext()", text)
    return text, n1, n2


def main():
    DST.mkdir(exist_ok=True)
    for f in SRC.iterdir():
        if f.suffix == ".safetensors":
            link = DST / f.name
            if not link.exists():
                link.symlink_to(f)
        elif f.is_file():
            if f.suffix == ".py":
                text, n1, n2 = patch(f.read_text(encoding="utf-8"))
                if n1 or n2:
                    text = PRELUDE + text
                    print(f"{f.name}: {n1} x .cuda(), {n2} x autocast")
                (DST / f.name).write_text(text, encoding="utf-8")
            else:
                shutil.copy(f, DST / f.name)


if __name__ == "__main__":
    main()
