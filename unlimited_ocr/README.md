# Unlimited-OCR trên Mac (MPS)

Thử [baidu/Unlimited-OCR](https://huggingface.co/baidu/Unlimited-OCR) (3B, MIT) trên PDF tiếng Việt trong `tests/`:
OCR từng trang, trả cả layout (nhãn + hộp), tự xoay trang scan bị xoay, chạy lại trang nghi lỗi.

## Cài đặt (làm trong `unlimited_ocr/`)

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python torch torchvision "transformers==4.57.1" pymupdf rapidfuzz addict easydict matplotlib psutil tqdm einops onnxruntime "huggingface_hub[hf_xet]"

# 1. model OCR (~6,7 GB) + vá .cuda() -> MPS
.venv/bin/hf download baidu/Unlimited-OCR --local-dir official
.venv/bin/python setup_mps.py

# 2. model phát hiện hướng trang (PP-LCNet_x1_0_doc_ori, ~6,8 MB), chuyển sang ONNX một lần
.venv/bin/hf download PaddlePaddle/PP-LCNet_x1_0_doc_ori --local-dir doc_ori
uv venv --python 3.12 .venv_conv
uv pip install --python .venv_conv/bin/python paddlepaddle==3.3.1 paddle2onnx==2.0.2rc3 onnx packaging
.venv_conv/bin/paddle2onnx --model_dir doc_ori --model_filename inference.json --params_filename inference.pdiparams --save_file doc_ori.onnx --opset_version 14
```

## Chạy

```bash
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python run.py ../tests/*.pdf --pages 1-4
```

Kết quả ở `results/<pdf>/`: `page_NNN.md` (markdown sạch), `.raw.txt` (output thô), `.layout.json` (`{label, bbox pixel, text}`),
`vis/page_NNN.png` (ảnh vẽ hộp), `merged.md`, `report.json`. Trang đã có kết quả được bỏ qua, `--force` để chạy lại.

* `--chunk N` (mặc định 1): số trang mỗi lần gọi `Multi page parsing.`. **Giữ ở 1.** Gộp 4 trang làm trộn nội dung giữa các trang
  (trang 1 chứa phần cuối trang 3, trang 2 mất) mà không có dấu hiệu lỗi nào; mới thử trên một tài liệu.
* `--mode single`: gundam (1024/640) cho từng trang. `--no-auto-rotate`: tắt tự xoay.
* Tự xoay: `orient.py` phân loại 0/90/180/270 (nhãn L nghĩa là ảnh lệch L°, sửa bằng `rotate(L)`). Trang scan xoay 90° mà không xoay lại
  thì model sinh chữ vô nghĩa rồi lặp, và làm hỏng cả chunk.
* Trang nghi lỗi (`suspect()`: rỗng, vòng lặp, từ lặp, khối trùng tọa độ) được chạy lại riêng, rồi ở chế độ gundam. Chưa bắt được:
  trộn trang, và câu LaTeX bịa ở cuối trang (từng thấy `\( \therefore m = \frac{3}{11} \)`).
* Tọa độ model là 0-999 theo chiều rộng/cao trang; nhãn đã thấy: title, text, list, page_number, page_footnote, header, footer, equation, table, image.
* CER so với text layer chỉ có khi PDF có text layer (hai PDF thử nghiệm là bản scan nên không có); chưa có nhãn gõ tay để đo chất lượng.

Bản MLX cộng đồng (`LoJexLLM/Unlimited-OCR-MLX`) **không dùng được**: lỗi nạp weight (tên `neck`, layout conv, shape `rel_pos`) và
`_get_rel_pos` bỏ trống phần relative position của SAM.
