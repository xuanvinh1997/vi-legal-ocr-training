# Tinh chỉnh OCR cho văn bản pháp lý tiếng Việt

Tinh chỉnh một bộ **PaddleOCR nhỏ, chạy song song được**: phát hiện chữ (detection), nhận dạng chữ (recognition) và
bố cục (layout) trên văn bản pháp lý tiếng Việt. Chạy được trên Colab T4: mở
[notebooks/vi_legal_ocr_finetune.ipynb](notebooks/vi_legal_ocr_finetune.ipynb), hoặc chạy không giao diện bằng
[scripts/run_pipeline.sh](scripts/run_pipeline.sh) (notebook gọi đúng script này theo từng pha, nên hai đường giống nhau).

| Mô hình | Điểm xuất phát | Kích thước | Kết quả |
|---|---|---|---|
| Recognition | `PP-OCRv6_small_rec` hoặc `_tiny_rec`, **từ điển tiếng Việt mới** | bản tiny với từ điển compact: ONNX 2,3 MB (đo ở lần chạy thử) | `export/rec` |
| Detection | `PP-OCRv6_small_det` hoặc `_tiny_det` | small ≈ 9,6 MB, tiny ≈ 1,9 MB (số của upstream) | `export/det` |
| Layout | `PP-DocLayout-S` (PicoDet-S, ≈ 4,8 MB), 11 lớp văn bản pháp lý | | `export/layout` |

## Vì sao nhận dạng tiếng Việt kém (phát hiện quyết định thiết kế)

Bộ nhận dạng CTC chỉ xuất được những ký tự có trong từ điển. Từ điển của `latin_PP-OCRv5` và `PP-OCRv6` **thiếu 88–92 trên 134
chữ cái tiếng Việt**: toàn bộ chữ mang dấu thứ hai (ạ ả ấ ầ ẩ ẫ ậ ắ ằ ế ề ể ệ ị ọ ố ồ ợ ụ ử ữ ự ỳ ỷ ỹ …). Thêm dữ liệu
trên từ điển cũ không sửa được. Vì vậy mô hình recognition dùng từ điển mới:

* `compact` (mặc định): ASCII + chữ Việt + chữ Latin-1 + ký hiệu hay gặp trong văn bản pháp lý, **294 lớp**. Lớp đầu ra nhỏ nên mô hình
  nhỏ và nhanh hơn, và không còn ký tự CJK để nhầm với chữ Việt.
* `extend`: từ điển upstream cộng các chữ còn thiếu. Giữ CJK, mô hình lớn hơn.

[ocr_train/head_surgery.py](ocr_train/head_surgery.py) chép lớp phân loại đã huấn luyện sang từ điển mới theo từng cột (chữ mới khởi tạo từ chữ bỏ dấu thanh
của nó, ế ← ê), nên việc huấn luyện bắt đầu từ mô hình đã biết đọc chữ Latin. `python -m ocr_train charset --upstream-dict <file>` in danh sách chữ thiếu.

## Quy trình

```
corpus ─┬─ synth-lines ───────────────► rec/synth_lines ─┐
        └─ synth-pages ─┬─ nhãn det ──► det/synth_pages  ├─► merge ─► train rec / det
PDF thật ─ pdf-pages ───┼─ crop rec ──► rec/*            │
                        └─ COCO ──────► layout ──────────┴──► train layout (PaddleX)
                                                               └─► export ► ONNX ► đối chiếu số học
```

Mọi bước là `python -m ocr_train <lệnh>` (`--help` liệt kê); notebook gọi đúng các lệnh này.

### Dữ liệu

| Shard | Nguồn | Nhãn |
|---|---|---|
| `synth_lines` | Văn bản pháp luật thật từ corpus CC-BY-4.0 [th1nhng0/vietnamese-legal-documents](https://huggingface.co/datasets/th1nhng0/vietnamese-legal-documents) (vbpl.vn), font tương đương Times, làm hỏng kiểu scan; thêm số hiệu, ngày tháng, dẫn chiếu điều khoản, số tiền | chính xác |
| `synth_pages` | Trang A4 đầy đủ theo Nghị định 30/2020/NĐ-CP: quốc hiệu, cơ quan ban hành, số/ngày, tên văn bản, thân văn bản căn đều, bảng, nơi nhận, chữ ký, con dấu | chính xác, cho cả det + layout + rec |
| `pdf` | PDF gốc có text layer (Công báo, xuất từ Word) được render rồi làm hỏng; chữ và hộp lấy từ text layer | chính xác |
| ngoài | Thư mục ảnh dòng `ảnh<TAB>chữ` hoặc dataset ảnh/chữ trên Hugging Face (`import-rec`, `import-hf-rec`) | tùy nguồn |

`python -m ocr_train sources` liệt kê từng nguồn công khai kèm giấy phép (đọc từ dataset card ngày 2026-10-24; kiểm tra lại trước khi phát hành mô hình).
Văn bản quy phạm pháp luật không thuộc đối tượng bảo hộ quyền tác giả ở Việt Nam (Luật Sở hữu trí tuệ, Điều 15); corpus CC-BY vẫn yêu cầu ghi nguồn.

Đường PDF thật **từ chối và đếm**: trang scan (có ảnh phủ cả trang, thường kèm lớp OCR của ai đó) và trang có text layer giải mã ra rác do font
cũ VNI/TCVN3 (`charset.looks_vietnamese`). Những trang đó cần gán nhãn thủ công hoặc mô hình thầy, không được coi là nhãn chuẩn.

### Vòng 2: train tiếp từ checkpoint tốt nhất, thêm dữ liệu, GPU lớn hơn

Không ghi đè vòng 1: dữ liệu mới vào shard mới, run mới vào thư mục riêng.

```
# trên VM mới, sau khi khôi phục data.tar từ Drive
PHASES="setup install more rec" REC_TAG=rec_r2 INIT_FROM=rec/best_accuracy \
  MORE_SEED=1 MORE_LINES=400000 MORE_PAGES=2000 REC_BATCH=128 REC_EVAL=500 REC_LR_SCALE=0.6 bash scripts/run_pipeline.sh
EXPORT_ONLY=rec REC_TAG=rec_r2 PHASES=export bash scripts/run_pipeline.sh     # -> export/rec_r2
```

* `more` sinh `synth_lines_<MORE_TAG>` / `synth_pages_<MORE_TAG>` bằng `--seed` mới (cùng seed sẽ sinh lại đúng các dòng cũ) và `--val-ratio 0`, nên
  `val_list.txt` giữ nguyên và độ chính xác hai vòng so sánh trực tiếp được. Shard trang mới không ghi lại COCO của layout.
* `INIT_FROM` nạp **chỉ trọng số** (optimizer và lịch learning rate bắt đầu lại), khác `RESUME` (khôi phục `latest` cả optimizer và epoch, dùng khi run bị ngắt giữa chừng).
  `check-init` từ chối checkpoint train với từ điển khác, vì PaddleOCR bỏ qua âm thầm tensor lệch shape. `--like` lấy lại bề rộng đầu vào và `max_text_length` của
  config vòng 1 và dừng nếu dữ liệu mới có nhãn dài hơn (PaddleOCR sẽ lặng lẽ bỏ các nhãn đó).
* Script từ chối chạy khi `INIT_FROM` đi với `REC_TAG=rec`, hoặc khi thư mục `REC_TAG` đã có run mà không đặt `RESUME=1`.
* Batch lớn: bộ nhớ tăng theo batch (T4, batch 32: 6,6 GB allocated, 10,8 GB reserved). `REC_LR_SCALE` chưa được kiểm chứng cho batch lớn; nếu lần eval đầu thấp hơn nhiều so với best của vòng 1 thì giảm xuống.

### Khác biệt so với công thức upstream, và lý do

| Thiết lập | Upstream | Ở đây | Lý do |
|---|---|---|---|
| `max_text_length` | 25 | theo dữ liệu (≈ 104) | Nhãn dài hơn bị bộ mã hóa nhãn **bỏ qua âm thầm**. Dòng pháp lý dài ~90 ký tự. |
| Độ rộng ảnh recognition | 320 | theo dữ liệu (≤ 1600) | Dòng 100 ký tự ở chiều cao 48 dài ~1600 px; ép vào 320 thì không đọc nổi. |
| `RecConAug` | bật | tắt | Nối mẫu để đủ 25 ký tự; mẫu của ta đã dài. |
| `RecAug` | bật | tắt | Dữ liệu đã được làm hỏng như scan; trên Colab 2 lõi CPU, `RecAug` kéo thông lượng xuống một nửa. Bật lại bằng `config-rec --rec-aug`. |
| Batch recognition | đệm mọi dòng tới độ rộng cố định | `ds_width`: dòng cùng độ rộng chung một batch | Dòng trung vị rộng ~500 px nhưng dòng dài nhất ~1700 px; đệm hết lên 1600 là lãng phí. Đo trên T4: 45 → ~100 dòng/giây. |
| Batch detection | 16 | 8 | Batch 16 hết bộ nhớ trên T4 (15 GB). |
| Độ rộng `RecResizeImg` trong `inference.yml` | – | vá về 320 khi export | PaddleX dùng nó làm độ rộng pad **tối thiểu** của mọi dòng; để 1600 thì dòng ngắn cũng bị pad lên 1600 px. |
| Xoay khi augment detection | ±45° | ±10° | Văn bản gần nằm ngang; trang bị xoay ngang là việc của mô hình hướng trang. |
| Vùng `###` | – | con dấu, chữ ký | Detection bỏ qua thay vì học như chữ. |

### Môi trường và export

Script dùng hai virtualenv Python 3.12 (Python 3.13 của Colab không có wheel `paddle2onnx`): một bản Paddle **GPU** để train, một bản Paddle **CPU** để export.
Lý do tách: bản GPU ghi toán tử `linear_v2` vào `inference.json` và Paddle2ONNX báo "unsupported operators" (gặp trên Colab với rec small).

`paddle2onnx==2.0.2rc3` (bản PaddleX dùng). Mô hình detection cần một bản vá: Paddle 3.3 ghi thuộc tính `scale` của `nearest_interp` dạng float64,
Paddle2ONNX từ chối với lỗi `ir::FloatAttribute`; `export.to_onnx` sửa trên một bản sao tạm. `export.verify_onnx` so recognition và detection với Paddle trên đầu vào ngẫu nhiên
(tắt oneDNN vì CPU x86 không chạy được thuộc tính float64). Trên T4 lần chạy thử: rec lệch 7e-5.

## Đã kiểm chứng gì, chưa kiểm chứng gì

Chạy đầu-cuối hai lần với PaddlePaddle 3.3.1, PaddleOCR 3.7.0, PaddleX 3.7.2: trên CPU MacBook, và trên **Colab T4** (smoke nhỏ): dựng dữ liệu → merge → check → phẫu thuật head →
`tools/train.py` cho recognition và detection → huấn luyện layout bằng PaddleX → export → ONNX (+ đối chiếu số học) → nạp kết quả qua `TextRecognition`,
`TextDetection`, `LayoutDetection` và `PaddleOCR(...)`. Đường GPU bắt được 5 lỗi mà CPU không có (OOM det batch 16, `linear_v2`, wheel Python 3.13, `MPLBACKEND`, treo khi đóng luồng Hugging Face); cả 5 đã sửa. 27 unit test phủ phần code dữ liệu (các biến `OCR_TEST_FONTS`, `OCR_TEST_CORPUS`, `OCR_TEST_PADDLEOCR`,
`OCR_TEST_PDPARAMS` bật các test cần tải về).

**Chưa kiểm chứng:** mọi con số độ chính xác. Lần chạy thử chỉ chứng minh đường ống nối đúng, không chứng minh mô hình tốt.
`REC_BATCH=32` ở độ rộng 1600 dùng gần hết bộ nhớ T4 (14,3/15,3 GB đã dành); gặp hết bộ nhớ thì giảm. ONNX của layout chỉ kiểm tra chuyển đổi được. Trang tổng hợp dạy lớp và hình học,
không dạy vẻ ngoài scan thật của bạn: sẽ có khoảng cách cho tới khi có trang thật trong tập (PDF cho rec/det, vài trăm trang tự gán nhãn cho layout).

## Chạy song song

Mô hình nhỏ là điều kiện để làm việc này: mỗi mô hình một phiên ONNX Runtime (hoặc một tiến trình) cùng thread pool, giữ `intra_op_num_threads` ở 1–2 cho mỗi phiên để các phiên
không tranh nhau lõi CPU, và gom crop recognition theo độ rộng.

## Đo trên bản quét thật và chạy thử PDF

Điểm đánh giá của PaddleOCR trộn dòng tổng hợp với dòng thật nên không theo kịp chất lượng trên bản quét (vòng r3: điểm đó đứng yên hoặc giảm trong khi CER trên trang thật giảm từ 3,0% xuống dưới 1%). Hai script chạy pipeline của `services/ocr` (monorepo private-ai; đặt `OCR_SERVICE_DIR` nếu repo này nằm chỗ khác) trên model ONNX đã export (`det/`, `rec/`, và `layout/`, `doc_ori/` nếu muốn Markdown và tự xoay trang):

```bash
# CER và số chữ có dấu mũ + dấu thanh đọc sai, trên 3 trang quét có đáp án gõ tay (scripts/real_pages/, PDF ở tests/)
python scripts/eval_real.py --models <thư mục model> [--unclip 2.0] [--json kq.json]
# cả file PDF qua pipeline: document.md, từng trang pNN.md, ảnh layout pNN_layout.png, summary.json
python scripts/run_pdf.py "tests/<file>.pdf" --models <thư mục model> --out <thư mục kết quả> [--unclip 2.0] [--pages 1-5]
```

Cần onnxruntime, opencv, pyclipper (như `services/ocr`), PyMuPDF và RapidFuzz. Model cũ (trước r3) cho CER 3,0% và đọc sai 47% chữ dấu mũ + dấu thanh; `--unclip 2.0` nới khung chữ để giữ dấu nặng dưới chữ. Chỉ có 3 trang, nên đây là phép đo để so các lần train với nhau, không phải độ chính xác tổng quát.

## Test

```bash
pip install -e .[dev]
pytest tests                                                          # test nhanh
OCR_TEST_FONTS=/tmp/fonts OCR_TEST_CORPUS=corpus.jsonl pytest tests   # thêm test bộ sinh dữ liệu
```
