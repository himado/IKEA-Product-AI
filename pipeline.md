# PIPELINE – Tiến độ thực hiện

> File này theo dõi pipeline **theo thứ tự thực hiện**: mỗi bước dùng gì, ra output gì, đã làm đến đâu, kết quả thế nào và bước tiếp theo là gì.
> Thiết kế chi tiết và lý do lựa chọn xem trong [PLAN.md](PLAN.md). Cập nhật file này sau mỗi bước.
>
> Cập nhật lần cuối: **2026-10-15 (chiều)**

**Ký hiệu:** ✅ xong · 🟡 đang làm / làm một phần · ⏸ tạm dừng · ⏭ bỏ qua · ⬜ chưa làm · ⛔ bị chặn

**Tiền tệ:** mọi giá và ngân sách hiển thị bằng **VND** (quyết định 2026-10-15). Catalog vẫn lưu giá IKEA Nhật (JPY) để lọc chính xác; quy đổi theo tỉ giá cố định **1 JPY = 170 VND** trong `backend/currency.py`.

**Python:** máy đã cài thêm Miniconda (Python 3.13) đứng trước trong PATH, nên lệnh `python` giờ **không có** thư viện của dự án. Hãy chạy bằng Python 3.11 của dự án: `C:\Users\Tan\AppData\Local\Programs\Python\Python311\python.exe` (hoặc tạo `.venv` riêng cho dự án).

---

## 1. Tổng quan trạng thái

| # | Bước | Giai đoạn | Công cụ chính | Output | Trạng thái |
|---|---|---|---|---|---|
| 0 | Chốt kiến trúc & kế hoạch | – | – | `PLAN.md` | ✅ |
| 1 | Crawl sản phẩm + mô hình 3D | P3 | Python `requests`, trang danh mục IKEA, JSON-LD | `data/models`, `data/products`, `catalog.jsonl` | ✅ 426 sản phẩm / 18 danh mục |
| 2 | Kiểm tra & lọc mô hình 3D (kích thước, độ nặng, gốc toạ độ) | P3 | `numpy`, `DracoPy`, `matplotlib` | `model_meta.jsonl`, `previews/`, `excluded/` | ✅ giữ 388/426, loại 38 |
| 3 | Làm giàu thuộc tính (enrichment) | P3 | – | – | ⏭ **bỏ qua** (quyết định 2026-10-15); dùng tạm `TYPE_RULES` + `USAGE` viết tay ở B7 |
| 4 | Embedding + index + tìm kiếm | P3 | `multilingual-e5-small` (sentence-transformers), numpy | `data/index/` | ✅ 388 vector · top-1 93% trên bộ test |
| 5 | Unity AR hello world (đặt 1 file `.glb` lên sàn) | P5 | Unity 6, AR Foundation, glTFast + Draco | `unity-app/` (Unity project) → `Builds/DoHaAR.apk` | ✅ **đạt trên Galaxy A55**: 7/7 model load, đo bằng thước khá đúng, 30 FPS |
| 6 | Hiểu ý định người dùng | P2 | Gemini 3.8 → 3.5 → 2.5 Flash (dự phòng) + luật dự phòng, Pydantic | `Intent` (`backend/schemas.py`) | ✅ 30/30 câu test · 7/7 kịch bản từ đầu đến cuối |
| 7 | Gợi ý từ text (lọc cứng + vector + chọn bộ + **rerank LLM**) với phòng giả lập | P4 | `retrieve.py` + `rerank.py` (Gemini) | `Recommendation` (VND, lý do tiếng Việt) | ✅ 7/7 kịch bản đạt mọi kiểm tra cứng, có rerank + lý do |
| 8 | Quét phòng trong app và upload | P1 | AR Foundation (planes, poses, keyframes, độ sâu) + FastAPI; **8b: phòng ảo** (ARCore ghi → thay video → phát lại) | `data/scans/<id>/` | 🟡 chạy trên A55; chưa quét phòng thật có tường, thay bằng phòng ảo |
| 9 | Hình học phòng (sàn, tường, kích thước) | P1 | SegFormer + tia chiếu xuống sàn, giả định tường vuông góc | `room_profile.json` | 🟡 bản đầu chạy trên bản quét phòng ảo |
| 10 | Nhận diện đồ vật có sẵn và nâng lên 3D | P1 | SegFormer (thay YOLO-World), tia chiếu xuống sàn | `room_profile.objects` | 🟡 bản đầu |
| 11 | Hiểu ngữ nghĩa phòng | P1 | Gemini Flash (vision) + luật dự phòng | `room_profile.semantics` | 🟡 bản đầu |
| 12 | Bản đồ không gian trống | P1 | lưới 5 cm, `scipy` | `free_grid.npy` + vùng trống | 🟡 bản đầu |
| 13 | Tìm vị trí đặt (placement engine) | P4 | Luật nhóm (góc đọc sách), giữ tầm nhìn TV ↔ sofa, tránh cửa | vị trí + hướng | 🟡 bản đầu |
| 14 | API `/recommend` hoàn chỉnh (phòng thật) | P2 + P4 | FastAPI `POST /scans/{id}/recommend`, `POST /sim/recommend` + nút **Ask** trong app | `recommendation.json` có toạ độ AR | 🟡 chạy từ điện thoại (20–90 s) |
| 15 | AR tự đặt đồ theo gợi ý + tương tác | P5 | Nút **Show picks**: phát lại + tải model từ server, đặt theo toạ độ AR | Demo trên điện thoại | 🟡 chạy trên phòng ảo (xem lại, chưa tương tác) |
| 16 | Đánh giá + quay demo | – | `backend/eval/run_eval.py` | `eval/report.md` | 🟡 bảng chỉ số xong (phòng ảo); chưa quay video |

**Tiến độ tổng (2026-10-16):** 7/17 bước xong (B0, B1, B2, B4, B5, B6, B7); B3 bỏ qua; **B8–B16 đều đã có bản đầu chạy được từ đầu đến cuối trên phòng ảo** (🟡). Chưa có phòng thật.

```
M0 Dữ liệu + AR hello world   [B1–B5]   ██████████  ✅ B1 ✅ B2 ✅ B3 ⏸ B4 ✅ B5 ✅
M1 Gợi ý từ text             [B6–B7]   ██████████  ✅ B6 ✅ · B7 ✅ (rerank LLM)
M2 Hiểu phòng                [B8–B9,11] ██████░░░░  🟡 bản đầu trên phòng ảo
M3 Đồ vật + vị trí đặt       [B10,12–14] ██████░░░░  🟡 bản đầu
M4 AR hoàn chỉnh             [B15]     █████░░░░░  🟡 xem lại đồ gợi ý trên đoạn phát lại
M5 Đánh giá                  [B16]     ████░░░░░░  🟡 bảng chỉ số, chưa video
```

---

## 2. Chi tiết từng bước

### Bước 0 – Chốt kiến trúc & kế hoạch ✅
- **Output:** `PLAN.md`
- **Kết quả:** đã chốt các lựa chọn sau:
  - Nền tảng: Unity AR Foundation (chạy cả iOS lẫn Android).
  - Quét phòng: **quay ngay trong phiên AR** của app.
  - LLM: Claude API (Sonnet 5.5 và Haiku 4.5).
  - Quy mô: đồ án demo trong 10–12 tuần.

---

### Bước 1 – Crawl sản phẩm + mô hình 3D ✅
**Mục đích:** có catalog sản phẩm IKEA kèm mô hình 3D và kích thước thật. Đây là nguồn dữ liệu cho mọi bước sau.

| | |
|---|---|
| **Input** | 18 danh mục nội thất của IKEA Nhật (`/jp/en/cat/<slug>/`), danh sách nằm trong `CATEGORIES` của crawler |
| **Công cụ** | `crawler/ikea_crawler.py` (Python + `requests`, không cần đăng nhập hay API key) |
| **Cách làm** | (1) Mở trang danh mục để lấy ~30–110 link sản phẩm.<br>(2) Mở trang từng sản phẩm, lấy link `.glb` từ JSON-LD `3DModel` và metadata từ khối `<script type="text/hydrate">`.<br>(3) Mỗi danh mục lấy tối đa N sản phẩm **có 3D**, mỗi dòng sản phẩm tối đa 2 biến thể màu.<br>(4) Kiểm tra mã sản phẩm và thị trường trong trang có khớp URL không, không khớp thì tải lại.<br>(5) Chạy lại được giữa chừng: sản phẩm đã tải được dùng lại từ cache. |
| **Lệnh chạy** | `python crawler/ikea_crawler.py --per-category 25`<br>`python crawler/ikea_crawler.py --per-category 25 --only floor_lamp,table_lamp,rug`<br>`python crawler/ikea_crawler.py --catalog-only` (dựng lại `catalog.jsonl` từ các sản phẩm đã lưu) |
| **Output** | `data/models/<item_no>.glb` (glTF, 415/426 nén Draco)<br>`data/images/<item_no>.jpg`<br>`data/products/<item_no>.json` (có thêm `crawl_category`)<br>`data/catalog.jsonl`<br>`data/crawl.log`, `crawl_lamps.log`, `crawl_rugs.log` |

**Kết quả (2026-10-09), chạy làm 2 lần do lần đầu bị dừng vì thiếu RAM:**
- **426 sản phẩm có 3D, đủ 18/18 danh mục.** Tổng 284 MB model, trung bình khoảng 0,67 MB mỗi model.
- Đã mở 631 trang sản phẩm: 430 trang có 3D, 201 trang không có. Tỉ lệ có 3D khoảng **68%**, cao hơn nhiều so với 45% khi lấy từ sitemap. Thảm có tỉ lệ 3D thấp nhất (25/50).
- Tổng thời gian khoảng 45 phút với độ trễ 1 giây mỗi request.

| Danh mục | SL | Giá (JPY) | Danh mục | SL | Giá (JPY) |
|---|---|---|---|---|---|
| sofa | 25 | 11,990 – 193,990 | desk | 25 | 3,990 – 39,990 |
| armchair | 25 | 8,999 – 82,980 | desk_chair | 25 | 2,990 – 39,990 |
| coffee_side_table | 22 | 1,499 – 24,990 | bed | 25 | 9,990 – 129,980 |
| tv_bench | 21 | 2,499 – 214,970 | bedside_table | 25 | 2,999 – 29,990 |
| ottoman_pouffe | 22 | 1,499 – 29,990 | wardrobe | 20 | 4,599 – 78,000 |
| dining_table | 23 | 4,990 – 139,990 | chest_of_drawers | 20 | 6,999 – 99,990 |
| dining_chair | 25 | 2,000 – 25,000 | bookcase_shelf | 25 | 2,499 – 34,990 |
| sideboard_cabinet | 25 | 3,999 – 125,000 | floor_lamp | 23 | 1,699 – 14,440 |
| rug | 25 | 399 – 79,990 | table_lamp | 25 | 899 – 8,999 |

Một số danh mục dưới 25 sản phẩm vì trang danh mục đã hết sản phẩm có 3D, hoặc bị giới hạn 2 biến thể màu.

**Chất lượng metadata:**
- Thiếu `dimensions_cm`: 15 sản phẩm. Đa số là các bộ (bàn kèm 4 ghế, kệ ghép). Bước 2 lấy kích thước từ model để bù.
- Không có chiều cao: 92 sản phẩm, trong đó có 25 tấm thảm. Bước 2 cũng lấy từ model để bù.
- Thiếu `summary`: 1 sản phẩm.

**Vấn đề gặp và đã xử lý:**
1. ✅ **Lọc theo từ khoá trong slug không chính xác** (từng chọn nhầm đèn tủ YTBERG). Đã chuyển sang lấy sản phẩm trực tiếp từ trang danh mục.
2. ✅ **IKEA đổi cấu trúc trang:** `pageProps` giờ nằm trong `{"0":{"loaderData":...}}`. Parser đã chuyển sang tìm key ở mọi độ sâu, không phụ thuộc vị trí cố định.
3. ✅ **CDN của IKEA đôi khi trả dữ liệu của sản phẩm khác.** Ví dụ trang sofa HEMLINGBY (Nhật) mang dữ liệu rèm BENGTA (Malaysia). Crawler giờ kiểm tra mã sản phẩm và thị trường, sai thì tải lại với cache-buster. Lỗi này xảy ra hàng chục lần ở lần chạy đầu và 2 lần ở lần chạy sau; lần tải lại nào cũng thành công. Đã xoá 1 bản ghi hỏng từ lần chạy trước.
4. ✅ **Lần crawl đầu bị dừng giữa chừng vì máy hết RAM** (Claude Code tự dừng tác vụ nền, không phải lỗi của crawler). Đã thêm `build_catalog()` / `--catalog-only` để dựng lại catalog từ những gì đã lưu, và từ nay catalog được gộp từ mọi lần chạy. 3 danh mục còn thiếu đã được crawl bù bằng `--only`, chạy trực tiếp thay vì chạy nền.
5. ℹ️ Danh mục gốc của IKEA hơi rộng: `desk` có lẫn giá đỡ laptop, `tv_bench` có cả combo tủ TV giá 215k. Bước 3 sẽ chuẩn hoá lại `category_norm`.

---

### Bước 2 – Kiểm tra & lọc mô hình 3D ✅
**Mục đích:** chỉ giữ những mô hình mà khi đưa vào AR sẽ **hiển thị đúng kích thước thật**, **đứng đúng trên sàn** và **đủ nhẹ để điện thoại render mượt**. Mô hình sai kích thước thì người dùng sẽ thấy một chiếc sofa to hơn hoặc nhỏ hơn thực tế, và toàn bộ việc "đặt vừa phòng" mất ý nghĩa.

| | |
|---|---|
| **Input** | `data/catalog.jsonl` (số đo IKEA ghi trên trang sản phẩm) và `data/models/*.glb` (mô hình 3D) |
| **Công cụ** | `pipeline/process_models.py`: đọc GLB bằng `numpy`, giải nén lưới bằng `DracoPy` để vẽ preview, vẽ bằng `matplotlib` |
| **Lệnh chạy** | `python pipeline/process_models.py --previews --exclude`<br>`python pipeline/stats.py` (thống kê theo danh mục) |
| **Output** | `data/model_meta.jsonl` (kích thước chuẩn, số tam giác, trạng thái của từng model)<br>`data/model_report.csv` (mở bằng Excel)<br>`data/previews/<item_no>.png` (3 hình chiếu: trước, cạnh, trên)<br>`data/excluded/` + `data/excluded.json` (các model bị loại và lý do) |

#### Đo dựa trên cơ sở nào?

So **hai nguồn độc lập** cho cùng một sản phẩm:

| Nguồn | Lấy từ đâu | Ví dụ: ghế YNGVAR |
|---|---|---|
| **A. Kích thước IKEA công bố** | Mục *Measurements* trên trang sản phẩm (Width, Depth, Height, Length, Diameter), bước 1 lưu vào `dimensions_cm` | rộng 53 × sâu 51 × cao 71 cm |
| **B. Kích thước thật của file 3D** | Hộp bao (bounding box) của mô hình. Chuẩn glTF quy định đơn vị là **mét** và bắt buộc mỗi lưới phải ghi `min`/`max` toạ độ đỉnh. Script lấy `min`/`max` này, nhân với ma trận biến đổi của các node, rồi lấy `max − min` theo từng trục X (rộng), Z (sâu), Y (cao). | X: −0.2665 → 0.2666 m = **53.3 cm**<br>Z: −0.2539 → 0.2543 m = **50.8 cm**<br>Y: 0 → 0.712 m = **71.2 cm** |

**Sai số** của mỗi chiều = |B / A − 1|. Lấy **chiều lệch nhiều nhất** làm sai số của model. Với YNGVAR: 53.3/53 = 1.006, 50.8/51 = 0.996, 71.2/71 = 1.003, nên sai số lớn nhất là **0.6%**.

Cách ghép chiều:
- IKEA đặt tên chiều ngang không thống nhất (giường dùng *length × width*, kệ dùng *width × depth*). Vì vậy hai chiều ngang được ghép theo thứ tự **lớn với lớn, nhỏ với nhỏ**.
- Nếu IKEA chỉ ghi một chiều ngang thì so với chiều gần nhất của model.
- Đồ tròn dùng *diameter* cho cả hai chiều.
- Chiều cao luôn là trục Y.

Tại sao tin được: A và B do hai quy trình khác nhau tạo ra (số đo sản phẩm thật so với file 3D do bộ phận khác dựng), nên khi khớp nhau thì gần như chắc chắn model đúng kích thước. Nếu lệch, hoặc model sai, hoặc IKEA ghi số đo không đầy đủ. Script không tự phân biệt được hai trường hợp này, nên **mọi model lệch quá ngưỡng đều bị loại**.

#### Tiêu chí giữ / loại

| Tiêu chí | Ngưỡng | Lý do chọn ngưỡng | Kết quả |
|---|---|---|---|
| Sai số kích thước | **≤ 20%** thì giữ (`ok` nếu ≤ 10%, `ok_minor` nếu 10–20%) | Dưới 10% gần như là sai số đo. Từ 10–20% thường do tay nắm, núm hoặc đệm không được tính vào số đo của IKEA. | Loại **25** model lệch > 20% |
| Độ nặng của model | **≤ 100.000 tam giác** và **≤ 5 MB** | Mức thường được khuyến nghị cho asset AR trên điện thoại. Nhiều model nặng cùng lúc sẽ làm giảm FPS và load lâu. | Loại **14** model nặng (13 vượt số tam giác, 1 vượt 5 MB) |
| Gốc toạ độ | đáy ở y = 0, tâm ở x = z = 0 (sai lệch ≤ 1 cm) | Để Unity đặt model thẳng lên điểm anchor trên sàn mà không phải bù | 426/426 đạt |
| Đơn vị | mét | Chuẩn glTF | 426/426 đạt |
| Không có số IKEA để so (`no_reference`) | – | Thường là các bộ như bàn kèm 4 ghế. Vẫn giữ, kích thước lấy từ model. | Giữ 15 |

Có 1 model vừa lệch vừa nặng (BÖJA), nên **tổng cộng loại 38**: 426 → **388 model**.

Các model bị loại được **chuyển** (không xoá) sang `data/excluded/{models,images,products,previews}/`. Lý do từng cái ghi trong `data/excluded.json`. Crawler đọc file này để **không tải lại** những mã đã loại.

#### Kết quả sau khi lọc

| Chỉ số | Giá trị |
|---|---|
| Model giữ lại | **388 / 426**, đủ 18 danh mục |
| Sai số kích thước | `ok` (≤ 10%) 348 · `ok_minor` (10–20%) 25 · `no_reference` 15. Sai số lớn nhất còn lại 19.7%. |
| Độ nặng | Trung vị 7.6k tam giác, lớn nhất 86k. File lớn nhất 4.8 MB. Tổng 244 MB. |
| Gốc toạ độ / đơn vị | 388/388 đáy ở y=0, tâm ở x=z=0, đơn vị mét |
| Hướng mặt trước | **+Z** (chuẩn glTF). Đã xác minh bằng preview của sofa, ghế, giường, bàn. |
| Nén Draco | 378/388 |

**Ảnh hưởng tới từng danh mục:** sofa còn 19 (mất 6: các sofa có chaise và KIVIK nặng), đèn bàn còn 18 (mất 7, chủ yếu do mô hình có kèm dây điện), giường còn 21, ghế làm việc còn 22. Các danh mục khác mất 0–4 model.

**Lý do lệch quan sát được trên preview** (để tham khảo; ghi chú chi tiết trong `data/excluded/review_notes.json`):
- **Model đúng nhưng IKEA ghi số đo không đầy đủ:**
  - Sofa có chaise: IKEA chỉ ghi độ sâu phần ngồi 98 cm, model sâu 165 cm.
  - Đèn có kèm dây điện nằm trên sàn hoặc trên bàn (FADO: quả cầu 25 cm, tính cả dây thì rộng 56 cm).
  - Bàn BURVIK có quai xách cao 77 cm, còn IKEA ghi chiều cao mặt bàn 45 cm.
- **Model là biến thể khác cỡ:** TARVA (model là bản giường 90 cm, sản phẩm là bản 120 cm), đèn BLIDVÄDER, đèn IKEA PS 2026.

Cả hai loại đều đã bị loại theo quy tắc ở trên.

**Quy ước cho các bước sau:**
- `model_meta.dims_cm` (w = trục X, d = trục Z, h = trục Y) là **kích thước chuẩn** để tính footprint và chỗ đặt trong P4. Số IKEA ghi chỉ dùng để hiển thị cho người dùng.
- Unity load model với scale 1, đặt gốc model thẳng lên điểm anchor trên sàn và xoay theo yaw. Không cần bù gì thêm.
- Nếu muốn bù số lượng cho các danh mục bị hụt, chạy lại crawler với `--per-category 25`. Crawler sẽ tự bỏ qua các mã đã loại và lấy sản phẩm khác thay vào, sau đó chạy lại bước 2.

---

### Bước 3 – Làm giàu thuộc tính ⏭ (bỏ qua theo quyết định ngày 2026-10-15; code `pipeline/enrich.py` vẫn giữ lại)
| | |
|---|---|
| **Input** | `catalog.jsonl` (text) + `images/<item_no>.jpg` (ảnh chính) + `model_meta.jsonl` (kích thước thật của model) + `ANTHROPIC_API_KEY` trong `.env` |
| **Công cụ** | `pipeline/enrich.py`: Anthropic Python SDK, `client.messages.parse()` + schema Pydantic (structured outputs, nên JSON trả về luôn đúng schema). Model **`claude-haiku-5-5`**, effort `low`. |
| **Lệnh chạy** | `python pipeline/enrich.py --limit 10` (test; mỗi danh mục lấy 1 sản phẩm trước)<br>`python pipeline/enrich.py` (toàn bộ)<br>`python pipeline/enrich.py --items 60417637 --force` (chạy lại 1 sản phẩm) |
| **Output** | `data/enriched/<item_no>.json` (cache; chạy lại sẽ bỏ qua sản phẩm đã có)<br>`data/enriched.jsonl` (gộp tất cả) |
| **Tại sao** | IKEA không có sẵn các thuộc tính về cách đặt và phong cách, mà bước lọc (B7) và tìm vị trí đặt (B13) đều cần chúng. Bước này cũng sửa lại danh mục rộng (giá đỡ laptop nằm trong `desk`). |

**Schema output** (giá trị bị giới hạn trong danh sách cố định để lọc SQL được):

| Trường | Kiểu / giá trị | Dùng ở bước |
|---|---|---|
| `category_norm` | 28 loại: sofa, armchair, coffee_table, side_table, tv_bench, dining_table, desk, bed, wardrobe, bookcase, floor_lamp, rug, …, `other` | B7 lọc theo loại |
| `placement_type` | floor / wall_mounted / tabletop / ceiling | B13 |
| `against_wall` | required / preferred / free | B13 (tủ bắt buộc sát tường, bàn trà đặt giữa phòng) |
| `front_clearance_cm`, `side_clearance_cm` | số nguyên (cm) | B13 (chừa chỗ mở cửa tủ, kéo ghế, lên giường) |
| `seats` | số chỗ ngồi/nằm hoặc null | B7 ("sofa cho 3 người") |
| `room_types` | living_room, bedroom, dining_room, home_office, kids_room, hallway, kitchen | B7 |
| `styles` | scandinavian, modern, minimalist, industrial, rustic, japandi, … | B7 + rerank |
| `colors`, `materials` | danh sách cố định | B7 |
| `functions` | tag tiếng Anh tự do (storage, reading, extendable…) | B4 embedding |
| `summary_vi` | 1 câu mô tả tiếng Việt | Hiển thị trong app, B4 embedding |

**Kết quả lần chạy thử 10 sản phẩm (2026-10-10): ⛔ chưa có.** API key đọc được từ `.env` và hợp lệ, nhưng API trả `400 – Your credit balance is too low to access the Anthropic API`. Request bị từ chối nên **không tốn tiền**. Script giờ **dừng ngay ở lỗi đầu tiên** thay vì gửi tiếp.

**Chi phí ước tính khi có credit:** mỗi sản phẩm gồm 1 ảnh và khoảng 1–2k token text vào, khoảng 300–600 token ra. Với giá Haiku 5.5 ($0.10 / $0.50 mỗi triệu token), **388 sản phẩm tốn dưới 0,5 USD**. Số thật sẽ được in ra cuối mỗi lần chạy.

> Ghi chú: PLAN ghi Haiku 4.5. Đã chuyển sang **Haiku 5.5**, là bản Haiku hiện hành và rẻ hơn khoảng 10 lần ($0.10 so với $1.00 mỗi triệu token input). Muốn dùng model khác thì truyền `--model`.

### Bước 4 – Embedding + index + tìm kiếm ✅
**Mục đích:** biến mỗi sản phẩm thành một vector "ý nghĩa", để câu hỏi tiếng Việt của người dùng tìm ra đúng sản phẩm IKEA mô tả bằng tiếng Anh. Đồng thời giữ các trường có cấu trúc (giá, kích thước, danh mục) để **lọc cứng** chính xác.

| | |
|---|---|
| **Input** | `data/catalog.jsonl` (388 sản phẩm đã lọc), `data/model_meta.jsonl` (kích thước thật của model), `data/enriched.jsonl` (bước 3; tự gộp vào khi có) |
| **Công cụ** | `pipeline/index.py`: `sentence-transformers` + model **`intfloat/multilingual-e5-small`** (đa ngôn ngữ, có tiếng Việt, 384 chiều, ~470 MB, chạy CPU)<br>`pipeline/search.py`: lọc cứng rồi xếp hạng bằng cosine, trên `numpy` |
| **Lệnh chạy** | `python pipeline/index.py` (khoảng 45 giây)<br>`python pipeline/search.py "ghế đọc sách nhỏ gọn" --max-price 20000 --category armchair`<br>`python pipeline/search.py --eval` (chạy bộ test) |
| **Output** | `data/index/embeddings.npy`: ma trận 388 × 384 float32, đã chuẩn hoá L2 (600 KB)<br>`data/index/products.jsonl`: dòng i = sản phẩm của vector i (các trường lọc + văn bản đã embed)<br>`data/index/meta.json`: tên model, số chiều, thời điểm build |

**Cách làm:**
1. **Ghép văn bản cho mỗi sản phẩm** (`doc_text`), thông tin quan trọng nhất đặt trước:
   - nhãn danh mục **song ngữ** (ví dụ `thảm, thảm trải sàn | rug, carpet`);
   - tên và loại sản phẩm, mô tả, tóm tắt, đường dẫn danh mục IKEA;
   - kích thước thật từ model, vật liệu, vài câu mô tả dài.

   Khi có kết quả bước 3 thì thêm phòng, phong cách, màu và câu mô tả tiếng Việt.
2. **Embed** bằng e5 với tiền tố `passage: ` cho sản phẩm và `query: ` cho câu hỏi (cách model e5 được huấn luyện). Vector được chuẩn hoá nên tích vô hướng chính là cosine.
3. **Tìm kiếm** (`ProductIndex.search`):
   - **lọc cứng** trước theo danh mục, giá min/max, rộng/sâu/cao tối đa;
   - sau đó xếp hạng phần còn lại theo cosine và trả về top-k.

   Bước 7 sẽ gọi đúng hàm này.

**Các điều chỉnh so với PLAN, và lý do:**

| PLAN | Thực tế | Lý do |
|---|---|---|
| `BAAI/bge-m3` (1024 chiều) | `intfloat/multilingual-e5-small` (384 chiều) | Lúc build, máy chỉ còn khoảng **370 MB RAM trống** (16 GB tổng) và ổ C còn 14 GB. bge-m3 cần khoảng 2,5 GB RAM. Đổi lại bằng `python pipeline/index.py --model BAAI/bge-m3` khi máy rảnh. |
| Postgres + pgvector | File numpy, tìm kiếm brute-force | Docker đã cài nhưng **engine chưa chạy**. Postgres 17 có chạy nhưng **chưa có pgvector**. Với 388 sản phẩm, brute-force mất dưới 1 ms nên không cần database. Các file này nạp vào pgvector được mà không phải embed lại. |
| Thuộc tính từ bước 3 | Chưa có (bước 3 tạm dừng) | Thay bằng nhãn danh mục song ngữ viết tay (`CATEGORY_TEXT`) |

**Kết quả đánh giá** (`python pipeline/search.py --eval`, top-5; "đúng" = sản phẩm thuộc danh mục hợp lý cho câu hỏi):

| Bộ test | Lần build 1 (chưa có nhãn song ngữ) | Lần build 2 (có nhãn song ngữ) |
|---|---|---|
| 15 câu chính (ví dụ "ghế đọc sách nhỏ gọn" ≤ ¥20.000, "kệ để tivi", "tủ đầu giường") | Top-1 **47%** · P@5 **52%** | Top-1 **93%** (14/15) · P@5 **85%** |
| 5 câu held-out, diễn đạt gián tiếp, không chứa tên danh mục (ví dụ "nơi cất quần áo cho phòng ngủ") | (chưa đo) | Top-1 **60%** (3/5) · P@5 **56%** |

- Nhãn song ngữ có một số từ trùng với câu test chính, nên con số 93% hơi lạc quan. Bộ held-out (60%) phản ánh khả năng tổng quát thật sát hơn.
- Các câu còn sai:
  - "chỗ ngồi êm ái để xem phim" → trả kệ TV;
  - "trải dưới bàn trà cho ấm chân" → trả bàn trà thay vì thảm;
  - "đèn đọc sách đặt cạnh ghế" → kệ sách đứng đầu.

  Model nhỏ bắt theo từ khoá ("phim" → TV, "bàn trà" → bàn trà) thay vì ý định.
- **Hướng sửa (đã nằm sẵn trong kế hoạch):** bước 6 dùng LLM đọc ý định ra danh mục (ví dụ `rug`) và viết lại câu truy vấn bằng tiếng Anh. Sau đó `search()` **lọc cứng theo danh mục**, nên các lỗi kiểu này biến mất. Thêm vào đó có thể đổi sang bge-m3 khi đủ RAM.
- Truy vấn mẫu theo tiêu chí xong của bước này, "ghế đọc sách nhỏ gọn" ≤ ¥20.000, **đạt khi có lọc danh mục**: `--category armchair` trả về 6 ghế bành ≤ ¥20.000. Không lọc thì kết quả số 1 là ghế bành TULLSTA, nhưng 6/8 kết quả là kệ sách (do chữ "sách"). Xem hình 4 trong `embedding.md`.

### Bước 5 – Unity AR hello world ✅ (đạt trên Galaxy A55, 2026-10-12)
**Mục đích:** kiểm tra **rủi ro lớn nhất còn lại** trên điện thoại thật:
- model IKEA có load được bằng glTFast không;
- có đúng kích thước thật không;
- có chạy mượt không.

**Môi trường đã cài (2026-10-12):**

| Thành phần | Phiên bản | Ghi chú |
|---|---|---|
| Unity Hub | 3.22.2 (MSIX, cài qua winget, cần UAC) | License **Unity Personal** đã kích hoạt |
| Unity Editor | **6000.3.26f1 (Unity 6.3 LTS)** | `C:\Users\Tan\Unity\Hub\Editor`, kèm Android Build Support + OpenJDK + SDK (API 34–37) + NDK |
| Gói Unity | AR Foundation 6.6.2, ARCore 6.6.2, XR Management 4.6.1, Input System 1.20.1, **glTFast 6.20.0**, **Draco 5.5.0** | `unity-app/Packages/manifest.json`, Built-in Render Pipeline |
| glTF-Transform CLI | 4.5.1 | `tools/gltf/` (cài cục bộ bằng npm) |
| Điện thoại | **Samsung Galaxy A55 (SM-A556E), Android 16 / API 36, ARCore 1.56** | Kết nối qua ADB. Kết nối USB đôi khi bị "offline", khởi động lại ADB thì nhận lại. |

**Tự động hoá** (`unity-app/Assets/DoHa/Editor/DoHaSetup.cs`): một lệnh làm tất cả, không cần mở giao diện Unity:
```
Unity.exe -batchmode -quit -nographics -buildTarget Android -projectPath unity-app -executeMethod DoHa.Editor.DoHaSetup.All -logFile data/unity_build.log
adb install -r unity-app/Builds/DoHaAR.apk
```
Lệnh này làm lần lượt:
1. **Cấu hình**: Android, IL2CPP ARM64, min API 25, OpenGL ES 3, bật ARCore loader, Input System, đưa 3 shader glTFast vào *Always Included Shaders*.
2. **Dựng scene**: AR Session, XR Origin + camera AR, Plane/Raycast/Anchor Manager, prefab mặt phẳng, `ARModelPlacer`.
3. **Build APK** (khoảng 31 MB). Lần đầu mất 14 phút, các lần sau 1–3 phút.

**Kết quả thử trên điện thoại:**

| # | Kiểm tra | Kết quả |
|---|---|---|
| 1 | Camera AR + nhận mặt sàn | ✅ nhận **102 mặt phẳng**, lớp xanh hiện trên sàn |
| 2 | Hiệu năng | ✅ **30 FPS**, bạn đánh giá "khá mượt" |
| 3 | Load model nén Draco | ✅ **7/7 model** load được (sau khi sửa lỗi 2 và 3 bên dưới), không model nào bị hồng |
| 4 | Kích thước trong Unity so với file 3D | ✅ **khớp chính xác**: HEKTAR 40,7 × 30 × 176 cm, SKANSNÄS 47,9 × 51 × 78 cm |
| 5 | Thời gian load | ✅ 171–198 ms lần đầu, 0–1 ms khi đặt lại (cache). Model nặng nhất (86k tam giác) chỉ mất 171 ms. |
| 6 | Đo bằng thước (HEMLINGBY 145 cm, SKÅLSTA cao 81 cm) | ✅ **khá đúng**. Bạn đề xuất sau này thêm một lớp *chỉnh scale riêng cho từng model*; hiện giữ nguyên 1:1 cho tất cả. |
| 7 | Mặt trước quay về người dùng · đứng trên sàn · đứng yên khi đi ra xa | ✅ cả 3: mặt trước hướng về người dùng, sát sàn, giữ nguyên vị trí |
| 8 | FPS khi có 5 model | ✅ vẫn **30 FPS** |

**Lỗi gặp và đã sửa:**

| # | Hiện tượng | Nguyên nhân | Cách sửa |
|---|---|---|---|
| 1 | Build lần 2 bị dừng giữa chừng | Máy thiếu RAM (Claude Code tự dừng tác vụ nền) | Tắt tiến trình Unity còn sót, chạy lại khi RAM trống khoảng 7 GB |
| 2 | Mọi model báo `Failed to load`, log không có chi tiết | `GltfImport` được tạo **không có logger**. glTFast tự tải `jar:file://…` | Tự đọc bytes bằng `UnityWebRequest` (cách đã đọc được `models.json`), rồi gọi `LoadGltfBinary`. Thêm `CollectingLogger` và hiện lý do lỗi lên màn hình. |
| 3 | 5/7 model vẫn lỗi `ExtensionUnsupported; EXT_texture_webp` | Texture **WebP**: glTFast/Unity không giải mã được | `export_unity_models.py` tự **chuyển WebP → PNG** bằng glTF-Transform rồi **nén Draco lại**. Kích thước model không đổi (đã kiểm tra bounding box). File có thể nặng hơn, ví dụ kệ IKEA PS 207 KB → 1,4 MB. |
| 4 | Model có thể hiện màu hồng | Shader glTFast (Built-in) nằm trong gói nên không tự vào bản build | Thêm `glTF/PbrMetallicRoughness`, `glTF/PbrSpecularGlossiness`, `glTF/Unlit` vào *Always Included Shaders* |

**Rút ra cho các bước sau:**
- Server gửi model cho app (bước 14) **phải chuyển WebP → PNG/JPEG trước**. Trong catalog có khoảng 84 file dùng WebP. Có thể chạy `make_unity_compatible()` cho toàn bộ catalog một lần.
- Kích thước đo trong Unity = kích thước bước 2 đo từ file, nên **`model_meta.dims_cm` dùng thẳng cho việc đặt đồ trong AR được**.

**Kết luận bước 5:** rủi ro lớn nhất (model IKEA hiển thị đúng kích thước thật trên điện thoại) **đã được gỡ**. Pipeline dữ liệu (bước 1–2) và pipeline hiển thị (Unity + glTFast) khớp nhau.

**Việc để sau (không chặn):**
- Thêm **hệ số scale riêng cho từng model** (mặc định 1.0), ví dụ trường `scale` trong `models.json` hoặc `model_meta`, cho những model sau này đo thấy lệch. Theo đề xuất của bạn, hiện giữ 1:1.
- Chạy `make_unity_compatible()` (WebP → PNG) cho toàn bộ catalog khi làm server gửi model (bước 14).

### Bước 6 – Hiểu yêu cầu người dùng ✅
**Mục đích:** biến câu nói tự nhiên (tiếng Việt, không dấu hoặc tiếng Anh) thành `Intent` có cấu trúc, là đầu vào của bước 7. Ví dụ:

> *"Phòng khách nhỏ, cần sofa 2 chỗ màu xám, bàn trà và một đèn để đọc sách, tổng dưới 1,5 triệu"*
> → `sofa` ("2-seat sofa grey") · `coffee_side_table` ("coffee table") · `floor_lamp` ("reading floor lamp") · ngân sách 1.500.000 VND → **¥8.824**

| | |
|---|---|
| **Input** | Câu yêu cầu (string) |
| **Công cụ** | `backend/llm.py`: gọi Gemini bằng SDK `google-genai` với **structured output** (schema Pydantic), nhiệt độ 0, thinking LOW. Mỗi request tối đa 45 s, có thử lại, **chuỗi model dự phòng 3.8 → 3.5 → 2.5 Flash** và **cache trên đĩa** (`data/llm_cache/`).<br>`backend/intent/parse.py`: prompt + schema + hậu xử lý + **luật dự phòng** khi mọi model đều lỗi |
| **Lệnh chạy** | `python backend/intent/parse.py "câu yêu cầu"`<br>`python backend/intent/eval_intent.py` (bộ test 30 câu)<br>`python backend/recommend/run_scenarios.py --nl` (từ đầu đến cuối với 7 kịch bản) |
| **Output** | `IntentResult`: `intent` (danh sách món: danh mục, câu tìm kiếm tiếng Anh, số lượng, giá tối đa; ngân sách quy ra JPY), **hoặc** `clarification_question` (1 câu hỏi lại); kèm `source` (model nào trả lời, hoặc `rules`) |

**Chia việc giữa LLM và code** (cùng nguyên tắc với PLAN: "LLM lo ngữ nghĩa, code lo tính toán"):

| LLM (Gemini) | Code |
|---|---|
| Món nào thuộc danh mục nào trong 18 danh mục (enum cố định, không bịa được danh mục) | **Quy đổi tiền** ra JPY theo tỉ giá cố định trong `JPY_PER_UNIT` (giả định: 1 JPY = 170 VND, 1 USD = 150 JPY; sửa được) |
| Viết câu tìm kiếm tiếng Anh cho từng món ("compact 2-seat sofa, grey") | Ghế ăn hoặc ghế văn phòng đi kèm bàn thì không cần vùng sàn riêng (`needs_zone = False`) |
| Tách số lượng, số tiền và **đơn vị như người dùng nói** ("1,5 triệu" → 1.500.000 VND) | Đổi tên danh mục sang tên trong catalog (`coffee_table` → `coffee_side_table`) |
| Câu mơ hồ hoặc món không có trong catalog (tủ lạnh) → trả 1 câu hỏi lại | Chỉ có 1 món mà không có ngân sách tổng → lấy giá tối đa của món đó làm ngân sách |

**Kết quả đánh giá** (`eval/intent_cases.json`: 30 câu do tôi soạn, gồm câu nhiều món, câu có số lượng, nhiều cách nói về tiền, câu gián tiếp, câu không dấu, tiếng Anh, câu mơ hồ, món IKEA không bán, và câu bẫy như "đèn ngủ để **trên tủ đầu giường**" hay "kệ tivi dài **1m6**"):

| Bộ phân tích | Kết quả | Ghi chú |
|---|---|---|
| **Gemini** | **30/30 (100%)** | Lần chạy đầu 29/30. Câu sai là "ghế gaming dưới 15000 yên": Gemini đặt 15000 làm *giá tối đa của món* thay vì *ngân sách tổng*. Đã thêm bước chuẩn hoá ở code. Bộ test và bước chuẩn hoá đều do tôi làm, nên **97% (lần đầu) sát với năng lực thật hơn**. |
| Luật dự phòng (không LLM) | 21/30 (70%) | Sai ở câu gián tiếp ("chỗ ngồi xem phim", "trải dưới bàn trà"), "sofa giường" (đọc thành sofa + giường), "đèn **trên** tủ đầu giường" (thêm nhầm tủ). Chỉ dùng khi Gemini không phản hồi. |
| **Từ đầu đến cuối** (câu nói → Gemini → chọn sản phẩm), 7 kịch bản | **7/7 `Intent` giống bản viết tay** · **7/7 qua mọi kiểm tra cứng** | Sản phẩm chọn được **hợp hơn**, nhờ câu tìm kiếm tiếng Anh: S1 ra bàn trà MOXBODA (trước là bàn phụ BRUKSVARA), S3 ra đèn bàn RÖDFLIK (trước là đèn kẹp FORSÅ). Kết quả trong `eval/results_nl/`. |

**Vấn đề gặp phải: API Gemini rất chập chờn trong lúc test**
- Lần gọi đầu tiên mất **245 s**. Sau đó các model báo **503** (quá tải) và **504** (60 s không trả lời).
- Trong bộ test 30 câu: **`gemini-3.8-flash` gần như không phản hồi**. 20 câu do 3.5 Flash trả lời, 10 câu do 2.5 Flash trả lời.
- Thời gian mỗi câu **5–9 s** khi API ổn định, **45–95 s** khi quá tải (gồm cả thời gian chờ thử lại).
- Vì vậy mới thêm timeout, chuỗi model dự phòng, luật dự phòng và cache.
- Mỗi câu tốn khoảng 500 token vào và 80 token ra.
- ⚠️ Với app thật, độ trễ này **quá cao**. Hướng xử lý sau này:
  - gọi bất đồng bộ, có màn hình chờ;
  - hiện ngay kết quả từ luật dự phòng, rồi cập nhật khi Gemini trả lời;
  - hoặc dùng model/endpoint có SLA.

### Bước 7 – Gợi ý sản phẩm ✅ (chọn bộ an toàn + rerank bằng Gemini)
**Mục đích:** từ *yêu cầu* và *căn phòng*, chọn ra **một bộ sản phẩm** đảm bảo **3 điều kiện cứng**:
1. đúng loại đồ;
2. **vừa chỗ** trong phòng, tính cả khoảng trống để sử dụng (mở cửa tủ, ngồi xuống, lối lên giường);
3. **không vượt tổng ngân sách**.

Phần rerank bằng LLM (PLAN §5 4.3) sau này chỉ sắp xếp lại và viết lý do trên danh sách này, nên **không bao giờ** phá được các điều kiện cứng.

| | |
|---|---|
| **Input** | `Intent` (yêu cầu: các món, danh mục, câu mô tả, số lượng, tổng ngân sách) và `RoomProfile` (kích thước phòng + các **vùng trống** hình chữ nhật, mỗi vùng ghi có sát tường hay không). Hiện cả hai **viết tay** trong `eval/scenarios.json`. Sau này bước 6 (LLM) tạo ra `Intent`, bước 1/12 (quét phòng) tạo ra `RoomProfile`, cùng định dạng. |
| **Công cụ** | `backend/schemas.py`: định dạng dữ liệu bằng Pydantic (Intent, RoomProfile, Zone, Placement, Pick, Recommendation)<br>`backend/recommend/retrieve.py`: thuật toán<br>`backend/recommend/run_scenarios.py`: chạy kịch bản + kiểm tra độc lập + vẽ sơ đồ |
| **Lệnh chạy** | `python backend/recommend/run_scenarios.py` |
| **Output** | `eval/results/<kịch bản>.json` (Recommendation: món được chọn, giá, kích thước, vị trí, hướng, lý do tiếng Việt, 3 lựa chọn thay thế)<br>`eval/results/<kịch bản>.png` (**sơ đồ nhìn từ trên xuống**)<br>`eval/results/summary.md` (bảng tổng hợp) |

**Thuật toán** (mỗi yêu cầu):
1. **Lấy ứng viên:** với mỗi món, gọi `ProductIndex.search()` (bước 4): lọc cứng theo danh mục và đơn giá ≤ ngân sách, rồi xếp hạng theo nghĩa, lấy 60 kết quả.
2. **Lọc theo loại** (`TYPE_RULES`): bỏ sản phẩm có `type_name` không khớp danh mục. Trang danh mục IKEA có lẫn đồ khác loại: *ghế xoay* trong "sofa", *tấm lót sàn* trong "ghế văn phòng", *khay laptop* trong "bàn làm việc", *tủ đầu giường* trong "bàn trà". Đây là giải pháp tạm cho tới khi bước 3 có `category_norm`.
3. **Gộp biến thể màu:** mỗi dòng sản phẩm chỉ giữ biến thể giống nhất.
4. **Kiểm tra vừa chỗ** (`place()`): footprint (kích thước thật từ model 3D) cộng khoảng trống sử dụng phải lọt vùng trống.
   - Vùng sát tường: đặt lưng sản phẩm áp tường, mặt quay vào phòng.
   - Vùng trống giữa phòng: đặt giữa vùng, thử cả 2 hướng.
   - Khoảng trống sử dụng (`USAGE`, tạm viết tay; bước 3 sẽ thay): sofa 60 cm phía trước; tủ quần áo 80 cm; giường 50 cm phía trước và 50 cm mỗi bên; bàn ăn 70 cm phía trước và mỗi bên; …
5. **Chọn bộ:** lấy 6 ứng viên tốt nhất của mỗi món, thử mọi tổ hợp. Một tổ hợp hợp lệ khi tổng giá (nhân số lượng) ≤ ngân sách **và** gán được mỗi món một vùng riêng. Thảm được nằm chung vùng với đồ khác; đèn bàn và ghế quanh bàn ăn không cần vùng. Trong các tổ hợp hợp lệ, chọn theo thứ tự ưu tiên:
   1. nhiều món được **đặt tự nhiên** nhất (đồ sát tường ở vùng sát tường, đồ đứng tự do ở vùng giữa);
   2. tổng độ giống nghĩa cao nhất;
   3. rẻ hơn.
6. Nếu không có tổ hợp nào hợp lệ thì trả về `unmet` kèm lý do, **không bao giờ** gợi ý sai.

**Kiểm tra độc lập** (`run_scenarios.py`; logic kiểm tra viết riêng, chỉ dùng chung hằng số SHARES_ZONE với `retrieve.py`):
- tổng tiền ≤ ngân sách;
- đúng danh mục;
- sản phẩm và khoảng trống nằm trong vùng và trong phòng;
- mỗi vùng chỉ một món;
- **footprint các sản phẩm không đè lên sản phẩm khác hay lên khoảng trống của nhau** (trừ thảm).

**Kết quả trên 7 kịch bản (2026-10-12): 7/7 đạt mọi kiểm tra**

| Kịch bản | Ngân sách | Tổng | Được chọn |
|---|---|---|---|
| S1 phòng khách nhỏ 3,5 × 3 m: sofa + bàn trà + đèn cây | ¥60.000 | ¥43.429 | ROCKSJÖN sofa 2 chỗ (sát tường) · BRUKSVARA bàn phụ (giữa phòng) · ISJAKT đèn đọc sách (góc) |
| S2 phòng ngủ 4 × 4 m: giường đôi + tủ đầu giường + tủ quần áo | ¥120.000 | ¥64.970 | STJÄRNÖ giường 142 × 198 · VIKHAMMER tủ đầu giường · SMÅGÖRA tủ quần áo |
| S3 góc làm việc: bàn + ghế xoay + đèn bàn | ¥40.000 | ¥26.979 | TORALD bàn (65 × 40) · SMÖRKULL ghế văn phòng · FORSÅ đèn làm việc |
| S4 phòng ăn: bàn ăn + 4 ghế | ¥50.000 | ¥26.990 | SANDSBERG bàn 110 × 67 · PINNTORP ghế × 4 |
| S5 sofa 3 chỗ dưới ¥10.000 (**cố tình không khả thi**) | ¥10.000 | – | **Đúng là không gợi ý gì**: sofa rẻ nhất trong catalog là ¥11.990 |
| S6 kệ lọt hốc tường rộng 50 cm | – | ¥12.990 | LACK kệ treo tường 30 × 28 cm |
| S7 góc đọc sách: ghế bành + thảm + đèn đứng | ¥30.000 | ¥29.988 | TULLSTA ghế bành · VÄGSKYLT thảm (dưới ghế) · LERSTA đèn |

**Các lỗi phát hiện và đã sửa trong quá trình làm** (từ lần chạy đầu đến lần chạy cuối):

| Lần chạy | Vấn đề | Cách sửa |
|---|---|---|
| 1 | "sofa" ra ghế xoay DYVLINGE; "bàn làm việc" ra khay laptop; "ghế xoay" ra **tấm lót sàn** KOLON | Thêm `TYPE_RULES` lọc theo `type_name` |
| 1 | Sofa bị đặt giữa phòng, bàn trà lại sát tường | Thêm ưu tiên đặt tự nhiên (`zone_preference`). Gán vùng tìm phương án tốt nhất thay vì phương án đầu tiên. |
| 1–2 | S2, S6: kịch bản do tôi khai báo vùng quá nhỏ (vùng giường rộng 200 cm < giường đôi + 2 lối đi; hốc tường không tính lối đi phía trước) | Sửa lại kích thước vùng trong `scenarios.json` |
| 2 | "bàn trà" ra tủ đầu giường KNARREVIK, vì "bed**side table**" chứa "side table" | Thêm loại trừ "bedside" |
| 3 | S7: hai vùng trống chồng nhau; bộ kiểm tra không bắt được | Sửa vùng và thêm kiểm tra **footprint không đè nhau** |

**Hạn chế còn lại** (sẽ được giải quyết ở các bước sau):
- Chất lượng chọn **theo gu** còn đơn giản: chỉ dựa trên độ giống nghĩa của câu mô tả. Ví dụ S3 chọn bàn TORALD 65 × 40 cm, đúng loại và rẻ nhưng hơi nhỏ cho "bàn làm việc máy tính". → Rerank bằng LLM (§5 4.3), cùng thông tin phong cách và chỗ ngồi từ bước 3.
- **Vị trí đặt còn thô:** đặt ở giữa vùng; chưa xét quan hệ giữa các món (bàn trà *trước* sofa, tủ đầu giường *sát* giường, đèn *cạnh* ghế) hay lối đi tổng thể. → Bước 13 (placement engine trên occupancy grid).
- Ghế quanh bàn ăn và ghế văn phòng được coi là không cần vùng riêng. Khoảng trống của bàn ăn (70 cm mỗi phía) đã tính phần chỗ cho ghế.
- **Phòng là giả lập**, vùng trống do tôi vẽ tay. → Bước 8–12 (quét phòng thật).

#### Bước 7 (phần 2) – Rerank bằng LLM ✅ (2026-10-15)

**Khái niệm:** gợi ý sản phẩm chia làm **hai tầng**.
- **Tầng 1 – retrieval** (`retrieve.py`): embedding và luật hình học/ngân sách duyệt cả 388 sản phẩm, ra một **shortlist an toàn**, khoảng 6 ứng viên mỗi món. Ứng viên nào cũng đã vừa chỗ, vừa giá và đúng loại. Mỗi sản phẩm được chấm **riêng lẻ**, theo độ giống chữ (cosine), nên nhanh nhưng thô. Ví dụ "bàn làm việc máy tính" có thể ra bàn 65 × 40 cm.
- **Tầng 2 – rerank** (`rerank.py`): **Gemini đọc cùng lúc** câu yêu cầu, căn phòng và shortlist, rồi chọn **một ứng viên cho mỗi món** sao cho cả bộ hợp ý người dùng (kích thước, công dụng, màu, phong cách, chia ngân sách), và **viết lý do bằng tiếng Việt**.

| | |
|---|---|
| **Input** | `Intent` (bước 6), `RoomProfile`, shortlist của tầng 1 (gồm tên, giá VND, kích thước, vùng vừa, thứ hạng retrieval, mô tả ngắn từ catalog) |
| **Công cụ** | `backend/recommend/rerank.py`: Gemini structured output (`RerankOut`: mỗi món một `item_no` kèm `reason_vi`, thêm `comment_vi` cho cả bộ), thinking MEDIUM, dùng lại `backend/llm.py` (timeout, model dự phòng, cache) |
| **Lệnh chạy** | `python backend/recommend/run_scenarios.py --nl --rerank` (từ câu nói tự nhiên đến rerank) |
| **Output** | `Recommendation` có thêm: `price_vnd`, `total_vnd`, `budget_vnd`, `reason_vi` (lý do Gemini), `fit_vi` (số liệu code đã kiểm tra), `ranker` (model nào chọn, hoặc `score`), `rerank_note` (lý do bị từ chối), `comment_vi`. File: `eval/results_nl_rerank/*.json`, `*.png`, **`compare.md`** (so sánh bộ theo điểm và bộ Gemini chọn, kèm lý do) |

**Rào chắn an toàn** (Gemini không thể làm hỏng kết quả):
1. Gemini **chỉ được trả mã sản phẩm có trong shortlist** của món đó. Mã lạ hoặc thiếu món thì bị từ chối.
2. Bộ Gemini chọn được **kiểm tra lại bằng code**: tổng tiền ≤ ngân sách, và mỗi món sàn có một vùng riêng (`check_bundle`).
3. Nếu bị từ chối hoặc API lỗi, hệ thống **dùng bộ theo điểm** và ghi lý do vào `rerank_note`. App luôn có kết quả hợp lệ.

**Kết quả (7 kịch bản, câu nói bằng VND, 3 lượt chạy):**

| | Lượt 1 (chủ yếu 3.8 Flash) | Lượt cuối (chủ yếu 2.5 Flash, sau khi thêm luật vùng vào prompt) |
|---|---|---|
| Kiểm tra cứng | **7/7 đạt** | **7/7 đạt** |
| Rerank được dùng | 5/6 (S1: cả 3 model báo **503**, nên dùng bộ theo điểm) | 6/6 |
| Gemini đổi bộ | **4/6** | 1/6 |
| Bộ bị từ chối | 0 (một lượt chạy thử: S7 chọn 2 món cùng chỉ vừa vùng z1, **code bắt được và từ chối**) | 0 |

- **Ví dụ đổi bộ hợp lý (S3, góc làm việc):** theo điểm thì chọn bàn TORALD **65 × 40 cm**; Gemini chọn bàn **120 × 60 cm** (BETESHÄST ở lượt 1, MULLSJÖ ở lượt cuối), hợp với "bàn làm việc máy tính". Lý do mẫu: *"Bàn BETESHÄST bằng tre sáng màu, 120 × 60 cm vừa khu vực tường, có ngăn kéo và rãnh quản lý dây cáp, giá 3.398.000 ₫"*.
- **Kiểm tra lý do có bịa không:** đối chiếu các chi tiết Gemini nêu (rãnh dây cáp, ngăn kéo đóng êm, bánh xe tự hãm, đệm mút, thép mạ niken) với dữ liệu IKEA gốc: **có đủ cả**.
- **Phát hiện quan trọng:** chất lượng rerank **phụ thuộc vào model trả lời**. Gemini 3.x có suy nghĩ thì hay sửa lại thứ hạng (4/6). Model dự phòng 2.5 Flash tắt suy nghĩ thì hầu như **giữ nguyên thứ tự retrieval** (1/6). Vì API Gemini 3.x thường quá tải, kết quả thay đổi giữa các lượt chạy.
- **Sửa trong lúc làm:**
  - không gọi Gemini khi danh sách rỗng (S5);
  - ghi rõ lý do bị từ chối;
  - **thêm vào prompt luật "mỗi vùng một món"**, vì lúc đầu Gemini không biết hai món không được chung một vùng.

**Đánh giá của người** (bạn chấm bản v1, lưu ở `eval/results_nl_rerank/compare_v1_rated.md`):

| Kịch bản | Bộ có hợp lý? | Lý do Gemini viết | Nhận xét |
|---|---|---|---|
| S1 phòng khách | Tạm | Hợp lý | Các món đều rẻ hơn nhiều so với ngân sách (dùng 30%) |
| S2 phòng ngủ | Tạm | Hợp lý | Tủ quần áo nhỏ; tiêu ít so với ngân sách (33%) |
| S3 góc làm việc | **Tốt** | – | |
| S4 phòng ăn | **Tốt** | – | |
| S5 không khả thi | **Tốt** | – | |
| S6 hốc tường | **Tốt** | – | |
| S7 góc đọc sách | Tạm | Hợp lý | Tiêu ít so với ngân sách (49%) |

**Nguyên nhân** (đã kiểm tra bằng số):
1. Shortlist phần lớn đã có món đắt hơn (sofa tới 9,35 tr, đèn tới 2,46 tr), nhưng **prompt bảo Gemini "đừng tiêu tiền"**, còn bộ chọn theo điểm khi hoà thì **ưu tiên bộ rẻ hơn**.
2. Shortlist bị cắt ở 6 ứng viên giống nghĩa nhất nên **mất các món cao cấp**. Ví dụ tủ quần áo vừa chỗ có giá tới 13,3 tr, nhưng shortlist chỉ lên tới 6,1 tr.

**Sửa (v2):**
- Prompt: ngân sách là số tiền khách *sẵn sàng chi*; bộ tốt thường dùng khoảng 60–100% ngân sách cho món to hơn, tốt hơn, hợp hơn; tiêu quá ít là điểm trừ, trừ khi khách muốn đồ rẻ hoặc nhỏ.
- Shortlist = 6 ứng viên giống nghĩa nhất **+ 2 ứng viên đắt nhất** trong số còn lại (vẫn vừa chỗ và vừa giá).
- Bộ chọn theo điểm (dùng khi Gemini lỗi): điểm liên quan làm tròn 0,01, hoà thì **chọn bộ dùng ngân sách nhiều hơn**.
- 2.5 Flash được bật suy nghĩ (1.024 token) khi rerank.
- **Sửa thay vì bỏ cả bộ:** nếu bộ Gemini chọn vi phạm luật, code chọn bộ hợp lệ **giữ được nhiều món Gemini chọn nhất** (`ranker = "<model>+repair"`). Món bị thay thì hiển thị lý do bằng số liệu, và bỏ nhận xét chung của Gemini.
- `compare.md` có thêm **% ngân sách đã dùng**.

**Kết quả v2** (`eval/results_nl_rerank/compare.md`; 7/7 đạt kiểm tra cứng):

| Kịch bản | v1 (bạn đã chấm) | v2 | Thay đổi chính |
|---|---|---|---|
| S1 phòng khách | 2,97 tr (30%) | **7,22 tr (72%)** | Sofa ROCKSJÖN 142 cm (thay GLOSTAD 123 cm); **bàn trà kính VITTSJÖ 75 × 75 cm** (thay bàn khay GLADOM 50 cm) |
| S2 phòng ngủ | 6,63 tr (33%) | **12,40 tr (62%)** | Giường TARVA gỗ thông 161 × 210 cm; tủ đầu giường và **tủ quần áo NORDKISA tre rộng 120 cm** (thay tủ 80 cm); đồng bộ tre và gỗ |
| S3 góc làm việc | 6,46 tr (92%) | 6,46 tr (92%) | Giữ nguyên (bạn chấm Tốt) |
| S4 phòng ăn | 5,27 tr (66%) | 5,27 tr (66%) | Giữ nguyên (Tốt) |
| S5 / S6 | – | – | Giữ nguyên (Tốt) |
| S7 góc đọc sách | 2,47 tr (49%) | **3,82 tr (76%)** | Ghế bập bênh POÄNG 2,9 tr. Gemini 2.5 Flash chọn 2 món tranh cùng vùng; **sửa lại được, giữ 2/3 món** Gemini chọn, chỉ thay chiếc đèn. |

**Hạn chế và việc nên làm tiếp:**
- (✅ đã làm ở v2) Bật suy nghĩ cho model dự phòng 2.5 Flash khi rerank (`thinking_budget` > 0) để kết quả ổn định hơn khi 3.x quá tải.
- Thêm **đánh giá của người**: bạn đọc `eval/results_nl_rerank/compare.md` và chấm từng kịch bản "Gemini hợp lý hơn / như nhau / kém hơn".
- Mỗi lần rerank là một lần gọi Gemini, mất 5–10 s khi API ổn định.

#### Bước 7 (phần 3) – Phân bổ ngân sách theo thị trường ✅ (2026-10-15)

**Yêu cầu của bạn:** giá các món trong một bộ phải theo tỉ lệ hợp lý, có quy luật như ngoài thị trường. Ví dụ ghế ăn không thể đắt gần bằng bàn ăn. Trước khi sửa, S4 chọn **ghế TEODORES 1,02 tr trong khi bàn SANDSBERG chỉ 1,19 tr**, tức mỗi ghế bằng 86% giá bàn.

**Nghiên cứu: 2 nguồn**

*A. Hướng dẫn thị trường* (blog của các studio thiết kế và nhà bán lẻ; con số không đồng nhất, chỉ dùng như quy tắc kinh nghiệm):

| Phòng | Phân bổ thường gặp | Nguồn |
|---|---|---|
| Phòng khách | sofa 25–40% (món chính); ghế phụ ~10%; thảm ~10%; bàn trà 8–9%; bàn phụ ~7%; đèn và trang trí trong phần còn lại | Welsh Design Studio, Havenly |
| Phòng ăn | **bàn chiếm phần lớn nhất** (~45–55%); 2–4 ghế cộng lại ~30–40%; "chọn giữa bàn tốt và ghế tốt thì ưu tiên bàn" | Megafurniture, The New Daily |
| Phòng ngủ | khung giường 25–30% (không tính nệm); tủ đồ ~20%; tủ ngăn kéo 15–16%; **tủ đầu giường là chỗ nên tiết kiệm** | Welsh Design Studio, Megafurniture |
| Góc làm việc | **ghế ≥ bàn** (60/40 hoặc 40/30), vì ngồi trên ghế nhiều giờ mỗi ngày | Evetech, FlexiSpot/blog văn phòng, Megafurniture |

*B. Giá thật trong catalog IKEA* (388 sản phẩm, giá trung vị):

| Danh mục | Trung vị (tr ₫) | | Tỉ lệ giá trung vị | |
|---|---|---|---|---|
| sofa | 9,35 | | ghế ăn / bàn ăn | **0,31** |
| bàn ăn | 5,95 | | tủ đầu giường / giường | **0,33** |
| giường | 5,10 | | đèn cây / sofa | **0,18** |
| tủ quần áo | 3,96 | | thảm / sofa | 0,27 |
| bàn làm việc | 3,06 | | bàn trà / sofa | 0,08 |
| ghế làm việc | 2,72 | | ghế làm việc / bàn làm việc | **0,89** |
| ghế ăn | 1,87 | | đèn bàn / bàn làm việc | 0,19 |

→ Cả hai nguồn khớp nhau ở các điểm chính: món chính đắt nhất, ghế ăn khoảng 0,2–0,3 lần bàn, tủ đầu giường và đèn rẻ. Riêng ghế làm việc thì xấp xỉ hoặc hơn bàn.

**Thiết kế** (`backend/recommend/allocation.py`):

| Thành phần | Nội dung |
|---|---|
| **Trọng số theo vai trò** (`WEIGHTS`) | sofa 30, ghế bành 12, thảm 10, kệ TV 10, bàn trà 8, đôn 6, đèn cây 5, đèn bàn 3 · bàn ăn 40, **ghế ăn 8/chiếc** (4 ghế = 44%, bàn 56%) · giường 28, tủ quần áo 20, tủ ngăn kéo 15, tủ đầu giường 7 · **ghế làm việc 30, bàn làm việc 25** |
| **Ngân sách mục tiêu từng món** | trọng số × số lượng / tổng × **85% ngân sách** (chừa 15%; dùng 60–100% là hợp lý) |
| **Luật tỉ lệ giá** (`RATIOS`, so giá 1 chiếc) | ghế ăn ≤ 50% bàn ăn · bàn trà ≤ 60% sofa · đèn cây ≤ 50% sofa và ≤ ghế bành · đèn bàn ≤ 50% bàn làm việc · tủ đầu giường ≤ 50% giường · đôn ≤ 70% sofa · thảm ≤ sofa / ghế bành · ghế làm việc ≤ 2 lần bàn |
| **Độ lệch phân bổ** | Σ\|chi thực − mục tiêu\| / Σ mục tiêu (0 = đúng mục tiêu). Phạt **cả sai tỉ lệ lẫn tiêu quá ít hoặc quá nhiều**. |

**Áp dụng ở 3 chỗ:**
1. **Shortlist:** 6 ứng viên giống nghĩa nhất **+ 2 ứng viên có giá gần mục tiêu nhất**.
2. **Bộ chọn theo điểm:** thứ tự ưu tiên là (đạt luật tỉ lệ) → (đặt tự nhiên) → (độ liên quan − 0,03 × số món × độ lệch phân bổ) → (dùng nhiều ngân sách hơn).
3. **Gemini rerank:** prompt có `target_vnd` cho từng món cùng `price_rules`. Bộ Gemini chọn mà sai luật tỉ lệ thì được **sửa** (giữ nhiều món Gemini chọn nhất trong số các bộ đạt luật).

**Sai lầm trong quá trình làm (đã sửa):**

| Lần thử | Vấn đề | Sửa |
|---|---|---|
| 1 | Đặt "khớp phân bổ" **lên trên** độ liên quan, nên chọn **giường đơn 92 cm** cho "giường đôi" chỉ để khớp giá | Đổi phân bổ thành điểm trừ nhẹ trên độ liên quan |
| 2 | Độ lệch chỉ so **tỉ lệ**, nên bộ rẻ nhưng đúng tỉ lệ vẫn được coi là tốt, và mức dùng ngân sách tụt (S1 72%→52%) | Đo lệch theo **số tiền mục tiêu tuyệt đối** |
| 3 | Embedding **không phân biệt được giường đơn và giường đôi** (0,841 so với 0,843) | Thêm **chiều rộng tối thiểu/tối đa** vào `Intent`: Gemini tách từ câu nói ("giường đôi" → ≥ 140 cm, "lọt hốc 50 cm" → ≤ 50 cm), retrieval **lọc cứng** |
| 4 | Luật dự phòng đọc "bàn ăn **kèm 4 ghế**" thành chỉ có bàn | "ghế" đứng một mình được hiểu theo ngữ cảnh (có bàn ăn → ghế ăn; có bàn làm việc → ghế làm việc). Luật dự phòng tăng từ 21/30 lên **23/30** |

**Kết quả lượt cuối** (`eval/results_nl_rerank/compare.md`; **7/7 đạt kiểm tra cứng và đạt mọi luật tỉ lệ giá**):

| Kịch bản | Tổng / ngân sách | Bộ được chọn (tỉ lệ đáng chú ý) |
|---|---|---|
| S1 phòng khách | 6,54 tr / 10 tr (65%) | sofa ROCKSJÖN 4,59 tr · bàn trà HOLMERUD 90×55 cm 1,36 tr (**30%** giá sofa) · đèn LERSTA 0,60 tr (13%) |
| S2 phòng ngủ | 13,76 tr / 20 tr (69%) | **giường đôi** STJÄRNÖ 142 cm 5,95 tr · tủ đầu giường 1,70 tr (**29%** giá giường) · tủ quần áo tre NORDKISA 120 cm 6,12 tr |
| S3 góc làm việc | 5,78 tr / 7 tr (83%) | bàn KOMPISHÄNG 2,55 tr · **ghế LÅNGFJÄLL 2,89 tr (≥ bàn)** · đèn 0,34 tr |
| S4 phòng ăn | 6,63 tr / 8 tr (83%) | bàn IKEA PS 96×96 cm 2,55 tr · 4 ghế TEODORES 1,02 tr/chiếc (**40%** giá bàn, trước đây 86%) |
| S7 góc đọc sách | 4,16 tr / 5 tr (83%) | ghế bành POÄNG 1,87 tr · thảm LOHALS 130 cm 1,70 tr · đèn 0,60 tr |

**⚠️ Hết hạn mức Gemini:** lượt chạy cuối nhận lỗi `429 RESOURCE_EXHAUSTED: You exceeded your current quota` sau 2 kịch bản đầu (key nhiều khả năng là gói miễn phí giới hạn theo ngày). Từ S3 trở đi, phần hiểu yêu cầu dùng **luật dự phòng** và phần chọn bộ dùng **bộ chọn theo điểm**. Hệ thống vẫn cho kết quả đạt mọi luật. Muốn Gemini rerank kèm ngân sách mục tiêu chạy đủ 7 kịch bản thì cần chờ hạn mức reset hoặc nâng gói.

### Bước 8 – Quét phòng trong app và upload 🟡 (2026-10-15)
**Mục đích:** ghi lại căn phòng **trong cùng phiên AR** sẽ dùng để đặt đồ, nên toạ độ server tính ra dùng thẳng được để đặt đồ, không phải định vị lại camera. Dữ liệu ghi gồm mặt phẳng sàn, tường, bàn kèm đường biên theo mét; ảnh chụp tại các điểm then chốt; vị trí và hướng camera; thông số ống kính.

| | |
|---|---|
| **Input** | Người dùng bấm **`Scan room`**, đi chậm quanh phòng 30–60 s, bấm **`Stop scan`** rồi **`Upload scan`** |
| **App** | `unity-app/Assets/DoHa/Scripts/ARRoomScanner.cs`:<br>• chụp keyframe khi camera di chuyển ≥ 0,4 m hoặc xoay ≥ 20° (cách nhau tối thiểu 0,5 s, tối đa 40 ảnh): ảnh CPU của ARCore 640×480 nén JPEG, vị trí và hướng camera (ma trận 4×4), thông số ống kính đã quy đổi theo kích thước ảnh;<br>• khi upload: thêm mọi mặt phẳng (bỏ mặt phẳng đã bị gộp và mặt phẳng chưa có đường biên), đường biên chuyển sang toạ độ thế giới;<br>• gửi `multipart/form-data` tới `/scans`. Nút và trạng thái nằm chung bảng giao diện của `ARModelPlacer`. |
| **Server** | `backend/server/app.py` (FastAPI): `POST /scans` (kiểm tra định dạng bằng Pydantic, lưu file, tóm tắt, vẽ sơ đồ), `GET /scans`, `GET /scans/{id}`, `GET /scans/{id}/topdown.png`, `GET /health`. Lệnh xử lý lại một bản quét: `python backend/server/app.py --reprocess <id>` |
| **Kết nối điện thoại ↔ máy tính** | Qua cáp USB: `adb reverse tcp:8000 tcp:8000`, app gọi `http://127.0.0.1:8000`. **Bấm đúp `start_backend.bat`** để vừa nối vừa chạy server. App được phép dùng HTTP thường (`insecureHttpOption`). |
| **Output** | `data/scans/<id>/scan.json`, `keyframes/kf_XXX.jpg`, `summary.json`, **`topdown.png`** (sàn xanh, tường đen, mặt ngang khác xám, đường đi camera cam) |
| **Kiểm thử không cần điện thoại** | `python backend/server/test_upload.py`: gửi bản quét giả lập (sàn 4×3 m, 2 tường, 3 ảnh), kiểm tra diện tích 12,0 m², 2 tường, camera cao 1,4 m → **PASS** |

**Lần quét thật đầu tiên** (`20261015_194501_4b6be7`, Galaxy A55):
- **Đạt:** 129 s, camera đi 14,3 m, **40 ảnh** (1,7 MB, rõ nét), **không mất tracking lần nào**, 8 mặt phẳng.
- **Lỗi và cách sửa:**

| Lỗi | Nguyên nhân | Sửa |
|---|---|---|
| Upload báo lỗi kết nối cổng 8000 | Đường nối `adb reverse` bị mất (mất mỗi khi ADB khởi động lại hoặc rút cáp ra cắm lại) | Nối lại. Thêm `start_backend.bat`. |
| **0 mặt tường** | Cấu hình từ bước 5 chỉ cho ARCore nhận mặt ngang | Bật `Horizontal \| Vertical`. Thao tác đặt đồ vẫn chỉ nhận mặt ngang hướng lên. |
| **Sàn sai:** lấy mặt ngang thấp nhất (mảng 2,5 × 4,9 m, thấp hơn 25 cm), nên camera "cao" 1,79 m | Cách chọn sàn quá đơn giản | Sàn = **mặt ngang lớn nhất nằm thấp hơn camera 1,0–2,0 m**. Kết quả: sàn ở +0,03 m, **camera cao 1,54 m**, sàn 29,5 m² (khung bao 6,3 × 9,6 m), trùng đường đi. Mảng thấp hơn được tách thành "mặt ngang thấp hơn sàn". |
| Mặt phẳng trùng hoặc rỗng | ARCore giữ mặt phẳng cũ đã bị gộp; mặt phẳng mới chưa có đường biên | App bỏ `subsumedBy != null` và đường biên < 3 điểm; server bỏ trùng |

- **Lưu ý cho bước 9:** ảnh lưu theo **hướng cảm biến (nằm ngang)** dù cầm máy dọc (`screen_orientation = Portrait`), nên bước 9/10 cần xoay 90° trước khi chạy nhận diện. Thông số ống kính đúng với ảnh 640×480 (fx ≈ fy ≈ 442, cx ≈ 320, cy ≈ 238).

**Lần quét thứ hai** (`20261015_200351_1048fd`, sau khi bật nhận tường): 90 s, 40 ảnh, không mất tracking, **camera cao 1,26 m**, sàn **6,25 m²** ghép từ 3 mảnh (khung bao 3,7 × 4,0 m).
- Upload đầu tiên bị **timeout**: cáp USB tuột khi đi quanh phòng, `adb devices` trống. Cắm lại và upload lại thì thành công (dữ liệu quét vẫn còn trong app).
- 2 mặt đứng ARCore tìm được **không phải tường**: một mặt chỉ cao 17 cm, mặt kia có đỉnh khoảng 0,75 m so với sàn, tức là thành đồ nội thất hoặc len chân tường. Server giờ **phân loại**: đỉnh ≥ 1,2 m so với sàn → tường; thấp hơn → "mặt đứng thấp" (vẽ màu xám đậm).
- **Kết luận: ARCore nhận tường trơn rất kém** (tường một màu, không có hoa văn). Cần một cách khác để có tường, xem các lựa chọn bên dưới.

**Lựa chọn để có tường (chờ bạn quyết định):**

| Cách | Mô tả | Ưu | Nhược |
|---|---|---|---|
| **A. Chạm góc phòng** (đề xuất) | Sau khi quét, người dùng chạm vào **chân tường ở từng góc phòng** trên màn hình. Raycast xuống sàn cho ra các điểm góc, nối lại thành đa giác phòng và các bức tường. | Chính xác, làm được ngay, giống các app đo phòng. Không phụ thuộc tường có hoa văn hay không. | Thêm 20–30 s thao tác cho người dùng |
| B. Depth API của ARCore | Lưu thêm ảnh độ sâu cùng mỗi keyframe; server ghép đám điểm và tìm mặt đứng | Tự động | Phải kiểm tra A55 có hỗ trợ không; phức tạp, dễ nhiễu |
| C. Suy từ mép sàn | Lấy đường bao của vùng sàn làm tường | Không thêm thao tác | Sàn quét chưa kín (đang rời từng mảnh), nên rất không chính xác |

**Bước 8 xong khi:** một lần quét có **sàn + ít nhất 2 tường**, và sơ đồ khớp với căn phòng thật (bạn xác nhận).

### Bước 8b – Phòng ảo thay phòng thật (ARCore ghi → thay video → phát lại) 🟡 (2026-10-16)
**Vì sao:** chưa có phòng thật để quét. Một phòng ảo đóng vai phòng thật. Pipeline quét **không đọc** thông số phòng ảo; thông số đó chỉ là **đáp án** để chấm.
**Cách làm:** ARCore chỉ nhận hình từ camera thật, nên dùng tính năng **Recording & Playback** của ARCore:
1. **Ghi** (điện thoại, `SimRecorder.cs`): màn hình chỉ hiện phòng ảo (`VirtualRoom.cs`, phòng khách 4×5×2,7 m có cửa, cửa sổ, tranh, TV, đồ IKEA thật: sofa ROCKSJÖN, kệ TV FRYKSÅS, kệ BYAKORRE, thảm STOENSE, đèn LAUTERS). Người dùng đứng ở chỗ trống ngẫu nhiên. ARCore ghi `rec.mp4` (video + con quay + gia tốc kế); app dựng ảnh phòng ảo cho từng khung hình theo đúng vị trí camera (`frames/`, `frames.jsonl`).
2. **Thay video** (máy tính, `backend/sim/swap_video.py` + `mp4box.py`): sửa trực tiếp cấu trúc MP4, **chỉ thay mẫu video**, giữ nguyên từng byte của 4 rãnh dữ liệu ARCore (`arcore-video-0`, gyro, accel, custom-event). Ghép ảnh ảo theo thời điểm cảm biến lưu trong `arcore-video-0`; **tự đo và bù độ trễ** giữa ảnh ảo và video (đo được 2 khung hình).
3. **Phát lại** (điện thoại): ARCore chạy **thuật toán gốc** trên video phòng ảo. `ARRoomScanner` chụp keyframe và mặt phẳng rồi upload như bản quét thật (`source = arcore_playback:<file>`); báo cáo diễn biến lưu ở `data/sim/reports/`.

| Lệnh | Việc |
|---|---|
| `python backend/sim/swap_video.py pull` | kéo bản ghi và báo cáo từ điện thoại về `data/sim/` |
| `python backend/sim/swap_video.py make <rec_id>` | tạo các bản `copy` / `reencode` / `virtual` trong `data/sim/playback/` |
| `python backend/sim/swap_video.py push <rec_id>` | đẩy lên `sim/playback` trên điện thoại |

**Kết quả bước thử** (`rec_20261016_095026`, 58 s; so quỹ đạo camera với bản `copy`):

| Bản phát lại | Tracking | Sai lệch quỹ đạo | Mặt phẳng |
|---|---|---|---|
| copy (video thật, đóng gói lại) | 58 s liên tục | — | 20 (17 ngang, 3 đứng), sàn 8,5 m² |
| reencode (nén lại) | 58 s liên tục | 1,3 cm | 18 |
| virtual, chưa bù trễ | mất 13 s | 31 cm, quãng đường 8,9 m (thật 3,8 m) | tối đa 9 ngang, cuối còn 0 |
| **virtual, đã bù trễ** | khởi tạo lại 1 lần + mất 11 s khi nhìn sát tường trơn | **15,6 cm**, quãng đường 3,6 m | tối đa 12 ngang, 0 đứng; upload 3 mặt (sàn 3,7 m²) |

- **Đã chứng minh:** ARCore chấp nhận file đã thay video và chạy được trên hình phòng ảo. A55 **có ảnh độ sâu** (160×90).
- **Chưa đạt:**
  - Tracking bị khởi tạo lại làm hệ toạ độ đổi giữa chừng, nên chưa chấm được vị trí mặt phẳng so với đáp án.
  - App rớt 45 ảnh ảo (dồn ở giây 21–22 và 35–36).
  - Mặt phẳng mất hết sau khi tracking khởi tạo lại.

**Lần 2** (`rec_20261016_173655`, 80 s, sau khi sửa app: dựng ảnh ngay trước khi vẽ, 2 luồng nén, tường loang + ổ cắm/công tắc, lưu mặt phẳng mỗi giây, đánh số lần khởi tạo lại; chấm bằng `python backend/sim/evaluate.py <scan_id>` → `data/sim/eval/`):

| | Lần 1 (đã bù trễ) | **Lần 2** |
|---|---|---|
| Hình ảo bị sót | 45 | 10 (8 ở 0,2 s đầu) |
| Độ trễ còn lại | 2 khung | 1 khung (tự bù) |
| Tracking | khởi tạo lại 1 lần, mất 11 s | **liên tục 80 s** |
| Mặt phẳng | 12 ngang, 0 đứng | 5 ngang, **6 đứng** (mặt sofa, kệ TV, kệ sách, tường bắc) |
| Sàn | — | 1 mặt, **lệch cao 1,4 cm**, 2,6 m² (phủ 10% chỗ đi được: người quay chỉ đi trong khoảng 1,5 × 1,5 m) |
| Sai lệch quỹ đạo | 15,6 cm | 23–29 cm, tỷ lệ đúng (1,02); trôi dần: 14–18 cm ở 3/4 đầu, 43 cm ở 1/4 cuối |

- Còn sai: 5 mặt ngang nằm thấp hơn sàn 17–37 cm và vượt ra ngoài tường đông. Tường bắc lệch khoảng 0,4 m.
- Tiêu cự đo từ chuyển động xoay: ảnh ảo 492 px, ảnh thật 474 px. Phép đo nhiễu, nhưng không thấy chênh lệch FOV lớn.
- Lỗi thao tác: lúc kéo dữ liệu, adb khởi động lại làm mất `adb reverse`, nên lần upload đầu bị lỗi kết nối.

**Bước 8b xong khi:** một lần phát lại phòng ảo tracking liên tục, có sàn khớp đáp án (sai < 5 cm độ cao, phủ > 50% diện tích sàn trống đã đi qua).

### Bước 9–14 – Từ bản quét ra gợi ý có vị trí đặt 🟡 (2026-10-16, bản đầu)
Chạy trên bản quét phòng ảo `20261016_174640_f741c4` (ARCore phát lại, 40 keyframe, 35 ảnh độ sâu). Chỗ nào chưa thấy thì **dự đoán** và đánh dấu `predicted`.

| Bước | File | Cách làm | Kết quả trên bản quét này |
|---|---|---|---|
| 9 Hình học | `backend/space/room_profile.py` | SegFormer gắn nhãn → chân tường (ranh giới tường/sàn) chiếu xuống mặt sàn ARCore → tìm hướng chính, coi các tường vuông góc → mỗi phía chọn đường chân tường mạnh nhất (ARCore trôi nên có đường bị lặp); phía thiếu dữ liệu thì tường đặt ngay sau phần sàn xa nhất đã thấy. Cửa và cửa sổ: tia chiếu từ pixel cửa giao với mặt tường | phòng **4,00 × 4,25 m** (thật 4 × 5), 4 tường đều có bằng chứng; cửa sổ rộng 1,72 m (thật 1,6); cửa ra vào 0,72 m (thật 0,9) |
| 10 Đồ vật | như trên | mỗi cột pixel của món đồ: điểm thấp nhất chiếu xuống sàn là chỗ đặt, điểm cao nhất cho ra chiều cao. Chỉ thấy mặt trước nên độ sâu được dự đoán (sát tường hoặc theo độ sâu thường gặp). Gộp các nhãn trùng (kệ TV bị gắn là cabinet + table + coffee table); món có TV bên trên thì là kệ TV | sofa, **kệ TV** (có TV), kệ thấp (đúng chỗ kệ sách), thảm. Đèn cây không nhận ra |
| 11 Ngữ nghĩa | như trên | Gemini vision với 4 ảnh (mô hình dự phòng 2.5 Flash), luật dự phòng | phòng khách, hiện đại, sàn gỗ, nhiều ánh sáng tự nhiên, 3 nhận xét tiếng Việt |
| 12 Không gian trống | như trên | lưới 5 cm: trừ đồ vật, giữ trống 0,8 m trước cửa; vùng dọc tường (sâu 0,9 m) và vùng giữa phòng; vùng chưa quét tới thì ghi "dự đoán" | 6 vùng; đã thấy 74% sàn |
| 13 Vị trí đặt | `backend/recommend/placement.py` | nhóm góc đọc (ghế bành + đèn cây + bàn phụ) đặt cạnh nhau dọc một tường, không chắn tầm nhìn TV ↔ sofa (chỉ chỗ để chân được lấn vào), tránh cửa, ưu tiên gần cửa sổ và góc phòng | bản đầu (bước 7) đặt ghế giữa thảm, chắn TV → sau khi sửa: **góc đọc dưới cửa sổ** |
| 14 Gợi ý cho phòng quét | `backend/recommend/recommend_scan.py <scan_id> "<yêu cầu>" [--rerank]` | bước 6 (Gemini) → bước 7 (chọn bộ + rerank) → bước 13 → đổi sang toạ độ AR (vị trí trên sàn + hướng) | "ghế bành đọc sách + đèn cây + bàn trà nhỏ, 8 triệu" → POÄNG 2.888.000 ₫ + LAUTERS 1.588.000 ₫ + GUTTANE 1.698.000 ₫ = **6.174.000 ₫** |

Output trong `data/scans/<id>/`: `room_profile.json`, `room_topdown.png`, `free_grid.npy`, `recommendation.json` (kèm `world_position`, `world_yaw_deg`), `recommend_topdown.png`.

**Còn thiếu:**
- kệ sách và đèn cây có sẵn không nhận ra (quay ít tới chỗ đó);
- tường trên nằm ở 4,25 m thay vì 5 m (ARCore trôi);
- bước 14 chưa thành API;
- bước 15 (điện thoại tự hiện đồ theo toạ độ) chưa làm.

### Bước 8–16 ⬜
Chưa bắt đầu. Input, công cụ và output của từng bước đã mô tả trong PLAN.md:

| Bước | Xem PLAN.md mục |
|---|---|
| 7 Rerank bằng LLM | §5 4.3 |
| 8 Quét phòng | §3 1.1 |
| 9 Hình học phòng | §3 1.2 |
| 10 Nhận diện đồ vật | §3 1.3 |
| 11 Ngữ nghĩa phòng | §3 1.4 |
| 12 Không gian trống | §3 1.5 |
| 13 Vị trí đặt | §5 4.2 |
| 14 API hoàn chỉnh | §7 |
| 15 AR đặt đồ | §6 |
| 16 Đánh giá | §9 |

---

## 3. Luồng dữ liệu giữa các bước

```
[B1 crawl] ──► catalog.jsonl + *.glb
                  │
          ┌───────┴────────┐
          ▼                ▼
   [B2 model_meta] ✅  [B3 enrichment]
   (dims chuẩn,        (placement_type,
    ar_usable)          style, room_types)
          └───────┬────────┘
                  ▼
           [B4 DB + vector] ─────────────────────────┐
                                                     ▼
[B8 scan bundle] ─► [B9 hình học] ─┐         [B7 lấy ứng viên + rerank]
                    [B10 đồ vật] ──┼─► room_profile ─┤
                    [B11 ngữ nghĩa]┘        │        │
                    [B12 vùng trống] ◄──────┘        │
                          │                          │
[text người dùng] ─► [B6 intent] ────────────────────┤
                          └──► [B13 vị trí đặt] ─────┤
                                                     ▼
                                     [B14 recommendations.json]
                                                     ▼
                                   [B15 Unity AR: anchor + glTFast]
```

---

## 4. Cấu trúc thư mục hiện tại

```
DoHa/
├── PLAN.md                    ✅ kế hoạch tổng thể
├── pipeline.md                ✅ file này
├── 3d_create.md               ✅ IKEA tạo mô hình 3D thế nào + hướng dẫn tự tạo
├── embedding.md               ✅ Bước 4 giải thích chi tiết + hình minh hoạ
├── .env                       🔒 ANTHROPIC_API_KEY (không chia sẻ)
├── requirements.txt           ✅ requests, numpy, matplotlib, DracoPy
├── crawler/
│   └── ikea_crawler.py        ✅ Bước 1
├── pipeline/
│   ├── process_models.py      ✅ Bước 2
│   ├── stats.py               ✅ thống kê catalog
│   ├── enrich.py              ⏸ Bước 3 (code xong, tạm dừng)
│   ├── index.py               ✅ Bước 4: embedding
│   ├── search.py              ✅ Bước 4: lọc cứng + tìm theo nghĩa, bộ test --eval
│   ├── visualize_index.py     ✅ Bước 4: vẽ 4 hình minh hoạ embedding
│   └── export_unity_models.py ✅ Bước 5: chọn + copy model test sang Unity
├── backend/
│   ├── sim/
│   │   ├── swap_video.py      🟡 Bước 8b: kéo bản ghi ARCore, thay video bằng phòng ảo, đẩy lên điện thoại
│   │   └── mp4box.py          🟡 Bước 8b: đọc/sửa cấu trúc MP4 (giữ nguyên rãnh dữ liệu ARCore)
│   ├── server/
│   │   ├── app.py             🟡 Bước 8: FastAPI nhận bản quét, tóm tắt, vẽ sơ đồ từ trên xuống
│   │   └── test_upload.py     ✅ Bước 8: bản quét giả lập để kiểm thử server
│   ├── schemas.py             ✅ định dạng dữ liệu (Intent, RoomProfile, Recommendation…)
│   ├── llm.py                 ✅ Bước 6: gọi Gemini (structured output, timeout, model dự phòng, cache)
│   ├── currency.py            ✅ tỉ giá cố định JPY ↔ VND, định dạng "1.650.000 ₫"
│   ├── intent/
│   │   ├── parse.py           ✅ Bước 6: câu nói → Intent (Gemini + luật dự phòng)
│   │   └── eval_intent.py     ✅ Bước 6: đánh giá 30 câu
│   └── recommend/
│       ├── retrieve.py        ✅ Bước 7: shortlist + vừa chỗ + chọn bộ trong ngân sách + kiểm tra bộ của Gemini
│       ├── rerank.py          ✅ Bước 7: Gemini chọn bộ từ shortlist + lý do tiếng Việt
│       ├── allocation.py      ✅ Bước 7: trọng số theo vai trò, ngân sách mục tiêu từng món, luật tỉ lệ giá
│       └── run_scenarios.py   ✅ Bước 7: chạy kịch bản + kiểm tra độc lập + vẽ sơ đồ
├── eval/
│   ├── scenarios.json         ✅ 7 kịch bản phòng + yêu cầu viết tay
│   ├── intent_cases.json      ✅ 30 câu test bước 6 + đáp án
│   ├── results_nl/            ✅ 7 kịch bản chạy từ câu nói tự nhiên (--nl)
│   ├── results_nl_rerank/     ✅ 7 kịch bản + rerank Gemini (--nl --rerank), compare.md
│   └── results/               ✅ <kịch bản>.json, <kịch bản>.png (sơ đồ từ trên xuống), summary.md, intent_eval.md/json
├── start_backend.bat          ✅ Bước 8: nối USB với điện thoại (adb reverse) + chạy server
├── tools/gltf/                ✅ glTF-Transform CLI (npm cục bộ): chuyển WebP → PNG, nén Draco
├── unity-app/                 🟡 Bước 5: Unity project (Packages/manifest.json, Assets/DoHa/Scripts/ARModelPlacer.cs, Assets/DoHa/Editor/DoHaSetup.cs, Assets/StreamingAssets/Models/ 7 .glb đã chuyển WebP), Builds/DoHaAR.apk
├── space_understanding/       ⬜ thư mục trống (dành cho Bước 9–12?)
└── data/
    ├── catalog.jsonl          ✅ 388 sản phẩm / 18 danh mục (đã lọc)
    ├── models/                ✅ 398 .glb (388 trong catalog + 10 bản cũ không thuộc danh mục nào)
    ├── images/                ✅ ảnh sản phẩm
    ├── products/              ✅ metadata từng sản phẩm
    ├── previews/              ✅ 388 PNG
    ├── model_meta.jsonl       ✅ Bước 2
    ├── model_report.csv       ✅ Bước 2
    ├── excluded.json          ✅ 38 mã bị loại + lý do (crawler bỏ qua các mã này)
    ├── index/                 ✅ embeddings.npy (388×384), products.jsonl, meta.json, viz/ (4 hình PNG)
    ├── excluded/              ✅ file của 38 model bị loại (models, images, products, previews) + review_notes.json
    ├── crawl.log, crawl_lamps.log, crawl_rugs.log
    └── process_models.log
```
Các thư mục sẽ tạo sau: `pipeline/enrich.py` (B3), `pipeline/index.py` (B4), `backend/` (B6–B14), `unity-app/` (B5, B8, B15), `eval/` (B16), `docker-compose.yml`.

> 11 sản phẩm từ lần crawl đầu (theo sitemap) không thuộc danh mục nào, ví dụ STARKVIND và YTBERG. Chúng vẫn nằm trong `data/products` và `data/models` nhưng **không có trong catalog**, nên các bước sau tự bỏ qua.

---

## 5. ▶ Hiện trạng & bước tiếp theo

### 5.1 Hiện tại chạy được gì (2026-10-16)

| Thành phần | Chạy thế nào | Trạng thái |
|---|---|---|
| Dữ liệu IKEA | 388 sản phẩm có 3D, 18 danh mục | ✅ |
| Hiểu yêu cầu + gợi ý bộ đồ (VND, tỷ lệ giá theo thị trường, lý do tiếng Việt) | `python backend/recommend/run_scenarios.py --nl --rerank` | ✅ 7/7 kịch bản |
| App AR đặt model IKEA đúng kích thước | App **DoHa AR** trên Galaxy A55 | ✅ 30 FPS |
| Quét phòng (ảnh, vị trí camera, mặt phẳng, **độ sâu 160×90**) + upload | nút **Scan room / Upload scan**; server `start_backend.bat` | 🟡 phòng thật chưa có tường |
| **Phòng ảo thay phòng thật** | nút **Virtual room → Record**; máy tính `swap_video.py pull/make/push`; điện thoại **Play** | 🟡 ARCore tracking liên tục 80 s, lệch khoảng 23 cm |
| Hồ sơ phòng: tường, cửa, cửa sổ, đồ có sẵn, kiểu phòng, vùng trống | `python backend/space/room_profile.py <scan_id>` | 🟡 4,00 × 4,25 m (thật 4 × 5) |
| Gợi ý cho phòng đã quét, xếp chỗ theo nhóm, có toạ độ AR | `python backend/recommend/recommend_scan.py <scan_id> "<yêu cầu>" --rerank` hoặc API `POST /sim/recommend` | 🟡 khoảng 9 s mỗi lượt |
| Xem đồ gợi ý trong AR | nút **Show picks** (phát lại + đặt 3 model IKEA) | 🟡 bạn xác nhận hiển thị đúng |
| Bảng đánh giá | `python backend/eval/run_eval.py` → `eval/report.md` | 🟡 |

**Kết quả đánh giá** (`eval/report.md`, phòng ảo):

| Chỉ số | Mục tiêu | Kết quả |
|---|---|---|
| Sai số kích thước phòng | < 10% | 0% và 15% (4,00 × 4,25 m so với 4 × 5 m) |
| Recall đồ vật lớn | ≥ 80% | đúng chỗ 3/3 (kệ sách bị gắn nhãn "bench"), đúng loại 2/3 |
| Trích xuất slot | ≥ 90% | 30/30 (Gemini) |
| Vi phạm ngân sách / tỷ lệ giá | 0% | 0/9 bộ |
| Gợi ý hợp lý (người chấm) | ≥ 70% | cần chấm lại |
| Độ trôi vị trí | < 5 cm | 23 cm (phòng ảo, chưa đo phòng thật) |
| FPS / load model | ≥ 30 fps, < 3 s | 30 FPS, 171–198 ms |

### 5.2 Việc còn lại

| Thứ tự | Việc | Vì sao |
|---|---|---|
| 1 | **Quay video demo:** quét phòng ảo → Ask → Show picks (quay màn hình bằng `adb shell screenrecord`) | phần cuối của bước 16 |
| 3 | Khi có phòng thật: quét lại, chạy `room_profile.py`, so với đo bằng thước | các chỉ số P1/P5 thật |
| 4 | Tương tác trong AR: chạm để đổi, xoay, xoá món; đi lại tự do trong cùng phiên quét | bước 15 đầy đủ |
| 5 | Nhận diện đèn và đồ nhỏ tốt hơn (SegFormer lớn hơn hoặc YOLO-World) | recall đồ vật |

### 5.3 Việc để sau (đã ghi nhận)
- Scale riêng cho từng model (mặc định 1.0).
- Bù số lượng sản phẩm cho các danh mục bị hụt (sofa, đèn bàn, tủ quần áo).
- Đổi model embedding sang `BAAI/bge-m3` khi máy đủ RAM.
- Server Python bị Claude Code tắt khi máy thiếu RAM: chạy bằng `start_backend.bat` cho ổn định.
- Mỗi lần kéo dữ liệu bằng adb mà daemon khởi động lại thì `adb reverse` bị mất: chạy lại `adb reverse tcp:8000 tcp:8000`.

### 5.4 Cần bạn quyết định
- [ ] Chấm lại `eval/results_nl_rerank/compare.md` (Tốt / Tạm / Kém cho mỗi kịch bản), để có chỉ số "gợi ý hợp lý".
- [ ] Quay video demo: cắm điện thoại rồi báo để bắt đầu.

---

## 6. Nhật ký thay đổi

| Ngày | Bước | Thay đổi |
|---|---|---|
| 2026-10-06 | B1 | Viết `crawler/ikea_crawler.py`, crawl 10 sản phẩm có 3D (22 URL thử) |
| 2026-10-08 | B0 | Tạo `PLAN.md`, chốt Unity AR Foundation, quét trong app, Claude API, demo 10–12 tuần |
| 2026-10-09 | – | Tạo `pipeline.md` để theo dõi tiến độ |
| 2026-10-09 | B1 | Crawler chuyển sang lấy từ trang danh mục và có quota mỗi danh mục. Sửa parser theo cấu trúc trang mới của IKEA. Thêm kiểm tra dữ liệu sai trang và tải lại. Thêm `--catalog-only`. Crawl được 353 sản phẩm / 15 danh mục (bị dừng do thiếu RAM). |
| 2026-10-09 | B2 | Viết `pipeline/process_models.py` + `pipeline/stats.py`. 349/353 model dùng được cho AR. Xác nhận đơn vị mét, gốc ở đáy-giữa, mặt trước +Z. Duyệt tay 15 model lệch. |
| 2026-10-09 | B1 | Crawl bù 3 danh mục còn thiếu (đèn cây 23, đèn bàn 25, thảm 25). **Bước 1 xong: 426 sản phẩm / 18 danh mục.** |
| 2026-10-09 | B2 | Chạy lại cho 73 model mới, duyệt tay thêm 10 đèn (8 nhận, 2 loại). **420/426 dùng được cho AR.** |
| 2026-10-10 | B2 | Bỏ cơ chế duyệt tay. Quy tắc mới: **loại mọi model lệch > 20% và mọi model > 100k tam giác hoặc > 5 MB**. Thêm `--exclude` (chuyển file sang `data/excluded/`, ghi `excluded.json`) và crawler bỏ qua các mã đã loại. **Còn 388/426 model.** |
| 2026-10-10 | B3 | Viết `pipeline/enrich.py` (Claude Haiku 5.5, structured outputs + Pydantic, cache theo sản phẩm). Chạy thử 10 sản phẩm: **thất bại do tài khoản hết credit**. |
| 2026-10-10 | – | Viết `3d_create.md`: quy trình IKEA tạo mô hình 3D (suy ra từ metadata file) và hướng dẫn tự tạo bằng Blender, quét ảnh hoặc AI. |
| 2026-10-10 | B3 | **Tạm dừng** bước 3 theo yêu cầu. |
| 2026-10-10 | B4 | Viết `pipeline/index.py` + `pipeline/search.py`. Dùng `multilingual-e5-small` + numpy (máy thiếu RAM, Docker chưa chạy). Thêm nhãn danh mục song ngữ: top-1 từ 47% lên 93% trên 15 câu chính, held-out 60%. **Bước 4 xong.** |
| 2026-10-10 | B4 | Viết `pipeline/visualize_index.py` và `embedding.md` (4 hình: vector, bản đồ t-SNE, độ giống giữa danh mục, truy vấn có/không lọc). Sản phẩm gần nhất cùng danh mục: 94,3%. |
| 2026-10-11 | B5 | Viết `unity-app/Assets/DoHa/Scripts/ARModelPlacer.cs` (chạm → raycast sàn → glTFast load → ARAnchor; giao diện đo kích thước và FPS), `pipeline/export_unity_models.py` (7 model test) và `unity-app/README.md` (hướng dẫn từng nút + bảng kiểm tra). **Chưa build**: máy chưa có Unity, ổ C còn ~4 GB. |
| 2026-10-12 | B7 | Viết `backend/schemas.py`, `backend/recommend/retrieve.py` (lấy ứng viên, lọc loại, vừa chỗ + khoảng trống sử dụng, chọn bộ trong ngân sách, ưu tiên đặt tự nhiên), `run_scenarios.py` (kiểm tra độc lập + sơ đồ từ trên xuống), `eval/scenarios.json` (7 kịch bản). **7/7 đạt.** Rerank LLM chờ credit. |
| 2026-10-12 | B5 | Cài Unity Hub 3.22 + Unity 6.3 LTS (6000.3.26f1) + Android. Viết `DoHaSetup.cs` để tự cấu hình, dựng scene và build APK. Chạy trên **Galaxy A55**: AR ✅ (102 mặt phẳng, 30 FPS); kích thước trong Unity khớp file 3D; load 171–198 ms. Sửa lỗi load (logger + `LoadGltfBinary`) và lỗi **WebP** (chuyển sang PNG bằng glTF-Transform). |
| 2026-10-12 | – | Đổi LLM sang **Gemini 3.8 Flash** (Google key). Key Claude không còn credit. |
| 2026-10-12 | B5 | Bạn thử trên điện thoại: **7/7 model load**, đo bằng thước khá đúng, mặt trước đúng, sát sàn, đứng yên, 30 FPS với 5 model. **Bước 5 xong.** Ghi nhận ý tưởng thêm scale riêng cho từng model (để sau). |
| 2026-10-13 | B6 | Viết `backend/llm.py` (Gemini structured output, timeout 45 s, dự phòng 3.8→3.5→2.5 Flash, cache) và `backend/intent/parse.py` (prompt 18 danh mục, quy đổi tiền ở code, luật dự phòng). Bộ test 30 câu: **Gemini 30/30** (lần đầu 29/30), luật 21/30. Từ đầu đến cuối: 7/7 kịch bản. API Gemini chập chờn (503/504, 3.8 Flash gần như không phản hồi). **Bước 6 xong.** |
| 2026-10-15 | – | Mở lại app DoHa AR trên Galaxy A55 qua ADB: chạy bình thường (30 FPS). Viết lại mục 5: hiện trạng, lộ trình còn lại, việc để sau, việc cần quyết định. |
| 2026-10-15 | B3 | **Bỏ qua** bước 3 theo quyết định. |
| 2026-10-15 | – | Đổi sang **VND**: `backend/currency.py` (1 JPY = 170 VND); 7 kịch bản viết lại ngân sách bằng VND; Pick/Recommendation có `price_vnd`/`total_vnd`. Phát hiện Miniconda chiếm lệnh `python` nên chạy bằng Python 3.11 của dự án. |
| 2026-10-15 | B7 | **Rerank bằng Gemini** (v1) (`rerank.py`): chọn bộ trong shortlist + lý do tiếng Việt, code kiểm tra lại, có dự phòng. 7/7 kịch bản đạt; Gemini đổi bộ 4/6 (3.8 Flash) hoặc 1/6 (2.5 Flash). Lý do đã đối chiếu với dữ liệu IKEA. Thêm luật "mỗi vùng một món" vào prompt sau khi code từ chối một bộ sai. **Bước 7 xong.** |
| 2026-10-15 | B7 | Bạn chấm v1: S3–S6 **Tốt**; S1, S2, S7 **Tạm** (tiêu quá ít so với ngân sách, tủ quần áo nhỏ). Sửa v2: prompt về ngân sách, shortlist thêm 2 món đắt nhất, hoà thì ưu tiên dùng ngân sách, 2.5 Flash có suy nghĩ, **sửa bộ sai luật thay vì bỏ cả bộ**. Mức dùng ngân sách: S1 30%→72%, S2 33%→62%, S7 49%→76%; 7/7 đạt. |
| 2026-10-15 | B7 | **Phân bổ ngân sách theo thị trường** (`allocation.py`): nghiên cứu hướng dẫn thị trường + giá trung vị IKEA → trọng số theo vai trò, ngân sách mục tiêu từng món (85%), luật tỉ lệ giá (ví dụ ghế ăn ≤ 50% bàn). Thêm lọc chiều rộng (giường đôi ≥ 140 cm) vào `Intent`. Luật dự phòng 23/30. Lượt cuối: 7/7 đạt mọi luật tỉ lệ, dùng 65–83% ngân sách. Gemini **hết hạn mức** (429) giữa chừng. |
| 2026-10-15 | B8 | Viết `ARRoomScanner.cs` (keyframe + mặt phẳng + upload) và server FastAPI `backend/server/app.py` (lưu, tóm tắt, sơ đồ từ trên xuống). Kết nối qua `adb reverse`, thêm `start_backend.bat`. Lần quét thật đầu tiên: 40 ảnh, không mất tracking. Sửa: bật nhận tường, chọn sàn theo độ cao camera, bỏ mặt phẳng trùng hoặc rỗng. Đang quét lại để có tường. |
| 2026-10-15 | B8 | Lần quét thứ hai: sàn 6,25 m², camera cao 1,26 m; 2 mặt đứng là thành đồ nội thất. Server phân loại tường (đỉnh ≥ 1,2 m) và mặt đứng thấp. ARCore không nhận được tường trơn → đề xuất cách A: chạm góc phòng. |
| 2026-10-16 | B8b | Phòng ảo thay phòng thật: `VirtualRoom.cs` + `SimRecorder.cs` (ghi ARCore + dựng ảnh phòng ảo), `backend/sim/swap_video.py` (thay video giữ nguyên rãnh IMU). Phát lại: bản copy/reencode chạy như gốc; bản phòng ảo tracking được, sai 31 cm → **15,6 cm** sau khi bù trễ 2 khung hình; còn mất tracking khi nhìn sát tường trơn, mặt phẳng mất sau khi khởi tạo lại. A55 có ảnh độ sâu 160×90. |
| 2026-10-16 | B8b | Lần 2: sửa thời điểm dựng ảnh, nén 2 luồng, tường thật hơn, giữ mặt phẳng khi mất tracking, thêm `evaluate.py`. Kết quả: tracking liên tục 80 s, 6 mặt đứng, sàn lệch cao 1,4 cm; quỹ đạo vẫn trôi 23–29 cm, nhiều mặt ngang sai độ cao. |
| 2026-10-16 | B9–B14 | Bản đầu từ bản quét ra gợi ý có vị trí đặt: `room_profile.py` (tường vuông góc, cửa và cửa sổ, đồ vật, Gemini vision, lưới trống, vùng trống), `placement.py` (góc đọc sách, giữ tầm nhìn TV), `recommend_scan.py` (toạ độ AR). Phòng ảo: đo 4,0 × 4,25 m (thật 4 × 5), nhận đúng sofa, kệ TV, thảm, cửa, cửa sổ; gợi ý 6,17 tr / 8 tr đặt thành góc đọc dưới cửa sổ. |
| 2026-10-16 | B15 | Server thêm `/sim/recommendation` + `/models/<id>.glb` (đổi WebP sang PNG khi cần). App thêm nút **Show picks**: phát lại đoạn quay phòng ảo, đặt 3 món IKEA được gợi ý đúng toạ độ và hướng tính ở bước 14. Thử trên A55: tải đủ gợi ý và 3 model, bạn xác nhận hiển thị đúng. |
| 2026-10-16 | B14 | API gợi ý: `POST /scans/{id}/recommend` và `POST /sim/recommend` (chưa có hồ sơ phòng thì tự dựng, bộ chỉ mục sản phẩm nạp một lần). App thêm nút **Ask**: gõ yêu cầu → server gợi ý → phát lại kèm đồ mới. Thử: "kệ sách + đèn cây, 5 triệu" → DALRIPA + ÅRSTID, 3,9 tr, 92 s. |
| 2026-10-16 | B16 | `backend/eval/run_eval.py` → `eval/report.md`: kích thước phòng lệch 0% và 15%, đồ lớn đúng chỗ 3/3 (đúng loại 2/3), slot 30/30, vi phạm ngân sách 0/9, trôi 23 cm (phòng ảo), 30 FPS. Cập nhật mục 5 (hiện trạng). |
| 2026-10-16 | B14 | Rút thời gian một lượt gợi ý từ 90–113 s xuống **khoảng 9 s**: (1) server nạp sẵn sentence-transformers và mô hình embedding ngay khi khởi động (`/warm`); việc này trước đây tốn 140–160 s ở yêu cầu đầu tiên; (2) `llm.py` nhớ model Gemini đang lỗi (429 thì bỏ qua 10 phút, quá tải thì 1 phút) và không chờ để thử lại khi hết hạn mức. |
