# Bước 4 – Embedding: biến sản phẩm thành con số để "tìm theo nghĩa"

> **Một câu tóm tắt:** mỗi sản phẩm IKEA (một file JSON) được viết thành **một đoạn văn**. Đoạn văn được một mô hình ngôn ngữ biến thành **384 con số** (gọi là vector). Câu hỏi của người dùng cũng được biến thành 384 con số theo cùng cách. Hai dãy số **càng giống nhau** (cosine càng cao) thì sản phẩm **càng hợp** với câu hỏi. Vì mô hình hiểu nhiều ngôn ngữ, câu hỏi **tiếng Việt** vẫn tìm ra được sản phẩm IKEA mô tả bằng **tiếng Anh**.
>
> Code: `pipeline/index.py` (tạo vector), `pipeline/search.py` (tìm kiếm), `pipeline/visualize_index.py` (vẽ các hình bên dưới). Cập nhật: 2026-10-10.

---

## 1. Biến cái gì thành cái gì – theo dõi một sản phẩm thật

Ví dụ: sofa **HEMLINGBY** (`10295025`).

```
 ① JSON sản phẩm            ② Đoạn văn (doc)            ③ Token                ④ Mô hình e5-small        ⑤ Vector
 data/products/*.json  ──►   1 đoạn ~230 token     ──►   271 mảnh từ     ──►   12 lớp transformer  ──►  384 số thực
 (tên, mô tả, vật liệu,      nhãn danh mục song ngữ      ('▁ghế','▁sofa',…)    → 271 vector 384 chiều    độ dài = 1
  danh mục, giá, …)          + tên + mô tả + kích thước                         → lấy trung bình          [0.047, -0.015, …]
 + kích thước thật từ        + vật liệu                                          (mean pooling)
   model_meta.jsonl                                                              → chuẩn hoá về độ dài 1
```

### ① Đầu vào: dữ liệu đã có từ bước 1–2
- `data/catalog.jsonl`: tên, loại, mô tả, danh mục, vật liệu, giá của 388 sản phẩm.
- `data/model_meta.jsonl`: kích thước **thật** đo từ file 3D.

### ② Ghép thành một đoạn văn (`doc_text()` trong `index.py`)
Thông tin quan trọng nhất đặt **lên đầu**. Mô hình chỉ đọc tối đa 512 token; đoạn văn của ta trung bình 229 token, dài nhất 351, nên không bị cắt. Đoạn văn thật của HEMLINGBY:

> **ghế sofa, sofa, ghế băng dài phòng khách | sofa, couch.** HEMLINGBY Two-seat sofa. Two-seat sofa, Bomstad black HEMLINGBY sofa is ideal wherever space is limited - in front of a bay window, in a student flat … Category: Sofas & armchairs > Sofas > Two-seat fabric sofas **Size: width 145.0 cm, depth 71.5 cm, height 71.5 cm.** Materials: Coated fabric parts: 100% polyurethane … A sofa with small, neat dimensions which is easy to furnish with, even when space is limited.

Phần in đậm đầu tiên là **nhãn danh mục song ngữ** do ta tự thêm vào (bảng `CATEGORY_TEXT`, 18 danh mục). Vì dữ liệu IKEA toàn tiếng Anh, nhãn này giúp câu hỏi tiếng Việt khớp tốt hơn rất nhiều (xem §5).

### ③ Cắt thành token
Tokenizer **SentencePiece** (bộ từ vựng 250.037 mảnh, có tiếng Việt) cắt đoạn văn thành **271 token**. Phía trước có thêm tiền tố `passage: `, vì mô hình e5 được huấn luyện như vậy: tài liệu dùng `passage:`, câu hỏi dùng `query:`.
```
['<s>', '▁passage', ':', '▁ghế', '▁sofa', ',', '▁sofa', ',', '▁ghế', '▁băng', '▁dài', '▁phòng', '▁khách', '▁', '|', '▁sofa', …]
```

### ④ Mô hình biến token thành vector
- **Mô hình:** `intfloat/multilingual-e5-small`
  - kiến trúc BERT, 12 lớp transformer, 384 chiều;
  - khoảng 118 triệu tham số, trong đó phần lớn là bảng từ vựng 250k từ;
  - huấn luyện trên khoảng 100 ngôn ngữ để **câu cùng nghĩa ở các ngôn ngữ khác nhau cho ra vector gần nhau**.
- Mỗi token đi qua 12 lớp và ra một vector 384 chiều. Lấy **trung bình** 271 vector đó (mean pooling) được **1 vector cho cả đoạn văn**.
- **Chuẩn hoá** vector về độ dài 1. Khi đó tích vô hướng giữa hai vector chính là **cosine**, nằm trong khoảng −1…1, càng cao càng giống nghĩa.

### ⑤ Đầu ra: 384 con số
```
HEMLINGBY sofa = [0.0472, -0.0149, -0.0262, -0.0765, 0.0471, -0.0609, 0.0252, 0.0251, … (384 số)]   độ dài = 1.0
```
Từng con số riêng lẻ **không có ý nghĩa** mà con người đọc được. Chỉ khi **so hai vector với nhau** thì mới có ý nghĩa (hình 1).

---

## 2. Tìm kiếm: câu hỏi → vector → xếp hạng

```
"ghế đọc sách nhỏ gọn" ──► "query: ghế đọc sách nhỏ gọn" ──► e5-small ──► vector q (384 số)
                                                                              │
 388 vector sản phẩm ─► ① LỌC CỨNG (danh mục, giá, rộng/sâu/cao tối đa) ──────┤
                        ② cosine(q, từng sản phẩm còn lại) = tích vô hướng ◄──┘
                        ③ sắp xếp giảm dần → top-k
```
- **Lọc cứng trước, xếp theo nghĩa sau.** Vector chỉ biết "giống nghĩa", **không biết** "dưới 20.000 yên" hay "rộng dưới 80 cm". Vì vậy giá và kích thước phải lọc chính xác bằng điều kiện, nếu không một sofa 80.000 yên vẫn có thể lọt vào kết quả "dưới 30.000 yên".
- Với 388 sản phẩm, phép tính cosine là một phép nhân ma trận 388 × 384, mất **dưới 1 ms**. Không cần database.

---

## 3. Công cụ sử dụng

| Việc | Công cụ | Vì sao |
|---|---|---|
| Chạy mô hình embedding | `sentence-transformers` 4.1 (trên PyTorch 2.9, CPU) | Thư viện chuẩn: tải mô hình, tokenize, pooling, chuẩn hoá chỉ trong 1 lệnh `encode()` |
| Mô hình | `intfloat/multilingual-e5-small` (~470 MB, tải từ Hugging Face lần đầu) | Đa ngôn ngữ, có tiếng Việt, nhỏ. Lúc chạy, máy chỉ còn khoảng 370 MB RAM trống, nên `BAAI/bge-m3` (~2,5 GB, tốt hơn) chưa chạy được. |
| Lưu vector và tìm kiếm | `numpy` (`embeddings.npy`, tích vô hướng) | 388 sản phẩm thì không cần vector database. Khi cần có thể nạp sang Postgres + pgvector mà không phải tính lại. |
| Vẽ hình | `matplotlib`; `scikit-learn` (t-SNE) | Ép 384 chiều xuống 2 chiều để nhìn được |

---

## 4. Output của bước này

| File | Nội dung | Kích thước |
|---|---|---|
| `data/index/embeddings.npy` | Ma trận **388 × 384** số float32, mỗi dòng là một sản phẩm, đã chuẩn hoá | 596 KB |
| `data/index/products.jsonl` | Dòng *i* là thông tin của vector *i*: `item_no`, tên, danh mục, giá, `w_cm/d_cm/h_cm`, đường dẫn `.glb` và ảnh, và đoạn văn `doc` đã dùng để embed | 491 KB |
| `data/index/meta.json` | Tên mô hình, số chiều, số sản phẩm, thời điểm build | – |
| `data/index/viz/*.png` | 4 hình ở §5 | – |
| `data/index/viz/stats.json` | Các con số dùng trong file này | – |

---

## 5. Nhìn thấy embedding

### Hình 1 – Một sản phẩm "trông" như thế nào sau khi embed
![vectors](data/index/viz/1_vectors.png)

- Mỗi dải là 384 con số của một sản phẩm: xanh là số âm, đỏ là số dương, đậm là trị tuyệt đối lớn.
- **Hai sofa HEMLINGBY** (hai màu vải khác nhau) có hoa văn gần như giống hệt, **cosine = 0,973**. **Đèn cây LERSTA** khác thấy rõ, **cosine với sofa = 0,854**.
- Lưu ý: cosine của mô hình e5 thường **dồn trong khoảng 0,8–1,0**. Vì vậy cái cần quan tâm là **thứ hạng** và **khoảng cách tương đối**, không phải con số tuyệt đối.

### Hình 2 – Bản đồ 388 sản phẩm
![map](data/index/viz/2_map.png)

- 384 chiều được thuật toán **t-SNE** ép xuống 2 chiều. t-SNE giữ các điểm vốn gần nhau thì vẫn gần nhau.
- Mỗi khung làm nổi bật một nhóm đồ (điểm xanh), các sản phẩm còn lại là điểm xám.
- Đọc hình:
  - **Các danh mục tự gom thành cụm riêng**: thảm nằm tách hẳn ở góc trên, đèn cây và đèn bàn nằm sát nhau, bàn ăn nằm cạnh ghế ăn.
  - Mô hình **không hề được cho biết danh mục**. Nó chỉ đọc đoạn văn, vậy mà xếp đúng những thứ cùng loại vào cạnh nhau.
- Kiểm tra bằng số: với **94,3%** sản phẩm, sản phẩm gần nó nhất (cosine cao nhất) **thuộc cùng danh mục**.

  | Mức | Danh mục |
  |---|---|
  | 100% | giường, ghế ăn, ghế văn phòng, đèn cây, đèn bàn, kệ TV, thảm |
  | Thấp nhất | sofa 79%, bàn ăn 83%, bàn làm việc 86% |

  Sofa hay "lẫn" với ghế bành, bàn ăn hay "lẫn" với bộ bàn ghế. Hai trường hợp này vốn cũng là đồ gần nhau trong đời thật.

### Hình 3 – Danh mục nào gần nghĩa nhau
![category similarity](data/index/viz/3_category_similarity.png)

- Mỗi ô là cosine giữa **tâm** (vector trung bình) của hai danh mục. Ô càng đậm thì hai danh mục càng gần nghĩa.
- Gần nhau nhất: **sofa – ghế bành (0,98)**. Xa nhau nhất: **tủ ngăn kéo – thảm (0,88)**.
- Các khối đậm khớp với cách con người nghĩ:
  - nhóm ghế (sofa, ghế bành, đôn, ghế ăn);
  - nhóm bàn (bàn trà, bàn ăn, bàn làm việc);
  - nhóm tủ;
  - đèn cây với đèn bàn.

### Hình 4 – Một truy vấn thật: vì sao cần lọc cứng
![query](data/index/viz/4_query.png)

- **Trái, chỉ tìm theo nghĩa:**
  - kết quả số 1 đúng là ghế bành TULLSTA;
  - nhưng **6/8 kết quả là kệ sách**, vì mô hình bắt theo chữ "**sách**" trong "ghế đọc sách".
- **Phải, lọc cứng trước** (danh mục = ghế bành, giá ≤ ¥20.000) rồi mới xếp hạng: cả 6 kết quả đều là ghế bành đúng ngân sách. Chỉ có 6 ghế bành thoả điều kiện, nên chỉ ra 6.
- Bài học cho các bước sau: ở bước 6, LLM sẽ đọc câu nói của người dùng, tách ra **danh mục** và **ngân sách** để lọc cứng, rồi mới để embedding xếp hạng theo "gu".

---

## 6. Đánh giá chất lượng

Lệnh: `python pipeline/search.py --eval`. Mỗi câu hỏi tiếng Việt lấy top-5. Một kết quả được tính "đúng" nếu nó thuộc danh mục hợp lý cho câu hỏi.

| Bộ test | Không có nhãn song ngữ | Có nhãn song ngữ |
|---|---|---|
| 15 câu chính ("kệ để tivi", "tủ đầu giường", "bàn ăn cho 4 người", …) | Top-1 47% · P@5 52% | **Top-1 93% · P@5 85%** |
| 5 câu **diễn đạt gián tiếp**, không chứa tên danh mục ("nơi cất quần áo cho phòng ngủ", "trải dưới bàn trà cho ấm chân", …) | (chưa đo) | Top-1 60% · P@5 56% |
| Sản phẩm gần nhất cùng danh mục (hình 2) | – | 94,3% |

- Nhãn song ngữ có một số từ trùng với 15 câu chính, nên **93% hơi lạc quan**. Bộ câu gián tiếp (**60%**) sát với thực tế hơn.
- Các lỗi còn lại đều do **bắt theo từ khoá thay vì ý định**:
  - "trải dưới bàn trà" → ra bàn trà, thay vì thảm;
  - "chỗ ngồi xem phim" → ra kệ TV;
  - "ghế đọc **sách**" → ra kệ sách.

---

## 7. Giới hạn và cách nâng cấp

| Giới hạn | Cách xử lý |
|---|---|
| Bắt theo từ khoá ("sách", "phim", "bàn trà") | **Bước 6**: LLM tách danh mục và ngân sách, rồi lọc cứng (hình 4 phải). Đây là cách xử lý chính. |
| Mô hình nhỏ (384 chiều) | `python pipeline/index.py --model BAAI/bge-m3` khi máy có ≥ 3 GB RAM trống, sau đó chạy lại `--eval` để so sánh |
| Đoạn văn chưa có phong cách, màu, phòng phù hợp | Khi bước 3 (enrichment) chạy, `index.py` **tự gộp** các trường này vào đoạn văn. Chỉ cần chạy lại `index.py`. |
| Chỉ embed chữ, chưa dùng ảnh | (Mở rộng) thêm embedding ảnh bằng CLIP/SigLIP để so "gu" giữa ảnh phòng và ảnh sản phẩm |

---

## 8. Lệnh chạy

```bash
python pipeline/index.py                                   # tạo vector (khoảng 45 giây)
python pipeline/search.py "ghế đọc sách nhỏ gọn" --category armchair --max-price 20000
python pipeline/search.py --eval                           # chạy bộ test 20 câu
python pipeline/visualize_index.py                         # vẽ lại 4 hình ở §5
```
