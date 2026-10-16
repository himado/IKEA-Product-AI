# Đánh giá DoHa (bước 16)

Bản quét dùng để đo: `20261016_174640_f741c4` (phòng ảo, ARCore phát lại). Đáp án phòng ảo chỉ dùng ở file này.

| Giai đoạn | Chỉ số | Mục tiêu | Kết quả | Ghi chú |
|---|---|---|---|---|
| P1 | Sai số kích thước phòng | < 10% | 0.0% và 15.0% (4.0 × 4.25 m so với 4.0 × 5.0 m) | phòng ảo, 1 bản quét |
| P1 | Recall đồ vật lớn | ≥ 80% | 2/3 đúng loại, 3/3 đúng chỗ (mọi đồ: 3/5) | đèn cây không nhận ra |
| P2 | Trích xuất slot (danh mục, ngân sách…) | ≥ 90% | 30/30 câu (Gemini); 23/30 khi chỉ dùng luật dự phòng | 30 câu thay vì 50 |
| P4 | Vi phạm ngân sách / tỷ lệ giá | 0% | 0/9 bộ gợi ý | 7 kịch bản + phòng quét |
| P4 | Gợi ý hợp lý (người chấm) | ≥ 70% | lần chấm v1: 4/7 Tốt, 3/7 Tạm; đã sửa phân bổ ngân sách, chưa chấm lại | cần bạn chấm lại |
| P5 | Độ trôi vị trí | < 5 cm | phòng ảo phát lại: 23 cm | ảnh dựng khác ảnh camera thật; chưa đo trên phòng thật |
| P5 | FPS / thời gian load model | ≥ 30 fps, < 3 s | 30 FPS (5 model); load 171–198 ms | Galaxy A55, bước 5 |
| – | Thời gian một lượt gợi ý (API) | – | ≈ 9 s (trước là 90–113 s) | server nạp sẵn mô hình khi khởi động (12–75 s); bỏ qua model Gemini đang hết hạn mức |

## Đồ vật: đáp án ↔ máy tìm được

| Đồ thật | Máy nhận là | Lệch tâm |
|---|---|---|
| FRYKSÅS TV bench | tv bench | 0.66 m |
| ROCKSJÖN 2-seat sofa | sofa | 0.16 m |
| BYAKORRE shelving | bench (đúng chỗ, sai loại) | 0.76 m |
| STOENSE rug | rug | 0.62 m |
| LAUTERS floor lamp | — (không thấy) | — m |
