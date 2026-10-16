# Mô hình 3D của IKEA được tạo ra như thế nào

> File này trả lời 3 câu hỏi:
> 1. IKEA tạo ra các file `.glb` mà ta crawl về bằng quy trình gì, dùng công cụ gì? (§1–§3)
> 2. Dự án này lấy và xử lý các file đó ra sao? (§4)
> 3. Muốn **tự tạo** một mô hình tương đương (cho sản phẩm không có 3D, hoặc đồ của riêng mình) thì làm thế nào, bấm nút nào? (§5–§7)
>
> Cập nhật: 2026-10-10

**Độ tin cậy của thông tin.** IKEA **không công bố** quy trình dựng 3D nội bộ: ai dựng, dùng phần mềm CAD nào, bấm nút gì. Vì vậy nội dung dưới đây chia làm 3 loại, có ghi nhãn rõ:

| Nhãn | Ý nghĩa |
|---|---|
| **[Đo được]** | Đọc trực tiếp từ metadata bên trong 388 file `.glb` trong `data/models` (trường `asset.generator`, `asset.extras`, extensions, materials, URL tải). Đây là bằng chứng chắc chắn. |
| **[Công khai]** | Thông tin đã được báo chí hoặc IKEA công bố (có nguồn ở §8). |
| **[Suy luận]** | Kết luận rút ra từ hai loại trên. Hợp lý nhưng chưa được IKEA xác nhận. |

---

## 1. Bức tranh tổng thể

- **[Công khai]** IKEA dùng ảnh dựng 3D (CGI) thay cho ảnh chụp từ giữa những năm 2000. Ảnh catalogue được dựng bằng **Autodesk 3ds Max** và render bằng **V-Ray**. Khoảng năm 2014, tới ~75% ảnh *sản phẩm đơn lẻ* là ảnh dựng 3D.
- **[Công khai]** IKEA giữ một **kho mô hình trung tâm** gồm hàng chục nghìn mô hình sản phẩm và texture. Kho này phục vụ nhiều đầu ra: ảnh render, cảnh phòng, xem 3D tương tác và AR. IKEA cũng tự vận hành hạ tầng render và phân phối 3D.
- **[Công khai]** Năm 2020, Ingka Group (tập đoàn sở hữu phần lớn cửa hàng IKEA) mua **Geomagical Labs**, công ty làm công nghệ quét phòng bằng điện thoại, xoá đồ cũ và đặt đồ mới đúng tỉ lệ. Về sau công nghệ này trở thành IKEA Kreativ. IKEA cũng đưa hơn 10.000 sản phẩm lên Google Search để xem 3D/AR đúng kích thước.
- **[Đo được]** Mọi file ta tải về đều nằm dưới đường dẫn `web-api.ikea.com/dimma/assets/geomagical/...`. Vậy là các file AR/web hiện nay được phân phối qua hạ tầng gắn với Geomagical.

**[Suy luận]** Từ đó suy ra quy trình gồm 2 tầng:
1. **Mô hình gốc (master):** chi tiết cao, dùng để render ảnh catalogue. Rất nặng, thường hàng triệu tam giác, không chạy được trên điện thoại.
2. **Mô hình real-time:** được **tự động sinh ra từ master** qua một pipeline tối ưu (§2). Đây chính là các file `.glb` khoảng 0,1–5 MB mà ta crawl về.

```
 ┌───────────────────────┐   ┌────────────────────────────────────┐   ┌─────────────────────┐   ┌──────────────────┐
 │ A. Mô hình gốc         │   │ B. Tối ưu tự động (Simplygon)       │   │ C. Đóng gói          │   │ D. Phát hành      │
 │ 3ds Max (+V-Ray)       │──►│ remeshing HOẶC reduction            │──►│ glTF-Transform       │──►│ CDN web-api.ikea  │
 │ chi tiết cao, render   │   │ nướng texture 1024–2048, nướng AO   │   │ nén Draco, WebP      │   │ /dimma/assets/... │
 │ catalogue              │   │ (wrapper nội bộ "optimiser")        │   │ 1 node, 1 scene      │   │ + JSON-LD trên    │
 │ [Công khai + Suy luận] │   │ [Đo được]                           │   │ [Đo được]            │   │ trang sản phẩm    │
 └───────────────────────┘   └────────────────────────────────────┘   └─────────────────────┘   └──────────────────┘
```

---

## 2. Pipeline suy ra từ metadata của file [Đo được]

Mỗi file glTF có trường `asset.generator` (công cụ ghi ra file) và `asset.extras` (tuỳ ý, IKEA ghi thông số tối ưu vào đây). Thống kê trên 388 file trong catalog:

### 2.1 Công cụ ghi ra file (`asset.generator`)

| Công cụ | Số file | Là gì |
|---|---|---|
| **glTF-Transform** (v0.8 → v4.4) | ~300 | Thư viện/CLI mã nguồn mở để chỉnh sửa và tối ưu glTF (nén Draco, đổi texture sang WebP, dọn dữ liệu thừa). Đây là **bước đóng gói cuối cùng**. |
| **Khronos glTF Blender I/O** (v1.6, v3.3) | 54 | Bộ xuất glTF mặc định của **Blender**. Một phần model được dựng hoặc chỉnh trong Blender rồi xuất thẳng. |
| **pygltflib** | 16 | Thư viện Python. Model được ghi bằng script tự động. |
| **Microsoft Simplygon glTF Exporter** | 4 | Xuất thẳng từ Simplygon. |
| **"ICOM real-time pipeline"** | 1 | Tên một pipeline nội bộ (không có thông tin công khai). |

Có nhiều phiên bản glTF-Transform khác nhau (từ 0.8 năm 2020 đến 4.4 năm 2025), cho thấy model được sinh ra **rải rác qua nhiều năm** bằng một pipeline được nâng cấp dần.

### 2.2 Tối ưu bằng Simplygon (228/388 file có ghi lại thông số)

**Simplygon** (của Microsoft) là phần mềm chuyên **tự động giảm độ phức tạp** của mô hình 3D, thường dùng trong ngành game. Các trường IKEA ghi lại:

| Trường trong `asset.extras` | Giá trị hay gặp | Ý nghĩa |
|---|---|---|
| `simplygon-version` | `9.1.42900` (178), `9.1.6200` (32), `9.1.159` (18) | Phiên bản Simplygon SDK |
| `optimiser-version` | `1.23.0.494`, `1.16…`, `1.11…`, `1.10…`, `1.7…` | **[Suy luận]** phiên bản công cụ nội bộ của IKEA bọc quanh Simplygon. Simplygon không có số phiên bản dạng này. |
| `input` | `in.glb` | Đầu vào của bước tối ưu là **một file glb**. Nghĩa là mô hình gốc được xuất ra glb trước, rồi mới tối ưu. |
| `remeshing` | `{screen-size: 1250, hard-edge-angle: 25, hole-filling: "low"}` (~95 file) | **Chế độ 1 – Remeshing:** Simplygon *dựng lại một lớp vỏ lưới mới* bao ngoài vật thể, rồi nướng toàn bộ vật liệu gốc lên texture mới. `screen-size 1250` = tối ưu để vật trông đúng khi chiếm khoảng 1250 px trên màn hình. `hard-edge-angle 25°` = cạnh gấp hơn 25° thì giữ sắc. `hole-filling low` = lấp nhẹ các lỗ hoặc hốc nhỏ. |
| `reduction` | `{screen-size: 1250, welding: 0}` (131 file) | **Chế độ 2 – Reduction:** *giữ lưới gốc nhưng xoá bớt tam giác* cho tới khi đủ nhẹ ở kích thước hiển thị 1250 px. |
| `texture-size` | `2048` (170), `1024` (56) | Kích thước texture được nướng ra (2048×2048 hoặc 1024×1024) |
| `multisampling` | `8` | Lấy mẫu 8 lần mỗi pixel khi nướng texture, giúp texture mịn và không răng cưa |
| `occlusion-opaque` / `occlusion-blend` | `{rays: 50, strength: 0.4}` (221 file) | **Nướng ambient occlusion (AO):** bắn 50 tia cho mỗi điểm để tính độ khuất sáng, tạo bóng tối ở khe và góc. Kết quả lưu thành `occlusionTexture` (377 material có). |
| `texture-area` | `texel-density`, `uv-area`, `world-area` | Thống kê độ phân giải texture trên mỗi m² bề mặt. Dùng để kiểm soát chất lượng. |
| `normal-repair` | `enable: false` (224) | Chức năng sửa vector pháp tuyến, phần lớn tắt |
| `visibility` | `{coverage, fidelity, pitch, yaw}` (5 file) | Bỏ các tam giác không bao giờ nhìn thấy được từ những góc nhìn đã chọn |

Ví dụ thật, đọc từ file ghế YNGVAR (`60417637.glb`):
```json
"generator": "glTF-Transform v0.8.4",
"extras": { "input": "in.glb", "simplygon-version": "9.1.159", "optimiser-version": "1.10.0.404",
            "remeshing": {"screen-size": 1250, "hard-edge-angle": 25, "hole-filling": "low"},
            "texture-size": 2048, "multisampling": 8,
            "occlusion-opaque": {"rays": 50, "strength": 0.4, "simple-mode": true, "radius": -1} }
```

### 2.3 Kết quả đầu ra (cấu trúc file)

| Đặc điểm | Thống kê | Ý nghĩa |
|---|---|---|
| Nén lưới **Draco** (`KHR_draco_mesh_compression`) | 378/388 | Lưới được nén mạnh để tải nhanh. App phải có bộ giải nén Draco. |
| Texture **WebP** (`EXT_texture_webp`) | 84 file | Texture nhẹ hơn JPEG/PNG. Tổng có 590 ảnh JPEG, 620 PNG và 326 WebP. |
| Vật liệu PBR đủ bộ (baseColor + metallicRoughness + normal + occlusion) | 326 material | Chuẩn vật liệu vật lý (PBR), hiển thị thật dưới ánh sáng AR |
| Trong suốt (`alphaMode: BLEND`) | 69 material | Kính, chụp đèn, vải mỏng |
| Cấu trúc | Thường **1 scene** (`cascade-scene-0`) với **1 node** (`ProcessedMeshNode`) | Cả vật thể được gộp thành một khối, tối ưu cho việc hiển thị |
| Đơn vị / gốc toạ độ | Mét. Gốc ở **đáy-giữa**. Mặt trước hướng **+Z**. | Đặt thẳng lên sàn trong AR không cần chỉnh |
| Độ nặng | Trung vị khoảng 7,6k tam giác (sau khi lọc ở bước 2) | Rất nhẹ so với master |

### 2.4 Phát hành [Đo được]

URL mẫu:
```
https://web-api.ikea.com/dimma/assets/geomagical/80501951/PS01_S01_NV_01/simple/glb_draco/
        a01c5f0e…-G-80501951-382eed02…-simple+draco.glb
```
- `geomagical/80501951`: mã của **mô hình**. Mã này **không nhất thiết trùng mã sản phẩm**: chỉ 67/388 URL chứa đúng `item_no`. Ví dụ STARKVIND `70501956` dùng model `80501951`. **[Suy luận]** IKEA quản lý mô hình theo một mã hình học riêng, và nhiều biến thể có thể dùng chung một mô hình. Điều này giải thích các ca "model là biến thể khác cỡ" ở bước 2, như giường TARVA.
- `simple/glb_draco` hoặc `simple/glb`: hồ sơ chất lượng (`simple`) và định dạng. Một số file mới hơn nằm dưới `dimma/assets/1.2/<id>/.../iqp3/` hoặc `rqp3/`. **[Suy luận]** đây là các hồ sơ chất lượng đời sau.
- `PS01_S01_NV_01`: ý nghĩa **không rõ** (không có thông tin công khai).
- Trang sản phẩm nhúng link này trong một khối JSON-LD `{"@type": "3DModel"}`. Crawler của ta đọc link từ đó.

### 2.5 Những gì **không** biết được

- Ai dựng mô hình gốc (nhân viên IKEA hay agency), dựng từ bản vẽ kỹ thuật hay từ ảnh, dùng phần mềm CAD nào ngoài 3ds Max.
- Pipeline được chạy như thế nào: tự động hoàn toàn hay có người duyệt, chạy trên server nào.
- "Bấm nút nào": pipeline tối ưu gần như chắc chắn **chạy bằng script, không có thao tác tay**. Bằng chứng là các thông số giống hệt nhau trên hàng trăm file (`screen-size 1250`, `rays 50`, `strength 0.4`) và đầu vào luôn tên `in.glb`.

---

## 3. Tóm tắt quy trình của IKEA (theo thứ tự)

| # | Bước | Công cụ | Đầu ra | Nhãn |
|---|---|---|---|---|
| A | Dựng mô hình gốc chi tiết cao, gán vật liệu thật | 3ds Max (+ V-Ray để render ảnh); một phần dùng Blender | Master (dùng cho ảnh catalogue) | Công khai / Đo được (Blender) |
| B1 | Xuất master ra `in.glb` | Exporter nội bộ | `in.glb` | Đo được (`input: in.glb`) |
| B2 | Tối ưu tự động: **remeshing** (dựng lưới mới) hoặc **reduction** (giảm tam giác), mục tiêu hiển thị 1250 px | Simplygon 9.1 qua công cụ nội bộ "optimiser" | Lưới nhẹ | Đo được |
| B3 | Nướng vật liệu vào texture 1024/2048 px (8× multisampling) và nướng AO (50 tia, độ đậm 0.4) | Simplygon | baseColor, normal, metallicRoughness, occlusion | Đo được |
| C | Gộp thành 1 node, nén lưới Draco, đổi texture sang WebP (một số file) | glTF-Transform | `*-simple+draco.glb` | Đo được |
| D | Đẩy lên CDN, nhúng link vào JSON-LD của trang sản phẩm | Hạ tầng `web-api.ikea.com/dimma` (Geomagical) | URL công khai | Đo được |

---

## 4. Quy trình của dự án này (lấy và kiểm tra model)

Dự án **không tạo** mô hình 3D, chỉ **tải và kiểm tra** model của IKEA. Chi tiết xem `pipeline.md` (bước 1–2).

| Bước | Lệnh | Kết quả |
|---|---|---|
| 1. Crawl | `python crawler/ikea_crawler.py --per-category 25` | `.glb` + metadata của 426 sản phẩm, 18 danh mục |
| 2. Kiểm tra & lọc | `python pipeline/process_models.py --previews --exclude` | Giữ 388 model: lệch kích thước ≤ 20% so với số IKEA ghi, ≤ 100k tam giác, ≤ 5 MB. 38 model bị loại được chuyển sang `data/excluded/`. |
| Xem nhanh 1 file | Kéo file `.glb` vào https://gltf-viewer.donmccurdy.com hoặc https://modelviewer.dev/editor/ | Xem 3D trong trình duyệt |

---

## 5. Tự tạo mô hình 3D tương đương – chọn cách nào?

Khi nào cần: khoảng 32% sản phẩm IKEA **không có** file 3D (bước 1 thấy 201/631 trang không có), cộng 38 model bị loại ở bước 2. Muốn đưa các sản phẩm đó vào app thì phải tự tạo.

| Cách | Công cụ | Độ chính xác kích thước | Thời gian / model | Phù hợp khi |
|---|---|---|---|---|
| **A. Dựng tay** | Blender (miễn phí) | ⭐⭐⭐ đúng tuyệt đối (nhập số đo) | 1–4 giờ | Đồ hình khối đơn giản: kệ, bàn, tủ, giường |
| **B. Quét bằng ảnh** (photogrammetry) | Điện thoại: Polycam, RealityScan, Reality Composer (Object Capture trên iPhone) | ⭐⭐ cần chỉnh lại tỉ lệ | 15–30 phút | Có món đồ thật trong tay |
| **C. AI tạo 3D từ ảnh** | TRELLIS, Hunyuan3D-2 (mã nguồn mở), Meshy (dịch vụ) | ⭐ phải scale lại, mặt sau thường bị "bịa" | Vài phút | Cần nhanh, chấp nhận chất lượng trung bình |

Cách nào cũng phải làm thêm **§6.6 (tối ưu)** và **§7 (kiểm tra)** để ra file giống chuẩn IKEA.

---

## 6. Cách A – Dựng tay trong Blender, từng bước bấm nút

> Hướng dẫn cho Blender 4.x (tải ở https://www.blender.org). Ví dụ: dựng một kệ 80 × 30 × 120 cm. Phím tắt ghi trong `[ ]`.

### 6.1 Cài đơn vị đo (làm 1 lần)
1. Bên phải màn hình, ở cột icon dọc của **Properties**, bấm icon **Scene** (hình nón và quả cầu).
2. Mở mục **Units**: chọn *Unit System* = **Metric**, *Unit Scale* = **1.000**, *Length* = **Centimeters**.
   → 1 đơn vị Blender = 1 mét, khớp với chuẩn glTF.

### 6.2 Dựng khối theo đúng kích thước
1. Xoá khối mặc định: chọn khối, bấm `[X]` → **Delete**.
2. Thêm khối mới: `[Shift+A]` → **Mesh** → **Cube**.
3. Mở thanh bên: `[N]` → tab **Item** → mục **Dimensions**: nhập **X = 80 cm** (rộng), **Y = 30 cm** (sâu), **Z = 120 cm** (cao).
4. Chốt tỉ lệ: `[Ctrl+A]` → **Scale**. Bắt buộc làm, nếu không file xuất ra sẽ mang hệ số scale lạ.
5. Tạo chi tiết: `[Tab]` vào **Edit Mode**, rồi dùng:
   - `[Ctrl+R]` loop cut để chia cạnh;
   - `[I]` inset mặt;
   - `[E]` extrude đẩy mặt ra hoặc vào (tạo ngăn kệ);
   - `[Ctrl+B]` bevel bo cạnh nhẹ (2–3 mm cho đồ gỗ).

   Xong thì `[Tab]` để quay về **Object Mode**.

### 6.3 Đặt gốc toạ độ và hướng (giống model IKEA)
Model IKEA có **gốc ở đáy-giữa** và **mặt trước hướng +Z (glTF)**. Trong Blender (trục Z hướng lên), mặt trước tương ứng là hướng **−Y**.
1. Bấm `[Numpad 1]` (Front view, nhìn từ phía −Y): **mặt trước của kệ phải quay về phía bạn**. Nếu chưa đúng thì `[R]` `[Z]` `[180]` `[Enter]`.
2. Đưa con trỏ 3D về gốc: `[Shift+S]` → **Cursor to World Origin**.
3. Di chuyển vật sao cho **đáy chạm Z = 0** và tâm nằm trên trục: trong tab **Item**, đặt *Location X = 0, Y = 0, Z = 60 cm* (bằng nửa chiều cao).
4. Đặt gốc của vật về con trỏ: menu **Object** → **Set Origin** → **Origin to 3D Cursor**. Gốc giờ nằm ở đáy-giữa.
5. Đưa vật về toạ độ 0: `[Alt+G]`.

### 6.4 UV và vật liệu
1. `[Tab]` → `[A]` chọn hết → `[U]` → **Smart UV Project** → **OK** → `[Tab]`.
2. Properties → icon **Material** (quả cầu đỏ) → **New**. Blender tạo sẵn *Principled BSDF* (chuẩn PBR, xuất sang glTF được).
3. Ở ô *Base Color*, bấm vào chấm tròn bên cạnh → **Image Texture** → **Open**, rồi chọn ảnh vân gỗ hoặc vải. Có thể thêm *Roughness* và *Normal Map* theo cách tương tự.

### 6.5 (Tuỳ chọn) Nướng ambient occlusion giống IKEA
1. Properties → icon **Render** → *Render Engine* = **Cycles**.
2. Mở **Shading** workspace (tab trên cùng). Trong Shader Editor, thêm node: `[Shift+A]` → **Texture** → **Image Texture** → **New**, đặt tên `AO`, kích thước **2048 × 2048**. Để node này được chọn, không nối dây vào đâu.
3. Properties → **Render** → mục **Bake** → *Bake Type* = **Ambient Occlusion** → bấm **Bake**.
4. Ở Image Editor: **Image** → **Save As** → `ao.png`. Nối ảnh này vào vật liệu (bộ xuất glTF hiểu AO khi nó được nối qua node *glTF Material Output*).

### 6.6 Giảm số tam giác (tương đương bước "reduction" của Simplygon)
1. Properties → icon **Modifier** (cờ lê) → **Add Modifier** → **Generate** → **Decimate**.
2. Chỉnh *Ratio* (ví dụ 0.3 = giữ 30% tam giác). Góc dưới màn hình hiện số *Faces* để theo dõi. Mục tiêu dưới 100.000 tam giác, tốt nhất 5.000–30.000.

### 6.7 Xuất file `.glb`
1. Chọn vật cần xuất. Menu **File** → **Export** → **glTF 2.0 (.glb/.gltf)**.
2. Trong bảng bên phải:
   - *Format* = **glTF Binary (.glb)**
   - **Include** → *Limit to*: ☑ **Selected Objects**
   - **Transform**: ☑ **+Y Up** (mặc định; chuyển trục Z-lên của Blender sang Y-lên của glTF, nhờ vậy mặt −Y trong Blender thành +Z trong glTF)
   - **Data → Mesh**: ☑ **Apply Modifiers** (để Decimate có hiệu lực)
   - **Data → Compression**: ☑ bật nén **Draco**, *Compression Level* = 6 (Blender 3.x để mục *Compression* thành panel riêng)
3. Đặt tên file (ví dụ `my_shelf.glb`) → bấm **Export glTF 2.0**.

---

## 6B. Cách B – Quét bằng điện thoại (photogrammetry)

> Giao diện các app thay đổi theo phiên bản, nên dưới đây là các bước chung.

1. Đặt món đồ ở nơi **đủ sáng, ánh sáng đều, không nắng gắt**, xung quanh có khoảng trống để đi vòng.
2. Mở app (Polycam hoặc RealityScan) → nút **tạo bản quét mới** → chọn chế độ **Object / Photo**.
3. Đi vòng quanh vật **2–3 vòng ở các độ cao khác nhau** (ngang tầm, chếch xuống 45°, từ trên xuống). Tổng 60–150 ảnh, ảnh sau chồng lên ảnh trước khoảng 70%.
4. Bấm **Process / Upload**, chờ app dựng mô hình.
5. Trong app: **Crop** để cắt bỏ sàn và vật thừa → **Export** → **GLB**.
6. Mở file trong Blender (**File → Import → glTF 2.0**):
   - Đo lại bằng `[N]` → *Dimensions* và **scale cho đúng số đo thật** (đo bằng thước).
   - Làm lại §6.3 (gốc, hướng), §6.6 (giảm tam giác) và §6.7 (xuất).

## 6C. Cách C – AI tạo 3D từ ảnh sản phẩm

1. Lấy ảnh sản phẩm nền trắng (`data/images/<item_no>.jpg`).
2. Đưa vào mô hình image-to-3D: **TRELLIS** hoặc **Hunyuan3D-2** (chạy local, cần GPU), hoặc dịch vụ web như **Meshy** → tải về `.glb`.
3. Mô hình AI **không có kích thước thật**, nên trong Blender phải scale theo `dimensions_cm` của IKEA (nhập vào *Dimensions* như §6.2, rồi `[Ctrl+A]` → *Scale*), sau đó làm §6.3, §6.6 và §6.7.
4. Lưu ý: mặt sau và mặt dưới thường bị AI "bịa". Chỉ nên dùng cho demo, và **phải ghi rõ** đây là mô hình minh hoạ, không phải mô hình chính thức của IKEA.

## 6D. Tối ưu bằng dòng lệnh (giống bước C của IKEA)

Dùng **glTF-Transform**, chính là công cụ IKEA dùng ở bước đóng gói (cần Node.js; máy đã có Node v24):
```bash
npm install --global @gltf-transform/cli

gltf-transform inspect my_shelf.glb                      # xem số tam giác, texture, dung lượng
gltf-transform resize my_shelf.glb tmp.glb --width 2048 --height 2048   # giới hạn texture 2048
gltf-transform optimize tmp.glb my_shelf_final.glb --compress draco --texture-compress webp
```
`optimize` gộp nhiều bước: dọn dữ liệu thừa, gộp lưới, giản lược lưới, nén Draco và đổi texture sang WebP. Kết quả là một file có cấu trúc tương tự file IKEA.

---

## 7. Kiểm tra trước khi đưa vào app

Mô hình tự tạo phải đạt **cùng tiêu chí** như model IKEA ở bước 2:

| Tiêu chí | Yêu cầu | Cách kiểm tra |
|---|---|---|
| Đơn vị | mét | `gltf-transform inspect` → xem bounding box |
| Kích thước | lệch ≤ 20% so với số đo thật | So bounding box với số đo |
| Gốc toạ độ | đáy ở y = 0, tâm ở x = z = 0 | Bounding box: `min.y ≈ 0`, `(min.x + max.x)/2 ≈ 0`, `(min.z + max.z)/2 ≈ 0` |
| Hướng | mặt trước +Z | Xem preview "FRONT (camera at +Z)" do `process_models.py` vẽ |
| Độ nặng | ≤ 100.000 tam giác, ≤ 5 MB | `gltf-transform inspect` |
| Định dạng | `.glb`, nén Draco (khuyến nghị) | — |

Để chạy kiểm tra tự động bằng script của dự án, đặt file vào `data/models/<mã>.glb` và tạo `data/products/<mã>.json` (có `item_no`, `model_file`, `dimensions_cm`, `crawl_category`). Sau đó chạy:
```bash
python crawler/ikea_crawler.py --catalog-only
python pipeline/process_models.py --previews --exclude
```

---

## 8. Nguồn

- TechCrunch (2020) – Ikea acquires AI imaging startup Geomagical Labs: https://techcrunch.com/2020/04/02/ikea-acquires-ai-imaging-startup-geomagical-labs-to-supercharge-room-visualisations
- Retail Dive – Ingka Group acquires 3D visual AI tech firm: https://www.retaildive.com/news/ingka-group-owner-of-ikea-acquires-3d-visual-ai-tech-firm/575714/
- AWE (2024) – Mapping IKEA's 3D journey (kho mô hình trung tâm, hạ tầng render): https://www.awexr.com/blog/1015-awe-talks-mapping-ikea-s-3d-journey
- Mixed News – IKEA đưa sản phẩm lên Google Search dạng AR: https://mixed-news.com/en/ikea-products-ar-models-google-search/
- Slate (2014) – 75% ảnh sản phẩm trong catalogue là CGI: https://www.slate.com/blogs/the_eye/2014/09/05/ikea_catalog_images_75_percent_are_computer_generated.html
- JTB World (2008) – IKEA dùng 3ds Max + V-Ray: https://blog.jtbworld.com/2008/11/ikea-using-computer-generated-and.html
- Khronos – Simplygon hỗ trợ glTF: https://www.khronos.org/news/permalink/microsoft-announces-simplygon-cloud-with-gltf-support
- Simplygon SDK – RemeshingSettings (hard edge angle, hole filling, screen size): https://documentation.simplygon.com/SimplygonSDK_9.0.666.0/api/reference/processingandmaterialcasting/settingsobject/remeshingsettings.html
- Số liệu [Đo được]: tự trích từ 388 file trong `data/models` (`asset.generator`, `asset.extras`, `extensionsUsed`, materials, URL).
