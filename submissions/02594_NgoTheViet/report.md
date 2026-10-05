# Báo cáo Lab Day 2 — Backbone, công thức huấn luyện và suy luận trên DeepWeeds
**Học viên:** Ngô Thế Việt — **MSSV:** 02594

---

## 1. Tóm tắt
- **Bài toán:** Phân loại 9 lớp cỏ dại trên tập dữ liệu DeepWeeds (gồm 8 loài cỏ dại nguy hại và 1 lớp Negative). Dữ liệu mất cân bằng nghiêm trọng với lớp Negatives chiếm khoảng 52%.
- **Chỉ số đánh giá chính:** Macro-F1 trên 9 lớp (trọng số đều giữa các lớp) để phản ánh trung thực chất lượng trên các loài hiếm, kết hợp Top-1 Accuracy, Balanced Accuracy và ECE.
- **Tiến độ hiện tại:** Đã hoàn thành toàn bộ Bước 0 (chuẩn bị dữ liệu, kiểm tra tính toàn vẹn S1–S6, xây dựng pipeline mô hình, loss, training loop và kiểm thử sanity checks).

---

## 2. Dữ liệu và Thiết lập thực nghiệm

### 2.1 Kiểm tra phân chia dữ liệu (Fold 0 — Quy tắc S1–S6)
Kết quả kiểm tra thực tế trên tập dữ liệu Fold 0:
- **Tập Train:** 10.501 ảnh (59.97%)
- **Tập Val:** 3.501 ảnh (20.00%)
- **Tập Test:** 3.507 ảnh (20.03%)
- **Tổng số ảnh:** 17.509 ảnh.
- **Giao giữa các tập:**
  - $\text{Train} \cap \text{Val} = 0$
  - $\text{Train} \cap \text{Test} = 0$
  - $\text{Val} \cap \text{Test} = 0$ (Hoàn toàn rỗng, không xảy ra rò rỉ dữ liệu).
- **Hợp 3 tập:** Đúng 17.509 ảnh, không thiếu bất kỳ file ảnh nào (missing = 0).

### 2.2 Phân bố lớp và nhận xét mất cân bằng (EDA)
- **Lớp lớn nhất:** Negatives chiếm 5.464 ảnh trong tập train (~52.0%), áp đảo hoàn toàn các lớp còn lại.
- **8 loài cỏ dại:** Số lượng dao động khoảng 600–675 ảnh/loài trong tập train (Chinee Apple: 675, Snake Weed: 610, Lantana: 638, v.v.).
- **Tỉ lệ mất cân bằng:** Lớp Negatives lớn gấp ~8.5 lần so với các lớp hiếm. Do đó, nếu mô hình dự đoán thiên lệch về Negatives, Top-1 Accuracy vẫn có thể cao nhưng Macro-F1 sẽ rất thấp.

### 2.3 Kiểm tra tính đúng đắn của Pipeline (Sanity Checks — Slide tr. 59)
1. **Cố định seed:** Đã cố định seed 42 trên toàn bộ `random`, `numpy`, `torch` và worker của DataLoader.
2. **Mất mát ban đầu (Initial Loss):**
   - Với 9 lớp, loss Cross-Entropy lý thuyết lúc khởi tạo ngẫu nhiên head là $-\ln(1/9) = \ln(9) \approx 2.1972$.
   - Thực nghiệm trên `resnet50`: Loss ban đầu đo được là **2.2462** (sai số tuyệt đối chỉ 0.049, khớp kỳ vọng lý thuyết).
3. **Overfit một batch nhỏ:**
   - Huấn luyện một mini-batch gồm 4 mẫu qua 60 bước lặp bằng optimizer AdamW (lr=1e-3).
   - Loss giảm từ 2.24 xuống **0.000182** ($< 0.05$), chứng minh pipeline forward/backward/optimizer hoạt động chính xác 100%.
4. **Kiểm tra tiền xử lý và nhãn:**
   - Ảnh sau augmentation cơ bản (RandomResizedCrop + HorizontalFlip) được giải chuẩn hóa ImageNet và vẽ trực quan cùng nhãn, đảm bảo đúng loài và không bị sai lệch kênh màu.

---

## 3. Kết quả và Phân tích
*(Đang tiến hành huấn luyện Bước 1: So sánh các Backbone)*
