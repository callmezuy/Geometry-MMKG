# experiment_results/ (legacy, v1)

Mọi file trong thư mục này là kết quả **trước 12/11/2026**:

- chạy trên dữ liệu **v1** (valid của Merged-MMKG bị rò rỉ; nhãn VRD và split VRD/VG cũ);
- giao thức cũ: GeometryMMKG chỉ đánh giá chiều tail; NativE chạy cố định số epoch thay vì chọn checkpoint theo valid;
- không có file rank theo từng triplet.

**Chỉ dùng để tham khảo**, không đưa vào bảng so sánh. Kết quả mới phải chạy với giao thức trong
`RESEARCH_PLAN.md` mục 4 / `CONTRIBUTING.md` mục 4, và tính số bằng `scripts/compute_metrics.py`.
