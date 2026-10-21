# 04a_extended_cmf_static.ipynb — CMF Static Extended

Đây là bản mở rộng của `04a_cmf_static.ipynb` với mục tiêu tìm **điểm tối ưu thực sự** của pipeline CMF Static, không bị giới hạn bởi ceiling epoch thấp hoặc patience quá ngắn.

## Thay đổi so với 04a_raw (MAX=30, PATIENCE=3)

| Tham số | 04a_raw | 04a_extended | Lý do thay đổi |
|---|---|---|---|
| `MAX_EPOCHS_04A` | 30 | **50** | Ceiling cao hơn, đề phòng các method cần nhiều epoch để converge thực sự |
| `PATIENCE_04A` | 3 | **5** | Tránh dừng sớm khi `dist` đang oscillate ngắn (tăng 2–3 epoch rồi giảm lại) |
| `MIN_EPOCHS_04A` | 3 | **5** | Đảm bảo model đã bắt đầu học trước khi early stopping có thể kích hoạt |
| `SEED_BATCH` | 1 (batch1) | **None** (all seeds) | Chạy đủ 10 seeds trong 1 lần |
| Checkpoint prefix | `cmf_04a_` | **`cmf_04a_ext_`** | Không ghi đè checkpoint của 04a_raw |

## Tại sao cần bản này?

**Vấn đề với PATIENCE=3:**
Tiêu chí `dist = |NCC_forget − oracle_NCC| + (1 − out_retain) × 0.5` là hàm **không đơn điệu** theo epoch:
- `NCC_forget` oscillate khi gradient ascent (D_F) xung đột với gradient descent (D_R)
- Với PATIENCE=3: nếu `dist` tăng 3 epoch liên tiếp (giai đoạn oscillation) rồi sẽ giảm ở epoch thứ 4 → early stopping kích hoạt sớm, bỏ qua minimum sâu hơn

**PATIENCE=5** cho phép chịu đựng oscillation ngắn (2–4 epoch) trước khi quyết định dừng.

**Vấn đề với MAX=30:**
Nếu một số method/seed cần >25 epoch để đạt `dist` tối thiểu và early stopping chưa kích hoạt → bị cắt bởi ceiling. MAX=50 là safety buffer.

## Tiêu chí dừng sớm (giữ nguyên)

```
dist = |NCC_forget − oracle_NCC| + (1 − out_retain) × 0.5
```

Dừng khi `dist` không cải thiện sau **PATIENCE=5** epoch liên tiếp (tính từ epoch ≥ MIN_EPOCHS=5).

## Output

- Checkpoint: `artifacts/checkpoints/cmf_04a_ext_{method}_{model}_seed_{seed}.pt`
- Metrics: `artifacts/metrics/cmf_04a_ext_{method}_{model}_seed_{seed}.json`
- Summary: dùng cùng script tổng hợp, filter theo prefix `cmf_04a_ext_`

## So sánh dự kiến

Sau khi chạy xong, so sánh `04a_extended` với `04a_raw` và `04c` để xem:
1. PATIENCE=5 có cải thiện Forget gap và Retain không?
2. Method nào được hưởng lợi nhiều nhất từ patience dài hơn? (Dự đoán: scrub và neggrad_plus)
3. `actual_epochs` trung bình có tăng đáng kể không (nếu không tăng → PATIENCE=3 đã đủ)
