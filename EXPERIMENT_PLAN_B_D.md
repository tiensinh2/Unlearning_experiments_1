# Kế hoạch chạy thí nghiệm — Config B và D

> Mục tiêu: so sánh **04a_extended (B)** vs **pipeline 3-stage extended (D)** để tìm config tối ưu.
> Chạy **1 seed trước** (seed=0) để validate, sau đó mới chạy đủ 10 seeds.

---

## Tổng quan 4 config

| Config | Notebook(s) | MAX epochs | PATIENCE | Checkpoint prefix |
|---|---|---|---|---|
| **A** (baseline, đã có) | `04a_cmf_static.ipynb` | 30 | 3/3/3 | `cmf_04a_` |
| **B** (mới) | `04a_extended_cmf_static.ipynb` | 50 | 5/5/5 | `cmf_04a_ext_` |
| **C** (baseline, đã có) | sub_04a (10/18) → 04b → 04c (10) | 10/18/20/10 | 3/2/3 | `cmf_04c_` |
| **D** (mới) | sub_ext (15/25) → 04b_ext → 04c_ext (15) | 15/25/30/15 | 5/5/3/5 | `cmf_04c_ext_` |

---

## Notebooks mới tạo

### Config B — CMF Static Extended

| Notebook | Thay đổi |
|---|---|
| `04a_extended_cmf_static.ipynb` | MAX=50, PATIENCE=5, MIN=5 · SEED_BATCH=None · checkpoint: `cmf_04a_ext_` |

### Config D — CMF 3-Stage Extended

| Notebook | Thứ tự | Thay đổi | Checkpoint |
|---|---|---|---|
| `04a_sub_extended_cmf_static.ipynb` | 1 | MAX=15 (rl/ng/salun), scrub=25, PATIENCE=5, MIN=5 | ghi: `cmf_04a_ext_` |
| `04b_extended_cmf_posthoc.ipynb` | 2 | max_epochs=30, patience=3 | đọc: `cmf_04a_ext_` · ghi: `cmf_04b_ext_` |
| `04c_extended_cmf_adaptive.ipynb` | 3 | MAX=15, PATIENCE=5 | đọc: `cmf_04b_ext_` · ghi: `cmf_04c_ext_` |

---

## Bước 1 — Chạy thử seed=0

### Config B (seed=0)
```
Notebook: 04a_extended_cmf_static.ipynb
SEED_BATCH = None  (đã set, chạy tất cả seeds — nhưng bạn có thể set = 1 để chỉ seed 0-4)
Hoặc set SEED_BATCH = 1 rồi interrupt sau seed=0
```

### Config D (seed=0) — chạy theo thứ tự
```
1. 04a_sub_extended_cmf_static.ipynb   (tạo cmf_04a_ext_*_seed_0.pt)
2. 04b_extended_cmf_posthoc.ipynb      (đọc cmf_04a_ext_*, tạo cmf_04b_ext_*_seed_0.pt)
3. 04c_extended_cmf_adaptive.ipynb     (đọc cmf_04b_ext_*, tạo cmf_04c_ext_*_seed_0.pt)
```

---

## Bước 2 — Đánh giá seed=0, quyết định tiếp

So sánh B0 vs A0 và D0 vs C0 trên 3 chỉ số quyết định:

| Chỉ số | Tốt khi | Nguồn |
|---|---|---|
| `Out_retain` | ↑ gần Oracle (99.9%) | metrics JSON |
| `\|Out_forget − oracle_forget\|` | ↓ gần 0 | metrics JSON |
| `LP_illusion_gap` | ↓ | metrics JSON |

**Quy tắc quyết định:**
- Nếu B0 > A0 trên ≥ 2/3 chỉ số → chạy B đủ 10 seeds
- Nếu D0 > C0 trên ≥ 2/3 chỉ số → chạy D đủ 10 seeds
- Nếu cả hai không cải thiện → dừng, config A và C đã tốt nhất

---

## Bước 3 — Chạy đủ 10 seeds (nếu cần)

### Config B (10 seeds)
```
04a_extended_cmf_static.ipynb  (SEED_BATCH=None, chạy seeds 0-9)
```

### Config D (10 seeds)
```
1. 04a_sub_extended_cmf_static.ipynb  (SEED_BATCH=None)
2. 04b_extended_cmf_posthoc.ipynb
3. 04c_extended_cmf_adaptive.ipynb
```

---

## So sánh cuối cùng

Sau khi có đủ data, tổng hợp vào bảng so sánh 4 config:

| Config | Out Retain | Δ Forget | LP Gap | R angle° | Runtime |
|---|---|---|---|---|---|
| A (04a_raw) | 89–98% | -0.19 → +3.2 | 0.37–2.2 | 3–8° | 566–1672s |
| **B (04a_ext)** | ? | ? | ? | ? | ~2–3× A |
| C (04c) | 91–98% | -0.42 → +3.9 | 0.45–1.64 | 3–10° | 719–1357s |
| **D (04c_ext)** | ? | ? | ? | ? | ~1.5–2× C |

---

## Lưu ý quan trọng

1. **B và D dùng cùng checkpoint prefix `cmf_04a_ext_`** — sub_04a_ext và 04a_extended đều ghi vào `cmf_04a_ext_*`. Nếu chạy cả hai config B và D cùng lúc (trên cùng máy), checkpoint của config B sẽ bị ghi đè bởi sub_04a_ext (hoặc ngược lại). **Chạy tuần tự hoặc đổi prefix cho 1 trong 2.**

2. **Giải pháp nếu cần chạy song song:** Đổi prefix sub_04a_ext thành `cmf_04a_subext_` và cập nhật 04b_ext để đọc `cmf_04a_subext_`.

3. **Thời gian ước tính (10 seeds, 5 methods):**
   - Config B: ~2–3× thời gian config A (~3000–5000s/method)
   - Config D: ~1.5–2× thời gian config C (~1100–2700s/method)
