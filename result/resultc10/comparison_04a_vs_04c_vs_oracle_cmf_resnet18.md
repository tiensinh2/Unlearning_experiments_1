# CMF ResNet18 — Comparison: 04a (seeds 0–9) vs 04c (seeds 0–9) vs Oracle

> **Model:** ResNet18 | **Dataset:** CIFAR-10 | **04a** = batch1 + batch2 (seeds 0–9) | **04c** = seeds 0–9
>
> **Runtime 04c (total)** = sub_04a + 04b + 04c_raw (full pipeline cost)
>
> **Criteria:** Retain ↑ · Forget ≈ Oracle (Δ gần 0) · LP/NCC Illusion Gap ↓ · Runtime ↓

## Oracle Reference (retrain from scratch, 10 seeds)

| Metric | Value |
|---|---|
| Val Acc (Out) | 87.0700 |
| Retain Acc (Out) | 99.9200 |
| Forget Acc (Out) | 86.6100 |
| Test Acc (Out) | 86.3300 |
| Retain Acc (LP) | 99.9400 |
| Forget Acc (LP) | 86.5000 |
| Test Acc (LP) | 86.3300 |
| LP Illusion Gap | 0.0200 |
| Retain Acc (NCC) | 99.9200 |
| Forget Acc (NCC) | 86.4900 |
| Test Acc (NCC) | 86.3300 |
| NCC Illusion Gap | 0.0100 |
| U-MIA (D_F vs D_V) | 50.2600 |
| Indisc (D_V) | 99.1700 |
| T-MIA (D_F vs D_T) | 49.6700 |
| Indisc (D_T) | 99.0000 |

---

## random_label

### Out-of-distribution Evaluation

| Source | Val Acc | Retain Acc | Forget Acc | Δ Forget | Test Acc |
|---|---|---|---|---|---|
| Oracle | 87.070 | 99.920 | 86.610 | — | — |
| 04a | 85.711 | 97.843 | 89.535 | +2.925 | 85.119 |
| 04c | 85.723 | 97.816 | 90.097 | +3.487 | 85.190 |

### Linear Probe (LP) Evaluation

| Source | Retain Acc | Forget Acc | Δ Forget | Test Acc | LP Illusion Gap |
|---|---|---|---|---|---|
| Oracle | 99.940 | 86.500 | — | 86.330 | 0.020 |
| 04a | 98.546 | 90.260 | +3.760 | 85.516 | 0.7525 |
| 04c | 98.545 | 90.920 | +4.420 | 85.575 | 0.8225 |

### Nearest-Class Centroid (NCC) Evaluation

| Source | Retain Acc | Forget Acc | Δ Forget | Test Acc | NCC Illusion Gap |
|---|---|---|---|---|---|
| Oracle | 99.920 | 86.490 | — | 86.330 | 0.010 |
| 04a | 97.889 | 89.610 | +3.120 | 85.201 | 0.1100 |
| 04c | 97.953 | 90.170 | +3.680 | 85.281 | 0.1375 |

### Membership Inference Attack (MIA)

| Source | U-MIA (D_F vs D_V) | Indisc (D_V) | T-MIA (D_F vs D_T) | Indisc (D_T) |
|---|---|---|---|---|
| Oracle | 50.260 | 99.170 | 49.670 | 99.000 |
| 04a | 51.266 | 97.468 | 51.620 | 96.760 |
| 04c | 51.451 | 97.097 | 51.849 | 96.303 |

### Runtime

| Source | Runtime (s) | Breakdown |
|---|---|---|
| 04a | 1634.1 | 04a_raw only |
| 04c | 1092.4 | sub_04a 287.5 + 04b 368.4 + 04c_raw 436.4 |

---

## neggrad_plus

### Out-of-distribution Evaluation

| Source | Val Acc | Retain Acc | Forget Acc | Δ Forget | Test Acc |
|---|---|---|---|---|---|
| Oracle | 87.070 | 99.920 | 86.610 | — | — |
| 04a | 84.304 | 97.420 | 86.422 | -0.188 | 83.759 |
| 04c | 84.375 | 97.716 | 86.192 | -0.418 | 83.886 |

### Linear Probe (LP) Evaluation

| Source | Retain Acc | Forget Acc | Δ Forget | Test Acc | LP Illusion Gap |
|---|---|---|---|---|---|
| Oracle | 99.940 | 86.500 | — | 86.330 | 0.020 |
| 04a | 98.328 | 86.905 | +0.405 | 84.297 | 0.4825 |
| 04c | 98.544 | 86.642 | +0.142 | 84.356 | 0.4500 |

### Nearest-Class Centroid (NCC) Evaluation

| Source | Retain Acc | Forget Acc | Δ Forget | Test Acc | NCC Illusion Gap |
|---|---|---|---|---|---|
| Oracle | 99.920 | 86.490 | — | 86.330 | 0.010 |
| 04a | 97.463 | 86.380 | -0.110 | 83.832 | 0.0625 |
| 04c | 97.806 | 86.290 | -0.200 | 83.997 | 0.1175 |

### Membership Inference Attack (MIA)

| Source | U-MIA (D_F vs D_V) | Indisc (D_V) | T-MIA (D_F vs D_T) | Indisc (D_T) |
|---|---|---|---|---|
| Oracle | 50.260 | 99.170 | 49.670 | 99.000 |
| 04a | 51.125 | 97.750 | 51.446 | 97.107 |
| 04c | 50.562 | 98.235 | 51.019 | 97.573 |

### Runtime

| Source | Runtime (s) | Breakdown |
|---|---|---|
| 04a | 1650.1 | 04a_raw only |
| 04c | 1355.7 | sub_04a 412.9 + 04b 349.5 + 04c_raw 593.3 |

---

## salun

### Out-of-distribution Evaluation

| Source | Val Acc | Retain Acc | Forget Acc | Δ Forget | Test Acc |
|---|---|---|---|---|---|
| Oracle | 87.070 | 99.920 | 86.610 | — | — |
| 04a | 85.736 | 97.988 | 89.828 | +3.218 | 85.298 |
| 04c | 85.828 | 97.969 | 90.508 | +3.898 | 85.355 |

### Linear Probe (LP) Evaluation

| Source | Retain Acc | Forget Acc | Δ Forget | Test Acc | LP Illusion Gap |
|---|---|---|---|---|---|
| Oracle | 99.940 | 86.500 | — | 86.330 | 0.020 |
| 04a | 98.603 | 90.180 | +3.680 | 85.521 | 0.3700 |
| 04c | 98.596 | 91.145 | +4.645 | 85.758 | 0.6375 |

### Nearest-Class Centroid (NCC) Evaluation

| Source | Retain Acc | Forget Acc | Δ Forget | Test Acc | NCC Illusion Gap |
|---|---|---|---|---|---|
| Oracle | 99.920 | 86.490 | — | 86.330 | 0.010 |
| 04a | 98.012 | 89.833 | +3.343 | 85.359 | 0.0600 |
| 04c | 98.041 | 90.573 | +4.083 | 85.448 | 0.0900 |

### Membership Inference Attack (MIA)

| Source | U-MIA (D_F vs D_V) | Indisc (D_V) | T-MIA (D_F vs D_T) | Indisc (D_T) |
|---|---|---|---|---|
| Oracle | 50.260 | 99.170 | 49.670 | 99.000 |
| 04a | 51.217 | 97.565 | 51.479 | 97.042 |
| 04c | 51.623 | 96.755 | 51.898 | 96.205 |

### Runtime

| Source | Runtime (s) | Breakdown |
|---|---|---|
| 04a | 1672.2 | 04a_raw only |
| 04c | 1183.8 | sub_04a 304.4 + 04b 423.4 + 04c_raw 456.0 |

---

## scrub

### Out-of-distribution Evaluation

| Source | Val Acc | Retain Acc | Forget Acc | Δ Forget | Test Acc |
|---|---|---|---|---|---|
| Oracle | 87.070 | 99.920 | 86.610 | — | — |
| 04a | 81.390 | 89.095 | 85.987 | -0.623 | 80.967 |
| 04c | 84.956 | 93.105 | 87.583 | +0.973 | 84.356 |

### Linear Probe (LP) Evaluation

| Source | Retain Acc | Forget Acc | Δ Forget | Test Acc | LP Illusion Gap |
|---|---|---|---|---|---|
| Oracle | 99.940 | 86.500 | — | 86.330 | 0.020 |
| 04a | 91.906 | 88.185 | +1.685 | 82.785 | 2.1975 |
| 04c | 94.372 | 88.255 | +1.755 | 85.032 | 0.6775 |

### Nearest-Class Centroid (NCC) Evaluation

| Source | Retain Acc | Forget Acc | Δ Forget | Test Acc | NCC Illusion Gap |
|---|---|---|---|---|---|
| Oracle | 99.920 | 86.490 | — | 86.330 | 0.010 |
| 04a | 89.632 | 86.472 | -0.017 | 81.555 | 0.4875 |
| 04c | 93.406 | 87.737 | +1.248 | 84.618 | 0.2150 |

### Membership Inference Attack (MIA)

| Source | U-MIA (D_F vs D_V) | Indisc (D_V) | T-MIA (D_F vs D_T) | Indisc (D_T) |
|---|---|---|---|---|
| Oracle | 50.260 | 99.170 | 49.670 | 99.000 |
| 04a | 52.275 | 95.450 | 52.416 | 95.167 |
| 04c | 51.221 | 97.558 | 51.715 | 96.570 |

### Runtime

| Source | Runtime (s) | Breakdown |
|---|---|---|
| 04a | 566.8 | 04a_raw only |
| 04c | 1356.6 | sub_04a 723.3 + 04b 223.0 + 04c_raw 410.3 |

---

## tarun_unsir

### Out-of-distribution Evaluation

| Source | Val Acc | Retain Acc | Forget Acc | Δ Forget | Test Acc |
|---|---|---|---|---|---|
| Oracle | 87.070 | 99.920 | 86.610 | — | — |
| 04a | 82.077 | 91.285 | 89.365 | +2.755 | 81.765 |
| 04c | 82.602 | 91.765 | 89.032 | +2.422 | 82.260 |

### Linear Probe (LP) Evaluation

| Source | Retain Acc | Forget Acc | Δ Forget | Test Acc | LP Illusion Gap |
|---|---|---|---|---|---|
| Oracle | 99.940 | 86.500 | — | 86.330 | 0.020 |
| 04a | 94.150 | 91.648 | +5.148 | 83.521 | 2.2825 |
| 04c | 93.999 | 90.675 | +4.175 | 83.731 | 1.6425 |

### Nearest-Class Centroid (NCC) Evaluation

| Source | Retain Acc | Forget Acc | Δ Forget | Test Acc | NCC Illusion Gap |
|---|---|---|---|---|---|
| Oracle | 99.920 | 86.490 | — | 86.330 | 0.010 |
| 04a | 92.113 | 90.205 | +3.715 | 82.392 | 0.8400 |
| 04c | 92.186 | 89.340 | +2.850 | 82.611 | 0.3350 |

### Membership Inference Attack (MIA)

| Source | U-MIA (D_F vs D_V) | Indisc (D_V) | T-MIA (D_F vs D_T) | Indisc (D_T) |
|---|---|---|---|---|
| Oracle | 50.260 | 99.170 | 49.670 | 99.000 |
| 04a | 53.848 | 92.305 | 54.061 | 91.877 |
| 04c | 53.166 | 93.667 | 53.449 | 93.102 |

### Runtime

| Source | Runtime (s) | Breakdown |
|---|---|---|
| 04a | 128.8 | 04a_raw only |
| 04c | 719.3 | sub_04a 115.5 + 04b 516.8 + 04c_raw 87.1 |

---

## Phân tích Runtime — Tại sao SCRUB và UNSIR nhanh hơn?

### 1. SCRUB — nhanh vì chỉ fine-tune trên subset nhỏ

SCRUB hoạt động theo cơ chế **gradient ascent trên D_forget + gradient descent trên D_retain**:
- Số bước update rất ít (vài epoch trên subset nhỏ), không cần forward pass qua toàn bộ dataset
- 04c_raw chỉ ~410s/run — nhanh hơn random_label (~436s) hay neggrad_plus (~593s) dù số epoch không ít hơn nhiều

### 2. TARUN_UNSIR — nhanh nhất vì không train thêm nhiều

UNSIR (Unlearning via Selective Impair and Repair) có pipeline đặc biệt:
- **Impair step**: inject noise vào activations của forget samples — không backprop qua toàn mạng
- **Repair step**: fine-tune rất ít bước trên retain data để recover accuracy
- 04c_raw chỉ ~87s/run → **5–7× nhanh hơn** các method khác

### 3. Tại sao cộng 3 notebook (sub_04a + 04b + 04c) lại lâu hơn?

sub_04a và 04b là **overhead chung** — chạy như nhau cho mọi method, không phụ thuộc thuật toán unlearning:

| Method | sub_04a | 04b | 04c_raw | 04c Total | % overhead (sub+04b) |
|---|---|---|---|---|---|
| random_label | 287.5 | 368.4 | 436.4 | 1092.4 | 60% |
| neggrad_plus | 412.9 | 349.5 | 593.3 | 1355.7 | 56% |
| salun | 304.4 | 423.4 | 456.0 | 1183.8 | 61% |
| scrub | 723.3 | 223.0 | 410.3 | 1356.6 | 70% |
| tarun_unsir | 115.5 | 516.8 | 87.1 | 719.3 | 88% |

> **UNSIR cực đoan nhất**: 04c_raw chỉ 87s nhưng 04b chiếm 517s (~72% tổng thời gian) — thuật toán unlearning nhanh nhưng pipeline overhead lấn át hoàn toàn.

### 4. Random Label / SalUn / NegGrad+ chậm vì gì?

- **Random Label**: relabel toàn bộ D_forget rồi fine-tune nhiều epoch → nhiều forward/backward pass
- **NegGrad+**: gradient ascent trên D_forget cần nhiều step để converge ổn định, ít step dễ catastrophic forgetting
- **SalUn**: thêm bước **saliency map computation** (tính gradient magnitude trên D_forget) trước khi unlearn → tốn thêm 1 full forward+backward pass trước khi bắt đầu training

---

*Generated from: summary_04a_cmf_resnet18_batch1/2.xlsx + summary_sub_04a + summary_04b + summary_04c + oracle*