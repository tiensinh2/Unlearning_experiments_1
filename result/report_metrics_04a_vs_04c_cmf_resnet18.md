# Báo cáo Kết quả: CMF ResNet18 — 04a vs 04c vs Oracle

> **Model:** ResNet18 | **Dataset:** CIFAR-10 | **Seeds:** 0–9 (10 seeds)
> **04a** = batch1 (seeds 0–4) + batch2 (seeds 5–9) | **04c** = seeds 0–9
> **Runtime 04c** = sub_04a + 04b + 04c_raw | **Runtime 04a** = 04a_raw only
>
> Chi tiết thiết kế pipeline, epoch, và tiêu chí dừng sớm → xem `report_experiment_04a_vs_04c_cmf_resnet18.md`

---

## Oracle Reference (retrain from scratch, 10 seeds)

| | Retain (Out) | Forget (Out) | Retain (LP) | Forget (LP) | LP Gap | Retain (NCC) | Forget (NCC) | NCC Gap | U-MIA | Indisc(D_V) | T-MIA | Indisc(D_T) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Oracle | 99.920 | 86.610 | 99.940 | 86.500 | 0.020 | 99.920 | 86.490 | 0.010 | 50.26 | 99.17 | 49.67 | 99.00 |

---

## Bảng kết quả tổng hợp — Out-of-distribution Head

> **Δ Forget** = Forget Acc − Oracle Forget. Gần 0 = tốt. Dương = quên quá nhiều. Âm = quên quá ít.

| Method | Src | Retain (Out) | Forget (Out) | Δ Forget | Test (Out) |
|---|---|---|---|---|---|
| Oracle | — | 99.920 | 86.610 | — | 86.330 |
| random_label | 04a | 97.843 | 89.535 | +2.925 | 85.119 |
| random_label | 04c | 97.816 | 90.097 | +3.487 | 85.190 |
| neggrad_plus | 04a | 97.420 | 86.422 | **−0.188** | 83.759 |
| neggrad_plus | 04c | 97.716 | 86.192 | **−0.418** | 83.886 |
| salun | 04a | 97.988 | 89.828 | +3.218 | 85.298 |
| salun | 04c | 97.969 | 90.508 | +3.898 | 85.355 |
| scrub | 04a | 89.095 | 85.987 | −0.623 | 80.967 |
| scrub | 04c | **93.105** | 87.583 | +0.973 | **84.356** |
| tarun_unsir | 04a | 91.285 | 89.365 | +2.755 | 81.765 |
| tarun_unsir | 04c | 91.765 | 89.032 | +2.422 | 82.260 |

## Bảng kết quả tổng hợp — Linear Probe (LP) Head

| Method | Src | Retain (LP) | Forget (LP) | Δ Forget | Test (LP) | LP Illusion Gap |
|---|---|---|---|---|---|---|
| Oracle | — | 99.940 | 86.500 | — | 86.330 | 0.020 |
| random_label | 04a | 98.546 | 90.260 | +3.760 | 85.516 | 0.753 |
| random_label | 04c | 98.545 | 90.920 | +4.420 | 85.575 | 0.823 |
| neggrad_plus | 04a | 98.328 | 86.905 | +0.405 | 84.297 | 0.483 |
| neggrad_plus | 04c | 98.544 | 86.642 | **+0.142** | 84.356 | **0.450** |
| salun | 04a | 98.603 | 90.180 | +3.680 | 85.521 | **0.370** |
| salun | 04c | 98.596 | 91.145 | +4.645 | 85.758 | 0.638 |
| scrub | 04a | 91.906 | 88.185 | +1.685 | 82.785 | 2.198 |
| scrub | 04c | **94.372** | 88.255 | +1.755 | **85.032** | **0.678** |
| tarun_unsir | 04a | 94.150 | 91.648 | +5.148 | 83.521 | 2.283 |
| tarun_unsir | 04c | 93.999 | 90.675 | +4.175 | 83.731 | **1.643** |

## Bảng kết quả tổng hợp — Nearest-Class Centroid (NCC) Head

| Method | Src | Retain (NCC) | Forget (NCC) | Δ Forget | Test (NCC) | NCC Illusion Gap |
|---|---|---|---|---|---|---|
| Oracle | — | 99.920 | 86.490 | — | 86.330 | 0.010 |
| random_label | 04a | 97.889 | 89.610 | +3.120 | 85.201 | 0.110 |
| random_label | 04c | 97.953 | 90.170 | +3.680 | 85.281 | 0.138 |
| neggrad_plus | 04a | 97.463 | 86.380 | **−0.110** | 83.832 | **0.063** |
| neggrad_plus | 04c | 97.806 | 86.290 | **−0.200** | 83.997 | 0.118 |
| salun | 04a | 98.012 | 89.833 | +3.343 | 85.359 | **0.060** |
| salun | 04c | 98.041 | 90.573 | +4.083 | 85.448 | 0.090 |
| scrub | 04a | 89.632 | 86.472 | −0.017 | 81.555 | 0.488 |
| scrub | 04c | **93.406** | 87.737 | +1.248 | **84.618** | **0.215** |
| tarun_unsir | 04a | 92.113 | 90.205 | +3.715 | 82.392 | 0.840 |
| tarun_unsir | 04c | 92.186 | 89.340 | +2.850 | 82.611 | **0.335** |

---

## Bảng kết quả — Membership Inference Attack (MIA)

> U-MIA / T-MIA tốt khi ≈ 50% (random guess). Indisc tốt khi ↑ (gần Oracle).

| Method | Src | U-MIA | Indisc (D_V) | T-MIA | Indisc (D_T) |
|---|---|---|---|---|---|
| Oracle | — | 50.260 | 99.170 | 49.670 | 99.000 |
| random_label | 04a | 51.266 | **97.468** | 51.620 | **96.760** |
| random_label | 04c | 51.451 | 97.097 | 51.849 | 96.303 |
| neggrad_plus | 04a | 51.125 | 97.750 | 51.446 | 97.107 |
| neggrad_plus | 04c | **50.562** | **98.235** | **51.019** | **97.573** |
| salun | 04a | **51.217** | **97.565** | **51.479** | **97.042** |
| salun | 04c | 51.623 | 96.755 | 51.898 | 96.205 |
| scrub | 04a | 52.275 | 95.450 | 52.416 | 95.167 |
| scrub | 04c | **51.221** | **97.558** | **51.715** | **96.570** |
| tarun_unsir | 04a | 53.848 | 92.305 | 54.061 | 91.877 |
| tarun_unsir | 04c | **53.166** | **93.667** | **53.449** | **93.102** |

---

## Bảng kết quả — Geometry (Retain Set)

> cos↑ · angle°↓ · norm ratio ≈ 1 · rel dist↓ · procrustes↓ · gram MAE↓

| Method | Src | R cos | R angle° | R norm ratio | R rel dist | R procrustes | R gram MAE |
|---|---|---|---|---|---|---|---|
| random_label | 04a | **0.9985** | **2.991** | 0.8687 | **0.1416** | **0.0040** | **0.0265** |
| random_label | 04c | 0.9980 | 3.491 | 0.7892 | 0.2183 | 0.0052 | 0.0322 |
| neggrad_plus | 04a | 0.9983 | 3.219 | 1.3866 | 0.3928 | **0.0044** | 0.0300 |
| neggrad_plus | 04c | **0.9984** | **3.125** | **1.3819** | **0.3877** | 0.0050 | **0.0288** |
| salun | 04a | **0.9985** | **2.991** | 0.8740 | **0.1363** | **0.0037** | **0.0270** |
| salun | 04c | 0.9983 | 3.205 | 0.7753 | 0.2308 | 0.0047 | 0.0288 |
| scrub | 04a | 0.9953 | 5.240 | 0.8881 | 0.1479 | 0.0151 | 0.0460 |
| scrub | 04c | **0.9965** | **4.183** | **0.9122** | **0.1156** | **0.0103** | **0.0364** |
| tarun_unsir | 04a | **0.9889** | **8.246** | **1.2848** | **0.3407** | **0.0326** | **0.0818** |
| tarun_unsir | 04c | 0.9828 | 10.215 | 1.2863 | 0.3725 | 0.0514 | 0.1061 |

## Bảng kết quả — Geometry (Forget Set)

| Method | Src | F cos | F angle° | F norm ratio | F rel dist |
|---|---|---|---|---|---|
| random_label | 04a | **0.9966** | **4.332** | 0.7825 | **0.2281** |
| random_label | 04c | 0.9954 | 5.015 | 0.7161 | 0.2941 |
| neggrad_plus | 04a | 0.9940 | 5.508 | 1.2367 | 0.2748 |
| neggrad_plus | 04c | 0.9940 | **5.468** | **1.2261** | **0.2660** |
| salun | 04a | **0.9963** | **4.472** | 0.7866 | **0.2247** |
| salun | 04c | 0.9959 | 4.700 | 0.7045 | 0.3040 |
| scrub | 04a | 0.9937 | 5.899 | 0.8612 | 0.1729 |
| scrub | 04c | 0.9935 | **5.466** | **0.8651** | **0.1634** |
| tarun_unsir | 04a | **0.9879** | **8.563** | 1.2668 | **0.3346** |
| tarun_unsir | 04c | 0.9806 | 10.728 | **1.2607** | 0.3644 |

---

## Bảng kết quả — Runtime

| Method | 04a (s) | 04c total (s) | sub_04a (s) | 04b (s) | 04c_raw (s) | % overhead (sub+04b) |
|---|---|---|---|---|---|---|
| random_label | 1634.1 | 1092.4 | 287.5 | 368.4 | 436.4 | 60% |
| neggrad_plus | 1650.1 | 1355.7 | 412.9 | 349.5 | 593.3 | 56% |
| salun | 1672.2 | 1183.8 | 304.4 | 423.4 | 456.0 | 61% |
| scrub | 566.8 | 1356.6 | 723.3 | 223.0 | 410.3 | 70% |
| tarun_unsir | 128.8 | 719.3 | 115.5 | 516.8 | 87.1 | 88% |

> **UNSIR cực đoan nhất**: 04c_raw chỉ 87s nhưng 04b chiếm 517s (~72% tổng) — algorithm nhanh nhưng pipeline overhead lấn át.

---

## Phân tích: Tại sao scrub khác biệt lớn (04a Retain = 89% vs 04c = 93%)?

SCRUB là method duy nhất dùng cơ chế **teacher-student KD** trong khi unlearning:

```
L = L_ascent(D_forget) + λ · L_KD(D_retain, teacher)
```

**Nguyên nhân trong 04a**: tiêu chí dừng 04a dùng `out_retain × 0.5` (weight thấp) → early stopping cho phép gradient ascent chạy đủ mạnh để làm giảm Retain accuracy trước khi dừng. Kết quả: Retain=89%, LP Illusion Gap=2.198 (cao bất thường — dấu hiệu catastrophic degradation một phần).

**Cách 04c khắc phục**: tiêu chí 04c dùng `(1−NCC_retain) × 1.0` (weight gấp đôi, dùng representation-based metric) → dừng sớm hơn khi representation retain bị ảnh hưởng → Retain=93%, LP Gap=0.678 (giảm 3×).

**Biên độ khác biệt (Δ = 04c − 04a):**

| Method | ΔRetain (Out) | ΔForget (Out) | ΔLP Gap | ΔIndisc(D_V) |
|---|---|---|---|---|
| random_label | −0.027 | +0.562 | +0.070 | −0.371 |
| neggrad_plus | +0.296 | −0.230 | −0.033 | +0.485 |
| salun | −0.019 | +0.680 | +0.268 | −0.810 |
| tarun_unsir | +0.480 | −0.333 | −0.640 | +1.362 |
| **scrub** | **+4.010** | **+1.596** | **−1.520** | **+2.108** |

→ scrub lệch gấp **5–15×** so với các method khác trên metric Retain và LP Gap.

---

## Verdict tổng hợp

| Method | Config tốt hơn | Lý do chính |
|---|---|---|
| random_label | **04a** nhỉnh | Forget gap nhỏ hơn, geometry gần Oracle hơn |
| neggrad_plus | **04c** nhỉnh | Retain cao hơn +0.3%, MIA tốt hơn |
| salun | **04a** tốt hơn | Forget gap nhỏ hơn ~0.7%, Illusion Gap thấp hơn, geometry tốt hơn |
| scrub | **04c** tốt hơn rõ | Retain cao hơn +4%, LP Gap thấp 3×, MIA tốt hơn, geometry tốt hơn |
| tarun_unsir | Trái chiều | 04a geometry tốt hơn; 04c Illusion Gap và MIA tốt hơn |

**Method tốt nhất tổng thể:** `neggrad_plus` — Forget gap &lt;0.5%, Retain ~97–98%, Illusion Gap thấp nhất, MIA gần Oracle nhất trong cả hai config.

**Method kém nhất:** `tarun_unsir` — Retain thấp ~91%, Forget overshot ~3–5%, MIA xa Oracle nhất.

---

*Sources: `result/resultc10/` (unlearning + MIA) · `result/distance_c10/` (geometry) · `result/resultc10/summary_oracle_resnet18_by_method.xlsx` (Oracle)*
