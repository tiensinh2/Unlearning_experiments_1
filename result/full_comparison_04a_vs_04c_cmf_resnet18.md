# CMF ResNet18 — Báo cáo tổng hợp: 04a vs 04c vs Oracle

> **Model:** ResNet18 | **Dataset:** CIFAR-10 | **Seeds:** 0–9 (10 seeds mỗi config)
> **04a** = batch1 (seeds 0–4) + batch2 (seeds 5–9) | **04c** = seeds 0–9 (pipeline mới)
> **Runtime 04c (total)** = sub_04a + 04b + 04c_raw

---

## Hướng dẫn đọc chỉ số

| Nhóm | Metric | Tốt khi |
|---|---|---|
| Unlearning | Retain Acc (Out/LP/NCC) | ↑ càng cao càng tốt |
| Unlearning | Forget Acc (Out/LP/NCC) | ≈ Oracle (Δ gần 0) |
| Unlearning | LP / NCC Illusion Gap | ↓ càng thấp càng tốt |
| MIA | U-MIA, T-MIA | ≈ 50% (random guess) |
| MIA | Indisc (D_V), Indisc (D_T) | ↑ càng cao càng tốt |
| Geometry | cos (R/F) | ↑ gần 1 |
| Geometry | angle(deg) (R/F) | ↓ gần 0 |
| Geometry | norm ratio (R/F) | ≈ 1 |
| Geometry | rel dist, procrustes, gram MAE | ↓ càng thấp càng tốt |
| Runtime | Runtime (s) | ↓ càng thấp càng tốt |

---

## Oracle Reference (retrain from scratch, 10 seeds)

| Metric | Value |
|---|---|
| Val Acc (Out) | 87.070 |
| Retain Acc (Out/LP/NCC) | 99.920 / 99.940 / 99.920 |
| Forget Acc (Out/LP/NCC) | 86.610 / 86.500 / 86.490 |
| Test Acc (Out/LP/NCC) | 86.330 / 86.330 / 86.330 |
| LP Illusion Gap | 0.020 |
| NCC Illusion Gap | 0.010 |
| U-MIA / Indisc (D_V) | 50.260 / 99.170 |
| T-MIA / Indisc (D_T) | 49.670 / 99.000 |

---

## 1. random_label

### Unlearning Metrics

| Source | Retain (Out) | Forget (Out) | Δ Fgt | Test (Out) | Retain (LP) | Forget (LP) | Δ Fgt | LP Gap | Retain (NCC) | Forget (NCC) | Δ Fgt | NCC Gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Oracle | 99.920 | 86.610 | — | 86.330 | 99.940 | 86.500 | — | 0.020 | 99.920 | 86.490 | — | 0.010 |
| 04a | 97.843 | 89.535 | +2.925 | 85.119 | 98.546 | 90.260 | +3.760 | 0.753 | 97.889 | 89.610 | +3.120 | 0.110 |
| 04c | 97.816 | 90.097 | +3.487 | 85.190 | 98.545 | 90.920 | +4.420 | 0.823 | 97.953 | 90.170 | +3.680 | 0.138 |

### MIA

| Source | U-MIA | Indisc (D_V) | T-MIA | Indisc (D_T) |
|---|---|---|---|---|
| Oracle | 50.260 | 99.170 | 49.670 | 99.000 |
| 04a | 51.266 | 97.468 | 51.620 | 96.760 |
| 04c | 51.451 | 97.097 | 51.849 | 96.303 |

### Geometry

| Source | R cos | R angle° | R norm ratio | R rel dist | R procrustes | R gram MAE | F cos | F angle° | F norm ratio | F rel dist |
|---|---|---|---|---|---|---|---|---|---|---|
| 04a | 0.9985 | 2.991 | 0.8687 | 0.1416 | 0.0040 | 0.0265 | 0.9966 | 4.332 | 0.7825 | 0.2281 |
| 04c | 0.9980 | 3.491 | 0.7892 | 0.2183 | 0.0052 | 0.0322 | 0.9954 | 5.015 | 0.7161 | 0.2941 |
| Δ(04a−04c) | +0.0005 | **−0.501** | +0.080 | **−0.077** | **−0.001** | **−0.006** | **+0.001** | **−0.684** | +0.066 | **−0.066** |

### Runtime

| Source | Runtime (s) | Breakdown |
|---|---|---|
| 04a | 1634.1 | 04a_raw only |
| 04c | 1092.4 | sub_04a 287.5 + 04b 368.4 + 04c_raw 436.4 |

### Nhận xét — random_label

- **Forget gap**: 04a (+2.93) tốt hơn 04c (+3.49) — cả 3 eval head đều vậy. Tức là 04a quên ít hơn (Forget Acc gần Oracle hơn).
- **Retain**: Hai config gần bằng nhau (~97.8–97.9%), đều kém Oracle ~2%.
- **Illusion Gap**: 04a thấp hơn (LP: 0.753 vs 0.823; NCC: 0.110 vs 0.138) → 04a ít bị LP/NCC đánh giá nhầm hơn.
- **MIA**: Cả hai gần nhau, Indisc 04a nhỉnh hơn (~97.5 vs 97.1) → 04a khó phân biệt forget/val hơn.
- **Geometry**: 04a geometry gần oracle hơn trên tất cả metrics (angle thấp hơn, rel dist thấp hơn, procrustes thấp hơn).
- **Runtime**: 04a chậm hơn (1634s vs 1092s do pipeline 04c tích hợp sub_04a+04b).
- **Verdict**: **04a tốt hơn** về chất lượng unlearning và geometry; 04c nhanh hơn ~33%.

---

## 2. neggrad_plus

### Unlearning Metrics

| Source | Retain (Out) | Forget (Out) | Δ Fgt | Test (Out) | Retain (LP) | Forget (LP) | Δ Fgt | LP Gap | Retain (NCC) | Forget (NCC) | Δ Fgt | NCC Gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Oracle | 99.920 | 86.610 | — | 86.330 | 99.940 | 86.500 | — | 0.020 | 99.920 | 86.490 | — | 0.010 |
| 04a | 97.420 | 86.422 | −0.188 | 83.759 | 98.328 | 86.905 | +0.405 | 0.483 | 97.463 | 86.380 | −0.110 | 0.063 |
| 04c | 97.716 | 86.192 | −0.418 | 83.886 | 98.544 | 86.642 | +0.142 | 0.450 | 97.806 | 86.290 | −0.200 | 0.118 |

### MIA

| Source | U-MIA | Indisc (D_V) | T-MIA | Indisc (D_T) |
|---|---|---|---|---|
| Oracle | 50.260 | 99.170 | 49.670 | 99.000 |
| 04a | 51.125 | 97.750 | 51.446 | 97.107 |
| 04c | 50.562 | 98.235 | 51.019 | 97.573 |

### Geometry

| Source | R cos | R angle° | R norm ratio | R rel dist | R procrustes | R gram MAE | F cos | F angle° | F norm ratio | F rel dist |
|---|---|---|---|---|---|---|---|---|---|---|
| 04a | 0.9983 | 3.219 | 1.3866 | 0.3928 | 0.0044 | 0.0300 | 0.9940 | 5.508 | 1.2367 | 0.2748 |
| 04c | 0.9984 | 3.125 | 1.3819 | 0.3877 | 0.0050 | 0.0288 | 0.9940 | 5.468 | 1.2261 | 0.2660 |
| Δ(04a−04c) | −0.0001 | +0.094 | +0.005 | +0.005 | −0.001 | +0.001 | 0.000 | +0.040 | +0.011 | +0.009 |

### Runtime

| Source | Runtime (s) | Breakdown |
|---|---|---|
| 04a | 1650.1 | 04a_raw only |
| 04c | 1355.7 | sub_04a 412.9 + 04b 349.5 + 04c_raw 593.3 |

### Nhận xét — neggrad_plus

- **Forget gap**: Cả hai đều rất gần Oracle (<0.5%) — đây là method tốt nhất về forgetting. 04c nhỉnh hơn nhẹ (−0.42 vs −0.19 ở Out head).
- **Retain**: 04c cao hơn ~0.3% (97.72 vs 97.42 ở Out) — nhỏ nhưng nhất quán qua 3 eval heads.
- **Illusion Gap**: 04a thấp hơn ở NCC (0.063 vs 0.118), 04c thấp hơn nhẹ ở LP (0.450 vs 0.483).
- **MIA**: 04c tốt hơn — Indisc cao hơn (98.24 vs 97.75), U-MIA gần 50 hơn (50.56 vs 51.13).
- **Geometry**: Cực kỳ gần nhau, 04c nhỉnh hơn rất nhẹ trên hầu hết metrics.
- **Runtime**: 04a chậm hơn (1650s vs 1356s).
- **Verdict**: **04c nhỉnh hơn** toàn diện (Retain, MIA, Geometry) nhưng sự khác biệt rất nhỏ (<0.5% đa số); 04a có NCC Illusion Gap thấp hơn.

---

## 3. salun

### Unlearning Metrics

| Source | Retain (Out) | Forget (Out) | Δ Fgt | Test (Out) | Retain (LP) | Forget (LP) | Δ Fgt | LP Gap | Retain (NCC) | Forget (NCC) | Δ Fgt | NCC Gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Oracle | 99.920 | 86.610 | — | 86.330 | 99.940 | 86.500 | — | 0.020 | 99.920 | 86.490 | — | 0.010 |
| 04a | 97.988 | 89.828 | +3.218 | 85.298 | 98.603 | 90.180 | +3.680 | 0.370 | 98.012 | 89.833 | +3.343 | 0.060 |
| 04c | 97.969 | 90.508 | +3.898 | 85.355 | 98.596 | 91.145 | +4.645 | 0.638 | 98.041 | 90.573 | +4.083 | 0.090 |

### MIA

| Source | U-MIA | Indisc (D_V) | T-MIA | Indisc (D_T) |
|---|---|---|---|---|
| Oracle | 50.260 | 99.170 | 49.670 | 99.000 |
| 04a | 51.217 | 97.565 | 51.479 | 97.042 |
| 04c | 51.623 | 96.755 | 51.898 | 96.205 |

### Geometry

| Source | R cos | R angle° | R norm ratio | R rel dist | R procrustes | R gram MAE | F cos | F angle° | F norm ratio | F rel dist |
|---|---|---|---|---|---|---|---|---|---|---|
| 04a | 0.9985 | 2.991 | 0.8740 | 0.1363 | 0.0037 | 0.0270 | 0.9963 | 4.472 | 0.7866 | 0.2247 |
| 04c | 0.9983 | 3.205 | 0.7753 | 0.2308 | 0.0047 | 0.0288 | 0.9959 | 4.700 | 0.7045 | 0.3040 |
| Δ(04a−04c) | **+0.0002** | **−0.213** | +0.099 | **−0.095** | **−0.001** | **−0.002** | **+0.000** | **−0.228** | +0.082 | **−0.079** |

### Runtime

| Source | Runtime (s) | Breakdown |
|---|---|---|
| 04a | 1672.2 | 04a_raw only |
| 04c | 1183.8 | sub_04a 304.4 + 04b 423.4 + 04c_raw 456.0 |

### Nhận xét — salun

- **Forget gap**: 04a tốt hơn rõ (Out: +3.22 vs +3.90; LP: +3.68 vs +4.65; NCC: +3.34 vs +4.08) — 04a quên chính xác hơn.
- **Retain**: Gần bằng nhau (~98.0%) — cả hai rất tốt so với các method khác.
- **Illusion Gap**: 04a thấp hơn đáng kể (LP: 0.370 vs 0.638; NCC: 0.060 vs 0.090).
- **MIA**: 04a tốt hơn — Indisc cao hơn (97.57 vs 96.76), MIA gần 50 hơn.
- **Geometry**: 04a geometry tốt hơn trên tất cả metrics (rel dist thấp hơn ~0.09, angle thấp hơn ~0.2°).
- **Runtime**: 04a chậm hơn (1672s vs 1184s).
- **Verdict**: **04a tốt hơn rõ ràng** về forget quality, illusion gap, MIA, và geometry.

---

## 4. scrub

### Unlearning Metrics

| Source | Retain (Out) | Forget (Out) | Δ Fgt | Test (Out) | Retain (LP) | Forget (LP) | Δ Fgt | LP Gap | Retain (NCC) | Forget (NCC) | Δ Fgt | NCC Gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Oracle | 99.920 | 86.610 | — | 86.330 | 99.940 | 86.500 | — | 0.020 | 99.920 | 86.490 | — | 0.010 |
| 04a | 89.095 | 85.987 | −0.623 | 80.967 | 91.906 | 88.185 | +1.685 | 2.198 | 89.632 | 86.472 | −0.017 | 0.488 |
| 04c | 93.105 | 87.583 | +0.973 | 84.356 | 94.372 | 88.255 | +1.755 | 0.678 | 93.406 | 87.737 | +1.248 | 0.215 |

### MIA

| Source | U-MIA | Indisc (D_V) | T-MIA | Indisc (D_T) |
|---|---|---|---|---|
| Oracle | 50.260 | 99.170 | 49.670 | 99.000 |
| 04a | 52.275 | 95.450 | 52.416 | 95.167 |
| 04c | 51.221 | 97.558 | 51.715 | 96.570 |

### Geometry

| Source | R cos | R angle° | R norm ratio | R rel dist | R procrustes | R gram MAE | F cos | F angle° | F norm ratio | F rel dist |
|---|---|---|---|---|---|---|---|---|---|---|
| 04a | 0.9953 | 5.240 | 0.8881 | 0.1479 | 0.0151 | 0.0460 | 0.9937 | 5.899 | 0.8612 | 0.1729 |
| 04c | 0.9965 | 4.183 | 0.9122 | 0.1156 | 0.0103 | 0.0364 | 0.9935 | 5.466 | 0.8651 | 0.1634 |
| Δ(04a−04c) | −0.0012 | **+1.058** | −0.024 | **+0.032** | **+0.005** | **+0.010** | +0.0002 | **+0.433** | −0.004 | **+0.010** |

### Runtime

| Source | Runtime (s) | Breakdown |
|---|---|---|
| 04a | 566.8 | 04a_raw only |
| 04c | 1356.6 | sub_04a 723.3 + 04b 223.0 + 04c_raw 410.3 |

### Nhận xét — scrub

- **Retain**: 04c cao hơn **+4%** (93.1 vs 89.1 ở Out) — đây là sự khác biệt lớn nhất trong tất cả methods. 04a bị mất accuracy retain đáng kể.
- **Forget gap**: 04a Forget Out (−0.623) gần Oracle hơn 04c (+0.973), nhưng 04a bị **LP gap cao bất thường** (2.198 vs 0.678) — LP nhìn thấy forget samples bị xử lý khác biệt.
- **Illusion Gap**: 04c thấp hơn rõ rệt (LP: 0.678 vs 2.198; NCC: 0.215 vs 0.488) — 04a có vấn đề lớn về LP illusion.
- **MIA**: 04c tốt hơn — Indisc cao hơn (97.56 vs 95.45), U-MIA gần 50 hơn (51.22 vs 52.28).
- **Geometry**: 04c geometry tốt hơn toàn diện (angle thấp hơn ~1°, rel dist thấp hơn, procrustes/gram MAE thấp hơn).
- **Runtime**: 04a nhanh hơn nhiều (567s vs 1357s) vì scrub là method nhẹ.
- **Verdict**: **04c tốt hơn rõ ràng** — Retain cao hơn +4%, Illusion Gap thấp hơn, MIA tốt hơn, geometry tốt hơn. 04a nhanh hơn nhưng chất lượng unlearning kém.

---

## 5. tarun_unsir

### Unlearning Metrics

| Source | Retain (Out) | Forget (Out) | Δ Fgt | Test (Out) | Retain (LP) | Forget (LP) | Δ Fgt | LP Gap | Retain (NCC) | Forget (NCC) | Δ Fgt | NCC Gap |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Oracle | 99.920 | 86.610 | — | 86.330 | 99.940 | 86.500 | — | 0.020 | 99.920 | 86.490 | — | 0.010 |
| 04a | 91.285 | 89.365 | +2.755 | 81.765 | 94.150 | 91.648 | +5.148 | 2.283 | 92.113 | 90.205 | +3.715 | 0.840 |
| 04c | 91.765 | 89.032 | +2.422 | 82.260 | 93.999 | 90.675 | +4.175 | 1.643 | 92.186 | 89.340 | +2.850 | 0.335 |

### MIA

| Source | U-MIA | Indisc (D_V) | T-MIA | Indisc (D_T) |
|---|---|---|---|---|
| Oracle | 50.260 | 99.170 | 49.670 | 99.000 |
| 04a | 53.848 | 92.305 | 54.061 | 91.877 |
| 04c | 53.166 | 93.667 | 53.449 | 93.102 |

### Geometry

| Source | R cos | R angle° | R norm ratio | R rel dist | R procrustes | R gram MAE | F cos | F angle° | F norm ratio | F rel dist |
|---|---|---|---|---|---|---|---|---|---|---|
| 04a | 0.9889 | 8.246 | 1.2848 | 0.3407 | 0.0326 | 0.0818 | 0.9879 | 8.563 | 1.2668 | 0.3346 |
| 04c | 0.9828 | 10.215 | 1.2863 | 0.3725 | 0.0514 | 0.1061 | 0.9806 | 10.728 | 1.2607 | 0.3644 |
| Δ(04a−04c) | **+0.006** | **−1.968** | −0.002 | **−0.032** | **−0.019** | **−0.024** | **+0.007** | **−2.165** | +0.006 | **−0.030** |

### Runtime

| Source | Runtime (s) | Breakdown |
|---|---|---|
| 04a | 128.8 | 04a_raw only |
| 04c | 719.3 | sub_04a 115.5 + 04b 516.8 + 04c_raw 87.1 |

### Nhận xét — tarun_unsir

- **Forget gap**: Cả hai đều overshot Oracle ~2.4–2.8% — không tốt. 04c nhỉnh hơn nhẹ (LP: +4.18 vs +5.15; NCC: +2.85 vs +3.72).
- **Retain**: Cả hai thấp đáng kể so với Oracle (~91–92%), 04c nhỉnh hơn nhẹ.
- **Illusion Gap**: 04c thấp hơn rõ ràng (LP: 1.643 vs 2.283; NCC: 0.335 vs 0.840) — 04a có LP illusion gap cao nhất.
- **MIA**: 04c tốt hơn — Indisc cao hơn (~93.7 vs 92.3), MIA gần 50 hơn. Cả hai đều kém Oracle nhất trong tất cả methods.
- **Geometry**: 04a geometry tốt hơn rõ rệt (angle thấp hơn ~2°, rel dist thấp hơn, procrustes/gram MAE thấp hơn) — paradox với unlearning metrics.
- **Runtime**: 04a nhanh hơn nhiều (129s vs 719s vì 04b chiếm 517s).
- **Verdict**: **Kết quả trái chiều** — 04a tốt hơn về geometry, 04c tốt hơn về Illusion Gap và MIA. Cả hai đều là method yếu nhất về unlearning quality (Retain thấp, Forget overshot, MIA xa 50).

---

## Tổng kết so sánh 04a vs 04c

### Bảng tổng hợp verdict

| Method | Forget quality | Retain | Illusion Gap | MIA | Geometry | Runtime | Verdict tổng |
|---|---|---|---|---|---|---|---|
| random_label | **04a** tốt hơn | ~bằng | **04a** thấp hơn | **04a** nhỉnh | **04a** tốt hơn | 04a chậm hơn | **04a tốt hơn** |
| neggrad_plus | 04c nhỉnh nhẹ | **04c** cao hơn | Gần bằng | **04c** tốt hơn | ~bằng | 04a chậm hơn | **04c nhỉnh nhẹ** |
| salun | **04a** tốt hơn | ~bằng | **04a** thấp hơn | **04a** tốt hơn | **04a** tốt hơn | 04a chậm hơn | **04a tốt hơn rõ** |
| scrub | 04a Fgt Out gần hơn nhưng LP Gap cao bất thường | **04c** cao hơn +4% | **04c** thấp hơn | **04c** tốt hơn | **04c** tốt hơn | **04a** nhanh hơn | **04c tốt hơn rõ** |
| tarun_unsir | 04c nhỉnh nhẹ | **04c** nhỉnh nhẹ | **04c** thấp hơn | **04c** tốt hơn | **04a** tốt hơn | **04a** nhanh hơn | **Trái chiều** |

### Nhận xét chung

1. **neggrad_plus là method tốt nhất** về unlearning quality trong cả 04a và 04c — Forget gap dưới 0.5% so với Oracle, Retain ~97–98%, Illusion Gap thấp.

2. **04a vs 04c không có winner tuyệt đối** — phụ thuộc vào method:
   - 04a tốt hơn với **random_label** và **salun** (forget chính xác hơn, geometry gần Oracle hơn)
   - 04c tốt hơn với **scrub** (Retain cao hơn +4%, LP Illusion Gap thấp hơn 3×)
   - 04c nhỉnh hơn với **neggrad_plus** và **tarun_unsir** (nhưng không đáng kể)

3. **scrub trong 04a có vấn đề**: LP Illusion Gap = 2.198 (cao nhất trong tất cả) trong khi Retain chỉ 89% — cho thấy cấu hình 04a không phù hợp với scrub.

4. **tarun_unsir yếu nhất** trong cả hai config: Retain chỉ ~91–92%, Forget overshot ~2.4–5%, MIA xa 50 nhất — geometry tốt nhưng unlearning kém.

5. **Runtime**: 04a nhanh hơn về 04a_raw riêng lẻ với scrub (567s) và tarun_unsir (129s). 04c nhanh hơn tổng thể với random_label (1092s vs 1634s) và salun (1184s vs 1672s). Với neggrad_plus gần bằng nhau (1356s vs 1650s).

---

*Generated from: resultc10/ (unlearning metrics) + distance_c10/ (geometry metrics)*
*Sources: summary_04a_cmf_resnet18_batch1/2.xlsx + summary_sub_04a + summary_04b + summary_04c + summary_oracle + summary_geometry_04a_batch1/2 + summary_geometry_04c*
