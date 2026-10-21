# Báo cáo Thí nghiệm: CMF ResNet18 — Thiết kế Pipeline 04a vs 04c

> **Model:** ResNet18 | **Dataset:** CIFAR-10 | **Seeds:** 0–9 (10 seeds mỗi config)
> **So sánh:** Pipeline **04a** (CMF Static Early Stopping) vs Pipeline **04c** (CMF Adaptive Early Stopping)
> **Reference:** Oracle (retrain from scratch) — pipeline 02

---

## 1. Tổng quan thí nghiệm

Thí nghiệm so sánh hai chiến lược **post-hoc CMF (Classifier-Matching Fine-tuning)** áp dụng sau bước unlearning encoder để căn chỉnh classifier head về phía Oracle. Cả hai pipeline đều dùng cùng 5 unlearning method, cùng 10 seeds, cùng dataset split — sự khác biệt duy nhất là **số epoch tối đa**, **tiêu chí dừng sớm**, và **cấu trúc notebook**.

> **Lưu ý quan trọng**: `04a_cmf_static.ipynb` được chạy **hai lần** với mục đích khác nhau:
> 1. **sub_04a** (lần chạy đầu, epoch thấp): tạo checkpoint `cmf_04a_*.pt` làm input cho 04b và 04c — `MAX_EPOCHS`: 10 (random_label / neggrad_plus / salun) và **18** (scrub); tarun_unsir cố định.
> 2. **04a_raw** (lần chạy sau, epoch đầy đủ): `MAX_EPOCHS_04A = 30` với early stopping — dùng để **so sánh trực tiếp với 04c**.

### 1.1. Danh sách pipeline trong chuỗi thí nghiệm

| Notebook | Tên | Vai trò |
|---|---|---|
| `01_train_full.ipynb` | Full Model Training | Train mô hình đầy đủ trên D_train (D_R ∪ D_F) |
| `02_oracle_retrain.ipynb` | Oracle Retrain | Retrain từ đầu chỉ trên D_R → reference chuẩn |
| `03_standard_mu.ipynb` | Standard MU | Baseline: unlearning cố định epoch, không CMF |
| `04a_cmf_static.ipynb` **(sub_04a)** | CMF Static — lần 1 (epoch thấp) | Tạo checkpoint `cmf_04a_*.pt` cho 04b và 04c; rl/ng/salun=10ep, scrub=18ep |
| `04b_cmf_posthoc.ipynb` | CMF Post-hoc Head | Đọc `cmf_04a_*.pt`, freeze encoder, train lại head trên D_R |
| `04c_cmf_adaptive.ipynb` | CMF Adaptive | Đọc `cmf_04b_*.pt`, unlearning encoder, early stopping theo NCC |
| `04a_cmf_static.ipynb` **(04a_raw)** | CMF Static — lần 2 (30 ep) | Chạy lại với MAX_EPOCHS=30; checkpoint so sánh trực tiếp với 04c |

### 1.2. Cái gì so sánh với cái gì?

- **04a_raw** (CMF Static MAX_EPOCHS=30, seeds 0–9) so sánh trực tiếp với **04c** (CMF Adaptive, seeds 0–9)
- Cả hai đều được so sánh với **Oracle** (pipeline 02)
- **04a_raw** = runtime notebook 04a lần 2 (MAX_EPOCHS=30, early stopping)
- **04c_total** = sub_04a (lần 1, 10/18ep) + 04b + 04c_raw → tổng chi phí pipeline 3-stage
- **03 (Standard MU)** là baseline riêng, không nằm trong so sánh chính nhưng dùng cùng hyperparameter nội tại của từng method

---

## 2. Cấu hình Epoch

### 2.1. Notebook 01 — Full Model Training

| Tham số | Giá trị |
|---|---|
| Epochs | 182 (CIFAR-10/CIFAR-100), 50 (MNIST/Fashion/UTKFace) |
| Batch size | 256 |
| Optimizer | SGD · lr=0.1 · momentum=0.9 · weight_decay=5e-4 |
| LR scheduler | CosineAnnealingLR (T_max=EPOCHS) |
| Early stopping | Không — chạy đủ số epoch |

### 2.2. Notebook 02 — Oracle Retrain

| Tham số | Giá trị |
|---|---|
| Epochs | 182 (CIFAR-10) |
| Tập dữ liệu | D_R only (không có D_F) |
| Optimizer/Scheduler | Giống NB01 |
| Early stopping | Không — chạy đủ số epoch |

### 2.3. Notebook 03 — Standard MU (Baseline, cố định epoch)

Không có early stopping — chạy đúng số epoch đã thiết lập.

| Method | Epochs (max = actual) | lr | Ghi chú |
|---|---|---|---|
| random_label | 10 | 0.013 | Fine-tune với nhãn ngẫu nhiên |
| neggrad_plus | 10 | 0.010 | Gradient ascent (D_F) + descent (D_R), alpha=0.95 |
| salun | 10 | 0.013 | Saliency-weighted unlearning, threshold=0.5 |
| scrub | 20 | 0.006514 | Teacher-student KD, msteps=3, kd_T=3.888, α=0.9696, γ=1.7154 |
| tarun_unsir | fixed* | varies | Impair (noise_epochs=2, noise_steps=15) + Repair (repair_epochs=1) |

> *tarun_unsir không có epoch structure thông thường — chạy theo cấu hình cố định: noise_lr=0.9345, impair_lr=6.638e-5, impair_batches=14, repair_lr=9.582e-4.

### 2.4. Notebook 04a — CMF Static (hai lần chạy khác nhau)

`04a_cmf_static.ipynb` được chạy **hai lần** với mục đích khác nhau, dùng cùng tiêu chí dừng sớm nhưng khác `MAX_EPOCHS`:

**Tiêu chí dừng sớm (chung cho cả sub_04a và 04a_raw):**

```
dist = |NCC_forget − oracle_NCC| + (1 − out_retain) × 0.5
```

Minimize `dist` → dừng khi `dist` không cải thiện sau `PATIENCE_04A=3` epoch liên tiếp (tính từ epoch ≥ MIN_EPOCHS=3). Lưu best model theo `dist` nhỏ nhất.

#### 2.4a. sub_04a — lần chạy đầu (epoch thấp, tạo checkpoint cho 04b/04c)

| Tham số | Giá trị |
|---|---|
| PATIENCE_04A | **3** |
| MIN_EPOCHS_04A | **3** |
| Mục đích | Tạo `cmf_04a_*.pt` làm input cho 04b và 04c |

| Method | Max Epochs (sub_04a) | lr | Ghi chú |
|---|---|---|---|
| random_label | **10** | 0.013 | Dừng sớm theo dist |
| neggrad_plus | **10** | 0.010 | Dừng sớm theo dist |
| salun | **10** | 0.013 | Dừng sớm theo dist |
| scrub | **18** | 0.006514 | Dừng sớm theo dist; msteps=3 |
| tarun_unsir | fixed | varies | Không có epoch structure |

#### 2.4b. 04a_raw — lần chạy sau (MAX_EPOCHS=30, để so sánh với 04c)

| Tham số | Giá trị |
|---|---|
| MAX_EPOCHS_04A | **30** |
| PATIENCE_04A | **3** |
| MIN_EPOCHS_04A | **3** |
| Mục đích | So sánh trực tiếp **04a_raw vs 04c** |

| Method | Max Epochs (04a_raw) | lr | Ghi chú |
|---|---|---|---|
| random_label | **30** | 0.013 | Dừng sớm theo dist |
| neggrad_plus | **30** | 0.010 | Dừng sớm theo dist |
| salun | **30** | 0.013 | Dừng sớm theo dist |
| scrub | **30** | 0.006514 | Dừng sớm theo dist; msteps=3 |
| tarun_unsir | fixed | varies | Không có epoch structure |

### 2.5. Notebook 04b — CMF Post-hoc Head (chỉ fine-tune head)

> Chạy **sau** 04a — encoder đã frozen, chỉ train `model.head` trên D_R.

| Tham số | Giá trị |
|---|---|
| max_epochs | **20** |
| min_epochs | **3** |
| patience | **2** |
| lr | **1e-3** |
| momentum | 0.9 |
| weight_decay | 5e-4 |
| Optimizer | SGD (chỉ trên head parameters) |

**Tiêu chí dừng sớm (04b):** Maximize `retain_acc` (Retain Accuracy trên D_R) — dừng khi không cải thiện sau `patience=2` epoch.

### 2.6. Notebook 04c — CMF Adaptive (encoder unlearning với early stopping thích nghi)

| Tham số | Giá trị |
|---|---|
| MAX_EPOCHS_04C | **10** |
| PATIENCE_04C | **3** |
| MIN_EPOCHS | không có (có thể dừng từ epoch 1 nếu patience đủ) |

**Tiêu chí dừng sớm (04c):**

```
mix_score = (1 − NCC_retain) + |NCC_forget − oracle_NCC|
```

Minimize `mix_score` → cân bằng đồng thời giữa **giữ representation retain set** (NCC Retain↑) và **erasure forget set** (NCC Forget ≈ Oracle). Dừng khi `mix_score` không cải thiện sau `PATIENCE_04C=3` epoch.

**Hyperparameter method trong 04c** (giống 03/04a):

| Method | Max Epochs | lr | Ghi chú |
|---|---|---|---|
| random_label | 10 | 0.013 | Dừng sớm theo mix_score |
| neggrad_plus | 10 | 0.010 | Dừng sớm theo mix_score |
| salun | 10 | 0.013 | Dừng sớm theo mix_score |
| scrub | 10 | 0.006514 | Dừng sớm theo mix_score; msteps=3 |
| tarun_unsir | fixed | varies | Không có epoch structure → không dừng sớm |

---

## 3. So sánh thiết kế 04a vs 04c

| Khía cạnh | 04a (CMF Static) | 04c (CMF Adaptive) |
|---|---|---|
| Max epochs | **30** | **10** |
| Patience | **3** | **3** |
| Min epochs | **3** | Không có |
| Tiêu chí minimize | `\|NCC_forget − oracle\| + (1−out_retain)×0.5` | `(1−NCC_retain) + \|NCC_forget − oracle\|` |
| Thành phần retain | Out_retain (accuracy) × 0.5 | NCC_retain (1 − loss) × 1.0 |
| Thành phần forget | |NCC_forget − oracle| × 1.0 | |NCC_forget − oracle| × 1.0 |
| Pipeline phụ thuộc | Độc lập (self-contained) | Cần sub_04a (checkpoint) + 04b (head) |
| Cách tính runtime | 04a_raw only | sub_04a + 04b + 04c_raw |

**Sự khác biệt thiết kế chính:**

1. **Max epochs**: 04c chỉ cho phép 10 epoch (⅓ so với 04a). Với patience=3, 04c có thể dừng sớm tối thiểu ở epoch 4.
2. **Thành phần retain trong loss**: 04a dùng `out_retain` (accuracy trực tiếp, weight=0.5); 04c dùng `NCC_retain` (representation-based, weight=1.0) — 04c cân bằng retain mạnh hơn.
3. **Pipeline**: 04c là pipeline 3-stage (sub_04a → 04b → 04c), tận dụng checkpoint từ 04a.

---

## 4. Cấu hình Dataset và Model (chung cho tất cả pipeline)

| Tham số | Giá trị |
|---|---|
| Dataset | CIFAR-10 |
| Model | ResNet18 |
| SPLIT_SEED | 123 (cố định cho mọi thí nghiệm) |
| Forget set (D_F) | 1 class (10% of training data) |
| Model seeds | 0, 1, 2, 3, 4, 5, 6, 7, 8, 9 (10 seeds) |
| Batch size | 256 |
| 04a seeds | batch1 = {0–4}, batch2 = {5–9} → gộp = seeds 0–9 |
| 04c seeds | 0–9 (single run) |

---

## 5. Evaluation Protocol

Sau mỗi unlearning run, model được đánh giá qua **3 eval head** và **2 MIA**:

| Eval | Mô tả |
|---|---|
| Out-of-distribution (Out) | Forward pass qua full model; không train thêm |
| Linear Probe (LP) | Train thêm linear layer trên frozen encoder; measure representation quality |
| Nearest-Class Centroid (NCC) | Tính class centroid trên D_R; classify D_F dựa trên cosine distance |
| U-MIA | Unlearn MIA: phân biệt D_F vs D_V (validation) — tốt khi ≈50% |
| T-MIA | Test MIA: phân biệt D_F vs D_T (test) — tốt khi ≈50% |
| Indisc | Indiscrimination: tỉ lệ D_V/D_T bị đoán đúng nhãn — tốt khi ↑ |

**Thứ tự ưu tiên metric:**
1. NCC Retain, NCC Forget ← tiêu chí early stopping
2. Out Retain, Out Forget ← chỉ số primary
3. LP Illusion Gap, NCC Illusion Gap ← chỉ số phụ trợ
4. MIA (U-MIA, T-MIA, Indisc) ← bảo mật
5. Geometry (cos, angle, rel dist, procrustes, gram MAE) ← representation quality

---

## 6. Cấu trúc Output Files

```
result/
  report_experiment_04a_vs_04c_cmf_resnet18.md   ← file này (setup)
  report_metrics_04a_vs_04c_cmf_resnet18.md       ← kết quả số liệu + verdict

result/resultc10/
  comparison_04a_vs_04c_vs_oracle_cmf_resnet18.md  ← chi tiết per-method + runtime
  comparison_04a_vs_04c_vs_oracle_cmf_resnet18.xlsx ← Excel 4 sheets

result/distance_c10/
  comparison_geometry_04a_vs_04c_cmf_resnet18.md   ← geometry metrics
```

---

*Sources: `04a_cmf_static.ipynb` · `04b_cmf_posthoc.ipynb` · `04c_cmf_adaptive.ipynb` · `03_standard_mu.ipynb` · `02_oracle_retrain.ipynb` · `01_train_full.ipynb`*
