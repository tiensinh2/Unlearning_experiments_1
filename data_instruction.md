# Machine Unlearning Dataset Preparation and Splitting Instructions

This document provides exact guidelines and instructions for partitioning datasets into **Train ($D$)**, **Validation ($D_V$)**, **Test ($D_T$)**, **Retain ($D_R$)**, and **Forget ($D_F$)** splits to replicate the experimental benchmark in **"Deep Unlearn: Benchmarking Machine Unlearning for Image Classification"** (Cadet et al., arXiv:2410.01276v2).

---

## 1. Core Principle & Interpretation of the Forget Set

The foundational rule of machine unlearning:

> **A forget sample must have been used to train the original model before it can be unlearned.**

Therefore:
$$D = D_R \cup D_F$$
is the training set of the original model:
$$\theta_O = \text{Train}(D_R \cup D_F)$$

and:
* **$D_R$ (Retain Set):** Training data that must be preserved and retained.
* **$D_F$ (Forget Set):** Data that was **actually used to train the original model** and must subsequently be forgotten.
* **$D_V$ (Validation Set):** Data **never used to train the original model**; used strictly for model selection, early stopping, and as the **unseen non-member reference** for Membership Inference Attacks (U-MIA).
* **$D_T$ (Official Test Set):** Official held-out test data **never used during training or unlearning**; used strictly for final out-of-sample generalization reporting.

> **Rules:**
> 1. Do **NOT** redefine $D_F$ to include $D_V$ or $D_T$.
> 2. Do **NOT** claim that $D_T$ or $D_V$ are part of the forget set merely because they are useful as unseen evaluation data.

---

## 2. Dataset Hierarchy & Invariants

For CIFAR-10 (and CIFAR-100):

```text
Full dataset = 60,000
│
├── D_T = 10,000       official test set (train=False)
│
└── D_dev = 50,000     official training/development set (train=True)
    │
    ├── D_V = 10,000   validation / MIA non-member reference (20% of D_dev)
    │
    └── D = 40,000     original training data (80% of D_dev)
        │
        ├── D_R = 36,000   retain set (90% of D)
        └── D_F = 4,000    forget set (10% of D)
```

### Required Mathematical Invariants:
$$D_{train} \cap D_V = \emptyset$$
$$D_R \cap D_F = \emptyset$$
$$D_R \cup D_F = D_{train}$$
$$D_{dev} \cap D_T = \emptyset$$
$$D_T \cap D_R = D_T \cap D_F = D_T \cap D_V = \emptyset$$

---

## 3. Dataset Split Sizes (Matching Benchmark Table 2)

| Dataset | Total Size | Retain ($D_R$) | Forget ($D_F$) | Validation ($D_V$) | Test ($D_T$) | Classes | Task Type | Input Size |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| **MNIST** | 70,000 | 43,200 | 4,800 | 12,000 | 10,000 | 10 | Handwritten Digits | 32×32 (3-ch) |
| **FashionMNIST** | 70,000 | 43,200 | 4,800 | 12,000 | 10,000 | 10 | Fashion Items | 32×32 (3-ch) |
| **CIFAR-10** | 60,000 | 36,000 | 4,000 | 10,000 | 10,000 | 10 | Natural Objects | 32×32 (3-ch) |
| **CIFAR-100** | 60,000 | 36,000 | 4,000 | 10,000 | 10,000 | 100 | Fine-grained Objects | 32×32 (3-ch) |
| **UTKFace** | 23,708 | 13,657 | 1,517 | 3,793 | 4,741 | 5 | Age Classification | 224×224 (3-ch) |

Split protocol (paper §4): seed 123; $D_{dev}$ → $D$ / $D_V$ = 80% / 20%; $D$ → $D_R$ / $D_F$ = 90% / 10% (uniform). Original models use 10 seeds (`model_seeds = [0..9]`).

---

## 4. Preprocessing & Transformations

| Dataset | Train Data Augmentations (`train_tf`) | Test / Eval Transformations (`eval_tf`) | Normalization (Mean / Std) |
|:---|:---|:---|:---|
| **CIFAR-10** | Random Crop 32×32 (pad 4), Random H-Flip (0.5) | Resize 32×32 | `mean=[0.4919, 0.4822, 0.4465]`, `std=[0.2023, 0.1994, 0.2010]` |
| **CIFAR-100** | Random Crop 32×32 (pad 4), Random H-Flip (0.5) | Resize 32×32 | `mean=[0.5071, 0.4865, 0.4409]`, `std=[0.2673, 0.2564, 0.2762]` |
| **MNIST** | Grayscale→3ch, Resize 32×32, Random H-Flip (0.5) | Grayscale→3ch, Resize 32×32 | `mean=[0.1307, 0.1307, 0.1307]`, `std=[0.3081, 0.3081, 0.3081]` |
| **FashionMNIST** | Grayscale→3ch, Resize 32×32, Random H-Flip (0.5) | Grayscale→3ch, Resize 32×32 | `mean=[0.2860, 0.2860, 0.2860]`, `std=[0.3560, 0.3560, 0.3560]` |
| **UTKFace** | Resize 224×224 | Resize 224×224 | `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` |

---

## 5. Roles of Each Data Subset in the Experiment Pipeline

### A. $D_R$ — Retain Set (36,000 samples)
* **Original Model Training:** Part of $D = D_R \cup D_F$.
* **Oracle Training:** The **sole training dataset** for the oracle reference $\theta_{\text{oracle}}$.
* **Unlearning Execution:** Training set for retain utility preservation across all unlearning algorithms.
* **CMF Classifier Construction:** Together with $D_F$, forms $D = D_R \cup D_F$ (`cmf_loader`, eval transforms) from which the Class-Mean Feature (CMF) head is rebuilt at the end of every unlearning epoch in 04a.
* **Evaluation:**
  - `output_retain`: Retention accuracy on the retain training set.
  - `lp_retain`, `ncc_retain`: In-sample representation diagnostic.
  - Linear Probe & NCC training: $D_R$ representations are used **exclusively** to train the probe and compute class centroids.

### B. $D_F$ — Forget Set (4,000 samples)
* **Original Model Training:** Part of $D = D_R \cup D_F$ (must be in original training to be unlearned).
* **Oracle Training:** **Strictly excluded** from oracle training.
* **Unlearning Execution:** Target data passed to unlearning algorithms (e.g. for random labeling, gradient ascent, KL divergence maximization, or error-maximizing noise).
* **Evaluation:**
  - `output_forget`: Accuracy on forgotten training samples.
  - `lp_forget`, `ncc_forget`: Out-of-sample latent representation unlearning metric.
  - **U-MIA Member Population:** Evaluated as the member population against non-members $D_V$ (U-MIA) and $D_T$ (T-MIA).

### C. $D_V$ — Validation Set (10,000 samples)
* **Original & Oracle Training:** Used strictly for checkpoint selection and early stopping.
* **Unlearning Execution:** Never used for gradient updates.
* **Evaluation:**
  - **U-MIA Non-Member Population:** Used as the unseen reference population when computing Membership Inference Attack (U-MIA) privacy metrics:
    $$\boxed{\text{MIA: } D_F \text{ (members)} \text{ vs. } D_V \text{ (non-members)}}$$
    Per-sample cross-entropy losses of both populations are shuffled and trimmed to equal size; a logistic regression is evaluated with 10-fold stratified CV and MIA = mean CV accuracy.
    $$\text{Disc} = |2 \times \text{MIA} - 1|, \quad \text{Indisc} = 1 - \text{Disc}$$

### D. $D_T$ — Official Test Set (10,000 samples)
* **Training & Unlearning:** **Never used** in training, checkpoint selection, or unlearning.
* **Evaluation:**
  - `output_test`: Final out-of-sample generalization accuracy.
  - `lp_test`, `ncc_test`: Out-of-sample representation retention.
  - **T-MIA Non-Member Population:** In addition to the $D_V$-based U-MIA, $D_F$ vs. $D_T$ is reported as T-MIA (`mia_test`, `indisc_test`), matching the paper's appendix tables. $D_T$ never replaces $D_V$ for U-MIA.

---

## 6. Correct Evaluation Metric Specification

| Evaluation Category | Metric Name | Data Split | Semantic Purpose |
|---|---|---|---|
| **Accuracy / Utility** | `output_retain` | $D_R$ | Retained utility on training retain partition |
| | `output_forget` | $D_F$ | Output-level forget behavior on unlearned training samples |
| | `output_test` | $D_T$ | Final out-of-sample generalization accuracy |
| **Representation Diagnostic** | `lp_retain` | $D_R$ | In-sample linear probe accuracy (probe fitted on $D_R$) |
| | `lp_forget` | $D_F$ | Out-of-sample representation-level unlearning efficacy |
| | `lp_test` | $D_T$ | Out-of-sample representation retention on unseen test set |
| | `ncc_retain` | $D_R$ | In-sample Nearest Class Center accuracy (centroids from $D_R$) |
| | `ncc_forget` | $D_F$ | Out-of-sample NCC forget accuracy |
| | `ncc_test` | $D_T$ | Out-of-sample NCC test accuracy |
| | `illusion_gap_lp` | $D_F$ | $\max(\text{lp\_forget} - \text{output\_forget}, 0)$ |
| | `illusion_gap_ncc` | $D_F$ | $\max(\text{ncc\_forget} - \text{output\_forget}, 0)$ |
| **Privacy / U-MIA** | `mia_forget` | $D_F \text{ vs. } D_V$ | U-MIA: mean 10-fold CV accuracy of a logistic regression on per-sample losses ($D_F$ members vs. $D_V$ non-members, equal-size) |
| | `disc_forget` | $D_F \text{ vs. } D_V$ | Discernibility: $\|2 \times \text{mia\_forget} - 1\| \in [0, 1]$ |
| | `indisc_forget` | $D_F \text{ vs. } D_V$ | Indiscernibility: $1 - \text{disc\_forget} \in [0, 1]$ ($1.0$ = ideal privacy) |
| | `mia_auc_forget` | $D_F \text{ vs. } D_V$ | Supplementary loss-based AUC-ROC (not used for Disc/Indisc) |
| **Privacy / T-MIA** | `mia_test` | $D_F \text{ vs. } D_T$ | Same attack with $D_T$ as non-members (requires `test_eval_loader`) |
| | `disc_test`, `indisc_test` | $D_F \text{ vs. } D_T$ | Disc / Indisc from `mia_test` |
| | `mia_auc_test` | $D_F \text{ vs. } D_T$ | Supplementary AUC-ROC |

---

## 7. Python Implementation Reference

```python
import os
import json
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

SPLIT_SEED = 123

# Function call for three-tier evaluation across notebooks:
metrics = evaluate_three_tier_metrics(
    model=model,
    probe_train_loader=train_r_eval_loader,   # D_R: fits Linear Probe & NCC centroids
    retain_eval_loader=train_r_eval_loader,   # D_R: measures output_retain, lp_retain, ncc_retain
    forget_eval_loader=train_f_eval_loader,   # D_F: measures output_forget, lp_forget, ncc_forget
    num_classes=10,
    seed=0,
    device=device,
    test_eval_loader=test_loader,             # D_T: measures output_test, lp_test, ncc_test
    nonmember_loader=val_loader               # D_V: non-member population for U-MIA
)
```

---

## 8. Protocol Consistency Checklist

1. [x] Original model $\theta_O$ is trained on all samples in $D = D_R \cup D_F$.
2. [x] Original model $\theta_O$ has **never** trained on $D_V$ or $D_T$.
3. [x] Oracle model $\theta_{\text{oracle}}$ is trained from scratch **strictly on $D_R$**.
4. [x] Unlearning algorithms receive $\theta_O$, $D_R$, and $D_F$.
5. [x] Forget metrics are evaluated on $D_F$.
6. [x] U-MIA strictly compares $D_F$ (members) against $D_V$ (non-members); T-MIA additionally compares $D_F$ against $D_T$.
7. [x] Generalization metrics are strictly evaluated on $D_T$.
8. [x] $D_T$ is never treated as the forget set.
9. [x] $D_V$ is never added to $D_F$.
10. [x] In Static CMF (04a), the CMF head is frozen w.r.t. gradients and rebuilt from $D = D_R \cup D_F$ at the end of every unlearning epoch (`epoch_end_callback`); no post-unlearning reset.
11. [x] In Post-Hoc Alignment (04b), encoder is frozen and classifier head is fine-tuned on $D_R$ only.
12. [x] In Adaptive Refinement (04c), 04b-aligned head is frozen and encoder is refined on $D_R$ and $D_F$.
