# Machine Unlearning Dataset Preparation and Splitting Instructions

This document provides exact guidelines and instructions for partitioning datasets into **Train ($D$)**, **Validation ($D_V$)**, **Test ($D_T$)**, **Retain ($D_R$)**, and **Forget ($D_F$)** splits to replicate the experimental benchmark in **"Deep Unlearn: Benchmarking Machine Unlearning for Image Classification"** (Cadet et al., arXiv:2410.01276v2).

---

## 1. Split Definitions & Hierarchy

```
                                  Full Dataset
                                       │
                ┌──────────────────────┴──────────────────────┐
                ▼                                             ▼
     Development Set ($D_{\text{dev}}$)                 Test Set ($D_T$)
 (Predefined train split / 80% full)             (Predefined test split / 20% full)
                │                                             │
      ┌─────────┴─────────┐                                   │
      ▼                   ▼                                   │
Training Set ($D$)  Validation Set ($D_V$)                    │
   (80% / 85% dev)     (20% / 15% dev)                        │
      │                   │                                   │
  ┌───┴───┐               │                                   │
  ▼       ▼               ▼                                   ▼
Retain  Forget       Validation                             Test
($D_R$) ($D_F$)        ($D_V$)                             ($D_T$)
 (90%)   (10%)
```

### Hierarchy Breakdown:
1. **Development ($D_{\text{dev}}$) vs. Test ($D_T$)**:
   - For datasets with predefined standard splits (**MNIST, FashionMNIST, CIFAR-10, CIFAR-100**): Keep the official test split as $D_T$ and the official training split as $D_{\text{dev}}$.
   - For unpartitioned datasets (**UTKFace**): Partition into 80% Development ($D_{\text{dev}}$) and 20% Test ($D_T$).
2. **Training ($D$) vs. Validation ($D_V$)**:
   - $D_{\text{dev}}$ is divided into Training set $D$ and Validation set $D_V$.
   - $D$ is used to train the original baseline model $f_O$.
   - $D_V$ is held out for hyperparameter optimization and Membership Inference Attack (U-MIA) evaluation.
3. **Retain ($D_R$) vs. Forget ($D_F$)**:
   - Training set $D$ is uniformly split into **90% Retain ($D_R$)** and **10% Forget ($D_F$)** (10% unlearning budget).
   - $D_R$ is used to train the retrained reference model $f_R$ and execute unlearning methods.
   - $D_F$ is the target data to be unlearned.

---

## 2. Dataset Split Sizes (Matching Benchmark Table 2)

| Dataset | Total Size | Retain ($D_R$) | Forget ($D_F$) | Validation ($D_V$) | Test ($D_T$) | Classes | Task Type | Input Size |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|
| **MNIST** | 70,000 | 45,900 | 5,100 | 9,000 | 10,000 | 10 | Handwritten Digits | 32×32 (3-ch) |
| **FashionMNIST** | 70,000 | 45,900 | 5,100 | 9,000 | 10,000 | 10 | Fashion Items | 32×32 (3-ch) |
| **CIFAR-10** | 60,000 | 38,250 | 4,250 | 7,500 | 10,000 | 10 | Natural Objects | 32×32 (3-ch) |
| **CIFAR-100** | 60,000 | 38,250 | 4,250 | 7,500 | 10,000 | 100 | Fine-grained Objects | 32×32 (3-ch) |
| **UTKFace** | 23,708 | 14,508 | 1,613 | 2,846 | 4,741 | 5 | Age Classification | 224×224 (3-ch) |

---

## 3. Data Preprocessing & Transformations

| Dataset | Train Data Augmentations | Test / Eval Transformations | Normalization (Mean / Std) |
|:---|:---|:---|:---|
| **CIFAR-10** | Random Crop 32×32 (pad 4), Random H-Flip (0.5) | Resize 32×32 | `mean=[0.4919, 0.4822, 0.4465]`, `std=[0.2023, 0.1994, 0.2010]` |
| **CIFAR-100** | Random Crop 32×32 (pad 4), Random H-Flip (0.5) | Resize 32×32 | `mean=[0.5071, 0.4865, 0.4409]`, `std=[0.2673, 0.2564, 0.2762]` |
| **MNIST** | Grayscale→3ch, Resize 32×32, Random H-Flip (0.5) | Grayscale→3ch, Resize 32×32 | `mean=[0.1307, 0.1307, 0.1307]`, `std=[0.3081, 0.3081, 0.3081]` |
| **FashionMNIST** | Grayscale→3ch, Resize 32×32, Random H-Flip (0.5) | Grayscale→3ch, Resize 32×32 | `mean=[0.2860, 0.2860, 0.2860]`, `std=[0.3560, 0.3560, 0.3560]` |
| **UTKFace** | Resize 224×224 | Resize 224×224 | `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]` |

---

## 4. Implementation Specification

Use fixed random seed `seed = 123` for deterministic dataset splitting.

```python
import torch
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

SPLIT_SEED = 123

DATASET_CONFIGS = {
    "cifar10": {
        "train_size": 42500,
        "val_size": 7500,
        "retain_size": 38250,
        "forget_size": 4250,
        "mean": [0.4919, 0.4822, 0.4465],
        "std": [0.2023, 0.1994, 0.2010],
    },
    "cifar100": {
        "train_size": 42500,
        "val_size": 7500,
        "retain_size": 38250,
        "forget_size": 4250,
        "mean": [0.5071, 0.4865, 0.4409],
        "std": [0.2673, 0.2564, 0.2762],
    },
    "mnist": {
        "train_size": 51000,
        "val_size": 9000,
        "retain_size": 45900,
        "forget_size": 5100,
        "mean": [0.1307, 0.1307, 0.1307],
        "std": [0.3081, 0.3081, 0.3081],
    },
    "fashion_mnist": {
        "train_size": 51000,
        "val_size": 9000,
        "retain_size": 45900,
        "forget_size": 5100,
        "mean": [0.2860, 0.2860, 0.2860],
        "std": [0.3560, 0.3560, 0.3560],
    }
}

class TransformSubset(Dataset):
    """Subset wrapper that applies a specific transform."""
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        x, y = self.subset[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)


def build_cifar10_splits(root="./data"):
    cfg = DATASET_CONFIGS["cifar10"]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"])
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=cfg["mean"], std=cfg["std"])
    ])

    raw_dev = datasets.CIFAR10(root=root, train=True, download=True)
    raw_test = datasets.CIFAR10(root=root, train=False, download=True)

    generator = torch.Generator().manual_seed(SPLIT_SEED)
    dev_perm = torch.randperm(len(raw_dev), generator=generator).tolist()

    train_indices = dev_perm[:cfg["train_size"]]
    val_indices = dev_perm[cfg["train_size"]:cfg["train_size"] + cfg["val_size"]]

    train_perm = torch.randperm(len(train_indices), generator=generator).tolist()
    retain_indices = [train_indices[i] for i in train_perm[:cfg["retain_size"]]]
    forget_indices = [train_indices[i] for i in train_perm[cfg["retain_size"]:]]

    retain_set = TransformSubset(Subset(raw_dev, retain_indices), train_transform)
    forget_set = TransformSubset(Subset(raw_dev, forget_indices), eval_transform)
    val_set = TransformSubset(Subset(raw_dev, val_indices), eval_transform)
    test_set = TransformSubset(raw_test, eval_transform)

    return {
        "retain": retain_set,
        "forget": forget_set,
        "validation": val_set,
        "test": test_set
    }
```

---

## 5. Usage in the Machine Unlearning Lifecycle

1. **Original Model Training ($f_O$):**
   - Trained on the full **Training Set ($D = D_R \cup D_F$)**.
2. **Retrained Reference Model ($f_R$):**
   - Trained from scratch on the **Retain Set ($D_R$)** only.
3. **Machine Unlearning Execution ($f_U$):**
   - An unlearning method receives the pre-trained weights $\theta_O$, Retain set $D_R$, Forget set $D_F$, and Validation set $D_V$.
4. **Evaluation Metrics:**
   - **Retain Accuracy (RA):** Classification accuracy on $D_R$.
   - **Forget Accuracy (FA):** Classification accuracy on $D_F$.
   - **Test Accuracy (TA):** Generalization accuracy on $D_T$.
   - **Indiscernibility / U-MIA:** Evaluated by testing whether a membership classifier can distinguish sample loss distributions between $D_F$ (members to unlearn) and $D_V$ (unseen validation non-members).
