# Model & Training Configuration Guide

This document outlines the standard model architectures, hyperparameters, initialization seeds, and unlearning configurations matching the experimental benchmark protocol.

---

## 1. Benchmark Training Configurations

| Dataset | Unlearner Modes | Model Architecture | Epochs | Batch Size | Learning Rate | Optimizer | Model Seeds |
|:---|:---|:---|:---:|:---:|:---:|:---|:---|
| **MNIST** | `original`, `naive` (oracle) | `resnet18`, `vit11m` | 50 | 256 | 0.1 | SGD (CosineAnnealing, Momentum=0.9, Weight Decay=5e-4) | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` |
| **FashionMNIST** | `original`, `naive` (oracle) | `resnet18`, `vit11m` | 50 | 256 | 0.1 | SGD (CosineAnnealing, Momentum=0.9, Weight Decay=5e-4) | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` |
| **CIFAR-10** | `original`, `naive` (oracle) | `resnet18`, `vit11m` | 182 | 256 | 0.1 | SGD (CosineAnnealing, Momentum=0.9, Weight Decay=5e-4) | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` |
| **CIFAR-100** | `original`, `naive` (oracle) | `resnet18`, `vit11m` | 182 | 256 | 0.1 | SGD (CosineAnnealing, Momentum=0.9, Weight Decay=5e-4) | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` |
| **UTKFace** | `original`, `naive` (oracle) | `resnet18`, `vit11m` | 50 | 256 | 0.1 | SGD (CosineAnnealing, Momentum=0.9, Weight Decay=5e-4) | `[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]` |

---

## 2. YAML Specification

```yaml
mnist:
  unlearner:
    - original
    - naive
  unlearner.cfg.num_epochs: 50
  unlearner.cfg.batch_size: 256
  unlearner.cfg.optimizer.learning_rate: 0.1
  model:
    - resnet18
    - vit11m
  model_seed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

fashion_mnist:
  unlearner:
    - original
    - naive
  unlearner.cfg.num_epochs: 50
  unlearner.cfg.batch_size: 256
  unlearner.cfg.optimizer.learning_rate: 0.1
  model:
    - resnet18
    - vit11m
  model_seed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

cifar10:
  unlearner:
    - original
    - naive
  unlearner.cfg.num_epochs: 182
  unlearner.cfg.batch_size: 256
  unlearner.cfg.optimizer.learning_rate: 0.1
  model:
    - resnet18
    - vit11m
  model_seed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

cifar100:
  unlearner:
    - original
    - naive
  unlearner.cfg.num_epochs: 182
  unlearner.cfg.batch_size: 256
  unlearner.cfg.optimizer.learning_rate: 0.1
  model:
    - resnet18
    - vit11m
  model_seed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

utkface:
  unlearner:
    - original
    - naive
  unlearner.cfg.num_epochs: 50
  unlearner.cfg.batch_size: 256
  unlearner.cfg.optimizer.learning_rate: 0.1
  model:
    - resnet18
    - vit11m
  model_seed: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

---

## 3. Python Configuration Dictionary

```python
MODEL_CONFIGS = {
    "mnist": {
        "unlearners": ["original", "naive"],
        "num_epochs": 50,
        "batch_size": 256,
        "lr": 0.1,
        "models": ["resnet18", "vit11m"],
        "model_seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    },
    "fashion_mnist": {
        "unlearners": ["original", "naive"],
        "num_epochs": 50,
        "batch_size": 256,
        "lr": 0.1,
        "models": ["resnet18", "vit11m"],
        "model_seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    },
    "cifar10": {
        "unlearners": ["original", "naive"],
        "num_epochs": 182,
        "batch_size": 256,
        "lr": 0.1,
        "models": ["resnet18", "vit11m"],
        "model_seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    },
    "cifar100": {
        "unlearners": ["original", "naive"],
        "num_epochs": 182,
        "batch_size": 256,
        "lr": 0.1,
        "models": ["resnet18", "vit11m"],
        "model_seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    },
    "utkface": {
        "unlearners": ["original", "naive"],
        "num_epochs": 50,
        "batch_size": 256,
        "lr": 0.1,
        "models": ["resnet18", "vit11m"],
        "model_seeds": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    },
}
```

> **Note:** Deep Unlearn Table 7 uses batch size **128** for UTKFace; this repo currently uses 256 for all datasets.

---

## 4. Unlearning Method Hyperparameters (defaults in `unlearning_methods/`)

All methods start from the original checkpoint $\theta_O$, receive $D_R$ and $D_F$, and run a fixed epoch budget (no hyperparameter search is implemented yet; the paper tunes with Optuna, 3 × 100 trials).

| Method | File | Optimizer | Key hyperparameters |
|---|---|---|---|
| Random Label (paper: SRL) | `random_label.py` | SGD, momentum 0.9, wd 5e-4 | `lr=0.013`, `epochs=10` |
| SalUn | `salun.py` | SGD, momentum 0.9, wd 5e-4 | `lr=0.013`, `epochs=10`, saliency `threshold=0.5` |
| NegGrad+ | `neggrad_plus.py` | SGD, momentum 0.9, wd 5e-4 | `alpha` (β) `=0.95`, `lr=0.01`, `epochs=10` |
| SCRUB | `scrub.py` | Adam, wd 0.1 | `lr=5e-4`, `epochs=10`, `msteps=3`, `kd_T=2`, `gamma=1`, `alpha=0.5`, retain batch 32, forget batch 64, lr decay ×0.1 at `[0.4E, 0.7E, 0.9E] (= epoch 4, 7, 9)` |
| UNSIR | `tarun_unsir.py` | Adam (noise, impair, repair) | noise: `lr=0.1`, 5 epochs × 8 steps; impair: `lr=5e-4`, 1 epoch, 20 batches; repair: `lr=0.01`, 1 epoch |

Every method accepts `epoch_end_callback(model)`, used by 04a to rebuild the CMF head from $D = D_R \cup D_F$ after each epoch.
