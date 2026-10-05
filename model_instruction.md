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
