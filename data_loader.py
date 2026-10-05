"""
Dataset preparation, transformations, and deterministic splitting for Machine Unlearning.
Complies with "Deep Unlearn: Benchmarking Machine Unlearning for Image Classification" (Cadet et al., arXiv:2410.01276v2)
and data_instruction.md.
"""

from typing import Dict, List, Optional, Tuple, Union
import copy
import os
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


SPLIT_SEED = 123

DATASET_CONFIGS = {
    "cifar10": {
        "num_classes": 10,
        "input_size": 32,
        "in_channels": 3,
        "dev_size": 50000,
        "test_size": 10000,
        "train_size": 42500,  # 85% of dev
        "val_size": 7500,     # 15% of dev
        "retain_size": 38250, # 90% of train
        "forget_size": 4250,  # 10% of train
        "mean": [0.4919, 0.4822, 0.4465],
        "std": [0.2023, 0.1994, 0.2010],
    },
    "cifar100": {
        "num_classes": 100,
        "input_size": 32,
        "in_channels": 3,
        "dev_size": 50000,
        "test_size": 10000,
        "train_size": 42500,  # 85% of dev
        "val_size": 7500,     # 15% of dev
        "retain_size": 38250, # 90% of train
        "forget_size": 4250,  # 10% of train
        "mean": [0.5071, 0.4865, 0.4409],
        "std": [0.2673, 0.2564, 0.2762],
    },
    "mnist": {
        "num_classes": 10,
        "input_size": 32,
        "in_channels": 3,
        "dev_size": 60000,
        "test_size": 10000,
        "train_size": 51000,  # 85% of dev
        "val_size": 9000,     # 15% of dev
        "retain_size": 45900, # 90% of train
        "forget_size": 5100,  # 10% of train
        "mean": [0.1307, 0.1307, 0.1307],
        "std": [0.3081, 0.3081, 0.3081],
    },
    "fashion_mnist": {
        "num_classes": 10,
        "input_size": 32,
        "in_channels": 3,
        "dev_size": 60000,
        "test_size": 10000,
        "train_size": 51000,  # 85% of dev
        "val_size": 9000,     # 15% of dev
        "retain_size": 45900, # 90% of train
        "forget_size": 5100,  # 10% of train
        "mean": [0.2860, 0.2860, 0.2860],
        "std": [0.3560, 0.3560, 0.3560],
    },
    "utkface": {
        "num_classes": 5,
        "input_size": 224,
        "in_channels": 3,
        "total_size": 23708,
        "dev_size": 18967,
        "test_size": 4741,
        "train_size": 16121,
        "val_size": 2846,
        "retain_size": 14508,
        "forget_size": 1613,
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
    },
}


class TransformSubset(Dataset):
    """
    Dataset wrapper applying transform dynamically on indexed subset samples.
    """
    def __init__(self, subset_or_dataset: Union[Dataset, Subset], transform=None):
        self.dataset = subset_or_dataset
        self.transform = transform

    def __getitem__(self, idx: int):
        x, y = self.dataset[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self) -> int:
        return len(self.dataset)


def get_transforms(dataset_name: str) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Builds training (with augmentation) and evaluation transforms.
    Grayscale images (MNIST, FashionMNIST) are expanded to 3 channels and resized to 32x32.
    """
    dataset_name = dataset_name.lower().replace("-", "_")
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Supported: {list(DATASET_CONFIGS.keys())}")

    cfg = DATASET_CONFIGS[dataset_name]
    mean, std = cfg["mean"], cfg["std"]

    if dataset_name in ["cifar10", "cifar100"]:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        eval_transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    elif dataset_name in ["mnist", "fashion_mnist"]:
        train_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        eval_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    elif dataset_name == "utkface":
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        eval_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])

    return train_transform, eval_transform


def load_raw_dataset(dataset_name: str, root: str = "./data") -> Tuple[Dataset, Optional[Dataset]]:
    """
    Downloads and loads raw torchvision datasets without predefined transform.
    """
    dataset_name = dataset_name.lower().replace("-", "_")
    os.makedirs(root, exist_ok=True)

    if dataset_name == "cifar10":
        raw_dev = datasets.CIFAR10(root=root, train=True, download=True)
        raw_test = datasets.CIFAR10(root=root, train=False, download=True)
    elif dataset_name == "cifar100":
        raw_dev = datasets.CIFAR100(root=root, train=True, download=True)
        raw_test = datasets.CIFAR100(root=root, train=False, download=True)
    elif dataset_name == "mnist":
        raw_dev = datasets.MNIST(root=root, train=True, download=True)
        raw_test = datasets.MNIST(root=root, train=False, download=True)
    elif dataset_name == "fashion_mnist":
        raw_dev = datasets.FashionMNIST(root=root, train=True, download=True)
        raw_test = datasets.FashionMNIST(root=root, train=False, download=True)
    elif dataset_name == "utkface":
        # Expect standard ImageFolder directory structure or root if already prepared
        utk_dir = os.path.join(root, "utkface")
        if not os.path.exists(utk_dir):
            raise FileNotFoundError(f"UTKFace directory not found at '{utk_dir}'. Please extract UTKFace dataset there.")
        raw_full = datasets.ImageFolder(root=utk_dir)
        return raw_full, None
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    return raw_dev, raw_test


def build_unlearning_splits(
    dataset_name: str,
    root: str = "./data",
    seed: int = SPLIT_SEED,
) -> Dict[str, Dataset]:
    """
    Generates deterministic partitions (Train, Retain, Forget, Validation, Test)
    according to Table 2 of arXiv:2410.01276v2 and data_instruction.md.
    """
    dataset_name = dataset_name.lower().replace("-", "_")
    cfg = DATASET_CONFIGS[dataset_name]
    train_transform, eval_transform = get_transforms(dataset_name)

    raw_dev, raw_test = load_raw_dataset(dataset_name, root=root)
    generator = torch.Generator().manual_seed(seed)

    if raw_test is None:
        # Handle unpartitioned datasets like UTKFace (80% dev, 20% test)
        total_len = len(raw_dev)
        total_perm = torch.randperm(total_len, generator=generator).tolist()
        dev_len = cfg["dev_size"]
        dev_indices = total_perm[:dev_len]
        test_indices = total_perm[dev_len:]

        dev_dataset = Subset(raw_dev, dev_indices)
        test_dataset = TransformSubset(Subset(raw_dev, test_indices), eval_transform)
        dev_len_actual = len(dev_indices)
    else:
        dev_dataset = raw_dev
        test_dataset = TransformSubset(raw_test, eval_transform)
        dev_len_actual = len(raw_dev)

    # 1. Split Dev -> Train (D) and Validation (D_V)
    dev_perm = torch.randperm(dev_len_actual, generator=generator).tolist()
    train_indices = dev_perm[:cfg["train_size"]]
    val_indices = dev_perm[cfg["train_size"]:cfg["train_size"] + cfg["val_size"]]

    # 2. Split Train (D) -> Retain (D_R) [90%] and Forget (D_F) [10%]
    train_perm = torch.randperm(len(train_indices), generator=generator).tolist()
    retain_indices = [train_indices[i] for i in train_perm[:cfg["retain_size"]]]
    forget_indices = [train_indices[i] for i in train_perm[cfg["retain_size"]:]]

    # 3. Wrap with transforms
    full_train_set = TransformSubset(Subset(dev_dataset, train_indices), train_transform)
    retain_train_set = TransformSubset(Subset(dev_dataset, retain_indices), train_transform)
    retain_eval_set = TransformSubset(Subset(dev_dataset, retain_indices), eval_transform)
    forget_eval_set = TransformSubset(Subset(dev_dataset, forget_indices), eval_transform)
    val_set = TransformSubset(Subset(dev_dataset, val_indices), eval_transform)

    # Validate exact disjoint partition
    assert len(retain_indices) == cfg["retain_size"], f"Retain size mismatch: {len(retain_indices)} vs {cfg['retain_size']}"
    assert len(forget_indices) == cfg["forget_size"], f"Forget size mismatch: {len(forget_indices)} vs {cfg['forget_size']}"
    assert len(val_indices) == cfg["val_size"], f"Val size mismatch: {len(val_indices)} vs {cfg['val_size']}"
    assert len(test_dataset) == cfg["test_size"], f"Test size mismatch: {len(test_dataset)} vs {cfg['test_size']}"

    return {
        "train": full_train_set,
        "retain": retain_train_set,
        "retain_eval": retain_eval_set,
        "forget": forget_eval_set,
        "validation": val_set,
        "test": test_dataset,
    }


def get_unlearning_dataloaders(
    dataset_name: str,
    root: str = "./data",
    batch_size: int = 128,
    num_workers: int = 2,
    pin_memory: bool = True,
    seed: int = SPLIT_SEED,
) -> Dict[str, DataLoader]:
    """
    Creates ready-to-use PyTorch DataLoaders for all unlearning experiment splits.
    """
    splits = build_unlearning_splits(dataset_name=dataset_name, root=root, seed=seed)

    loaders = {
        "train": DataLoader(
            splits["train"], batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory
        ),
        "retain": DataLoader(
            splits["retain"], batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=pin_memory
        ),
        "retain_eval": DataLoader(
            splits["retain_eval"], batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        ),
        "forget": DataLoader(
            splits["forget"], batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        ),
        "validation": DataLoader(
            splits["validation"], batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        ),
        "test": DataLoader(
            splits["test"], batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=pin_memory
        ),
    }

    return loaders
