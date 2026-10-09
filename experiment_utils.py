"""
Shared core utilities for Representation-Level Class Unlearning Experiment:
- ResNet-18 Backbone & Split Architectures (Encoder + Linear / CMF Head)
- Class-wise and Whole-Class Retain/Forget Split partitioning and persistence
- Three-tier Evaluation Metrics: Output Accuracy, Linear Probe (LP), Nearest Class Center (NCC)
- Illusion Gap Computation: max(LP_f - Output_f, 0) and max(NCC_f - Output_f, 0)
"""

from typing import Dict, List, Optional, Tuple, Union
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from torchvision.models import resnet18, ResNet18_Weights
from sklearn.linear_model import LogisticRegression
import numpy as np


# ==============================================================================
# 1. Modular Model Architecture (Encoder + Classifier Head)
# ==============================================================================

class ResNet18Backbone(nn.Module):
    """
    Standard ResNet-18 feature encoder returning a d-dimensional representation (512-dim).
    """
    def __init__(self, pretrained: bool = False, in_channels: int = 3):
        super().__init__()
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        base_resnet = resnet18(weights=weights)
        
        if in_channels != 3:
            base_resnet.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )

        self.conv1 = base_resnet.conv1
        self.bn1 = base_resnet.bn1
        self.relu = base_resnet.relu
        self.maxpool = base_resnet.maxpool
        self.layer1 = base_resnet.layer1
        self.layer2 = base_resnet.layer2
        self.layer3 = base_resnet.layer3
        self.layer4 = base_resnet.layer4
        self.avgpool = base_resnet.avgpool
        self.feature_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        return features


class ClassifierHead(nn.Module):
    """Linear classifier head W * h + b."""
    def __init__(self, in_features: int = 512, num_classes: int = 10, bias: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, num_classes, bias=bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear(features)


class FullClassifier(nn.Module):
    """End-to-end model comprising encoder + classifier head."""
    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(x)
        return self.head(feats)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class ViT11MBackbone(nn.Module):
    """
    Vision Transformer (ViT-11M / ViT-Tiny variant for 32x32 / small image benchmarks).
    Image size: 32x32 (or resized), Patch size: 4x4, Embed dim: 384, Depth: 6 layers, Heads: 6.
    Output feature dimension: 384.
    """
    def __init__(self, in_channels: int = 3, img_size: int = 32, patch_size: int = 4, embed_dim: int = 384, depth: int = 6, num_heads: int = 6):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.feature_dim = embed_dim

        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=0.0)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=embed_dim * 4,
            dropout=0.0, activation="gelu", batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.transformer(x)
        x = self.norm(x)
        return x[:, 0]  # CLS token feature


def build_resnet18_classifier(num_classes: int = 10, in_channels: int = 3, pretrained: bool = False) -> FullClassifier:
    encoder = ResNet18Backbone(pretrained=pretrained, in_channels=in_channels)
    head = ClassifierHead(in_features=encoder.feature_dim, num_classes=num_classes)
    return FullClassifier(encoder, head)


def build_vit11m_classifier(num_classes: int = 10, in_channels: int = 3, img_size: int = 32) -> FullClassifier:
    encoder = ViT11MBackbone(in_channels=in_channels, img_size=img_size)
    head = ClassifierHead(in_features=encoder.feature_dim, num_classes=num_classes)
    return FullClassifier(encoder, head)


def build_classifier(model_name: str = "resnet18", num_classes: int = 10, in_channels: int = 3, img_size: int = 32) -> FullClassifier:
    model_name = model_name.lower().replace("-", "").replace("_", "")
    if model_name in ["resnet18", "resnet"]:
        return build_resnet18_classifier(num_classes=num_classes, in_channels=in_channels)
    elif model_name in ["vit11m", "vit"]:
        return build_vit11m_classifier(num_classes=num_classes, in_channels=in_channels, img_size=img_size)
    else:
        raise ValueError(f"Unsupported model architecture: {model_name}")


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
        "model_seeds": [0, 1, 2],
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


# ==============================================================================
# 2. Dataset Partitioning & Whole-Class Retain/Forget Split
# ==============================================================================

class IndexedDataset(Dataset):
    """Wrapper that applies transform dynamically."""
    def __init__(self, dataset: Dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx: int):
        x, y = self.dataset[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self) -> int:
        return len(self.dataset)


def get_dataset_transforms(dataset_name: str = "cifar10"):
    dataset_name = dataset_name.lower().replace("-", "_")
    if dataset_name == "cifar10":
        mean, std = [0.4919, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        eval_tf = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    elif dataset_name == "cifar100":
        mean, std = [0.5071, 0.4865, 0.4409], [0.2673, 0.2564, 0.2762]
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        eval_tf = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    elif dataset_name in ["mnist", "fashion_mnist"]:
        mean, std = ([0.1307, 0.1307, 0.1307], [0.3081, 0.3081, 0.3081]) if dataset_name == "mnist" else ([0.2860, 0.2860, 0.2860], [0.3560, 0.3560, 0.3560])
        train_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
        eval_tf = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
        train_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
        eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=mean, std=std)])
    return train_tf, eval_tf


def create_and_persist_splits(
    dataset_name: str = "cifar10",
    root: str = "./data",
    seed: int = 123,
    save_dir: str = "./artifacts/splits",
) -> Dict[str, Union[List[int], str, int]]:
    """
    Creates and persists deterministic dataset split exactly according to data_instruction.md:
    D_dev -> D (85%, 42500) and D_V (15%, 7500)
    D -> D_R (90%, 38250) and D_F (10%, 4250)
    D_T (10000)
    """
    os.makedirs(save_dir, exist_ok=True)
    split_filename = os.path.join(save_dir, f"{dataset_name}_seed_{seed}_split.json")

    if os.path.exists(split_filename):
        with open(split_filename, "r") as f:
            split_artifact = json.load(f)
        # Validate loaded split integrity and disjointness
        train_idx = set(split_artifact["train_indices"])
        val_idx = set(split_artifact["val_indices"])
        train_r_idx = set(split_artifact["train_r_indices"])
        train_f_idx = set(split_artifact["train_f_indices"])
        assert len(train_idx.intersection(val_idx)) == 0, "Loaded split: train and val overlap"
        assert len(train_r_idx.intersection(train_f_idx)) == 0, "Loaded split: retain and forget overlap"
        assert train_r_idx.union(train_f_idx) == train_idx, "Loaded split: retain union forget != train"
        return split_artifact

    if dataset_name.lower() == "cifar10":
        raw_train = datasets.CIFAR10(root=root, train=True, download=True)
        raw_test = datasets.CIFAR10(root=root, train=False, download=True)
        train_size, val_size = 42500, 7500
        retain_size, forget_size = 38250, 4250
    elif dataset_name.lower() == "cifar100":
        raw_train = datasets.CIFAR100(root=root, train=True, download=True)
        raw_test = datasets.CIFAR100(root=root, train=False, download=True)
        train_size, val_size = 42500, 7500
        retain_size, forget_size = 38250, 4250
    elif dataset_name.lower() == "mnist":
        raw_train = datasets.MNIST(root=root, train=True, download=True)
        raw_test = datasets.MNIST(root=root, train=False, download=True)
        train_size, val_size = 51000, 9000
        retain_size, forget_size = 45900, 5100
    elif dataset_name.lower() == "fashion_mnist":
        raw_train = datasets.FashionMNIST(root=root, train=True, download=True)
        raw_test = datasets.FashionMNIST(root=root, train=False, download=True)
        train_size, val_size = 51000, 9000
        retain_size, forget_size = 45900, 5100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # 1. Deterministic Dev -> Train (D) & Validation (D_V)
    gen = torch.Generator().manual_seed(seed)
    dev_perm = torch.randperm(len(raw_train), generator=gen).tolist()
    train_indices = dev_perm[:train_size]
    val_indices = dev_perm[train_size:train_size + val_size]
    test_indices = list(range(len(raw_test)))

    # 2. Deterministic Train (D) -> Retain (D_R) & Forget (D_F) (90% / 10%)
    train_perm = torch.randperm(len(train_indices), generator=gen).tolist()
    train_r_indices = [train_indices[i] for i in train_perm[:retain_size]]
    train_f_indices = [train_indices[i] for i in train_perm[retain_size:]]

    # Validate disjoint assertions
    assert len(set(train_indices).intersection(set(val_indices))) == 0
    assert len(train_indices) + len(val_indices) == len(raw_train)
    assert len(set(train_r_indices).intersection(set(train_f_indices))) == 0
    assert len(train_r_indices) + len(train_f_indices) == len(train_indices)
    assert len(train_r_indices) == retain_size
    assert len(train_f_indices) == forget_size

    split_artifact = {
        "dataset_name": dataset_name,
        "seed": seed,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "test_indices": test_indices,
        "train_r_indices": train_r_indices,
        "train_f_indices": train_f_indices,
    }

    with open(split_filename, "w") as f:
        json.dump(split_artifact, f)

    return split_artifact


# ==============================================================================
# 3. Three-Tier Evaluation: Output, Linear Probe (LP), Nearest Class Center (NCC)
# ==============================================================================

@torch.no_grad()
def extract_features(
    encoder: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extracts features with encoder in eval() mode and no_grad.
    """
    encoder.eval()
    all_feats = []
    all_targets = []

    for images, targets in dataloader:
        images = images.to(device)
        feats = encoder(images)
        all_feats.append(feats.cpu().numpy())
        all_targets.append(targets.numpy() if isinstance(targets, torch.Tensor) else np.array(targets))

    return np.concatenate(all_feats, axis=0), np.concatenate(all_targets, axis=0)


def compute_output_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = "cuda"
) -> float:
    """Computes output accuracy (argmax of logits) in [0, 1]."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            targets = targets.to(device) if isinstance(targets, torch.Tensor) else torch.tensor(targets, device=device)
            outputs = model(images)
            preds = outputs.argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)

    acc = correct / total if total > 0 else 0.0
    assert 0.0 <= acc <= 1.0, f"Accuracy out of range [0, 1]: {acc}"
    return float(acc)


def compute_linear_probe_accuracy(
    train_feats: np.ndarray,
    train_targets: np.ndarray,
    eval_feats: np.ndarray,
    eval_targets: np.ndarray,
    seed: int = 123
) -> float:
    """
    Trains a fresh deterministic linear probe (Logistic Regression) on frozen training features
    and evaluates accuracy on the target evaluation features.
    """
    if len(eval_feats) == 0:
        return 0.0
    clf = LogisticRegression(max_iter=1000, random_state=seed, solver="lbfgs")
    clf.fit(train_feats, train_targets)
    preds = np.array(clf.predict(eval_feats))
    eval_targets_arr = np.array(eval_targets)
    acc = float(np.mean(preds == eval_targets_arr))
    assert 0.0 <= acc <= 1.0, f"LP Accuracy out of range [0, 1]: {acc}"
    return acc


def compute_ncc_accuracy(
    train_feats: np.ndarray,
    train_targets: np.ndarray,
    eval_feats: np.ndarray,
    eval_targets: np.ndarray,
    num_classes: int = 10
) -> float:
    """
    Nearest Class Center (NCC): computes class means from training features
    and assigns evaluation samples to the nearest class mean.
    """
    if len(eval_feats) == 0:
        return 0.0

    class_means = []
    for c in range(num_classes):
        mask = (train_targets == c)
        if mask.sum() > 0:
            mean = train_feats[mask].mean(axis=0)
        else:
            mean = np.zeros(train_feats.shape[1])
        class_means.append(mean)

    class_means = np.stack(class_means, axis=0) # [C, D]

    # Compute Euclidean distances from eval_feats to each class mean
    dists = np.linalg.norm(eval_feats[:, None, :] - class_means[None, :, :], axis=2) # [N, C]
    preds = np.argmin(dists, axis=1)
    eval_targets_arr = np.array(eval_targets)
    acc = float(np.mean(preds == eval_targets_arr))
    assert 0.0 <= acc <= 1.0, f"NCC Accuracy out of range [0, 1]: {acc}"
    return acc


@torch.no_grad()
def compute_mia_metrics(
    model: nn.Module,
    forget_loader: DataLoader,
    nonmember_loader: DataLoader,
    device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluates Membership Inference Attack (MIA) and Indiscernibility metrics on forgotten samples (D_F)
    versus unseen non-members (D_V or D_T).
    
    Computes per-sample cross-entropy loss: s_i = -loss_i.
    Higher score indicates higher likelihood of training membership.
    
    Returns:
        mia_forget: AUC-ROC of separating D_F from non-members in [0, 1].
        disc_forget: Discernibility |2 * MIA - 1| in [0, 1].
        indisc_forget: Indiscernibility 1 - Disc in [0, 1].
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="none")

    forget_losses = []
    for images, targets in forget_loader:
        images = images.to(device)
        targets = targets.to(device) if isinstance(targets, torch.Tensor) else torch.tensor(targets, device=device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        forget_losses.extend(loss.cpu().numpy().tolist())

    nonmember_losses = []
    for images, targets in nonmember_loader:
        images = images.to(device)
        targets = targets.to(device) if isinstance(targets, torch.Tensor) else torch.tensor(targets, device=device)
        outputs = model(images)
        loss = criterion(outputs, targets)
        nonmember_losses.extend(loss.cpu().numpy().tolist())

    if len(forget_losses) == 0 or len(nonmember_losses) == 0:
        return {"mia_forget": 0.5, "disc_forget": 0.0, "indisc_forget": 1.0}

    y_true = np.concatenate([np.ones(len(forget_losses)), np.zeros(len(nonmember_losses))])
    # Lower loss indicates higher membership likelihood -> score = -loss
    scores = -np.concatenate([np.array(forget_losses), np.array(nonmember_losses)])

    try:
        from sklearn.metrics import roc_auc_score
        mia = float(roc_auc_score(y_true, scores))
    except Exception:
        mia = 0.5

    disc = float(abs(2.0 * mia - 1.0))
    indisc = float(1.0 - disc)

    return {
        "mia_forget": mia,
        "disc_forget": disc,
        "indisc_forget": indisc,
    }


def evaluate_three_tier_metrics(
    model: FullClassifier,
    probe_train_loader: DataLoader,
    retain_eval_loader: DataLoader,
    forget_eval_loader: DataLoader,
    num_classes: int = 10,
    seed: int = 123,
    device: str = "cuda",
    test_eval_loader: Optional[DataLoader] = None,
    nonmember_loader: Optional[DataLoader] = None,
) -> Dict[str, float]:
    """
    Evaluates three distinct categories of unlearning metrics:
    
    1. Accuracy / Utility:
       - output_retain: Retain training set accuracy on D_R (matches Deep Unlearn benchmark definition).
       - output_forget: Forget set accuracy on D_F.
       - output_test: Generalization accuracy on official unseen test set D_T.
       
    2. Representation Diagnostic (Linear Probe & NCC):
       - lp_retain, ncc_retain: In-sample representation diagnostics on D_R.
       - lp_forget, ncc_forget: Latent representation unlearning effectiveness on D_F.
       - lp_test, ncc_test: Out-of-sample representation retention on D_T.
       - illusion_gap_lp, illusion_gap_ncc: Illusion gap metrics on D_F.
       
    3. Privacy / Membership Inference (MIA):
       - mia_forget: AUC-ROC of distinguishing D_F from unseen non-members (D_V or D_T).
       - disc_forget: Discernibility |2 * MIA - 1|.
       - indisc_forget: Indiscernibility 1 - Disc.
    """
    # --- Category 1: Accuracy / Utility ---
    out_r = compute_output_accuracy(model, retain_eval_loader, device=device)
    out_f = compute_output_accuracy(model, forget_eval_loader, device=device)

    # --- Category 2: Representation Diagnostic ---
    train_feats, train_targets = extract_features(model.encoder, probe_train_loader, device=device)
    r_feats, r_targets = extract_features(model.encoder, retain_eval_loader, device=device)
    f_feats, f_targets = extract_features(model.encoder, forget_eval_loader, device=device)

    lp_r = compute_linear_probe_accuracy(train_feats, train_targets, r_feats, r_targets, seed=seed)
    lp_f = compute_linear_probe_accuracy(train_feats, train_targets, f_feats, f_targets, seed=seed)

    ncc_r = compute_ncc_accuracy(train_feats, train_targets, r_feats, r_targets, num_classes=num_classes)
    ncc_f = compute_ncc_accuracy(train_feats, train_targets, f_feats, f_targets, num_classes=num_classes)

    gap_lp = max(lp_f - out_f, 0.0)
    gap_ncc = max(ncc_f - out_f, 0.0)

    res = {
        # Category 1: Accuracy / Utility
        "output_retain": out_r,
        "output_forget": out_f,
        # Category 2: Representation
        "lp_retain": lp_r,
        "lp_forget": lp_f,
        "ncc_retain": ncc_r,
        "ncc_forget": ncc_f,
        "illusion_gap_lp": gap_lp,
        "illusion_gap_ncc": gap_ncc,
    }

    if test_eval_loader is not None:
        out_t = compute_output_accuracy(model, test_eval_loader, device=device)
        t_feats, t_targets = extract_features(model.encoder, test_eval_loader, device=device)
        lp_t = compute_linear_probe_accuracy(train_feats, train_targets, t_feats, t_targets, seed=seed)
        ncc_t = compute_ncc_accuracy(train_feats, train_targets, t_feats, t_targets, num_classes=num_classes)
        res["output_test"] = out_t
        res["lp_test"] = lp_t
        res["ncc_test"] = ncc_t

    # --- Category 3: Privacy / Membership Inference (U-MIA) ---
    # Non-member reference population MUST be D_V (passed via nonmember_loader).
    # Falling back to retain_eval_loader (D_R) would silently compare members against members,
    # producing ~0.5 AUC and falsely indicating ideal unlearning.  Fail loudly instead.
    if nonmember_loader is None:
        raise ValueError(
            "nonmember_loader is required for MIA computation. "
            "Pass the D_V validation loader (never D_R or D_T) as nonmember_loader."
        )
    mia_res = compute_mia_metrics(model, forget_eval_loader, nonmember_loader, device=device)
    res.update(mia_res)

    return res


# ==============================================================================
# 4. Feature Space Visualization (t-SNE & Grouped Metric Bar Charts)
# ==============================================================================

def visualize_feature_space_and_boundaries(
    model: nn.Module,
    retain_loader: DataLoader,
    forget_loader: DataLoader,
    num_classes: int = 10,
    forget_class: int = 0,
    title: str = "t-SNE Feature Space Visualization",
    max_samples_per_class: int = 100,
    device: str = "cuda",
    save_path: Optional[str] = None,
    projection_method: str = "tsne",
    normalize_features: bool = True,
):
    """
    Visualizes representations in 2D space matching Figure 3 & Figure 6 of Gao et al. (arXiv:2604.08271v1).
    1. Extracts high-dimensional representations for retain & forget samples.
    2. Applies L2 normalization (as in Figure 6 of paper).
    3. Projects to 2D via t-SNE (or PCA if requested).
    4. Plots retain classes (pastel/tab10 points) and forget class (distinct highlighted points)
       to show whether forget representations remain linearly separable or collapse/overlap with retain.
    """
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA

    # Extract encoder
    encoder = getattr(model, "encoder", model)
    encoder.eval()

    # Collect retain and forget features
    feats_r, labels_r = extract_features(encoder, retain_loader, device=device)
    feats_f, labels_f = extract_features(encoder, forget_loader, device=device)

    # Subsample for clean, fast visualization
    selected_feats = []
    selected_labels = []
    is_forget = []

    # Forget samples
    if len(feats_f) > max_samples_per_class * 2:
        idx_f = np.random.choice(len(feats_f), max_samples_per_class * 2, replace=False)
    else:
        idx_f = np.arange(len(feats_f))
    selected_feats.append(feats_f[idx_f])
    selected_labels.append(labels_f[idx_f])
    is_forget.append(np.ones(len(idx_f), dtype=bool))

    # Retain samples per class
    for c in range(num_classes):
        mask_c = (labels_r == c)
        if mask_c.sum() > 0:
            c_feats = feats_r[mask_c]
            c_labels = labels_r[mask_c]
            if len(c_feats) > max_samples_per_class:
                idx_c = np.random.choice(len(c_feats), max_samples_per_class, replace=False)
            else:
                idx_c = np.arange(len(c_feats))
            selected_feats.append(c_feats[idx_c])
            selected_labels.append(c_labels[idx_c])
            is_forget.append(np.zeros(len(idx_c), dtype=bool))

    X = np.concatenate(selected_feats, axis=0)
    y = np.concatenate(selected_labels, axis=0)
    is_f = np.concatenate(is_forget, axis=0)

    # Optional L2 feature normalization matching paper Figure 6
    if normalize_features:
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        X = X / norms

    # 2D Projection: t-SNE (paper Figure 3/6) or PCA
    if projection_method.lower() == "tsne":
        perp = min(30, max(5, (len(X) - 1) // 3))
        reducer = TSNE(n_components=2, perplexity=perp, random_state=42, init="pca", learning_rate="auto")
        X_2d = reducer.fit_transform(X)
        xlabel = "t-SNE Dimension 1"
        ylabel = "t-SNE Dimension 2"
    else:
        reducer = PCA(n_components=2, random_state=42)
        X_2d = reducer.fit_transform(X)
        xlabel = f"PCA Component 1 ({reducer.explained_variance_ratio_[0]*100:.1f}%)"
        ylabel = f"PCA Component 2 ({reducer.explained_variance_ratio_[1]*100:.1f}%)"

    plt.figure(figsize=(9, 7))
    cmap = plt.get_cmap("tab10")

    # Plot retain samples
    retain_mask = ~is_f
    for c in range(num_classes):
        c_mask = retain_mask & (y == c)
        if c_mask.sum() > 0:
            plt.scatter(
                X_2d[c_mask, 0], X_2d[c_mask, 1],
                color=cmap(c % 10), label=f"Retain Class {c}",
                alpha=0.45, s=25, edgecolors="none"
            )

    # Plot forget samples (highlighted in red, matching Figure 6 of paper)
    forget_mask = is_f
    plt.scatter(
        X_2d[forget_mask, 0], X_2d[forget_mask, 1],
        color="#d62728", label=f"Forgotten Class ({forget_class})",
        marker="o", s=35, edgecolors="#800000", linewidths=0.8, alpha=0.9
    )

    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, frameon=True)
    plt.grid(True, linestyle=":", alpha=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved feature visualization plot to: {save_path}")
    plt.show()


def plot_illusion_gap_comparison(
    results: List[Dict],
    title: str = "Comparison of Forget Accuracies: Output vs LP vs NCC (Figure 1)",
    save_path: Optional[str] = None,
):
    """
    Renders grouped bar chart matching Figure 1 of Gao et al. (arXiv:2604.08271v1):
    - Blue bars: Output-level Forget Accuracy
    - Red bars: Feature-level Linear Probe (LP) Forget Accuracy
    - Grey bars: Nearest Class Center (NCC) Forget Accuracy
    """
    import matplotlib.pyplot as plt

    methods = []
    output_f = []
    lp_f = []
    ncc_f = []

    for r in results:
        name = r.get("method") or r.get("variant") or "unknown"
        tm = r.get("test_metrics", {})
        methods.append(name.replace("_", " ").title())
        output_f.append(tm.get("output_forget", 0.0) * 100.0)
        lp_f.append(tm.get("lp_forget", 0.0) * 100.0)
        ncc_f.append(tm.get("ncc_forget", 0.0) * 100.0)

    x = np.arange(len(methods))
    width = 0.25

    plt.figure(figsize=(max(8, len(methods) * 1.8), 6))
    plt.bar(x - width, output_f, width, label="Output", color="#1f77b4")
    plt.bar(x, lp_f, width, label="Linear Probe", color="#d62728")
    plt.bar(x + width, ncc_f, width, label="NCC", color="#7f7f7f")

    plt.ylabel("Forget Accuracy (%)", fontsize=12, fontweight="bold")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xticks(x, methods, rotation=30, ha="right", fontsize=10)
    plt.legend(frameon=True, fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.ylim(0, 105)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"Saved Illusion Gap bar chart to: {save_path}")
    plt.show()


def print_metrics_summary_table(
    results: List[Dict],
    title: str = "Unlearning Benchmark Metrics Summary",
    save_csv_path: Optional[str] = None,
):
    """
    Renders and prints a consolidated, formatted markdown summary table across
    methods/models/seeds for all 3 tiers of metrics:
    - Output Accuracy (Retain / Forget)
    - Linear Probe (LP) Accuracy (Retain / Forget & Illusion Gap)
    - Nearest Class Center (NCC) Accuracy (Retain / Forget & Illusion Gap)
    - Runtime (seconds)
    Optionally saves the structured summary table to a CSV file.
    """
    import pandas as pd

    records = []
    for r in results:
        method = r.get("method") or r.get("variant") or "unknown"
        model_name = r.get("model_name", "resnet18")
        seed = r.get("seed", 0)
        runtime = r.get("runtime_sec", 0.0)
        tm = r.get("test_metrics", {})

        records.append({
            "Method / Variant": method,
            "Model": model_name,
            "Seed": seed,
            "Retain Acc (Out)": f"{tm.get('output_retain', 0.0)*100:.2f}%",
            "Forget Acc (Out)": f"{tm.get('output_forget', 0.0)*100:.2f}%",
            "Test Acc (Out)": f"{tm.get('output_test', 0.0)*100:.2f}%" if "output_test" in tm else "-",
            "Retain Acc (LP)": f"{tm.get('lp_retain', 0.0)*100:.2f}%",
            "Forget Acc (LP)": f"{tm.get('lp_forget', 0.0)*100:.2f}%",
            "Test Acc (LP)": f"{tm.get('lp_test', 0.0)*100:.2f}%" if "lp_test" in tm else "-",
            "LP Illusion Gap": f"{tm.get('illusion_gap_lp', 0.0)*100:.2f}%",
            "Retain Acc (NCC)": f"{tm.get('ncc_retain', 0.0)*100:.2f}%",
            "Forget Acc (NCC)": f"{tm.get('ncc_forget', 0.0)*100:.2f}%",
            "Test Acc (NCC)": f"{tm.get('ncc_test', 0.0)*100:.2f}%" if "ncc_test" in tm else "-",
            "NCC Illusion Gap": f"{tm.get('illusion_gap_ncc', 0.0)*100:.2f}%",
            "MIA (Forget)": f"{tm.get('mia_forget', 0.0)*100:.2f}%" if "mia_forget" in tm else "-",
            "Indiscernibility": f"{tm.get('indisc_forget', 0.0)*100:.2f}%" if "indisc_forget" in tm else "-",
            "Runtime (s)": f"{runtime:.2f}",
        })

    df = pd.DataFrame(records)
    print(f"\n{'=' * 100}")
    print(f" {title.upper()} ")
    print(f"{'=' * 100}\n")
    try:
        print(df.to_markdown(index=False))
    except (ImportError, Exception):
        print(df.to_string(index=False))
    print(f"\n{'=' * 100}\n")

    if save_csv_path:
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        df.to_csv(save_csv_path, index=False)
        print(f"Summary table saved to: {save_csv_path}")
    return df
