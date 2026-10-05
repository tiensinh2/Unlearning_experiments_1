from typing import Sequence, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def unlearn_tarun_unsir(
    model: nn.Module,
    retain_loader: DataLoader,
    forget_loader: DataLoader = None,
    num_classes: int = 10,
    img_shape: Tuple[int, int, int] = (3, 32, 32),
    noise_lr: float = 0.1,
    noise_epochs: int = 5,
    impair_epochs: int = 1,
    repair_epochs: int = 2,
    lr: float = 5e-5,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    device: str = "cuda",
) -> nn.Module:
    """
    UNSIR / TarUn (Fast Machine Unlearning via Error-Maximizing Impair-Repair).
    
    Source: Tarun et al. (Fast Yet Effective Machine Unlearning, TNNLS/TIFS).
    1. Error-maximizing noise optimization on forget data/targets.
    2. Impair phase: quick high-LR weight perturbation on synthesized noise.
    3. Repair phase: brief standard fine-tuning on retain data to restore utility.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()

    # --- Step 1: Synthesize Error-Maximizing Noise Matrix ---
    model.eval()
    batch_sz = retain_loader.batch_size or 128
    noise_param = torch.nn.Parameter(torch.randn((batch_sz, *img_shape), device=device), requires_grad=True)
    noise_opt = optim.Adam([noise_param], lr=noise_lr)

    # Get sample targets from forget loader or retain loader
    if forget_loader is not None:
        sample_batch = next(iter(forget_loader))
        sample_targets = sample_batch[1].to(device)
    else:
        sample_batch = next(iter(retain_loader))
        sample_targets = sample_batch[1].to(device)

    if sample_targets.size(0) != batch_sz:
        sample_targets = sample_targets[:batch_sz]
        if sample_targets.size(0) < batch_sz:
            sample_targets = torch.cat([sample_targets, torch.zeros(batch_sz - sample_targets.size(0), dtype=torch.long, device=device)])

    for _ in range(noise_epochs):
        noise_opt.zero_grad()
        out = model(noise_param)
        loss = -criterion(out, sample_targets) + 0.1 * torch.mean(torch.sum(torch.square(noise_param), dim=[1, 2, 3]))
        loss.backward()
        noise_opt.step()

    noise_tensor = noise_param.detach()

    # --- Step 2: Impair Step (Perturbation with noisy data) ---
    model.train()
    optimizer_impair = optim.Adam(model.parameters(), lr=lr * 10)

    for _ in range(impair_epochs):
        rand_label = torch.randint(0, num_classes, (noise_tensor.size(0),), device=device)
        optimizer_impair.zero_grad()
        out = model(noise_tensor)
        loss = criterion(out, rand_label)
        loss.backward()
        optimizer_impair.step()

    # --- Step 3: Repair Step (Fine-tuning on Retain Set) ---
    optimizer_repair = optim.SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    for _ in range(repair_epochs):
        for images, targets in retain_loader:
            images = images.to(device)
            if not isinstance(targets, torch.Tensor):
                targets = torch.tensor(targets, device=device)
            else:
                targets = targets.to(device)

            optimizer_repair.zero_grad()
            out = model(images)
            loss = criterion(out, targets)
            loss.backward()
            optimizer_repair.step()

    return model
