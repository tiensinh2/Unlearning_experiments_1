from typing import Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


def unlearn_tarun_unsir(
    model: nn.Module,
    retain_loader: DataLoader,
    forget_loader: Optional[DataLoader] = None,
    forget_classes: Optional[Sequence[int]] = None,
    num_classes: int = 10,
    img_shape: Tuple[int, int, int] = (3, 32, 32),
    noise_lr: float = 0.1,
    noise_epochs: int = 5,
    noise_steps: int = 8,
    impair_lr: float = 0.02,
    impair_epochs: int = 1,
    impair_batches: int = 20,
    repair_lr: float = 0.01,
    repair_epochs: int = 1,
    device: str = "cuda",
) -> nn.Module:
    """
    UNSIR / TarUn (Fast Machine Unlearning via Error-Maximizing Impair-Repair).
    
    Source: Tarun et al. (Fast Yet Effective Machine Unlearning, TNNLS / arXiv:2111.08947).
    Official Repository: https://github.com/vikram2000b/Fast-Machine-Unlearning
    
    1. Error-maximizing noise optimization:
       Learns class-wise noise tensor per forgotten class minimizing -CE(y_forget) + 0.1 * ||noise||_2^2.
    2. Impair phase:
       Perturbs weights using synthesized noise labeled with pseudo class 0 (or retain mix) with Adam optimizer.
    3. Repair phase:
       Restores retain accuracy through brief fine-tuning on retain data with Adam optimizer.
    """
    model.to(device)
    batch_sz = retain_loader.batch_size or 256

    # Determine classes to forget
    if forget_classes is None:
        if forget_loader is not None:
            # Extract distinct classes from forget_loader
            classes_set = set()
            for _, targets in forget_loader:
                if isinstance(targets, torch.Tensor):
                    classes_set.update(targets.cpu().numpy().tolist())
                else:
                    classes_set.update(list(targets))
            forget_classes = sorted(list(classes_set))
        else:
            forget_classes = [0]

    # --- Step 1: Synthesize Error-Maximizing Noise Matrix per Forget Class ---
    model.eval()
    noises = {}
    for cls in forget_classes:
        noise_param = nn.Parameter(torch.randn((batch_sz, *img_shape), device=device), requires_grad=True)
        noise_opt = optim.Adam([noise_param], lr=noise_lr)
        cls_target = torch.full((batch_sz,), cls, dtype=torch.long, device=device)

        for epoch in range(noise_epochs):
            for step in range(noise_steps):
                noise_opt.zero_grad()
                out = model(noise_param)
                loss = -F.cross_entropy(out, cls_target) + 0.1 * torch.mean(
                    torch.sum(torch.square(noise_param), dim=[1, 2, 3])
                )
                loss.backward()
                noise_opt.step()

        noises[cls] = noise_param.detach()

    # --- Step 2: Impair Step (Weight perturbation with noise + retain data) ---
    # Construct noisy dataset (synthesized noise with pseudo label 0 + retain samples)
    model.train()
    optimizer_impair = optim.Adam(model.parameters(), lr=impair_lr)

    # Collect retain sample subset for mix
    retain_samples = []
    for images, targets in retain_loader:
        if not isinstance(targets, torch.Tensor):
            targets = torch.tensor(targets)
        for img, tgt in zip(images, targets):
            retain_samples.append((img, tgt))
        if len(retain_samples) >= batch_sz * impair_batches:
            break

    # Build impair dataset
    impair_data = []
    pseudo_label = torch.tensor(0, dtype=torch.long)
    for cls in forget_classes:
        noise_tensor = noises[cls].cpu()
        for _ in range(impair_batches):
            for idx in range(noise_tensor.size(0)):
                impair_data.append((noise_tensor[idx], pseudo_label))

    # Add retain samples
    impair_data.extend(retain_samples)
    impair_loader = DataLoader(impair_data, batch_size=batch_sz, shuffle=True)

    for _ in range(impair_epochs):
        for inputs, labels in impair_loader:
            inputs = inputs.to(device)
            labels = labels.to(device) if isinstance(labels, torch.Tensor) else torch.tensor(labels, device=device)

            optimizer_impair.zero_grad()
            outputs = model(inputs)
            loss = F.cross_entropy(outputs, labels)
            loss.backward()
            optimizer_impair.step()

    # --- Step 3: Repair Step (Fine-tuning on Retain Set) ---
    optimizer_repair = optim.Adam(model.parameters(), lr=repair_lr)

    for _ in range(repair_epochs):
        for images, targets in retain_loader:
            images = images.to(device)
            if not isinstance(targets, torch.Tensor):
                targets = torch.tensor(targets, device=device)
            else:
                targets = targets.to(device)

            optimizer_repair.zero_grad()
            outputs = model(images)
            loss = F.cross_entropy(outputs, targets)
            loss.backward()
            optimizer_repair.step()

    return model
