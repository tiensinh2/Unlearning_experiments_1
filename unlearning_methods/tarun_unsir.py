from typing import Callable, Optional, Sequence, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


def unlearn_tarun_unsir(
    model: nn.Module,
    retain_loader: DataLoader,
    forget_loader: DataLoader,
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
    retain_finetune: bool = True,
    epoch_end_callback: Optional[Callable[[nn.Module], None]] = None,
) -> nn.Module:
    """
    UNSIR / TarUn (Fast Machine Unlearning via Error-Maximizing Impair-Repair).

    Source: Tarun et al. (Fast Yet Effective Machine Unlearning, TNNLS / arXiv:2111.08947).
    Official Repository: https://github.com/vikram2000b/Fast-Machine-Unlearning

    1. Error-maximizing noise optimization:
       Learns class-wise noise tensor per forgotten class minimizing -CE(y_forget) + 0.1 * ||noise||_2^2.
    2. Impair phase:
       Perturbs weights using synthesized noise labeled with its forget class (plus retain mix) with Adam optimizer.
    3. Repair phase:
       Restores retain accuracy through brief fine-tuning on retain data with Adam optimizer.

    epoch_end_callback(model) is called after every impair and repair epoch
    (e.g. CMF head reconstruction, Alg. 2 line 7).

    Notes on random-subset unlearning:
    ------------------------------------
    This benchmark uses random-subset unlearning where D_F contains samples from every class.
    The noise synthesis step is applied *per forget class* — every class represented in D_F
    gets its own error-maximizing noise tensor.  The impair step then mixes all noise tensors
    with a subset of retain samples before the repair step restores retain utility.

    When `forget_classes` is provided explicitly (e.g. from the test suite), that list is used
    directly.  When derived from `forget_loader`, ALL unique classes in D_F are targeted —
    not just class 0.
    """
    model.to(device)
    batch_sz = retain_loader.batch_size or 256

    # ------------------------------------------------------------------
    # Determine classes to forget
    # ------------------------------------------------------------------
    if forget_classes is None:
        if forget_loader is not None:
            classes_set = set()
            for _, targets in forget_loader:
                if isinstance(targets, torch.Tensor):
                    classes_set.update(targets.cpu().tolist())
                else:
                    classes_set.update(list(targets))
            # Use ALL distinct classes found in D_F — do NOT collapse to a single class.
            forget_classes = sorted(classes_set)
        else:
            forget_classes = list(range(num_classes))

    # ------------------------------------------------------------------
    # Step 1: Synthesize Error-Maximizing Noise Matrix per Forget Class
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Step 2: Impair Step (Weight perturbation with noise + retain data)
    # ------------------------------------------------------------------
    # Collect a retain sample subset for the impair mix.
    retain_samples = []
    if retain_finetune:
        for images, targets in retain_loader:
            if not isinstance(targets, torch.Tensor):
                targets = torch.tensor(targets)
            for img, tgt in zip(images, targets):
                retain_samples.append((img, tgt))
            if len(retain_samples) >= batch_sz * impair_batches:
                break

    # Build impair dataset: each noise tensor labeled with the class it was optimized for
    # (as in the official UNSIR implementation) + optional retain samples.
    impair_data = []
    for cls in forget_classes:
        noise_tensor = noises[cls].cpu()
        noise_label = torch.tensor(cls, dtype=torch.long)
        for _ in range(impair_batches):
            for idx in range(noise_tensor.size(0)):
                impair_data.append((noise_tensor[idx], noise_label))

    impair_data.extend(retain_samples)

    model.train()
    trainable_impair = [p for p in model.parameters() if p.requires_grad]
    optimizer_impair = optim.Adam(trainable_impair, lr=impair_lr)
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
        if epoch_end_callback is not None:
            epoch_end_callback(model)
            model.train()

    # ------------------------------------------------------------------
    # Step 3: Repair Step (Fine-tuning on Retain Set only — D_F / D_V / D_T never used)
    # ------------------------------------------------------------------
    if retain_finetune:
        trainable_repair = [p for p in model.parameters() if p.requires_grad]
        optimizer_repair = optim.Adam(trainable_repair, lr=repair_lr)

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
            if epoch_end_callback is not None:
                epoch_end_callback(model)
                model.train()

    return model
