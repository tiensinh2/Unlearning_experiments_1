from typing import Callable, Optional, Sequence
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader


class DistillKL(nn.Module):
    """
    KL Divergence with Temperature Scaling for Knowledge Distillation.
    Source: Kurmanji et al., NeurIPS 2023 / RepDistiller.
    """
    def __init__(self, T: float = 2.0):
        super().__init__()
        self.T = T

    def forward(self, y_s: torch.Tensor, y_t: torch.Tensor) -> torch.Tensor:
        p_s = F.log_softmax(y_s / self.T, dim=1)
        p_t = F.softmax(y_t / self.T, dim=1)
        loss = F.kl_div(p_s, p_t, reduction="batchmean") * (self.T ** 2)
        return loss


def unlearn_scrub(
    model: nn.Module,
    retain_loader: DataLoader,
    forget_loader: DataLoader,
    epochs: int = 10,
    msteps: int = 3,
    lr: float = 5e-4,
    kd_T: float = 2.0,
    alpha: float = 0.5,
    gamma: float = 1.0,
    weight_decay: float = 5e-4,
    momentum: float = 0.9,
    optim_name: str = "sgd",
    lr_decay_epochs: Optional[Sequence[int]] = None,
    lr_decay_rate: float = 0.1,
    batch_size: Optional[int] = 32,
    forget_batch_size: Optional[int] = 64,
    device: str = "cuda",
    retain_finetune: bool = True,
    epoch_end_callback: Optional[Callable[[nn.Module], None]] = None,
) -> nn.Module:
    """
    SCRUB (Student-Teacher Relabeling & Unlearning Bound).
    
    Source: Kurmanji et al. (Towards Unbounded Machine Unlearning, NeurIPS 2023).
    - Maximizes KL divergence on forget set (for the first msteps epochs): loss = -KL(student, teacher).
    - Minimizes weighted combination on retain set (controlled via retain_finetune, default True).
    Default: SGD, lr 5e-3, momentum 0.9, weight_decay 5e-4, 10 epochs, msteps 3,
    kd_T 2, gamma 1, alpha 0.5, retain batch 32, forget batch 64, step decay x0.1 after
    lr_decay_epochs = [int(0.5E), int(0.8E), int(0.9E)] = [5, 8, 9] (repdistiller `adjust_learning_rate`).
    Pass optim_name="adam", lr=5e-4, weight_decay=0.1 to match the official small_scale_unlearning.ipynb.
    momentum is ignored when optim_name="adam".
    batch_size / forget_batch_size rebuild the loaders from their datasets (None keeps the given loader).
    epoch_end_callback(student) is called after every epoch (e.g. CMF head reconstruction, Alg. 2 line 7);
    the teacher stays the original model.
    """
    model.to(device)
    teacher = copy.deepcopy(model).eval()
    student = model
    student.train()

    criterion_cls = nn.CrossEntropyLoss()
    criterion_kd = DistillKL(T=kd_T)
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    if optim_name == "adam":
        optimizer = optim.Adam(trainable_params, lr=lr, weight_decay=weight_decay)
    elif optim_name == "sgd":
        optimizer = optim.SGD(trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        raise ValueError(f"optim_name must be 'adam' or 'sgd', got {optim_name!r}")

    if lr_decay_epochs is None:
        lr_decay_epochs = [int(epochs * 0.5), int(epochs * 0.8), int(epochs * 0.9)]
    if batch_size is not None:
        retain_loader = DataLoader(retain_loader.dataset, batch_size=batch_size, shuffle=True,
                                   num_workers=retain_loader.num_workers, pin_memory=retain_loader.pin_memory)
    if forget_batch_size is not None:
        forget_loader = DataLoader(forget_loader.dataset, batch_size=forget_batch_size, shuffle=True,
                                   num_workers=forget_loader.num_workers, pin_memory=forget_loader.pin_memory)

    for epoch in range(1, epochs + 1):
        # Step LR decay (repdistiller adjust_learning_rate)
        steps = int(np.sum(epoch > np.asarray(lr_decay_epochs)))
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * (lr_decay_rate ** steps)

        # Phase 1: Maximize divergence on Forget set
        if epoch <= msteps:
            student.train()
            for images, _ in forget_loader:
                images = images.to(device)
                with torch.no_grad():
                    t_out = teacher(images)
                s_out = student(images)

                # Negated KL loss to push student outputs away from teacher
                loss_forget = -criterion_kd(s_out, t_out)

                optimizer.zero_grad()
                loss_forget.backward()
                optimizer.step()

        # Phase 2: Optional minimize divergence & standard classification loss on Retain set
        if retain_finetune:
            student.train()
            for images, targets in retain_loader:
                images = images.to(device)
                if not isinstance(targets, torch.Tensor):
                    targets = torch.tensor(targets, device=device)
                else:
                    targets = targets.to(device)

                with torch.no_grad():
                    t_out = teacher(images)
                s_out = student(images)

                loss_cls = criterion_cls(s_out, targets)
                loss_div = criterion_kd(s_out, t_out)
                loss_retain = gamma * loss_cls + alpha * loss_div

                optimizer.zero_grad()
                loss_retain.backward()
                optimizer.step()

        if epoch_end_callback is not None:
            epoch_end_callback(student)
            student.train()

    return student
