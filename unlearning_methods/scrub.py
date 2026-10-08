import copy
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
    epochs: int = 3,
    msteps: int = 2,
    lr: float = 1e-4,
    kd_T: float = 2.0,
    alpha: float = 0.5,
    gamma: float = 1.0,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    device: str = "cuda",
) -> nn.Module:
    """
    SCRUB (Student-Teacher Relabeling & Unlearning Bound).
    
    Source: Kurmanji et al. (Towards Unbounded Machine Unlearning, NeurIPS 2023).
    - Maximizes KL divergence on forget set (for the first msteps epochs): loss = -KL(student, teacher).
    - Minimizes weighted combination on retain set: loss = gamma * CE(student, targets) + alpha * KL(student, teacher).
    """
    model.to(device)
    teacher = copy.deepcopy(model).eval()
    student = model
    student.train()

    criterion_cls = nn.CrossEntropyLoss()
    criterion_kd = DistillKL(T=kd_T)
    trainable_params = [p for p in student.parameters() if p.requires_grad]
    optimizer = optim.SGD(trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    for epoch in range(1, epochs + 1):
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

        # Phase 2: Minimize divergence & standard classification loss on Retain set
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

    return student
