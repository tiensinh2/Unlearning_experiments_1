import copy
from itertools import cycle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def unlearn_neggrad_plus(
    model: nn.Module,
    retain_loader: DataLoader,
    forget_loader: DataLoader,
    alpha: float = 0.95,
    lr: float = 1e-4,
    epochs: int = 3,
    grad_clip: float = 1.0,
    weight_decay: float = 5e-4,
    device: str = "cuda",
) -> nn.Module:
    """
    NegGrad+ (Gradient Ascent on Forget + Gradient Descent on Retain with L2 penalty to original initialization).
    
    Source: Kurmanji et al. (SCRUB repo) / Choi & Na (2023).
    """
    model.to(device)
    model.train()
    model_init = copy.deepcopy(model).eval()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    def l2_penalty(m: nn.Module, m_init: nn.Module) -> torch.Tensor:
        loss = torch.tensor(0.0, device=device)
        for (k, p), (k_i, p_i) in zip(m.named_parameters(), m_init.named_parameters()):
            if p.requires_grad:
                loss += (p - p_i).pow(2).sum()
        return (weight_decay / 2.0) * loss

    for epoch in range(epochs):
        for (x_r, y_r), (x_f, y_f) in zip(retain_loader, cycle(forget_loader)):
            x_r, y_r = x_r.to(device), y_r.to(device)
            x_f, y_f = x_f.to(device), y_f.to(device)

            optimizer.zero_grad()
            out_r = model(x_r)
            out_f = model(x_f)

            loss_r = criterion(out_r, y_r) + l2_penalty(model, model_init)
            loss_f = criterion(out_f, y_f)

            # Maximize forget loss (negative sign) and minimize retain loss
            loss = alpha * loss_r - (1.0 - alpha) * loss_f
            loss.backward()

            if grad_clip is not None and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

    return model
