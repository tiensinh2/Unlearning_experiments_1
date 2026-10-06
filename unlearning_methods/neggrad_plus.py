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
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    device: str = "cuda",
) -> nn.Module:
    """
    NegGrad+ (Gradient Ascent on Forget + Gradient Descent on Retain).
    
    Source: Kurmanji et al. (SCRUB repo / repdistiller helper loops `train_negrad`).
    Loss formulation: loss = alpha * loss_retain - (1.0 - alpha) * loss_forget.
    """
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    for epoch in range(epochs):
        for (x_r, y_r), (x_f, y_f) in zip(retain_loader, cycle(forget_loader)):
            x_r = x_r.to(device)
            if not isinstance(y_r, torch.Tensor):
                y_r = torch.tensor(y_r, device=device)
            else:
                y_r = y_r.to(device)

            x_f = x_f.to(device)
            if not isinstance(y_f, torch.Tensor):
                y_f = torch.tensor(y_f, device=device)
            else:
                y_f = y_f.to(device)

            optimizer.zero_grad()
            out_r = model(x_r)
            out_f = model(x_f)

            loss_r = criterion(out_r, y_r)
            loss_f = criterion(out_f, y_f)

            # Maximize forget loss (negative sign) and minimize retain loss
            loss = alpha * loss_r - (1.0 - alpha) * loss_f
            loss.backward()
            optimizer.step()

    return model
