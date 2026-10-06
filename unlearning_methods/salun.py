from typing import Dict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def _apply_mask_to_grads(model: nn.Module, mask: Dict[str, torch.Tensor]) -> None:
    """Applies binary mask to gradients in-place."""
    for name, param in model.named_parameters():
        if param.grad is not None and name in mask:
            param.grad *= mask[name]


def _restore_masked_params(
    model: nn.Module,
    mask: Dict[str, torch.Tensor],
    theta0: Dict[str, torch.Tensor],
    optimizer: optim.Optimizer = None,
) -> None:
    """
    Restores unselected weights back to initial snapshot theta0 and clears momentum buffer.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name not in mask:
                continue
            mask_tensor = mask[name].to(device=param.device, dtype=param.dtype)
            inv_mask_tensor = 1.0 - mask_tensor
            if torch.count_nonzero(inv_mask_tensor) == 0:
                continue
            param.data.mul_(mask_tensor).add_(theta0[name].to(param.device) * inv_mask_tensor)

            if optimizer is not None:
                state = optimizer.state.get(param, None)
                if state is not None and "momentum_buffer" in state:
                    state["momentum_buffer"].mul_(mask_tensor)


def generate_salun_mask(
    model: nn.Module,
    forget_loader: DataLoader,
    threshold: float = 0.5,
    device: str = "cuda",
) -> Dict[str, torch.Tensor]:
    """
    Computes weight saliency mask from the first-order gradient magnitude on the forget set.
    Source: Fan et al., ICLR 2024 Spotlight (OPTML Group/Unlearn-Saliency).
    
    Accumulates signed gradients across forget batches, takes element-wise absolute value,
    and identifies the top-k percentile most salient parameters using argsort ranking.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    gradients = {name: torch.zeros_like(param) for name, param in model.named_parameters()}

    for images, targets in forget_loader:
        images, targets = images.to(device), targets.to(device)
        model.zero_grad()
        outputs = model(images)
        loss = -criterion(outputs, targets)  # Gradient ascent direction
        loss.backward()

        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.grad is not None:
                    gradients[name] += param.grad.data

    with torch.no_grad():
        for name in gradients:
            gradients[name] = torch.abs_(gradients[name])

        # Concatenate all tensors and calculate positions
        all_elements = -torch.cat([tensor.flatten() for tensor in gradients.values()])
        threshold_index = int(len(all_elements) * threshold)

        positions = torch.argsort(all_elements)
        ranks = torch.argsort(positions)

        mask = {}
        start_index = 0
        for name, tensor in gradients.items():
            num_elements = tensor.numel()
            tensor_ranks = ranks[start_index : start_index + num_elements]
            threshold_tensor = torch.zeros_like(tensor_ranks, dtype=torch.float32)
            threshold_tensor[tensor_ranks < threshold_index] = 1.0
            mask[name] = threshold_tensor.reshape(tensor.shape).to(device)
            start_index += num_elements

    return mask


def unlearn_salun(
    model: nn.Module,
    retain_loader: DataLoader,
    forget_loader: DataLoader,
    num_classes: int,
    threshold: float = 0.5,
    lr: float = 1e-4,
    epochs: int = 3,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    device: str = "cuda",
) -> nn.Module:
    """
    SalUn (Saliency-Guided Unlearning with Random Labeling).
    Only updates parameters with highest saliency while keeping remaining parameters intact.
    """
    mask = generate_salun_mask(model, forget_loader, threshold=threshold, device=device)
    theta0 = {name: param.detach().clone() for name, param in model.named_parameters()}

    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    for epoch in range(epochs):
        # 1. Update on forget set using randomized labels
        for images, targets in forget_loader:
            images = images.to(device)
            rand_targets = torch.randint(0, num_classes, targets.shape, device=device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, rand_targets)
            loss.backward()

            _apply_mask_to_grads(model, mask)
            optimizer.step()
            _restore_masked_params(model, mask, theta0, optimizer)

        # 2. Update on retain set using true labels
        for images, targets in retain_loader:
            images, targets = images.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()

            _apply_mask_to_grads(model, mask)
            optimizer.step()
            _restore_masked_params(model, mask, theta0, optimizer)

    return model
