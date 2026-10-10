from typing import Callable, Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


def unlearn_random_label(
    model: nn.Module,
    retain_loader: DataLoader,
    forget_loader: DataLoader,
    num_classes: int,
    lr: float = 0.013,
    epochs: int = 10,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    device: str = "cuda",
    retain_finetune: bool = True,
    epoch_end_callback: Optional[Callable[[nn.Module], None]] = None,
) -> nn.Module:
    """
    Random Label (RL) Unlearning baseline (Golatkar et al., CVPR 2020 / OPTML Group).
    
    Iteratively trains on the forget set with uniformly randomized labels,
    followed by fine-tuning on the retain set with true labels (controlled by retain_finetune, default True).
    epoch_end_callback(model) is called after every epoch (e.g. CMF head reconstruction, Alg. 2 line 7).
    """
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(trainable_params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    for epoch in range(epochs):
        # 1. Update on forget set using randomized target labels
        for images, targets in forget_loader:
            images = images.to(device)
            rand_targets = torch.randint(0, num_classes, targets.shape, device=device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, rand_targets)
            loss.backward()
            optimizer.step()

        # 2. Optional update on retain set using true ground truth labels
        if retain_finetune:
            for images, targets in retain_loader:
                images = images.to(device)
                if not isinstance(targets, torch.Tensor):
                    targets = torch.tensor(targets, device=device)
                else:
                    targets = targets.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

        if epoch_end_callback is not None:
            epoch_end_callback(model)
            model.train()

    return model
