import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset


def unlearn_random_label(
    model: nn.Module,
    retain_loader: DataLoader,
    forget_loader: DataLoader,
    num_classes: int,
    lr: float = 1e-4,
    epochs: int = 3,
    momentum: float = 0.9,
    weight_decay: float = 5e-4,
    device: str = "cuda",
) -> nn.Module:
    """
    Random Label (RL) Unlearning baseline (Golatkar et al., CVPR 2020 / OPTML Group).
    
    Reassigns random class labels to the forget set and trains on the combined
    (retain + random-labeled forget) dataset.
    """
    model.to(device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Deepcopy forget dataset and randomize target labels
    forget_dataset = copy.deepcopy(forget_loader.dataset)
    if hasattr(forget_dataset, "targets"):
        if isinstance(forget_dataset.targets, torch.Tensor):
            forget_dataset.targets = torch.randint(0, num_classes, forget_dataset.targets.shape)
        else:
            forget_dataset.targets = np.random.randint(0, num_classes, len(forget_dataset.targets)).tolist()

    # Combine retain and randomized forget datasets
    train_dataset = ConcatDataset([forget_dataset, retain_loader.dataset])
    batch_size = retain_loader.batch_size if retain_loader.batch_size is not None else 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        for images, targets in train_loader:
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

    return model
