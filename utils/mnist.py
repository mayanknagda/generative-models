import torch
from torchvision import datasets, transforms
from typing import List, Any, Dict


def get_mnist_data(batch_size=128) -> Dict[str, Any]:
    """
    Used to get PyTorch dataloader for MNIST dataset.
    """
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data",
            train=True,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "data",
            train=False,
            download=True,
            transform=transforms.Compose(
                [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
            ),
        ),
        batch_size=batch_size,
        shuffle=True,
    )
    return {"train_dl": train_loader, "val_dl": test_loader, "test_dl": test_loader}
