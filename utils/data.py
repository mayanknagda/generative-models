import torch
from .mnist import get_mnist_data
from .ng import get_ng_data
from typing import Any


def get_data(name: str) -> Any:
    """
    Used to get PyTorch dataloader for a specified dataset.
    Two datasets that are currently supported are:
    1. MNIST (name = "mnist")
    2. 20 Newsgroups (name = "ng")
    """
    if name == "mnist":
        return get_mnist_data()
    elif name == "ng":
        return get_ng_data()
