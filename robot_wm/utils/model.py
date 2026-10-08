from typing import Callable, Optional

import torch.nn as nn


def get_model_size(model: nn.Module, requires_grad: Optional[bool] = None):
    requires_grad = [False, True] if requires_grad is None else [requires_grad]
    parameters = [p for p in model.parameters() if p.requires_grad in requires_grad]
    return sum([p.numel() for p in parameters])


def log_model_size(
    model: nn.Module, logger: Callable[[str], None], name: Optional[str] = None
):
    name = model.__class__.__name__ if name is None else name

    total_size = get_model_size(model)
    logger(f"{name} size: {total_size:,}")

    trainable_size = get_model_size(model, requires_grad=True)
    if trainable_size != total_size and trainable_size > 0:
        logger(f"{name} size (trainable): {trainable_size:,}")
