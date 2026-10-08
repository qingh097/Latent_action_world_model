"""Fine-tuning policy for the Wan DiT.

The Wan checkpoint is stored in BF16. Dense fine-tuning deliberately promotes the
trainable transformer to FP32 before it is moved to CUDA. BF16 autocast still performs
the expensive matmuls in BF16, while gradients and AdamW moments remain FP32.
"""

from __future__ import annotations

import torch
import torch.nn as nn


VALID_FINETUNE_MODES = ("lora", "full")
FULL_FINETUNE_PARAM_DTYPE = "float32"


def normalize_finetune_mode(value: str) -> str:
    mode = str(value).strip().lower()
    if mode not in VALID_FINETUNE_MODES:
        raise ValueError(
            f"finetune_mode must be one of {VALID_FINETUNE_MODES}, got {value!r}"
        )
    return mode


def configure_full_finetune_transformer(
    transformer: nn.Module, *, param_dtype: str = FULL_FINETUNE_PARAM_DTYPE
) -> None:
    """Make every Wan transformer parameter trainable and FP32 for dense fine-tuning."""

    if str(param_dtype).lower() != FULL_FINETUNE_PARAM_DTYPE:
        raise ValueError(
            "full Wan fine-tuning requires FP32 trainable parameters so BF16 "
            "autocast uses FP32 gradients/AdamW moments; "
            f"got {param_dtype!r}"
        )
    parameters = list(transformer.parameters())
    if not parameters:
        raise ValueError("Wan transformer has no parameters")
    transformer.float()
    for parameter in parameters:
        parameter.requires_grad = True
    bad = [
        parameter.dtype
        for parameter in parameters
        if parameter.dtype != torch.float32
    ]
    if bad:
        raise RuntimeError(f"failed to promote Wan transformer parameters to FP32: {bad[:3]}")
