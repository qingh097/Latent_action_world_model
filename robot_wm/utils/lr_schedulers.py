# References:
#     https://github.com/pytorch/torchtitan/blob/99f73969e1bc4b67115f3421c69b855ca4ef35c6/torchtitan/components/optimizer.py
import math


def linear_warmup_cosine_decay(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    start_learning_rate: float = 0.0,
    max_learning_rate: float = 1.0,
    final_learning_rate: float = 0.0
):
    """
    Learning rate scheduler with linear warmup and cosine decay.
    Compatible with PyTorch's LambdaLR when base learning rate equals max_learning_rate.

    Args:
        step (int): Current step number.
        warmup_steps (int): Number of warmup steps.
        total_steps (int): Total number of training steps.
        start_learning_rate (float): Initial learning rate.
        max_learning_rate (float): Maximum learning rate after warmup.
        final_learning_rate (float): Final learning rate at the end of training.

    Returns:
        float: A multiplier for the base learning rate.
    """
    # Calculate multiplier based on max_learning_rate being 1.0
    # This ensures compatibility with LambdaLR

    # Linear warmup phase
    if step < warmup_steps:
        # Start from start_lr/max_lr and increase to 1.0
        return (
            start_learning_rate
            + (max_learning_rate - start_learning_rate) * (step / warmup_steps)
        ) / max_learning_rate

    # Cosine decay phase
    decay_steps = total_steps - warmup_steps
    current_decay_step = min(step - warmup_steps, decay_steps)

    # Calculate cosine decay factor (1 to 0)
    cosine_factor = 0.5 * (1 + math.cos(math.pi * current_decay_step / decay_steps))

    # Apply cosine decay from 1.0 to final_lr/max_lr
    return (
        final_learning_rate + (max_learning_rate - final_learning_rate) * cosine_factor
    ) / max_learning_rate
