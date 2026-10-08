import math
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn


def _get_1d_sincos_pos_embed_from_grid(pos: np.ndarray, dim: int) -> np.ndarray:
    assert dim % 2 == 0
    omega = np.arange(dim // 2, dtype=np.float32)
    omega /= dim / 2.0
    omega = 1.0 / 10_000**omega

    pos = pos.reshape(-1)
    pos = np.einsum("m,d->md", pos, omega)
    pos_sin = np.sin(pos)
    pos_cos = np.cos(pos)

    pos_embed = np.concatenate([pos_sin, pos_cos], axis=1)
    return pos_embed


def _get_grid_dims(dim: int, num: int) -> list[int]:
    assert dim % 2 == 0
    small = int(math.floor(dim / num))
    if small % 2 == 1:
        small -= 1
    assert small % 2 == 0 and small > 0
    big = dim - small * (num - 1)
    assert big % 2 == 0 and big > 0
    return [big] + [small] * (num - 1)


def _get_nd_sincos_pos_embed(*size: int, dim: int):
    grid = [np.arange(s, dtype=np.float32) for s in size]
    grid = np.meshgrid(*grid, indexing="ij")
    grid_dims = _get_grid_dims(dim, len(size))
    pos_embed = [
        _get_1d_sincos_pos_embed_from_grid(g, gd) for g, gd in zip(grid, grid_dims)
    ]
    pos_embed = np.concatenate(pos_embed, axis=1)
    return pos_embed.reshape(list(size) + [dim])


class PositionalEmbedding(nn.Module):
    def __init__(self, *size: int, dim: int, use_sincos: bool):
        super().__init__()
        self._size = size
        self._dim = dim
        self._use_sincos = use_sincos
        self._pos_embed = nn.Parameter(
            torch.zeros(*size, dim), requires_grad=not use_sincos
        )

    def init_weights(self):
        if self._use_sincos:
            pos_embd = _get_nd_sincos_pos_embed(*self._size, dim=self._dim)
            self._pos_embed.copy_(torch.from_numpy(pos_embd))
        else:
            nn.init.trunc_normal_(self._pos_embed, mean=0.0, std=self._dim**-0.5)

    def _get_indices(self, x: torch.Tensor) -> Sequence[torch.Tensor]:
        return torch.meshgrid([torch.arange(s) for s in x.shape[1:]], indexing="ij")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert self._pos_embed.shape[1:] == x.shape[2:], (
            f"Positional embedding shape {self._pos_embed.shape[1:]} "
            f"inconsistent with input shape {x.shape[2:]}"
        )
        indices = self._get_indices(x)
        res = x + self._pos_embed[indices]
        return res
