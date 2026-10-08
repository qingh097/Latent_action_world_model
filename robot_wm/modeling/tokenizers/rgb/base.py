from abc import ABC, abstractmethod
from typing import Sequence

import torch
import torch.nn as nn


class RGBTokenizer(ABC, nn.Module):
    def __init__(
        self,
        output_dim: int,
        input_range: str,
        input_mean: Sequence[float],
        input_std: Sequence[float],
        chunk_decoded_shape: Sequence[int],
        chunk_encoded_shape: Sequence[int],
    ):
        super().__init__()
        self._output_dim = output_dim
        self._input_range = input_range
        self._input_mean = input_mean
        self._input_std = input_std
        self._chunk_decoded_shape = chunk_decoded_shape
        self._chunk_encoded_shape = chunk_encoded_shape

    @property
    def dim(self) -> int:
        """Returns the output channel size."""
        return self._output_dim

    
    @property
    def chunk_decoded_shape(self) -> tuple[int, ...]:
        """Returns the shape of the decoded image tokens. (e.g. (3, 180, 320) for a DROID image). Chunk refers to one prediction timestep. """
        return self._chunk_decoded_shape

    @property
    def chunk_encoded_shape(self) -> tuple[int, ...]:
        """Returns the shape of the encoded image tokens. (e.g. (13, 23, 1024) for an encoded DROID image). Chunk refers to one prediction timestep. """
        return self._chunk_encoded_shape


    @property
    @abstractmethod
    def scale_factors(self) -> Sequence[int]:
        """Returns the scaling factors for the last n dimensions.
        Returns:
            Sequence[int]
        """

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor [N, *, C, H, W]
        Returns:
            torch.Tensor [N, *, d, h, w]
        """

    @abstractmethod
    def decode(self, z: torch.Tensor, out_shape: Sequence[int]) -> torch.Tensor:
        """
        Args:
            z: torch.Tensor [N, *, d, h, w]
            out_shape: Sequence[int] [N, *, C, H, W]
        Returns:
            torch.Tensor [N, *, C, H, W]
        """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encode(x)
