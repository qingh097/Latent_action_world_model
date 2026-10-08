# References:
#     Genie: Generative Interactive Environments (https://arxiv.org/abs/2402.15391)
#     Latent Action Pretraining from Videos (https://arxiv.org/abs/2410.11758)
#     https://github.com/pytorch/torchtitan/blob/6cb13c7a97eed890f7df872261c9bf1523959dee/torchtitan/models/llama/model.py

import logging
from typing import Callable

import torch
import torch.nn as nn
from einops import repeat

from robot_wm.modeling.modules.attention import precompute_freqs_cis
from robot_wm.modeling.modules.video_blocks import (
    STAttentionBlock,
    VideoCrossAttentionBlock,
)

logger = logging.getLogger(__name__)


class InverseModelOutput(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        num_actions: int,
        output_dim: int,
        build_norm: Callable[[int], nn.Module] = nn.RMSNorm,
    ):
        super().__init__()
        self.query = nn.Embedding(num_actions, embedding_dim=dim)
        self.layers = nn.ModuleDict()
        for idx in range(num_layers):
            self.layers[str(idx)] = VideoCrossAttentionBlock(dim, num_heads)
        self.norm = build_norm(dim)
        self.proj = nn.Linear(dim, output_dim, bias=True)

    def init_weights(self, offset: int):
        nn.init.trunc_normal_(self.query.weight, mean=0.0, std=0.02)
        for idx, layer in enumerate(self.layers.values(), offset):
            init_std = 0.02 / (2 * (idx + 1)) ** 0.5
            layer.init_weights(init_std=init_std)
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=0.02)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        self.norm.reset_parameters()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor [N, T, H, W, D]
        Returns:
            torch.Tensor [N, T, A, D]
        """
        N, T, _, _, _ = x.shape
        query = repeat(self.query.weight, "A D -> N T A D", N=N, T=T)
        for layer in self.layers.values():
            query = layer(query=query, context=x)
        query = self.norm(query)
        query = self.proj(query)
        return query


class STInverseModel(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_layers: int,
        num_output_layers: int,
        num_actions: int,
        input_dim: int,
        output_dim: int,
        rope_end: int,
        rope_theta: int,
        attn_drop_prob: float = 0.0,
        proj_drop_prob: float = 0.0,
        mlp_drop_prob: float = 0.0,
        residual_drop_prob: float = 0.0,
        build_norm: Callable[[int], nn.Module] = nn.RMSNorm,
        gradient_checkpointing: bool = False,
    ):
        super().__init__()
        # This model attends over every latent pixel of every frame of every view, so its
        # activations dominate training memory (far more than the LoRA'd DiT, which is
        # already checkpointed). Recomputing them in the backward pass trades ~30% step
        # time for a large memory saving, and is numerically identical.
        self.gradient_checkpointing = gradient_checkpointing
        self.input_proj = nn.Linear(input_dim, dim, bias=False)
        self.layers = nn.ModuleDict()
        for idx in range(num_layers):
            self.layers[str(idx)] = STAttentionBlock(
                dim=dim,
                num_heads=num_heads,
                attn_drop_prob=attn_drop_prob,
                proj_drop_prob=proj_drop_prob,
                mlp_drop_prob=mlp_drop_prob,
                residual_drop_prob=residual_drop_prob,
            )
        self.norm = build_norm(dim)
        self.output_proj = InverseModelOutput(
            dim, num_heads, num_output_layers, num_actions, output_dim
        )
        self.proj_dropout = nn.Dropout(p=proj_drop_prob)
        self.register_buffer(
            name="freqs_cis",
            tensor=precompute_freqs_cis(
                dim=dim // num_heads, end=rope_end, theta=rope_theta
            ),
            persistent=False,
        )

    def init_weights(self):
        self.norm.reset_parameters()
        nn.init.trunc_normal_(self.input_proj.weight, mean=0.0, std=0.02)
        for idx, layer in enumerate(self.layers.values()):
            init_std = 0.02 / (2 * (idx + 1)) ** 0.5
            layer.init_weights(init_std=init_std)
        self.output_proj.init_weights(offset=len(self.layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor [N, T, H, W, D]
        Returns:
            torch.Tensor [N, T, A, D]
        """
        x = self.input_proj(x)
        x = self.proj_dropout(x)
        for layer in self.layers.values():
            if self.gradient_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    layer, x, self.freqs_cis, use_reentrant=False
                )
            else:
                x = layer(x, self.freqs_cis)
        x = self.norm(x)
        z = self.output_proj(x)
        z = self.proj_dropout(z)
        return z


def get_arch(arch: str, **kwargs) -> STInverseModel:
    if arch == "small":
        kwargs["dim"] = 512
        kwargs["num_heads"] = 16
        kwargs["num_layers"] = 8
        kwargs["num_output_layers"] = 4
    elif arch == "base":
        kwargs["dim"] = 768
        kwargs["num_heads"] = 16
        kwargs["num_layers"] = 12
        kwargs["num_output_layers"] = 6
    elif arch == "large":
        kwargs["dim"] = 1024
        kwargs["num_heads"] = 16
        kwargs["num_layers"] = 18
        kwargs["num_output_layers"] = 9
    elif arch == "huge":
        kwargs["dim"] = 1536
        kwargs["num_heads"] = 24
        kwargs["num_layers"] = 18
        kwargs["num_output_layers"] = 9
    else:
        raise ValueError(f"invalid arch: {arch}")
    return STInverseModel(**kwargs)
