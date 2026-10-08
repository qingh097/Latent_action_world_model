# References:
#     https://github.com/pytorch/torchtitan/blob/8a92fb6649de607fbcf5983d13571ca91da593f2/torchtitan/models/llama_multimodal/model.py

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        attn_drop_prob: float = 0.0,
        proj_drop_prob: float = 0.0,
        is_causal: bool = False,
        build_norm: Callable[[int], nn.Module] = nn.RMSNorm,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        self.attn_drop_prob = attn_drop_prob
        self.proj_drop_prob = proj_drop_prob
        self.is_causal = is_causal

        self.head_dim = self.dim // self.num_heads
        assert self.head_dim * self.num_heads == self.dim

        self.q = nn.Linear(self.dim, self.num_heads * self.head_dim, bias=False)
        self.k = nn.Linear(self.dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v = nn.Linear(self.dim, self.num_kv_heads * self.head_dim, bias=False)
        self.proj = nn.Linear(self.dim, self.dim, bias=False)

        self.q_norm = build_norm(self.head_dim)
        self.k_norm = build_norm(self.head_dim)

        self.sdpa_backend_order = [
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
            torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
            torch.nn.attention.SDPBackend.MATH,
        ]

    def init_weights(self, init_std: float):
        for norm in [self.q_norm, self.k_norm]:
            norm.reset_parameters()
        for layer in (self.q, self.k, self.v):
            nn.init.trunc_normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=init_std)

    def forward(
        self,
        query: torch.Tensor,
        context: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        Hq, Hc, E = self.num_heads, self.num_kv_heads, self.head_dim
        N, Lq, D = query.shape
        _, Lc, _ = context.shape

        # q, k, v projection
        q = self.q(query).reshape(N, Lq, Hq, E)
        k = self.k(context).reshape(N, Lc, Hc, E)
        v = self.v(context).reshape(N, Lc, Hc, E)

        # q, k normalization
        q = self.q_norm(q)
        k = self.k_norm(k)

        # reshape for attention
        q = q.permute(0, 2, 1, 3)  # N, Hq, Lq, E
        k = k.permute(0, 2, 1, 3)  # N, Hc, Lc, E
        v = v.permute(0, 2, 1, 3)  # N, Hc, Lc, E

        # attention
        with torch.nn.attention.sdpa_kernel(self.sdpa_backend_order, set_priority=True):
            attn_drop_prob = self.attn_drop_prob if self.training else 0.0
            enable_gqa = self.num_heads != self.num_kv_heads
            x = F.scaled_dot_product_attention(
                query=q,
                key=k,
                value=v,
                attn_mask=attn_mask,
                dropout_p=attn_drop_prob,
                is_causal=self.is_causal and attn_mask is None,
                enable_gqa=enable_gqa,
            )  # N, Hq, Lq, E

        # reshape for output
        x = x.transpose(1, 2).reshape(N, Lq, D)  # N, Lq, D

        # output projection
        proj_drop_prob = self.proj_drop_prob if self.training else 0.0
        x = F.dropout(self.proj(x), p=proj_drop_prob)  # N, Lq, D

        return x


# usage example
