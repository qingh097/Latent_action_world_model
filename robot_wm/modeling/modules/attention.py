# References:
#     https://github.com/pytorch/torchtitan/blob/6cb13c7a97eed890f7df872261c9bf1523959dee/torchtitan/models/llama/model.py

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    freqs_cis = torch.view_as_real(freqs_cis)
    return freqs_cis


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    L, D = x.shape[1], x.shape[-1]
    freqs_cis = freqs_cis[0:L]
    assert freqs_cis.shape == (L, D)
    new_shape = [1 for _ in range(x.ndim)]
    new_shape[1], new_shape[-1] = L, D
    return freqs_cis.view(*new_shape)


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    x_out = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(torch.view_as_complex(freqs_cis.float()), x_out)
    x_out = torch.view_as_real(x_out * freqs_cis).reshape_as(x)
    return x_out.type_as(x)


class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        attn_drop_prob: float = 0.0,
        proj_drop_prob: float = 0.0,
        is_causal: bool = False,
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

        self.sdpa_backend_order = [
            torch.nn.attention.SDPBackend.FLASH_ATTENTION,
            torch.nn.attention.SDPBackend.EFFICIENT_ATTENTION,
            torch.nn.attention.SDPBackend.CUDNN_ATTENTION,
            torch.nn.attention.SDPBackend.MATH,
        ]

    def init_weights(self, init_std: float):
        for layer in (self.q, self.k, self.v):
            nn.init.trunc_normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.proj.weight, mean=0.0, std=init_std)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        Hq, Hkv, E = self.num_heads, self.num_kv_heads, self.head_dim
        N, L, D = x.shape

        # q, k, v projection
        q = self.q(x).reshape(N, L, Hq, E)
        k = self.k(x).reshape(N, L, Hkv, E)
        v = self.v(x).reshape(N, L, Hkv, E)

        # rotary positional embeddings (RoPE)
        q = apply_rotary_emb(q, freqs_cis=freqs_cis)
        k = apply_rotary_emb(k, freqs_cis=freqs_cis)

        # reshape for attention
        q = q.permute(0, 2, 1, 3)  # N, Hq, L, E
        k = k.permute(0, 2, 1, 3)  # N, Hkv, L, E
        v = v.permute(0, 2, 1, 3)  # N, Hkv, L, E

        # attention. For SHORT sequences (e.g. temporal attention over a few frames) the FLASH
        # fused backward can emit NaN gradients when keys are degenerate (identical across the
        # sequence -- happens here because masked/padded views are constant across frames). The
        # MATH backend is exact, stable, and cheap at small L, so use it for short sequences and
        # keep the fast flash/efficient order for long ones (e.g. spatial attention over h*w).
        backend_order = (
            [torch.nn.attention.SDPBackend.MATH] if q.shape[-2] <= 256 else self.sdpa_backend_order
        )
        with torch.nn.attention.sdpa_kernel(backend_order, set_priority=True):
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
            )  # N, Hq, L, E

        # reshape for output
        x = x.transpose(1, 2).reshape(N, L, D)  # N, L, D

        # output projection
        proj_drop_prob = self.proj_drop_prob if self.training else 0.0
        x = F.dropout(self.proj(x), p=proj_drop_prob)  # N, L, D

        return x


# usage example
