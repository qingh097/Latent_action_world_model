# References:
#     https://github.com/myscience/open-genie/blob/732b9f9b746f18fff1a0fb22f83638224f2f7cc6/genie/module/attention.py

import math
import warnings
from typing import Callable, Optional

import torch
import torch.nn as nn
from einops import pack, rearrange, repeat, unpack

from robot_wm.modeling.modules.cross_attention import CrossAttention


class VideoCrossAttention(CrossAttention):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        attn_drop_prob: float = 0.0,
        proj_drop_prob: float = 0.0,
        is_causal: bool = True,
        build_norm: Callable[[int], nn.Module] = nn.RMSNorm,
    ):
        assert is_causal is True  # attend over the full spatial extent
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            attn_drop_prob=attn_drop_prob,
            proj_drop_prob=proj_drop_prob,
            is_causal=is_causal,
            build_norm=build_norm,
        )

    @staticmethod
    def _get_causal_attn_mask(
        query: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """Returns a block causal attention mask.
        Args:
            query: torch.Tensor [N, T, *, D]
            context: torch.Tensor [N, T, *, D]
        Returns:
            torch.Tensor [Lq, Lc]
        """
        device = query.device
        T = query.shape[1]
        rq = math.prod(query.shape[2:-1])
        rc = math.prod(context.shape[2:-1])
        mask = torch.ones(T, T, device=device, dtype=torch.bool).tril(diagonal=0)
        mask = repeat(mask, "i j -> (i rq) (j rc)", rq=rq, rc=rc)
        return mask.contiguous()

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        assert query.shape[:2] == context.shape[:2]
        assert query.shape[-1] == context.shape[-1]
        mask = self._get_causal_attn_mask(query, context)
        query, ps_thw = pack([query], "n * d")  # n (t h w) d
        context, _ = pack([context], "n * d")  # n (t h w) d --or-- n (t a) d
        query = super().forward(query, context, mask)  # n (t h w) d
        query = unpack(query, ps_thw, "n * d")[0]  # n t h w d
        return query


class SpatialCrossAttention(CrossAttention):
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
        # print(
        #     "SpatialCrossAttention is deprecated; use VideoCrossAttention",
        #     DeprecationWarning,
        # )
        assert is_causal is False  # attend over the full spatial extent
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            attn_drop_prob=attn_drop_prob,
            proj_drop_prob=proj_drop_prob,
            is_causal=is_causal,
            build_norm=build_norm,
        )

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        assert query.shape[:2] == context.shape[:2]
        assert query.shape[-1] == context.shape[-1]
        query, ps_nt = pack([query], "n t * d")  # n t (h w) d
        query, ps_hw = pack([query], "* hw d")  # (n t) (h w) d
        context, _ = pack([context], "n t * d")  # n t (h w) d --or-- n t a d
        context, _ = pack([context], "* hw d")  # (n t) (h w) d --or-- (n t) a d
        query = super().forward(query, context)  # (n t) (h w) d
        query = unpack(query, ps_hw, "* hw d")[0]  # n t (h w) d
        query = unpack(query, ps_nt, "n t * d")[0]  # n t h w d
        return query


class TemporalCrossAttention(CrossAttention):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        attn_drop_prob: float = 0.0,
        proj_drop_prob: float = 0.0,
        is_causal: bool = True,
        build_norm: Callable[[int], nn.Module] = nn.RMSNorm,
    ):
        print(
            "TemporalCrossAttention is deprecated; use VideoCrossAttention",
            DeprecationWarning,
        )
        assert is_causal is True  # attention should be causal over time
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            attn_drop_prob=attn_drop_prob,
            proj_drop_prob=proj_drop_prob,
            is_causal=is_causal,
            build_norm=build_norm,
        )

    @staticmethod
    def _get_causal_attn_mask(
        query: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """Returns a block causal attention mask.
        Args:
            query: torch.Tensor [N, Lq, D]
            context: torch.Tensor [N, Lc, D]
        Returns:
            torch.Tensor [Lq, Lc]
        """
        device = query.device
        Lq, Lc = query.shape[1], context.shape[1]
        mask = torch.ones(Lq, Lq, device=device, dtype=torch.bool).tril(diagonal=0)
        mask = repeat(mask, "i j -> i (j r)", r=Lc // Lq)
        return mask.contiguous()

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        assert query.shape[:2] == context.shape[:2]
        assert query.shape[-1] == context.shape[-1]

        # convert `query` and `context` to: n t * d
        query, ps_q1 = pack([query], "n t * d")  # n t hw_q d
        context, _ = pack([context], "n t * d")  # n t hw_c d

        # convert `context` to: (n hw_q) * d
        if query.shape[2] == context.shape[2]:
            msg = "Are you sure you want to use patchwise temporal cross-attention?"
            warnings.warn(msg)
            context = rearrange(context, "n t hw_c d -> n hw_c t d")
            context, _ = pack([context], "* t d")  # (n hw_q) t d
        else:
            hw_q = query.shape[2]
            context = repeat(context, "n t hw_c d -> n hw_q t hw_c d", hw_q=hw_q)
            context, _ = pack([context], "n hw_q * d")  # n hw_q (t hw_c) d
            context, _ = pack([context], "* t_hw_c d")  # (n hw_q) (t hw_c) d

        # convert `query` to: (n hw_q) t d
        query = rearrange(query, "n t hw_q d -> n hw_q t d")
        query, ps_q2 = pack([query], "* t d")  # (n hw_q) t d

        # generate causal attention mask
        attn_mask = self._get_causal_attn_mask(query, context)

        # cross attention
        query = super().forward(query, context, attn_mask)

        # unpack `query` to (n t h w d)
        query = unpack(query, ps_q2, "* t d")[0]  # n hw_q t d
        query = rearrange(query, "n hw_q t d -> n t hw_q d")
        query = unpack(query, ps_q1, "n t * d")[0]  # n t h w d

        return query


# usage example
