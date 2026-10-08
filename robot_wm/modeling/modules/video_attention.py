# References:
#     https://github.com/myscience/open-genie/blob/732b9f9b746f18fff1a0fb22f83638224f2f7cc6/genie/module/attention.py

from typing import Optional

import torch
from einops import pack, rearrange, unpack

from robot_wm.modeling.modules.attention import Attention


class SpatialAttention(Attention):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        attn_drop_prob: float = 0.0,
        proj_drop_prob: float = 0.0,
        is_causal: bool = False,
    ):
        assert is_causal is False  # attend over the full spatial extent
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            attn_drop_prob=attn_drop_prob,
            proj_drop_prob=proj_drop_prob,
            is_causal=is_causal,
        )

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor [n t * d]
            freqs_cis: torch.Tensor [max_len head_dim 2]
        Returns:
            torch.Tensor [n t * d]
        """
        x, ps_hw = pack([x], "n t * d")  # n t (h w) d
        x, ps_nt = pack([x], "* hw d")  # (n t) (h w) d
        x = super().forward(x, freqs_cis)  # (n t) (h w) d
        x = unpack(x, ps_nt, "* hw d")[0]  # n t (h w) d
        x = unpack(x, ps_hw, "n t * d")[0]  # n t h w d
        return x


class TemporalAttention(Attention):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        attn_drop_prob: float = 0.0,
        proj_drop_prob: float = 0.0,
        is_causal: bool = True,
    ):
        assert is_causal is True  # attention should be causal over time
        super().__init__(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            attn_drop_prob=attn_drop_prob,
            proj_drop_prob=proj_drop_prob,
            is_causal=is_causal,
        )

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor [n t * d]
            freqs_cis: torch.Tensor [max_len head_dim 2]
        Returns:
            torch.Tensor [n t * d]
        """
        x, ps_hw = pack([x], "n t * d")  # n t (h w) d
        x = rearrange(x, "n t hw d -> n hw t d")  # n hw t d
        x, ps_nhw = pack([x], "* t d")  # (n h w) t d
        x = super().forward(x, freqs_cis)  # (n h w) t d
        x = unpack(x, ps_nhw, "* t d")[0]  # n h w t d
        x = rearrange(x, "n hw t d -> n t hw d")  # n t h w d
        x = unpack(x, ps_hw, "n t * d")[0]  # n t h w d
        return x


# usage example
