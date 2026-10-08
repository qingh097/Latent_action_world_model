# References:
#     https://github.com/pytorch/torchtitan/blob/6cb13c7a97eed890f7df872261c9bf1523959dee/torchtitan/models/llama/model.py
#     https://github.com/pytorch/torchtitan/blob/8a92fb6649de607fbcf5983d13571ca91da593f2/torchtitan/models/llama_multimodal/model.py
#     https://github.com/myscience/open-genie/blob/732b9f9b746f18fff1a0fb22f83638224f2f7cc6/genie/module/attention.py

from typing import Callable, Optional

import torch
import torch.nn as nn

from robot_wm.modeling.modules.mlp import MLP
from robot_wm.modeling.modules.video_attention import (
    SpatialAttention,
    TemporalAttention,
)
from robot_wm.modeling.modules.video_cross_attention import VideoCrossAttention


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth)
    Gao Huang, Yu Sun, Zhuang Liu, Daniel Sedra, and Kilian Q Weinberger. Deep networks with stochastic depth. In ECCV, 2016. 14, 17
    https://arxiv.org/abs/1603.09382
    """
    
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
        
    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor
    
    def extra_repr(self):
        return f'drop_prob={self.drop_prob}'


class VideoCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        attn_drop_prob: float = 0.0,
        proj_drop_prob: float = 0.0,
        build_norm: Callable[[int], nn.Module] = nn.RMSNorm,
        mlp_ratio: float = 4.0,
        mlp_build_act: nn.Module = nn.GELU,
        mlp_drop_prob: float = 0.0,
        residual_drop_prob: float = 0.0,
        drop_path: float = 0.0,
        use_checkpointing: bool = False,
    ):
        super().__init__()

        self.video_attn = VideoCrossAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            attn_drop_prob=attn_drop_prob,
            proj_drop_prob=proj_drop_prob,
            build_norm=build_norm,
        )
        self.video_attn_norm = build_norm(dim)

        self.mlp = MLP(
            input_dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            build_act=mlp_build_act,
            drop_prob=mlp_drop_prob,
        )
        self.mlp_norm = build_norm(dim)
        self.use_checkpointing = use_checkpointing
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.residual_dropout = nn.Dropout(residual_drop_prob) if residual_drop_prob > 0 else nn.Identity()

    def init_weights(self, init_std: float):
        for norm in [self.video_attn_norm, self.mlp_norm]:
            norm.reset_parameters()
        self.video_attn.init_weights(init_std=init_std)
        self.mlp.init_weights(init_std=init_std)

    def maybe_checkpoint(self, func, *args, **kwargs):
        if self.use_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, **kwargs, use_reentrant=False
            )
        return func(*args, **kwargs)

    def forward(self, query: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query: torch.Tensor [N, T, H, W, D]
            context: torch.Tensor [N, T, A, D]
        Returns:
            torch.Tensor [N, T, H, W, D]
        """
        attn_out = self.maybe_checkpoint(
            self.video_attn, self.video_attn_norm(query), context
        )
        attn_out = self.residual_dropout(attn_out)
        query = query + self.drop_path(attn_out)
        
        mlp_out = self.mlp(self.mlp_norm(query))
        mlp_out = self.residual_dropout(mlp_out)
        query = query + self.drop_path(mlp_out)
        return query


class STAttentionBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: Optional[int] = None,
        attn_drop_prob: float = 0.0,
        proj_drop_prob: float = 0.0,
        build_norm: Callable[[int], nn.Module] = nn.RMSNorm,
        mlp_ratio: float = 4.0,
        mlp_build_act: nn.Module = nn.GELU,
        mlp_drop_prob: float = 0.0,
        residual_drop_prob: float = 0.0,
        drop_path: float = 0.0,
        use_checkpointing: bool = False,
    ):
        super().__init__()

        self.spatial_attn = SpatialAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            attn_drop_prob=attn_drop_prob,
            proj_drop_prob=proj_drop_prob,
        )
        self.spatial_attn_norm = build_norm(dim)

        self.temporal_attn = TemporalAttention(
            dim=dim,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            attn_drop_prob=attn_drop_prob,
            proj_drop_prob=proj_drop_prob,
        )
        self.temporal_attn_norm = build_norm(dim)

        self.mlp = MLP(
            input_dim=dim,
            hidden_dim=int(dim * mlp_ratio),
            build_act=mlp_build_act,
            drop_prob=mlp_drop_prob,
        )
        self.mlp_norm = build_norm(dim)
        self.use_checkpointing = use_checkpointing
        
        self.drop_path1 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.drop_path3 = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        
        self.residual_dropout1 = nn.Dropout(residual_drop_prob) if residual_drop_prob > 0 else nn.Identity()
        self.residual_dropout2 = nn.Dropout(residual_drop_prob) if residual_drop_prob > 0 else nn.Identity()
        self.residual_dropout3 = nn.Dropout(residual_drop_prob) if residual_drop_prob > 0 else nn.Identity()

    def maybe_checkpoint(self, func, *args, **kwargs):
        if self.use_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, **kwargs, use_reentrant=False
            )
        return func(*args, **kwargs)

    def init_weights(self, init_std: float):
        for norm in [self.spatial_attn_norm, self.temporal_attn_norm, self.mlp_norm]:
            norm.reset_parameters()
        self.spatial_attn.init_weights(init_std=init_std)
        self.temporal_attn.init_weights(init_std=init_std)
        self.mlp.init_weights(init_std=init_std)

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: torch.Tensor [N, T, H, W, D]
            freqs_cis: torch.Tensor [L_max, D]
        Returns:
            torch.Tensor [N, T, H, W, D]
        """
        attn_out = self.maybe_checkpoint(
            self.spatial_attn, self.spatial_attn_norm(x), freqs_cis=freqs_cis
        )
        attn_out = self.residual_dropout1(attn_out)
        x = x + self.drop_path1(attn_out)
        
        attn_out = self.maybe_checkpoint(
            self.temporal_attn, self.temporal_attn_norm(x), freqs_cis=freqs_cis
        )
        attn_out = self.residual_dropout2(attn_out)
        x = x + self.drop_path2(attn_out)
        
        mlp_out = self.mlp(self.mlp_norm(x))
        mlp_out = self.residual_dropout3(mlp_out)
        x = x + self.drop_path3(mlp_out)
        return x


# usage example
