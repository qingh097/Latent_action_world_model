import torch
import torch.nn as nn

from robot_wm.modeling.modules.cross_attention import CrossAttention
from robot_wm.modeling.modules.mlp import MLP


class ActionXattenDecoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        rgb_latent_dim: int = 1408,
    ):
        super().__init__()
        self.final_mlp = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
        )
        self.xatten = CrossAttention(
            dim=input_dim,
            num_heads=2,
            attn_drop_prob=0.0,
            proj_drop_prob=0.0,
            is_causal=False,
        )
        self.conv = nn.Conv2d(
            in_channels=rgb_latent_dim, out_channels=input_dim, kernel_size=4, stride=4
        )

    def init_weights(self, init_std: float = 0.02):
        self.final_mlp.init_weights(init_std=init_std)
        self.xatten.init_weights(init_std=init_std)

    def forward(
        self, action_embed: torch.Tensor, rgb_tokens: torch.Tensor
    ) -> torch.Tensor:
        B, T, H, W, D = rgb_tokens.shape
        rgb_flat_tokens = rgb_tokens.flatten(0, 1).permute(0, 3, 1, 2)  # (N*T) D H W
        rgb_reduced = self.conv(rgb_flat_tokens)  # (N*T) D H/4 W/4
        rgb_reduced = rgb_reduced.permute(0, 2, 3, 1).flatten(1, 2)  # (N*T) (H/4*W/4) D
        x = self.xatten(action_embed.flatten(0, 1)[:, None], rgb_reduced).squeeze(
            1
        )  # (N*T) D
        x = self.final_mlp(x)
        return x.view(B, T, -1)

