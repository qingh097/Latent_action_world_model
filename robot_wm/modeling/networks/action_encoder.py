# Explicit action encoder: an MLP that maps a per-frame GT robot-action chunk directly
# into a latent action used as the world-model conditioning (replaces the inverse model
# that derives a latent action from image frames).
#
# Actions arrive as [N, T, chunk_size, padding_dim] (heterogeneous morphologies are
# zero-padded to padding_dim=157). A morphology embedding is concatenated to the flattened
# chunk so the single MLP can interpret the (morphology-dependent) action layout.

import torch
import torch.nn as nn
from einops import rearrange


class ActionEncoderMLP(nn.Module):
    def __init__(
        self,
        action_dim: int = 157,      # padding_dim
        chunk_size: int = 5,
        latent_dim: int = 64,
        morph_dim: int = 64,
        hidden: int = 512,
        num_layers: int = 3,
    ):
        super().__init__()
        in_dim = action_dim * chunk_size + morph_dim
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(max(num_layers - 2, 0)):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers += [nn.Linear(hidden, latent_dim)]
        self.net = nn.Sequential(*layers)

    def init_weights(self, init_std: float = 0.02):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=init_std)
                nn.init.zeros_(m.bias)

    def forward(self, actions: torch.Tensor, morph_emb: torch.Tensor) -> torch.Tensor:
        """actions: [N, T, chunk, action_dim]; morph_emb: [N, morph_dim]
        returns latent action [N, T, 1, latent_dim]."""
        x = rearrange(actions, "n t a d -> n t (a d)")
        m = morph_emb.unsqueeze(1).expand(-1, x.shape[1], -1)
        x = torch.cat([x, m.to(x.dtype)], dim=-1)
        z = self.net(x)            # [N, T, latent_dim]
        return z.unsqueeze(2)      # [N, T, 1, latent_dim]
