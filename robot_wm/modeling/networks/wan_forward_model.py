# Wan2.1-Fun-1.3B-Control as the forward (world) model for the latent-action world model.
#
# The Wan DiT supports either frozen-base LoRA or dense full fine-tuning. Latent actions
# reach it through up to two zero-initialized heads:
#   ActionToControl - per-latent-frame action -> the Fun-Control "control" channel (16-ch)
#   ActionToTokens  - per-action tokens -> the i2v image cross-attention slots (clip_fea)
# History frames go in the "reference" channel. Control and reference are
# channel-concatenated into `y` (32-ch); with the 16-ch noisy latent the DiT sees the
# expected in_dim=48.
#
# y = concat([control(16), reference(16)], dim=channel)  ->  transformer(x=noise, y=y, ...)

import logging
import math
import os

import torch
import torch.nn as nn
from einops import rearrange
from omegaconf import OmegaConf
from peft import LoraConfig, inject_adapter_in_model

from videox_fun.models.wan_transformer3d import WanTransformer3DModel

from robot_wm.utils.finetuning import (
    configure_full_finetune_transformer,
    normalize_finetune_mode,
)

logger = logging.getLogger(__name__)


class ActionToControl(nn.Module):
    """Project a per-(latent-)frame latent action vector into a 16-ch control latent,
    broadcast over the spatial grid. Zero-initialized so it starts as a no-op."""

    def __init__(self, action_dim: int = 64, latent_ch: int = 16, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(action_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_ch),
        )

    def init_weights(self, init_std: float = 0.02):
        nn.init.trunc_normal_(self.net[0].weight, std=init_std)
        nn.init.zeros_(self.net[0].bias)
        nn.init.zeros_(self.net[2].weight)  # zero-init last -> control starts as no-op
        nn.init.zeros_(self.net[2].bias)

    def forward(self, z: torch.Tensor, h: int, w: int) -> torch.Tensor:
        # z: [N, Fp, action_dim] -> [N, latent_ch, Fp, h, w]
        c = self.net(z)  # [N, Fp, latent_ch]
        c = rearrange(c, "n f d -> n d f")
        c = c[:, :, :, None, None].expand(-1, -1, -1, h, w)
        return c


class ActionToTokens(nn.Module):
    """Project the per-action latent sequence into clip_fea token slots for the Wan i2v
    image cross-attention branch (dedicated k_img/v_img in every block). The A action
    tokens occupy the first A of num_slots slots; the rest stay zero. A learned temporal
    position embedding (added before the zero-initialized output layer) tells the
    cross-attention WHICH future chunk each token controls. Zero-init output -> the
    module starts by emitting exactly the all-zero clip_fea the model has always seen."""

    def __init__(self, action_dim: int = 64, clip_dim: int = 1280, num_slots: int = 257,
                 hidden: int = 512, max_actions: int = 64):
        super().__init__()
        self.num_slots = num_slots
        self.pos = nn.Parameter(torch.zeros(max_actions, hidden))
        self.inp = nn.Linear(action_dim, hidden)
        self.act = nn.SiLU()
        self.out = nn.Linear(hidden, clip_dim)

    def init_weights(self, init_std: float = 0.02):
        nn.init.trunc_normal_(self.inp.weight, std=init_std)
        nn.init.zeros_(self.inp.bias)
        nn.init.trunc_normal_(self.pos, std=init_std)
        nn.init.zeros_(self.out.weight)  # zero-init last -> clip_fea starts all-zero (no-op)
        nn.init.zeros_(self.out.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [N, A, action_dim] -> [N, num_slots, clip_dim]
        n, a, _ = z.shape
        assert a <= self.num_slots, f"{a} action tokens > {self.num_slots} clip slots"
        tok = self.out(self.act(self.inp(z) + self.pos[:a]))
        pad = tok.new_zeros(n, self.num_slots - a, tok.shape[-1])
        return torch.cat([tok, pad], dim=1)


class WanForwardModel(nn.Module):
    def __init__(
        self,
        model_path: str = os.environ.get("WAN_DIR", "./checkpoints/wan_fun_1.3b_control"),
        config_path: str = os.path.join(os.environ.get("VIDEOX_HOME", "./VideoX-Fun"), "config/wan2.1/wan_civitai.yaml"),
        latent_action_dim: int = 64,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        control_hidden: int = 256,
        gradient_checkpointing: bool = True,
        finetune_mode: str = "lora",
        full_finetune_param_dtype: str = "float32",
        # how latent actions condition the DiT:
        #   "control" - pooled onto future latent frames via the spatial control channel
        #   "xattn"   - per-action tokens through the i2v image cross-attn branch (clip_fea)
        #   "both"    - both pathways
        action_conditioning: str = "control",
        clip_dim: int = 1280,
        clip_seq_len: int = 257,
        token_hidden: int = 512,
    ):
        super().__init__()
        assert action_conditioning in ("control", "xattn", "both"), action_conditioning
        self.action_conditioning = action_conditioning
        self.finetune_mode = normalize_finetune_mode(finetune_mode)
        self.full_finetune_param_dtype = str(full_finetune_param_dtype).lower()
        cfg = OmegaConf.load(config_path)
        subpath = cfg["transformer_additional_kwargs"].get("transformer_subpath", "./")
        self.transformer = WanTransformer3DModel.from_pretrained(
            os.path.join(model_path, subpath),
            transformer_additional_kwargs=OmegaConf.to_container(
                cfg["transformer_additional_kwargs"]
            ),
            low_cpu_mem_usage=True,
        )
        self.patch_size = self.transformer.config.patch_size

        if self.finetune_mode == "lora":
            # Freeze the full DiT, then inject + unfreeze LoRA adapters.
            for parameter in self.transformer.parameters():
                parameter.requires_grad = False
            lora_cfg = LoraConfig(
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                target_modules=["q", "k", "v", "o"],
            )
            self.transformer = inject_adapter_in_model(lora_cfg, self.transformer)
            for name, parameter in self.transformer.named_parameters():
                parameter.requires_grad = "lora_" in name
                if "lora_" in name:
                    # Keep master weights/grads/AdamW state in FP32 while autocast
                    # performs the adapter's forward matmuls in BF16.
                    parameter.data = parameter.data.float()
        else:
            # Dense mode has no PEFT topology.  FP32 parameters are the optimizer
            # master weights; the trainer computes in BF16 autocast.
            configure_full_finetune_transformer(
                self.transformer,
                param_dtype=self.full_finetune_param_dtype,
            )

        if gradient_checkpointing:
            try:
                self.transformer.enable_gradient_checkpointing()
            except Exception as e:  # noqa
                self.transformer.gradient_checkpointing = True
                logger.warning(f"enable_gradient_checkpointing fallback: {e}")

        # Only instantiate the pathway modules the mode actually uses, so DDP never
        # sees parameters that receive no gradient.
        self.action_to_control = (
            ActionToControl(latent_action_dim, 16, control_hidden)
            if action_conditioning in ("control", "both") else None
        )
        self.action_to_tokens = (
            ActionToTokens(latent_action_dim, clip_dim, clip_seq_len, token_hidden)
            if action_conditioning in ("xattn", "both") else None
        )

    def init_weights(self):
        if self.action_to_control is not None:
            self.action_to_control.init_weights()
        if self.action_to_tokens is not None:
            self.action_to_tokens.init_weights()

    def forward(
        self,
        noisy_latents: torch.Tensor,  # [N, 16, Fp, h, w]
        timesteps: torch.Tensor,      # [N]
        z_control: torch.Tensor,      # [N, Fp, action_dim] or None (xattn-only mode)
        ref_latents: torch.Tensor,    # [N, 16, Fp, h, w]
        context,                      # list of [L, text_dim]
        clip_fea: torch.Tensor = None,
        z_tokens: torch.Tensor = None,  # [N, num_actions, action_dim] per-action latents
    ) -> torch.Tensor:
        n, c, fp, h, w = noisy_latents.shape
        if self.action_to_control is not None and z_control is not None:
            control = self.action_to_control(z_control, h, w).to(noisy_latents.dtype)
        else:
            control = torch.zeros_like(ref_latents)
        if self.action_to_tokens is not None and z_tokens is not None:
            clip_fea = self.action_to_tokens(z_tokens).to(noisy_latents.dtype)
        y = torch.cat([control, ref_latents], dim=1)  # [N, 32, Fp, h, w]
        seq_len = int(math.ceil(h * w / (self.patch_size[1] * self.patch_size[2])) * fp)
        out = self.transformer(
            x=noisy_latents,
            t=timesteps,
            context=context,
            seq_len=seq_len,
            y=y,
            clip_fea=clip_fea,
        )
        return out[0] if isinstance(out, (list, tuple)) else out
