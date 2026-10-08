# Wrapper for the Wan2.1 (VideoX-Fun) VAE, used as the RGB tokenizer for the
# Wan-DiT latent-action world model.
#
# The Wan VAE is a causal 3D VAE: 8x spatial + 4x temporal compression, 16-channel
# continuous latent. We use it two ways:
#   * encode_per_frame: encode each frame independently (image mode, F=1) -> one
#     latent per frame, for the latent-action inverse model.
#   * encode_temporal / decode_temporal: causal video encode/decode -> temporal
#     latent, for the diffusion forward (world) model + visualization.
#
# Pixel inputs are expected in [-1, 1] (the dataset transform maps [0,1] -> [-1,1]
# with mean=std=0.5). The VAE's internal `self.scale` (mean/std) normalizes the
# *latent*, not the pixels, so no extra pixel normalization is applied here.

import logging
import os
from typing import Sequence

import torch
from einops import rearrange
from omegaconf import OmegaConf

from robot_wm.modeling.tokenizers.rgb.base import RGBTokenizer

# Import the VAE class directly from the submodule to avoid videox_fun.models.__init__,
# which eagerly imports audio encoders (librosa etc.).
from videox_fun.models.wan_vae import AutoencoderKLWan

logger = logging.getLogger(__name__)


def _ceil_to_multiple(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


class WanVAETokenizer(RGBTokenizer):
    """Frozen Wan2.1 VAE wrapped to match the RGBTokenizer interface."""

    def __init__(
        self,
        model_path: str = os.environ.get("WAN_DIR", "./checkpoints/wan_fun_1.3b_control"),
        vae_subpath: str = "Wan2.1_VAE.pth",
        config_path: str = os.path.join(os.environ.get("VIDEOX_HOME", "./VideoX-Fun"), "config/wan2.1/wan_civitai.yaml"),
        output_dim: int = 16,
        input_range: str = "normal",
        input_mean: Sequence[float] = (0.5, 0.5, 0.5),
        input_std: Sequence[float] = (0.5, 0.5, 0.5),
        chunk_decoded_shape: Sequence[int] = (3, -1, -1),
        chunk_encoded_shape: Sequence[int] = (16, -1, -1),
        spatial_compression_ratio: int = 8,
        temporal_compression_ratio: int = 4,
        pad_multiple: int = 16,
    ):
        super().__init__(
            output_dim=output_dim,
            input_range=input_range,
            input_mean=list(input_mean),
            input_std=list(input_std),
            chunk_decoded_shape=chunk_decoded_shape,
            chunk_encoded_shape=chunk_encoded_shape,
        )
        assert output_dim == 16, "Wan VAE latent is 16-channel"
        self.spatial_ratio = spatial_compression_ratio
        self.temporal_ratio = temporal_compression_ratio
        # pixel H,W must be divisible by spatial(8) * patch(2) = 16 so the DiT patchify works
        self.pad_multiple = pad_multiple

        cfg = OmegaConf.load(config_path)
        vae_kwargs = OmegaConf.to_container(cfg["vae_kwargs"])
        self.model = AutoencoderKLWan.from_pretrained(
            os.path.join(model_path, vae_kwargs.get("vae_subpath", vae_subpath)),
            additional_kwargs=vae_kwargs,
        )
        self.model = self.model.float()  # keep the VAE in fp32 (bf16 conv VAE is numerically unstable -> nan)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @property
    def scale_factors(self) -> Sequence[int]:
        return (self.spatial_ratio, self.spatial_ratio)

    # ------------------------------------------------------------------ helpers
    def _pad_hw(self, x: torch.Tensor) -> torch.Tensor:
        """Pad the last two dims (H, W) up to a multiple of pad_multiple with -1 (black)."""
        h, w = x.shape[-2], x.shape[-1]
        ph, pw = _ceil_to_multiple(h, self.pad_multiple) - h, _ceil_to_multiple(w, self.pad_multiple) - w
        if ph or pw:
            x = torch.nn.functional.pad(x, (0, pw, 0, ph), mode="constant", value=-1.0)
        return x

    def latent_temporal_len(self, num_frames: int) -> int:
        return (num_frames - 1) // self.temporal_ratio + 1

    # ------------------------------------------------------------------ encode
    @torch.no_grad()
    def encode_per_frame(self, rgb: torch.Tensor) -> torch.Tensor:
        """Encode each frame independently (image mode).
        Args:  rgb [N, T, C, H, W] in [-1, 1]
        Returns: [N, T, d, h, w]
        """
        n, t = rgb.shape[0], rgb.shape[1]
        x = rearrange(rgb, "n t c h w -> (n t) c h w")
        x = self._pad_hw(x)
        x = x.unsqueeze(2)  # (n t) c 1 H W  -> single-frame videos
        with torch.autocast(device_type="cuda", enabled=False):
            z = self.model.encode(x.float()).latent_dist.mode()  # (n t) d 1 h w
        z = z.squeeze(2)
        z = rearrange(z, "(n t) d h w -> n t d h w", n=n, t=t)
        return z

    @torch.no_grad()
    def encode_temporal(self, video: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """Causal video encode.
        Args:  video [N, C, F, H, W] in [-1, 1]
        Returns: [N, d, F', h, w]
        """
        x = rearrange(video, "n c f h w -> (n) c f h w")
        # pad H, W (dims -2, -1)
        x = self._pad_hw(x)
        with torch.autocast(device_type="cuda", enabled=False):
            dist = self.model.encode(x.float()).latent_dist
            return dist.sample() if sample else dist.mode()

    # ------------------------------------------------------------------ decode
    @torch.no_grad()
    def decode_temporal(self, z: torch.Tensor, out_hw: Sequence[int] = None) -> torch.Tensor:
        """Decode a temporal latent back to pixels in [-1, 1].
        Args:  z [N, d, F', h, w]
        Returns: [N, C, F, H, W]
        """
        with torch.autocast(device_type="cuda", enabled=False):
            dec = self.model.decode(z.float()).sample  # [N, C, F, H, W], clamped to [-1,1]
        if out_hw is not None:
            dec = dec[..., : out_hw[0], : out_hw[1]]
        return dec

    # --------------------------------------------------- RGBTokenizer interface
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # default encode == per-frame (used by the inverse model path)
        return self.encode_per_frame(x)

    def decode(self, z: torch.Tensor, out_shape: Sequence[int] = None) -> torch.Tensor:
        out_hw = (out_shape[-2], out_shape[-1]) if out_shape is not None else None
        return self.decode_temporal(z, out_hw=out_hw)
