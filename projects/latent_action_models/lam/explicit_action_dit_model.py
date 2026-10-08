# Explicit action-conditioned world model, used for adaptation stages 2 and 3: a variant of
# LatentActionDiTModel where the conditioning latent comes from an MLP projector that
# encodes the GROUND-TRUTH action (robot action + camera motion) instead of an inverse
# model deriving a latent action from image frames.
#
# Pipeline:
#   future GT action chunks --ActionEncoderMLP(+morphology)--> z (in the latent-action space)
#       --action_pool + ActionToControl--> control channel
#       --ActionToTokens--> clip_fea tokens
#   history frames --WanVAE(temporal)--> reference channel
#   future frames  --WanVAE+noise--> noisy --> Wan DiT --flow-matching--> future frames
#
# Everything downstream of z is the pretrained latent-action world model, so the projector
# learns to map explicit actions into the latent-action space. No inverse model and no
# action-decoding loss: the projector is trained by the flow-matching loss alone.

import logging

from lam.latent_action_dit_model import LatentActionDiTModel

logger = logging.getLogger(__name__)


class ExplicitActionDiTModel(LatentActionDiTModel):
    def __init__(self, *, action_encoder, **kwargs):
        # force off the image-derived latent-action path + action-decoding supervision
        kwargs["inverse_model"] = None
        kwargs["rgb_pos_embed"] = None
        kwargs["action_pos_embed"] = None
        kwargs["action_decoder"] = None
        super().__init__(**kwargs)
        self.action_encoder = action_encoder
        if self.action_encoder is not None:
            self.action_encoder.init_weights(init_std=0.02)
        logger.info("ExplicitActionDiTModel: conditioning on GT actions via ActionEncoderMLP")

    def _latent_actions(self, rgb, actions, morphology_index, idm_rgb=None):
        """Encode the future GT action chunks directly into the latent action.

        ``idm_rgb`` (the cross-augmented view) is accepted for signature compatibility and
        ignored: there is no inverse model here, so there is no shortcut to break.
        Returns (z_future, z_cond [N, num_future, D], future_tokens=None); no
        action-decoding supervision."""
        assert actions is not None, "ExplicitActionDiTModel requires GT actions"
        fut = actions[:, self.num_history_frames:]              # [N, num_future, chunk, action_dim]
        if self.morphology_tokens is not None and morphology_index is not None:
            morph_emb = self.morphology_tokens(morphology_index)  # [N, morph_dim]
        else:
            morph_emb = fut.new_zeros(fut.shape[0], self.latent_action_dim)
        z_future = self.action_encoder(fut, morph_emb)          # [N, num_future, 1, D]
        return z_future, z_future.mean(dim=2), None
