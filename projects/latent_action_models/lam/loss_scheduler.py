import logging
import math
from typing import Any

import torch
from torch.distributed.checkpoint.stateful import Stateful

logger = logging.getLogger(__name__)


class CustomLossScheduler(Stateful):
    def __init__(
        self,
        pretrain_steps: int,
        warmup_steps: int,
        total_steps: int,
        seed: int,
        losses: list[str],
        action_only_steps: int = 0,
        forward_only_steps: int = 0,
        cb_only_steps: int = 0,
        ae_only_steps: int = 0,
        decode_z_step: int = 0,
        action_decode_steps: int = 0,
    ):
        assert 1 <= len(losses) <= 2
        self.pretrain_steps = pretrain_steps
        self.warmup_steps = pretrain_steps + warmup_steps
        self.action_only_steps = action_only_steps
        self.forward_only_steps = forward_only_steps
        self.cb_only_steps = cb_only_steps
        self.decode_z_step = decode_z_step
        self.ae_only_steps = ae_only_steps
        self.action_decode_steps = action_decode_steps

        self.total_steps = total_steps
        self.losses = losses

        self._cur_step = 0
        self._gen = torch.Generator()
        self._gen.manual_seed(seed)

    def reset(self):
        self._cur_step = 0
        logger.info("Loss scheduler reset.")

    def step(self):
        self._cur_step += 1

    def _get_loss(self) -> int:
        prefix = ""
        if self._cur_step < self.pretrain_steps:
            prefix = "pretrain-"

        if self._cur_step < self.decode_z_step:
            prefix += "decode-z-"

        if self._cur_step < self.cb_only_steps:
            prefix += "cb-only-"

        if self._cur_step < self.ae_only_steps:
            prefix += "ae-only-"

        if self._cur_step < self.action_only_steps:
            prefix += "action-only-"

        if self._cur_step < self.action_decode_steps:
            prefix += "action-decode-"

        if (
            self._cur_step < self.forward_only_steps
            and self._cur_step >= self.action_only_steps
        ):
            prefix += "forward-only-"

        if len(self.losses) == 1:
            return prefix + self.losses[0]

        if self._cur_step < self.warmup_steps:
            return prefix + self.losses[0]

        decay_steps = self.total_steps - self.warmup_steps
        current_decay_step = min(self._cur_step - self.warmup_steps, self.total_steps)

        cosine_factor = 0.5 * (1 + math.cos(math.pi * current_decay_step / decay_steps))
        sampled_prob = torch.rand(1, generator=self._gen).item()
        if sampled_prob < cosine_factor:
            return prefix + self.losses[0]
        return prefix + self.losses[1]

    def get_loss(self) -> int:
        selected_loss = self._get_loss()
        logger.debug(f"{selected_loss = }")
        return selected_loss

    def state_dict(self) -> dict[str, Any]:
        output = {"_cur_step": self._cur_step, "_gen": self._gen.get_state()}
        return output

    def load_state_dict(self, state_dict: dict[str, Any]):
        self._cur_step = state_dict["_cur_step"]
        self._gen.set_state(state_dict["_gen"].cpu())
