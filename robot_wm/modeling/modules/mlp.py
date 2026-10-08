# References:
#     https://github.com/pytorch/torchtitan/blob/6cb13c7a97eed890f7df872261c9bf1523959dee/torchtitan/models/llama/model.py

import copy
import logging
from typing import Callable, Optional

import torch
import torch.nn as nn
from hydra.utils import get_class

logger = logging.getLogger(__name__)

# decoder keys we've already warned about, so a missing morphology decoder is reported
# once instead of on every forward pass (avoids flooding the training log).
_MISSING_DECODER_WARNED: set = set()


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        output_dim: Optional[int] = None,
        build_act: Callable[[], nn.Module] = nn.GELU,
        drop_prob: float = 0.0,
    ):
        super().__init__()
        hidden_dim = input_dim if hidden_dim is None else hidden_dim
        output_dim = input_dim if output_dim is None else output_dim
        if isinstance(hidden_dim, int):
            hidden_dim = [hidden_dim]
        hidden_dim = [input_dim] + hidden_dim
        self.layers = nn.ModuleList()
        for i in range(len(hidden_dim) - 1):
            in_dim = int(hidden_dim[i])
            out_dim = int(hidden_dim[i + 1])
            self.layers.append(nn.Linear(in_dim, out_dim))
            self.layers.append(build_act())
            self.layers.append(nn.Dropout(p=drop_prob))
        self.out = nn.Linear(hidden_dim[-1], output_dim)

    def init_weights(self, init_std: float = 0.02):
        for layer in self.layers[:-1]:
            if isinstance(layer, nn.Linear):
                nn.init.trunc_normal_(layer.weight, mean=0.0, std=0.02)
        nn.init.trunc_normal_(self.out.weight, mean=0.0, std=init_std)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        x = self.out(x)
        return x


class SplitMultiMLP(nn.Module):
    def __init__(
        self,
        decoders,
        output_dim: Optional[int] = None,
        input_dim: Optional[int] = None,
        split_type=None,
    ):
        super().__init__()

        self.output_dim = output_dim
        self.input_dim = input_dim
        self.split_type = split_type
        morphology_ids = []
        split_decoders = dict()
        for ind, decoder in decoders.items():
            morphology_ids.append(ind)
            cls = get_class(decoder["target"])
            for k, v in split_type[ind].items():
                new_decoder = copy.deepcopy(decoder)
                new_decoder["input_dim"] = int(
                    self.split_type["action_type_split"][k][1]
                    - self.split_type["action_type_split"][k][0]
                )
                new_decoder["output_dim"] = v
                module = cls(
                    **{key: val for key, val in new_decoder.items() if key != "target"}
                )
                split_decoders[f"{ind}_{k}"] = module
        decoders = split_decoders

        self.decoders = nn.ModuleDict(decoders)
        self.morphology_ids = morphology_ids

    def init_weights(self, init_std: float):
        for k, decoder in self.decoders.items():
            decoder.init_weights(init_std=init_std)

    def forward(self, x: torch.Tensor, morphology_index: torch.Tensor, rgb_token=None):
        res = {mid: dict() for mid in self.morphology_ids}

        assert (
            morphology_index is not None
        ), "morphology_index is None, cannot use SplitMultiMLP"

        for morph_id in self.morphology_ids:
            mask = morphology_index == int(morph_id)
            res[morph_id]["mask"] = mask
            if not mask.any():
                continue
            x_in = x[mask]
            rgb_in = rgb_token[mask]
            cur_action = []
            for k, split_dim in self.split_type["action_type_split"].items():
                if f"{morph_id}_{k}" not in self.decoders:
                    key = f"{morph_id}_{k}"
                    if key not in _MISSING_DECODER_WARNED:
                        _MISSING_DECODER_WARNED.add(key)
                        logger.warning("no decoder for %s; skipping that action key", key)
                    continue
                decoder = self.decoders[f"{morph_id}_{k}"]
                input_latent = x_in[..., int(split_dim[0]) : int(split_dim[1])]
                out_action = decoder(input_latent, rgb_tokens=rgb_in)
                cur_action.append(out_action)

            res[morph_id]["actions"] = torch.cat(cur_action, dim=-1)

        return res


# usage example
