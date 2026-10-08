from typing import Optional

import wandb
from omegaconf import DictConfig, OmegaConf


def init_wandb_from_config(cfg: DictConfig, run_id: Optional[str] = None):
    assert "wandb" in cfg
    kwargs = OmegaConf.to_container(cfg.wandb, resolve=True)
    kwargs.pop("enabled", None)
    kwargs["id"] = kwargs.get("id", run_id)
    kwargs["config"] = kwargs.get("config", OmegaConf.to_container(cfg, resolve=True))
    kwargs["resume"] = kwargs.get("resume", "allow")
    wandb.init(**kwargs)
