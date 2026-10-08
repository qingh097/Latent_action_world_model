import logging
import random
import sys
import traceback

import hydra
import numpy as np
import torch
from custom_resolvers import *
from omegaconf import DictConfig

import robot_wm.utils.distributed as dist
from robot_wm.utils.trainer import Trainer

logger = logging.getLogger(__name__)


def _setup(cfg: DictConfig) -> Trainer:
    # initialize distributed process group
    dist.init_process_group()

    # set random seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)

    # initialize trainer
    trainer: Trainer = hydra.utils.instantiate(cfg.trainer)

    # initialize wandb
    trainer.initialize_wandb(cfg)

    return trainer


def _teardown(trainer: Trainer):
    # finalize wandb
    trainer.finalize_wandb()

    # destroy distributed process group
    dist.destroy_process_group()


@hydra.main(version_base=None, config_path="configs", config_name="train.yaml")
def main(cfg: DictConfig):
    if cfg.debug:
        import debugpy  # only needed for interactive debugging

        logger.info(f"{cfg = }")
        debugpy.listen(("0.0.0.0", 5678))
        print("Waiting for debugger attach...")
        debugpy.wait_for_client()
    trainer = _setup(cfg)
    if cfg.debug:
        trainer.train()
        _teardown(trainer)
    else:
        try:
            trainer.train()
        except Exception as e:
            logger.error(e)
            logger.error(traceback.format_exc())
        finally:
            _teardown(trainer)


if __name__ == "__main__":
    sys.exit(main())
