"""Lightweight wrapper for torch.distributed with `torchrun` support."""

import logging
import os

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def _log_distributed_env(prefix: str):
    msg = prefix
    msg += f" {os.environ['RANK'] = }"
    msg += f" {os.environ['LOCAL_RANK'] = }"
    msg += f" {os.environ['WORLD_SIZE'] = }"
    logger.info(msg)


def _set_distributed_env():
    # If already set, don't do anything, this happens when using torchrun
    if (
        "RANK" in os.environ
        and "WORLD_SIZE" in os.environ
        and "LOCAL_RANK" in os.environ
    ):
        return

    # If not, set them up; this happens when using hydra.submitit
    try:
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        os.environ["MASTER_ADDR"] = os.environ["HOSTNAME"]
        os.environ["MASTER_PORT"] = str(23437)

        logger.info(f"{os.environ['WORLD_SIZE']=}, {os.environ['RANK']=}")
        logger.info(f"{os.environ['HOSTNAME']=}")
        logger.info(f"{os.environ['SLURM_LOCALID']=}")

    # Cannot handle this case
    except Exception as e:
        logger.warning("SLURM vars not set (distributed training not available)")
        raise e


def init_process_group(backend: str = "nccl", set_device: bool = True) -> int:
    _set_distributed_env()
    _log_distributed_env("initializing")
    dist.init_process_group(backend=backend)
    if set_device:
        torch.cuda.set_device(get_local_rank())


def destroy_process_group():
    _log_distributed_env("destroying")
    dist.destroy_process_group()


def is_initialized() -> bool:
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    for key in ["RANK", "LOCAL_RANK", "WORLD_SIZE"]:
        if key not in os.environ:
            return False
    return True


def get_local_rank() -> int:
    assert is_initialized()
    return int(os.environ["LOCAL_RANK"])


def get_global_rank() -> int:
    assert is_initialized()
    return int(os.environ["RANK"])


def get_world_size() -> int:
    assert is_initialized()
    return int(os.environ["WORLD_SIZE"])


def barrier(device_ids=None):
    """Synchronize all processes."""
    assert is_initialized()
    dist.barrier(device_ids=device_ids)


def all_reduce(tensor, op=dist.ReduceOp.SUM):
    world_size = get_world_size()

    if world_size == 1:
        return tensor

    dist.all_reduce(tensor, op=op)

    return tensor