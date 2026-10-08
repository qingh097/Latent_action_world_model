import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from torch.distributed.checkpoint.stateful import Stateful
from torch.utils.data import IterableDataset

import robot_wm.utils.distributed as dist

logger = logging.getLogger(__name__)


class Dataset(ABC, IterableDataset, Stateful):
    def __init__(
        self,
        seed: int = 0,
        infinite: bool = False,
        transform: Optional[Stateful] = None,
    ):
        self._seed = seed
        self._infinite = infinite
        self._transform = transform

        self._start_idx = 0
        self._gen = torch.Generator()

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @abstractmethod
    def _get_length(self) -> int:
        pass

    @abstractmethod
    def _get_sample(self, index: int) -> dict[str, Any]:
        pass

    def __iter__(self):
        self._gen.manual_seed(self._seed)

        # shard dataset across processes
        process_id, num_processes = 0, 1
        if dist.is_initialized():
            process_id = dist.get_global_rank()
            num_processes = dist.get_world_size()

        N = self._get_length()
        total = int(math.ceil(N / num_processes)) * num_processes
        process_indices = torch.randperm(total, generator=self._gen) % N
        process_indices = process_indices[process_id:total:num_processes]

        # shard dataset across workers
        worker_id, num_workers = 0, 1
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        N = len(process_indices)
        total = int(math.ceil(N / num_workers)) * num_workers
        worker_indices = torch.randperm(total, generator=self._gen) % N
        worker_indices = worker_indices[worker_id:total:num_workers]

        # get the subset of `process_indices` for this worker
        final_indices = process_indices[worker_indices]

        # generate worker data
        while True:
            for idx in range(self._start_idx, len(final_indices)):
                self._start_idx += 1
                index = final_indices[idx]
                yield self._get_sample(index)
            if not self._infinite:
                break
            self._start_idx = 0
            logger.log(level=logging.DEBUG, msg=f"Finished {self.name}; restarting")

    def state_dict(self) -> dict[str, Any]:
        output = {
            "_start_idx": self._start_idx,
            "_gen": self._gen.get_state(),
        }
        if self._transform is not None:
            output["_transform"] = self._transform.state_dict()
        return output

    def load_state_dict(self, state_dict: dict[str, Any]):
        self._start_idx = state_dict["_start_idx"]
        self._gen.set_state(state_dict["_gen"])
        if self._transform is not None:
            self._transform.load_state_dict(state_dict["_transform"])


class Transform(ABC, Stateful):
    @abstractmethod
    def __call__(self, episode: dict[str, Any]) -> dict[str, Any]:
        pass
