import abc
from enum import Enum

import torch


class BenchmarkType(Enum):
    CONTINUOUS = 1
    BINARY = 2


class Benchmark(abc.ABC):

    def __init__(
            self,
            dim: int,
            lb: torch.Tensor,
            ub: torch.Tensor,
            type: BenchmarkType = BenchmarkType.CONTINUOUS,
    ):
        self.dim = dim
        self.lb = lb
        self.ub = ub
        self.type = type

    @abc.abstractmethod
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()
