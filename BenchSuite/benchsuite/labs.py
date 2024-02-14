import torch

from benchsuite import settings
from benchsuite.benchmark import Benchmark, BenchmarkType


class Labs(Benchmark):

    def __init__(self):
        dim = 50
        super().__init__(dim=dim, lb=(-1) * torch.ones(dim, device=settings.DEVICE, dtype=settings.DTYPE),
                         ub=torch.ones(dim, device=settings.DEVICE, dtype=settings.DTYPE), type=BenchmarkType.BINARY)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 1

        n2 = self.dim ** 2

        e = 0
        for k in range(1, self.dim):
            _e = 0
            for i in range(self.dim - k):
                _e += x[i] * x[i + k]
            e += _e ** 2
        return -torch.tensor(n2 / e, device=settings.DEVICE, dtype=settings.DTYPE).unsqueeze(-1)
