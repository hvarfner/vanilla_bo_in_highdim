import os
from pathlib import Path

import numpy as np
import torch
from pysat.formula import WCNF

from benchsuite import settings
from benchsuite.benchmark import Benchmark, BenchmarkType


class MaxSat60(Benchmark):

    def __init__(self):
        dim = 60
        super().__init__(dim=dim, lb=torch.zeros(dim, device=settings.DEVICE, dtype=settings.DTYPE),
                         ub=torch.ones(dim, device=settings.DEVICE, dtype=settings.DTYPE), type=BenchmarkType.BINARY)
        wcnf = WCNF(from_file=os.path.join(Path(__file__).parent.parent, "data", "maxsat", "frb-frb10-6-4.wcnf"))
        weights = np.array(wcnf.wght)
        # normalize weights to unit standard deviation and zero mean
        self.weights = (weights - weights.mean()) / weights.std()
        self.clauses = [np.array(c) for c in wcnf.soft]
        self.dim = wcnf.nv
        assert self.dim == 60, "Something is really off here"
        self._arange = np.arange(self.dim)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.detach().cpu().numpy().squeeze()
        assert x.ndim == 1

        x = x.astype(np.bool)

        print("x", x)

        trues = self._arange[x] + 1
        falses = -(self._arange[~x] + 1)

        assignments = np.concatenate([trues, falses])

        weights_sum = 0

        print(assignments)

        for i, soft in enumerate(self.clauses):
            if np.any(np.isin(assignments, soft)):
                weights_sum += self.weights[i]
        return -torch.tensor(weights_sum, device=settings.DEVICE, dtype=settings.DTYPE).unsqueeze(-1)
