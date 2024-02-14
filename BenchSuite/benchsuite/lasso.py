import numpy as np
import torch

from benchsuite import settings
from benchsuite.benchmark import Benchmark


class LassoBenchmark(Benchmark):

    def __init__(
            self,
            dim: int,
            lb: torch.Tensor,
            ub: torch.Tensor,
    ):
        super().__init__(dim=dim, lb=lb, ub=ub)
        self._b = None

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tensor(self._b.evaluate(x.cpu().numpy().astype(np.double)), device=settings.DEVICE,
                            dtype=settings.DTYPE).unsqueeze(-1)


class LassoDNA(LassoBenchmark):

    def __init__(self):
        super().__init__(dim=180, lb=torch.ones(180, device=settings.DEVICE, dtype=settings.DTYPE) * (-1),
                         ub=torch.ones(180, device=settings.DEVICE, dtype=settings.DTYPE))
        from LassoBench import LassoBench
        self._b: LassoBench.RealBenchmark = LassoBench.RealBenchmark(
            pick_data="dna", mf_opt="discrete_fidelity"
        )


class LassoSimple(LassoBenchmark):

    def __init__(self):
        dim = 60
        super().__init__(dim=dim,
                         lb=torch.ones(dim, device=settings.DEVICE, dtype=settings.DTYPE) * (-1),
                         ub=torch.ones(dim, device=settings.DEVICE, dtype=settings.DTYPE))
        from LassoBench import LassoBench
        self._b: LassoBench.SyntheticBenchmark = LassoBench.SyntheticBenchmark(
            pick_bench="synt_simple"
        )


class LassoMedium(LassoBenchmark):

    def __init__(self):
        dim = 100
        super().__init__(dim=dim,
                         lb=torch.ones(dim, device=settings.DEVICE, dtype=settings.DTYPE) * (-1),
                         ub=torch.ones(dim, device=settings.DEVICE, dtype=settings.DTYPE))
        from LassoBench import LassoBench
        self._b: LassoBench.SyntheticBenchmark = LassoBench.SyntheticBenchmark(
            pick_bench="synt_medium"
        )


class LassoHigh(LassoBenchmark):

    def __init__(self):
        dim = 300
        super().__init__(dim=dim,
                         lb=torch.ones(dim, device=settings.DEVICE, dtype=settings.DTYPE) * (-1),
                         ub=torch.ones(dim, device=settings.DEVICE, dtype=settings.DTYPE))
        from LassoBench import LassoBench
        self._b: LassoBench.SyntheticBenchmark = LassoBench.SyntheticBenchmark(
            pick_bench="synt_high"
        )


class LassoHard(LassoBenchmark):

    def __init__(self):
        dim = 1000
        super().__init__(dim=dim,
                         lb=torch.ones(dim, device=settings.DEVICE, dtype=settings.DTYPE) * (-1),
                         ub=torch.ones(dim, device=settings.DEVICE, dtype=settings.DTYPE))
        from LassoBench import LassoBench
        self._b: LassoBench.SyntheticBenchmark = LassoBench.SyntheticBenchmark(
            pick_bench="synt_hard"
        )
