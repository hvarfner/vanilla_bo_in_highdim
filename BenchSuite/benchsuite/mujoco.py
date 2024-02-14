import warnings
from typing import Any

import torch

from benchsuite import settings
from benchsuite.benchmark import Benchmark
from benchsuite.utils.mujoco import func_factories


class MujocoBenchmark(Benchmark):

    def __init__(
        self,
        dim: int,
        ub: torch.Tensor,
        lb: torch.Tensor,
        benchmark: Any
    ):
        super().__init__(dim=dim, lb=lb, ub=ub)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.benchmark = benchmark.make_object()

    def __call__(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y = self.benchmark(x)[0]
        return -torch.tensor(y)


class MujocoSwimmer(MujocoBenchmark):
    def __init__(
        self
    ):
        super().__init__(
            dim=16,
            ub=torch.ones(16, dtype=settings.DTYPE, device=settings.DEVICE),
            lb=-1 * torch.ones(16, dtype=settings.DTYPE, device=settings.DEVICE),
            benchmark=func_factories["swimmer"]
        )


class MujocoHumanoid(MujocoBenchmark):
    def __init__(
        self
    ):
        super().__init__(
            dim=6392,
            ub=torch.ones(6392, dtype=settings.DTYPE, device=settings.DEVICE),
            lb=-1 * torch.ones(6392, dtype=settings.DTYPE, device=settings.DEVICE),
            benchmark=func_factories["humanoid"]
        )


class MujocoAnt(MujocoBenchmark):
    def __init__(
        self
    ):
        super().__init__(
            dim=888,
            ub=torch.ones(888, dtype=settings.DTYPE, device=settings.DEVICE),
            lb=-1 * torch.ones(888, dtype=settings.DTYPE, device=settings.DEVICE),
            benchmark=func_factories["ant"]
        )


class MujocoHopper(MujocoBenchmark):
    def __init__(
        self
    ):
        super().__init__(
            dim=33,
            ub=1.4 * torch.ones(33, dtype=settings.DTYPE, device=settings.DEVICE),
            lb=-1.4 * torch.ones(33, dtype=settings.DTYPE, device=settings.DEVICE),
            benchmark=func_factories["hopper"]
        )


class MujocoWalker(MujocoBenchmark):
    def __init__(
        self
    ):
        super().__init__(
            dim=102,
            ub=0.9 * torch.ones(102, dtype=settings.DTYPE, device=settings.DEVICE),
            lb=-1.8 * torch.ones(102, dtype=settings.DTYPE, device=settings.DEVICE),
            benchmark=func_factories["walker_2d"]
        )


class MujocoHalfCheetah(MujocoBenchmark):
    def __init__(
        self
    ):
        super().__init__(
            dim=102,
            ub=torch.ones(102, dtype=settings.DTYPE, device=settings.DEVICE),
            lb=-1.0 * torch.ones(102, dtype=settings.DTYPE, device=settings.DEVICE),
            benchmark=func_factories["half_cheetah"]
        )
