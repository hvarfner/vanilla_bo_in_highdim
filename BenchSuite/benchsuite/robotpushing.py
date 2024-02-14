import torch
from ebo.test_functions.push_function import PushReward

from benchsuite import settings
from benchsuite.benchmark import Benchmark


class RobotPushingBenchmark(Benchmark):
    def __init__(
        self
    ):
        self._pr = PushReward()
        super().__init__(
            dim=14,
            ub=torch.tensor(self._pr.xmax, dtype=settings.DTYPE, device=settings.DEVICE),
            lb=torch.tensor(self._pr.xmin, dtype=settings.DTYPE, device=settings.DEVICE),
        )

    def __call__(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        if x.ndim == 1:
            x = x.unsqueeze(0)
        rewards = [self._pr(y) for y in x]
        rewards = -torch.tensor(rewards)
        return rewards
