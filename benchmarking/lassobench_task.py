
from botorch.test_functions.base import BaseTestProblem
import numpy as np
from torch import Tensor
import LassoBench


class LassoRealFunction(BaseTestProblem):

    def __init__(self, pick_data: str, noise_std: float = None, negate: bool = False, seed: int = 42):
        self.seed = seed
        self.benchmark = LassoBench.RealBenchmark(pick_data=pick_data)
        self.dim = self.benchmark.n_features
        self._bounds = [(-1.0, 1.0) for _ in range(self.dim)]
        super().__init__(noise_std=noise_std, negate=negate)
        
    def evaluate_true(self, X: Tensor, seed=None) -> Tensor:
        X_np = X.detach().numpy().flatten().astype(np.float64)
        val = self.benchmark.evaluate(X_np)

        return Tensor([val])
