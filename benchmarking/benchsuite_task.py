import torch
from torch import Tensor

from benchsuite.mopta08 import Mopta08
from benchsuite.svm import SVM

from botorch.test_functions.synthetic import SyntheticTestFunction

BENCHMARKS = {
    'mopta': Mopta08,
    'svm': SVM,

}

class BenchSuiteFunction(SyntheticTestFunction):
    def __init__(
        self,
        noise_std: float = 0,
        negate: bool = True,
        task_id: 'str' = None,
    ) -> None:
        self.task_id = task_id
        self.f = BENCHMARKS[task_id]()
        self.dim = self.f.dim
        self._bounds = torch.cat((self.f.ub.unsqueeze(0), self.f.lb.unsqueeze(0)), dim=0).T
        super().__init__(noise_std=noise_std, negate=negate, bounds=self._bounds)

    def evaluate_true(self, X: Tensor) -> Tensor:
        if self.task_id == 'svm':
            X = X.flatten()
        return self.f(X)