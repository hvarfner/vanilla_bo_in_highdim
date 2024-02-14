import numpy as np
import torch
from torch import Tensor

import subprocess
import os

from botorch.test_functions.synthetic import SyntheticTestFunction

class MujocoFunction(SyntheticTestFunction):
    def __init__(
        self,
        bounds: list,
        noise_std: float = 0,
        negate: bool = True,
        container: 'str' = None,
        task_id: 'str' = None,

    ) -> None:
        self.task_id = task_id
        self.dim = len(bounds)
        self._bounds = torch.Tensor(bounds)
        
        self.container = os.environ[f"{container}".upper()]
        self.benchmark = task_id

        self.ARG_LIST = [self.container, '--benchmark_name', self.benchmark, '--x']

        super().__init__(noise_std=noise_std, negate=negate, bounds=self._bounds)


    def evaluate_true(self, X: Tensor) -> Tensor:
        x_str = [np.format_float_positional(x.to(torch.float16).item()) + '0' for x in X.flatten()] 
        result = subprocess.run(self.ARG_LIST + x_str, capture_output=True, text=True, check=False)
        return Tensor([-float(result.stdout)])
