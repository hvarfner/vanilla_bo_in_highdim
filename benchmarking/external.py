import os
from os.path import dirname

import torch
from torch import Tensor
from botorch.test_functions import SyntheticTestFunction
from benchmarking.mappings import get_test_function
import pandas as pd


class RecordedTrajectory(SyntheticTestFunction):

    def __init__(
        self, 
        function: SyntheticTestFunction, 
        function_name: str, 
        method_name: str, 
        experiment_name: str, 
        seed: int,
    ) -> None:
        self._bounds = function._bounds
        self.dim = function.dim
        super().__init__(noise_std=function.noise_std, negate=function.negate)
        self.function = function
        df_columns = [function_name, 'True Eval']
        df_columns.extend([f'x_{i}' for i in range(self.function.dim)])
        self.data = {col: [] for col in df_columns}
        self.function_name = function_name
        self.save_path  = f'{experiment_name}/{function_name}/{method_name}/{method_name}_run_{seed}.csv'


    def evaluate_true(self, X: Tensor) -> Tensor:
        assert X.ndim == 2, f'X does not have the expected number of dimensions: {X.ndim}'
        for i in range(self.function.dim):
            self.data[f'x_{i}'].append(X[:, i].item()) 
        res = self.function.evaluate_true(X)
        self.data['True Eval'].append(res.item())
        return res

    def __call__(self, X: Tensor) -> Tensor:

        noisy_res = super().__call__(X)
        self.data[self.function_name].append(noisy_res.item())
        self.save()
        return noisy_res

    def save(self):
        trajectory = pd.DataFrame(self.data)
        os.makedirs(dirname(self.save_path), exist_ok=True)
        trajectory.to_csv(self.save_path)
