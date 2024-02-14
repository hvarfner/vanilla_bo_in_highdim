from typing import List
import math
import torch
import numpy as np
from torch import Tensor
from botorch.utils.transforms import unnormalize

from botorch.test_functions.synthetic import SyntheticTestFunction


class Embedded(SyntheticTestFunction):

    def __init__(
        self,
        function: SyntheticTestFunction,
        dim=2,
        noise_std: float = 0.0,
        negate: bool = False,
        bounds: Tensor = None,
    ) -> None:
        r"""
        Args:
            dim: The (input) dimension.
            noise_std: Standard deviation of the observation noise.
            negate: If True, negate the function.
            bounds: Custom bounds for the function specified as (lower, upper) pairs.
        """
        assert dim >= function.dim, 'The effective function dimensionality is larger than the embedding dimension.'
        self.dim = dim
        self._bounds = [(0.0, 1.0) for _ in range(self.dim)]
        self._bounds[0: function.dim] = function._bounds
        super().__init__(noise_std=noise_std, negate=negate, bounds=bounds)
        self.register_buffer(
            "i", torch.tensor(tuple(range(1, self.dim + 1)), dtype=torch.float)
        )
        self.embedded_function = function

    def evaluate_true(self, X: Tensor) -> Tensor:

        embedded_X = X[:, 0: self.embedded_function.dim]
        return self.embedded_function.evaluate_true(embedded_X)
