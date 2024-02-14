import gzip
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVR

from benchsuite import settings
from benchsuite.benchmark import Benchmark


class SVM(Benchmark):
    """
    The interior benchmark is just the benchmark in the lower-dimensional effective embedding.
    """

    def __init__(
            self,
    ):
        dim = 388
        super().__init__(
            dim=dim,
            lb=torch.zeros(dim, device=settings.DEVICE, dtype=settings.DTYPE),
            ub=torch.ones(dim, device=settings.DEVICE, dtype=settings.DTYPE),
        )
        self.X, self.y = self._load_data()
        np.random.seed(388)
        idxs = np.random.choice(np.arange(len(self.X)), min(500, len(self.X)), replace=False)
        half = len(idxs) // 2
        self._X_train = self.X[idxs[:half]]
        self._X_test = self.X[idxs[half:]]
        self._y_train = self.y[idxs[:half]]
        self._y_test = self.y[idxs[half:]]

    def _load_data(self):
        then = time.time()
        data_folder = os.path.join(Path(__file__).parent.parent, "data", "svm")
        try:
            X = np.load(os.path.join(data_folder, "CT_slice_X.npy"))
            y = np.load(os.path.join(data_folder, "CT_slice_y.npy"))
        except:
            fx = gzip.GzipFile(os.path.join(data_folder, "CT_slice_X.npy.gz"), "r")
            fy = gzip.GzipFile(os.path.join(data_folder, "CT_slice_y.npy.gz"), "r")
            X = np.load(fx)
            y = np.load(fy)
            fx.close()
            fy.close()
        X = MinMaxScaler().fit_transform(X)
        y = MinMaxScaler().fit_transform(y.reshape(-1, 1)).squeeze()
        now = time.time()
        print(f"Loaded data in {now - then:.2f} seconds")
        return X, y

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        y = x.numpy()
        C = 0.01 * (500 ** y[387])
        gamma = 0.1 * (30 ** y[386])
        epsilon = 0.01 * (100 ** y[385])
        length_scales = np.exp(4 * y[:385] - 2)

        svr = SVR(gamma=gamma, epsilon=epsilon, C=C, cache_size=1500, tol=0.001)
        svr.fit(self._X_train / length_scales, self._y_train)
        pred = svr.predict(self._X_test / length_scales)
        error = np.sqrt(np.mean(np.square(pred - self._y_test)))

        res = torch.tensor(error, dtype=settings.DTYPE).unsqueeze(-1)
        return res
