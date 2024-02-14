import numpy as np
import torch

from benchsuite.benchmark import Benchmark
from benchsuite.utils.contamination import sample_init_points, generate_contamination_dynamics, _contamination


class Contamination(Benchmark):
    """
    Contamination Control Problem with the simplest graph
    """

    def __init__(self):
        super().__init__(25, torch.zeros(25), torch.ones(25))
        self.lamda = 1e-2  # 1e-4 1e-2
        self.n_vertices = np.array([2] * 25)
        self.suggested_init = torch.empty(0).long()
        self.suggested_init = torch.cat([self.suggested_init,
                                         sample_init_points(self.n_vertices, 20 - self.suggested_init.size(0),
                                                            random_seed=42)], dim=0)
        self.adjacency_mat = []
        self.fourier_freq = []
        self.fourier_basis = []
        for i in range(len(self.n_vertices)):
            n_v = self.n_vertices[i]
            adjmat = torch.diag(torch.ones(n_v - 1), -1) + torch.diag(torch.ones(n_v - 1), 1)
            self.adjacency_mat.append(adjmat)
            laplacian = torch.diag(torch.sum(adjmat, dim=0)) - adjmat
            eigval, eigvec = torch.symeig(laplacian, eigenvectors=True)
            self.fourier_freq.append(eigval)
            self.fourier_basis.append(eigvec)
        # In all evaluation, the same sampled values are used.
        self.init_Z, self.lambdas, self.gammas = generate_contamination_dynamics(random_seed=42)

    def __call__(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        assert x.size(1) == len(self.n_vertices)
        return torch.cat([self._evaluate_single(x[i]) for i in range(x.size(0))], dim=0).to(dtype=torch.double)

    def _evaluate_single(self, x):
        assert x.dim() == 1
        assert x.numel() == len(self.n_vertices)
        if x.dim() == 2:
            x = x.squeeze(0)
        evaluation = _contamination(x=(x.cpu() if x.is_cuda else x).numpy(), cost=np.ones(x.numel()),
                                    init_Z=self.init_Z, lambdas=self.lambdas, gammas=self.gammas, U=0.1, epsilon=0.05)
        evaluation += self.lamda * float(torch.sum(x))
        return evaluation * x.new_ones((1,)).float()
