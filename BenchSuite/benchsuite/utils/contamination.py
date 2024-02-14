import numpy as np
import torch


def generate_contamination_dynamics(random_seed=None):
    n_stages = 25
    n_simulations = 100

    init_alpha = 1.0
    init_beta = 30.0
    contam_alpha = 1.0
    contam_beta = 17.0 / 3.0
    restore_alpha = 1.0
    restore_beta = 3.0 / 7.0
    init_Z = np.random.RandomState(random_seed).beta(init_alpha, init_beta, size=(n_simulations,))
    lambdas = np.random.RandomState(random_seed).beta(contam_alpha, contam_beta, size=(n_stages, n_simulations))
    gammas = np.random.RandomState(random_seed).beta(restore_alpha, restore_beta, size=(n_stages, n_simulations))

    return init_Z, lambdas, gammas


def sample_init_points(n_vertices, n_points, random_seed=None):
    """
    :param n_vertices: 1D array
    :param n_points:
    :param random_seed:
    :return:
    """
    if random_seed is not None:
        rng_state = torch.get_rng_state()
        torch.manual_seed(random_seed)
    init_points = torch.empty(0).long()
    for _ in range(n_points):
        init_points = torch.cat(
            [init_points, torch.cat([torch.randint(0, int(elm), (1, 1)) for elm in n_vertices], dim=1)], dim=0)
    if random_seed is not None:
        torch.set_rng_state(rng_state)
    return init_points


def _contamination(x, cost, init_Z, lambdas, gammas, U, epsilon):
    assert x.size == 25

    rho = 1.0
    n_simulations = 100

    Z = np.zeros((x.size, n_simulations))
    Z[0] = lambdas[0] * (1.0 - x[0]) * (1.0 - init_Z) + (1.0 - gammas[0] * x[0]) * init_Z
    for i in range(1, 25):
        Z[i] = lambdas[i] * (1.0 - x[i]) * (1.0 - Z[i - 1]) + (1.0 - gammas[i] * x[i]) * Z[i - 1]

    below_threshold = Z < U
    constraints = np.mean(below_threshold, axis=1) - (1.0 - epsilon)

    return np.sum(x * cost - rho * constraints)
