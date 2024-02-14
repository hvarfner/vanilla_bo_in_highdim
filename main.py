import math

from ax.service.utils.instantiation import ObjectiveProperties
from ax.service.ax_client import AxClient
from ax.models.torch.botorch_modular.surrogate import Surrogate
from ax.modelbridge.registry import Models
from ax.modelbridge.generation_strategy import GenerationStep, GenerationStrategy
from omegaconf import DictConfig
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.likelihoods.gaussian_likelihood import (
    GaussianLikelihood,
)

from operator import attrgetter
from gpytorch.priors import GammaPrior
from benchmarking.eval_utils import get_model_hyperparameters
from benchmarking.mappings import (
    get_test_function,
    ACQUISITION_FUNCTIONS,
    INITS
)
from benchmarking.gp_priors import (
    MODELS,
    get_covar_module
)
import numpy as np
import os
from os.path import dirname, abspath, join
import sys
import json
import hydra
import torch
from time import time


@hydra.main(config_path='./configs', config_name='conf')
def main(cfg: DictConfig) -> None:
    torch.manual_seed(int(cfg.seed))
    np.random.seed(int(cfg.seed))
    q = cfg.q
    benchmark = cfg.benchmark.name
    if hasattr(cfg.benchmark, 'outputscale'):
        test_function = get_test_function(
            benchmark, float(cfg.benchmark.noise_std), cfg.seed, float(cfg.benchmark.outputscale))

    else:
        test_function = get_test_function(
            name=cfg.benchmark.benchmark,
            noise_std=float(cfg.benchmark.noise_std),
            seed=cfg.seed,
            bounds=cfg.benchmark.bounds,
        )
    
    if cfg.init == "sqrt":
        factor = cfg.init_factor
        num_init = math.ceil(factor * len(cfg.benchmark.bounds) ** 0.5)

    elif isinstance(cfg.benchmark.num_init, int):
        num_init = max(cfg.benchmark.num_init, cfg.q)

    num_bo = cfg.benchmark.num_iters - num_init

    if cfg.acq.name == 'Sampling':
        num_init = cfg.benchmark.num_iters
        num_bo = 0
    acq_func = ACQUISITION_FUNCTIONS[cfg.acq.acq_func]
    bounds = torch.transpose(torch.Tensor(cfg.benchmark.bounds), 1, 0)

    if hasattr(cfg.acq, 'acq_kwargs'):
        acq_func_kwargs = dict(cfg.acq.acq_kwargs)
    else:
        acq_func_kwargs = {}


    model_kwargs = get_covar_module(cfg.model.model_name, len(
        bounds.T), 
        gp_params=cfg.model.get('gp_params', None),
        gp_constraints=cfg.model.get('gp_constraints', {})
    )

    model_enum = Models.BOTORCH_MODULAR
    init_type = INITS['sobol']
    init_kwargs = {"seed": int(cfg.seed)}
    steps = [
        GenerationStep(
            model=init_type,
            num_trials=num_init,
            model_kwargs=init_kwargs,
        )
    ]
    opt_setup = cfg.acq_opt
    model = MODELS[cfg.model.gp]

    bo_step = GenerationStep(
        model=model_enum,
        num_trials=num_bo,
        model_kwargs={  # Kwargs to pass to `BoTorchModel.__init__`
            "surrogate": Surrogate(
                botorch_model_class=model,
                covar_module_class=model_kwargs["covar_module_class"],
                covar_module_options=model_kwargs["covar_module_options"],
                likelihood_class=model_kwargs["likelihood_class"],
                likelihood_options=model_kwargs["likelihood_options"],
            ),
            "botorch_acqf_class": acq_func,
            "acquisition_options": {**acq_func_kwargs},
        },
        model_gen_kwargs={"model_gen_options": {  # Kwargs to pass to `BoTorchModel.gen`
            "optimizer_kwargs": dict(opt_setup)},
        },
    )
    steps.append(bo_step)

    def evaluate(parameters, seed=None):
        x = torch.tensor(
            [[parameters.get(f"x_{i+1}") for i in range(len(cfg.benchmark.bounds))]])

        if seed is not None:
            bc_eval = test_function.evaluate_true(x, seed=seed).squeeze().tolist()
        else:
            bc_eval = test_function(x).squeeze().tolist()

        return {benchmark: bc_eval}

    gs = GenerationStrategy(
        steps=steps
    )
    # Initialize the client - AxClient offers a convenient API to control the experiment
    ax_client = AxClient(generation_strategy=gs)
    # Setup the experiment
    ax_client.create_experiment(
        name=cfg.experiment_name,
        parameters=[
            {
                "name": f"x_{i+1}",
                "type": "range",
                "bounds": bounds[:, i].tolist(),
            }
            for i in range(len(cfg.benchmark.bounds))
        ],
        objectives={
            benchmark: ObjectiveProperties(minimize=False),
        },
    )
    true_vals = []
    hyperparameters = {}
    bo_times = []

    total_iters = num_init + num_bo
    total_batches = math.ceil((num_init + num_bo) / q)
    current_count = 0

    for i in range(total_batches):

        current_count = (q * i)
        batch_data = []
        q_curr = min(q, total_iters - current_count)
        if current_count >= num_init:
            start_time = time()

        for q_rep in range(q_curr):
            batch_data.append(ax_client.get_next_trial())
        if current_count >= num_init:
            end_time = time()
            bo_times.append(end_time - start_time)
        # Local evaluation here can be replaced with deployment to external system.
        for q_rep in range(q_curr):
            parameters, trial_index = batch_data[q_rep]
            ax_client.complete_trial(
                trial_index=trial_index, raw_data=evaluate(parameters))


        results_df = ax_client.get_trials_data_frame()
        configs = torch.tensor(
            results_df.loc[:, ['x_' in col for col in results_df.columns]].to_numpy())

        if cfg.benchmark.get('synthetic', True):
            for q_idx in range(q_curr):
                true_vals.append(test_function.evaluate_true(
                    configs[-q_curr + q_idx].unsqueeze(0)).item())
            results_df['True Eval'] = true_vals
            if current_count > num_init:
                model = ax_client._generation_strategy.model.model.surrogate.model
                current_data = ax_client.get_trials_data_frame()[benchmark].to_numpy()
                hps = get_model_hyperparameters(model, current_data)
                hyperparameters[f'iter_{i}'] = hps
        savepath = join(dirname(os.path.abspath(__file__)), cfg.result_path)
        os.makedirs(savepath, exist_ok=True)
        with open(f"{savepath}/{ax_client.experiment.name}_hps.json", "w") as f:
            json.dump(hyperparameters, f, indent=2)
        results_df.to_csv(f"{savepath}/{ax_client.experiment.name}.csv")


if __name__ == '__main__':
    main()
