import math

from typing import Optional, Dict
import torch
from torch import Tensor

from gpytorch.priors.torch_priors import GammaPrior
from botorch.models import (
    FixedNoiseGP,
    SingleTaskGP,
    SingleTaskVariationalGP,
)
from gpytorch.means import (
    ConstantMean,
)

from gpytorch.kernels import (
    ScaleKernel,
    MaternKernel,
    RBFKernel
)
from gpytorch.priors import (
    NormalPrior,
    GammaPrior,
    LogNormalPrior
)
from gpytorch.constraints.constraints import GreaterThan
from gpytorch.likelihoods import GaussianLikelihood

MODELS = {
    'FixedNoiseGP': FixedNoiseGP,
    'SingleTaskGP': SingleTaskGP,
}
DIM_SCALING = {
               "default": (0.5, 0), # scaling factor of mu and sigma in the dim
               "with_ops": (0.5, 0),
               }



def parse_hyperparameters(gp_params: Dict[str, float], dims: int, dim_scaling: float = None):
    ls_params = gp_params.get('ls', {})
    ops_params = gp_params.get('ops', {})
    noise_params = gp_params.get('noise', {})
    if dim_scaling is not None:
        ls_params['loc'] = ls_params['loc'] + math.log(dims) * dim_scaling[0]
        # toyed with scaling the scale parameter as well
        ls_params['scale'] = (ls_params['scale'] ** 2 + math.log(dims) * dim_scaling[1]) **0.5 # Since it's std and not var, we divide by 2


    return ls_params, ops_params, noise_params


def parse_constraints(gp_constraints):
    ls_constraint = gp_constraints.get('ls', 1e-4)
    scale_constraint = gp_constraints.get('scale', 1e-4)
    noise_constraint = gp_constraints.get('noise', 1e-4)

    return ls_constraint, scale_constraint, noise_constraint


def get_covar_module(model_name, dims, gp_params: Dict = None, gp_constraints: Dict = {}):

    ls_params, ops_params, noise_params = parse_hyperparameters(
        gp_params, dims, dim_scaling=(DIM_SCALING.get(model_name)))
    ls_constraint, scale_constraint, noise_constraint = parse_constraints(
        gp_constraints)

    COVAR_MODULES = {
        'gamma_3_6':
        {
            'covar_module_class': None,
            'covar_module_options': None,
            'likelihood_class': None,
            'likelihood_options': None,
        },
        
        'rbf_mle':
        {
            'covar_module_class': RBFKernel,
            'covar_module_options': dict(
                    ard_num_dims=dims,
                    lengthscale_prior=None,
            ),
            'likelihood_class': GaussianLikelihood,
            'likelihood_options': dict(
                noise_prior=None
            ),
        },        
        
        'default':
        {
            'covar_module_class': RBFKernel,
            'covar_module_options': dict(
                    ard_num_dims=dims,
                    lengthscale_prior=LogNormalPrior(**ls_params),
                    lengthscale_constraint=GreaterThan(ls_constraint)
            ),
            'likelihood_class': GaussianLikelihood,
            'likelihood_options': dict(
                noise_prior=LogNormalPrior(**noise_params),
                noise_constraint=GreaterThan(noise_constraint)
            ),
        },
        
        'with_ops':
        {
            'covar_module_class': ScaleKernel,
            'covar_module_options': dict(
                base_kernel=RBFKernel(
                    ard_num_dims=dims,
                    lengthscale_prior=LogNormalPrior(**ls_params),
                    lengthscale_constraint=GreaterThan(ls_constraint)
                ),
                outputscale_prior=GammaPrior(2, 0.15),
                outputscale_constraint=GreaterThan(scale_constraint)
            ),
            'likelihood_class': GaussianLikelihood,
            'likelihood_options': dict(
                noise_prior=LogNormalPrior(**noise_params),
                noise_constraint=GreaterThan(noise_constraint)
            ),
        },
        
    }
    return COVAR_MODULES[model_name]
