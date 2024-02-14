import torch
from gpytorch.kernels import ScaleKernel
from gpytorch.means import ConstantMean


def get_model_hyperparameters(model, current_data, scale_hyperparameters=True):
    has_outputscale = isinstance(model.covar_module, ScaleKernel)
    has_mean = isinstance(model.mean_module, ConstantMean)

    def tolist(l): return l.detach().to(torch.float32).numpy().tolist()
    hp_dict = {}
    data_mean = current_data.mean()

    if scale_hyperparameters:
        data_variance = current_data.var()
    # print('Data variance', data_variance)
    else:
        data_variance = torch.Tensor([1])

    if has_outputscale:
        hp_dict['outputscale'] = tolist(model.covar_module.outputscale * data_variance)
        hp_dict['lengthscales'] = tolist(model.covar_module.base_kernel.lengthscale)
        hp_dict['noise'] = tolist(model.likelihood.noise * data_variance)
    else:
        hp_dict['lengthscales'] = tolist(model.covar_module.lengthscale)
        hp_dict['noise'] = tolist(model.likelihood.noise)

    if has_mean:
        hp_dict['mean'] = tolist(model.mean_module.constant
                                 * data_variance ** 0.5 - data_mean)

    return hp_dict