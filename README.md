##### NOTE: As of [BoTorch 0.12.0](https://github.com/pytorch/botorch/releases/tag/v0.12.0), the dimension-scaled prior is the default. As such, the priors used in the paper can be used by running a standard BoTorch SingleTaskGP, with no additional changes to either the `covar_module` or the `likelihood`. See [this post](https://github.com/pytorch/botorch/discussions/2451) for more information. So, if you simply want to run the priors from the paper for your high-dimensional problems, you can just use BoTorch as usual. Just don't forget to cite the paper if you do. =)

```
@InProceedings{pmlr-v235-hvarfner24a,
  title = 	 {Vanilla {B}ayesian Optimization Performs Great in High Dimensions},
  author =       {Hvarfner, Carl and Hellsten, Erik Orm and Nardi, Luigi},
  booktitle = 	 {Proceedings of the 41st International Conference on Machine Learning},
  pages = 	 {20793--20817},
  year = 	 {2024},
  editor = 	 {Salakhutdinov, Ruslan and Kolter, Zico and Heller, Katherine and Weller, Adrian and Oliver, Nuria and Scarlett, Jonathan and Berkenkamp, Felix},
  volume = 	 {235},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {21--27 Jul},
  publisher =    {PMLR},
  url = 	 {https://proceedings.mlr.press/v235/hvarfner24a.html},
}
```

# Official repository for "Vanilla Bayesian Optimization Performs Great in High Dimensions"


To run any experiment in the paper:

```pip install ax-platform hydra-core```

Then, any of the synthetic benchmarks are runnable.

### Basic Usage
Run Vanilla BO (with the proposed lengthscale scaling) by entering

```python main.py```

this will run 25D-embedded Levy4 by default. Change the benchmark by adding 
```benchmark=**some_benchmark_name**```. The complete list of synthetic benchmarks is

```levy4_25, levy4_100, levy4_300, levy4_1000, hartmann6_25, hartmann6_100, hartmann6_300, hartmann6_1000```

and the real ones (which require some more installation) are
```mopta, lasso_dna, svm, ant, humanoid```.

Moreover, there are 4 different model variants```model=**model_name**```, reflecting the models used in the paper including the appendix. The specification for each can be found in ```benchmarking/gp_priors```:
```default, with_ops, rbf_mle, gamma_3_6```.

Changing the prior of the models can be done by entering the arguments:
```model.gp_params.ls.loc=2``` or ```model.gp_params.ls.scale=5``` to set $\mu_0=2$ and $\sigma_0 = 5$. If nothing is entered, the default paper settings are run.

Finally, a complete example:
```python main.py model=default benchmark=mopta model.gp_params.ls.loc=2 seed=13 experiment_group=vanilla_bo_testrun algorithm=qlognei```

Then, the results will land in ```results/vanilla_bo_testrun/mopta/qLogNEI/mopta_qLogNEI_run13.csv```.

Other modifiable parameters can be found by checking the various options in ```/configs```, such as ```benchmark.num_iters```, ```benchmark.noise_std```, ```acq_opt.raw_samples```, and ```acq_opt.num_restarts```.


**If no options are specified, the default algorithm (used in the main results in the paper) is run, with the default number of iterations, acquisition budget and noise level.**

### Running real-world tasks
#### Lasso-DNA
To run Lasso-DNA, clone [LassoBench](https://github.com/ksehic/LassoBench) and add it to PYTHONPATH. Run the DNA task by entering 
```benchmark=lasso_dna```

#### MOPTA & SVM
To run MOPTA and SVM, add [BenchSuite](https://arxiv.org/abs/2304.11468) (modified version included in the repo) to PYTHONPATH:
```export PYTHONPATH=${PYTHONPATH}:$PWD/BenchSuite```

 Run either benchmark as
```benchmark=mopta``` or ```benchmark=svm```.

#### MuJoCo Ant & Humanoid
Build the ```recipes/mujoco_container``` with Singularity, and add the container path:

```
sudo singularity build containers/mujoco recipes/mujoco_container
export MUJOCO=${PWD}/containers/mujoco     
```


Then, run either benchmark as:
```benchmark=ant``` or ```benchmark=humanoid```



