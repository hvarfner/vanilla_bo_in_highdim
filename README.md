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



