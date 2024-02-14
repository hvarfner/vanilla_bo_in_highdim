import argparse
import os
from pathlib import Path

dir = Path(__file__).parent.absolute()  # noqa
os.environ["LD_LIBRARY_PATH"] = f"{dir}/data/mujoco210/bin:/usr/lib/nvidia"  # noqa
os.environ["MUJOCO_PY_MUJOCO_PATH"] = f"{dir}/data/mujoco210"  # noqa

import torch

from benchsuite import settings
from benchsuite.benchmarks import benchmark_options

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='BenchSuite',
        description='Provides some benchmarks',
        epilog='Enjoy the program!'
    )
    parser.add_argument(
        "--name",
        help=f"Name of the benchmark. Options: {list(benchmark_options.keys())}",
        type=str,
        required=True
    )
    parser.add_argument("--x", "-x", help="The point", type=float, nargs="+", required=True)
    args = parser.parse_args()

    if args.name not in benchmark_options:
        raise RuntimeError(f"Unknown benchmark {args.name}")
    else:
        bench = benchmark_options[args.name]()

    x = torch.tensor(args.x, dtype=settings.DTYPE, device=settings.DEVICE)
    # scale x to the correct range
    x = bench.lb + (bench.ub - bench.lb) * x
    
    # set LIBSVMDATA_HOME /tmp/libsvmdata
    os.environ["LIBSVMDATA_HOME"] = "/tmp/libsvmdata"

    y = bench(x)
    print(y.detach().cpu().numpy().tolist()[0])
