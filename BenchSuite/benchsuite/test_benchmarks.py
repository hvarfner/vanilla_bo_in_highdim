import pytest
import torch

from benchsuite import settings
from benchsuite.benchmarks import benchmark_options


def dict_parametrize(
    data,
    **kwargs
):
    args = list(list(data.values())[0].keys())
    formatted_data = [[item[a] for a in args] for item in data.values()]
    ids = list(data.keys())
    return pytest.mark.parametrize(args, formatted_data, ids=ids, **kwargs)


class TestAllBenchmarks:
    @dict_parametrize(
        {
            f"test{k}": {'benchmark': v} for k, v in benchmark_options.items()
        }
    )
    def test_benchmarks(
        self,
        benchmark,
    ):
        benchmark = benchmark()
        x = torch.rand(benchmark.dim, dtype=settings.DTYPE, device=settings.DEVICE)
        y = benchmark(x)
