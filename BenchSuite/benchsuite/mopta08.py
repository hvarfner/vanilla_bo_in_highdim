import os
import subprocess
import sys
import tempfile
from pathlib import Path
from platform import machine

import torch

from benchsuite import settings
from benchsuite.benchmark import Benchmark


class Mopta08(Benchmark):

    def __init__(self):
        dim = 124
        super().__init__(
            dim=dim,
            lb=torch.zeros(dim, device=settings.DEVICE, dtype=settings.DTYPE),
            ub=torch.ones(dim, device=settings.DEVICE, dtype=settings.DTYPE),
        )

        self.sysarch = 64 if sys.maxsize > 2 ** 32 else 32
        self.machine = machine().lower()

        if self.machine == "armv7l":
            assert self.sysarch == 32, "Not supported"
            self._mopta_exectutable = "mopta08_armhf.bin"
        elif self.machine == "x86_64":
            assert self.sysarch == 64, "Not supported"
            self._mopta_exectutable = "mopta08_elf64.bin"
        elif self.machine == "i386":
            assert self.sysarch == 32, "Not supported"
            self._mopta_exectutable = "mopta08_elf32.bin"
        elif self.machine == "amd64":
            assert self.sysarch == 64, "Not supported"
            self._mopta_exectutable = "mopta08_amd64.exe"
        else:
            raise RuntimeError("Machine with this architecture is not supported")

        self._mopta_exectutable = os.path.join(
            Path(__file__).parent.parent, "data", "mopta08", self._mopta_exectutable
        )
        self.directory_file_descriptor = tempfile.TemporaryDirectory()
        self.directory_name = self.directory_file_descriptor.name

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Evaluate Mopta08 benchmark for one point
        :param x: one input configuration
        :return: value with soft constraints
        """
        x = x.squeeze()
        assert x.ndim == 1
        # write input to file in dir
        with open(os.path.join(self.directory_name, "input.txt"), "w+") as tmp_file:
            for _x in x:
                tmp_file.write(f"{_x.detach().cpu().numpy()}\n")
        # pass directory as working directory to process
        popen = subprocess.Popen(
            self._mopta_exectutable,
            stdout=subprocess.PIPE,
            cwd=self.directory_name,
        )
        popen.wait()
        # read and parse output file
        output = (
            open(os.path.join(self.directory_name, "output.txt"), "r")
            .read()
            .split("\n")
        )
        output = [x.strip() for x in output]
        output = torch.tensor([float(x) for x in output if len(x) > 0], dtype=settings.DTYPE, device=settings.DEVICE)
        value = output[0]
        constraints = output[1:]
        # see https://arxiv.org/pdf/2103.00349.pdf E.7
        return (value + 10 * torch.sum(torch.clip(constraints, min=0, max=None))).unsqueeze(-1)
