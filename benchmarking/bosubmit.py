""" Submits to slurm using a hydra API.
"""
import argparse
import itertools
import os
from pathlib import Path


def parse_argument_string(args):
    def get_argument_settings(argument):
        """Transforms 'a=1,2,3' to ('a=1', 'a=2', 'a=3')."""
        name, values = argument.split("=")
        if values.startswith("range("):
            range_params = values[len("range(") : -1]
            range_params = (int(param) for param in range_params.split(", "))
            return [f"{name}={value}" for value in range(*range_params)]
        else:
            return [f"{name}={value}" for value in values.split(",")]

    def get_all_argument_settings(arguments):
        """
        Transforms ['a=1,2,3', 'b=1'] to [('a=1', 'b=1'), ('a=2', 'b=1'), ('a=3', 'b=1')]
        """
        return itertools.product(
            *(get_argument_settings(argument) for argument in arguments)
        )

    def get_all_argument_strings(argument_settings):
        """Transforms [('a=1', 'b=1')] to ('a=1 b=1',)"""
        return (" ".join(argument_setting) for argument_setting in argument_settings)

    argument_settings = get_all_argument_settings(args.arguments)
    argument_strings = list(get_all_argument_strings(argument_settings))
    argument_string = "\n".join(argument_strings)
    argument_final_string = f"ARGS=(\n{argument_string}\n)"

    return argument_final_string, len(argument_strings)


def construct_script(args, cluster_oe_dir):
    argument_string, num_tasks = parse_argument_string(args)

    script = list()
    script.append("#!/bin/bash")
    script.append(f"#SBATCH --time {args.time}")
    # script.append(f"#SBATCH -A {args.project}")

    script.append(f'#SBATCH -n 1')
    script.append(f'#SBATCH --array 0-{num_tasks-1}')
    script.append(f'#SBATCH --mem-per-cpu 8000')

    script.append(f"#SBATCH --job-name {args.job_name}")
    script.append(f"#SBATCH --time {args.time}")

    script.append("")
    script.append(argument_string)
    script.append("")
    script.append(
        f"python benchmarking/ax_run.py "
        f"${{ARGS[@]:{len(args.arguments)}*$SLURM_ARRAY_TASK_ID:{len(args.arguments)}}}"
    )

    return "\n".join(script) + "\n", argument_string  # type: ignore[assignment]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_group", default="test")
    parser.add_argument("--project", default="")
    parser.add_argument("--time", default="48:00:00")
    parser.add_argument("--job_name", default="test")
    parser.add_argument("--memory", default=0, type=int)
    parser.add_argument("--arguments", nargs="+")

    args = parser.parse_args()

    experiment_group_dir = Path("results", args.experiment_group)
    cluster_oe_dir = Path(experiment_group_dir, ".cluster_oe")
    scripts_dir = Path(experiment_group_dir, ".submit")

    experiment_group_dir.mkdir(parents=True, exist_ok=True)
    cluster_oe_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir.mkdir(parents=True, exist_ok=True)

    script, runs = construct_script(args, cluster_oe_dir)

    num_scripts = len(list(scripts_dir.glob("*.sh")))
    script_path = Path(scripts_dir, f"{num_scripts}.sh")
    submission_commmand = f"sbatch {script_path}"
    print(f"Running {submission_commmand} with runs:\n\n{runs}")
    if input("Ok? [Y|n] -- ").lower() in {"y", ""}:
        script_path.write_text(script, encoding="utf-8")  # type: ignore[arg-type]
        os.system(submission_commmand)
    else:
        print("Not submitting.")
