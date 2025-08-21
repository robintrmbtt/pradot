import itertools
import os
from argparse import ArgumentParser

def get_parameters(args):
    sweep_parameters = {'+task':[args.task]}
    for arg in vars(args):
        if arg in ['task', 'qos', 'max_time']:
            continue

        arg_value = getattr(args, arg)
        if not isinstance(arg_value, list):
            arg_value = [arg_value]
        sweep_parameters[arg] = arg_value
    return sweep_parameters


def submit_job(command, qos='t3', max_time=None):
    if max_time is None:
        if qos=='t3':
            max_time='20:00:00'
        elif qos=='t4':
            max_time='100:00:00'
        elif qos=='dev':
            max_time='2:00:00'
    else:
        max_time = f'{max_time}:00:00'

    sbatch_script = f"""#!/bin/bash
#SBATCH --account=[account]
#SBATCH --job-name=pradot
#SBATCH --chdir=/path/to/logs
#SBATCH --qos=qos_gpu-{qos}
#SBATCH --constraint v100-32g
#SBATCH --output=_%x_%j.out
#SBATCH --error=_%x_%j.err
#SBATCH --gres=gpu:1
#SBATCH --time={max_time}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --hint=nomultithread

module purge
module load pytorch-gpu/py3/2.2.0
set -x

cd /path/to/working/dir

{command}
"""
    script_filename = "tmp_job.sh"
    with open(script_filename, "w") as f:
        f.write(sbatch_script)

    os.system(f"sbatch {script_filename}")
    os.remove("tmp_job.sh")

if __name__ == "__main__":
    aparser = ArgumentParser()
    aparser.add_argument('--task', required=True, type=str)
    aparser.add_argument('--obj', required=True, nargs='+', type=str)
    aparser.add_argument('--alpha', required=True, nargs='+', type=str)
    aparser.add_argument('--qos', required=False, choices=['t3','t4','dev'], default='t3')
    aparser.add_argument('--max_time', required=False, default=None, type=int)

    args = aparser.parse_args()

    base_command = "srun python pradot/train.py"

    sweep_parameters = get_parameters(args)

    keys = list(sweep_parameters.keys())
    values = list(sweep_parameters.values())
    combinations = list(itertools.product(*values))

    for i, c in enumerate(combinations):
        command = " ".join([f"+{key}={val}" for key, val in zip(keys, c)])
        full_command = f"{base_command} {command}"
        print(full_command)
        submit_job(full_command, qos=args.qos, max_time=args.max_time)