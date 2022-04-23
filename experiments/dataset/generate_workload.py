"""
   scripts is used to generate
   initial dataset for the experiments
   it uses functions implemented in
   the gym_edgesimulator.dataset module to
   generate a dataset with given specs
"""
import os
import sys
import pickle
import json
import click
from typing import List
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

from smart_scheduler.dataset import WorkloadGenerator

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))

from experiments.utils.constants import (
    DATASETS_PATH,
    CONFIGS_PATH
)

from experiments.utils import config_workload_generation_check

# generaing the workloads
def generate_workload(notes: str, dataset_id: int,
                      workloads_var: List[List], timesteps: int,
                      services_types: int, plot_smoothing: int,
                      seed: int):
    """
        generate a random workload
    """
    # dataset path for the dataset
    dataset_path = os.path.join(DATASETS_PATH, str(dataset_id))

    # read the dataset and start workload
    try:
        with open(os.path.join(dataset_path, 'dataset.pickle'), 'rb')\
            as in_pickle:
            dataset = pickle.load(in_pickle)
    except:
        raise FileNotFoundError(f"dataset <{dataset_id}> does not exist")
    dataset_info_path = os.path.join(dataset_path, 'info.json')
    with open(dataset_info_path) as cf:
        start_workloads = json.loads(cf.read())['start_workload']

    # fix foldering per datast
    workload_path = os.path.join(dataset_path, 'workloads')
    content = os.listdir(workload_path)
    new_workload = len(content)
    dir2save = os.path.join(workload_path, str(new_workload))
    os.mkdir(dir2save)

    # generate the workload
    workload_generator = WorkloadGenerator(dataset=dataset,
                                           workloads_var=workloads_var,
                                           timesteps=timesteps,
                                           services_types=services_types,
                                           start_workloads=start_workloads,
                                           plot_smoothing=plot_smoothing,
                                           seed=seed)
    workloads, figs = workload_generator.make_workloads()

    # information of the generated workload
    info = {
        'notes': notes,
        'dataest_id': dataset_id,
        'timesteps': timesteps,
        'services_types': services_types,
        'workload_var': workloads_var,
        'plot_smoothing': plot_smoothing,
        'seed': seed
    }

    # save the information and workload in the folder
    with open(os.path.join(dir2save, 'info.json'), 'x') as out_file:
        json.dump(info, out_file, indent=4)
    with open(os.path.join(dir2save, 'workload.pickle'), 'wb') as out_pickle:
        pickle.dump(workloads, out_pickle)
    print(f"\n\nGenerated data saved in <{dir2save}>\n\n")

    # save figs
    figures_dir = os.path.join(dir2save, 'figures')
    os.mkdir(figures_dir)
    for i, fig in enumerate(figs):
        fig.savefig(os.path.join(figures_dir, f'services_type_{i}.png'))


def main():
    # read the config file
    config_file_path = os.path.join(
        CONFIGS_PATH,
        'generation-configs',
        'workload.json')
    with open(config_file_path) as cf:
        config = json.loads(cf.read())
    config_workload_generation_check(config=config)
    generate_workload(**config)


if __name__ == "__main__":
    main()
