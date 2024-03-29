"""
   scripts is used to generate
   initial cluster for the experiments
   it uses functions implemented in
   the gym_edgesimulator.cluster module to
   generate a cluster with given specs
"""
import os
import sys
import pickle
import json
import click
from typing import List
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

from smart_scheduler.cluster_generator import (
    WorkloadGeneratorRandom,
    WorkloadGeneratorArabesque,
    WorkloadGeneratorAlibaba
)

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))

from experiments.utils.constants import (
    CLUSTERS_PATH,
    CONFIGS_PATH,
    ARABESQUE_PATH,
    ALIBABA_PATH
)

from experiments.utils import config_workload_generation_check

# generaing the workloads
# def generate_workload(notes: str, cluster_id: int,
#                       workloads_var: List[List], timesteps: int,
#                       services_types: int, plot_smoothing: int,
#                       workload_type: str, seed: int):
def generate_workload(notes: str, cluster_id: int,
                      workload_type: str, seed: int,
                      plot_smoothing: int,
                      **kwargs):
    """
        generate a worload random or dataset
    """
    # cluster path for the cluster
    cluster_path = os.path.join(CLUSTERS_PATH, str(cluster_id))

    # read the cluster and start workload
    try:
        with open(os.path.join(cluster_path, 'cluster.pickle'), 'rb')\
            as in_pickle:
            cluster = pickle.load(in_pickle)
    except:
        raise FileNotFoundError(f"cluster <{cluster_id}> does not exist")
    cluster_info_path = os.path.join(cluster_path, 'info.json')
    with open(cluster_info_path) as cf:
        start_workloads = json.loads(cf.read())['start_workload']

    # fix foldering per datast
    workload_path = os.path.join(cluster_path, 'workloads')
    content = os.listdir(workload_path)
    new_workload = len(content)
    dir2save = os.path.join(workload_path, str(new_workload))
    os.mkdir(dir2save)

    if workload_type == 'random':
        # generate the workload
        workload_generator = WorkloadGeneratorRandom(
            cluster=cluster,
            workloads_var=kwargs['workloads_var'],
            timesteps=kwargs['timesteps'],
            num_services_types=kwargs['services_types'],
            start_workloads=start_workloads,
            plot_smoothing=plot_smoothing,
            seed=seed)
        num_services = len(cluster['services_types'])
        workloads, figs = workload_generator.make_workloads()
        # information of the generated workload
    elif workload_type == 'arabesque':
        # generate the workload
        workload_generator = WorkloadGeneratorArabesque(
            cluster=cluster,
            min_timesteps=kwargs['min_timesteps'],
            dataset_path=ARABESQUE_PATH,
            num_services=kwargs['num_services'],
            plot_smoothing=plot_smoothing,
            seed=seed)
        num_services = kwargs['num_services']
        # workloads, figs = workload_generator.make_workloads()
        workloads = workload_generator.make_workloads()
    elif workload_type == 'alibaba':
        b = 1
    info = {
        'notes': notes,
        'dataest_id': cluster_id,
        'plot_smoothing': plot_smoothing,
        'workload_type': workload_type,
        'num_services': num_services,
        'seed': seed
    }
    workloads_save = {
        'workload_type': workload_type,
        'workloads': workloads
    }

    # save the information and workload in the folder
    with open(os.path.join(dir2save, 'info.json'), 'x') as out_file:
        json.dump(info, out_file, indent=4)
    with open(os.path.join(dir2save, 'workload.pickle'), 'wb') as out_pickle:
        pickle.dump(workloads_save, out_pickle)
    print(f"\n\nGenerated data saved in <{dir2save}>\n\n")

    # save figs
    # figures_dir = os.path.join(dir2save, 'figures')
    # os.mkdir(figures_dir)
    # for i, fig in enumerate(figs):
    #     fig.savefig(os.path.join(figures_dir, f'services_type_{i}.png'))

@click.command()
@click.option('--workload-type',
              type=click.Choice(
                  ['random', 'arabesque', 'alibaba']), default='random')
def main(workload_type: str):
    # read the config file
    config_file_path = os.path.join(
        CONFIGS_PATH,
        'generation-configs',
        f'workload_{workload_type}.json')
    with open(config_file_path) as cf:
        config = json.loads(cf.read())
    config_workload_generation_check(config=config)
    generate_workload(**config)


if __name__ == "__main__":
    main()
