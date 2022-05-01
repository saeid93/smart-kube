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
from copy import deepcopy
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

from smart_scheduler.cluster_generator import ClusterGenerator

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))

from experiments.utils.constants import (
    CLUSTERS_PATH,
    CONFIGS_PATH
)
from experiments.utils import config_cluster_generation_check


def generate_cluster(config):
    """
        use the random_initializer.py and random_state_initializer.py
        to make and save initial_states
    """
    # generate the cluster
    generator_config = deepcopy(config)
    del generator_config['notes']
    cluster_generator = ClusterGenerator(**generator_config)
    cluster = cluster_generator.make_cluster()

    # fix the paths to save the newly generated datset
    content = os.listdir(CLUSTERS_PATH)
    new_cluster = len(content)
    dir2save = os.path.join(CLUSTERS_PATH, str(new_cluster))
    os.mkdir(dir2save)

    # information of the generated cluster
    info = config
    info['capacities'] = {}
    info['capacities']['nodes_resources'] = \
        cluster['nodes_resources_cap'].tolist()
    info['capacities']['services_resources'] = \
        cluster['services_resources_request'].tolist()
    info['services_nodes'] = \
        cluster['services_nodes'].tolist()

    # save the info and cluster in the folder
    with open(os.path.join(dir2save, 'info.json'), 'x') as out_file:
        json.dump(info, out_file, indent=4)
    with open(os.path.join(dir2save, 'cluster.pickle'), 'wb') as out_pickle:
        pickle.dump(cluster, out_pickle)
    print(f"\n\nGenerated data saved in <{dir2save}>\n\n")

    # empty folder for the workload and networks
    os.mkdir(os.path.join(dir2save, 'workloads'))
    # os.mkdir(os.path.join(dir2save, 'networks'))


def main():
    # read the config file
    config_file_path = os.path.join(
        CONFIGS_PATH,
        'generation-configs',
        'cluster.json')
    with open(config_file_path) as cf:
        config = json.loads(cf.read())
    print('generating cluster from the following config:')
    pp.pprint(config)
    config_cluster_generation_check(config)
    generate_cluster(config)


if __name__ == "__main__":
    main()
