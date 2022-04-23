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
from copy import deepcopy
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

from smart_scheduler.dataset import DatasetGenerator

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))

from experiments.utils.constants import (
    DATASETS_PATH,
    CONFIGS_PATH
)
from experiments.utils import config_dataset_generation_check


def generate_dataset(config):
    """
        use the random_initializer.py and random_state_initializer.py
        to make and save initial_states
    """
    # generate the dataset
    generator_config = deepcopy(config)
    del generator_config['notes']
    dataset_generator = DatasetGenerator(**generator_config)
    dataset = dataset_generator.make_dataset()

    # fix the paths to save the newly generated datset
    content = os.listdir(DATASETS_PATH)
    new_dataset = len(content)
    dir2save = os.path.join(DATASETS_PATH, str(new_dataset))
    os.mkdir(dir2save)

    # information of the generated dataset
    info = config
    info['capacities'] = {}
    info['capacities']['nodes_resources'] = \
        dataset['nodes_resources_cap'].tolist()
    info['capacities']['services_resources'] = \
        dataset['services_resources_request'].tolist()
    info['services_nodes'] = \
        dataset['services_nodes'].tolist()

    # save the info and dataset in the folder
    with open(os.path.join(dir2save, 'info.json'), 'x') as out_file:
        json.dump(info, out_file, indent=4)
    with open(os.path.join(dir2save, 'dataset.pickle'), 'wb') as out_pickle:
        pickle.dump(dataset, out_pickle)
    print(f"\n\nGenerated data saved in <{dir2save}>\n\n")

    # empty folder for the workload and networks
    os.mkdir(os.path.join(dir2save, 'workloads'))
    os.mkdir(os.path.join(dir2save, 'networks'))


def main():
    # read the config file
    config_file_path = os.path.join(
        CONFIGS_PATH,
        'generation-configs',
        'dataset.json')
    with open(config_file_path) as cf:
        config = json.loads(cf.read())
    print('generating dataset from the following config:')
    pp.pprint(config)
    config_dataset_generation_check(config)
    generate_dataset(config)


if __name__ == "__main__":
    main()
