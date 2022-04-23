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
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

from smart_scheduler.dataset import TraceGenerator

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))

from experiments.utils.constants import (
    DATASETS_PATH,
    CONFIGS_PATH
)

from experiments.utils import config_trace_generation_check

def generate_workload(dataset_id: int, network_id: int,
                      from_dataset: bool, seed: int,
                      speed: int, timesteps: int):
    """
        generate a movement trace
    """
    # dataset, dataset_metadata and network path
    dataset_path = os.path.join(DATASETS_PATH, str(dataset_id))
    network_path = os.path.join(dataset_path, 'networks', str(network_id))

    # read the network
    try:
        with open(os.path.join(network_path, 'edge_simulator_config.pickle'), 'rb')\
            as in_pickle:
            if not os.path.isdir(dataset_path):
                raise FileNotFoundError((f"dataset {dataset_id}"
                                         " does not exist"))
            edge_simulator_config = pickle.load(in_pickle)
    except:
        raise FileNotFoundError((f"network <{network_id}> does not exist"
                                 f" for dataset <{dataset_id}>")) 

    if from_dataset and not edge_simulator_config['from_dataset']:
        raise ValueError("random stations cannot go with real movements")

    # fix foldering per dataset
    traces_path = os.path.join(network_path, 'traces')
    content = os.listdir(traces_path)
    new_trace = len(content)
    dir2save = os.path.join(traces_path, str(new_trace))
    os.mkdir(dir2save)

    del edge_simulator_config['from_dataset']
    # generate the trace
    movement_trace_generator = TraceGenerator(
        edge_simulator_config=edge_simulator_config,
        timesteps=timesteps,
        from_dataset=from_dataset,
        seed=seed)
    traces = movement_trace_generator.make_traces()

    # information of the generated trace
    info = {
        'from_dataset': from_dataset,
        'seed': seed,
        'speed': speed,
        'timesteps': timesteps
    }

    # save the information and trace in the folder
    with open(os.path.join(dir2save, 'info.json'), 'x') as out_file:
        json.dump(info, out_file, indent=4)
    with open(os.path.join(dir2save, 'trace.pickle'), 'wb') as out_pickle:
        pickle.dump(traces, out_pickle)
    print(f"\n\nGenerated data saved in <{dir2save}>\n\n")


def main():
    # read the config file
    config_file_path = os.path.join(
        CONFIGS_PATH,
        'generation-configs',
        'trace.json')
    with open(config_file_path) as cf:
        config = json.loads(cf.read())
    print('generating trace from the following config:')
    pp.pprint(config)
    config_trace_generation_check(config)
    generate_workload(**config)

if __name__ == "__main__":
    main()
