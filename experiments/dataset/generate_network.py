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

from smart_scheduler.dataset import NetworkGenerator

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))

from experiments.utils.constants import (
    DATASETS_PATH,
    CONFIGS_PATH,
    DATASETS_METADATA_PATH,
)

from experiments.utils import config_network_generation_check

def generate_network(
    notes: str, dataset_id: int, num_stations: int,
    num_users: int, width: float, length: float,
    speed_limit: int, nodes_stations_con: int,
    users_services_distributions: str,
    from_dataset: bool, dataset_metadata: int,
    colocated: bool,
    nodes_selection: str, nodes_list: list,
    seed: int):
    """
        generate a random workload
    """
    # dataset and dataset_metadata path
    dataset_path = os.path.join(DATASETS_PATH, str(dataset_id))
    if from_dataset:
        dataset_metadata_folder = os.path.join(
            DATASETS_METADATA_PATH,
            str(dataset_metadata))
        stations_dataset_path = os.path.join(
            dataset_metadata_folder,
            'stations.txt')
        users_dataset_path = os.path.join(
            dataset_metadata_folder,
            'users.txt')
    else:
        stations_dataset_path = None
        users_dataset_path = None

    # read the dataset
    try:
        with open(
            os.path.join(
                dataset_path, 'dataset.pickle'), 'rb') as in_pickle:
            dataset = pickle.load(in_pickle)
    except:
        raise FileNotFoundError(f"dataset <{dataset_id}> does not exist")

    # fix foldering per dataset
    networks_path = os.path.join(dataset_path, 'networks')
    content = os.listdir(os.path.join(networks_path))
    new_network = len(content)
    dir2save = os.path.join(networks_path, str(new_network))
    os.mkdir(dir2save)

    # generate the network
    network_generator = NetworkGenerator(
        dataset=dataset,
        num_stations=num_stations,
        num_users=num_users,
        width=width,
        length=length,
        speed_limit=speed_limit,
        nodes_stations_con=nodes_stations_con,
        from_dataset=from_dataset,
        users_services_distributions=users_services_distributions,
        dataset_metadata=dataset_metadata,
        stations_dataset_path=stations_dataset_path,
        users_dataset_path=users_dataset_path,
        nodes_selection=nodes_selection,
        colocated=colocated,
        nodes_list=nodes_list,
        seed=seed)
    edge_simulator, edge_simulator_config = network_generator.make_network()

    # information of the generated network
    info = {
        'notes': notes,
        'dataset_id': dataset_id,
        'num_users': num_users,
        'num_stations': num_stations,
        'from_dataset': from_dataset,
        'width': width,
        'length': length,
        'speed_limit': speed_limit,
        'nodes_stations_con': nodes_stations_con,
        'seed': seed
    }

    if from_dataset:
        info.update({
            'selected_nodes_indexes':\
                edge_simulator.selected_nodes_indexes.tolist()})

    # save the info and network in the folder
    with open(os.path.join(dir2save, 'info.json'), 'x') as out_file:
        json.dump(info, out_file, indent=4)
    with open(os.path.join(
        dir2save, 'edge_simulator_config.pickle'), 'wb') as out_pickle:
        pickle.dump(edge_simulator_config, out_pickle)
    print(f"\n\nGenerated data saved in <{dir2save}>\n\n")

    # save the figures of the network
    edge_simulator.visualize_debug().savefig(
        os.path.join(dir2save, 'fig.png'))
    edge_simulator.visualize_paper_style().savefig(
        os.path.join(dir2save, 'fig-paper-style.png'))

    # empty folder for the traces of the network
    os.mkdir(os.path.join(dir2save, 'traces'))


def main():
    # read the config file
    config_file_path = os.path.join(
        CONFIGS_PATH,
        'generation-configs',
        'network.json')
    with open(config_file_path) as cf:
        config = json.loads(cf.read())
    print('generating dataset from the following config:')
    pp.pprint(config)
    config_network_generation_check(config)
    generate_network(**config)


if __name__ == "__main__":
    main()
