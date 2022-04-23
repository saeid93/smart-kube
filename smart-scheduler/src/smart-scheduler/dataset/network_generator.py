import numpy as np
import random
from tqdm import tqdm

from smart_scheduler.network import (
    NetworkBuilderDataset,
    NetworkBuilderRandom
)


class NetworkGenerator:
    def __init__(self, *, dataset: np.array,
                 num_stations: int,
                 num_users: int,
                 width: float, length: float,
                 speed_limit: int,
                 nodes_stations_con: int,
                 from_dataset: bool,
                 users_services_distributions: str,
                 dataset_metadata,
                 stations_dataset_path=None,
                 users_dataset_path=None,
                 nodes_selection,
                 nodes_list,
                 colocated,
                 seed):
        """
            edge network generator
        """
        self.services_nodes = dataset['services_nodes']
        self.num_nodes: int = dataset['nodes_resources_cap'].shape[0]
        self.num_services: int = dataset[
            'services_resources_request'].shape[0]
        self.nodes_stations_con = nodes_stations_con
        self.num_users = num_users
        self.num_stations = num_stations
        self.width = width
        self.length = length
        self.speed_limit = speed_limit
        self.seed = seed
        self.from_dataset = from_dataset
        self.users_services_distributions =\
            users_services_distributions
        self.dataset_metadata = dataset_metadata
        self.stations_dataset_path = stations_dataset_path
        self.users_dataset_path = users_dataset_path
        self.nodes_selection = nodes_selection
        self.nodes_list = nodes_list
        self.colocated = colocated
        np.random.seed(self.seed)
        random.seed(self.seed)

    def make_network(self):
        """
        """
        # users_services fixed in all the environment
        if self.users_services_distributions == 'random':
            users_services = np.random.randint(self.num_services,
                                                 size=self.num_users)
        elif self.users_services_distributions == 'equal':
            users_services = []
            service = 0
            for i in range(self.num_users):
                users_services.append(service)
                service += 1
                if service % self.num_services == 0:
                    service = 0
            users_services = np.array(users_services)

        if self.from_dataset:
            edge_simulator = NetworkBuilderDataset(
                services_nodes=self.services_nodes,
                users_services=users_services,
                num_nodes=self.num_nodes,
                num_stations=self.num_stations,
                width=self.width, length=self.length,
                speed_limit=self.speed_limit,
                nodes_stations_con=self.nodes_stations_con,
                stations_dataset_path=self.stations_dataset_path,
                users_dataset_path=self.users_dataset_path,
                nodes_selection=self.nodes_selection,
                nodes_list=self.nodes_list,
                seed=self.seed)
        else:
            edge_simulator  = NetworkBuilderRandom(
                services_nodes=self.services_nodes,
                users_services=users_services,
                num_nodes=self.num_nodes,
                num_stations=self.num_stations,
                width=self.width, length=self.length,
                speed_limit=self.speed_limit,
                nodes_stations_con=self.nodes_stations_con,
                colocated=self.colocated,
                seed=self.seed)

        edge_simulator_config = {
           'services_nodes': self.services_nodes,
           'users_services': users_services,
           'num_nodes': self.num_nodes,
           'num_stations': self.num_stations,
           'width': self.width,
           'length': self.length,
           'from_dataset': self.from_dataset,
           'speed_limit': self.speed_limit,
           'nodes_stations_con': self.nodes_stations_con,
           'seed': self.seed,
           'network': edge_simulator.network,
           'raw_network': edge_simulator.raw_network,
           'selected_nodes':\
               edge_simulator.selected_nodes_indexes\
                   if self.from_dataset else None
        }
        if self.from_dataset:
            edge_simulator_config.update({
                'stations_dataset_path': self.stations_dataset_path,
                'users_dataset_path': self.users_dataset_path,
            })
        return edge_simulator, edge_simulator_config
