import numpy as np
from typing import Dict
from smart_scheduler.util import rounding

# TODO complete based on the new needs

class Preprocessor():
    def __init__(self, max_services_nodes: int,
                 cluster_nodes_capacities: np.ndarray,
                 services_resources_request: np.ndarray,):
        self.nodes_resources_cap = cluster_nodes_capacities
        self.services_resources_request = services_resources_request
        self.num_nodes = cluster_nodes_capacities.shape[0]
        self.max_services_nodes = max_services_nodes

    @rounding
    def transform(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        """
        transform the input observation as the dictionary
        sends each key of the dictionary to the approperiate preprocessor
        and returns the concatenated and flattened numpy array
        """
        obs = np.array([])
        transformers = {
            "backlog_services_requests": self._nodes_normalizer,
            "nodes_capacities": self._nodes_normalizer,
            "nodes_usages": self._nodes_normalizer,
            "nodes_requests": self._nodes_normalizer,
            "nodes_available": self._nodes_normalizer,
            "nodes_unused": self._nodes_normalizer,
            "nodes_slack": self._nodes_normalizer,
            "nodes_usages_frac":self._none,
            "nodes_requests_frac": self._none,
            "backlog_services_requests_frac": self._none,
            "nodes_requests_available_frac": self._none,
            "nodes_resources_unused_frac": self._none,
            "nodes_requests_available_frac_avg": self._none,
            "nodes_resources_unused_avg": self._none,
            "num_consolidated": self._one_hot_consolidation,
            "num_services_nodes": self._one_hot_services_nodes
        }
        for key, val in observation.items():
            obs = np.concatenate((obs, transformers.get(
                key, self._invalid_operation)(val).flatten()))
        return obs

    def _services_usage_normalizer(self, obs: np.ndarray) -> np.ndarray:
        """
        divides the largest available of each resource by the
        capacity of the largest size of that resource in the cluster
        in any service
        e.g for ram:
            ram_usage_of_a_service / largest_ram_capacity_of_any_conrainer
        """
        lst = []
        for index in range(self.services_resources_request.shape[1]):
            lst.append(max(self.services_resources_request[:, index]))
        return obs/lst

    def _nodes_normalizer(self, obs: np.ndarray) -> np.ndarray:
        """
        divides the largest available of each resource by the
        capacity of the largest size of that resource in the cluster
        in any node
        e.g for ram:
            ram_usage_of_a_node / largest_ram_capacity_of_any_node
        """
        lst = []
        for index in range(self.nodes_resources_cap.shape[1]):
            lst.append(max(self.nodes_resources_cap[:, index]))
        return obs/lst

    def _none(self, obs: np.ndarray) -> np.ndarray:
        return obs

    def _one_hot_services_nodes(
        self, obs: np.ndarray) -> np.ndarray:
        """
        one hot encoding of the services_nodes
        e.g in a cluster of 2 nodes and 4 services:
            [0, 1, 1, 0]
        results in:
            [0, 0, 0, 1, 0, 1, 0, 0]
        """
        obs_prep = np.array([])

        for node in obs:
            one_hot_encoded = np.zeros(self.max_services_nodes + 1)
            one_hot_encoded[node] = 1
            obs_prep = np.concatenate((obs_prep, one_hot_encoded))
        return obs_prep

    def _one_hot_consolidation(
        self, obs: np.ndarray) -> np.ndarray:
        """
        one hot encoding of the number of consolidated
        servers in the cluster
        """
        one_hot_encoded = np.zeros(self.num_nodes)
        one_hot_encoded[obs-1] = 1
        return one_hot_encoded

    def _invalid_operation(self, obs: np.ndarray) -> None:
        raise ValueError(f"invalid observation: <{obs}>")
