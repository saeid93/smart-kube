import abc
import numpy as np
from copy import deepcopy
from typing import (
    List,
    Dict,
    Any,
    Literal,
    Tuple
)
import numpy as np

from smart_scheduler.util import (
    rounding
)
from .node import Node
from .service import Service


class Cluster:
    def __init__(self, cluster_schema: Dict[str, np.ndarray]):

        self.nodes_resources_cap: np.ndarray = cluster_schema['nodes_resources_cap']


        # find the number of nodes, services, service types and timesteps
        self.num_resources: int = self.nodes_resources_cap.shape[1]
        self.num_nodes: int = self.nodes_resources_cap.shape[0]

        self.serving_services: List[Service] = []
        self.nodes: List[Node] = []

        for node_id in range(self.num_nodes):
            self.nodes.append(Node(
                node_id=node_id,
                capacities=self.nodes_resources_cap[node_id], 
            ))


    def schedule(self, service: Service, node_id: int) -> bool:
        """schedule one of the services on a target node

        Args:
            service (Service): id of the service
                to be scheduled
            node_id (int): id of the node
                to be scheduled

        Returns:
            bool: returns whether the service
                has been scheduled or not
        """
        # schedule the service on the node
        schedule_success = self.nodes[
            node_id].add_service(service)
        # return true if successful
        return schedule_success

    @property
    def nodes_resources_usage(self):
        """return the amount of resource usage
        on each node
                     ram - cpu
                    |         |
            nodes   |         |
                    |         |

            range:
                row inidices: (0, num_nodes]
                columns indices: (0, num_resources]
                enteries: [0, node_resource_cap] type: float
        """
        nodes_resources_usage = np.array(
            list(map(lambda node: node.usages, self.nodes)))
        return nodes_resources_usage

    @property
    def nodes_resources_request(self):
        """return the amount of resource usage
        on each node
        """
        nodes_resources_requests = np.array(
            list(map(lambda node: node.requests, self.nodes)))
        return nodes_resources_requests

    @property
    def nodes_requests_available(self):
        # The amount of non requested resources on the nodes
        nodes_requests_available = np.array(
            list(map(lambda node: node.requests_available, self.nodes)))
        return nodes_requests_available

    @property
    def nodes_resources_unused(self):
        # The amount of the available
        nodes_resources_unused = np.array(
            list(map(lambda node: node.resources_unused, self.nodes)))
        return nodes_resources_unused

    @property
    @rounding
    def nodes_resources_usage_frac(self) -> np.ndarray:
        """returns the resource usage of
        each node
                     ram - cpu
                    |         |
            nodes   |         |
                    |         |

            range:
                row inidices: (0, num_nodes]
                columns indices: (0, num_resources]
                enteries: [0, 1] type: float
        """
        return self.nodes_resources_usage / self.nodes_resources_cap

    @property
    @rounding
    def nodes_resources_request_frac(self):
        """returns the resource requested on
        each node
                     ram - cpu
                    |         |
            nodes   |         |
                    |         |

            range:
                row inidices: (0, num_nodes]
                columns indices: (0, num_resources]
                enteries: [0, 1] type: float
        """
        return self.nodes_resources_request / self.nodes_resources_cap

    @property
    def num_consolidated(self) -> int:
        """returns the number of consolidated nodes
        """
        num_consolidated = 0
        for node in self.nodes:
            if node.services == []:
                num_consolidated+=1
        return num_consolidated

    @property
    def nodes_services(self) -> List[List]:
        """
                node_id                    node_id
            [[service_id, service_id], ...,[service_id]]
        """
        nodes_services = list(
            map(lambda node: node.services_ids, self.nodes))
        return nodes_services

    @property
    def nodes_requests_available_frac(self):
        return self.nodes_requests_available / self.nodes_resources_cap

    @property
    def nodes_resources_unused_frac(self):
        return self.nodes_resources_unused / self.nodes_resources_cap

    @property
    def nodes_requests_available_frac_avg(self):
        return np.average(self.nodes_requests_available_frac, axis=1)

    @property
    def nodes_resources_unused_avg(self):
        return np.average(self.nodes_resources_unused_frac, axis=1)
