from this import d
import numpy as np
from .service import Service

class Node:
    def __init__(self, node_id: int, capacities: np.array, 
                 node_start_time: int = 0) -> None:
        self.services = []
        self.node_id = node_id
        self.capacities = capacities
        self.node_start_time = node_start_time
        self.time = node_start_time

    def clock_tick(self, time):
        # TODO update all services time
        raise NotImplementedError

    def add_service(self, service: Service) -> bool:
        # TODO check for available resources
        raise NotImplementedError

    @property
    def nodes_services(self):
        # TODO calculate from the services
        raise NotImplementedError

    @property
    def nodes_usage(self):
        # TODO calculate from the services
        raise NotImplementedError

    @property
    def node_request(self):
        # TODO calculate from the services
        raise NotImplementedError

    @property
    def nodes_available(self):
        # TODO calculate from the usage and capacities
        raise NotImplementedError

    @property
    def nodes_request_available(self):
        # TODO calculate from the request and capacities
        raise NotImplementedError
