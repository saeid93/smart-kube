import numpy as np
from .service import Service
from typing import List

class Node:
    def __init__(self, node_id: int, capacities: np.ndarray, 
                 start_time: int = 0) -> None:
        self.services: List[Service] = []
        self.served_services: List[Service] = []
        self.node_id = node_id
        self.capacities = capacities
        self.start_time = start_time
        self.time = start_time

    def clock_tick(self, time: int):
        """move the clock of the server forward

        Args:
            time (int): time in seconds

        Raises:
            ValueError: check the validation of time
        """
        if time < self.start_time:
            raise ValueError('Invalid time!')
        self.time = time
        for service in self.services:
            service.clock_tick(time)
        # TODO debug
        for service_index, service in enumerate(self.services):
            if service.done:
                self.deschedule(service_index)

    def deschedule(self, service_index):
        # TODO debug
        # schedule the service on the node
        self.served_services.append(service_index)
        # remove the service from the pending services
        self.services.pop(service_index)

    @property
    def nodes_usage(self):
        # TODO calculate from the services
        a = 1
        return np.array([0, 0])

    @property
    def requests(self):
        # TODO calculate from the services
        a = 1
        return np.array([100000, 100000])

    @property
    def resources_available(self):
        # TODO calculate from the usage and capacities
        a = 1
        return np.array([100000, 100000])

    def add_service(self, service: Service) -> bool:
        # TODO check for available resources
        # add if there is enough request available
        self.services.append(service)
        return True

    @property
    def services_ids(self):
        services_id = list(
            map(lambda l: l.service_id, self.services))
        return services_id

    @property
    def services_names(self):
        services_names = list(
            map(lambda l: l.service_name, self.services))
        return services_names

    @property
    def requests_available(self):
        # TODO calculate from the request and capacities
        a = 1
        return np.array([100000, 100000])