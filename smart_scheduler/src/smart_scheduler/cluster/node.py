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

    def add_service(self, service: Service) -> bool:
        # add if there is enough request available
        if np.alltrue(
            self.requests_available < service.requests):
            return False
        # check if the node has enough resource available
        if np.alltrue(
            self.resources_unused < service.requests): # TODO check against k8s
            return False
        self.services.append(service)
        return True

    def deschedule(self, service_index):
        # TODO debug
        # schedule the service on the node
        self.served_services.append(service_index)
        # remove the service from the pending services
        self.services.pop(service_index)

    @property
    def usages(self):
        usages = sum(
            list(map(
                lambda service: service.usages, self.services)))
        if type(usages) == int:
            return np.zeros(2)
        return usages

    @property
    def requests(self):
        requests = sum(
            list(map(
                lambda service: service.requests, self.services)))
        if type(requests) == int:
            return np.zeros(2)
        return requests

    @property
    def requests_available(self):
        """total available resource request on the node
        """
        return self.capacities - self.requests

    @property
    def resources_unused(self):
        """total available resource on the node
        """
        return self.capacities - self.usages

    @property
    def slack(self):
        """total unused requested resources
        """
        return self.requests - self.usages

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
