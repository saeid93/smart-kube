import numpy as np
from .service import Service
from typing import List

class Node:
    def __init__(self, node_id: int, capacities: np.ndarray) -> None:
        self.services: List[Service] = []
        self.served_services: List[Service] = []
        self.node_id = node_id
        self.capacities = capacities
        self.time = 0

    def clock_tick(self):
        """move the clock of the server forward

        Args:
            time (int): time in seconds

        Raises:
            ValueError: check the validation of time
        """
        self.time += 1
        list(map(lambda a: a.clock_tick(), self.services))
        for service_index, service in enumerate(self.services):
            if service.done:
                self.deschedule(service_index)

    def reset_node(self):
        self.time = 0
        self.served_services = []

    def add_service(self, service: Service) -> bool:
        service.start_time_update(self.time)
        # add if there is enough request available
        if np.alltrue(
            self.requests_available < service.requests):
            return False
        # check if the node has enough resource available
        if np.alltrue(
            self.unused < service.requests): # TODO check against k8s
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
    def requests(self):
        """total resource request on the node
        """
        requests = sum(
            list(map(
                lambda service: service.requests, self.services)))
        if type(requests) == int:
            return np.zeros(2)
        return requests

    @property
    def requests_fraction(self):
        """total resource request on the node fraction
        """
        return self.requests / self.capacities

    @property
    def requests_available(self):
        """total available resource request on the node
        """
        return self.capacities - self.requests

    @property
    def requests_available_fraction(self):
        """total available resource request on the node fraction
        """
        return self.requests_available / self.capacities

    @property
    def usages(self):
        usages = sum(
            list(map(
                lambda service: service.usages, self.services)))
        if type(usages) == int:
            return np.zeros(2)
        return usages

    @property
    def usages_fraction(self):
        """total resource usages on the node fraction
        """
        return self.usages / self.capacities

    @property
    def unused(self):
        """total available resource on the node
        """
        return self.capacities - self.usages

    @property
    def unused_fraction(self):
        """percentage of total available resource fraction
        """
        return self.resources_unused / self.capacities

    @property
    def slack(self):
        """total unused requested resources
        """
        return self.requests - self.usages

    @property
    def slack_fraction(self):
        """total unused requested resources fraction
        """
        return self.slack / self.capacities

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
    def is_overloaded(self):
        return np.alltrue(self.usages > self.capacities)

    @property
    def all_jobs_done(self):
        if self.services == []:
            return True
        else:
            return False
