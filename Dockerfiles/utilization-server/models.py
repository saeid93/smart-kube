import numpy as np


class Service:
    """Service Model"""

    def __init__(self, hostname: str, workload_id: int, is_up: bool = False):
        """Service

        :param hostname: str
            hostname of service

        :param workload_id: int
            hostname of service

        :param is_up: bool (default: False)
            service is up or not
        """
        self.is_up: bool = is_up
        self.hostname: str = hostname
        self.workload_id: int = workload_id

    def __str__(self):
        return "Service(hostname='{}', workload_id='{}', up='{}')".format(
            self.hostname, self.workload_id, self.is_up
        )


class Resources:
    """Resources Model"""

    def __init__(self, ram: str, cpu: str, storage: str):
        """Resources

        :param ram: str
            RAM usage of service

        :param cpu: str
            cpu load of service

        :param storage: str
            storage usage of service
        """
        self.ram: str = ram
        self.cpu: str = cpu
        self.storage: str = storage

    def __str__(self):
        return "Resources(RAM='{}', CPU='{}', STORAGE='{}')".format(
            self.ram, self.cpu, self.storage
        )


class WorkLoads:
    """WorkLoads Model"""

    def __init__(self, data: np.array):
        """WorkLoads

        :param data: np.array (nWorkloads x nTimesteps x nResources)
            all of workloads (it a 3D matrix)
            1D: number of Workloads
            2D: number of timesteps
            3D: number of resources
        """
        self.data: np.array = np.array(data).T # HACK transpose for temp fix for inconsistency between implementations
        self.nWorkloads, self.nTimesteps, self.nResources = self.data.shape

    def get_workload(self, service: Service) -> np.array:
        """Get Workload of a Service

        :param service: Service
            your expected service
        """
        return self.data[service.workload_id]

    def get_timesteps(self, workload_id: int) -> np.array:
        """Get Timesteps
            get all resources over timesteps

        :param workload_id: int
            refers to workload index

        :return: np.array
            1 x nTimesteps x nResources
        """
        return self.data[workload_id]

    def get_resources(self, workload_id: int, timestep: int) -> np.array:
        """Get Resoources

        :param workload_id: int
            refers to workload index

        :param timestep: int
            refers to timestep index

        :return: np.array
            1 x 1 x nResources
        """
        return self.data[workload_id, timestep]

    def __str__(self):
        return "WorkLoads(nWorkloads='{}', nTimesteps='{}', nResources='{}')".format(
            *self.data.shape
        )


class Cluster:
    """Cluster Model"""

    def __init__(self, data: dict):
        self.data = data
