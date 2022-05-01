import numpy as np

class Service:
    def __init__(self, service_id: int, requests: np.array,
                 limits: np.array, workload: np.array,
                 service_start_time: int, duration: int) -> None:
        self.service_id = service_id
        self.requests = requests
        self.limits = limits
        self.workload = workload
        self.service_start_time = service_start_time
        self.duration = duration

    def container_usage(self, time):
        return self.workload[self.service_start_time - time]
