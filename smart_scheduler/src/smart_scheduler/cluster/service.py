from turtle import done
import numpy as np
from pendulum import duration

class Service:
    def __init__(self, service_id: int,
                 service_name: str, requests: np.ndarray,
                 limits: np.ndarray, workload: np.ndarray,
                 serving_time: int,
                 start_time: int = 0) -> None:
        self.service_id = service_id
        self.service_name = service_name
        self.requests = requests
        self.limits = limits
        self.workload = workload
        self.start_time = start_time
        self.duration = serving_time
        self.time = start_time

    def clock_tick(self, time):
        if time < self.start_time\
            or time > self.start_time + self.duration:
            raise ValueError('Invalid time!')
        self.time = time

    @property
    def usages(self):
        return self.workload[:, self.time]

    @property
    def slack(self):
        return self.requests - self.usages
    
    @property
    def done(self):
        if self.start_time + self.duration == self.time:
            return True
        else:
            return False

