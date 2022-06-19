# TODO in case of integration with the Alibaba workload at
# some point


import numpy as np
import random
from tqdm import tqdm
from smart_scheduler.util import (
    plot_workload,
    rounding)
import random
import string
from typing import List


class WorkloadGeneratorAlibaba:
    def __init__(self, cluster, num_services, plot_smoothing, seed):
        """
            cluster generator
        """
        # self.cluster = cluster
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(seed)
        # self.services_types: np.array = self.cluster['services_types']

        self.plot_smoothing = plot_smoothing

    def make_workloads(self):
        """
        making random workload per each service type
        each channel (third dimension) is a different workload for
        a different machine
        generating workloads

                                timesteps
                          |                |
                        | |              | |
            ram       | |              | |
            cpu       |                |    types of services

        start workload:

                        different types
            ram       |                |
            cpu       |                |

        """

        # option 1: with variations in the workload resource usage
        # workloads = np.zeros((self.num_resources,
        #                       self.timesteps,
        #                       self.num_services_types))
        # workloads[:, 0] = self.start_workloads

        # for col in tqdm(range(1, self.timesteps)):
        #     num_steps = np.random.randint(-self.workloads_max_steps,
        #                                   self.workloads_max_steps+1)
        #     steps = num_steps * self.workloads_steps_units
        #     workloads[:, col] = workloads[:, col-1] + steps
        #     workloads[workloads < 0] = 0
        #     workloads[workloads > 1] = 1
            # workloads = np.round(workloads, 2)

        figs = []
        # TODO use once done
        # for i in range(self.num_services_types):
        #     workload_i = workloads[:, :, i]
        #     fig = plot_workload(self.timesteps, workload_i,
        #                         self.plot_smoothing, i)
        #     figs.append(fig)
        # return workloads, figs

