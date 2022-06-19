import numpy as np
import random
from tqdm import tqdm
from smart_scheduler.util import (
    plot_workload,
    rounding)
import random
import string
from typing import List


class WorkloadGeneratorRandom:
    def __init__(self, cluster, workloads_var, timesteps, num_services_types,
                 start_workloads, plot_smoothing, seed):
        """
            cluster generator
        """
        self.cluster = cluster
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(seed)
        self.services_types: np.array = self.cluster['services_types']

        self.num_resources = cluster['nodes_resources_cap'].shape[1]
        self.timesteps = timesteps
        self.num_services_types = num_services_types

        self.services_resources_request: np.array = self.cluster[
            'services_resources_request']
        self.num_services: int = self.services_resources_request.shape[0]
        self.nodes_resources_cap: np.array = self.cluster['nodes_resources_cap']
        self.num_nodes: int = self.nodes_resources_cap.shape[0]

        self.start_workloads = np.transpose(np.array(start_workloads))
        self.workloads_steps_units = \
            np.transpose(np.array(workloads_var['steps_unit']))
        self.workloads_max_steps = \
            np.transpose(np.array(workloads_var['max_steps']))
        assert self.num_resources == self.start_workloads.shape[0],\
            (f"num_resources: <{self.num_resources}> "
             f"not equals start_workloads.shape[0]: "
             f"<{self.start_workloads.shape[0]}>")
        assert self.num_services_types == self.start_workloads.shape[1],\
            (f"num_services_types: <{self.num_services_types}> "
             f"not equals start_workloads.shape[1]: "
             f"<{self.start_workloads.shape[1]}>")
        assert self.num_resources == self.workloads_steps_units.shape[0],\
            (f"num_resources: <{self.num_resources}> "
             f"workloads_steps_units.shape[0]: "
             f"{self.workloads_steps_units.shape[0]}")
        assert self.num_services_types == self.workloads_steps_units.shape[1],\
            (f"num_services_types: <{self.num_services_types}> "
             f"not equals workloads_steps_units.shape[1]: "
             f"{self.workloads_steps_units.shape[1]}")
        assert self.num_resources == self.workloads_max_steps.shape[0],\
            (f"num_resources: <{self.num_resources}> "
             f"self.workloads_max_steps.shape[0]: "
             f"<{self.workloads_max_steps.shape[0]}>")
        assert self.num_services_types == self.workloads_max_steps.shape[1],\
            (f"num_services_types: <{self.num_services_types}> "
             f"workloads_max_steps.shape[1]: "
             f"{self.workloads_max_steps.shape[1]}")

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
        workloads = np.zeros((self.num_resources,
                              self.timesteps,
                              self.num_services_types))
        workloads[:, 0] = self.start_workloads

        for col in tqdm(range(1, self.timesteps)):
            num_steps = np.random.randint(-self.workloads_max_steps,
                                          self.workloads_max_steps+1)
            steps = num_steps * self.workloads_steps_units
            workloads[:, col] = workloads[:, col-1] + steps
            workloads[workloads < 0] = 0  # TODO from input if necessary
            workloads[workloads > 1] = 1
            # workloads = np.round(workloads, 2)

        figs = []
        for i in range(self.num_services_types):
            workload_i = workloads[:, :, i]
            fig = plot_workload(self.timesteps, workload_i, i)
            figs.append(fig)
        return workloads, figs

