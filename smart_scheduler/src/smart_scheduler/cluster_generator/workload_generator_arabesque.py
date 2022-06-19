from distutils.log import set_verbosity
from cv2 import sepFilter2D
import numpy as np
import random
import os
import pickle
from tqdm import tqdm
from smart_scheduler.util import (
    plot_workload,
    rounding)
import random
import string
from typing import List


class WorkloadGeneratorArabesque:
    def __init__(
        self, cluster, min_timesteps, 
        dataset_path, num_services, plot_smoothing, seed):
        """
            cluster generator
        """
        self.cluster = cluster
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(seed)
        self.num_services = num_services
        self.min_timesteps = min_timesteps

        dataset_path = os.path.join(dataset_path, 'arabesque-single-file')
        dataset_file_names = os.listdir(dataset_path)
        datasets = {}
        for dataset_file_name in dataset_file_names:
            with open(os.path.join(dataset_path, dataset_file_name), 'rb') as dataset:
                dataset = pickle.load(dataset)
            datasets[dataset_file_name.replace('.pickle', '')] = dataset
        self.datasets = datasets
        # TODO read all the dataset workloads one by one with folder ordering

        self.plot_smoothing = plot_smoothing

    def is_fit(self, service_data):
        # check if a workload fit into any
        # of the servers - we are not experimenting
        # situation with server overload
        limits = np.array(list(service_data['limits'].values()))
        limits[1] = limits[1]/1000
        if np.alltrue(limits < self.cluster['nodes_resources_cap'])\
            and len(service_data['time']) >= self.min_timesteps:
            return True
        return False


    def make_workloads(self):
        """
        choosing containers from randomly the dataset
        """
        # grab containers from the list until the selected
        # services are proportial
        workloads = {}
        # while len(workloads) < self.num_services:
        # first grab the top ten containers
        for service_name, service_data in self.datasets[
            'engine-top-ten']['engine'].items():
            if not self.is_fit(service_data):
                continue
            workloads[service_name] = service_data
            if len(workloads) >= self.num_services:
                break
        for service_name, service_data in self.datasets[
            'portfolio-top-ten']['qryfolio-daily'].items():
            if not self.is_fit(service_data):
                continue
            workloads[service_name] = service_data
            if len(workloads) >= self.num_services:
                break
        for _, dataset in self.datasets[
            'engine-july-all'].items():
            for service_name, service_data in dataset.items():
                if not self.is_fit(service_data):
                    continue
                workloads[service_name] = service_data
                if len(workloads) >= self.num_services:
                    break
            if len(workloads) >= self.num_services:
                break
        for _, dataset in self.datasets[
            'portfolio-july-all'].items():
            for service_name, service_data in dataset.items():
                if not self.is_fit(service_data):
                    continue
                workloads[service_name] = service_data
                if len(workloads) >= self.num_services:
                    break
            if len(workloads) >= self.num_services:
                break
        figs = []
        for service_name, service in workloads.items():
            fig = plot_workload(
                service['workload'].shape[1], service['workload'], service_name)
            figs.append(fig)
        # for i in range(self.num_services_types):
        #     workload_i = workloads[:, :, i]
        #     fig = plot_workload(self.timesteps, workload_i,
        #                         self.plot_smoothing, i)
        #     figs.append(fig)
        return workloads, figs

