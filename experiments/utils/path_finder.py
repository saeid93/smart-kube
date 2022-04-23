"""
change the ids to full paths
"""
import json
import os
import pickle
import sys
import numpy as np
from typing import Union, Any, Dict

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))

from experiments.utils.constants import DATASETS_PATH


def add_path_to_config_edge(
    config: Dict[str, Any], dataset_id: int, workload_id: int,
    network_id: int,trace_id: int) -> Dict[str, Any]:

    dataset_path = os.path.join(DATASETS_PATH, str(dataset_id))
    workload_path = os.path.join(dataset_path, 'workloads', str(workload_id))
    network_path = os.path.join(dataset_path, 'networks', str(network_id))
    trace_path = os.path.join(network_path, 'traces', str(trace_id))
    config.update({
        'dataset_path': os.path.join(dataset_path, 'dataset.pickle'),
        'workload_path': os.path.join(workload_path, 'workload.pickle'),
        'network_path':\
            os.path.join(network_path, 'edge_simulator_config.pickle'),
        'trace_path': os.path.join(trace_path, 'trace.pickle')})
    return config
