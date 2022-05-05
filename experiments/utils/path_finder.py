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

from experiments.utils.constants import CLUSTERS_PATH


def add_path_to_config(
    config: Dict[str, Any], cluster_id: int, workload_id: int) -> Dict[str, Any]:

    cluster_path = os.path.join(CLUSTERS_PATH, str(cluster_id))
    workload_path = os.path.join(cluster_path, 'workloads', str(workload_id))
    config.update({
        'cluster_path': os.path.join(cluster_path, 'cluster.pickle'),
        'workload_path': os.path.join(workload_path, 'workload.pickle')})
    return config
