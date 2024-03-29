"""
read clusters, workloads etc
"""
from typing import (
    Dict,
    Union
)
import os
import pickle
import numpy as np


def load_object(path: str):
    """
    read the cluster, workload and network from the disk
    and their path
    """

    # load cluster
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no data at <{path}>")
    with open(path, 'rb') as in_pickle:
        object = pickle.load(in_pickle)
    return object
