"""
read datasets, workloads etc
"""
import os
import pickle


def load_object(path: str):
    """
    read the dataset, workload and network from the disk
    and their path
    """

    # load dataset
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no data at <{path}>")
    with open(path, 'rb') as in_pickle:
        object = pickle.load(in_pickle)
    return object
