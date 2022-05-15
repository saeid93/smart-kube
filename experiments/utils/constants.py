import os
from smart_scheduler.envs import (
    SimSchedulerEnv,
    SimBinpackingEnv,
    KubeSchedulerEnv,
    KubeBinpackingEnv,
)
# dfined by the user
DATA_PATH = "/homes/sg324/smart-vpa/smart-scheduler/data/"

# generated baesd on the users' path
CLUSTERS_PATH = os.path.join(DATA_PATH, "clusters")
TRAIN_RESULTS_PATH = os.path.join(DATA_PATH, "train-results")
TESTS_RESULTS_PATH = os.path.join(DATA_PATH, "test-results")

CONFIGS_PATH = os.path.join(DATA_PATH, "configs")
BACKUP_PATH = os.path.join(DATA_PATH, "backup")
PLOTS_PATH = os.path.join(DATA_PATH, "plots")
CLUSTERS_METADATA_PATH = os.path.join(DATA_PATH, "cluster_metadata") 

def _create_dirs():
    """
    create directories if they don't exist
    """
    if not os.path.exists(CLUSTERS_PATH):
        os.makedirs(CLUSTERS_PATH)
    if not os.path.exists(TRAIN_RESULTS_PATH):
        os.makedirs(TRAIN_RESULTS_PATH)
    if not os.path.exists(CONFIGS_PATH):
        os.makedirs(CONFIGS_PATH)
    if not os.path.exists(BACKUP_PATH):
        os.makedirs(BACKUP_PATH)
    if not os.path.exists(TESTS_RESULTS_PATH):
        os.makedirs(TESTS_RESULTS_PATH)
    if not os.path.exists(CLUSTERS_METADATA_PATH):
        os.makedirs(CLUSTERS_METADATA_PATH)
    if not os.path.exists(PLOTS_PATH):
        os.makedirs(PLOTS_PATH)

_create_dirs()

ENVS = {
    'sim-scheduler': SimSchedulerEnv,
    'sim-binpacking': SimBinpackingEnv,
    'kube-scheduler': KubeSchedulerEnv,
    'kube-binpacking': KubeBinpackingEnv,
}

ENVSMAP = {
    'sim-scheduler': 'SimSchedulerEnv-v0',
    'sim-binpacking': 'SimBinpackingEnv-v0',
    'kube-scheduler': 'KubeSchedulerEnv-v0',
    'kube-binpacking': 'KubeBinpackingEnv-v0',
    'kube-greedy': 'KubeGreedyEnv-v0',
}
