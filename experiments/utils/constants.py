import os
from smart_scheduler.envs import (
    SimSchedulerEnv,
    KubeSchedulerEnv,
)
# dfined by the user
DATA_PATH = "/homes/sg324/smart-scheduler/data/"
# DATA_PATH = "/home/user/smart-scheduler/data"

# generated baesd on the users' path
CLUSTERS_PATH = os.path.join(DATA_PATH, "clusters")
TRAIN_RESULTS_PATH = os.path.join(DATA_PATH, "train-results")
TESTS_RESULTS_PATH = os.path.join(DATA_PATH, "test-results")

CONFIGS_PATH = os.path.join(DATA_PATH, "configs")
BACKUP_PATH = os.path.join(DATA_PATH, "backup")
PLOTS_PATH = os.path.join(DATA_PATH, "plots")
ARABESQUE_PATH = os.path.join(DATA_PATH, "arabesque")
ALIBABA_PATH = os.path.join(DATA_PATH, "alibaba")

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
    if not os.path.exists(PLOTS_PATH):
        os.makedirs(PLOTS_PATH)

# _create_dirs()

ENVS = {
    'sim-scheduler': SimSchedulerEnv,
    'kube-scheduler': KubeSchedulerEnv,
}

ENVSMAP = {
    'sim-scheduler': 'SimSchedulerEnv-v0',
    'sim-binpacking': 'SimBinpackingEnv-v0',
    'kube-scheduler': 'KubeSchedulerEnv-v0',
    'kube-binpacking': 'KubeBinpackingEnv-v0',
    'kube-greedy': 'KubeGreedyEnv-v0',
}
