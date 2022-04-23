import os
from smart_scheduler.envs import (
    SimEdgeEnv,
    SimBinpackingEnv,
    KubeEdgeEnv,
    KubeBinpackingEnv,
)
# dfined by the user
DATA_PATH = "/Users/saeid/Codes/smart-vpa/mobile-kube/data"

# generated baesd on the users' path
DATASETS_PATH = os.path.join(DATA_PATH, "datasets")
TRAIN_RESULTS_PATH = os.path.join(DATA_PATH, "train-results")
TESTS_RESULTS_PATH = os.path.join(DATA_PATH, "test-results")

CONFIGS_PATH = os.path.join(DATA_PATH, "configs")
BACKUP_PATH = os.path.join(DATA_PATH, "backup")
PLOTS_PATH = os.path.join(DATA_PATH, "plots")
DATASETS_METADATA_PATH = os.path.join(DATA_PATH, "dataset_metadata") 

def _create_dirs():
    """
    create directories if they don't exist
    """
    if not os.path.exists(DATASETS_PATH):
        os.makedirs(DATASETS_PATH)
    if not os.path.exists(TRAIN_RESULTS_PATH):
        os.makedirs(TRAIN_RESULTS_PATH)
    if not os.path.exists(CONFIGS_PATH):
        os.makedirs(CONFIGS_PATH)
    if not os.path.exists(BACKUP_PATH):
        os.makedirs(BACKUP_PATH)
    if not os.path.exists(TESTS_RESULTS_PATH):
        os.makedirs(TESTS_RESULTS_PATH)
    if not os.path.exists(DATASETS_METADATA_PATH):
        os.makedirs(DATASETS_METADATA_PATH)
    if not os.path.exists(PLOTS_PATH):
        os.makedirs(PLOTS_PATH)

_create_dirs()

ENVS = {
    'sim-edge': SimEdgeEnv,
    'sim-binpacking': SimBinpackingEnv,
    'kube-edge': KubeEdgeEnv,
    'kube-binpacking': KubeBinpackingEnv,
}

ENVSMAP = {
    'sim-edge': 'SimEdgeEnv-v0',
    'sim-binpacking': 'SimBinpackingEnv-v0',
    'sim-greedy': 'SimGreedyEnv-v0',
    'kube-edge': 'KubeEdgeEnv-v0',
    'kube-binpacking': 'KubeBinpackingEnv-v0',
    'kube-greedy': 'KubeGreedyEnv-v0',
}
