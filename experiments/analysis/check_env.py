import os
import sys
import gym
import click
import matplotlib
import numpy as np
from typing import Dict, Any
import time
import json
from pprint import PrettyPrinter
# import matplotlib.pyplot as plt
# matplotlib.use("Agg")
pp = PrettyPrinter(indent=4)


# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))

from experiments.utils.constants import (
    ENVSMAP,
    CONFIGS_PATH,
    DATA_PATH
)
from experiments.utils import (
    add_path_to_config_edge,
    action_pretty_print,
    config_check_env_check
)


def check_env(*, config: Dict[str, Any], type_env: str,
              cluster_id: int, workload_id: int):

    env_config_base = config["env_config_base"]
    env_config = add_path_to_config_edge(
        config=env_config_base,
        cluster_id=cluster_id,
        workload_id=workload_id,
    )
    if type_env not in ['CartPole-v0', 'Pendulum-v0']:
        type_env = ENVSMAP[type_env]
        env = gym.make(type_env, config=env_config)
    else:
        env = gym.make(type_env)

    i = 1
    total_timesteps = 1000
    _ = env.reset()


    consolidation_rewards = []
    reward_total = []
    users_distances = []
    episode_total_latency_reward = 0
    episode_total_latency_negative_reward = 0
    episode_total_consolidation_reward = 0
    while i < total_timesteps:
        action = env.action_space.sample()
        _, reward, done, info = env.step(action)
        consolidation_reward = info['rewards']['reward_consolidation']
        consolidation_rewards.append(consolidation_reward)
        reward_total.append(reward)
        episode_total_consolidation_reward += consolidation_reward
        print(f'episode_total_latency_reward: {episode_total_latency_reward}')
        print(f'episode_total_latency_negative_rewards: {episode_total_latency_negative_reward}')
        print(f'episode_total_consolidation_reward: {episode_total_consolidation_reward}')
    x = np.arange(total_timesteps-1)

@click.command()
@click.option('--type-env', required=True,
              type=click.Choice(['sim-scheduler', 'sim-binpacking',
                                 'kube-scheduler', 'kube-binpacking',
                                 'CartPole-v0', 'Pendulum-v0']),
              default='sim-scheduler')
@click.option('--cluster-id', required=True, type=int, default=2)
@click.option('--workload-id', required=True, type=int, default=0)
def main(type_env: str, cluster_id: int, workload_id: int):
    """[summary]

    Args:
        type_env (str): the type of the used environment
        cluster_id (int): used cluster cluster
        workload_id (int): the workload used in that cluster
    """
    config_file_path = os.path.join(
        CONFIGS_PATH, 'check',
        'check_env.json')
    with open(config_file_path) as cf:
        config = json.loads(cf.read())

    config_check_env_check(config)
    check_env(config=config,
              type_env=type_env,
              cluster_id=cluster_id,
              workload_id=workload_id)


if __name__ == "__main__":
    main()
