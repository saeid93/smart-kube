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
import matplotlib.pyplot as plt
matplotlib.use("Agg")
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
    add_path_to_config,
    action_pretty_print,
    config_check_env_check
)


def check_env(*, config: Dict[str, Any], type_env: str,
              cluster_id: int, workload_id: int):

    env_config_base = config["env_config_base"]
    env_config = add_path_to_config(
        config=env_config_base,
        cluster_id=cluster_id,
        workload_id=workload_id
    )
    if type_env not in ['CartPole-v0', 'Pendulum-v0']:
        type_env = ENVSMAP[type_env]
        env = gym.make(type_env, config=env_config)
    else:
        env = gym.make(type_env)

    i = 1
    total_timesteps = 10000
    _ = env.reset()

    rewards = []
    rewards_cv = []
    rewards_p = []
    rewards_g = []
    while i < total_timesteps:
        action = env.action_space.sample()
        # action = 1
        _, reward, done, info = env.step(action)
        if info['scheduling_timestep']:
            print('scheudling timestep')
            if info['scheduling_success']:
                print('scheduling successful :)')
            else:
                print('scheduling unsuccessful :(')
        rewards.append(reward)
        rewards_cv.append(info['rewards']['cv'])
        rewards_p.append(info['rewards']['p'])
        rewards_g.append(info['rewards']['g'])
        env.render()
        if env.time % 100 == 0:
            TEMP = 1
        print("time: {}".format(
            env.time
        ))
        print("timestep_episode: {}".format(
            env.timestep_episode
        ))
        print("reward: {}".format(
            reward
        ))
        print("rewards: {}".format(
            info['rewards']
        ))
        if done and not env.complete_done:
            _ = env.reset()
            print(20*'-' + ' Done for episode! ' + 20*'-')
        if done and env.complete_done:
            print( + 40*'=' + ' Done for ending the simulation ' + 40*'=')
            break
        i += 1
    x = np.arange(len(rewards))
    plt.plot(x, np.array(rewards), label = "Total Rewards")
    plt.plot(x, np.array(rewards_cv), label = "Variance reward")
    plt.plot(x, np.array(rewards_p), label = "Consolidation reward")
    plt.xlabel('Timesteps')
    plt.ylabel('Reward Value')
    # plt.plot(x, np.array(rewards_g), label = "G")
    plt.legend()
    plt.grid()
    plt.savefig(f'rewards-{cluster_id}.png')

@click.command()
@click.option('--type-env', required=True,
              type=click.Choice(['sim-scheduler', 'sim-binpacking',
                                 'kube-scheduler', 'kube-binpacking',
                                 'CartPole-v0', 'Pendulum-v0']),
              default='sim-scheduler')
@click.option('--cluster-id', required=True, type=int, default=21)
@click.option('--workload-id', required=True, type=int, default=2)
def main(type_env: str, cluster_id: int, workload_id: int):
    """[summary]

    Args:
        type_env (str): the type of the used environment
        cluster_id (int): the cluster metadata (size, #nodes, etc)
        workload_id (int): the workload used in that cluster
    """
    config_file_path = os.path.join(
        CONFIGS_PATH, 'check',
        f'check_env.json')
    with open(config_file_path) as cf:
        config = json.loads(cf.read())

    config_check_env_check(config)
    check_env(config=config,
              type_env=type_env,
              cluster_id=cluster_id,
              workload_id=workload_id)


if __name__ == "__main__":
    main()
