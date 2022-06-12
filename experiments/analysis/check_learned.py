"""
scripts to check a learned agent
based-on:
https://github.com/ray-project/ray/issues/9123
https://github.com/ray-project/ray/issues/7983
"""

# TODO Refine completely based on the new paper

import os
import sys
import numpy as np
import click
from typing import Dict, Any
import json

import ray
from ray import tune
from ray.rllib.utils.framework import try_import_torch
import pprint
import gym
import ray.rllib.agents.ppo as ppo
import ray.rllib.agents.a3c as a3c
import ray.rllib.agents.impala as impala
import ray.rllib.agents.pg as pg
import ray.rllib.agents.dqn as dqn
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)
from copy import deepcopy

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))

from experiments.utils.constants import (
    TRAIN_RESULTS_PATH,
    ENVSMAP
)
from experiments.utils import (
    add_path_to_config,
    make_env_class,
    CloudCallback
)

torch, nn = try_import_torch()


def learner(*, series: int, type_env: str, cluster_id: int,
            workload_id: int, checkpoint: str, experiment_id: int,
            local_mode: bool, episode_length: int,
            workload_id_test: int):
    """
    """
    path_env = type_env if type_env != 'kube-scheduler' else 'sim-scheduler'    
    experiments_config_path = os.path.join(
        TRAIN_RESULTS_PATH,
        "series",      str(series),
        "envs",        path_env,
        "clusters",    str(cluster_id),
        "workloads",   str(workload_id),
        "experiments", str(experiment_id),
        "experiment_config.json")

    with open(experiments_config_path) as cf:
        config = json.loads(cf.read())

    pp = pprint.PrettyPrinter(indent=4)
    print('start experiments with the following config:\n')
    pp.pprint(config)

    # extract differnt parts of the input_config
    learn_config = config['learn_config']
    algorithm = config["run_or_experiment"]
    env_config_base = config['env_config_base']

    # add evn_config_base updates
    env_config_base.update({
        'episode_length': episode_length,
        'no_action_on_overloaded': True
    })

    # add the additional nencessary arguments to the edge config
    env_config = add_path_to_config(
        config=env_config_base,
        cluster_id=cluster_id,
        workload_id=workload_id_test
    )

    # trained ray agent should always be simulation
    # however the agent outside it can be kuber agent or
    # other types of agent
    if type_env not in ['CartPole-v0', 'Pendulum-v0']:
        env = gym.make(ENVSMAP[type_env], config=env_config)
        ray_config = {"env": make_env_class('sim-scheduler'),
                    "env_config": env_config}
        ray_config.update(learn_config)
    else:
        ray_config = {"env": type_env}
        env = gym.make(type_env)
        ray_config.update(learn_config)

    path_env = type_env if type_env != 'kube-scheduler' else 'sim-scheduler'
    experiments_folder = os.path.join(TRAIN_RESULTS_PATH,
                                      "series",      str(series),
                                      "envs",        path_env,
                                      "clusters",    str(cluster_id),
                                      "workloads",   str(workload_id),
                                      "experiments", str(experiment_id),
                                      algorithm)
    for item in os.listdir(experiments_folder):
        if 'json' not in item:
            experiment_string = item
            break

    # make checkpoint folder
    # the checkpoint folder always should have six digits
    checkpoint_folder =\
        f"checkpoint_{'0' * (6 - len(checkpoint))}{checkpoint}"

    checkpoint_path = os.path.join(
        experiments_folder,
        experiment_string,
        checkpoint_folder,
        f"checkpoint-{checkpoint}"
    )

    ray.init(local_mode=local_mode)

    # TODO fix here
    if type_env not in ['CartPole-v0', 'Pendulum-v0']:
        alg_env = make_env_class(type_env)
    else:
        alg_env = type_env
    if algorithm == 'PPO':
        agent = ppo.PPOTrainer(
            config=ray_config,
            env=alg_env)
    if algorithm == 'IMPALA':
        agent = impala.ImpalaTrainer(
            config=ray_config,
            env=alg_env)
    elif algorithm == 'A3C' or algorithm == 'A2C':
        agent = a3c.A3CTrainer(
            config=ray_config,
            env=alg_env)
    elif algorithm == 'PG':
        agent = pg.PGTrainer(
            config=ray_config,
            env=alg_env)
    elif algorithm == 'DQN':
        agent = dqn.DQNTrainer(
            config=ray_config,
            env=alg_env)

    agent.restore(checkpoint_path=checkpoint_path)
    episode_reward = 0
    done = False
    obs = env.reset()
    env.render()
    action = 1
    while not done:
        prev_obs = deepcopy(obs)
        prev_action = deepcopy(action)
        action = agent.compute_action(obs)
        print("\n\n--------action--------")
        print(action)
        obs, reward, done, info = env.step(action)
        env.render()
        print('info:')
        pp.pprint(info)
        episode_reward += reward
        # if not np.alltrue(prev_obs==obs):
        #     a = 1
        # if not np.alltrue(prev_action==action):
        #     b = 1
    print(f"episode reward: {episode_reward}")

# /homes/sg324/smart-vpa/smart-scheduler/data/train-results/series/1/envs/
# sim-scheduler/clusters/0/workloads/0/experiments/0/PPO/
# PPO_SimSchedulerEnv_744f8_00000_0_2022-06-11_18-03-45/checkpoint_000100

@click.command()
@click.option('--local-mode', type=bool, default=True)
@click.option('--series', required=True, type=int, default=1)
@click.option('--type-env', required=True,
              type=click.Choice(['sim-scheduler', 'kube-scheduler',
                                 'CartPole-v0', 'Pendulum-v0']),
              default='sim-scheduler')
@click.option('--cluster-id', required=True, type=int, default=0)
@click.option('--workload-id', required=True, type=int, default=0)
@click.option('--experiment_id', required=True, type=int, default=0)
@click.option('--checkpoint', required=False, type=str, default="100")
@click.option('--episode-length', required=False, type=int, default=10)
@click.option('--workload-id-test', required=False, type=int, default=0)
def main(local_mode: bool, series: int,
         type_env: str, cluster_id: int, workload_id: int,
         experiment_id: int, checkpoint: int, episode_length: int,
         workload_id_test: int):
    """[summary]
    Args:
        local_mode (bool): run in local mode for having the 
        config_folder (str): name of the config folder (only used in real mode)
        use_callback (bool): whether to use callbacks or storing and visualising
        checkpoint (int): selected checkpoint to test
        series (int): to gather a series of clusters in a folder
        type_env (str): the type of the used environment
        cluster_id (int): used cluster cluster
        workload_id (int): the workload used in that cluster
    """

    learner(series=series, type_env=type_env,
            cluster_id=cluster_id, workload_id=workload_id,
            experiment_id=experiment_id, checkpoint=checkpoint,
            local_mode=local_mode, episode_length=episode_length,
            workload_id_test=workload_id_test)


if __name__ == "__main__":
    main()