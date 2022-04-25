"""
Testing phase of the experiments on the test data
"""

# TODO add kubernetes

import os
import sys
import pickle
import click
from typing import Dict, Any
import json
from gym.core import Env
from ray.rllib.utils.framework import try_import_torch
import pprint
import gym
import pandas as pd
from pprint import PrettyPrinter
pp = PrettyPrinter(indent=4)

# get an absolute path to the directory that contains parent files
project_dir = os.path.dirname(os.path.join(os.getcwd(), __file__))
sys.path.append(os.path.normpath(os.path.join(project_dir, '..', '..')))


from experiments.utils.constants import (
    TRAIN_RESULTS_PATH,
    TESTS_RESULTS_PATH,
    ENVSMAP
)
from experiments.utils import (
    add_path_to_config_edge,
    make_env_class
)

torch, nn = try_import_torch()



def run_experiments(
    *, test_series: int, train_series: int, type_env: str,
    cluster_id: int, workload_id: int, network_id: int,
    trace_id: int, trace_id_test: int,
    comp_experiment_id: int, episode_length):
    """
    """
    experiments_config_folder = os.path.join(
        TRAIN_RESULTS_PATH,
        "series",      str(train_series),
        "envs",        'sim-scheduler',
        "clusters",    str(cluster_id),
        "workloads",   str(workload_id),
        "networks",    str(network_id),
        "traces",      str(trace_id),
        "experiments", str(comp_experiment_id),
        "experiment_config.json")

    with open(experiments_config_folder) as cf:
        config = json.loads(cf.read())

    pp = pprint.PrettyPrinter(indent=4)
    print('start experiments with the following config:\n')
    pp.pprint(config)

    # extract differnt parts of the input_config
    algorithm = {
        'sim-binpacking': 'binpacking',
        'kube-binpacking': 'binpacking',
        'sim-greedy': 'latency_greedy',
        'kube-greedy': 'latency_greedy'}[type_env]
    env_config_base = config['env_config_base']

    # add evn_config_base updates
    env_config_base.update({
        'episode_length': episode_length,
        'no_action_on_overloaded': True,
        'timestep_reset': True,
        'placement_reset': True
    })

    # add the additional nencessary arguments to the edge config
    env_config = add_path_to_config_edge(
        config=env_config_base,
        cluster_id=cluster_id,
        workload_id=workload_id,
        network_id=network_id,
        trace_id=trace_id_test
    )
    fix_grid_searches(env_config)

    # trained ray agent should always be simulation
    # however the agent outside it can be kuber agent or
    # other types of agent
    if type_env not in ['CartPole-v0', 'Pendulum-v0']:
        env = gym.make(ENVSMAP[type_env], config=env_config)
    else:
        env = gym.make(type_env)

    episode_reward = 0
    done = False
    states = []
    _ = env.reset()
    while not done:
        action = env.action_space.sample()
        obs, reward, done, info = env.step(action)
        state = flatten(env.raw_observation, action, reward, info)
        states.append(state)
        episode_reward += reward
    states = pd.DataFrame(states)
    print(f"episode reward: {episode_reward}")
    info = {
        'cluster_id': cluster_id,
        'workload_id': workload_id,
        'network_id': network_id,
        'trace_id': trace_id,
        'trace_id_test': trace_id_test,
        'experiment_id': comp_experiment_id,
        'episode_length': episode_length,
        'algorithm': algorithm,
        'penalty_latency': env_config['penalty_latency'],
        'penalty_consolidated': env_config['penalty_consolidated']
    }
    # make the new experiment folder
    test_series_path = os.path.join(
        TESTS_RESULTS_PATH,
        'series', str(test_series),
        'tests')
    if not os.path.isdir(test_series_path):
        os.makedirs(test_series_path)
    content = os.listdir(test_series_path)
    new_test = len(content)
    this_test_folder = os.path.join(test_series_path,
                                    str(new_test))
    os.makedirs(this_test_folder)

    # save the necesarry information
    with open(os.path.join(this_test_folder, 'info.json'), 'x') as out_file:
        json.dump(info, out_file, indent=4)
    with open(os.path.join(
        this_test_folder, 'episodes.pickle'), 'wb') as out_pickle:
        pickle.dump([states], out_pickle)

def flatten(raw_obs, action, reward, info):
    return {
        'action': action,
        'services_nodes': raw_obs['services_nodes'],
        'users_stations': raw_obs['users_stations'],
        'num_consolidated': info['num_consolidated'],
        'num_moves': info['num_moves'],
        'num_overloaded': info['num_overloaded'],
        'users_distances': info['users_distances'],
        'reward_latency': info['rewards']['reward_latency'],
        'reward_move': info['rewards']['reward_move'],
        'reward_illegal': info['rewards']['reward_illegal'],
        'reward_variance': info['rewards']['reward_variance'],
        'reward': reward
    }

def fix_grid_searches(config):
    values = []
    for k, v in config.items():
        if type(v) == dict:
            values = v['grid_search']
            config[k] = values[-1]
            break
    return config



@click.command()
@click.option('--test-series', required=True, type=int, default=69)
@click.option('--comp-train-series', required=True, type=int, default=69)
@click.option('--type-env', required=True,
              type=click.Choice(['sim-binpacking', 'sim-greedy',
                                 'kube-binpacking', 'kube-greedy']),
              default='sim-greedy')
@click.option('--cluster-id', required=True, type=int, default=6)
@click.option('--workload-id', required=True, type=int, default=0)
@click.option('--network-id', required=False, type=int, default=7)
@click.option('--trace-id', required=False, type=int, default=2)
@click.option('--trace-id-test', required=False, type=int, default=0)
@click.option('--comp-experiment-id', required=True, type=int, default=0)
@click.option('--episode-length', required=False, type=int, default=3453)
def main(test_series: int, comp_train_series: int, type_env: str,
         cluster_id: int, workload_id: int, network_id: int,
         trace_id: int, trace_id_test: int, comp_experiment_id: int,
         episode_length: int):
    """[summary]

    Args: 
        test-series (int): series of the tests
        comp-train-series (int): series of the trainining phase for loading config
        test-series (int): testing series to save
        type_env (str): the type of the used environment
        cluster_id (int): used cluster cluster
        workload_id (int): the workload used in that cluster
        network_id (int): edge network of some cluster
        trace_id (int): user movement traces
        comp-experiment-id (int): the trained agent experiment-id
                    that we want to compare against
        episode-length (int): number of steps in the test episode
    """

    run_experiments(
        test_series=test_series,
        train_series=comp_train_series, type_env=type_env,
        cluster_id=cluster_id, workload_id=workload_id,
        network_id=network_id, trace_id=trace_id,
        trace_id_test=trace_id_test,
        comp_experiment_id=comp_experiment_id,
        episode_length=episode_length)


if __name__ == "__main__":
    main()
