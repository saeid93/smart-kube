"""base class of edge enviornments
"""
from random import seed
import numpy as np
from copy import deepcopy
from typing import (
    Dict,
    Any,
    Tuple
)

from gym.spaces import (
    Box,
    MultiDiscrete,
    Discrete
)


from smart_scheduler.util import (
    Preprocessor,
    override,
    load_object,
    Discrete2MultiDiscrete,
    logger
)

from .sim_base_env import SimBaseEnv

class SimEdgeEnv(SimBaseEnv):
    def __init__(self, config: Dict[str, Any]):
        # the network for edge simulators
        edge_simulator_config = load_object(config['network_path'])
        trace = load_object(config['trace_path'])

        # self.latency_reward_option = config['latency_reward_option']
        self.latency_lower = config['latency_lower']
        self.latency_upper = config['latency_upper']
        self.consolidation_lower = config['consolidation_lower']
        self.consolidation_upper = config['consolidation_upper']
        # self.normalise_latency = config['normalise_latency']
        # self.normalise_factor = self.edge_simulator.get_largest_station_node_path()
        super().__init__(config)

        self.observation_space, self.action_space =\
            self._setup_space()
        _ = self.reset()

    @override(SimBaseEnv)
    def _setup_space(self):
        """
        States:
            the whole or a subset of the following dictionary:
            observation = {
                    "services_resources_usage":
                        self.services_resources_usage,
                    "nodes_resources_usage":
                        self.nodes_resources_usage,
                    "services_resources_frac":
                        self.services_resources_frac,
                    "nodes_resources_frac":
                        self.nodes_resources_frac,
                    "services_nodes":
                        self.services_nodes,
                    "users_stations": --> always in the observation
                        self.users_stations
            }
        users_stations:
             user_id        user_id                 user_id
            [station_id,    station_id,    , ... ,  station_id  ]
                        range:
                            indices: [0, num_users)
                            contents: [0, num_stations)
        Actions:
                              nodes
            services [                   ]
        """
        # numuber of elements based on the obs in the observation space
        obs_size = 0
        for elm in self.obs_elements:
            if elm == "services_resources_usage":
                obs_size += self.num_services * self.num_resources
            elif elm == "nodes_resources_usage":
                obs_size += self.num_nodes * self.num_resources
            elif elm == "services_resources_usage_frac":
                obs_size += self.num_services * self.num_resources
            elif elm == "nodes_resources_usage_frac":
                obs_size += self.num_nodes * self.num_resources
            elif elm == "services_nodes":
                # add the one hot endoded services_resources
                # number of elements
                obs_size += (self.num_nodes) * self.num_services

        # add the one hot endoded users_stations
        # number of elements

        higher_bound = 10 # TODO TEMP just for test - find a cleaner way
        # generate observation and action spaces
        observation_space = Box(
            low=0, high=higher_bound, shape=(obs_size, ),
            dtype=np.float64, seed=self._env_seed)

        if self.discrete_actions:
            action_space = Discrete(
                self.num_nodes**self.num_services, seed=self._env_seed)
            self.discrete_action_converter = Discrete2MultiDiscrete(
                self.num_nodes, self.num_services)
        else:
            action_space = MultiDiscrete(np.ones(self.num_services) *
                                        self.num_nodes, seed=self._env_seed)
        # action_space = Box(
        #     low=0, high=self.num_nodes-1, shape=(
        #         self.num_services,), dtype=int, seed=self._env_seed)

        return observation_space, action_space

    def preprocessor(self, obs):
        """
        environment preprocessor
        depeiding on the observation (state) definition
        """
        prep = Preprocessor(self.nodes_resources_cap,
                            self.services_resources_request)
        obs = prep.transform(obs)
        return obs

    @property
    def raw_observation(self) -> Dict[str, np.ndarray]:
        """
        returns only the raw observations requested by the user
        in the config input through obs_elements
        """
        observation = super().raw_observation
        return observation

    @override(SimBaseEnv)
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, int, bool, dict]:
        """
        edge servers here
        1. move the services based-on current network and nodes state
        2. do one step of user movements
        3. update the nodes
        """
        # take the action
        prev_services_nodes = deepcopy(self.services_nodes)
        assert self.action_space.contains(action)

        if self.discrete_actions:
            action = self.discrete_action_converter[action]

        # TODO not possible to roll back in the real world
        # take the action in the real world only if possible
        # simulation therefore should co-exist
        self.services_nodes = deepcopy(action)
        if self.no_action_on_overloaded and self.num_overloaded > 0:
            print("overloaded state, reverting back ...")
            self.services_nodes = deepcopy(prev_services_nodes)

        # move to the next timestep
        self.global_timestep += 1
        self.timestep = self.global_timestep % self.workload.shape[1]

        num_moves = len(np.where(
            self.services_nodes != prev_services_nodes)[0])

        reward, rewards = self._reward(
            num_overloaded=self.num_overloaded,
            num_moves=num_moves
            )

        info = {'num_consolidated': self.num_consolidated,
                'num_moves': num_moves,
                'num_overloaded': self.num_overloaded,
                'total_reward': reward,
                'timestep': self.timestep,
                'global_timestep': self.global_timestep,
                'rewards': rewards,
                'seed': self.base_env_seed}

        assert self.observation_space.contains(self.observation),\
                (f"observation:\n<{self.raw_observation}>\noutside of "
                f"observation_space:\n <{self.observation_space}>")

        return self.observation, reward, self.done, info
