"""base class of edge enviornments
"""
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

from .kube_base_env import KubeBaseEnv

class KubeSchedulerEnv(KubeBaseEnv):
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)

        self.observation_space, self.action_space =\
            self._setup_space()
        # _ = self.reset()

    @override(KubeBaseEnv)
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
        obs_size += (self.num_users) * self.num_stations

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

        return observation_space, action_space

    def preprocessor(self, obs):
        """
        environment preprocessor
        depeiding on the observation (state) definition
        """
        prep = Preprocessor(self.nodes_resources_cap,
                            self.services_resources_request,
                            self.num_stations)
        obs = prep.transform(obs)
        return obs

    @property
    def raw_observation(self) -> Dict[str, np.ndarray]:
        """
        returns only the raw observations requested by the user
        in the config input through obs_elements
        """
        observation = super().raw_observation
        observation.update({'users_stations': self.users_stations})
        return observation

    @override(KubeBaseEnv)
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

        self.services_nodes = deepcopy(action)
        if self.no_action_on_overloaded and self.num_overloaded > 0:
            print("overloaded state, reverting back ...")
            self.services_nodes = deepcopy(prev_services_nodes)

        # add migraion here
        self.services_nodes_obj =  self.nodes[action]
        self._migrate(self.services_nodes_obj)

        # move to the next timestep
        self.global_timestep += 1
        self.timestep = self.global_timestep % self.workload.shape[1]

        # make user movements --> network parts
        self.users_stations = self.edge_simulator.sample_users_stations(
            timestep=self.timestep)
        users_distances = self.edge_simulator.users_distances
        # update network with the new placements
        self.edge_simulator.update_services_nodes(self.services_nodes)

        num_moves = len(np.where(
            self.services_nodes != prev_services_nodes)[0])

        reward, rewards = self._reward(
            num_overloaded=self.num_overloaded,
            users_distances=users_distances,
            num_moves=num_moves
            )

        info = {'num_consolidated': self.num_consolidated,
                'num_moves': num_moves,
                'num_overloaded': self.num_overloaded,
                'users_distances': np.sum(users_distances),
                'total_reward': reward,
                'timestep': self.timestep,
                'global_timestep': self.global_timestep,
                'rewards': rewards,
                'seed': self.base_env_seed}

        assert self.observation_space.contains(self.observation),\
                (f"observation:\n<{self.raw_observation}>\noutside of "
                f"observation_space:\n <{self.observation_space}>")

        return self.observation, reward, self.done, info
