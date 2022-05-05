"""base class of all simulation enviornments
"""
import abc
import numpy as np
from copy import deepcopy
from typing import (
    List,
    Dict,
    Any,
    Literal,
    Tuple
)

from colorama import (
    Fore,
    Style
)
import gym
from gym.utils import seeding
import types
from pendulum import duration
from smart_scheduler import cluster

from smart_scheduler.util import (
    Preprocessor,
    override,
    rounding,
    check_config,
    load_object,
    ACTION_MIN,
    ACTION_MAX,
    plot_resource_allocation,
    Discrete2MultiDiscrete,
    logger
)
from smart_scheduler.envs_extensions import (
    _reward
)

from gym.spaces import (
    Box,
    MultiDiscrete,
    Discrete
)

from smart_scheduler.util import (
    get_random_string
)
from smart_scheduler.cluster import (
    Service,
    Node,
    Cluster
)

class SimSchedulerEnv(gym.Env):
    """
    what differs between different enviornments
    1. States
    2. Actions and related functions
    3. Reward

    common variables:

        services_nodes:

             service_id     service_id            service_id
            [node_id,       node_id,    , ... ,   node_id     ]

            range:
                indices: [0, num_services)
                contents: [0, num_nodes)

    Remarks:
            the main indicator of the state (obseravation) if services_nodes
            the rest of the observation dictionary is updated with decorators
            automatically
    """

    # ------------------ common functions ------------------

    def __init__(self, config: Dict[str, Any]):
        # action min and max
        self.action_min, self.action_max = (
            ACTION_MIN, ACTION_MAX
        )

        # initialize seed to ensure reproducible resutls
        self.seed(config['seed'])
        check_config(config)

        # observation elements
        self.obs_elements: List[str] = config['obs_elements']

        # path of the cluster and workload
        self.cluster_path = config['cluster_path']
        self.workload_path = config['workload_path']

        # node, services resources and workload
        cluster_schema = load_object(self.cluster_path)
        self.workload_save = load_object(self.workload_path)

        # TODO different for other types of data
        sim_type = self.workload_save['workload_type']
        self.workload = self.workload_save['workloads']

        self.cluster = Cluster(cluster_schema)

        self.total_timesteps: int = self.workload.shape[1]


        self.services_resources_request: np.ndarray = cluster_schema[
            'services_resources_request']
        self.services_types: np.ndarray = cluster_schema['services_types']
        self.total_num_services: int = self.services_resources_request.shape[0]

        # reward penalties
        self.penalty_illegal: float = config['penalty_illegal']
        self.penalty_move: float = config['penalty_move']
        self.penalty_consolidated: float = config['penalty_consolidated']

        # episode length
        self.episode_length: int = config['episode_length']

        # whether to reset timestep and placement at every episode
        if 'timestep_reset' in config:
            self.timestep_reset: bool = config['timestep_reset']
        else:
            self.timestep_reset: bool = False
        if 'placement_reset' in config:
            self.placement_reset: bool = config['placement_reset']
        else:
            self.placement_reset: bool = False
        self.global_timestep: int = 0
        self.timestep: int = 0

        # set the reward method
        self._reward = types.MethodType(_reward, self)

        # value based methods needs to have a convertor of
        # discrete state to multidiscrete
        if 'discrete_actions' in config:
            self.discrete_actions: bool = config['discrete_actions']
        else:
            self.discrete_actions: bool = False

        # whether to take the overloaded action with negative reward or not
        self.no_action_on_overloaded = config['no_action_on_overloaded']

        # reward weighting variables
        self.consolidation_lower = config['consolidation_lower']
        self.consolidation_upper = config['consolidation_upper']

        self.pending_services: List[Service] = []

        # make services objects
        for service_id in range(self.total_num_services):
            if sim_type=='alibaba' or sim_type=='arabesque':
                pass # TODO
            else:
                service_workload = self.workload[
                    :, :, self.services_types[0]] * np.reshape(
                        self.services_resources_request[0], (2,1))
                service_name = get_random_string(6)
                serving_time = np.random.randint(
                    5, service_workload.shape[1])
            self.pending_services.append(Service(
                service_id=service_id,
                service_name=service_name,
                requests=self.services_resources_request[service_id],
                limits=self.services_resources_request[service_id],
                workload=service_workload,
                serving_time=serving_time))

        self.backlog = config['backlog']

        # TODO TEMP remove
        self.schedule(
            service_id=0,
            node_id=0
        )
        self.schedule(
            service_id=2,
            node_id=0
        )
        self.schedule(
            service_id=3,
            node_id=1
        )
        self.schedule(
            service_id=4,
            node_id=1
        )
        self.observation_space, self.action_space =\
            self._setup_space()
        _ = self.reset()

    def seed(self, seed):
        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        self._env_seed = seed
        self.base_env_seed = seed
        return [seed]

    @override(gym.Env)
    def reset(self) -> np.ndarray:
        """Resets the state of the environment and returns
        an initial observation.
        Returns:
            (object): the initial observation.
        Remarks:
            each time resets to a different initial state
        """
        if self.timestep_reset:
            self.global_timestep = 0
            self.timestep = 0
        if self.placement_reset:
            self.services_nodes = deepcopy(self.initial_services_nodes)
        return self.observation

    @override(gym.Env)
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

        # TODO clock tick for nodes (services are updated inside)

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


    def render(self, mode: Literal['human', 'ansi'] ='human') -> None:
        """
        """
        print("--------state--------")
        if not self.num_overloaded:
            print("nodes_resources_request_frac:")
            print(self.nodes_resources_request_frac)
            print("services_nodes:")
            print(self.services_nodes)
            if mode == 'ansi':
                plot_resource_allocation(self.services_nodes,
                                        self.nodes_resources_cap,
                                        self.services_resources_request,
                                        self.services_resources_usage,
                                        plot_length=80)
        else:
            print(Fore.RED, "agent's action lead to an overloaded state!")
            print("nodes_resources_usage_frac:")
            print(self.nodes_resources_request_frac)
            print("services_nodes:")
            print(self.services_nodes)
            if mode == 'ansi':
                plot_resource_allocation(self.services_nodes,
                                        self.nodes_resources_cap,
                                        self.services_resources_request,
                                        self.services_resources_usage,
                                        plot_length=80)
            print(Style.RESET_ALL)


    def schedule(self, service_id: int, node_id: int) -> bool: # TODO part of cluster's logic move the rest to the step function
        """schedule one of the services on a target node

        Args:
            service_id (int): id of the service
                to be scheduled
            node_id (int): id of the node
                to be scheduled

        Returns:
            bool: returns whether the service
                has been scheduled or not
        """
        if not service_id in self.pending_services_ids:
            raise ValueError(
                'Service {} does not exists in pending services'.format(
                service_id
            ))
        service_index = self.pending_services_ids.index(service_id)
        schedule_success = self.cluster.schedule(
            service=self.pending_services[service_index],
            node_id=node_id
        )
        # remove the service from the pending services if successful
        if schedule_success:
            self.pending_services.pop(service_index)
        # return true if successful
        return schedule_success


    @property
    def pending_services_ids(self) -> List[int]:
        ids = list(
            map(
                lambda service: service.service_id, self.pending_services))
        return ids

    @property
    def raw_observation(self) -> Dict[str, np.ndarray]:
        """returns only the raw observations requested by the user
        in the config input through obs_elements
        """
        observation = {
                "nodes_capacities": self.cluster.nodes_capacities,
                "nodes_usages": self.cluster.nodes_capacities,
                "nodes_requests": self.cluster.nodes_requests,
                "nodes_available": self.cluster.nodes_available,
                "nodes_unused": self.cluster.nodes_available,
                "nodes_slack": self.cluster.nodes_slack,
                "nodes_usages_frac": self.cluster.nodes_usages_frac,
                "nodes_requests_frac": self.cluster.nodes_requests_frac,
                "num_consolidated": self.cluster.num_consolidated,
                "nodes_requests_available_frac": self.cluster.nodes_requests_available_frac,
                "nodes_resources_unused_frac": self.cluster.nodes_resources_unused_frac,
                "nodes_requests_available_frac_avg": self.cluster.nodes_requests_available_frac_avg,
                "nodes_resources_unused_avg": self.cluster.nodes_resources_unused_avg

        }
        # TODO for now I have chosen just some of them as example
        # add more after finalized on satate-space
        selected = dict(zip(self.obs_elements,
                            [observation[k] for k in self.obs_elements]))
        return selected

    @property
    def observation(self) -> np.ndarray:
        """preprocessed observation of each environment
        """
        obs = self.preprocessor(self.raw_observation)
        obs = np.array(list(map(int, obs)))
        return obs


    def preprocessor(self, obs):
        """
        environment preprocessor
        depeiding on the observation (state) definition
        """
        prep = Preprocessor(self.nodes_resources_cap,
                            self.services_resources_request)
        obs = prep.transform(obs)
        return obs

    def _setup_space(self): # TODO change based on new need
        """
        """
        # TODO change action-space to arriving services
        # numuber of elements based on the obs in the observation space
        obs_size = 0
        for elm in self.obs_elements:
            if elm == "nodes_requests":
                obs_size += self.cluster.num_nodes * self.cluster.num_resources
            elif elm == "nodes_usages":
                obs_size += self.cluster.num_nodes * self.cluster.num_resources
            elif elm == "nodes_slack":
                obs_size += self.cluster.num_nodes * self.cluster.num_resources
        # TODO for now I have chosen just some of them as example
        # add more after finalized on satate-space

        higher_bound = 20 # TEMP just for test - find a cleaner way
        # generate observation and action spaces
        observation_space = Box(
            low=0, high=higher_bound, shape=(obs_size, ),
            dtype=np.float64, seed=self._env_seed)

        if self.discrete_actions:
            action_space = Discrete(
                self.cluster.num_nodes, seed=self._env_seed)
            # self.discrete_action_converter = Discrete2MultiDiscrete(
            #     self.back, self.total_num_services)
        else:
            action_space = MultiDiscrete(
                np.ones(self.backlog)*self.cluster.num_nodes, seed=self._env_seed)

        return observation_space, action_space