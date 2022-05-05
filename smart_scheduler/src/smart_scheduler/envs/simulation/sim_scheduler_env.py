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
        self.cluster = load_object(self.cluster_path)
        self.workload = load_object(self.workload_path)

        self.nodes_resources_cap: np.array = self.cluster['nodes_resources_cap']
        self.services_resources_request: np.array = self.cluster[
            'services_resources_request']
        self.services_types: np.array = self.cluster['services_types']

        # find the number of nodes, services, service types and timesteps
        self.num_resources: int = self.nodes_resources_cap.shape[1]
        self.num_nodes: int = self.nodes_resources_cap.shape[0]
        self.num_services: int = self.services_resources_request.shape[0]
        self.num_services_types: int = self.workload.shape[2]
        self.total_timesteps: int = self.workload.shape[1]

        # start and stop timestep
        stop_timestep: int = self.total_timesteps
        self.workload = self.workload[:, 0:stop_timestep, :]

        # initial states
        self.initial_services_nodes: np.array = self.cluster['services_nodes']

        # reward penalties
        self.penalty_illegal: float = config['penalty_illegal']
        self.penalty_move: float = config['penalty_move']
        self.penalty_variance: float = config['penalty_variance']
        self.penalty_consolidated: float = config['penalty_consolidated']
        self.penalty_latency: float = config['penalty_latency']

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
        self.services_nodes = deepcopy(self.initial_services_nodes)

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

        # TODO could be cleaner - come back to it if necessary
        self.pending_services: List[Service] = []
        self.nodes: List[Node] = []

        # make services objects
        for service_id in range(self.num_services):
            service_workload = self.workload[ # TODO an if statement here for check with random and real
                :, :, self.services_types[0]] * np.reshape(
                    self.services_resources_request[0], (2,1))
            self.pending_services.append(Service(
                service_id=service_id,
                service_name=get_random_string(6),
                requests=self.services_resources_request[service_id],
                limits=self.services_resources_request[service_id],
                workload=service_workload,
                duration=np.random.randint(
                    5, service_workload.shape[1])
            ))
        for node_id in range(self.num_nodes):
            self.nodes.append(Node(
                node_id=node_id,
                capacities=self.nodes_resources_cap[node_id], 
            ))
        # TODO start HERE
        # TEMP
        self.schedule(
            service_id=0,
            node_id=0
        )
        self.schedule(
            service_id=1,
            node_id=0
        )
        a1 = self.nodes[0].nodes_usage
        a2 = self.nodes[0].requests
        a3 = self.nodes[0].resources_available
        a5 = self.nodes[0].services_ids
        a5 = self.nodes[0].services_names
        a6 = self.nodes[0].requests_available

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


    def schedule(self, service_id: int, node_id: int) -> bool:
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
            raise ValueError('Service {} does not exists'.format(
                service_id
            ))
        service_index = self.pending_services_ids.index(service_id)
        # check if the node has enough request
        if np.alltrue(self.nodes[
            node_id].requests_available < self.pending_services[
                service_index].requests):
            return False
        # check if the node has enough resource available
        if np.alltrue(
            self.nodes[
                node_id].resources_available < np.zeros((2,1))):
            return False
        # schedule the service on the node
        self.nodes[node_id].add_service(
            self.pending_services[service_index])
        # remove the service from the pending services
        self.pending_services.pop(service_index)
        # return true if successful
        return True


    # ------------------ new properties ------------------
    @property
    def pending_services_ids(self) -> List[int]:
        ids = list(
            map(
                lambda service: service.service_id, self.pending_services))
        return ids

    # ------------------ old properties TODO to be changed ------------------

    @property
    @rounding
    def services_resources_usage(self) -> np.ndarray:
        # TODO query them from the nodes
        """return the fraction of resource usage for each node
        workload at current timestep e.g. at time step 0.
                         ram cpu
                        |       |
            services    |       |
                        |       |
            range:
                row inidices: (0, num_services]
                columns indices: (0, num_resources]
                enteries: [0, node_resource_cap] type: float
        """
        services_resources_usage = (self.services_resources_usage_frac *
                                      self.services_resources_request)
        return services_resources_usage

    @property
    def nodes_resources_usage(self): # TODO have two version for request and usage separately
        """return the amount of resource usage
        on each node
                     ram - cpu
                    |         |
            nodes   |         |
                    |         |

            range:
                row inidices: (0, num_nodes]
                columns indices: (0, num_resources]
                enteries: [0, node_resource_cap] type: float
        """
        nodes_resources_usage = []
        for node in range(self.num_nodes):
            services_in_node = np.where(self.services_nodes == node)[0]
            node_resources_usage = sum(self.services_resources_usage[
                services_in_node])
            if type(node_resources_usage) != np.ndarray:
                node_resources_usage = np.zeros(self.num_resources)
            nodes_resources_usage.append(node_resources_usage)
        return np.array(nodes_resources_usage)

    @property
    def nodes_resources_request(self): # TODO have two version for request and usage separately
        """return the amount of resource usage
        on each node
        """
        nodes_resources_request = []
        for node in range(self.num_nodes):
            services_in_node = np.where(
                self.services_nodes == node)[0]
            node_resources_usage = sum(
                self.services_resources_request[services_in_node])
            if type(node_resources_usage) != np.ndarray:
                node_resources_usage = np.zeros(self.num_resources)
            nodes_resources_request.append(node_resources_usage)
        return np.array(nodes_resources_request)

    @property
    def services_resources_remained(self) -> np.ndarray:  # TODO have two version for request and usage separately
        return self.services_resources_request - self.services_resources_usage

    @property
    def nodes_resources_remained(self):  # TODO have two version for request and usage separately
        # The amount of acutally used resources
        # on the nodes
        return self.nodes_resources_cap - self.nodes_resources_usage

    @property
    def nodes_resources_available(self):
        # The amount of the available
        # non-requested resources on the nodes
        return self.nodes_resources_cap - self.nodes_resources_request

    @property
    @rounding
    def services_types_usage(self) -> np.ndarray:
        """each service type resource usage

                                 ram  cpu
                                |        |
            services_types      |        |
                                |        |
        """
        services_types_usage = np.transpose(self.workload[
            :, self.timestep, :])
        return services_types_usage

    @property
    @rounding
    def services_resources_usage_frac(self) -> np.ndarray:
        """fraction of usage:

                         ram - cpu
                        |         |
            services    |         |
                        |         |

            range:
                row inidices: (0, num_services]
                columns indices: (0, num_resources]
                enteries: [0, 1] type: float
        """
        workload_services_types = self.workload[:, self.timestep, :]
        services_resources_usage_frac = list(map(lambda service_type:
                                                   workload_services_types
                                                   [:, service_type],
                                                   self.services_types))
        services_resources_usage_frac = np.array(
            services_resources_usage_frac)
        return services_resources_usage_frac

    @property
    @rounding
    def nodes_resources_usage_frac(self) -> np.ndarray:
        """returns the resource usage of
        each node
                     ram - cpu
                    |         |
            nodes   |         |
                    |         |

            range:
                row inidices: (0, num_nodes]
                columns indices: (0, num_resources]
                enteries: [0, 1] type: float
        """
        return self.nodes_resources_usage / self.nodes_resources_cap

    @property
    @rounding
    def nodes_resources_request_frac(self):
        """returns the resource requested on
        each node
                     ram - cpu
                    |         |
            nodes   |         |
                    |         |

            range:
                row inidices: (0, num_nodes]
                columns indices: (0, num_resources]
                enteries: [0, 1] type: float
        """
        return self.nodes_resources_request / self.nodes_resources_cap

    @property
    def num_consolidated(self) -> int:
        """returns the number of consolidated nodes
        """
        return self._num_consolidated(self.services_nodes)

    @property
    def num_overloaded(self) -> int:
        """return the number of resource exceeding nodes
        """
        overloaded_nodes = np.unique(np.where(
            self.nodes_resources_request_frac > 1)[0])
        return len(overloaded_nodes)

    @property
    def nodes_services(self) -> np.ndarray:
        """change the representation of placements from:
             contianer_id   contianer_id          contianer_id
            [node_id,       node_id,    , ... ,   node_id     ]
        to:
                node_id                    node_id
            [[service_id, service_id], ...,[service_id]]
        """
        nodes_services = []
        for node in range(self.num_nodes):
            services_in_node = np.where(self.services_nodes ==
                                          node)[0].tolist()
            nodes_services.append(services_in_node)
        return nodes_services

    @property
    def nodes_resources_remained_frac(self):
        return self.nodes_resources_remained / self.nodes_resources_cap

    @property
    def nodes_resources_available_frac(self):
        return self.nodes_resources_available / self.nodes_resources_cap

    @property
    def nodes_resources_remained_frac_avg(self):
        return np.average(self.nodes_resources_remained_frac, axis=1)

    @property
    def nodes_resources_available_frac_avg(self):
        return np.average(self.nodes_resources_available_frac, axis=1)

    @property
    def done(self):
        """check at every step that if we have reached the
        final state of the simulation of not
        """
        done = True if self.timestep % self.episode_length == 0 else False
        return done

    @property
    def complete_raw_observation(self) -> Dict[str, np.ndarray]:
        """complete observation with all the available elements
        """
        observation = {
                "services_resources_usage": self.services_resources_usage,
                "nodes_resources_usage": self.nodes_resources_usage,
                "services_resources_usage_frac":
                self.services_resources_usage_frac,
                "nodes_resources_usage_frac": self.nodes_resources_usage_frac,
                "services_nodes": self.services_nodes
        }
        return observation

    @property
    def raw_observation(self) -> Dict[str, np.ndarray]:
        """returns only the raw observations requested by the user
        in the config input through obs_elements
        """
        observation = {
                "services_resources_usage": self.services_resources_usage,
                "nodes_resources_usage": self.nodes_resources_usage,
                "services_resources_usage_frac":
                self.services_resources_usage_frac,
                "nodes_resources_usage_frac": self.nodes_resources_usage_frac,
                "services_nodes": self.services_nodes
        }
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

    def _num_consolidated(self, services_nodes) -> int:
        """functional version of num_services
        returns the number of consolidated nodes
        """
        a = set(services_nodes)
        b = set(np.arange(self.num_nodes))
        intersect = b - a
        return len(intersect)

    def preprocessor(self, obs):
        """
        environment preprocessor
        depeiding on the observation (state) definition
        """
        prep = Preprocessor(self.nodes_resources_cap,
                            self.services_resources_request)
        obs = prep.transform(obs)
        return obs

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
        # TODO change action-space to arriving services
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