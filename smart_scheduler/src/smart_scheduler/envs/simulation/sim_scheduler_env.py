"""base class of all simulation enviornments
"""
import numpy as np
from scipy.stats import bernoulli
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
    ServiceDummy,
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
        self.penalty_one: float = config['penalty_one']
        self.penalty_two: float = config['penalty_two']
        self.penalty_three: float = config['penalty_three']
        self.penalty_four: float = config['penalty_four']
        self.penalty_four: float = config['penalty_five']
        # reward weighting variables
        self.reward_var_one = config['reward_var_one']
        self.reward_var_two = config['reward_var_two']
        self.reward_var_three = config['reward_var_three']
        self.reward_var_four = config['reward_var_four']

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

        # maximum allowed services in a node
        self.max_services_nodes = config['max_services_nodes']

        # get the job arrival mode
        self.job_arrival = config['job_arrival']
        self.job_arrival_mode = self.job_arrival['mode']
        if self.job_arrival_mode == 'fixed':
            self.interval = self.job_arrival['interval']
        elif self.job_arrival_mode == 'bernoulli':
            self.probability = self.job_arrival['probability']

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
                    10, service_workload.shape[1])
            self.pending_services.append(Service(
                service_id=service_id,
                service_name=service_name,
                requests=self.services_resources_request[service_id],
                limits=self.services_resources_request[service_id],
                workload=service_workload,
                serving_time=serving_time))

        self.initil_pending_services = deepcopy(self.pending_services)
        self.backlog_size = config['backlog_size']
        self.time = 0
        self.observation_space, self.action_space =\
            self._setup_space()
        self.complete_done = False
        _ = self.reset()

    def seed(self, seed):
        np.random.seed(seed)
        self.np_random, seed = seeding.np_random(seed)
        self._env_seed = seed
        self.base_env_seed = seed
        # self.scheduling_timestep = False
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
        if self.complete_done:
            self.time = 0
            self.pending_services = deepcopy(self.initil_pending_services)
            self.cluster.reset_cluster()
            self.complete_done = False
        # self.next_scheduling_time = self.get_next_scheduling_time()
        return self.observation

    def clock_tick(self):
        """
            move the simulator forward proportional to
            tick_seconds
        """
        self.time += 1
        self.cluster.clock_tick()

    @override(gym.Env)
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, int, bool, dict]:
        """
        edge servers here
        1. move the services based-on current network and nodes state
        2. do one step of user movements
        3. update the nodes
        """
        # take the action
        scheduling_timestep = False
        success = False
        if self.scheduling_timestep:
            assert self.action_space.contains(action)
            if self.pending_services != []:
                success = self.schedule(
                    service_id=self.pending_services[-1].service_id,
                    node_id=action
                    )
                # self.next_scheduling_time = self.get_next_scheduling_time()
                scheduling_timestep = True
        else:
            scheduling_timestep = False

        self.clock_tick()

        # reward, rewards = 1, {1:1}
        # TODO
        reward, rewards = self._reward(
            num_overloaded=self.cluster.num_overloaded)

        info = {
            'scheduling_timestep': scheduling_timestep,
            'scheduling_success': success,
            'num_consolidated': self.cluster.num_consolidated,
            'num_overloaded': self.cluster.num_overloaded,
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
        print(f"node placements:\n{self.cluster.nodes_services}")

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
        # self.clock_tick()
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
    def scheduling_timestep(self):
        # next scheduling descition
        if self.job_arrival_mode == 'fixed':
            if self.time == 0:
                self.next_scheduling_time = self.interval
                return False
            elif self.time == self.next_scheduling_time:
                self.next_scheduling_time = self.time + self.interval
                return True
            else: False
        elif self.job_arrival_mode == 'bernoulli':
            dice = bernoulli.rvs(self.probability, loc=0)
            scheduling_interval = True if dice == 1 else False
            return scheduling_interval

    @property
    def backlog_services(self):
        """
        The next services to be scheduled on
        the cluster
        """
        backlog_services = self.pending_services[-self.backlog_size:]
        if len(backlog_services) < self.backlog_size:
            diff = self.backlog_size - len(backlog_services)
            backlog_services = backlog_services + [ServiceDummy()]*diff
        return backlog_services

    @property
    def backlog_services_requests(self):
        backlog_services_requests = np.array(
            list(map(
                lambda a: a.requests, self.backlog_services)))
        return backlog_services_requests

    @property
    def backlog_services_ids(self):
        ids = list(
            map(
                lambda service: service.service_id, self.backlog_services))
        return ids

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
                "backlog_services_requests": self.backlog_services_requests,
                "num_services_nodes": self.cluster.num_services_nodes,
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
        # add more if needed after finalizing on satate-space
        selected = dict(zip(self.obs_elements,
                            [observation[k] for k in self.obs_elements]))
        return selected

    @property
    def observation(self) -> np.ndarray:
        """preprocessed observation of each environment
        """
        obs = self.preprocessor(self.raw_observation)
        return obs

    def preprocessor(self, obs):
        """
        environment preprocessor
        depeiding on the observation (state) definition
        """
        prep = Preprocessor(self.max_services_nodes,
                            self.cluster.nodes_capacities,
                            self.services_resources_request)
        obs = prep.transform(obs)
        return obs

    def _setup_space(self):
        """
        """
        # numuber of elements based on the obs in the observation space
        obs_size = 0
        node_resource_size_items = [
            "nodes_capacities",
            "nodes_usages",
            "nodes_requests",
            "nodes_available",
            "nodes_unused",
            "nodes_slack",
            "nodes_usages_frac",
            "nodes_requests_frac",
            "nodes_requests_available_frac",
            "nodes_resources_unused_frac",
            "nodes_requests_available_frac_avg",
            "nodes_resources_unused_avg",
        ]
        node_services_size_items = ['num_services_nodes']
        backlog_size_items = ["backlog_services_requests"]
        node_size_items = ['num_consolidated']
        for elm in self.obs_elements:
            if elm in node_resource_size_items:
                obs_size += self.cluster.num_nodes * self.cluster.num_resources
            elif elm in node_services_size_items:
                obs_size += self.cluster.num_nodes * self.max_services_nodes
            elif elm in backlog_size_items:
                obs_size += self.backlog_size * self.cluster.num_resources
            elif elm in node_size_items:
                obs_size += self.cluster.num_nodes

        higher_bound = 20 # TEMP find a cleaner way
        # generate observation and action spaces
        observation_space = Box(
            low=-1, high=higher_bound, shape=(obs_size, ),
            dtype=np.float64, seed=self._env_seed)

        # if self.discrete_actions:
        action_space = Discrete(self.cluster.num_nodes, seed=self._env_seed)

        return observation_space, action_space

    @property
    def done(self):
        if self.time % self.episode_length == 0:
            if self.pending_services == [] and self.cluster.all_jobs_done:
                self.complete_done = True
            return True
        return False

