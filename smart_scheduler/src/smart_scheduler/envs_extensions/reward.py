from difflib import diff_bytes
import numpy as np
from typing import Tuple, Dict, Any

def _reward(self) -> Tuple[
        float, Dict[str, Any]]:
    if self.cluster.num_overloaded > 0:
        illegal = _illegal(self)
        return illegal, {
            "illegal": illegal,
            "u_t": 0,
            "c_t": 0,
            "v_t": 0,
            "g_t": 0,
            "p_t": 0
            }
    illegal = _illegal(self)
    u = _u(self)
    c = _c(self)
    v = _v(self)
    g = _g(self)
    p = _p(self)
    rewards = {
        "illegal": illegal,
        "u": u,
        "c": c,
        "v": v,
        "g": g,
        "p": p
    }
    if self.reward_option == 'rlsk':
        rewards_total = rlsk(self, u, c, v)
    if self.reward_option == 'proposed':
        rewards_total = proposed(self, p, g, c)
    return rewards_total, rewards

def rescale(values, old_min = 0, old_max = 1, new_min = 0, new_max = 100):
    output = []
    for v in values:
        new_v = (new_max - new_min) / (
            old_max - old_min) * (v - old_min) + new_min
        output.append(new_v)

    return np.array(output)

def reward_illegal(self):
    rewards_total = 1
    rewards = {}
    return rewards_total, rewards

def rlsk(self, u, c, v):
    """RLSK paper reward function
    """
    rewards_total = self.penalty_u * u - self.penalty_c * c\
        - self.penalty_v * v
    return rewards_total

def proposed(self, p, g, c):
    """Our paper proposed approach
    """
    rewards_total = self.penalty_p * p - self.penalty_g * g\
        - self.penalty_c * c
    return rewards_total

def _illegal(self):
    """reward for the illegal states which has led
        to having overloaded machines
    """
    reward = self.penalty_illegal * self.cluster.num_overloaded
    return reward

def _u(self):
    """reward for utilizations
    """
    # arverage of resource usage of all resources
    average_resource_usage_fraction = np.average(
        self.cluster.nodes_usages_frac, axis=1)
    overal_resource_usage = np.sum(
        average_resource_usage_fraction)
    return overal_resource_usage

def _c(self):
    """compute the variance reward
    """
    diff_usage_per_node = []
    for node in self.cluster.nodes:
        diff_node = 0
        for resource_i in node.usages_fraction:
            for resource_j in node.usages_fraction:
                diff_node += np.abs(resource_i-resource_j)
        diff_usage_per_node.append(diff_node/2)
    total_resource_difference_all_cluster = np.sum(diff_usage_per_node)
    return total_resource_difference_all_cluster

def _v(self):
    """reward for balancing the ultilization across servers
    """
    diff_usage_nodes = 0
    average_resource_usage_fraction = np.average(
        self.cluster.nodes_usages_frac, axis=1)
    for node_i in average_resource_usage_fraction:
        for node_j in average_resource_usage_fraction:
            diff_usage_nodes = np.abs(node_i-node_j)
    return diff_usage_nodes/2

def _g(self):
    """total difference across all clusters 
    """
    diff_from_target = np.abs(
        self.cluster.nodes_usages_frac - self.target_utilization)
    diff_from_target_sum = np.sum(diff_from_target)
    return diff_from_target_sum

def _p(self):
    """total difference across all clusters
    """
    reward = self.cluster.num_consolidated
    return reward
