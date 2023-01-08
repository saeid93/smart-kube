from difflib import diff_bytes
import numpy as np
from typing import Tuple, Dict, Any

def _reward(self) -> Tuple[
        float, Dict[str, Any]]:
    if self.cluster.num_overloaded > 0:
        illegal = _illegal(self)
        return illegal, {
            "illegal": illegal,
            "u": 0,
            "c": 0,
            "cv": 0,
            "v": 0,
            "g": 0,
            "p": 0
            }
    illegal = _illegal(self)
    u = _u(self)
    c = _c(self)
    cv = _cv(self)
    v = _v(self)
    g = _g(self)
    p = _p(self)
    rewards = {
        "illegal": illegal,
        "u": u,
        "c": c,
        "cv": cv,
        "v": v,
        "g": g,
        "p": p
    }
    if self.reward_option == 'rlsk':
        rewards_total = rlsk(self, u, c, v)
    if self.reward_option == 'proposed':
        rewards_total = proposed(self, p, g, c, cv)
    return rewards_total, rewards

def rescale(values, old_min = 0, old_max = 1, new_min = 0, new_max = 100):
    output = []
    for v in values:
        new_v = (new_max - new_min) / (
            old_max - old_min) * (v - old_min) + new_min
        output.append(new_v)

    return np.array(output)

def rlsk(self, u, c, v):
    """RLSK paper reward function
    """
    rewards_total = self.penalty_u * u - self.penalty_c * c\
        - self.penalty_v * v
    return rewards_total

def proposed(self, p, g, c, cv):
    """Our paper proposed approach
    """
    rewards_total = self.penalty_p * p - self.penalty_g * g\
        - self.penalty_c * c - self.penalty_cv * cv
    return rewards_total

def _illegal(self):
    """reward for the illegal states which has led
        to having overloaded machines
    """
    reward = self.penalty_illegal * self.cluster.num_overloaded
    reward = rescale(
        [reward], old_min = 0, old_max = self.reward_var_illegal_1,
        new_min = 0, new_max = self.reward_var_illegal_2)[0]
    return reward

def _u(self):
    """reward for utilizations
    """
    # arverage of resource usage of all resources
    average_resource_usage_fraction = np.average(
        self.cluster.nodes_usages_frac, axis=1)
    overal_resource_usage = np.sum(
        average_resource_usage_fraction)
    # overal_resource_usage_scaled = rescale(
    #     [overal_resource_usage], old_min = 0, old_max = self.reward_var_u_1,
    #     new_min = 0, new_max = self.reward_var_u_2)[0]
    overal_resource_usage_scaled = overal_resource_usage # TEMP
    return overal_resource_usage_scaled

def _c(self):
    """compute the difference reward
    """
    diff_usage_per_node = []
    for node in self.cluster.nodes:
        diff_node = 0
        for resource_i in node.usages_fraction:
            for resource_j in node.usages_fraction:
                diff_node += np.abs(resource_i-resource_j)
        diff_usage_per_node.append(diff_node/2)
    total_resource_difference_all_cluster = np.sum(diff_usage_per_node)
    # total_resource_difference_all_cluster_scaled = rescale(
    #     [total_resource_difference_all_cluster], old_min = 0, old_max = self.reward_var_c_1,
    #     new_min = 0, new_max = self.reward_var_c_2)[0]
    total_resource_difference_all_cluster_scaled = total_resource_difference_all_cluster # TEMP
    return total_resource_difference_all_cluster_scaled

def _cv(self):
    """compute the difference reward for variance
    """
    per_resource_var = []
    for resource in range(self.cluster.num_resources):
        per_resource_var.append(np.var(
            self.cluster.nodes_usages_frac[:, resource]))
    total_resource_difference_all_cluster = np.sum(per_resource_var)
    # total_resource_difference_all_cluster_scaled = rescale(
    #     [total_resource_difference_all_cluster], old_min = 0, old_max = self.reward_var_cv_1,
    #     new_min = 0, new_max = self.reward_var_cv_2)[0]
    total_resource_difference_all_cluster_scaled = total_resource_difference_all_cluster # TEMP
    return total_resource_difference_all_cluster_scaled

def _v(self):
    """reward for balancing the ultilization across servers
    """
    diff_usage_nodes = 0
    average_resource_usage_fraction = np.average(
        self.cluster.nodes_usages_frac, axis=1)
    for node_i in average_resource_usage_fraction:
        for node_j in average_resource_usage_fraction:
            diff_usage_nodes = np.abs(node_i-node_j)
    diff_usage_nodes /= 2
    # diff_usage_nodes_scaled = rescale(
    #     [diff_usage_nodes], old_min = 0, old_max = self.reward_var_v_1,
    #     new_min = 0, new_max = self.reward_var_v_2)[0]
    diff_usage_nodes_scaled = diff_usage_nodes # TEMP
    return diff_usage_nodes_scaled


def _g(self):
    """total difference across all clusters 
    """
    diff_from_target = np.abs(
        self.cluster.nodes_requests_frac - self.target_utilization)
    diff_from_target_sum = np.sum(diff_from_target)
    # diff_from_target_sum_scaled = rescale(
    #     [diff_from_target_sum], old_min = 0, old_max = self.reward_var_g_1,
    #     new_min = 0, new_max = self.reward_var_g_2)[0]
    diff_from_target_sum_scaled = diff_from_target_sum # TEMP
    return diff_from_target_sum_scaled

def _p(self):
    """binpacking/consolidation reward
    """
    reward = self.cluster.num_consolidated
    # reward_scaled = rescale(
    #     [reward], old_min = 0, old_max = self.reward_var_p_1,
    #     new_min = 0, new_max = self.reward_var_p_2)[0]
    reward_scaled = reward # TEMP
    return reward_scaled
