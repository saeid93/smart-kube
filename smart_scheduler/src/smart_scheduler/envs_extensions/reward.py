import numpy as np
from typing import Tuple, Dict, Any

def _reward(
    self, *,num_overloaded: int) -> Tuple[
        float, Dict[str, Any]]:
    if num_overloaded > 0:
        reward_illegal = _reward_one(self, num_overloaded)
        return reward_illegal, {
            "reward_move": 0,
            "reward_illegal": reward_illegal,
            "reward_consolidation": 0,
            "reward_variance": 0,
            "reward_latency": 0
            }


    reward_one = _reward_one(self)
    reward_two = _reward_two(self)
    reward_three = _reward_three(self)
    reward_four = _reward_three(self) + _reward_four(self)
    rewards = {
        "reward_one": reward_one,
        "reward_two": 0,
        "reward_three": reward_three,
        "reward_variance": reward_four
    }
    rewards_total = reward_one + reward_two + reward_three + reward_four
    return rewards_total, rewards

def rescale(values, old_min = 0, old_max = 1, new_min = 0, new_max = 100):
    output = []

    for v in values:
        new_v = (new_max - new_min) / (old_max - old_min) * (v - old_min) + new_min
        output.append(new_v)

    return np.array(output)


def _reward_one(self):
    """reward for the num_consolidated
    """
    # consolidation_factor = self.num_consolidated/self.num_nodes
    # reward_scaled = rescale(
    #     values=[consolidation_factor],
    #     old_min=self.consolidation_lower, 
    #     old_max=self.consolidation_upper,
    #     new_min=0, new_max=1)[0]
    # reward = self.penalty_consolidated * reward_scaled
    # if reward > 10000000000000:
    #     a = 5
    reward = 1
    return reward

def _reward_two(self):
# def _reward_move(self, num_moves: int):
    """reward for the number of moves
    """
    # movement_factor = num_moves/self.num_services
    # reward_move = self.penalty_move * movement_factor
    reward =1
    return reward

def _reward_three(self):
# def _reward_variance(self):
    """compute the variance reward
    """
    # reward_factor = np.sum(np.var(
    #     self.nodes_resources_request_frac, axis=1))
    # reward_variance = reward_factor * self.penalty_variance
    reward = 1
    return reward

def _reward_four(self):
# def _reward_illegal(self, prev_num_overloaded: int):
    """reward for the number of illegal factors
    """
    # nodes_overloaded_factor = prev_num_overloaded/self.num_nodes
    # reward_illegal = self.penalty_illegal * nodes_overloaded_factor
    reward = 1
    return reward
