import numpy as np
from typing import Dict
from tabulate import tabulate
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from typing import Dict, Optional, TYPE_CHECKING

from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.typing import AgentID, PolicyID


class CloudCallback(DefaultCallbacks):

    """
    callbakc for saving my own metrics
    functions to add necessary metrics either to the
    tensorboard or print it in the output during the
    training
    points (functions) to add metrics (see parent class
    description for each):
        1. on_episode_start
        2. on_episode_step
        3. on_episode_end
        4. on_sample_end
        5. on_postprocess_trajectory
        6. on_sample_end
        7. on_learn_on_batch
        8. on_train_result
    variables to add/store custom metrics:
        1. episode.user_data: to pass data between episodes
        2. episode.custom_metrics: what is being printed to the tensorboard
        3. episode.hist_data: histogram data saved to the json
    my metrics:
        1. num_consolidated
        2. num_overloaded
        3. greedy_num_consolidated
        4. num_moves
    """
    def __init__(self, legacy_callbacks_dict: Dict[str, callable] = None):
        super().__init__(legacy_callbacks_dict=legacy_callbacks_dict)
        self.workers_total_episodes = [0]*1000
        self.count = 0
        self.total = 0

    def on_episode_start(self, *, worker,
                         base_env: BaseEnv,
                         policies: Dict[str, Policy],
                         episode: MultiAgentEpisode,
                         env_index: int, **kwargs):

        # timestep of the cluster
        episode.user_data["timestep"] = []
        episode.hist_data["timestep"] = []
        # global timestep of the cluster
        episode.user_data["global_timestep"] = []
        episode.hist_data["global_timestep"] = []
        # num_consolidated placeholder lists
        episode.user_data["num_consolidated"] = []
        episode.hist_data["num_consolidated"] = []
        # num_overloaded placeholder lists
        episode.user_data["num_overloaded"] = []
        episode.hist_data["num_overloaded"] = []
        # num_moves
        episode.user_data["num_moves"] = []
        episode.hist_data["num_moves"] = []
        # rewards
        episode.user_data["rewards"] = []
        episode.hist_data["rewards"] = []
        # users distances
        episode.user_data["users_distances"] = []
        episode.hist_data["users_distances"] = []

    def on_episode_step(self, *, worker,
                        base_env: BaseEnv,
                        episode: MultiAgentEpisode,
                        env_index: int, **kwargs):

        if type(episode.last_info_for()) == dict:
            # extract the timestep of the current step from the dict
            timestep = episode.last_info_for()['timestep']
            episode.user_data["timestep"].append(timestep)
            # extract the global timestep of the current step from the dict
            global_timestep = episode.last_info_for()['global_timestep']
            episode.user_data["global_timestep"].append(global_timestep)
            # extract the number of conolidated from the info dict
            num_consolidated = episode.last_info_for()['num_consolidated']
            episode.user_data["num_consolidated"].append(num_consolidated)
            # extract the number of overloaded from the info dict
            num_overloaded = episode.last_info_for()['num_overloaded']
            episode.user_data["num_overloaded"].append(num_overloaded)
            # extract of the number of services moves from the dict
            num_moves = episode.last_info_for()['num_moves']
            episode.user_data["num_moves"].append(num_moves)
            # extract of the total sum of latencies
            users_distances = episode.last_info_for()['users_distances']
            episode.user_data["users_distances"].append(users_distances)
            # rewards
            rewards = episode.last_info_for()['rewards']
            episode.user_data["rewards"].append(rewards)


            # print(f"env: <{episode.env_id}>")
            # print(f"worker: <{worker.worker_index}>")
            # print(f"episode: <{episode.episode_id}>")
            # print(f"timestep: <{timestep}>")
            # print(f"last state unprocessed:\n<{episode.last_raw_obs_for()}>")
            # print(f"last state process:\n<{episode.last_observation_for()}>")
            # print(f"rllib action:\n<{episode.last_action_for()}>")
            # print(f"env action: <{episode.last_info_for()['action']}>")

    def on_episode_end(self, *, worker: RolloutWorker,
                       base_env: BaseEnv,
                       policies: Dict[str, Policy],
                       episode: MultiAgentEpisode,
                       env_index: int, **kwargs):

        # extract the episode information
        num_moves_avg = np.mean(episode.user_data["num_moves"])
        num_consolidated_avg = np.mean(episode.user_data["num_consolidated"])
        users_distances_avg = np.mean(episode.user_data["users_distances"])
        num_overloaded_avg = np.mean(episode.user_data["num_overloaded"])
        timestep = np.max(episode.user_data["timestep"])
        global_timestep = np.max(episode.user_data["global_timestep"])
        # episode_total_reward = episode.total_reward
        # action_logit_max = round(max(episode.last_action_for()).item(), 2)
        # action_logit_min = round(min(episode.last_action_for()).item(), 2)
        # action_logit_avg = round(np.mean(episode.last_action_for()).item(), 2)

        # extract episodes rewards info
        episode_reward_consolidation = [a[
            'reward_consolidation']for a in episode.user_data[
                "rewards"]]
        episode_reward_illegal = [a[
            'reward_illegal']for a in episode.user_data[
                "rewards"]]
        episode_reward_move = [a[
            'reward_move']for a in episode.user_data[
                "rewards"]]
        episode_reward_latency = [a[
            'reward_latency']for a in episode.user_data[
                "rewards"]]

        reward_consolidation_mean = np.mean(episode_reward_consolidation)
        reward_illegal_mean = np.mean(episode_reward_illegal)
        reward_move_mean = np.mean(episode_reward_move)
        reward_latency_mean = np.mean(episode_reward_latency)
        reward_consolidation_sum = np.sum(episode_reward_consolidation)
        reward_illegal_sum = np.sum(episode_reward_illegal)
        reward_move_sum = np.sum(episode_reward_move)
        reward_latency_sum = np.sum(episode_reward_latency)
        # print episode information in the ouput
        # print('-'*50)
        # print(f"<----- workder <{worker.worker_index}>,"
        #       f" episode <{episode.episode_id}>,"
        #       f" env <{env_index}> ----->")
        # headers = ['last_timestep', 'episode_length', 'num_consolidated_avg',
        #            'greedy_num_consolidated_avg', 'num_overloaded_avg',
        #            'action_logits_min', 'action_logits_max',
        #            'action_logits_avg']
        # table = [[last_timestep, episode.length, num_consolidated_avg,
        #           greedy_num_consolidated_avg, num_overloaded_avg,
        #           action_logit_max, action_logit_min, action_logit_avg]]
        # print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
        # print("[-- episode info --]")
        # print(f"last_timestep <{last_timestep}>"
        #       f", episode_length <{episode.length}>\n"
        #       f"num_consolidated_avg <{num_consolidated_avg}>"
        #       f", num_overloaded_avg <{num_overloaded_avg}>"
        #     #   f", greedy_num_consolidated_avg <{greedy_num_consolidated_avg}>\n"
        #       f"episode_total_reward <{episode_total_reward}>")
        # print("[-- reward info --]")
        # print(f"reward_consolidation_max <{reward_consolidation_max}>"
        #       f", reward_illegal_max <{reward_illegal_max}>"
        #       f", reward_move_max <{reward_move_max}>"
        #       f", reward_variance_max <{reward_variance_max}>")
        # print("[-- last episode state/action info --]")
        # print(f"action_logits_min <{action_logit_max}>"
        #       f", action_logits_max <{action_logit_min}>"
        #       f", action_logits_avg <{action_logit_avg}>\n")

        # add custom metrics to tensorboard
        episode.custom_metrics['num_moves'] = num_moves_avg
        episode.custom_metrics['num_consolidated'] = num_consolidated_avg
        episode.custom_metrics['users_distances'] = users_distances_avg
        episode.custom_metrics['num_overloaded'] = num_overloaded_avg
        # episode.custom_metrics['action_logit_max'] = action_logit_max
        episode.custom_metrics['reward_consolidation_mean'] =\
            reward_consolidation_mean
        episode.custom_metrics['reward_illegal_mean'] = reward_illegal_mean
        episode.custom_metrics['reward_move_mean'] = reward_move_mean
        episode.custom_metrics['reward_latency_mean'] = reward_latency_mean
        episode.custom_metrics['reward_consolidation_sum'] =\
            reward_consolidation_sum
        episode.custom_metrics['reward_illegal_sum'] = reward_illegal_sum
        episode.custom_metrics['reward_move_sum'] = reward_move_sum
        episode.custom_metrics['reward_latency_sum'] = reward_latency_sum
        episode.custom_metrics['timestep'] = timestep
        episode.custom_metrics['global_timestep'] = global_timestep

        # # add histogram data
        # episode.hist_data["timestep"] = \
        #     episode.user_data["timestep"]
        # episode.hist_data["num_consolidated"] = \
        #     episode.user_data["num_consolidated"]
        # episode.hist_data["greedy_num_consolidated"] = \
        #     episode.user_data["greedy_num_consolidated"]
        # episode.hist_data["num_overloaded"] = \
        #     episode.user_data["num_overloaded"]
        # episode.hist_data["num_moves"] = \
        #     episode.user_data["num_moves"]

    def on_postprocess_trajectory(
            self, *, worker: "RolloutWorker", episode: MultiAgentEpisode,
            agent_id: AgentID, policy_id,
            policies: Dict[PolicyID, Policy], postprocessed_batch: SampleBatch,
            original_batches: Dict[AgentID, SampleBatch], **kwargs) -> None:
        if self.legacy_callbacks.get("on_postprocess_traj"):
            self.legacy_callbacks["on_postprocess_traj"]({
                "episode": episode,
                "agent_id": agent_id,
                "pre_batch": original_batches[agent_id],
                "post_batch": postprocessed_batch,
                "all_pre_batches": original_batches,
            })




    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        num_episodes = np.unique(samples['eps_id']).size
        # self.workers_total_episodes[worker.worker_index] += num_episodes
        # headers = ['rollout_fragment_length', 'num_episodes',
        #  'total_episodes']
        # table = [[samples.count, num_episodes,
        #           self.workers_total_episodes[worker.worker_index]]]
        # print('*'*50)
        # print(f"episode: {self.count}")
        # print("<--- one sample batch of worker"
        #       f" <{worker.worker_index}> ended --->")
        # print(samples['actions'])
        # print(f"rollout_fragment_length <{samples.count}>"
        #       f", num_episodes <{num_episodes}>"
        #       ", total_episodes "
        #       f"<{self.workers_total_episodes[worker.worker_index]}>")
        # # print(tabulate(table, headers=headers, tablefmt="fancy_grid"))
        # self.count += 1
        # print('*'*50, '\n')
        pass

    def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch,
                          **kwargs) -> None:
        # print(f"train batch <{self.count}> of"
        #       f" size <{train_batch.count}>"
        #       f" total <{self.total}>")
        self.total += train_batch.count
        self.count += 1

    def on_train_result(self, *, trainer, result: dict, **kwargs):
        """Called at the end of Trainable.train().

        Args:
            trainer (Trainer): Current trainer instance.
            result (dict): Dict of results returned from trainer.train() call.
                You can mutate this object to add additional metrics.
            kwargs: Forward compatibility placeholder.
        """
        # print("trainer.train() result: <{}> -> <{}> episodes".format(
        #     trainer, result["episodes_this_iter"]))
        # you can mutate the result dict to add new fields to return
        result["callback_ok"] = True
        if self.legacy_callbacks.get("on_train_result"):
            self.legacy_callbacks["on_train_result"]({
                "trainer": trainer,
                "result": result,
            })

    def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch,
                          result: dict, **kwargs) -> None:
        """Called at the beginning of Policy.learn_on_batch().

        Note: This is called before 0-padding via
        `pad_batch_to_sequences_of_same_size`.

        Args:
            policy (Policy): Reference to the current Policy object.
            train_batch (SampleBatch): SampleBatch to be trained on. You can
                mutate this object to modify the samples generated.
            result (dict): A results dict to add custom metrics to.
            kwargs: Forward compatibility placeholder.
        """
        pass
