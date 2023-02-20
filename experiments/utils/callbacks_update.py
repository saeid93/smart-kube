# callback function updated for new versios

import numpy as np
from typing import Dict
import ray
from packaging import version
from tabulate import tabulate
if version.parse(ray.__version__) < version.parse('1.9.0'):
    from ray.rllib.agents.callbacks import DefaultCallbacks
else:
    from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from typing import Dict, Optional, TYPE_CHECKING

from ray.rllib.env import BaseEnv
from ray.rllib.policy import Policy
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.evaluation import MultiAgentEpisode, RolloutWorker
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.deprecation import deprecation_warning
from ray.rllib.utils.typing import AgentID, PolicyID


class CloudCallbackUpdate(DefaultCallbacks):

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

        episode.user_data["scheduling_timestep"] = []
        episode.hist_data["scheduling_timestep"] = []

        episode.user_data["scheduling_success"] = []
        episode.hist_data["scheduling_success"] = []

        episode.user_data["num_consolidated"] = []
        episode.hist_data["num_consolidated"] = []

        episode.user_data["num_overloaded"] = []
        episode.hist_data["num_overloaded"] = []

        episode.user_data["time"] = []
        episode.hist_data["time"] = []

        episode.user_data["timestep_episode"] = []
        episode.hist_data["timestep_episode"] = []

        episode.user_data["rewards"] = []
        episode.hist_data["rewards"] = []

    def on_episode_step(self, *, worker,
                        base_env: BaseEnv,
                        episode: MultiAgentEpisode,
                        env_index: int, **kwargs):


        if type(episode.last_info_for()) == dict:
            scheduling_timestep = episode.last_info_for()['scheduling_timestep']
            episode.user_data["scheduling_timestep"].append(scheduling_timestep)

            scheduling_success = episode.last_info_for()['scheduling_success']
            episode.user_data["scheduling_success"].append(scheduling_success)

            num_consolidated = episode.last_info_for()['num_consolidated']
            episode.user_data["num_consolidated"].append(num_consolidated)

            num_overloaded = episode.last_info_for()['num_overloaded']
            episode.user_data["num_overloaded"].append(num_overloaded)

            time = episode.last_info_for()['time']
            episode.user_data["time"].append(time)

            timestep_episode = episode.last_info_for()['timestep_episode']
            episode.user_data["timestep_episode"].append(timestep_episode)

            rewards = episode.last_info_for()['rewards']
            episode.user_data["rewards"].append(rewards)

    def on_episode_end(self, *, worker: RolloutWorker,
                       base_env: BaseEnv,
                       policies: Dict[str, Policy],
                       episode: MultiAgentEpisode,
                       env_index: int, **kwargs):

        scheduling_timestep_avg = np.mean(episode.user_data["scheduling_timestep"])
        scheduling_success_avg = np.mean(episode.user_data["scheduling_success"])
        num_consolidated_avg = np.mean(episode.user_data["num_consolidated"])
        num_overloaded_avg = np.mean(episode.user_data["num_overloaded"])
        time = np.max(episode.user_data["time"])
        timestep_episode = np.max(episode.user_data["timestep_episode"])


        # extract episodes rewards info
        episode_reward_u = [a[
            'u']for a in episode.user_data[
                "rewards"]]
        episode_reward_c = [a[
            'c']for a in episode.user_data[
                "rewards"]]
        episode_reward_cv = [a[
            'cv']for a in episode.user_data[
                "rewards"]]
        episode_reward_v = [a[
            'v']for a in episode.user_data[
                "rewards"]]
        episode_reward_g = [a[
            'g']for a in episode.user_data[
                "rewards"]]
        episode_reward_p = [a[
            'p']for a in episode.user_data[
                "rewards"]]

        episode_reward_u = np.mean(episode_reward_u)
        episode_reward_c = np.mean(episode_reward_c)
        episode_reward_cv = np.mean(episode_reward_cv)
        episode_reward_v = np.mean(episode_reward_v)
        episode_reward_g = np.mean(episode_reward_g)
        episode_reward_p = np.mean(episode_reward_p)

        # add custom metrics to tensorboard
        episode.custom_metrics['scheduling_timestep_avg'] = scheduling_timestep_avg
        episode.custom_metrics['scheduling_success_avg'] = scheduling_success_avg
        episode.custom_metrics['num_consolidated_avg'] = num_consolidated_avg
        episode.custom_metrics['num_overloaded_avg'] = num_overloaded_avg
        episode.custom_metrics['time'] = time
        episode.custom_metrics['timestep_episode'] = timestep_episode
        episode.custom_metrics['reward_u'] = episode_reward_u
        episode.custom_metrics['reward_c'] = episode_reward_c
        episode.custom_metrics['reward_cv'] = episode_reward_cv
        episode.custom_metrics['reward_v'] = episode_reward_v
        episode.custom_metrics['reward_g'] = episode_reward_g
        episode.custom_metrics['reward_p'] = episode_reward_p

    def on_postprocess_trajectory(
            self, *, worker: "RolloutWorker", episode: MultiAgentEpisode,
            agent_id: AgentID, policy_id,
            policies: Dict[PolicyID, Policy], postprocessed_batch: SampleBatch,
            original_batches: Dict[AgentID, SampleBatch], **kwargs) -> None:
        # masking all the scheduling states
        scheduling_timestep_mask = list(
            map(lambda a: a[
                'scheduling_timestep'],postprocessed_batch['infos']))
        # adding scheduling states former state to the mask
        for i in range(1, len(scheduling_timestep_mask)):
            if scheduling_timestep_mask[i]:
                scheduling_timestep_mask[i-1] = True
        for key in postprocessed_batch.keys():
            postprocessed_batch[
                key] = postprocessed_batch[
                    key][scheduling_timestep_mask]
        # postprocessed_batch['rewards'] = postprocessed_batch['rewards'][scheduling_timestep_mask]
        if version.parse(ray.__version__) < version.parse('1.9.0'):
            if self.legacy_callbacks.get("on_postprocess_traj"):
                self.legacy_callbacks["on_postprocess_traj"]({
                    "episode": episode,
                    "agent_id": agent_id,
                    "pre_batch": original_batches[agent_id],
                    "post_batch": postprocessed_batch,
                    "all_pre_batches": original_batches,
                })
        else:
            pass


    def on_sample_end(self, *, worker: RolloutWorker, samples: SampleBatch,
                      **kwargs):
        pass

    def on_learn_on_batch(self, *, policy: Policy, train_batch: SampleBatch,
                          **kwargs) -> None:
        # print(f"train batch <{self.count}> of"
        #       f" size <{train_batch.count}>"
        #       f" total <{self.total}>")
        self.total += train_batch.count
        self.count += 1

    def on_train_result(self, *, result: dict, **kwargs):
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
        if version.parse(ray.__version__) < version.parse('1.9.0'):
            result["callback_ok"] = True
            if self.legacy_callbacks.get("on_train_result"):
                self.legacy_callbacks["on_train_result"]({
                    "trainer": trainer,
                    "result": result,
                })
        else:
            pass

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
