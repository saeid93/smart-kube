```
{
    "notes": <some text explaining the experiment>,
    "env_config_base": {
        "obs_elements": ["services_resources_usage" |
                         "nodes_resources_usage" |
                         "services_resources_usage_frac" |
                         "nodes_resources_usage_frac" |
                         "services_nodes"],
        "penalty_illegal": [int],
        "penalty_move": [int],
        "penalty_variance": [int],
        "penalty_consolidatefd": [int],
        "penalty_latency": [int],
        "mitigation_tries": [int],
        "episode_length": [int],
        "timestep_reset": [bool],
        "placement_reset": [bool],
        "normalise_latency": [bool],
        "compute_greedy_num_consolidated": [bool],
        "seed": [int]
    } // for the full list see tune.run https://github.com/ray-project/ray/blob/master/python/ray/tune/tune.py line 151
    "run_or_experiment": "PPO",
    "learn_config": { // the learn_config is the config parameters for ray.tune.run see the full list of available options at https://docs.ray.io/en/master/rllib-algorithms.html
        "num_gpus": 0,
        "num_workers": 1,
        "lr": 1e-2,
        "model": {
            "fcnet_hiddens": [20, 20]
        },
        "framework": "torch",
        "rollout_fragment_length": 20,
        "train_batch_size": 40,
        "sgd_minibatch_size": 40,
        "num_sgd_iter": 5
    },
    "stop": {
        "training_iteration": 1000
    }
}
```