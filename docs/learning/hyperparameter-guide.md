# hyperparameters json files guide
The json files follows the follwoing structrue:
```
{
    "notes": ...,
    "metadata":...,
    "env_config_base": ...,
    "run_or_experiment": ...,
    "learn_config": ...,
    "stop": ... 
}
```
and consists of the following fields:

1. notes: This is for my own reference to save necessary information about each of experiments. 
Description:
```
"notes": {
        "env_model": "This is a test", // some description of my assumptions -- non-mandatory
        "description": "This is a test" // some description of the test -- non-mandatory
    }
```

2. metadata: metadata for the python script and the learnining
```
Description:
"metadata":{
    "type_env": 0, // type of the used environment -- mandatory
    "dataset_id": 0, // used dataset -- mandatory
    "workload_id": 0 // workload used for the experiments -- mandatory
}
```
3. env_config_base: configuration of the environment itself
```
Description:
"env_config_base": {
    "penalty_illegal": -1, // penalty for illegal steps -- mandatory
    "penalty_normal": 0, // penalty for normal steps -- mandatory
    "penalty_consolidated": 1, // penalty for consolidation reward -- mandatory
    "mitigation_tries": 4, // mitigation tries for the greey mitigator -- mandatory
    "episode_length": 5, // length of trainig episodes -- mandatory
    "latency_lower": 0.75, // whether to reset at each episode or not -- mandatory
    "latency_upper": 2, // whether to reset services_node at each episode or not -- mandatory
    "seed": 1 // random seed for the env environment dynamics -- mandatory
}
```
4. run_or_experiment: type of the used algorithm from rllib, see [rllib algorithms](https://docs.ray.io/en/latest/rllib-algorithms.html) for the full list of the available built-in algorithms in rllib.
```
"run_or_experiment": "PPO"
```
5. learn_config: learning configuration of the rllib library, this config is very much dependent on the used RL algorithm. For the complete list of hyperparameters of each algorithm see [rllib algorithms](https://docs.ray.io/en/latest/rllib-algorithms.html). Also, for the complete list of common configs (same hyperparameters across all rl envs) see the [common configs](https://docs.ray.io/en/latest/rllib-training.html#common-parameters). There are also a number of [tuned hyperparameter examples](https://github.com/ray-project/ray/tree/master/rllib/tuned_examples) per each of the algorithms.
To build the reinforcement learning model neural network see the [model catalog](https://docs.ray.io/en/latest/rllib-models.html#default-model-config-settings) from rllib documentation.
Complete list of the available [huperparameter search options for Tune](https://docs.ray.io/en/latest/tune/api_docs/search_space.html) ara available in the ray documentation.
For hyperparameter search with tune we have the following options:
```
"uniform": tune.uniform(-5, -1),  # Uniform float between -5 and -1
"quniform": tune.quniform(3.2, 5.4, 0.2),  # Round to increments of 0.2
"loguniform": tune.loguniform(1e-4, 1e-2),  # Uniform float in log space
"qloguniform": tune.qloguniform(1e-4, 1e-1, 5e-4),  # Round to increments of 0.0005
"randn": tune.randn(10, 2),  # Normal distribution with mean 10 and sd 2
"qrandn": tune.qrandn(10, 2, 0.2),  # Round to increments of 0.2
"randint": tune.randint(-9, 15),  # Random integer between -9 and 15
"qrandint": tune.qrandint(-21, 12, 3),  # Round to increments of 3 (includes 12)
"choice": tune.choice(["a", "b", "c"]),  # Choose one of these options uniformly
"func": tune.sample_from(lambda spec: spec.config.uniform * 0.01), # Depends on other value
"grid": tune.grid_search([32, 64, 128])  # Search over all these values
```
6. stoping criteria of the results. The stopping criteria of the hyperparameter searches. This is the returned object of the agent.train() function.
```
'episode_reward_max', 'episode_reward_min',
 'episode_reward_mean', 'episode_len_mean', 
 'episodes_this_iter', 'policy_reward_min', 'policy_reward_max', 
 'policy_reward_mean', 'custom_metrics', 'hist_stats', 
 'sampler_perf', 'off_policy_estimator', 'num_healthy_workers', 
 'timesteps_total', 'timers', 'info', 'done', 'episodes_total', 
 'training_iteration', 'experiment_id', 'date', 'timestamp', 
 'time_this_iter_s', 'time_total_s', 'pid', 'nodename', 
 'node_ip', 'config', 'time_since_restore', 
 'timesteps_since_restore', 'iterations_since_restore', 'perf'

```