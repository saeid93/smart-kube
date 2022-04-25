from gym.envs.registration import register

register(
    id='SimSchedulerEnv-v0',
    entry_point='smart_scheduler.envs:SimSchedulerEnv',
)
register(
    id='SimBinpackingEnv-v0',
    entry_point='smart_scheduler.envs:SimBinpackingEnv',
)
register(
    id='KubeSchedulerEnv-v0',
    entry_point='smart_scheduler.envs:KubeSchedulerEnv',
)
register(
    id='KubeBinpackingEnv-v0',
    entry_point='smart_scheduler.envs:KubeBinpackingEnv',
)
