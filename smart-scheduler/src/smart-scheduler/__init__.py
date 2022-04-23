from gym.envs.registration import register

register(
    id='SimEdgeEnv-v0',
    entry_point='smart_scheduler.envs:SimEdgeEnv',
)
register(
    id='SimBinpackingEnv-v0',
    entry_point='smart_scheduler.envs:SimBinpackingEnv',
)
register(
    id='KubeEdgeEnv-v0',
    entry_point='smart_scheduler.envs:KubeEdgeEnv',
)
register(
    id='KubeBinpackingEnv-v0',
    entry_point='smart_scheduler.envs:KubeBinpackingEnv',
)
