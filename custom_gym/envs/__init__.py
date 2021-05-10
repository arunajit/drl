from gym.envs.registration import register
#env_name = 'CustomEnv-v0'
#if env_name in gym.envs.registry.env_specs:
#    del gym.envs.registry.env_specs[env_name]

register(id='CustomEnv-v0',
    entry_point='custom_env:CustomEnv'
)
