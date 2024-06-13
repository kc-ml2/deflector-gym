import gymnasium as gym
from gymnasium.envs.registration import register

def make(*args, **kwargs):
    return gym.make(*args, **kwargs)

register(
    id='MeentIndexEfield-v0',
    entry_point='deflector_gym.envs.meent_env:MeentIndexEfield',
    kwargs=dict(
        obs_type='efield',
        rew_type='eff',
    )
)

register(
    id='MeentIndex-v0',
    entry_point='deflector_gym.envs.meent_env:MeentIndexEfield',
    kwargs=dict(
        obs_type='struct',
        rew_type='delta_eff',
    )
)
