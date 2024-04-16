import gymnasium as gym
from gymnasium.envs.registration import register

def make(*args, **kwargs):
    return gym.make(*args, **kwargs)

register(
    id='MeentIndexEfield-v0',
    entry_point='deflector_gym.envs.meent_env:MeentIndexEfield',
)
