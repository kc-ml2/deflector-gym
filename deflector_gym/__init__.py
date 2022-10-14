import gym
from gym.envs.registration import register

def make(*args, **kwargs):
    return gym.make(*args, **kwargs)


register(
    id='ReticoloIndex-v0',
    entry_point='deflector_gym.envs.reticolo_env:ReticoloIndexEnv',
)

register(
    id='ReticoloDirectrion-v0',
    entry_point='deflector_gym.envs.reticolo_env:ReticoloDirectionEnv',
)

register(
    id='MeentIndex-v0',
    entry_point='deflector_gym.envs.meent_env:ReticoloIndexEnv',
)

register(
    id='MeentDirection-v0',
    entry_point='deflector_gym.envs.meent_env:ReticoloDirectionEnv'
)


