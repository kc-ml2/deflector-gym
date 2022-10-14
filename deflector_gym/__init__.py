import gym
from gym.envs.registration import register

def make(*args, **kwargs):
    return gym.make(*args, **kwargs)

try:
    register(
        id='ReticoloIndex-v0',
        entry_point='deflector_gym.envs.reticolo_env:ReticoloIndexEnv',
    )

    register(
        id='ReticoloDirection-v0',
        entry_point='deflector_gym.envs.reticolo_env:ReticoloDirectionEnv',
    )
except Exception as e:
    raise Warning(f'Reticolo environments not available\n{e}')

register(
    id='MeentIndex-v0',
    entry_point='deflector_gym.envs.meent_env:MeentIndexEnv',
)

register(
    id='MeentDirection-v0',
    entry_point='deflector_gym.envs.meent_env:MeentDirectionEnv'
)


