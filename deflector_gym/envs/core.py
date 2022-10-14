import gym
import numpy as np

from .constants import Direction1d


def DirectionEnv(
        base,  # base env such as MeentBase or MatlabBase
        n_cells=256,
        wavelength=1100,
        desired_angle=70,
        initial_pos='random',  # initial agent's position
        *args,
        **kwargs
):
    class _cls(base):
        def __init__(self):
            super().__init__(n_cells, wavelength, desired_angle)

            self.observation_space = gym.spaces.Box(
                low=-1., high=1.,
                shape=(1, n_cells,),
                dtype=np.float64
            )
            self.action_space = gym.spaces.Discrete(len(Direction1d))
            self.initial_pos = initial_pos

        def reset(self):
            # initialize structure
            super().reset()

            # initialize agent
            if self.initial_pos == 'center':
                self.pos = self.n_cells // 2
            elif self.initial_pos == 'right_edge':
                self.pos = self.n_cells - 1
            elif self.initial_pos == 'left_edge':
                self.pos = 0
            elif self.initial_pos == 'random':
                self.pos = np.random.randint(self.n_cells)
            else:
                raise RuntimeError('Undefined inital position')

            return self.struct[np.newaxis, :]

        def step(self, ac):
            prev_eff = self.eff
            # left == -1, noop == 0, right == 1
            # this way we can directly use ac as index difference
            ac -= 1
            self.struct = self.flip(self.struct, self.pos + ac)
            self.eff = self.get_efficiency(self.struct)

            reward = self.eff - prev_eff

            return self.struct[np.newaxis, :], reward, False, {}

    return _cls
