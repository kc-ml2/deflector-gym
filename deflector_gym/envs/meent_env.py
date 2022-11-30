from functools import partial

import gym
import numpy as np

from JLAB.solver import JLABCode
from .base import DeflectorBase
from .actions import Action1D2, Action1D4


class MeentBase(DeflectorBase):
    def __init__(
            self,
            n_cells=256,
            wavelength=1100,
            desired_angle=70,
            order=40,
            thickness=325,
    ):
        super().__init__(n_cells, wavelength, desired_angle, order, thickness)

    def get_efficiency(self, struct):
        # struct [1, -1, 1, 1, ...]
        struct = struct[np.newaxis, np.newaxis, :]

        wls = np.array([self.wavelength])
        period = abs(wls / np.sin(self.desired_angle / 180 * np.pi))
        calc = JLABCode(
            grating_type=0,
            n_I=1.45, n_II=1., theta=0, phi=0.,
            fourier_order=self.order, period=period,
            wls=wls, pol=1,
            patterns=None, ucell=struct, thickness=np.array([self.thickness])
        )

        eff, _, _ = calc.reproduce_acs_cell('p_si__real', 1)

        return eff


class MeentIndex(MeentBase):
    def __init__(
            self,
            n_cells=256,
            wavelength=1100,
            desired_angle=70,
            *args,
            **kwargs
    ):
        super().__init__(n_cells, wavelength, desired_angle, *args, **kwargs)

        self.observation_space = gym.spaces.Box(
            low=-1., high=1.,
            shape=(n_cells,), #### TODO fix shape
            dtype=np.float64
        )
        self.action_space = gym.spaces.Discrete(n_cells)

    def reset(self):
        self.struct = self.initialize_struct()
        self.eff = self.get_efficiency(self.struct)

        return self.struct.copy()

    def step(self, action):
        prev_eff = self.eff

        self.flip(action)
        self.eff = self.get_efficiency(self.struct)

        reward = self.eff - prev_eff

        # unsqueeze for 1 channel
        return self.struct.copy(), reward, False, {}


def initialize_agent(initial_pos, n_cells):
    # initialize agent
    if initial_pos == 'center':
        pos = n_cells // 2
    elif initial_pos == 'right_edge':
        pos = n_cells - 1
    elif initial_pos == 'left_edge':
        pos = 0
    elif initial_pos == 'random':
        pos = np.random.randint(n_cells)
    else:
        raise RuntimeError('Undefined inital position')

    return pos


class MeentAction1D2(MeentBase):
    def __init__(
            self,
            n_cells=256,
            wavelength=1100,
            desired_angle=70,
            initial_pos='center',  # initial agent's position
            *args,
            **kwargs
    ):
        super().__init__(n_cells, wavelength, desired_angle)

        self.observation_space = gym.spaces.Box(
            low=-1., high=1.,
            shape=(2*n_cells,),
            dtype=np.float64
        )
        self.action_space = gym.spaces.Discrete(len(Action1D2)) # start=-1
        self.initial_pos = initial_pos
        self.onehot = np.eye(n_cells)

    def reset(self):
        # initialize structure

        self.struct = self.initialize_struct()
        self.eff = self.get_efficiency(self.struct)
        self.pos = initialize_agent(self.initial_pos, self.n_cells)

        return np.concatenate((self.struct, self.onehot[self.pos]))

    def step(self, ac):
        prev_eff = self.eff
        # left == -1, noop == 0, right == 1
        # this way we can directly use ac as index difference
        ac -= 1

        self.flip(self.pos + ac)
        self.eff = self.get_efficiency(self.struct)

        reward = self.eff - prev_eff

        return np.concatenate((self.struct, self.onehot[self.pos])), reward, False, {}

class MeentAction1D4(MeentBase):
    def __init__(
            self,
            n_cells=256,
            wavelength=1100,
            desired_angle=70,
            initial_pos='center',  # initial agent's position
            *args,
            **kwargs
    ):
        super().__init__(n_cells, wavelength, desired_angle)

        self.observation_space = gym.spaces.Box(
            low=-1., high=1.,
            shape=(2*n_cells,),
            dtype=np.float64
        )
        self.action_space = gym.spaces.Discrete(len(Action1D4))
        self.initial_pos = initial_pos
        self.onehot = np.eye(n_cells)

    def reset(self):
        # initialize structure

        self.struct = self.initialize_struct(n_cells=self.n_cells)
        self.eff = self.get_efficiency(self.struct)
        self.pos = initialize_agent(self.initial_pos, self.n_cells)

        return np.concatenate((self.struct, self.onehot[self.pos]))

    def step(self, ac):
        prev_eff = self.eff

        if ac == Action1D4.RIGHT_SI.value and self.pos + 1 < self.n_cells:
            self.pos += 1
            self.struct[self.pos] = 1
        elif ac == Action1D4.RIGHT_AIR.value and self.pos + 1 < self.n_cells:
            self.pos += 1
            self.struct[self.pos] = -1
        elif ac == Action1D4.LEFT_SI.value and 0 <= self.pos - 1:
            self.pos -= 1
            self.struct[self.pos] = 1
        elif ac == Action1D4.LEFT_AIR.value and 0 <= self.pos - 1:
            self.pos -= 1
            self.struct[self.pos] = -1

        self.eff = self.get_efficiency(self.struct)

        reward = self.eff - prev_eff

        return np.concatenate((self.struct, self.onehot[self.pos])), reward, False, {}
