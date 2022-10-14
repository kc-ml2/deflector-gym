from functools import partial

import gym
import numpy as np

from JLAB.solver import JLABCode
from .base import DeflectorBase
from .core import DirectionEnv


class MeentBase(DeflectorBase):
    def __init__(
            self,
            n_cells=256,
            wavelength=1100,
            desired_angle=70
    ):
        super().__init__(n_cells, wavelength, desired_angle)

    def get_efficiency(self, struct):
        # struct [1, -1, 1, 1, ...]
        struct = struct[np.newaxis, np.newaxis, :]

        wls = np.array([1100])
        period = abs(wls / np.sin(self.desired_angle / 180 * np.pi))
        calc = JLABCode(
            grating_type=0,
            n_I=1.45, n_II=1., theta=0, phi=0.,
            fourier_order=40, period=period,
            wls=wls, pol=1,
            patterns=None, ucell=struct, thickness=np.array([325])
        )

        eff, _, _ = calc.reproduce_acs_cell('p_si__real', 1)

        return eff


class ReticoloIndexEnv(MeentBase):
    def __init__(
            self,
            n_cells=256,
            wavelength=1100,
            desired_angle=70,
            *args,
            **kwargs
    ):
        super().__init__(n_cells, wavelength, desired_angle)

        self.observation_space = gym.spaces.Box(
            low=-1., high=1.,
            shape=(1, n_cells,),
            dtype=np.float64
        )
        self.action_space = gym.spaces.Discrete(n_cells)


ReticoloDirectionEnv = partial(DirectionEnv, base=MeentBase)
