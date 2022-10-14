import os
from functools import partial
from pathlib import Path

import gym
import numpy as np

from .base import DeflectorBase
from .core import DirectionEnv

try:
    import matlab.engine
except:
    raise Warning(
        'matlab python API not installed, '
        'try installing pip install matlabengine=={YOUR MATLAB VERSION}'
    )

RETICOLO_MATLAB = os.path.join(Path().absolute().parent, 'third_party/reticolo_allege')
SOLVER_MATLAB = os.path.join(Path().absolute().parent, 'third_party/solvers')


class MatlabBase(DeflectorBase):
    def __init__(
            self,
            n_cells=256,
            wavelength=1100,
            desired_angle=70
    ):
        super().__init__(n_cells, wavelength, desired_angle)

        self.eng = matlab.engine.start_matlab()
        self.eng.addpath(self.eng.genpath(RETICOLO_MATLAB))
        self.eng.addpath(self.eng.genpath(SOLVER_MATLAB))
        self.wavelength_mtl = matlab.double([wavelength])
        self.desired_angle_mtl = matlab.double([desired_angle])

    def get_efficiency(self, struct: np.array):
        return self.eng.Eval_Eff_1D(
            matlab.double(struct),
            self.wavelength_mtl,
            self.desired_angle_mtl
        )


class ReticoloIndexEnv(MatlabBase):
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


ReticoloDirectionEnv = partial(DirectionEnv, base=MatlabBase)
