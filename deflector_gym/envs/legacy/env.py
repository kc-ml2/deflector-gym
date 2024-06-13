import os
from functools import partial
from pathlib import Path

import gym
import numpy as np

from utils import random_bunch_init
from actions import Action1D2

try:
    import matlab.engine
except:
    raise Warning(
        'matlab python API not installed, '
        'try installing pip install matlabengine=={YOUR MATLAB VERSION}'
    )

from JLAB.solver import JLABCode
from .base import DeflectorBase
from .actions import Action1D2, Action1D4


RETICOLO_MATLAB = os.path.join(Path(__file__).parent.parent.parent.absolute(), 'third_party/reticolo_allege')
SOLVER_MATLAB = os.path.join(Path(__file__).parent.parent.parent.absolute(), 'third_party/solvers')


def badcell(img, mfs):
    img = np.array(img)
    len = img.size
    sz = mfs+2
    window = np.ones(sz)
    window[0] = -1
    window[-1] = -1
    imgextend = np.concatenate((img, img))
    volved = np.convolve(imgextend, window)
    output = volved[sz-1:len+sz-1]

    return np.sum(np.floor(np.abs(output/sz)))

def underMFS(img, mfs):
    num = 0
    for i in range(1,mfs):
        num += badcell(img, i)
    return num


class DeflectorBase(gym.Env):
    """
    This is interface class for deflection simulation
    You will need to inherit this interface and,
    implement abstract methods such as get_efficiency,
    define observation spaces, etc
    """
    def __init__(
            self,
            n_cells=256,
            wavelength=1100,
            desired_angle=70,
            order=40,
            thickness=325,
    ):
        self.thickness = thickness
        self.order = order
        self.n_cells = n_cells
        self.wavelength = wavelength
        self.desired_angle = desired_angle
        self.struct = None
        self.eff = None  # uninitialized

    def initialize_struct(self, *args, **kwargs):
        # default initialization is genetic algorithm(ga)
        return random_bunch_init(*args, **kwargs)

    def get_efficiency(self, struct: np.array) -> float:
        raise NotImplementedError

    def flip(self, pos):
        if 0 <= pos <= (self.n_cells - 1):
            self.struct[pos] = 1 if self.struct[pos] == -1 else -1
        else:
            # if out of boundary, do nothing
            # the agent will learn the boundary
            pass

    
    from functools import partial


class MeentBase(DeflectorBase):
    def __init__(
        self,
        n_cells=256,
        wavelength=1100,
        desired_angle=70,
        order=40,
        thickness=325,
        alpha=0.05,
        mfs=1 # minimum feature size
    ):
        super().__init__(n_cells, wavelength, desired_angle, order, thickness)
        self.alpha = 0.05
        self.mfs = mfs

    def get_efficiency(self, struct):
        # struct [1, -1, 1, 1, ...]
        penalty = 0.0
        if self.mfs > 1:
            penalty = underMFS(struct, self.mfs) * self.alpha
     
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

        return eff - penalty


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
            matlab.double(struct.tolist()),
            self.wavelength_mtl,
            self.desired_angle_mtl
        )

    def close(self):
        # recommend to close env when done
        self.eng.quite()


class ReticoloIndex(MatlabBase):
    """
    legacy env for compatibility with chaejin's UNet
    will move to meent later on
    """
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
            shape=(n_cells,),
            dtype=np.float64
        )
        self.action_space = gym.spaces.Discrete(n_cells)

    def reset(self):
        self.struct = self.initialize_struct()
        self.eff = self.get_efficiency(self.struct)

        return self.struct.copy()  # for 1 channel

    def step(self, action):
        prev_eff = self.eff

        self.flip(action)
        self.eff = self.get_efficiency(self.struct)

        reward = self.eff - prev_eff

        # unsqueeze for 1 channel
        return self.struct.copy(), reward, False, {}