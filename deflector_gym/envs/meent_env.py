from functools import partial

import gymnasium as gym
import numpy as np

from .meent_utils import get_efficiency, get_field
from .constants import AIR, SILICON


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

class MeentIndexEfield(gym.Env):
    def __init__(
        self,
        n_cells=256,
        init_func=np.ones,
        wavelength=1100,
        desired_angle=70,
        order=40,
        thickness=325,
        field_res=(256, 1, 32),
        mfs=1 # minimum feature size
    ):
        super().__init__()
        
        self.n_cells = n_cells
        self.init_func = init_func
        self.wavelength = wavelength
        self.desired_angle = desired_angle
        self.order = order
        self.thickness = thickness
        self.field_res = field_res
        self.mfs = mfs

        self.max_eff = 0.
        self.prev_eff = 0.
        self.eff = 0.
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2, 256, 256), dtype=np.float32) # fix shape
        self.action_space = gym.spaces.Discrete(n_cells)

    def reset(self, seed, options):
        self.struct = self.init_func(self.n_cells)
        _, field = get_field(self.struct,
            wavelength=self.wavelength,
            deflected_angle=self.desired_angle,
            fourier_order=self.order,
            field_res=self.field_res
        )
        field = np.stack([field.real, field.imag])

        info = {}
        info['max_eff'] = self.max_eff

        return field, info

    def step(self, action):
        info = {}

        self.flip(action)

        _, field = get_field(self.struct,
            wavelength=self.wavelength,
            deflected_angle=self.desired_angle,
            fourier_order=self.order,
            field_res=self.field_res
        )
        field = np.stack([field.real, field.imag])

        self.eff = get_efficiency(
            self.struct,
            wavelength=self.wavelength,
            deflected_angle=self.desired_angle,
            fourier_order=self.order
        )
        if self.eff > self.max_eff:
            self.max_eff = self.eff

        delta_eff = self.prev_eff - self.eff
        self.prev_eff = self.eff 

        info['max_eff'] = float(self.max_eff)

        return field, delta_eff, False, False, info

    def render(self, mode='human'):
        pass

    def flip(self, action):
        self.struct[action] = -self.struct[action]
        # self.struct[action] = AIR if self.struct[action] == SILICON else SILICON