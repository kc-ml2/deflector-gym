from functools import partial

import gymnasium as gym
import numpy as np

from .meent_utils import get_efficiency, get_field
from .constants import AIR, SILICON

from threadpoolctl import ThreadpoolController
controller = ThreadpoolController()


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
        mfs=1, # minimum feature size
        obs_type='efield', # 'efield' or 'struct'
        rew_type = 'eff' # 'eff' or 'delta_eff'
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
        self.obs_type = obs_type
        self.rew_type = rew_type

        self.max_eff = 0.
        self.prev_eff = 0.
        self.eff = 0.

        if obs_type == 'efield':
            self.observation_space = gym.spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(2, 256, 256), 
                dtype=np.float32
            )
        elif obs_type == 'struct':
            self.observation_space = gym.spaces.Box(
                low=-1., high=1.,
                shape=(1, n_cells),
                dtype=np.float32
            )
        else:
            raise NotImplementedError
        
        self.action_space = gym.spaces.Discrete(n_cells)

    @controller.wrap(limits=4)
    def reset(self, seed=42, options={}):
        info = {}
        
        self.struct = self.init_func(self.n_cells)
        if self.obs_type == 'efield':
            field = get_field(self.struct,
                wavelength=self.wavelength,
                deflected_angle=self.desired_angle,
                fourier_order=self.order,
                field_res=self.field_res
            )
            obs = np.stack([field.real, field.imag])
        elif self.obs_type == 'struct':
            obs = self.struct.copy()[np.newaxis]
        else:
            raise NotImplementedError

        self.eff = get_efficiency(
            self.struct,
            wavelength=self.wavelength,
            deflected_angle=self.desired_angle,
            fourier_order=self.order
        )

        if self.eff > self.max_eff:
            self.max_eff = self.eff
        info['max_eff'] = self.max_eff

        return obs, info

    @controller.wrap(limits=4)
    def step(self, action):
        info = {}

        self.flip(action)
        if self.obs_type == 'efield':
            field = get_field(self.struct,
                wavelength=self.wavelength,
                deflected_angle=self.desired_angle,
                fourier_order=self.order,
                field_res=self.field_res
            )
            obs = np.stack([field.real, field.imag])
        elif self.obs_type == 'struct':
            obs = self.struct.copy()[np.newaxis]
        else:
            raise NotImplementedError

        self.eff = get_efficiency(
            self.struct,
            wavelength=self.wavelength,
            deflected_angle=self.desired_angle,
            fourier_order=self.order
        )
        if self.eff > self.max_eff:
            self.max_eff = self.eff
        info['max_eff'] = self.max_eff

        if self.rew_type == 'eff':
            rew = float(self.eff)
        elif self.rew_type == 'delta_eff':
            rew = float(self.eff - self.prev_eff)
        
        return obs, rew, False, False, info

    def render(self, mode='human'):
        pass

    def flip(self, action):
        self.struct[action] = -self.struct[action]
        # self.struct[action] = AIR if self.struct[action] == SILICON else SILICON