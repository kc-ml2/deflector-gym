import numpy as np
import gym

from .utils import ga_init


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
            desired_angle=70
    ):
        self.n_cells = n_cells
        self.wavelength = wavelength
        self.desired_angle = desired_angle
        self.eff = None  # uninitialized

    def initialize_struct(self, *args, **kwargs):
        # default initialization is genetic algorithm(ga)
        return ga_init(*args, **kwargs)

    def get_efficiency(self, struct: np.array) -> float:
        raise NotImplementedError

    def flip(self, struct, pos):
        if 0 <= pos <= (self.n_cells - 1):
            struct[pos] = 1 if struct[pos] == -1 else -1
        else:
            # if out of boundary, do nothing
            # the agent will learn the boundary
            pass

        return struct

    def reset(self):  # initializing the env
        """
        usual routine
        """

        self.struct = self.initialize_struct()
        self.eff = self.get_efficiency(self.struct)

        return self.struct[np.newaxis, :]  # for 1 channel

    def step(self, action):
        """
        usual routine
        """
        # TODO: define done condition
        prev_eff = self.eff

        self.struct = self.flip(self.struct, action)
        self.eff = self.get_efficiency(self.struct)

        reward = self.eff - prev_eff

        # unsqueeze for 1 channel
        return self.struct[np.newaxis, :], reward, False, {}
