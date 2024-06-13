import gym
import numpy as np

class BestRecorder(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.best = (None, None)  # efficiency, structure

    def step(self, action):
        ret = super().step(action)

        if self.eff > self.best[0]:
            self.best = (self.eff, self.struct.copy())

        return ret

    def reset(self, *args, **kwargs):
        ret = super().reset(*args, **kwargs)

        self.best = (self.eff, self.struct.copy())

        return ret

class ExpandObservation(gym.Wrapper):
    def __init__(self, env):
        super(ExpandObservation, self).__init__(env)
        obs_space = self.observation_space
        self.observation_space = gym.spaces.Box(
            low=obs_space.low[0], high=obs_space.high[0],
            shape=(1, *obs_space.shape),
            dtype=np.float64
        )

    def step(self, action):
        obs, rew, done, info = super(ExpandObservation, self).step(action)
        obs = obs.reshape(1, -1)

        return obs, rew, done, info

    def reset(self, **kwargs):
        obs = super(ExpandObservation, self).reset()
        obs = obs.reshape(1, -1)

        return obs