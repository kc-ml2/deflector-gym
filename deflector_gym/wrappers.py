import gym


class BestRecorder(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.best = (None, None)  # efficiency, structure

    def step(self, action):
        ret = super().step(action)

        if self.eff > self.best[0]:
            self.best = (self.eff, self.struct)

        return ret

    def reset(self, *args, **kwargs):
        ret = super().reset(*args, **kwargs)

        self.best = (self.eff, self.struct)

        return ret