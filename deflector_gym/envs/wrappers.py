import gym


class BestRecorder(gym.Wrapper):
    def __init__(self, env):
        super(BestRecorder, self).__init__(env)
        
        self.max_eff = None
        self.best_struct = None

    def step(self, action):
        super(BestRecorder, self).step()

        if self.eff > self.max_eff:
            self.max_eff = self.eff
            self.best_struct = self.struct

    def reset(self):
        super(BestRecorder, self).reset()

        self.max_eff = self.eff
        self.best_struct = self.struct