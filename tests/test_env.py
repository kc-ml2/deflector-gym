import deflector_gym

env = deflector_gym.make('MeentIndex-v0')
print(env.reset())
env = deflector_gym.make('MeentDirection-v0')
print(env.reset())

env = deflector_gym.make('ReticoloIndex-v0')
print(env.reset())
env = deflector_gym.make('ReticoloDirection-v0')
print(env.reset())
