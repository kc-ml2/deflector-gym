# deflector-gym

Simple RL environment for beam deflector design, built on top of [Meent](https://github.com/kc-ml2/meent).

Currently, only one-dimensional metasurface environment is implemented.

## Setup
```
git clone https://github.com/kc-ml2/deflector-gym
cd deflector-gym
pip install -e .
```

## How to

```python
import deflector_gym
env = deflector_gym.make('MeentIndex-v0')
obs, info = env.reset()
for step in range(10):
  env.step(env.action_space.sample())
```

## Examples
`MeentIndex-v0` was used in [Park, Kim, Jung, 2024](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=oIw79eEAAAAJ&citation_for_view=oIw79eEAAAAJ:u-x6o8ySG0sC).

`MeentIndexEfield-v0` was used in model-based RL example of Meent.
