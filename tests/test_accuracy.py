import numpy as np
from numpy.testing import assert_almost_equal

import deflector_gym

for i in range(1000):
    test_struct = np.random.choice([-1, 1], 256)
    m_env = deflector_gym.make('MeentIndex-v0')
    m_eff = m_env.get_efficiency(test_struct)
    print(m_eff)
    r_env = deflector_gym.make('ReticoloIndex-v0')
    r_eff = r_env.get_efficiency(test_struct)
    print(f'{m_eff} {r_eff}')

    np.testing.assert_almost_equal(m_eff, r_eff)
