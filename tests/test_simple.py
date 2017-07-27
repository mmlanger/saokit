
import numpy as np

import saokit.saodft
from saokit.saodft import SpectralAnalysis

def test_success():
    assert True


def test_trivial():
    omega = 2.1234
    dt = 0.1
    n = 800

    t = np.arange(0, n*dt, dt)
    y = np.cos(omega * t)

    analysis = SpectralAnalysis(y, p=4, delta_t=dt)
    result = analysis.compute_frequency()


    assert abs(abs(result) - omega) < 1e-12
