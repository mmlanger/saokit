
import numpy as np

from saokit.saodft import SpectralAnalysis

def test_success():
    assert True


def test_trivial():
    omega = 2.1234
    time_step = 0.1
    n = 800

    t = np.arange(0, n*time_step + 0.5*time_step, time_step)
    y = np.cos(omega * t)

    analysis = SpectralAnalysis(y, p=4, dt=time_step)
    result = analysis.compute_frequency()


    assert abs(abs(result) - omega) < 1e-12
