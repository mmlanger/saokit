
import numpy as np

from .analysis import SpectralAnalysis, SolverError


def freq_diff(signal, p=4, dt=1.0):
    signal_arr = np.array(signal)
    n = signal_arr.size

    seg1 = signal_arr[:n//2]
    seg2 = signal_arr[n//2:]

    om1 = abs(SpectralAnalysis(seg1, p=p, dt=dt).compute_frequency())
    om2 = abs(SpectralAnalysis(seg2, p=p, dt=dt).compute_frequency())

    return abs(om1-om2)
