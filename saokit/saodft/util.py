
import numpy as np

from .analysis import SpectralAnalysis


def freq_diff(signal, p=4, dt=1.0, return_freq=False):
    signal_arr = np.array(signal)
    half_n = signal_arr.size // 2

    seg1 = signal_arr[:half_n]
    seg2 = signal_arr[half_n:]

    om1 = abs(SpectralAnalysis(seg1, p=p, dt=dt).compute_frequency())
    om2 = abs(SpectralAnalysis(seg2, p=p, dt=dt).compute_frequency())
    diff_val = abs(om1 - om2)

    if return_freq:
        om = abs(SpectralAnalysis(signal_arr, p=p, dt=dt).compute_frequency())
        return diff_val, om
    else:
        return diff_val
