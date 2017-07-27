import numpy as np

from numpy.fft import fftfreq
from scipy.fftpack import fft
from scipy.misc import factorial
from scipy.special import binom


class FilteredWaveDFT:
    def __init__(self, N, T, p):
        self.N = N
        self.T = T
        self.p = p

        self.prefac = factorial(p, exact=True)**2 / factorial(2*p, exact=True)
        self.binom_coeffs = np.array([(-1)**l * binom(2*p, l+p)
                                      for l in range(p+1)])

    def fourier_coeff(self, omega, k):
        theta = omega * self.T - 2 * k * np.pi
        return (np.exp(-1j * theta) - 1) / (
            np.exp(-1j * theta / self.N) - 1) / self.N

    def filtered_coeff(self, omega, k):
        freq_coeff = self.binom_coeffs[0] * self.fourier_coeff(omega, k)
        for l in range(1, self.p+1):
            shifted_coeffs = (self.fourier_coeff(omega, k-l) +
                              self.fourier_coeff(omega, k+l))
            freq_coeff += self.binom_coeffs[l] * shifted_coeffs

        return self.prefac * freq_coeff


class FilteredDFT:
    def __init__(self, signal, delta_t=1.0, p=4):
        self.delta_t = delta_t
        self.signal = np.array(signal)

        self.N = self.signal.size
        self.T = self.delta_t * self.N

        prefac = factorial(p, exact=True) ** 2 / factorial(2 * p, exact=True)
        binom_coeffs = np.array(
            [(-1) ** l * binom(2 * p, l + p) for l in range(p + 1)])

        self.p = p
        self.freqs = 2 * np.pi * fftfreq(self.N, self.delta_t)
        self.dft = fft(self.signal) / self.N

        self.filtered_dft = binom_coeffs[0] * self.dft
        for l in range(1, self.p + 1):
            shifted_coeffs = np.roll(self.dft, -l) + np.roll(self.dft, +l)
            self.filtered_dft += binom_coeffs[l] * shifted_coeffs

        self.filtered_dft *= prefac
