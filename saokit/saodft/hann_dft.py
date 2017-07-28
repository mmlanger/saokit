
from math import factorial

import numpy as np

from numpy.fft import fftfreq
from scipy.fftpack import fft
from scipy.special import binom


class FilteredWaveDFT:
    def __init__(self, N, T, p):
        self.N = N
        self.T = T
        self.p = p

        self.prefac = factorial(self.p)**2 / factorial(2*self.p)
        self.binom_coeffs = np.array([(-1)**l * binom(2*self.p, l+self.p)
                                      for l in range(self.p+1)])

    def fourier_coeff(self, omega, k):
        theta = omega*self.T - 2*k*np.pi
        dft_coeff = (np.exp(-1j*theta) - 1) / (np.exp(-1j*theta/self.N) - 1)
        return dft_coeff / self.N

    def filtered_coeff(self, omega, k):
        freq_coeff = self.binom_coeffs[0] * self.fourier_coeff(omega, k)
        for l in range(1, self.p+1):
            shifted_coeffs = (self.fourier_coeff(omega, k-l) +
                              self.fourier_coeff(omega, k+l))
            freq_coeff += self.binom_coeffs[l] * shifted_coeffs

        return self.prefac * freq_coeff


class FilteredDFT:
    def __init__(self, signal, dt=1.0, p=4):
        self.dt = dt
        self.p = p
        self.signal = np.array(signal)

        self.N = self.signal.size
        self.T = self.dt * self.N

        prefac = factorial(p)**2 / factorial(2*self.p)
        binom_coeffs = np.array([(-1)**l * binom(2*self.p, l+self.p)
                                 for l in range(self.p+1)])

        self.freqs = 2 * np.pi * fftfreq(self.N, self.dt)
        self.dft = fft(self.signal) / self.N

        self.filtered_dft = binom_coeffs[0] * self.dft
        for l in range(1, self.p+1):
            shifted_coeffs = np.roll(self.dft, -l) + np.roll(self.dft, +l)
            self.filtered_dft += binom_coeffs[l] * shifted_coeffs

        self.filtered_dft *= prefac
