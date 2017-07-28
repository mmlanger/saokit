
import numpy as np
from scipy.optimize import brentq

from .hann_dft import FilteredDFT, FilteredWaveDFT


class SolverError(Exception):
    pass


class FrequencyEquation:
    def __init__(self, wave_dft, k, coeff_ratio):
        self.wave_dft = wave_dft
        self.k = k
        self.coeff_ratio = coeff_ratio

    def __call__(self, omega):
        numer = (abs(self.wave_dft.filtered_coeff(omega, self.k)) +
                 abs(self.wave_dft.filtered_coeff(omega, self.k-1)))
        denom = (abs(self.wave_dft.filtered_coeff(omega, self.k)) +
                 abs(self.wave_dft.filtered_coeff(omega, self.k+1)))
        analytic_ratio = numer / denom

        return analytic_ratio - self.coeff_ratio


class SpectralAnalysis:
    def __init__(self, signal, p=4, dt=1.0):
        self.signal = signal
        self.p = p
        self.dt = dt

        self.data_dft = FilteredDFT(self.signal, self.dt, self.p)
        self.wave_dft = FilteredWaveDFT(self.data_dft.N, self.data_dft.T,
                                        self.p)

    def reset(self):
        self.data_dft = FilteredDFT(self.signal, self.dt, self.p)
        self.wave_dft = FilteredWaveDFT(self.data_dft.N, self.data_dft.T,
                                        self.p)

    def compute_frequency(self, return_k=False, solver_xtol=5e-16):
        k, ratio = self.select_ratio()
        equation = FrequencyEquation(self.wave_dft, k, ratio)

        limit_shift = 1e-14
        om_min = self.data_dft.freqs[k - 1] + limit_shift
        om_max = self.data_dft.freqs[k + 1] - limit_shift

        try:
            omega = brentq(equation, om_min, om_max, xtol=solver_xtol)
        except ValueError:
            if self.p > 1:
                raise SolverError
            msg = "Warning: solver failed for p={}, using fallback solver"
            print(msg.format(self.p))
            return self.fallback_solver(return_k, solver_xtol)

        if return_k:
            return omega, k
        else:
            return omega

    def fallback_solver(self, return_k=False, solver_xtol=5e-16):
        dft = self.data_dft.dft
        coeffs = np.abs(dft - (np.roll(dft, -1) + np.roll(dft, 1)) / 2)
        k = coeffs.argmax()

        limit_shift = 1e-14
        om_min = self.data_dft.freqs[k - 1] + limit_shift
        om_max = self.data_dft.freqs[k + 1] - limit_shift

        ratio = (coeffs[k] + coeffs[k - 1]) / (coeffs[k] + coeffs[k + 1])
        wave_dft = FilteredWaveDFT(self.data_dft.N, self.data_dft.T, 1)
        equation = FrequencyEquation(wave_dft, k, ratio)

        try:
            omega = brentq(equation, om_min, om_max, xtol=solver_xtol)
        except ValueError:
            raise SolverError

        if return_k:
            return omega, k
        else:
            return omega

    def extract_frequency(self, return_ampl=False):
        omega, k = self.compute_frequency(True)
        indices = np.arange(self.wave_dft.N)

        wave_dft = self.wave_dft.fourier_coeff(omega, indices)
        wave_dft_conj = np.insert(wave_dft[1:][::-1], 0,
                                  wave_dft[0]).conj()

        filtered_wave_dft = self.wave_dft.filtered_coeff(omega, indices)
        filtered_wave_dft_conj = np.insert(filtered_wave_dft[1:][::-1], 0,
                                           filtered_wave_dft[0]).conj()

        amplitude = self.data_dft.filtered_dft[k] / filtered_wave_dft[k]
        amplitude_conj = self.data_dft.filtered_dft[-k] / \
                         filtered_wave_dft_conj[-k]

        self.data_dft.dft -= amplitude * wave_dft + amplitude_conj * wave_dft_conj
        self.data_dft.filtered_dft -= (amplitude * filtered_wave_dft +
                                       amplitude_conj * filtered_wave_dft_conj)

        if return_ampl:
            return omega, amplitude, amplitude_conj
        else:
            return omega

    def select_ratio(self):
        # TODO: ignore coefficients F[-p:p+1] ? (constant mode problem)
        coeffs = np.abs(self.data_dft.filtered_dft)
        coeffs[:self.p + 1] = 0.0
        coeffs[::-1][:self.p + 1] = 0.0

        k = coeffs.argmax()
        ratio = (coeffs[k] + coeffs[k - 1]) / (coeffs[k] + coeffs[k + 1])

        return k, ratio

    def frequencies(self, n_freqs, coeff_limit=1e-10, return_ampl=False):
        for i in range(n_freqs):
            if np.abs(self.data_dft.filtered_dft).max() > coeff_limit:
                yield self.extract_frequency(return_ampl)
            else:
                break