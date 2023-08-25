"""
This module implements a biquad filter class.
"""

from math import pi
from enum import Enum
import numpy as np
import scipy.signal as sig
import matplotlib.pyplot as plt

class BiquadFilterOutputs(Enum):
    """Enumeration holding the different biquads filter output types."""
    LOW_PASS = 0
    HIGH_PASS = 1
    BAND_PASS = 2
    NOTCH = 3


class Biquad():
    """
    Biquad filter class.

    Parameters:
        sample_rate(float): Sample rate at which the filter run in Hz.
        f_c(float): Center frequency of the filter in Hz.
        q_fact(float): Quality factor of the filter.
        filter_type(BiquadFilterOutputs): Type of the filter.
    """
    def __init__(self, sample_rate, f_c, q_fact, filter_type):
        self.sample_rate = sample_rate
        self.f_c = f_c
        self.q_fact = q_fact
        self.filter_type = filter_type
        self.b = np.zeros(3)
        self.a = np.zeros(3)
        self._calculate_filter_coefficients()

    def _calculate_filter_coefficients(self):
        """Cacluclates the filter coefficients."""
        if self.filter_type == BiquadFilterOutputs.BAND_PASS:
            omega_0 = 2 * pi * self.f_c / self.sample_rate
            alpha = np.sin(omega_0) / (2 * self.q_fact)

            self.b[0] = alpha
            self.b[1] = 0
            self.b[2] = - alpha

            self.a[0] = 1 + alpha
            self.a[1] = - 2 * np.cos(omega_0)
            self.a[2] = 1 - alpha
    
    def theoretical_frequency_response(self, worN):
        """
        Returns the theoretical frequency response of the calculated biquad filter.

        Parameters:
            worN (float): The number of points the frequency response is calculated on.
        """
        return sig.freqz(self.b, self.a, worN=worN, fs=self.sample_rate)
