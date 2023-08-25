"""
This module implements a digital state variable filter class.

State variable filters are second order filters. They offer a an independent
control over the center frequency and quality factor. A single circuit can
provide low-pass, high-pass, band-pass and notch outputs simultaneously.
"""

from enum import Enum
from math import pi
import numpy as np


class StateVariableFilterOutputs(Enum):
    """Enumeration holding the different state variable filter output types."""
    LOW_PASS = 0
    HIGH_PASS = 1
    BAND_PASS = 2
    NOTCH = 3

class StateVariableFilter:
    """
    State variable filter class.

    Parameters:
        sample_rate(float): Sample rate at which the filter run in Hz.
        f_c(float): Center frequency of the filter in Hz.
        q_fact(float): Quality factor of the filter.
        filter_type(StateVariableFilterOutputs): Type of the filter.
    """
    def __init__(self, sample_rate, f_c, q_fact, filter_type):
        self.sample_rate = sample_rate
        self.outputs = np.zeros(4)
        self.d_1 = 0
        self.d_2 = 0

        self.f_c = f_c
        self.q_fact = q_fact
        self.filter_type = filter_type

    @property
    def f_c(self):
        return self._f_c

    @f_c.setter
    def f_c(self, f_c):
        self._f_c = f_c
        self.freq_control = 2 * pi * f_c / self.sample_rate

    @property
    def q_fact(self):
        return self._q

    @f_c.setter
    def q_fact(self, q_fact):
        self._q = q_fact
        self.q_control = 1 / q_fact

    def process_sample(self, xn):
        """
        Processes one sample through the state variable filter.

        Parameters
            xn (float): The input sample.
        """
        self.outputs[StateVariableFilterOutputs.LOW_PASS.value] = self.d_2 + self.freq_control * self.d_1
        self.outputs[StateVariableFilterOutputs.HIGH_PASS.value] = \
            xn - self.outputs[StateVariableFilterOutputs.LOW_PASS.value] - self.q_control * self.d_1
        self.outputs[StateVariableFilterOutputs.BAND_PASS.value] = \
            self.outputs[StateVariableFilterOutputs.HIGH_PASS.value] * self.freq_control + self.d_1
        self.outputs[StateVariableFilterOutputs.NOTCH.value] = \
            (self.outputs[StateVariableFilterOutputs.HIGH_PASS.value] +
             self.outputs[StateVariableFilterOutputs.LOW_PASS.value])

        self.d_1 = self.outputs[StateVariableFilterOutputs.BAND_PASS.value]
        self.d_2 = self.outputs[StateVariableFilterOutputs.LOW_PASS.value]

        return self.outputs[self.filter_type.value]
    
    def theoretical_frequency_response(self, frequency_vector):
        """
        Computes the theoretical frequency response of the state variable filter.

        Parameters:
            frequency_vector(numpy.ndarray): Array containing the frequencies at which the theoretical
                                             frequency response has to be evaluated.
        """
        resp = np.zeros(frequency_vector.shape, dtype=np.complex_)
        z = np.exp(-1 * 1j * 2 * np.pi * frequency_vector / self.sample_rate)

        if self.filter_type == StateVariableFilterOutputs.LOW_PASS:
            resp = 20 * np.log10(np.abs(np.divide(np.power(self.freq_control, 2) * z,
                                        1
                                        - z * (2 - self.freq_control * self.q_control - np.power(self.freq_control, 2))
                                        + np.power(z, 2) * (1 - self.freq_control * self.q_control))))
        else:
            # Not yet implemented.
            assert(False)

        return resp

    def reset(self):
        """
        Resets the filter's internals i.e. tap delays and outputs.
        """
        self.outputs = np.zeros(4)
        self.d_1 = 0
        self.d_2 = 0