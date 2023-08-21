"""
This module implements a digital state variable filter test class.
"""

import unittest
import numpy as np
from state_variable_filter import StateVariableFilter, StateVariableFilterOutputs


class TestStateVariableFilter(unittest.TestCase):
    """
    State variable filter test class.
    """

    INPUT_LENGTH = 3000
    FFT_LENGTH = 16 * INPUT_LENGTH
    SAMPLE_RATE = 44100
    CUTOFF_FREQUENCIES = [500, 1500, 5000]
    Q_FACTORS = [0.5, 2, 4, 8]

    def measure_frequency_response(self, dut):
        """
        Measure the frequency response of the test. It will first measure the impulse response
        of the filter and apply Fourier transform to it to deduce the frequency response of the filter.

        Parameters:
            dut(StateVariableFilterOutputs): The state variable filter to be tested.
        """
        impulse_input = np.zeros(TestStateVariableFilter.INPUT_LENGTH)
        impulse_input[0] = 1

        # Pre-allocate output vector
        filtered_signal = np.zeros(TestStateVariableFilter.INPUT_LENGTH)
        for idx in range(TestStateVariableFilter.INPUT_LENGTH):
            filtered_signal[idx] = dut.process_sample(impulse_input[idx])
        
        measured_response = 20 * np.log10(np.abs(np.fft.rfft(filtered_signal, 24 * len(filtered_signal))))
        frequencies = np.fft.rfftfreq(24 * len(filtered_signal), 1 / TestStateVariableFilter.SAMPLE_RATE)
        return frequencies, measured_response

    def test_low_pass(self):
        """
        Tests the low pass configuration of the state variable filter. 
        """
        dut = StateVariableFilter(TestStateVariableFilter.SAMPLE_RATE, 1500, 0.5, StateVariableFilterOutputs.LOW_PASS)

        for cutoff_freq in TestStateVariableFilter.CUTOFF_FREQUENCIES:
            for q_factor in TestStateVariableFilter.Q_FACTORS:
                dut.f_c = cutoff_freq
                dut.q_fact = q_factor
                dut.reset()
                frequencies, measured_frequency_response = self.measure_frequency_response(dut)
                theoretical_frequency_response = dut.theoretical_frequency_response(frequencies)
                self.assertIsNone(np.testing.assert_array_almost_equal(measured_frequency_response, theoretical_frequency_response, 4))
