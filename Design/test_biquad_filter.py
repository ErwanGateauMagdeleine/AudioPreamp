"""
This module implements a biquad filter test class.
"""

import unittest
import numpy as np
from biquad_filter import Biquad, BiquadFilterOutputs


class TestBiquadFilter(unittest.TestCase):
    """
    State variable filter test class.
    """

    NUM_POINTS_FREQ_RESPONSE = 96000

    def test_band_pass(self):
        """
        Test the band pass biquad filter.
        It first checks that no value of the frequency response is above 0dB.
        Then it checks that the set cutoff frequecies are correct.
        It finally checks that the gain are in the correct range for the different
        filter parts.
        """
        sample_rate = 44100
        fl = 75
        fh = 190
        bw = fh - fl
        f0 = np.sqrt(fl * fh)
        q = f0 / bw

        bq = Biquad(sample_rate, f0, q, BiquadFilterOutputs.BAND_PASS)
        w, h = bq.theoretical_frequency_response(self.NUM_POINTS_FREQ_RESPONSE)
        h = 20 * np.log10(abs(h))
        self.assertIsNone(np.testing.assert_array_less(h, 0))

        h  = [True if h_idx > -3 else False for h_idx in h]
        fl_index = np.nonzero(h)[0][0]
        fh_index = np.nonzero(h)[0][-1]
        fc1 = w[fl_index]
        fc2 = w[fh_index]
        
        self.assertAlmostEqual(fl, fc1, delta=0.3)
        self.assertAlmostEqual(fh, fc2, delta=0.3)

        self.assertTrue(np.all([x == False for x in h[0 : fl_index -1]]))
        self.assertTrue(np.all([x == True for x in h[fl_index : fh_index]]))
        self.assertTrue(np.all([x == False for x in h[fh_index + 1:]]))

