"""
This script implements a 4 way crossover and plots its frequency response
"""

import numpy as np
from biquad_filter import Biquad, BiquadFilterOutputs
import matplotlib.pyplot as plt

n_points = 50000
sample_rate = 44100
bands = [(28, 75), (75, 190), (190, 3500), (3500, 20000)]
band_names = ["SUB", "BASS", "MID", "HI"]

summed_answer = np.zeros(n_points)
for band_idx, freqs in enumerate(bands):
    center_freq = np.sqrt(freqs[0] * freqs[1])
    quality_factor = center_freq / (freqs[1] - freqs[0])
    band_filter = Biquad(sample_rate, center_freq, quality_factor, BiquadFilterOutputs.BAND_PASS)
    w, h = band_filter.theoretical_frequency_response(n_points)
    summed_answer = np.add(summed_answer, h)
    plt.semilogx(w, 20 * np.log10(abs(h)), label=band_names[band_idx])

plt.semilogx(w, 20 * np.log10(abs(summed_answer)), label="SUM")
plt.title("Crossover frequency response")
plt.legend()
plt.show()