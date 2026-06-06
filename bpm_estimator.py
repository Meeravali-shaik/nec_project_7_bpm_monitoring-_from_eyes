"""
bpm_estimator.py
Estimates BPM from a 1-D pulse signal using FFT peak detection.
"""

import numpy as np
from signal_processing import bandpass_filter, detrend, normalize

BPM_MIN = 45
BPM_MAX = 180


class BPMEstimator:
    def __init__(self, fs: float = 30.0):
        self.fs = fs          # sampling frequency (camera FPS)
        self.bpm   = 0.0
        self.freq  = 0.0
        self.fft_freqs  = np.array([])
        self.fft_power  = np.array([])

    def estimate(self, signal: np.ndarray) -> float:
        """
        Given a raw pulse signal array, return BPM.
        Also stores .freq, .fft_freqs, .fft_power for visualisation.
        """
        if len(signal) < 30:
            return 0.0

        # Pre-process
        sig = detrend(signal.copy())
        sig = bandpass_filter(sig, self.fs)
        sig = normalize(sig)

        # FFT
        N      = len(sig)
        window = np.hanning(N)
        fft_vals  = np.abs(np.fft.rfft(sig * window))
        fft_freqs = np.fft.rfftfreq(N, d=1.0 / self.fs)
        power     = fft_vals ** 2

        # Restrict to heart-rate band
        mask = (fft_freqs >= BPM_MIN / 60.0) & (fft_freqs <= BPM_MAX / 60.0)
        if not np.any(mask):
            return 0.0

        band_freqs = fft_freqs[mask]
        band_power = power[mask]

        peak_idx   = np.argmax(band_power)
        peak_freq  = band_freqs[peak_idx]

        self.freq      = peak_freq
        self.bpm       = peak_freq * 60.0
        self.fft_freqs = fft_freqs
        self.fft_power = power

        return self.bpm
