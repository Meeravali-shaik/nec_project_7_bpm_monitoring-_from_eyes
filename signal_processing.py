"""
signal_processing.py
Bandpass filter and utilities for rPPG signal cleaning.
"""

import numpy as np
from scipy.signal import butter, filtfilt


# Normal resting heart-rate band: 0.75 Hz – 3.0 Hz  (45–180 BPM)
BPM_LOW  = 0.75   # Hz
BPM_HIGH = 3.0    # Hz


def bandpass_filter(signal: np.ndarray, fs: float,
                    low: float = BPM_LOW, high: float = BPM_HIGH,
                    order: int = 4) -> np.ndarray:
    """
    Apply a zero-phase Butterworth bandpass filter.
    Requires len(signal) > padlen ~ 3*(order+1)*2 ≈ 30 samples.
    """
    nyq = fs / 2.0
    low_n  = low  / nyq
    high_n = high / nyq

    # Clamp to valid range
    low_n  = max(1e-4, min(low_n,  0.99))
    high_n = max(1e-4, min(high_n, 0.99))
    if low_n >= high_n:
        return signal

    min_len = 3 * (order + 1) * 2 + 1
    if len(signal) < min_len:
        return signal

    b, a = butter(order, [low_n, high_n], btype='band')
    return filtfilt(b, a, signal)


def detrend(signal: np.ndarray) -> np.ndarray:
    """Remove linear trend."""
    return signal - np.polyval(np.polyfit(np.arange(len(signal)), signal, 1),
                               np.arange(len(signal)))


def normalize(signal: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-variance normalisation."""
    std = np.std(signal)
    if std < 1e-9:
        return signal - np.mean(signal)
    return (signal - np.mean(signal)) / std
