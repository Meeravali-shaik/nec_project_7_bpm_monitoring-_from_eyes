"""
chrom_rppg.py
CHROM (Chrominance-based) rPPG method.
Reference: De Haan & Jeanne (2013), IEEE TBME.

Extracts the pulse signal from RGB colour channels by projecting
onto a chrominance plane that suppresses motion and illumination changes.
"""

import numpy as np


class CHROMExtractor:
    """
    Accumulates per-frame mean RGB values from an ROI (e.g. eye region)
    and computes the CHROM pulse signal on demand.
    """

    def __init__(self):
        self.rgb_buffer = []   # list of [R, G, B] means

    def update(self, roi_bgr: np.ndarray):
        """
        Add one frame's mean colour from a BGR ROI patch.
        roi_bgr: numpy array of shape (H, W, 3) in BGR order.
        """
        if roi_bgr is None or roi_bgr.size == 0:
            return
        mean_bgr = roi_bgr.mean(axis=(0, 1))   # [B, G, R]
        R, G, B = float(mean_bgr[2]), float(mean_bgr[1]), float(mean_bgr[0])
        self.rgb_buffer.append([R, G, B])

    def get_signal(self) -> np.ndarray:
        """
        Return the CHROM pulse signal for the buffered frames.
        Returns empty array if fewer than 10 samples.
        """
        if len(self.rgb_buffer) < 10:
            return np.array([])

        rgb = np.array(self.rgb_buffer, dtype=np.float64)   # (N, 3)

        # Normalise each channel by its mean (handles illumination drift)
        means = rgb.mean(axis=0)
        means[means < 1e-6] = 1e-6
        rgb_n = rgb / means   # normalised

        R, G, B = rgb_n[:, 0], rgb_n[:, 1], rgb_n[:, 2]

        # CHROM projection
        Xs =  3*R - 2*G          # skin-tone axis
        Ys =  1.5*R + G - 1.5*B  # chrominance axis

        std_xs = np.std(Xs)
        std_ys = np.std(Ys)
        alpha  = std_xs / (std_ys + 1e-9)

        pulse = Xs - alpha * Ys
        return pulse

    def clear(self):
        self.rgb_buffer.clear()

    def __len__(self):
        return len(self.rgb_buffer)
