"""
app.py
Main entry point for the Eye-based rPPG Heart Rate Monitor.
All visualisation is pure OpenCV — no Matplotlib, no Tkinter.

Safety gates added:
  1. EAR gate      — eyes must be open (checked in EyeDetector)
  2. Quality gate  — signal variance must be above MIN_SIGNAL_VARIANCE
  3. Buffer reset  — buffer & BPM cleared when face/eyes lost

Run:
    python app.py

Controls:
    Q / ESC  — quit
    R        — reset signal buffer manually
"""

import cv2
import numpy as np
import collections
import time

from webcam_capture   import WebcamCapture
from eye_detector     import EyeDetector
from chrom_rppg       import CHROMExtractor
from bpm_estimator    import BPMEstimator
from dashboard        import Dashboard

# ── Configuration ────────────────────────────────────────────────────────────
CAMERA_INDEX        = 0
BUFFER_SECONDS      = 20      # seconds of signal to keep
MIN_BPM_SECONDS     = 5       # seconds before attempting BPM estimate
WINDOW_NAME         = "rPPG Heart Rate Monitor"

# Gate 2: minimum signal variance — below this the signal is too flat
# (face covered, very still lighting, or eyes closed giving uniform ROI)
MIN_SIGNAL_VARIANCE = 1e-5

# How many consecutive missing-face frames before we reset the buffer
MAX_MISSING_FRAMES  = 15      # ~0.5 s at 30 fps


def get_eye_roi(frame, eye_pts):
    """Return a tightly-cropped BGR patch around the eye landmarks."""
    if eye_pts is None or len(eye_pts) == 0:
        return None
    x1 = max(0, int(eye_pts[:, 0].min()) - 4)
    y1 = max(0, int(eye_pts[:, 1].min()) - 4)
    x2 = min(frame.shape[1], int(eye_pts[:, 0].max()) + 4)
    y2 = min(frame.shape[0], int(eye_pts[:, 1].max()) + 4)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def _reset_all(chrom, signal_buf):
    """Clear CHROM buffer and signal deque."""
    chrom.clear()
    signal_buf.clear()


def main():
    cam       = WebcamCapture(CAMERA_INDEX)
    detector  = EyeDetector()
    chrom     = CHROMExtractor()
    estimator = BPMEstimator(fs=cam.fps)
    dash      = Dashboard()

    fps        = cam.fps or 30.0
    buf_size   = int(BUFFER_SECONDS * fps)
    signal_buf = collections.deque(maxlen=buf_size)

    bpm       = 0.0
    freq      = 0.0
    fft_freqs = np.array([])
    fft_power = np.array([])
    status    = "Warming up…"

    # Gate 3 counter — consecutive frames without valid eyes
    missing_frames = 0

    last_estimate_t = time.time()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    print("[INFO] Starting rPPG monitor. Press Q to quit, R to reset buffer.")

    while True:
        ok, frame = cam.read()
        if not ok:
            print("[WARN] Frame grab failed, retrying…")
            continue

        # ── Gate 1 — Eye detection + EAR check (inside EyeDetector) ──────
        all_pts, left_pts, right_pts = detector.detect(frame)

        face_found = all_pts is not None
        eyes_open  = left_pts is not None and right_pts is not None

        # ── Gate 3 — Buffer invalidation on sustained detection loss ──────
        if not eyes_open:
            missing_frames += 1
            if missing_frames >= MAX_MISSING_FRAMES:
                if len(signal_buf) > 0:          # only reset once
                    _reset_all(chrom, signal_buf)
                    bpm, freq = 0.0, 0.0
                    fft_freqs = np.array([])
                    fft_power = np.array([])
                    print("[INFO] Eyes lost — buffer cleared.")

            if not face_found:
                status = "⚠ No face — please face the camera"
            else:
                # Face found but eyes not open (closed or covered)
                l_ear = detector.left_ear
                r_ear = detector.right_ear
                status = (f"⚠ Eyes closed  "
                          f"(EAR L:{l_ear:.2f} R:{r_ear:.2f}  "
                          f"thresh:{detector.ear_threshold:.2f})")
        else:
            missing_frames = 0   # reset counter as soon as eyes are open again

            # ── CHROM signal extraction (only when eyes are open) ─────────
            left_roi  = get_eye_roi(frame, left_pts)
            right_roi = get_eye_roi(frame, right_pts)

            all_means = []
            for roi in [left_roi, right_roi]:
                if roi is not None and roi.size > 0:
                    all_means.append(roi.mean(axis=(0, 1)))

            if all_means:
                mean_bgr = np.mean(all_means, axis=0)
                pixel = np.array([[[mean_bgr[0], mean_bgr[1], mean_bgr[2]]]],
                                  dtype=np.uint8)
                chrom.update(pixel)

            l_ear = detector.left_ear
            r_ear = detector.right_ear
            status = (f"Eyes open  EAR L:{l_ear:.2f} R:{r_ear:.2f}  "
                      f"— collecting signal…")

        # Pull latest CHROM signal into deque
        raw_signal = chrom.get_signal()
        if len(raw_signal) > 0:
            signal_buf.clear()
            for v in raw_signal[-buf_size:]:
                signal_buf.append(float(v))

        # ── BPM estimation (once per second) ─────────────────────────────
        now = time.time()
        if (now - last_estimate_t) >= 1.0:
            last_estimate_t = now

            if not eyes_open:
                # No eyes → show --- immediately, do not keep stale BPM
                bpm, freq = 0.0, 0.0
            else:
                sig_arr     = np.array(signal_buf)
                min_samples = int(MIN_BPM_SECONDS * fps)

                if len(sig_arr) < min_samples:
                    status = f"Buffering… {len(sig_arr)}/{min_samples} samples"

                # ── Gate 2 — Signal quality check ─────────────────────────
                elif np.var(sig_arr) < MIN_SIGNAL_VARIANCE:
                    bpm, freq = 0.0, 0.0
                    status = "⚠ Signal too flat — stay still & ensure good lighting"

                else:
                    bpm = estimator.estimate(sig_arr)
                    freq      = estimator.freq
                    fft_freqs = estimator.fft_freqs
                    fft_power = estimator.fft_power
                    if bpm > 0:
                        status = f"BPM: {bpm:.1f}  |  {freq:.3f} Hz"

        # ── Buffer fill percentage ────────────────────────────────────────
        buffer_pct = len(signal_buf) / buf_size

        # ── Render dashboard ──────────────────────────────────────────────
        canvas = dash.render(
            cam_frame     = frame,
            left_eye_pts  = left_pts,
            right_eye_pts = right_pts,
            signal_buf    = list(signal_buf),
            fft_freqs     = fft_freqs,
            fft_power     = fft_power,
            bpm           = bpm,
            freq          = freq,
            buffer_pct    = buffer_pct,
            status        = status,
            eyes_open     = eyes_open,
            left_ear      = detector.left_ear,
            right_ear     = detector.right_ear,
        )

        cv2.imshow(WINDOW_NAME, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord('r'):
            _reset_all(chrom, signal_buf)
            bpm, freq = 0.0, 0.0
            fft_freqs = np.array([])
            fft_power = np.array([])
            print("[INFO] Buffer reset.")

    cam.release()
    detector.close()
    cv2.destroyAllWindows()
    print("[INFO] Exited cleanly.")


if __name__ == "__main__":
    main()
