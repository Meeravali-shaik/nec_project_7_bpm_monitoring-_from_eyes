"""
dashboard.py
Renders an all-in-one OpenCV dashboard — no Matplotlib, no Tkinter.

Layout (1280 x 720):
┌──────────────────────────┬──────────────────────────┐
│                          │   BPM / Freq panel       │
│   Camera feed (640x480)  │   Signal graph           │
│   + eye landmarks        │   FFT graph              │
└──────────────────────────┴──────────────────────────┘
"""

import cv2
import numpy as np

# ── colours (BGR) ──────────────────────────────────────────────────────────
C_BG        = (18,  18,  18)
C_PANEL     = (30,  30,  30)
C_ACCENT    = (0,   200, 255)   # orange-ish
C_GREEN     = (80,  220, 80)
C_RED       = (60,  60,  220)
C_TEXT      = (230, 230, 230)
C_GRID      = (55,  55,  55)
C_SIGNAL    = (80,  220, 80)
C_FFT       = (0,   180, 255)
C_BPM_OK    = (80,  220, 80)
C_BPM_WARN  = (0,   200, 255)
C_BPM_NONE  = (100, 100, 100)

FONT        = cv2.FONT_HERSHEY_SIMPLEX

# ── canvas dimensions ──────────────────────────────────────────────────────
CANVAS_W = 1280
CANVAS_H = 720
CAM_W    = 640
CAM_H    = 480
RIGHT_W  = CANVAS_W - CAM_W   # 640
TOP_H    = 120                 # BPM display height
GRAPH_H  = (CANVAS_H - TOP_H) // 2   # signal & FFT panels


def _make_canvas():
    c = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    c[:] = C_BG
    return c


def _draw_panel_bg(canvas, x, y, w, h, label=""):
    cv2.rectangle(canvas, (x, y), (x+w, y+h), C_PANEL, -1)
    cv2.rectangle(canvas, (x, y), (x+w, y+h), C_GRID,   1)
    if label:
        cv2.putText(canvas, label, (x+8, y+20), FONT, 0.5, C_ACCENT, 1, cv2.LINE_AA)


def _draw_grid_lines(canvas, x, y, w, h, n_h=4, n_v=6):
    for i in range(1, n_h):
        yy = y + i * h // n_h
        cv2.line(canvas, (x, yy), (x+w, yy), C_GRID, 1)
    for i in range(1, n_v):
        xx = x + i * w // n_v
        cv2.line(canvas, (xx, y), (xx, y+h), C_GRID, 1)


def _plot_line(canvas, data, x, y, w, h, colour, thickness=2):
    """Plot a 1-D array as a line graph inside rectangle (x,y,w,h)."""
    if len(data) < 2:
        return
    arr = np.array(data, dtype=np.float64)
    mn, mx = arr.min(), arr.max()
    rng = mx - mn
    if rng < 1e-9:
        rng = 1.0
    norm = (arr - mn) / rng    # [0, 1]

    pts = []
    for i, v in enumerate(norm):
        px = x + int(i * w / (len(norm) - 1))
        py = y + h - int(v * (h - 4)) - 2
        pts.append((px, py))

    for i in range(len(pts) - 1):
        cv2.line(canvas, pts[i], pts[i+1], colour, thickness, cv2.LINE_AA)


class Dashboard:
    """
    Call .render(frame, left_pts, right_pts, signal_buf, fft_freqs,
                  fft_power, bpm, freq, status)
    Returns the composed BGR canvas.
    """

    def __init__(self):
        self._canvas = _make_canvas()

    def render(self, cam_frame, left_eye_pts, right_eye_pts,
               signal_buf, fft_freqs, fft_power,
               bpm: float, freq: float,
               buffer_pct: float = 0.0,
               status: str = "Detecting…",
               eyes_open: bool = True,
               left_ear: float = 0.0,
               right_ear: float = 0.0):

        canvas = _make_canvas()

        # ── 1. Camera feed (left half) ─────────────────────────────────────
        if cam_frame is not None:
            h, w = cam_frame.shape[:2]
            # Resize to fit panel while keeping aspect ratio
            scale = min(CAM_W / w, CAM_H / h)
            nw, nh = int(w * scale), int(h * scale)
            resized = cv2.resize(cam_frame, (nw, nh))
            ox = (CAM_W - nw) // 2
            oy = (CAM_H - nh) // 2
            canvas[oy:oy+nh, ox:ox+nw] = resized

            # Eye landmarks (scale coordinates to match resized frame)
            def _draw_eye(pts, colour):
                if pts is None:
                    return
                for pt in pts:
                    px = int(pt[0] * scale) + ox
                    py = int(pt[1] * scale) + oy
                    cv2.circle(canvas, (px, py), 3, colour, -1, cv2.LINE_AA)

            _draw_eye(left_eye_pts,  (80,  220, 80))
            _draw_eye(right_eye_pts, (0,   200, 255))

        # Camera panel border
        cv2.rectangle(canvas, (0, 0), (CAM_W-1, CAM_H-1), C_GRID, 1)
        cv2.putText(canvas, "CAMERA FEED", (8, 20),
                    FONT, 0.5, C_ACCENT, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"Left eye: {16 if left_eye_pts is not None else 0} pts   "
                             f"Right eye: {16 if right_eye_pts is not None else 0} pts",
                    (8, CAM_H - 28), FONT, 0.42, C_TEXT, 1, cv2.LINE_AA)

        # EAR values row
        ear_colour_l = C_GREEN if left_ear  >= 0.20 else C_RED
        ear_colour_r = C_GREEN if right_ear >= 0.20 else C_RED
        cv2.putText(canvas, "EAR  L:", (8, CAM_H - 10), FONT, 0.42, C_TEXT, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{left_ear:.2f}", (70, CAM_H - 10), FONT, 0.42, ear_colour_l, 1, cv2.LINE_AA)
        cv2.putText(canvas, "R:", (115, CAM_H - 10), FONT, 0.42, C_TEXT, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"{right_ear:.2f}", (135, CAM_H - 10), FONT, 0.42, ear_colour_r, 1, cv2.LINE_AA)

        # ── Warning overlay when eyes are closed or face is covered ───────
        if not eyes_open:
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0, 0), (CAM_W, CAM_H), (0, 0, 180), -1)
            cv2.addWeighted(overlay, 0.25, canvas, 0.75, 0, canvas)

            warn_lines = ["SIGNAL PAUSED", "Eyes closed or face covered"]
            for i, line in enumerate(warn_lines):
                fs    = 0.9  if i == 0 else 0.55
                thick = 2    if i == 0 else 1
                col   = (60, 60, 255) if i == 0 else C_TEXT
                (tw, th), _ = cv2.getTextSize(line, FONT, fs, thick)
                tx = (CAM_W - tw) // 2
                ty = CAM_H // 2 - 20 + i * 40
                cv2.putText(canvas, line, (tx, ty), FONT, fs, col, thick, cv2.LINE_AA)

        # ── 2. BPM / Freq panel (top-right) ────────────────────────────────
        bx, by = CAM_W, 0
        _draw_panel_bg(canvas, bx, by, RIGHT_W, TOP_H)
        cv2.putText(canvas, "HEART RATE MONITOR", (bx+8, by+20),
                    FONT, 0.55, C_ACCENT, 1, cv2.LINE_AA)

        # BPM number
        if bpm > 0:
            bpm_colour = C_BPM_OK if 50 <= bpm <= 120 else C_BPM_WARN
            bpm_str = f"{bpm:.1f}"
        else:
            bpm_colour = C_BPM_NONE
            bpm_str = "---"

        cv2.putText(canvas, bpm_str, (bx + 20, by + 90),
                    FONT, 2.4, bpm_colour, 3, cv2.LINE_AA)
        cv2.putText(canvas, "BPM", (bx + 220, by + 90),
                    FONT, 1.0, bpm_colour, 2, cv2.LINE_AA)

        # Freq
        freq_str = f"{freq:.3f} Hz" if freq > 0 else "--- Hz"
        cv2.putText(canvas, freq_str, (bx + 340, by + 65),
                    FONT, 0.7, C_TEXT, 1, cv2.LINE_AA)

        # Buffer progress bar
        bar_x, bar_y, bar_w, bar_h = bx + 340, by + 75, 270, 12
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), C_GRID, -1)
        filled = int(bar_w * min(buffer_pct, 1.0))
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x+filled, bar_y+bar_h), C_ACCENT, -1)
        cv2.putText(canvas, f"Buffer {int(buffer_pct*100)}%",
                    (bar_x, bar_y - 4), FONT, 0.38, C_TEXT, 1, cv2.LINE_AA)

        # Status
        cv2.putText(canvas, status, (bx + 340, by + 108),
                    FONT, 0.45, C_TEXT, 1, cv2.LINE_AA)

        # ── 3. Signal graph (middle-right) ─────────────────────────────────
        sx, sy = CAM_W, TOP_H
        sw, sh = RIGHT_W, GRAPH_H
        _draw_panel_bg(canvas, sx, sy, sw, sh, "PULSE SIGNAL (CHROM)")
        _draw_grid_lines(canvas, sx, sy, sw, sh)

        if len(signal_buf) > 10:
            _plot_line(canvas, signal_buf, sx+2, sy+25, sw-4, sh-30, C_SIGNAL)

            # Latest value label
            cv2.putText(canvas, f"val: {signal_buf[-1]:.4f}",
                        (sx + sw - 140, sy + sh - 8),
                        FONT, 0.38, C_SIGNAL, 1, cv2.LINE_AA)

        # ── 4. FFT graph (bottom-right) ────────────────────────────────────
        fx, fy = CAM_W, TOP_H + GRAPH_H
        fw, fh = RIGHT_W, CANVAS_H - TOP_H - GRAPH_H
        _draw_panel_bg(canvas, fx, fy, fw, fh, "FFT SPECTRUM")
        _draw_grid_lines(canvas, fx, fy, fw, fh)

        if fft_freqs is not None and len(fft_freqs) > 5:
            # Show only 0–4 Hz range
            mask = fft_freqs <= 4.0
            freqs_plot = fft_freqs[mask]
            power_plot = fft_power[mask]
            if len(power_plot) > 2:
                _plot_line(canvas, power_plot, fx+2, fy+25, fw-4, fh-30, C_FFT)

                # Mark peak (BPM band)
                bpm_mask = (freqs_plot >= 0.75) & (freqs_plot <= 3.0)
                if np.any(bpm_mask):
                    pk = np.argmax(power_plot[bpm_mask])
                    pk_freq = freqs_plot[bpm_mask][pk]
                    # Map frequency to x-pixel
                    pk_x = fx + 2 + int(np.searchsorted(freqs_plot, pk_freq) * (fw-4) / len(freqs_plot))
                    cv2.line(canvas, (pk_x, fy+25), (pk_x, fy+fh-10),
                             (0, 80, 255), 1, cv2.LINE_AA)
                    cv2.putText(canvas, f"{pk_freq:.2f}Hz",
                                (pk_x+4, fy+40), FONT, 0.4, (0, 80, 255), 1, cv2.LINE_AA)

            # X-axis freq labels
            for f_label in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5]:
                lx = fx + 2 + int(f_label / 4.0 * (fw-4))
                cv2.putText(canvas, f"{f_label:.1f}", (lx-10, fy+fh-4),
                            FONT, 0.32, (115, 115, 115),
                            1, cv2.LINE_AA)

        # ── 5. Bottom status bar (below camera feed) ───────────────────────
        # Fill remaining camera column below 480
        if CAM_H < CANVAS_H:
            extra_y = CAM_H
            extra_h = CANVAS_H - CAM_H
            _draw_panel_bg(canvas, 0, extra_y, CAM_W, extra_h)
            hint = "Press Q to quit  |  rPPG eye-based heart rate detection"
            cv2.putText(canvas, hint, (10, extra_y + extra_h//2 + 6),
                        FONT, 0.48, C_TEXT, 1, cv2.LINE_AA)

            # Legend
            cv2.circle(canvas, (10 + 10, extra_y + extra_h//2 + 30), 5, (80, 220, 80), -1)
            cv2.putText(canvas, "Left eye", (30, extra_y + extra_h//2 + 35),
                        FONT, 0.4, (80, 220, 80), 1, cv2.LINE_AA)
            cv2.circle(canvas, (120, extra_y + extra_h//2 + 30), 5, (0, 200, 255), -1)
            cv2.putText(canvas, "Right eye", (135, extra_y + extra_h//2 + 35),
                        FONT, 0.4, (0, 200, 255), 1, cv2.LINE_AA)

        return canvas