# rPPG Eye-Based Heart Rate Monitor

Real-time contactless heart rate detection using a standard webcam.
**No Matplotlib / No Tkinter — pure OpenCV dashboard.**

---

## How It Works

```
Webcam → MediaPipe Eye Detection → Eye Landmarks (16 pts)
      → CHROM Signal Extraction → Bandpass Filter → FFT → BPM
```

---

## Quick Start

```bash
pip install -r requirements.txt
python app.py
```

### Controls
| Key | Action |
|-----|--------|
| `Q` or `ESC` | Quit |
| `R` | Reset signal buffer |

---

## Dashboard Layout (1280 × 720)

```
┌──────────────────────────┬──────────────────────────┐
│                          │  BPM: 72.4   0.207 Hz    │
│   Camera Feed            │  Buffer ████░░  60%       │
│   + Eye Landmarks        ├──────────────────────────┤
│   (green = left)         │  PULSE SIGNAL (CHROM)    │
│   (orange = right)       │  ~~~waveform~~~           │
│                          ├──────────────────────────┤
│                          │  FFT SPECTRUM            │
│                          │  |||peaks|||              │
└──────────────────────────┴──────────────────────────┘
```

---

## File Structure

| File | Purpose |
|------|---------|
| `app.py` | Main loop — ties everything together |
| `eye_detector.py` | MediaPipe Face Mesh, 16-pt eye landmarks |
| `webcam_capture.py` | OpenCV camera wrapper |
| `signal_processing.py` | Butterworth bandpass filter, detrend, normalise |
| `chrom_rppg.py` | CHROM chrominance-based pulse extraction |
| `bpm_estimator.py` | FFT peak detection → BPM |
| `dashboard.py` | Pure-OpenCV dashboard renderer |
| `requirements.txt` | Python dependencies |

---

## Current Status

| Feature | Status |
|---------|--------|
| Webcam | ✅ |
| Eye Detection (MediaPipe) | ✅ |
| Both Eyes + 16 Landmarks | ✅ |
| CHROM Signal Extraction | ✅ |
| Bandpass Filter | ✅ |
| FFT → BPM | ✅ |
| OpenCV Dashboard | ✅ |
output:
<img width="1919" height="1023" alt="Screenshot 2026-06-05 203557" src="https://github.com/user-attachments/assets/5ddffaea-724d-4ba2-8c3b-c8d8db445c1b" />

---


## Roadmap

```
CHROM (current)
  → CNN       — learn pulse waveform features
  → Fuzzy Logic   — handle noisy/uncertain BPM readings
  → Random Forest — ensemble for final BPM decision
```

---

## Tips for Best Results

- **Good lighting** — face should be evenly lit, avoid strong backlight
- **Keep still** — motion artifacts are the biggest source of error
- **Wait ~10 s** — the buffer needs time to fill before BPM is reliable
- Normal resting range shown: **45–180 BPM**

