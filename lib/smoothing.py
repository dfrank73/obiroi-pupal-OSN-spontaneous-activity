"""Zero-phase Butterworth low-pass filters applied to dF/F traces before
event detection (second order, applied forward and backward with filtfilt;
fs = 1.25 Hz, the frame rate).

- `butterworth_smooth` (0.02 Hz cutoff): the trace on
  which active episodes are detected. Episodes are sustained, minutes-long
  states; this cutoff removes noise on faster timescales without fragmenting
  them.
- `butterworth_smooth_peaks` (0.05 Hz cutoff): the trace on which peaks are
  detected. Bursty cells show ~30-60 s oscillations that the 0.02 Hz filter
  merges into single humps; 0.05 Hz resolves them.

Noise sigma (see event_detection.py) is always estimated from the raw,
unsmoothed trace, whichever filter is used for the signal.
"""
import numpy as np
from scipy.signal import butter, filtfilt

FS = 1.25  # frame rate
FC = 0.02  # Hz cutoff (episode detection)
FC_PEAKS = 0.05  # Hz cutoff (peak detection)
ORDER = 2

_b, _a = butter(ORDER, FC, btype="low", fs=FS)
_bp, _ap = butter(ORDER, FC_PEAKS, btype="low", fs=FS)


def butterworth_smooth(trace):
    trace = np.asarray(trace, dtype=float)
    return filtfilt(_b, _a, trace)


def butterworth_smooth_peaks(trace):
    trace = np.asarray(trace, dtype=float)
    return filtfilt(_bp, _ap, trace)
