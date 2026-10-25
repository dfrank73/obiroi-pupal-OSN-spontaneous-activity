"""Per-cell features of a dF/F trace.

`extract_features` returns the quantities used in the figures:

| feature | meaning |
|---|---|
| `median_duration_s` | median duration of the cell's active episodes (0 if none) |
| `duty_cycle` | fraction of the recording spent inside active episodes |
| `n_subpeaks` | number of peaks |
| `median_interpeak_interval_s` | median time between successive peaks (0 if fewer than two peaks) |
| `tau_s` | autocorrelation decay time: first lag at which the trace's autocorrelation drops below 1/e |
| `mean_dff` | mean of the trace over the recording |
| `max_dff` | maximum of the trace |

Episodes and peaks come from `event_detection.detect_events`. `normalize_trace`
rescales a trace to its own [0, 1] range; timing features are identical on the
raw and normalized trace, amplitude features are expressed as a fraction of the
cell's own dynamic range.
"""
import numpy as np
from scipy.signal import fftconvolve

from event_detection import detect_events


def normalize_trace(trace):
    """Per-cell min-max rescale to [0, 1] using the trace's own min/max."""
    trace = np.asarray(trace, dtype=float)
    mn, mx = trace.min(), trace.max()
    if mx - mn < 1e-9:
        return np.zeros_like(trace)
    return (trace - mn) / (mx - mn)


def autocorr_tau(trace, frame_interval_s):
    x = trace - trace.mean()
    n = len(x)
    result = fftconvolve(x, x[::-1], mode="full")
    result = result[n - 1:]
    if result[0] == 0:
        return n * frame_interval_s
    result = result / result[0]
    below = np.where(result < (1.0 / np.e))[0]
    tau_frames = below[0] if len(below) > 0 else n
    return tau_frames * frame_interval_s


def extract_features(trace, frame_interval_s):
    trace = np.asarray(trace, dtype=float)
    events, peak_frames = detect_events(trace, frame_interval_s)
    durations = np.array([e["duration_s"] for e in events])
    peak_t = np.asarray(peak_frames, dtype=float) * frame_interval_s
    return {
        "median_duration_s": float(np.median(durations)) if len(durations) else 0.0,
        "duty_cycle": float(durations.sum() / (len(trace) * frame_interval_s)),
        "n_subpeaks": int(len(peak_t)),
        "median_interpeak_interval_s": float(np.median(np.diff(peak_t))) if len(peak_t) > 1 else 0.0,
        "tau_s": autocorr_tau(trace, frame_interval_s),
        "mean_dff": float(trace.mean()),
        "max_dff": float(trace.max()),
    }
