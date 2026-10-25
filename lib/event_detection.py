"""Active-episode and peak detection for spontaneous calcium activity traces.

- `detect_events`: hysteresis episode detection on the 0.02 Hz-filtered trace
  -- sustained excursions from baseline, the unit for episode duration and
  the fraction of time spent active.
- `detect_peaks`: peak detection on the 0.05 Hz-filtered trace -- discrete
  transients, the unit for peak counts and inter-peak intervals. Peaks are
  not confined to episodes.

Quantities shared by both, computed from the raw (unfiltered) trace:
- Baseline = the 8th percentile of the trace.
- Noise sigma = MAD of frame-to-frame differences / 0.6745 / sqrt(2).
- Dynamic range = maximum of the filtered trace above baseline.

Episode rules (all thresholds relative to baseline):

  Start: the filtered trace exceeds baseline + max(high_k*sigma,
         rel_min_amp*range, rel_entry*(previous episode's peak above baseline)).
    - high_k*sigma (4 sigma)
    - rel_min_amp (10% of range)
    - rel_entry (30% of the previous episode's peak, applied only until the
      trace next returns below baseline + low_k*sigma)
  Onset: walked back from the trigger to the start of the uninterrupted
         stretch above baseline + max(low_k*sigma, rel_min_amp*range), i.e.
         the moment the trace departed from baseline.
  End:   the trace falls below baseline + max(low_k*sigma,
         rel_release*(this episode's running peak above baseline)).

Peak rules: scipy.signal.find_peaks on the 0.05 Hz-filtered trace with
prominence >= peak_prominence_k*sigma (3 sigma, raw-derived), prominence
evaluated within a window of +/-10 min around each candidate.
"""
import numpy as np
from scipy.signal import find_peaks
from smoothing import butterworth_smooth, butterworth_smooth_peaks


def estimate_baseline_and_noise(trace, baseline_pct=8.0):
    """Baseline and noise sigma of a raw (unfiltered) dF/F trace (see module docstring)."""
    baseline = np.percentile(trace, baseline_pct)
    d1 = np.diff(trace)
    sigma = np.median(np.abs(d1 - np.median(d1))) / 0.6745 / np.sqrt(2)
    return baseline, sigma


def detect_peaks(trace, frame_interval_s, baseline_pct=8.0,
                  peak_prominence_k=3.0, peak_wlen_s=1200.0):
    """Peak detection (see module docstring). Returns the frame index of every peak."""
    trace = np.asarray(trace, dtype=float)
    baseline, sigma = estimate_baseline_and_noise(trace, baseline_pct)
    work = butterworth_smooth_peaks(trace)
    locs, _ = find_peaks(
        work,
        prominence=peak_prominence_k * sigma,
        wlen=max(3, int(peak_wlen_s / frame_interval_s)),
    )
    return locs


def detect_events(trace, frame_interval_s, baseline_pct=8.0,
                   high_k=4.0, low_k=1.0,
                   rel_release=0.2, rel_entry=0.3, rel_min_amp=0.10,
                   smooth=True):
    """Detect active episodes via hysteresis (see module docstring for the
    rules), plus peaks via detect_peaks.

    Returns
    -------
    events : list of dicts with onset_frame, offset_frame (inclusive) and
             duration_s, one per episode
    peak_frames : 1D int array of peak frame indices
    """
    trace = np.asarray(trace, dtype=float)
    n = len(trace)

    # noise sigma comes from the raw trace (see module docstring)
    baseline, sigma = estimate_baseline_and_noise(trace, baseline_pct)

    work = butterworth_smooth(trace) if smooth else trace

    low_floor = baseline + low_k * sigma
    amp_floor = rel_min_amp * (work.max() - baseline)
    onset_floor = baseline + max(low_k * sigma, amp_floor)

    events = []
    last_end = -1          # inclusive end frame of the previous episode
    prev_peak_above = 0.0  # previous episode's peak above baseline; reset
                           # once the trace returns below the low floor
    i = 0
    while i < n:
        if work[i] <= low_floor:
            prev_peak_above = 0.0
            i += 1
            continue
        entry = baseline + max(high_k * sigma, rel_entry * prev_peak_above, amp_floor)
        if work[i] > entry:
            start = i
            while start > last_end + 1 and work[start - 1] > onset_floor:
                start -= 1
            runmax = work[i]
            j = i
            while j < n:
                runmax = max(runmax, work[j])
                release = baseline + max(low_k * sigma, rel_release * (runmax - baseline))
                if work[j] < release:
                    break
                j += 1
            end = j  # exclusive
            events.append({
                "onset_frame": start,
                "offset_frame": end - 1,
                "duration_s": (end - start) * frame_interval_s,
            })
            last_end = end - 1
            prev_peak_above = max(prev_peak_above, runmax - baseline)
            i = j + 1
        else:
            i += 1

    peak_frames = detect_peaks(trace, frame_interval_s, baseline_pct)
    return events, peak_frames
