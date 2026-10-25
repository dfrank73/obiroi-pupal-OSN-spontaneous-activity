"""Activity criterion applied to every segmented cell before any analysis.

A cell is included if its raw (unsmoothed) dF/F trace reaches at least
`ACTIVITY_THRESH_DFF` (a 100% increase over baseline) for at least
`ACTIVITY_MIN_CONSEC_FRAMES` consecutive frames (~2.4 s) at any point in the
recording. Cells that never do are excluded from all further analyses.
"""
import numpy as np

ACTIVITY_THRESH_DFF = 1.0
ACTIVITY_MIN_CONSEC_FRAMES = 3


def max_consec_above(trace, thresh):
    """Longest run of consecutive frames at or above `thresh`."""
    above = trace >= thresh
    max_run = 0
    run = 0
    for v in above:
        run = run + 1 if v else 0
        max_run = max(max_run, run)
    return max_run


def compute_cell_qc(trace):
    """Apply the activity criterion to one raw dF/F trace."""
    trace = np.asarray(trace, dtype=float)
    max_run = max_consec_above(trace, ACTIVITY_THRESH_DFF)
    is_active = max_run >= ACTIVITY_MIN_CONSEC_FRAMES
    return {
        "max_consec_above_1dff": max_run,
        "excluded": not is_active,
        "exclusion_reason": "none" if is_active else "inactive",
    }
