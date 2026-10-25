"""Pairwise-correlation panels (Fig 2G-H, Fig 3H-I) and the Fig S2 helpers.
Builds, per age group, the real pairwise Pearson-r distribution of every pupa
and two null distributions of 200 shuffle rounds each. Cell pairs on the same
or an adjacent imaging plane whose ROI outlines lie within
`BOUNDARY_DIST_THRESH_PX` of each other are excluded from all pairwise
statistics (`neighbor_pair_mask`)."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from data_io import animal_qc_traces_and_coords

BOUNDARY_DIST_THRESH_PX = 10   # ROI-outline distance (px) defining neighbouring pairs
MIN_SHIFT_S = 300              # minimum lag (s) of the circular-shift null
BINS = np.linspace(-1, 1, 200)
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42


def neighbor_pair_mask(planes, coords_list, boundary_thresh_px=BOUNDARY_DIST_THRESH_PX):
    """True for cell pairs on the same or an adjacent imaging plane whose ROI
    polygons come within `boundary_thresh_px` of each other (dendrites of
    neighbouring OSNs pass adjacent somata). The distance is measured between
    the polygon outlines. Pairs two or more planes apart are never flagged."""
    n = len(planes)
    artifact = np.zeros((n, n), dtype=bool)
    for i in range(n):
        for j in range(i + 1, n):
            if abs(int(planes[i]) - int(planes[j])) > 1:
                continue
            min_dist = cdist(coords_list[i], coords_list[j]).min()
            if min_dist <= boundary_thresh_px:
                artifact[i, j] = artifact[j, i] = True
    return artifact


def load_animal_group(dataset, cells, qc, coords):
    """Per-pupa QC-passing traces (raw dF/F), planes, ROI polygons and the
    upper-triangular keep-mask (artifact pairs removed) for one age group."""
    g = {"names": [], "traces": [], "planes": [], "coords": [], "keep": [], "pairs": []}
    for a in sorted(cells[cells["dataset"] == dataset]["animal_idx"].unique()):
        names, planes, traces, coords_list = animal_qc_traces_and_coords(dataset, a, cells, qc, coords)
        traces = np.asarray(traces, dtype=float)
        mask = neighbor_pair_mask(planes, coords_list)
        keep = ~mask
        np.fill_diagonal(keep, False)
        ant = cells[(cells["dataset"] == dataset) & (cells["animal_idx"] == a)]["ant"].iloc[0]
        n = len(traces)
        g["names"].append(ant); g["traces"].append(traces); g["planes"].append(planes)
        g["coords"].append(coords_list); g["keep"].append(np.triu(keep, k=1))
        g["pairs"].append({"ant": ant, "n_cells": n, "n_total_pairs": n * (n - 1) // 2,
                           "n_artifact_pairs": int(mask.sum() // 2),
                           "pct_excluded": 100 * (mask.sum() // 2) / (n * (n - 1) // 2)})
    g["pairs"] = pd.DataFrame(g["pairs"])
    return g


def _fake_group_keep_mask(planes, coords_list, origin):
    mask = neighbor_pair_mask(planes, coords_list, BOUNDARY_DIST_THRESH_PX)
    mask = mask & (origin[:, None] == origin[None, :])   # only same-origin pairs can be artifacts
    keep = ~mask
    np.fill_diagonal(keep, False)
    return np.triu(keep, k=1)


def pairwise_r_distributions(group, dataset, frame_interval_s, out_stem, n_shuffles=200,
                             seed_cross=7, seed_shift=0):
    """Real per-pupa pairwise-r histograms overlaid on (left) the within-pupa
    circular-shift null and (right) the across-pupae shuffle null, each pooled
    over `n_shuffles` rounds."""
    min_shift = int(MIN_SHIFT_S / frame_interval_s)
    traces_list, keep_masks, names = group["traces"], group["keep"], group["names"]
    sizes = [t.shape[0] for t in traces_list]
    real_vals = [np.corrcoef(t)[k] for t, k in zip(traces_list, keep_masks)]

    rng = np.random.default_rng(seed_cross)
    pool_traces = np.concatenate(traces_list, axis=0)
    pool_planes = np.concatenate(group["planes"], axis=0)
    pool_coords = [c for cl in group["coords"] for c in cl]
    pool_origin = np.concatenate([np.full(s, a) for a, s in enumerate(sizes)])
    cross_counts = np.zeros(len(BINS) - 1)
    for _ in range(n_shuffles):
        perm = rng.permutation(len(pool_traces)); start = 0
        for a, size in enumerate(sizes):
            idx = perm[start:start + size]; start += size
            keep_upper = _fake_group_keep_mask(pool_planes[idx], [pool_coords[i] for i in idx], pool_origin[idx])
            vals = np.corrcoef(pool_traces[idx])[keep_upper]
            cross_counts += np.histogram(vals, bins=BINS)[0]

    rng_shift = np.random.default_rng(seed_shift)
    circ_counts = np.zeros(len(BINS) - 1)
    for a, (tr, keep_upper) in enumerate(zip(traces_list, keep_masks)):
        n, T = tr.shape
        for _ in range(n_shuffles):
            shifted = np.empty_like(tr)
            for i in range(n):
                shifted[i] = np.roll(tr[i], rng_shift.integers(min_shift, T - min_shift))
            vals = np.corrcoef(shifted)[keep_upper]
            circ_counts += np.histogram(vals, bins=BINS)[0]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    for ax, counts, label in [(axes[0], circ_counts, "Within-pupae null\n(circular time-shift)"),
                              (axes[1], cross_counts, "Across-pupae null\n(OSNs shuffled)")]:
        ax.stairs(counts / counts.sum(), BINS, fill=True, color="gray", alpha=0.4, label=label)
        for name, vals, size in zip(names, real_vals, sizes):
            ax.hist(vals, bins=BINS, weights=np.full(len(vals), 1.0 / len(vals)),
                    histtype="step", linewidth=1.5, label=f"Observed ({name})")
        ax.set_xlabel("Pearson r"); ax.set_ylabel("Proportion")
        ax.legend(fontsize=8, frameon=False); ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight"); fig.savefig(f"{out_stem}.eps", bbox_inches="tight")

    return fig


def sliding_mean_r(traces_a, keep_upper, frame_interval_s, win_min=10, step_min=2):
    """Fig S2A: mean (signed) pairwise r in a sliding window."""
    win = int(win_min * 60 / frame_interval_s); step = int(step_min * 60 / frame_interval_s)
    T = traces_a.shape[1]
    centers, mean_rs = [], []
    for start in range(0, T - win + 1, step):
        seg = traces_a[:, start:start + win]
        ok = seg.std(axis=1) > 0
        vals = np.corrcoef(seg[ok])[keep_upper[np.ix_(ok, ok)]]
        centers.append((start + win / 2) * frame_interval_s / 60)
        mean_rs.append(vals.mean())
    return np.array(centers), np.array(mean_rs)
