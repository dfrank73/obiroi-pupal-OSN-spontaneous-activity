"""Panel-generating functions shared by the Figure 2 (early pupae) and
Figure 3 (intermediate pupae) notebooks, which run the same analyses on the
two age groups. Each function draws one panel (or one panel pair), saves it,
and returns the matplotlib figure."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats as sp_stats

from features import normalize_trace

PLANE_DEPTH_UM = {1: 0, 2: 25, 3: 50, 4: 75, 5: 100}
AGE_COLORS = {"P2P3": "#2a78d6", "P6P7": "#eb6834"}
PLOT_DECIMATION = 5   # plot every Nth frame of a trace: keeps vector PDFs light
plt.rcParams["pdf.fonttype"] = 42   # TrueType fonts -> text stays editable in Illustrator
plt.rcParams["ps.fonttype"] = 42


def _save(fig, out_stem):
    fig.savefig(f"{out_stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{out_stem}.eps", bbox_inches="tight")


def cell_counts_by_plane(cells, dataset, out_stem):
    """Fig 1H / Fig 3C: segmented somata per imaging plane, one panel per pupa."""
    sub = cells[cells["dataset"] == dataset]
    animal_idxs = sorted(sub["animal_idx"].unique())
    counts = sub.groupby(["animal_idx", "plane"]).size().unstack(fill_value=0)
    for p in PLANE_DEPTH_UM:
        if p not in counts.columns:
            counts[p] = 0
    counts = counts[sorted(PLANE_DEPTH_UM)]
    xmax = counts.values.max() * 1.1
    fig, axes = plt.subplots(3, 2, figsize=(9, 9))
    axes = axes.flatten()
    for ax, a in zip(axes, animal_idxs):
        ant = sub[sub["animal_idx"] == a]["ant"].iloc[0].replace("#", "")
        row = counts.loc[a]
        y = np.arange(len(row))
        ax.barh(y, row.values, height=0.7, color="#e8e8ec", edgecolor="black", linewidth=1.0)
        ax.set_yticks(y)
        ax.set_yticklabels([f"{PLANE_DEPTH_UM[p]} µm" for p in row.index])
        ax.invert_yaxis()
        ax.set_xlim(0, xmax)
        ax.set_title(f"{ant}, {int(row.sum())} Cells", fontsize=11)
        ax.set_xlabel("# Cells")
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[len(animal_idxs):]:
        ax.axis("off")
    plt.tight_layout()
    _save(fig, out_stem)
    counts.index = [sub[sub["animal_idx"] == a]["ant"].iloc[0] for a in counts.index]
    return fig, counts


def max_dff_scatter_with_qc_pie(qc_df, dataset, activity_thresh, out_stem, ylabel="Max ΔF/F"):
    """Fig 2A / Fig 3E: every cell's maximum dF/F, sorted, with the activity
    threshold; inset-style pie of the fraction passing the activity criterion."""
    sub = qc_df[qc_df["dataset"] == dataset].sort_values("max_dff", ascending=False)
    n_in, n_ex = int((~sub["excluded"]).sum()), int(sub["excluded"].sum())
    fig, (ax, axp) = plt.subplots(1, 2, figsize=(10, 4.5), gridspec_kw={"width_ratios": [2.2, 1]})
    ax.scatter(np.arange(1, len(sub) + 1), sub["max_dff"], c="steelblue", s=8, alpha=0.6, edgecolors="none")
    ax.axhline(activity_thresh, color="black", ls="--", lw=1, label=f"dF/F = {activity_thresh:g} (activity threshold)")
    ax.set_xlabel("Cell Number")
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    axp.pie([n_in, n_ex], colors=["white", "black"], startangle=90, counterclock=False,
            wedgeprops={"edgecolor": "black", "linewidth": 1})
    axp.set_title(f"Max\nΔF/F > {activity_thresh:g}", fontsize=10)
    axp.text(0, -1.25, f"{100 * n_in / len(sub):.2f}%", ha="center", fontsize=10)
    plt.tight_layout()
    _save(fig, out_stem)
    return fig, {"n_included": n_in, "n_excluded": n_ex, "pct_active": 100 * n_in / len(sub)}


def n_subpeaks_vs_max_dff(feat_df, dataset, out_stem):
    """Fig 2D: number of peaks vs maximum dF/F per active cell, with a linear fit."""
    sub = feat_df[feat_df["dataset"] == dataset]
    x, y = sub["n_subpeaks"].values.astype(float), sub["max_dff"].values
    fit = sp_stats.linregress(x, y)
    xs = np.linspace(x.min(), x.max(), 200)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.scatter(x, y, s=10, alpha=0.4, color=AGE_COLORS[dataset], zorder=1)
    ax.plot(xs, fit.slope * xs + fit.intercept, color="black", lw=2, zorder=2)
    p_txt = "p < 0.001" if fit.pvalue < 0.001 else f"p = {fit.pvalue:.3f}"
    ax.text(0.97, 0.95, f"r = {fit.rvalue:.2f}, {p_txt}", transform=ax.transAxes, ha="right", va="top", fontsize=11)
    ax.set_xlabel("Number of ΔF/F Peaks")
    ax.set_ylabel("Max ΔF/F")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _save(fig, out_stem)
    return fig, fit


def check_example_cells_distinct(feat_df, example_ids, min_dx=0.01, min_dy=1):
    """Refuse highlighted example cells whose markers would coincide in the
    activity-space scatter (same peak count and nearly identical mean dF/F)."""
    pts = [(cid, feat_df.loc[feat_df.cell_id == cid, ["mean_dff_normalized", "n_subpeaks"]].iloc[0].tolist())
           for ids in example_ids.values() for cid in ids]
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            if abs(pts[i][1][0] - pts[j][1][0]) < min_dx and abs(pts[i][1][1] - pts[j][1][1]) < min_dy:
                raise ValueError(f"example cells {pts[i][0]} and {pts[j][0]} overlap in the scatter")


def activity_space_examples(feat_df, cells, traces, dataset, example_ids, frame_interval_s, out_stem):
    """Fig 2E-F / Fig 3F-G: amplitude-normalized mean dF/F vs number of peaks
    (one point per cell), with hand-picked example cells highlighted and their
    normalized traces shown alongside. `example_ids` maps colour -> [cell_id, ...]."""
    check_example_cells_distinct(feat_df, example_ids)
    row_lookup = dict(zip(cells["cell_id"], cells["row_in_dataset"]))
    scale_bar_frames = int(300 / frame_interval_s)
    dset_norm = feat_df[(feat_df["dataset"] == dataset) & (feat_df["duty_cycle"] > 0)].reset_index(drop=True)
    example_groups = [(color, [dset_norm[dset_norm["cell_id"] == cid].iloc[0] for cid in ids])
                      for color, ids in example_ids.items()]

    fig = plt.figure(figsize=(16, 13))
    grid = fig.add_gridspec(1, 2, width_ratios=[1, 1.3], wspace=0.25)
    ax_scatter = fig.add_subplot(grid[0])
    ax_scatter.scatter(dset_norm["mean_dff_normalized"], dset_norm["n_subpeaks"], s=6, color="#333333", alpha=0.5, zorder=2)
    for color, rows_ in example_groups:
        for row in rows_:
            ax_scatter.scatter(row["mean_dff_normalized"], row["n_subpeaks"], s=90, color=color, zorder=3,
                               edgecolor="black", linewidth=0.5)
    ax_scatter.set_xlabel("Amplitude-Normalized Mean ΔF/F")
    ax_scatter.set_ylabel("Number of ΔF/F Peaks")
    ax_scatter.spines[["top", "right"]].set_visible(False)

    all_examples = [(color, row) for color, rows_ in example_groups for row in rows_]
    grid_traces = grid[1].subgridspec(len(all_examples), 1, hspace=0.15)
    for i, (color, row) in enumerate(all_examples):
        ax = fig.add_subplot(grid_traces[i])
        trace = normalize_trace(traces[int(row_lookup[row["cell_id"]])])
        t_frames = np.arange(len(trace))[::PLOT_DECIMATION]
        ax.plot(t_frames, trace[::PLOT_DECIMATION], color=color, linewidth=0.8)
        ax.axhline(0, color="black", linewidth=1)
        ax.set_xlim(0, len(trace) - 1)
        ax.set_ylim(-0.05, 1.05)
        ax.set_yticks([0, 1])
        ax.set_xticks([])
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        if i == 0:
            ax.plot([200, 200 + scale_bar_frames], [1.15, 1.15], color="black", lw=2, clip_on=False)
            ax.text(200 + scale_bar_frames / 2, 1.28, "5m", ha="center", fontsize=11, clip_on=False)
        if i == len(all_examples) // 2:
            ax.set_ylabel("Amplitude-Normalized ΔF/F", fontsize=12)
    _save(fig, out_stem)
    return fig
