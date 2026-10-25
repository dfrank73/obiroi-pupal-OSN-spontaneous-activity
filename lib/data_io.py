"""Loaders for the tables and trace arrays in data/."""
import os
import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")


def load_cells():
    """One row per cell across both datasets: dataset, pupa (`ant`), plane,
    ROI name, bounding box, and `row_in_dataset` (its row index into that dataset's
    trace arrays)."""
    return pd.read_parquet(os.path.join(DATA_DIR, "cells.parquet"))


def load_roi_coordinates():
    """Long-format ROI polygon vertices: one row per (cell_id, point_idx)."""
    return pd.read_parquet(os.path.join(DATA_DIR, "roi_coordinates.parquet"))


def load_traces(dataset, channel="dff"):
    """Full trace array for one dataset ('P2P3' or 'P6P7'), shape
    (n_cells_in_dataset, n_frames). Row i corresponds to the cell whose
    `row_in_dataset` == i in cells.parquet (after filtering to that
    dataset). channel is 'dff' or 'brightness'."""
    fname = f"traces_{channel}_{dataset}.npy"
    return np.load(os.path.join(DATA_DIR, fname))


def get_cell_polygon(coords_df, cell_id):
    """Return the (n_points, 2) array of [x, y] vertices for one cell."""
    sub = coords_df[coords_df["cell_id"] == cell_id].sort_values("point_idx")
    return sub[["x", "y"]].to_numpy()


def animal_qc_traces_and_coords(dataset, animal_idx, cells_df, qc_df, coords_df, channel="dff"):
    """QC-passing cells for one animal: names, planes, trace array, and a
    list of per-cell (n_points, 2) polygon coordinate arrays."""
    merged = cells_df.merge(qc_df[["cell_id", "excluded"]], on="cell_id")
    subset = merged[(merged["dataset"] == dataset) & (merged["animal_idx"] == animal_idx)
                     & (~merged["excluded"])].sort_values("row_in_dataset")
    traces_full = load_traces(dataset, channel=channel)
    traces = traces_full[subset["row_in_dataset"].to_numpy()]
    planes = subset["plane"].to_numpy()
    names = subset["roi_name"].to_numpy()
    coords_list = [get_cell_polygon(coords_df, cid) for cid in subset["cell_id"]]
    return names, planes, traces, coords_list

