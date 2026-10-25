# Spontaneous neural activity in the developing ant olfactory system

Analysis code for Frank & Kronauer, *Spontaneous neural activity in the developing
ant olfactory system*.

## Data availability

Data will be made publicly available prior to publication and are available from
the authors on request. Every notebook is provided with its executed outputs, so
all figure panels and numbers can be inspected.

## Layout

```
lib/                 analysis library
  qc.py                activity criterion (ΔF/F ≥ 1 for ≥ 3 consecutive frames)
  smoothing.py         zero-phase Butterworth filters (0.02 Hz episodes, 0.05 Hz peaks)
  event_detection.py   active-episode and peak detection
  features.py          per-cell features; amplitude normalization of traces
  figure_panels.py     panel functions shared by the Figure 2 and Figure 3 notebooks
  correlation_panels.py  ROI-proximity pair exclusion; pairwise-r distributions vs both nulls; sliding-window correlation
  data_io.py           loaders for data/
notebooks/
  00_build_dataset.ipynb          run first: activity criterion, episode/peak detection, features
  Fig1_cell_counts.ipynb
  Fig2_early_pupae.ipynb
  Fig3_intermediate_pupae.ipynb
  Fig4_features_by_age.ipynb
  Fig5_late_pupae.ipynb
```

## Running

```
conda env create -f environment.yml
conda activate obiroi-osn-activity
cd notebooks
jupyter nbconvert --to notebook --execute --inplace 00_build_dataset.ipynb
jupyter nbconvert --to notebook --execute --inplace Fig2_early_pupae.ipynb   # etc.
```

`00_build_dataset.ipynb` must run before any figure notebook; it writes
`data/cell_qc.parquet`, `data/episodes.parquet` and `data/cell_features.parquet`. The figure notebooks are independent of
one another. The two correlation notebooks (Fig 2, Fig 3) each build two null
distributions of 200 shuffle rounds and take several minutes.

## Conventions

- ΔF/F = (F − F₀)/F₀ with F₀ the 12th percentile of the ROI's raw fluorescence trace.
- Early (P2/P3) and intermediate (P6/P7) pupae: volumetric imaging, 0.806 s per volume, 2 h.
  Late (P16/P17) pupae: single plane, 1.00 s per frame, 60 min.
- Pupa numbering in the figures: Ant #1–5 early, Ant #6–10 intermediate, Ant #11–14 late.
