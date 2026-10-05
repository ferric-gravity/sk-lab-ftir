"""
Shared data-loading and participant-grouped splitting utilities for the
final analysis pipeline (manuscript Methods, S.2.9-S.2.13).

Expected input format
----------------------
A single CSV, e.g. data/fingerprint_spectra.csv, with one row per spectrum:

    participant_id, diagnosis, <wavenumber_1>, <wavenumber_2>, ..., <wavenumber_462>

- participant_id   : unique per tissue donor (NOT per spectrum -- a participant
                      contributes up to 5 ATR replicates, S.2.3, and all of
                      them must stay on the same side of any train/test split).
- diagnosis        : one of "control", "benign", "metastatic".
- wavenumber columns: absorbance values across the 1800-900 cm-1 fingerprint
                      region at 2 cm-1 resolution (462 variables, S.2.3).

This file is intentionally not committed -- patient-level spectra are not
public data. Place your own export at analysis/data/fingerprint_spectra.csv
(gitignored) or pass --data to any of the scripts in this folder.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

DIAGNOSIS_ORDER = ["control", "benign", "metastatic"]
_LABEL_MAP = {d: i for i, d in enumerate(DIAGNOSIS_ORDER)}


def load_spectra(path):
    df = pd.read_csv(path)
    meta_cols = {"participant_id", "diagnosis"}
    wn_cols = [c for c in df.columns if c not in meta_cols]
    wavenumbers = np.array([float(c) for c in wn_cols])

    unknown = set(df["diagnosis"].unique()) - set(DIAGNOSIS_ORDER)
    if unknown:
        raise ValueError(f"Unrecognised diagnosis label(s): {unknown}. Expected one of {DIAGNOSIS_ORDER}.")

    X = df[wn_cols].values.astype(float)
    y = df["diagnosis"].map(_LABEL_MAP).values.astype(int)
    groups = df["participant_id"].values
    return X, y, groups, wavenumbers


def sample_zscore(X):
    """Per-spectrum z-score, matching the normalisation used upstream of the
    original exploratory script. Normalisation choice does not affect the
    two ratio measures reported elsewhere in the manuscript (S.2.6) but does
    affect what a neural network trained directly on absorbance sees."""
    return (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)


def grouped_folds(X, y, groups, n_splits=10, seed=0):
    """
    StratifiedGroupKFold keeps every spectrum from one participant in the
    same fold, while still balancing diagnostic groups across folds. This
    is the computational form of the "verified participant-grouped
    partitioning" described in S.2.11, and it is the single most important
    safeguard in this pipeline: without it, replicate spectra from the same
    tissue block can land on opposite sides of a train/test split, letting
    a model memorise participant-specific noise rather than diagnosis --
    exactly the leakage mode the manuscript's Introduction cites in a
    comparable study (ref. 20) as inflating accuracy past 99%.
    """
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(X, y, groups))
