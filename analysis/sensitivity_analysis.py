"""
Perturbation-based sensitivity analysis (manuscript Methods S.2.12):
identifies which wavenumbers the trained three-way classifier relies on, by
perturbing each input variable in isolation and measuring how far the
network's output moves from the correct one-hot target.

This is the computational core of this author's contribution: the same
analysis that independently flagged 1632 cm-1 (beta-sheet) and 1032 cm-1
(glycogen/nucleic acid) as the most influential bands (S.3.5), agreeing
with the curve-fitted result obtained by an entirely separate method
(Voigt fitting, S.2.8) despite sharing no processing step with it.

Usage:
    python sensitivity_analysis.py --data data/fingerprint_spectra.csv
    python sensitivity_analysis.py --data data/fingerprint_spectra.csv --networks 5   # smoke test
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from classify_three_way import FNN3Way, train_one_fold
from data_utils import grouped_folds, load_spectra, sample_zscore


def perturbation_sensitivity(model, X, y, device, n_classes=3, pct_range=0.5, n_steps=21):
    """
    For each wavenumber j: hold every other input fixed, scale column j by
    (1 + p) for p spanning +/-pct_range, and record the mean-squared
    deviation of the softmax output from the one-hot ideal target, averaged
    over the perturbation range. A wavenumber the network ignores barely
    moves the output under perturbation; one it relies on moves it a lot --
    the resulting curve is a per-wavenumber importance profile that makes
    no assumption about which bands "should" matter, unlike curve fitting
    (S.2.8), which starts from fixed, literature-assigned band positions.
    """
    Xt = torch.tensor(X, dtype=torch.float32, device=device)
    ideal = torch.eye(n_classes, device=device)[torch.tensor(y, device=device)]
    perturb = np.linspace(-pct_range, pct_range, n_steps)

    model.eval()
    sensitivity = np.empty(X.shape[1])
    with torch.no_grad():
        for j in range(X.shape[1]):
            mvals = []
            for p in perturb:
                Xp = Xt.clone()
                Xp[:, j] *= (1 + p)
                out = model(Xp).softmax(dim=1)
                mvals.append(((out - ideal) ** 2).mean().item())
            sensitivity[j] = np.mean(mvals)
    return sensitivity


def average_over_networks(X, y, groups, device, n_networks=50, lr=1e-3):
    """
    S.2.12: "averaged across 50 independently trained networks to stabilise
    the estimate" -- a single network's sensitivity profile depends on its
    particular init and batch order and can be noisy; 50 independent
    trainings (one per seed, each on its own 10-fold grouped split) turn
    that noise into a stable, reportable importance curve.
    """
    all_sens = []
    for seed in range(n_networks):
        tr_idx, te_idx = next(iter(grouped_folds(X, y, groups, n_splits=10, seed=seed)))
        model = train_one_fold(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], lr, device, seed=seed)
        all_sens.append(perturbation_sensitivity(model, X, y, device))
    all_sens = np.array(all_sens)
    return all_sens.mean(axis=0), all_sens.std(axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/fingerprint_spectra.csv")
    ap.add_argument("--networks", type=int, default=50)
    ap.add_argument("--out", default="sensitivity_profile.png")
    args = ap.parse_args()

    X, y, groups, wavenumbers = load_spectra(args.data)
    Xn = sample_zscore(X)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    mean_sens, std_sens = average_over_networks(Xn, y, groups, device, n_networks=args.networks)

    top = np.argsort(mean_sens)[::-1][:10]
    print("Top 10 most influential wavenumbers:")
    for i in top:
        print(f"  {wavenumbers[i]:.0f} cm-1   sensitivity = {mean_sens[i]:.5f} +/- {std_sens[i]:.5f}")
    print("\n(cf. manuscript S.3.5: 1632 cm-1 beta-sheet and 1032 cm-1 glycogen/nucleic-acid "
          "flagged as most influential)")

    plt.figure(figsize=(10, 4))
    plt.plot(wavenumbers, mean_sens, color="darkgreen")
    plt.fill_between(wavenumbers, mean_sens - std_sens, mean_sens + std_sens, color="darkgreen", alpha=0.2)
    plt.gca().invert_xaxis()
    plt.title("Neural-network sensitivity, averaged over 50 independently trained networks")
    plt.xlabel("Wavenumber (cm$^{-1}$)")
    plt.ylabel("Mean-squared output deviation")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=200)
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
