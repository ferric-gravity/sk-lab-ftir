"""
Chemometric comparators (manuscript Methods S.2.13): PLS-DA and an
RBF-kernel SVM, fitted to the same spectra and the same participant-grouped
folds as the neural network (classify_three_way.py), so the
89.4% / 81.2% / 85.6% comparison reported in S.3.5 reflects a genuine
difference in classifier, not a difference in evaluation protocol.

Usage:
    python chemometric_comparators.py --data data/fingerprint_spectra.csv
"""

import argparse

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import label_binarize
from sklearn.svm import SVC

from data_utils import DIAGNOSIS_ORDER, grouped_folds, load_spectra, sample_zscore


def plsda_predict(Xtr, ytr, Xte, n_components=4):
    """PLS-DA = PLS regression against one-hot class labels, then argmax of
    the predicted scores. 4 latent variables per S.2.13, there selected by
    minimising cross-validated RMSE; fixed here to match the reported run."""
    Ytr = label_binarize(ytr, classes=list(range(len(DIAGNOSIS_ORDER))))
    pls = PLSRegression(n_components=n_components)
    pls.fit(Xtr, Ytr)
    return pls.predict(Xte).argmax(axis=1)


def svm_predict(Xtr, ytr, Xte, C=10, gamma=0.01):
    """C and gamma per S.2.13, there selected by grid search; fixed here."""
    svm = SVC(kernel="rbf", C=C, gamma=gamma)
    svm.fit(Xtr, ytr)
    return svm.predict(Xte)


def evaluate(X, y, groups, n_splits=10, n_trials=50, verbose=True):
    pls_accs, svm_accs = [], []
    for trial in range(n_trials):
        pls_fold, svm_fold = [], []
        for tr_idx, te_idx in grouped_folds(X, y, groups, n_splits=n_splits, seed=trial):
            pls_fold.append(balanced_accuracy_score(y[te_idx], plsda_predict(X[tr_idx], y[tr_idx], X[te_idx])))
            svm_fold.append(balanced_accuracy_score(y[te_idx], svm_predict(X[tr_idx], y[tr_idx], X[te_idx])))
        pls_accs.append(np.mean(pls_fold))
        svm_accs.append(np.mean(svm_fold))
        if verbose:
            print(f"trial {trial + 1}/{n_trials}: PLS-DA = {pls_accs[-1] * 100:.1f}%, "
                  f"SVM = {svm_accs[-1] * 100:.1f}%")
    return np.array(pls_accs), np.array(svm_accs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/fingerprint_spectra.csv")
    ap.add_argument("--trials", type=int, default=50)
    args = ap.parse_args()

    X, y, groups, _ = load_spectra(args.data)
    Xn = sample_zscore(X)

    pls_accs, svm_accs = evaluate(Xn, y, groups, n_trials=args.trials)
    print(f"\nPLS-DA balanced accuracy: {pls_accs.mean() * 100:.1f}% +/- {pls_accs.std() * 100:.1f}% "
          f"(cf. manuscript S.3.5: 81.2% +/- 3.4%)")
    print(f"SVM balanced accuracy:    {svm_accs.mean() * 100:.1f}% +/- {svm_accs.std() * 100:.1f}% "
          f"(cf. manuscript S.3.5: 85.6% +/- 2.8%)")


if __name__ == "__main__":
    main()
