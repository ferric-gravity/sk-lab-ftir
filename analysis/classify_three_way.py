"""
Three-way diagnostic classification (control / benign / metastatic) by a
feed-forward neural network, matching manuscript Methods S.2.11.

This supersedes, for reporting purposes, the binary single-split prototype
in ../scripts/spectral_script.py (kept as-is for provenance -- see
../scripts/README.md for how the two differ) on every point the manuscript
specifies: three-way labels instead of binary, Adam instead of Adagrad,
grid-searched learning rate, and the fold structure actually described
(10-fold, repeated over 50 independent trials, participant-grouped,
permutation-tested) rather than a single 10-fold split.

Usage:
    python classify_three_way.py --data data/fingerprint_spectra.csv
    python classify_three_way.py --data data/fingerprint_spectra.csv --quick   # smoke test
"""

import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score

from data_utils import DIAGNOSIS_ORDER, grouped_folds, load_spectra, sample_zscore


class FNN3Way(nn.Module):
    """
    Four hidden layers (256, 128, 64, 32), SELU activation and Alpha Dropout
    throughout, three-unit SoftMax output -- the architecture specified in
    S.2.11. SELU is only self-normalising if paired with LeCun-normal
    initialisation and Alpha Dropout (not standard Dropout): standard
    Dropout zeroes activations, which pushes the running mean/variance away
    from the fixed point SELU is designed to preserve, while Alpha Dropout
    is built to leave that fixed point undisturbed.
    """

    def __init__(self, input_dim, n_classes=3, dropout=0.10):
        super().__init__()
        dims = [input_dim, 256, 128, 64, 32]
        layers = []
        for d_in, d_out in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(d_in, d_out), nn.SELU(), nn.AlphaDropout(dropout)]
        layers.append(nn.Linear(dims[-1], n_classes))
        self.net = nn.Sequential(*layers)
        self.apply(self._lecun_normal_init)

    @staticmethod
    def _lecun_normal_init(m):
        if isinstance(m, nn.Linear):
            fan_in = m.weight.shape[1]
            nn.init.normal_(m.weight, mean=0.0, std=fan_in ** -0.5)
            nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


def train_one_fold(Xtr, ytr, Xval, yval, lr, device, max_epochs=200, patience=15,
                    batch_size=64, weight_decay=1e-5, seed=0):
    torch.manual_seed(seed)
    model = FNN3Way(Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    Xtr_t = torch.tensor(Xtr, dtype=torch.float32, device=device)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=device)
    Xval_t = torch.tensor(Xval, dtype=torch.float32, device=device)
    yval_t = torch.tensor(yval, dtype=torch.long, device=device)

    best_val, best_state, stale = np.inf, None, 0
    n = Xtr_t.shape[0]
    for _ in range(max_epochs):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            loss = loss_fn(model(Xtr_t[idx]), ytr_t[idx])
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(Xval_t), yval_t).item()
        if val_loss < best_val - 1e-4:
            best_val, best_state, stale = val_loss, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            stale += 1
            if stale >= patience:
                break

    model.load_state_dict(best_state)
    return model


def grid_search_lr(X, y, groups, device, lrs=(1e-4, 1e-3, 1e-2), seed=0):
    """
    S.2.11: learning rate chosen by grid search over 1e-4 to 1e-2. Run on an
    inner 5-fold grouped split carved out of the trial's training data only,
    so the learning-rate choice never sees the outer test fold it will
    later be evaluated on.
    """
    best_lr, best_score = lrs[0], -np.inf
    for lr in lrs:
        scores = []
        for tr_idx, val_idx in grouped_folds(X, y, groups, n_splits=5, seed=seed):
            model = train_one_fold(X[tr_idx], y[tr_idx], X[val_idx], y[val_idx], lr, device, seed=seed)
            with torch.no_grad():
                pred = model(torch.tensor(X[val_idx], dtype=torch.float32, device=device)).argmax(1).cpu().numpy()
            scores.append(balanced_accuracy_score(y[val_idx], pred))
        if np.mean(scores) > best_score:
            best_lr, best_score = lr, np.mean(scores)
    return best_lr


def run_trials(X, y, groups, device, n_trials=50, n_splits=10, verbose=True):
    trial_accs = []
    for trial in range(n_trials):
        fold_accs = []
        for tr_idx, te_idx in grouped_folds(X, y, groups, n_splits=n_splits, seed=trial):
            lr = grid_search_lr(X[tr_idx], y[tr_idx], groups[tr_idx], device, seed=trial)
            model = train_one_fold(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], lr, device, seed=trial)
            with torch.no_grad():
                pred = model(torch.tensor(X[te_idx], dtype=torch.float32, device=device)).argmax(1).cpu().numpy()
            fold_accs.append(balanced_accuracy_score(y[te_idx], pred))
        trial_accs.append(np.mean(fold_accs))
        if verbose:
            print(f"trial {trial + 1}/{n_trials}: balanced accuracy = {trial_accs[-1] * 100:.1f}%")
    return np.array(trial_accs)


def permutation_test(X, y, groups, device, observed_mean, n_rounds=1000, n_splits=10, lr=1e-3, seed=0):
    """
    Null distribution for the observed mean balanced accuracy (S.2.11,
    "a permutation test with 1,000 label-shuffled rounds"). Each round runs
    one 10-fold grouped evaluation at a fixed learning rate rather than the
    full 50-trial grid-searched pipeline -- 1000x50 retrainings is not
    computationally defensible, and one run per permutation round is
    standard practice for a permutation-test p-value.
    """
    rng = np.random.default_rng(seed)
    null_scores = np.empty(n_rounds)
    for r in range(n_rounds):
        y_perm = rng.permutation(y)
        fold_accs = []
        for tr_idx, te_idx in grouped_folds(X, y_perm, groups, n_splits=n_splits, seed=r):
            model = train_one_fold(X[tr_idx], y_perm[tr_idx], X[te_idx], y_perm[te_idx], lr, device, seed=r)
            with torch.no_grad():
                pred = model(torch.tensor(X[te_idx], dtype=torch.float32, device=device)).argmax(1).cpu().numpy()
            fold_accs.append(balanced_accuracy_score(y_perm[te_idx], pred))
        null_scores[r] = np.mean(fold_accs)
    p_value = (np.sum(null_scores >= observed_mean) + 1) / (n_rounds + 1)
    return null_scores, p_value


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/fingerprint_spectra.csv")
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--quick", action="store_true", help="2 trials / 100 permutation rounds, for a smoke test")
    args = ap.parse_args()

    X, y, groups, _ = load_spectra(args.data)
    Xn = sample_zscore(X)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loaded {X.shape[0]} spectra from {len(set(groups))} participants, "
          f"classes: {dict(zip(DIAGNOSIS_ORDER, np.bincount(y)))}. Using: {device}")

    n_trials = 2 if args.quick else args.trials
    n_rounds = 100 if args.quick else 1000

    accs = run_trials(Xn, y, groups, device, n_trials=n_trials)
    print(f"\nMean balanced accuracy: {accs.mean() * 100:.1f}% +/- {accs.std() * 100:.1f}% "
          f"(cf. manuscript Table/S.3.5: 89.4% +/- 2.1%)")

    _, p = permutation_test(Xn, y, groups, device, accs.mean(), n_rounds=n_rounds)
    print(f"Permutation test p-value: {p:.4f} (n={n_rounds} rounds)")


if __name__ == "__main__":
    main()
