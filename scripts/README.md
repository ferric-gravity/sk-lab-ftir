# scripts/ (exploratory)

Everything in this folder is the original exploratory work: the code that
first identified 1632 cm-1 and 1032 cm-1 as discriminating wavenumbers. It
is kept exactly as committed, unmodified, for provenance. For code that
matches what the manuscript's Methods section actually describes, see
`../analysis/`.

## Contents

- **`spectral_script.py`** — feed-forward neural network (4 hidden layers,
  SELU + dropout) trained on patient-vs-control spectra, followed by the
  perturbation-based sensitivity sweep that produced the first version of
  the wavenumber-importance plot. This is the prototype behind manuscript
  Methods S.2.11-S.2.12, but it differs from what S.2.11 describes in four
  ways worth knowing about before citing a number from it:

  | | This script | Manuscript S.2.11 |
  |---|---|---|
  | Classes | Binary (patient vs. control) | Three-way (control / benign / metastatic) |
  | Optimiser | Adagrad | Adam |
  | Hidden layers | 400 / 400 / 400 / 400 | 256 / 128 / 64 / 32 |
  | Evaluation | One `StratifiedKFold(10)` run, split by spectrum | 10-fold x 50 trials, split by participant |

  The last row matters most: splitting by spectrum rather than by
  participant risks letting replicate spectra from the same tissue block
  appear in both train and test folds, which can inflate accuracy. Treat
  any accuracy number from this script as exploratory, not as the figure
  reported in the manuscript.

- **`cleaned_spectral_analysis.py`** — an earlier, simpler pass at the same
  question: fit a univariate linear regression per wavenumber and rank
  wavenumbers by cross-validated MSE. Superseded by the perturbation
  sensitivity analysis above, which asks the question of an actual trained
  classifier rather than of 462 independent single-variable regressions.

- **`Figure 6.png`, `sens_abhay_plot.png`, `sens_data_abhay_plot.png`** —
  three renders of the same sensitivity sweep (two are the same plot with
  the wavenumber axis flipped). All show a dominant peak in the
  1600-1700 cm-1 range, consistent with the 1632 cm-1 beta-sheet band
  called out in the manuscript.

- **`s.txt`** — empty, unused. Left as-is.

## Running

Needs `fastai` and `tqdm` in addition to the base requirements
(`pip install -r ../requirements.txt`), plus `patients.csv` and
`control.csv` (not included — not public data) in this directory.
