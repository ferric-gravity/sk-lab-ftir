# analysis/

Final-methodology code, written to match the manuscript's Methods section
exactly rather than to explore it. Each script below maps to one subsection;
read them in this order.

| Script | Manuscript section | What it produces |
|---|---|---|
| `data_utils.py` | S.2.9 (zonal sampling is out of scope here) | Shared loading + participant-grouped fold splitting used by everything below |
| `classify_three_way.py` | S.2.11 | The three-way (control/benign/metastatic) FNN classifier and its reported balanced accuracy (89.4% +/- 2.1%) |
| `sensitivity_analysis.py` | S.2.12 | The perturbation-based wavenumber sensitivity profile (1632 cm-1, 1032 cm-1 flagged) |
| `chemometric_comparators.py` | S.2.13 | PLS-DA and SVM comparators under identical resampling |

Not implemented here: the pixel-level 1D CNN image classifier (S.2.14,
behind the still-missing Figure 5) and the zonal recomputation behind
Figure 4 (S.2.9/S.3.4) — both are outside this contribution's scope and
are left exactly as flagged in the manuscript.

## Why this exists alongside `../scripts/`

`../scripts/spectral_script.py` is the original exploratory prototype: a
**binary** (patient vs. control) classifier, trained with **Adagrad** on a
**single** 10-fold split that does not group by participant. It is what
first surfaced the 1632 cm-1 / 1032 cm-1 signal and is kept untouched as a
record of that. It does not, however, match what the manuscript's Methods
describe or what its reported numbers imply was run — see
`../scripts/README.md` for the full list of differences.

This folder is a from-scratch implementation of what S.2.11-S.2.13 actually
say: three diagnostic groups, Adam with a grid-searched learning rate,
`StratifiedGroupKFold` so no participant's spectra cross a train/test
boundary, 10-fold CV repeated over 50 trials, and a 1000-round permutation
test. Running it against the real cohort is how the manuscript's reported
numbers (89.4% / 81.2% / 85.6%, and the 1632/1032 cm-1 sensitivity result)
can actually be reproduced and verified end-to-end.

## Data

None of the three scripts ship with data — patient-level spectra are not
public. Each expects a single CSV at `data/fingerprint_spectra.csv` (or a
path passed via `--data`):

```
participant_id, diagnosis, <wavenumber_1>, <wavenumber_2>, ..., <wavenumber_462>
```

See the docstring in `data_utils.py` for the exact schema. `data/` is
gitignored so this file can sit there locally without being committed.

## Running

```bash
pip install -r ../requirements.txt

# smoke test on whatever data you have — few trials, fast
python classify_three_way.py --data data/fingerprint_spectra.csv --quick
python sensitivity_analysis.py --data data/fingerprint_spectra.csv --networks 5

# full run matching the manuscript (slow — 50 trials x 10 folds x grid search)
python classify_three_way.py --data data/fingerprint_spectra.csv
python sensitivity_analysis.py --data data/fingerprint_spectra.csv
python chemometric_comparators.py --data data/fingerprint_spectra.csv
```

A GPU is strongly recommended for the full run; each script falls back to
CPU automatically (`torch.cuda.is_available()`).
