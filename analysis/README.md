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

## Walkthrough — if you have the real data

This is the full sequence for whoever has access to the raw spectral
measurements and participant/diagnosis records (the one piece this repo
deliberately does not include).

### 1. Get the code

```bash
git clone https://github.com/ferric-gravity/sk-lab-ftir.git
cd sk-lab-ftir
pip install -r requirements.txt
```

### 2. Shape the data into one CSV

The only step that actually requires domain knowledge of the raw exports.
Produce a single file at `analysis/data/fingerprint_spectra.csv`, one row
per spectrum:

| participant_id | diagnosis | 1800.0 | 1798.0 | ... | 900.0 |
|---|---|---|---|---|---|
| P001 | control | 0.142 | 0.139 | ... | 0.088 |
| P001 | control | 0.140 | 0.141 | ... | 0.091 |
| P002 | metastatic | 0.201 | 0.198 | ... | 0.103 |

- A participant with 5 replicate ATR spectra (S.2.3) appears as 5 rows
  sharing the same `participant_id` — this is what lets `data_utils.py`
  group correctly.
- `diagnosis` must be exactly `control`, `benign`, or `metastatic`.
- The remaining columns are the 462 wavenumbers (1800→900 cm⁻¹); header =
  wavenumber, cell = absorbance.

This is almost certainly a reshape of files that already exist (per-tissue
exports from the Agilent software, or similar) — the work is making
`participant_id` consistent across replicates and matching the diagnosis
spelling above.

### 3. Smoke-test before committing to a full run

```bash
cd analysis
python classify_three_way.py --data data/fingerprint_spectra.csv --quick
python sensitivity_analysis.py --data data/fingerprint_spectra.csv --networks 5
```

`--quick` runs 2 trials instead of 50 — a couple of minutes, just to catch
schema problems early rather than after a long run.

### 4. Run the classifier for real

```bash
python classify_three_way.py --data data/fingerprint_spectra.csv
```

Slow — 10-fold CV × 50 trials × a small grid search. A GPU is strongly
recommended; each script falls back to CPU automatically
(`torch.cuda.is_available()`). Prints a balanced accuracy to compare
against the manuscript's 89.4% ± 2.1%.

### 5. Run the sensitivity analysis

```bash
python sensitivity_analysis.py --data data/fingerprint_spectra.csv
```

Prints the top 10 most influential wavenumbers and saves a plot — this is
what should reproduce 1632 cm⁻¹ and 1032 cm⁻¹ as the top hits, now from
the final three-group model rather than the exploratory binary one.

### 6. Run the comparators

```bash
python chemometric_comparators.py --data data/fingerprint_spectra.csv
```

Gives the PLS-DA and SVM numbers to check against 81.2% and 85.6%.

### 7. Compare against the manuscript

Whatever comes out either confirms the numbers already written, or it
doesn't — worth knowing before submission, not after a reviewer asks for
the code.

The only blocker for all of this is step 2; everything downstream just
works once that file exists.
