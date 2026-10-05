# sk-lab-ftir

Code behind the computational/machine-learning portion of *Protein-structural
gradient across the breast tumour margin: an ATR-FTIR and focal-plane-array
micro spectroscopic imaging study with neural-network analysis*
(Mishra, Mittal, Mathur, Kumar — manuscript in preparation, AIIMS New Delhi).

This repo covers one author's (Aditya Mittal's) contribution specifically:
training a classifier on ATR-FTIR spectra to distinguish control, benign and
metastatic breast tissue, and running a perturbation-based sensitivity
analysis on that classifier to identify *which infrared wavenumbers it
actually relies on*. That analysis independently flagged 1632 cm⁻¹
(β-sheet protein) and 1032 cm⁻¹ (glycogen/nucleic acid) as the most
discriminating bands — agreeing with, and cross-validating, the curve-fitted
result the rest of the paper derives by a separate, non-ML method.

It does **not** cover the FTIR acquisition, the amide I curve fitting, the
zonal/pathology analysis, or the pixel-level tissue-imaging classifier —
those are other authors' contributions and live outside this repo.

## Why two folders

| Folder | What it is |
|---|---|
| [`scripts/`](scripts/) | The original exploratory code. Binary classifier, single split, Adagrad — this is what first found the 1632/1032 cm⁻¹ signal. Kept untouched. |
| [`analysis/`](analysis/) | A clean implementation of what the manuscript's Methods section (S.2.11–S.2.13) actually specifies: three-way classification, Adam, participant-grouped 10-fold CV repeated 50 times, permutation testing, and the PLS-DA/SVM comparators. This is the code that reproduces the reported numbers. |

Each folder has its own README with the detailed breakdown and a script-by-script map to manuscript sections.

## Quick start

```bash
git clone https://github.com/ferric-gravity/sk-lab-ftir.git
cd sk-lab-ftir
pip install -r requirements.txt
```

Neither folder ships with data (patient spectra are not public). See
`analysis/README.md` for the expected CSV schema, or `scripts/README.md`
for the legacy two-file format.

## Status

Exploratory stage is done and its finding (1632/1032 cm⁻¹) is reported in
the manuscript. The `analysis/` pipeline is written to spec but has not yet
been run end-to-end against the full cohort — doing so, and recording the
resulting numbers against the manuscript's reported 89.4% / 81.2% / 85.6%,
is the remaining step.

## License

See [LICENSE](LICENSE).
