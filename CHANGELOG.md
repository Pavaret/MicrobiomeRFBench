# Changelog

All notable changes to this project are documented here.

## [1.0.0] - 2026-07-05

First public release.

### Added
- `scripts/rf_mtx_response_pipeline.py` — a single pipeline that combines the two
  original analysis scripts:
  - **permutation significance** (label-permutation p-value on the base-seed
    model), from the permutation pipeline; and
  - **repeated-seed 95% confidence band** on the ROC and AUROC, plus mean ± SD
    feature importance, from the repeated-seed CI pipeline.
  The base-seed repeat (seed 42) is the model the permutation test scores, so both
  analyses describe the same fitted model.
- Figures are written as **SVG** by default (`--fig-format` also allows `png` /
  `both`); every figure has a matching plot-data TSV.
- `scripts/plot_svg.py` — rebuild any figure as SVG from its plot-data TSV,
  including the permutation-null histogram.
- `scripts/run_pipeline.sh` — repository-relative convenience runner.
- Example inputs (`data/`), example outputs and figures for all four models
  (`results/`), and documentation (`README.md`, `docs/METHODS.md`,
  `data/README.md`, `results/README.md`).

