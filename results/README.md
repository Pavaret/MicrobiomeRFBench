# Example results

Outputs for all four models (RA/PsA × `pathway_only`/`species_diversity`),
produced by `scripts/rf_mtx_response_pipeline.py`. Each model lives under
`<group>/<feature_set>/all_features/`.

## Per-model files

### `tables/`

| File | Contents |
|------|----------|
| `*_Summary.tsv`          | one-row summary: mean/SD/CI AUROC, observed AUROC, permutation p, confusion-matrix metrics |
| `*_AUROC_CI.tsv`         | mean AUROC, SD, SEM, t-critical, 95% CI, min/max across the 100 repeats |
| `*_PerRepeatAUROC.tsv`   | AUROC of every seed repeat |
| `*_PValue.tsv`           | observed AUROC, permutations completed, #(null ≥ observed), p-value |
| `*_Permutation.tsv`      | the full null distribution (one row per permutation) |
| `*_FeatureImportance.tsv`| stability-weighted importance: mean/SD/SEM, mean selection frequency, per-seed columns |
| `*_Predictions.tsv`      | per-patient true label, mean & SD out-of-fold probability, per-seed probabilities |
| `*_ConfusionMatrix.tsv`  | 2×2 counts from the mean out-of-fold probability at threshold 0.5 |
| `*_Settings.json`        | full reproducibility record (seeds, permutations, grid, screen, source files) |

### `plot_data/`

One TSV per figure, sharing the figure's file stem, so any plot can be
regenerated or restyled without re-running the models:

- `*_AUROC.tsv` — grid of `mean_fpr`, mean/SD/SEM TPR, CI bounds, per-seed TPR
- `*_ImportantScore.tsv` — top-N features with mean ± SD
- `*_ConfusionMatrix.tsv` — 2×2 counts
- `*_PermutationNullAUROC.tsv` — histogram bins of the null AUROC (+ observed, p)

### `figures/`

The same four plots as **SVG**: `AUROC`, `ImportantScore`, `ConfusionMatrix`,
`PermutationNullAUROC`. Rebuild them from `plot_data/` with:

```bash
python scripts/plot_svg.py \
  --plot-data-dir results/PsA/pathway_only/all_features/plot_data \
  --out-dir results/PsA/pathway_only/all_features/figures --flatten
```

## `figures_gallery/`

All 16 SVGs (4 models × 4 plots) in one flat folder for quick browsing and for the
figures embedded in the top-level `README.md`.

## `combined/`

Cross-model rollups concatenated over the four models:

- `all_runs_summary.tsv`, `all_runs_auroc_ci.tsv`, `all_runs_pvalues.tsv`
- `all_runs_per_repeat_auroc.tsv`, `all_runs_predictions.tsv`,
  `all_runs_confusion_matrices.tsv`
- `input_manifest.tsv` — which input file feeds each model

## Note on this example bundle

The shipped numbers are the published results. The 100-seed-repeat confidence band
and the 1,000-permutation p-value were originally computed by two separate analysis
runs; because seed 42 of the repeat set reproduces the permutation run's observed
AUROC exactly, the two describe the *same* fitted model and are presented together
here, matching what a single run of the combined pipeline produces. Re-running
`scripts/rf_mtx_response_pipeline.py` regenerates this whole tree from scratch. The
large optional per-fold tables (`*_FoldParameters.tsv`,
`*_SelectedFeaturesByFold.tsv`, written only with `--write-fold-tables`) are omitted
to keep the repository lightweight.
