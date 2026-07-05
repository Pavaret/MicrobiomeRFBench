# Example input data

Four microbiome **relative-abundance** tables, one per disease group × modality.
They are the exact inputs the pipeline consumes to reproduce the example results.

| File | Group | Modality | Samples | Feature rows |
|------|:-----:|----------|:-------:|:------------:|
| `RA_pathways_relab_with_diversity.tsv`  | RA  | HUMAnN3 pathways    | 29 | 574 pathways + 2 diversity |
| `RA_species_relab_with_diversity.tsv`   | RA  | MetaPhlAn4 species  | 29 | 1420 species + 2 diversity |
| `PsA_pathways_relab_with_diversity.tsv` | PsA | HUMAnN3 pathways    | 30 | 597 pathways + 2 diversity |
| `PsA_species_relab_with_diversity.tsv`  | PsA | MetaPhlAn4 species  | 30 | 1381 species + 2 diversity |

(Feature counts are the raw rows in the file; the pipeline additionally drops
zero-variance features before modelling, so the modelled feature counts are
slightly lower — e.g. 569 for RA `pathway_only`.)

## Table format

Each table is tab-separated with **feature/label rows** and **sample columns**:

```text
sample            007_4640a   008_4655a   010_4671a   ...
MTX_response      remission   remission   remission   ...
<pathway or species #1>   0.00123   0.0    0.00087   ...
<pathway or species #2>   ...
...
observed          207   236   ...        # alpha-diversity: observed richness
shannon           3.62  3.11  ...        # alpha-diversity: Shannon index
```

- The first column header is literally `sample`; the remaining column headers are
  sample identifiers.
- The row named `MTX_response` is the classification **label**
  (`inefficiency` = non-responder, `remission` = responder).
- Rows named `observed` and `shannon` are the two **alpha-diversity** features.
- Every other row is a **feature** (a pathway or a species relative abundance).

The loader transposes the table to samples × features and keeps only these rows;
anything else is ignored.

## Feature sets

The pipeline builds three possible feature sets from these two files per group;
the two focus sets are the defaults:

| Feature set | Source file | Diversity rows | Default |
|-------------|-------------|:--------------:|:-------:|
| `pathway_only`      | `*_pathways_*` | excluded | ✅ |
| `species_diversity` | `*_species_*`  | included | ✅ |
| `pathway_diversity` | `*_pathways_*` | included | — (optional) |

## Provenance and privacy

These tables contain **only** what the model uses: the microbiome feature matrix,
the two alpha-diversity indices and the `MTX_response` label. Clinical and
identifying metadata that were present in the original analysis tables (age, sex,
serology such as RF / ACPA / HLA-B27, collection dates, therapy and sub-clustering
annotations) have been removed — the pipeline never reads them. Sample column IDs
are internal laboratory codes.

- **Relative abundances** were computed with HUMAnN3 (pathways) and MetaPhlAn4
  (species); each sample column sums are on the tool's native relative-abundance
  scale.
- **Alpha diversity** (`observed`, `shannon`) was computed from the species
  profiles.
