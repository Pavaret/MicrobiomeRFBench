#!/usr/bin/env python3
"""
Random-Forest MTX-response pipeline — permutation significance + repeated-seed 95% CI
=====================================================================================

One nested leave-one-out random-forest classifier of methotrexate (MTX) response
(inefficiency = 1, remission = 0) from gut-microbiome features, run per disease
group (RA, PsA) and feature set (``pathway_only``, ``species_diversity``).

For every disease-group x feature-set block the pipeline produces, in a single
run, both pieces of evidence that the two original analysis scripts produced
separately:

1.  **Permutation significance.**  The full nested LOO-CV is re-run on the
    base-seed model with many random label permutations; the one-sided p-value is
    ``(#{null AUROC >= observed} + 1) / (#permutations + 1)``.

2.  **Repeated-seed 95% confidence band.**  The whole nested LOO-CV is repeated
    with many base seeds (default 100: 42, 43, ..., 141).  The held-out samples and
    the patients are identical across repeats; only the stochastic parts (in-fold
    pre-screen, inner-CV shuffling, random forests) change with the seed, so the
    spread across repeats reflects run-to-run algorithm stability, not
    patient-sampling uncertainty.  The AU-ROC plot shows the mean ROC with a
    t-based 95% CI shade; the feature-importance plot shows mean +/- 1 SD bars.

The base-seed repeat (seed = ``--base-seed``, default 42) is the model that the
permutation test evaluates, so the two analyses describe the same fitted model.

Design notes
------------
* Input tables have feature/metadata rows and sample columns; after transposition
  rows are samples and columns are features.  Only the feature rows (pathways or
  species, plus ``observed`` / ``shannon`` alpha diversity) and the
  ``MTX_response`` label row are used.
* Each outer fold applies an in-fold feature pre-screen (default: keep the 50
  highest random-forest-importance features) followed by an inner
  stratified-CV random-forest grid search.
* Stability-weighted importance = per-fold impurity importance averaged over
  outer folds (unselected folds contribute zero).
* There is NO top-percentile-cutoff gradient — only the all-feature model is run.
* Every figure is written as SVG (``--fig-format`` also allows ``png`` / ``both``),
  and every figure has a matching plot-ready TSV with the same file stem so the
  plots can be regenerated or restyled with ``plot_svg.py``.

Paths default to the repository layout (``../data`` and ``../results`` relative to
this script); nothing outside the repository is referenced.
"""

from __future__ import annotations

# ---- thread caps must be set BEFORE importing numpy / sklearn ----
import os
for _thread_var in [
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]:
    os.environ.setdefault(_thread_var, "1")

import argparse
import json
import sys
import time
import traceback
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

from joblib import Parallel, delayed

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, LeaveOneOut, StratifiedKFold

try:
    from boruta import BorutaPy
    HAS_BORUTA = True
except Exception:  # pragma: no cover - only used when Boruta is unavailable
    BorutaPy = None
    HAS_BORUTA = False

warnings.filterwarnings("ignore")

# --- repository-relative defaults (no absolute / cluster paths) --------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent
DEFAULT_DATA_DIR = REPO_DIR / "data"
DEFAULT_OUT_DIR = REPO_DIR / "results"

DEFAULT_BASE_SEED = 42
DEFAULT_N_REPEATS = 100
DEFAULT_N_PERMUTATIONS = 1000

LABEL_COL = "MTX_response"
POSITIVE_CLASS = "inefficiency"
NEGATIVE_CLASS = "remission"
CLASS_LABELS = [NEGATIVE_CLASS, POSITIVE_CLASS]
DIVERSITY_FEATURES = {"observed", "shannon"}

# Metadata row names that may appear in an input table and must never be treated
# as features.  The loader drops rows by name, not by count, so tables with or
# without a given metadata row both load correctly.
METADATA_ROWS = {
    "patient-a_partner-b",
    "time_point",
    "diagnose",
    "collection_date",
    "difference_to_first_sample (in months)",
    "RF",
    "ACPA",
    "anti-CD74_qualitative",
    "HLA-B27",
    "gender",
    "age",
    "MTX_treatment",
    "MTX_response",
    "MTX_response_subclustering",
    "other_therapy",
    "clustering_other_therapy",
}

DEFAULT_PARAM_GRID = {
    "n_estimators": [300, 500],
    "max_depth": [None, 4, 6],
    "min_samples_leaf": [1, 2, 3],
    "max_features": ["sqrt", 0.3],
}


@dataclass(frozen=True)
class InputFiles:
    group: str
    pathway: Path
    species: Path


@dataclass
class FeatureSetData:
    group: str
    feature_set: str
    X: pd.DataFrame
    y: pd.Series
    feature_types: Dict[str, str]
    source_files: Dict[str, str]


# =============================================================================
# Generic helpers
# =============================================================================

def tsv(df: pd.DataFrame, path: Path) -> None:
    """Write a TSV, atomically when the filesystem allows it.

    The preferred path is write-to-temp + os.replace (atomic rename). On shared
    filesystems the rename can intermittently fail when several jobs touch the
    same global file at once, so it is retried a few times and then falls back to
    a direct write (last-writer-wins) rather than aborting the run.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    df.to_csv(tmp, sep="\t", index=False)
    for attempt in range(4):
        try:
            os.replace(tmp, path)
            return
        except OSError:
            time.sleep(0.2 * (attempt + 1))
    try:
        df.to_csv(path, sep="\t", index=False)
    finally:
        try:
            if tmp.exists():
                os.remove(tmp)
        except OSError:
            pass


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def make_unique(names: Sequence[object]) -> List[str]:
    """Return unique strings while preserving the original order."""
    seen: Dict[str, int] = {}
    out: List[str] = []
    for value in names:
        base = str(value)
        if base not in seen:
            seen[base] = 0
            out.append(base)
        else:
            seen[base] += 1
            out.append(f"{base}__dup{seen[base]}")
    return out


def zero_div(numer: float, denom: float) -> float:
    return float(numer / denom) if denom else float("nan")


def t_critical(df: int, alpha: float = 0.05) -> float:
    """Two-sided t critical value; uses SciPy when available, else a lookup table."""
    if df < 1:
        return float("nan")
    try:
        from scipy import stats

        return float(stats.t.ppf(1.0 - alpha / 2.0, df))
    except Exception:
        table = {
            1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
            7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
            13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
            19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
            25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
        }
        if df in table:
            return table[df]
        if df > 30:
            return 1.96
        return table[max(1, min(30, df))]


def resolve_seeds(args: argparse.Namespace) -> List[int]:
    if args.seeds:
        seeds = [int(s) for s in args.seeds]
    else:
        seeds = [int(args.base_seed) + i for i in range(int(args.n_repeats))]
    if len(seeds) < 1:
        raise ValueError("Need at least one seed/repeat.")
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Duplicate seeds requested: {seeds}")
    return seeds


def get_input_files(data_dir: Path) -> Dict[str, InputFiles]:
    return {
        "RA": InputFiles(
            group="RA",
            pathway=data_dir / "RA_pathways_relab_with_diversity.tsv",
            species=data_dir / "RA_species_relab_with_diversity.tsv",
        ),
        "PsA": InputFiles(
            group="PsA",
            pathway=data_dir / "PsA_pathways_relab_with_diversity.tsv",
            species=data_dir / "PsA_species_relab_with_diversity.tsv",
        ),
    }


# =============================================================================
# Data loading and feature-set construction
# =============================================================================

def load_one_transposed_matrix(
    path: Path,
    feature_family: str,
    include_diversity: bool,
) -> Tuple[pd.DataFrame, pd.Series, Dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    raw = pd.read_csv(path, sep="\t", low_memory=False)
    if "sample" not in raw.columns:
        raise ValueError(f"Missing first/ID column 'sample' in {path}")

    raw = raw.copy()
    raw["sample"] = make_unique(raw["sample"].tolist())
    df = raw.set_index("sample").T

    if LABEL_COL not in df.columns:
        raise ValueError(f"{LABEL_COL!r} not found in {path}")

    y = df[LABEL_COL].astype(str).str.strip()

    feature_cols: List[str] = []
    feature_types: Dict[str, str] = {}
    for col in df.columns:
        if col in METADATA_ROWS:
            continue
        is_div = col in DIVERSITY_FEATURES
        if is_div and not include_diversity:
            continue
        feature_cols.append(col)
        feature_types[col] = "diversity" if is_div else feature_family

    if not feature_cols:
        raise ValueError(f"No feature columns found in {path}")

    X = df.loc[:, feature_cols].apply(pd.to_numeric, errors="coerce")

    valid = y.isin([POSITIVE_CLASS, NEGATIVE_CLASS])
    X = X.loc[valid].fillna(0.0)
    y = y.loc[valid]

    nunique = X.nunique(dropna=False)
    X = X.loc[:, nunique > 1]
    feature_types = {c: feature_types[c] for c in X.columns}

    X = X.sort_index()
    y = y.loc[X.index]

    if len(np.unique(y)) < 2:
        raise ValueError(
            f"Only one valid class found in {path}; counts={y.value_counts().to_dict()}"
        )

    return X, y, feature_types


def load_feature_set(group: str, feature_set: str, data_dir: Path) -> FeatureSetData:
    inputs = get_input_files(data_dir)[group]

    if feature_set == "pathway_only":
        X, y, feature_types = load_one_transposed_matrix(
            inputs.pathway, feature_family="pathway", include_diversity=False
        )
        sources = {"pathway": inputs.pathway.name}

    elif feature_set == "species_diversity":
        X, y, feature_types = load_one_transposed_matrix(
            inputs.species, feature_family="species", include_diversity=True
        )
        sources = {"species": inputs.species.name}

    elif feature_set == "pathway_diversity":
        X, y, feature_types = load_one_transposed_matrix(
            inputs.pathway, feature_family="pathway", include_diversity=True
        )
        sources = {"pathway": inputs.pathway.name}

    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

    if X.columns.duplicated().any():
        old_cols = X.columns.tolist()
        new_cols = make_unique(old_cols)
        rename_map = dict(zip(old_cols, new_cols))
        X.columns = new_cols
        feature_types = {rename_map.get(k, k): v for k, v in feature_types.items()}

    nunique = X.nunique(dropna=False)
    X = X.loc[:, nunique > 1]
    feature_types = {c: feature_types.get(c, "unknown") for c in X.columns}

    print(
        f"[load] {group} {feature_set}: X={X.shape}, "
        f"y={y.value_counts().to_dict()}, sources={sources}"
    )
    return FeatureSetData(group, feature_set, X, y, feature_types, sources)


# =============================================================================
# Core ML pipeline
# =============================================================================

def build_rf_for_screen(n_estimators: int, random_state: int) -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=n_estimators,
        n_jobs=1,
        class_weight="balanced",
        max_depth=5,
        random_state=random_state,
    )


def fallback_feature_screen(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    max_features: int,
    n_estimators: int,
) -> np.ndarray:
    """RF-importance screen: keep the top ``max_features`` by impurity importance."""
    p = X.shape[1]
    n_keep = min(max_features, p)
    rf = build_rf_for_screen(n_estimators=n_estimators, random_state=random_state)
    rf.fit(X, y)
    order = np.argsort(rf.feature_importances_)[::-1]
    mask = np.zeros(p, dtype=bool)
    mask[order[:n_keep]] = True
    return mask


def run_preselector(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    prescreen: str,
    boruta_perc: int,
    boruta_max_iter: int,
    boruta_estimator_n: int,
    fallback_max_features: int,
) -> Tuple[np.ndarray, str]:
    """In-fold feature pre-screen, selected explicitly by ``prescreen``.

    * ``"rf_fallback"`` (default) - always keep the top-``fallback_max_features``
      RF-importance features.  This reproduces the primary/production models.
    * ``"boruta"`` - Boruta pre-screen; if Boruta is unavailable or rejects every
      feature in a fold, fall back to the RF-importance screen.
    * ``"none"`` - keep every feature (no screen).
    """
    p = X.shape[1]
    if p == 0:
        raise ValueError("No features were supplied to preselector")

    if prescreen == "none":
        return np.ones(p, dtype=bool), "none_all_features"

    if prescreen == "boruta" and HAS_BORUTA and BorutaPy is not None:
        try:
            rf_for_boruta = build_rf_for_screen(
                n_estimators=boruta_estimator_n, random_state=random_state
            )
            boruta = BorutaPy(
                estimator=rf_for_boruta,
                n_estimators="auto",
                perc=boruta_perc,
                max_iter=boruta_max_iter,
                random_state=random_state,
                verbose=0,
            )
            boruta.fit(X, y)
            mask = boruta.support_ | boruta.support_weak_
            if int(mask.sum()) > 0:
                return mask, "boruta_confirmed_or_tentative"
        except Exception:
            pass

    mask = fallback_feature_screen(
        X=X,
        y=y,
        random_state=random_state,
        max_features=fallback_max_features,
        n_estimators=boruta_estimator_n,
    )
    return mask, "rf_importance_fallback"


def tune_rf(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int,
    param_grid: Dict[str, Sequence[object]],
) -> Tuple[RandomForestClassifier, Dict[str, object]]:
    base = RandomForestClassifier(class_weight="balanced", random_state=random_state, n_jobs=1)

    counts = np.bincount(y.astype(int), minlength=2)
    min_class = int(counts[counts > 0].min())
    n_splits = min(3, min_class)

    if n_splits < 2:
        params = {
            "n_estimators": 500,
            "max_depth": None,
            "min_samples_leaf": 1,
            "max_features": "sqrt",
        }
        model = RandomForestClassifier(
            **params, class_weight="balanced", random_state=random_state, n_jobs=1
        )
        model.fit(X, y)
        return model, params

    inner_cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    gs = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=inner_cv,
        n_jobs=1,
        refit=True,
        error_score="raise",
    )
    gs.fit(X, y)
    return gs.best_estimator_, dict(gs.best_params_)


def _one_fold(
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    X_arr: np.ndarray,
    y_enc: np.ndarray,
    random_state: int,
    prescreen: str,
    boruta_perc: int,
    boruta_max_iter: int,
    boruta_estimator_n: int,
    fallback_max_features: int,
    param_grid: Dict[str, Sequence[object]],
    collect_details: bool,
) -> Dict[str, object]:
    X_tr, X_te = X_arr[train_idx], X_arr[test_idx]
    y_tr = y_enc[train_idx]

    fold_seed = random_state + fold_idx
    mask, preselector = run_preselector(
        X_tr,
        y_tr,
        random_state=fold_seed,
        prescreen=prescreen,
        boruta_perc=boruta_perc,
        boruta_max_iter=boruta_max_iter,
        boruta_estimator_n=boruta_estimator_n,
        fallback_max_features=fallback_max_features,
    )

    X_tr_sel = X_tr[:, mask]
    X_te_sel = X_te[:, mask]

    best_rf, best_params = tune_rf(
        X_tr_sel, y_tr, random_state=fold_seed, param_grid=param_grid
    )

    pos_col = list(best_rf.classes_).index(1)
    proba = float(best_rf.predict_proba(X_te_sel)[0, pos_col])

    out: Dict[str, object] = {
        "fold_idx": int(fold_idx),
        "test_i": int(test_idx[0]),
        "proba": proba,
    }
    if collect_details:
        out.update(
            {
                "mask": mask,
                "importances": best_rf.feature_importances_.astype(float),
                "params": best_params,
                "preselector": preselector,
            }
        )
    return out


def nested_loo(
    X: pd.DataFrame,
    y_enc: np.ndarray,
    random_state: int,
    n_jobs: int,
    prescreen: str,
    boruta_perc: int,
    boruta_max_iter: int,
    boruta_estimator_n: int,
    fallback_max_features: int,
    param_grid: Dict[str, Sequence[object]],
    collect_details: bool,
) -> Tuple[np.ndarray, Optional[Dict[str, object]]]:
    loo = LeaveOneOut()
    X_arr = X.to_numpy(dtype=float)
    splits = list(loo.split(X_arr))
    n = X.shape[0]

    args = (
        X_arr, y_enc, random_state, prescreen, boruta_perc, boruta_max_iter,
        boruta_estimator_n, fallback_max_features, param_grid, collect_details,
    )
    if n_jobs == 1:
        results = [_one_fold(i, tr, te, *args) for i, (tr, te) in enumerate(splits)]
    else:
        results = Parallel(n_jobs=n_jobs, backend="loky")(
            delayed(_one_fold)(i, tr, te, *args) for i, (tr, te) in enumerate(splits)
        )

    oof_proba = np.zeros(n, dtype=float)
    fold_masks: List[np.ndarray] = []
    fold_importances: List[np.ndarray] = []
    fold_params: List[Dict[str, object]] = []
    fold_preselectors: List[str] = []
    fold_test_indices: List[int] = []

    for r in sorted(results, key=lambda d: int(d["fold_idx"])):
        test_i = int(r["test_i"])
        oof_proba[test_i] = float(r["proba"])
        if collect_details:
            fold_test_indices.append(test_i)
            mask = np.asarray(r["mask"], dtype=bool)
            fold_masks.append(mask)
            imp_full = np.full(X.shape[1], np.nan, dtype=float)
            imp_full[mask] = np.asarray(r["importances"], dtype=float)
            fold_importances.append(imp_full)
            fold_params.append(dict(r["params"]))
            fold_preselectors.append(str(r["preselector"]))

    if not collect_details:
        return oof_proba, None

    details = {
        "feature_names": X.columns.to_numpy(dtype=object),
        "sample_names": X.index.to_numpy(dtype=object),
        "fold_masks": np.vstack(fold_masks),
        "fold_importances": np.vstack(fold_importances),
        "fold_params": fold_params,
        "fold_preselectors": fold_preselectors,
        "fold_test_indices": fold_test_indices,
    }
    return oof_proba, details


def repeated_nested_loo(
    X: pd.DataFrame,
    y_enc: np.ndarray,
    seeds: Sequence[int],
    n_jobs: int,
    prescreen: str,
    boruta_perc: int,
    boruta_max_iter: int,
    boruta_estimator_n: int,
    fallback_max_features: int,
    param_grid: Dict[str, Sequence[object]],
) -> Tuple[List[np.ndarray], List[float], List[Dict[str, object]]]:
    """Run the full nested LOO-CV once per base seed.

    The base-seed repeat (``seeds[0]``) is the model the permutation test scores.
    """
    oof_list: List[np.ndarray] = []
    auc_list: List[float] = []
    details_list: List[Dict[str, object]] = []
    for ri, seed in enumerate(seeds, start=1):
        t0 = time.time()
        oof, details = nested_loo(
            X,
            y_enc,
            random_state=int(seed),
            n_jobs=n_jobs,
            prescreen=prescreen,
            boruta_perc=boruta_perc,
            boruta_max_iter=boruta_max_iter,
            boruta_estimator_n=boruta_estimator_n,
            fallback_max_features=fallback_max_features,
            param_grid=param_grid,
            collect_details=True,
        )
        assert details is not None
        auc = float(roc_auc_score(y_enc, oof))
        oof_list.append(oof)
        auc_list.append(auc)
        details_list.append(details)
        print(
            f"  repeat {ri}/{len(seeds)} seed={seed}: AUROC={auc:.4f} "
            f"({time.time() - t0:.1f}s)",
            flush=True,
        )
    return oof_list, auc_list, details_list


# =============================================================================
# Permutation significance (evaluated on the base-seed model)
# =============================================================================

def _one_permutation(
    perm_seed: int,
    X: pd.DataFrame,
    y_enc: np.ndarray,
    prescreen: str,
    boruta_perc: int,
    boruta_max_iter: int,
    boruta_estimator_n: int,
    fallback_max_features: int,
    param_grid: Dict[str, Sequence[object]],
) -> float:
    rng = np.random.RandomState(perm_seed)
    y_shuf = rng.permutation(y_enc)
    oof, _ = nested_loo(
        X,
        y_shuf,
        random_state=perm_seed,
        n_jobs=1,
        prescreen=prescreen,
        boruta_perc=boruta_perc,
        boruta_max_iter=boruta_max_iter,
        boruta_estimator_n=boruta_estimator_n,
        fallback_max_features=fallback_max_features,
        param_grid=param_grid,
        collect_details=False,
    )
    return float(roc_auc_score(y_shuf, oof))


def permutation_pvalue(
    X: pd.DataFrame,
    y_enc: np.ndarray,
    observed_auc: float,
    n_perm: int,
    n_jobs: int,
    prescreen: str,
    boruta_perc: int,
    boruta_max_iter: int,
    boruta_estimator_n: int,
    fallback_max_features: int,
    param_grid: Dict[str, Sequence[object]],
) -> Tuple[float, np.ndarray, int]:
    if n_perm <= 0:
        return float("nan"), np.array([], dtype=float), 0

    seeds = np.arange(n_perm, dtype=int) + 10_000
    print(f"  permutations: n={n_perm}, n_jobs={n_jobs}", flush=True)
    null_aucs = Parallel(n_jobs=n_jobs, backend="loky", verbose=5)(
        delayed(_one_permutation)(
            int(s), X, y_enc, prescreen, boruta_perc, boruta_max_iter,
            boruta_estimator_n, fallback_max_features, param_grid,
        )
        for s in seeds
    )
    null = np.array([a for a in null_aucs if not np.isnan(a)], dtype=float)
    n_ge = int(np.sum(null >= observed_auc))
    p = float((n_ge + 1) / (len(null) + 1))
    return p, null, n_ge


# =============================================================================
# Repeated-run summaries (ROC CI + importance SD)
# =============================================================================

def compute_roc_ci(
    y_enc: np.ndarray,
    oof_list: Sequence[np.ndarray],
    n_grid: int = 201,
) -> Dict[str, object]:
    """Mean ROC curve with a t-based 95% CI band across repeats."""
    mean_fpr = np.linspace(0.0, 1.0, n_grid)
    interp_tprs: List[np.ndarray] = []
    per_auc: List[float] = []
    for oof in oof_list:
        fpr, tpr, _ = roc_curve(y_enc, oof)
        ti = np.interp(mean_fpr, fpr, tpr)
        ti[0] = 0.0
        interp_tprs.append(ti)
        per_auc.append(float(roc_auc_score(y_enc, oof)))

    interp = np.vstack(interp_tprs)
    n = interp.shape[0]

    mean_tpr = interp.mean(axis=0)
    mean_tpr[-1] = 1.0
    sd_tpr = interp.std(axis=0, ddof=1) if n > 1 else np.zeros_like(mean_tpr)
    sem_tpr = sd_tpr / np.sqrt(n)
    tcrit = t_critical(n - 1) if n > 1 else 0.0
    tpr_lower = np.clip(mean_tpr - tcrit * sem_tpr, 0.0, 1.0)
    tpr_upper = np.clip(mean_tpr + tcrit * sem_tpr, 0.0, 1.0)

    per_auc_arr = np.asarray(per_auc, dtype=float)
    auc_mean = float(per_auc_arr.mean())
    auc_sd = float(per_auc_arr.std(ddof=1)) if n > 1 else 0.0
    auc_sem = auc_sd / np.sqrt(n)
    auc_ci_low = float(np.clip(auc_mean - tcrit * auc_sem, 0.0, 1.0))
    auc_ci_high = float(np.clip(auc_mean + tcrit * auc_sem, 0.0, 1.0))

    return {
        "mean_fpr": mean_fpr,
        "mean_tpr": mean_tpr,
        "sd_tpr": sd_tpr,
        "sem_tpr": sem_tpr,
        "tpr_lower": tpr_lower,
        "tpr_upper": tpr_upper,
        "interp_tprs": interp,
        "tcrit": float(tcrit),
        "per_auc": per_auc_arr,
        "auc_mean": auc_mean,
        "auc_sd": auc_sd,
        "auc_sem": auc_sem,
        "auc_ci_low": auc_ci_low,
        "auc_ci_high": auc_ci_high,
        "n_repeats": int(n),
    }


def build_feature_importance_table_one(
    details: Dict[str, object],
    feature_types: Dict[str, str],
) -> pd.DataFrame:
    """Single-repeat stability-weighted importance table."""
    fnames = np.asarray(details["feature_names"], dtype=object)
    imps = np.asarray(details["fold_importances"], dtype=float)
    masks = np.asarray(details["fold_masks"], dtype=bool)
    n_folds = imps.shape[0]

    selected_counts = masks.sum(axis=0).astype(int)
    selection_frequency = selected_counts / float(n_folds)
    mean_when_selected = np.nanmean(imps, axis=0)
    mean_when_selected = np.where(np.isnan(mean_when_selected), 0.0, mean_when_selected)
    importance_score = np.nan_to_num(imps, nan=0.0).mean(axis=0)

    return pd.DataFrame(
        {
            "feature": fnames,
            "feature_type": [feature_types.get(str(f), "unknown") for f in fnames],
            "importance_score": importance_score,
            "mean_importance_when_selected": mean_when_selected,
            "selection_frequency": selection_frequency,
            "selected_in_n_folds": selected_counts,
            "n_outer_folds": n_folds,
        }
    )


def build_repeated_importance_table(
    details_list: Sequence[Dict[str, object]],
    seeds: Sequence[int],
    feature_types: Dict[str, str],
) -> pd.DataFrame:
    """Aggregate per-repeat stability-weighted importance into mean +/- SD."""
    score_cols = []
    selfreq_cols = []
    for det, seed in zip(details_list, seeds):
        tab = build_feature_importance_table_one(det, feature_types).set_index("feature")
        score_cols.append(tab["importance_score"].rename(f"importance_score_seed{seed}"))
        selfreq_cols.append(tab["selection_frequency"].rename(f"selection_frequency_seed{seed}"))

    score_df = pd.concat(score_cols, axis=1)
    selfreq_df = pd.concat(selfreq_cols, axis=1)
    n = score_df.shape[1]

    mean_imp = score_df.mean(axis=1)
    sd_imp = score_df.std(axis=1, ddof=1) if n > 1 else pd.Series(0.0, index=score_df.index)
    sem_imp = sd_imp / np.sqrt(n)
    mean_selfreq = selfreq_df.mean(axis=1)

    out = pd.DataFrame(index=mean_imp.index)
    out["feature_type"] = [feature_types.get(str(f), "unknown") for f in out.index]
    out["mean_importance_score"] = mean_imp
    out["sd_importance_score"] = sd_imp
    out["sem_importance_score"] = sem_imp
    out["mean_selection_frequency"] = mean_selfreq
    out["n_repeats"] = n
    out = out.join(score_df)
    out = out.sort_values(
        ["mean_importance_score", "sd_importance_score", "mean_selection_frequency"],
        ascending=[False, True, False],
    )
    out = out.reset_index()
    if "index" in out.columns and "feature" not in out.columns:
        out = out.rename(columns={"index": "feature"})
    out.insert(0, "rank_within_run", np.arange(1, len(out) + 1, dtype=int))
    return out


def prediction_and_confusion_tables(
    samples: Sequence[object],
    y_enc: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, float]]:
    y_pred = (y_proba >= threshold).astype(int)
    cm = confusion_matrix(y_enc, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(v) for v in cm.ravel()]

    pred_df = pd.DataFrame(
        {
            "sample": list(samples),
            "true_label_encoded": y_enc.astype(int),
            "true_label": [POSITIVE_CLASS if v == 1 else NEGATIVE_CLASS for v in y_enc],
            "mean_predicted_probability_inefficiency": y_proba.astype(float),
            "predicted_label_encoded": y_pred.astype(int),
            "predicted_label": [POSITIVE_CLASS if v == 1 else NEGATIVE_CLASS for v in y_pred],
            "classification_threshold": threshold,
        }
    )

    cm_df = pd.DataFrame(
        [
            {"true_label": NEGATIVE_CLASS, "predicted_label": NEGATIVE_CLASS, "count": tn},
            {"true_label": NEGATIVE_CLASS, "predicted_label": POSITIVE_CLASS, "count": fp},
            {"true_label": POSITIVE_CLASS, "predicted_label": NEGATIVE_CLASS, "count": fn},
            {"true_label": POSITIVE_CLASS, "predicted_label": POSITIVE_CLASS, "count": tp},
        ]
    )

    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_enc, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_enc, y_pred)),
        "sensitivity_recall_inefficiency": zero_div(tp, tp + fn),
        "specificity_remission": zero_div(tn, tn + fp),
        "precision_ppv_inefficiency": float(precision_score(y_enc, y_pred, zero_division=0)),
        "npv_remission": zero_div(tn, tn + fn),
        "tn": tn, "fp": fp, "fn": fn, "tp": tp,
    }
    return pred_df, cm_df, metrics


# =============================================================================
# Plots (SVG by default; PNG optional)
# =============================================================================

def _fig_paths(fig_dir: Path, stem: str, fig_format: str) -> List[Path]:
    formats = ["svg", "png"] if fig_format == "both" else [fig_format]
    return [fig_dir / f"{stem}.{fmt}" for fmt in formats]


def _save(fig, paths: Sequence[Path]) -> None:
    for p in paths:
        p.parent.mkdir(parents=True, exist_ok=True)
        if p.suffix == ".png":
            fig.savefig(p, dpi=300, bbox_inches="tight")
        else:
            fig.savefig(p, format="svg", bbox_inches="tight")
    plt.close(fig)


def set_plot_grid(ax: plt.Axes) -> None:
    ax.grid(True, which="major", color="grey", linewidth=0.6, alpha=0.45)
    ax.grid(True, which="minor", color="grey", linewidth=0.35, alpha=0.25)
    try:
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
    except Exception:
        pass
    ax.tick_params(axis="both", which="major", labelsize=12)
    ax.tick_params(axis="both", which="minor", labelsize=10)


def save_roc_ci_plot(
    roc_ci: Dict[str, object],
    seeds: Sequence[int],
    fig_dir: Path,
    stem: str,
    plot_data_path: Path,
    title: str,
    fig_format: str,
) -> None:
    mean_fpr = np.asarray(roc_ci["mean_fpr"], dtype=float)
    mean_tpr = np.asarray(roc_ci["mean_tpr"], dtype=float)
    sd_tpr = np.asarray(roc_ci["sd_tpr"], dtype=float)
    sem_tpr = np.asarray(roc_ci["sem_tpr"], dtype=float)
    lower = np.asarray(roc_ci["tpr_lower"], dtype=float)
    upper = np.asarray(roc_ci["tpr_upper"], dtype=float)
    interp = np.asarray(roc_ci["interp_tprs"], dtype=float)

    plot_df = pd.DataFrame(
        {
            "mean_fpr": mean_fpr,
            "mean_tpr": mean_tpr,
            "sd_tpr": sd_tpr,
            "sem_tpr": sem_tpr,
            "tpr_ci95_lower": lower,
            "tpr_ci95_upper": upper,
        }
    )
    for i, seed in enumerate(seeds):
        plot_df[f"tpr_seed{seed}"] = interp[i]
    tsv(plot_df, plot_data_path)

    auc_mean = float(roc_ci["auc_mean"])
    auc_lo = float(roc_ci["auc_ci_low"])
    auc_hi = float(roc_ci["auc_ci_high"])
    n_rep = int(roc_ci["n_repeats"])

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    for i in range(interp.shape[0]):
        ax.plot(mean_fpr, interp[i], color="tab:blue", linewidth=0.8, alpha=0.25)
    ax.fill_between(mean_fpr, lower, upper, color="tab:blue", alpha=0.25,
                    linewidth=0, label="95% CI (t-based)")
    mean_label = (
        f"Mean AUROC = {auc_mean:.3f}\n"
        f"95% CI [{auc_lo:.3f}, {auc_hi:.3f}]\n"
        f"({n_rep} seed repeats)"
    )
    ax.plot(mean_fpr, mean_tpr, color="tab:blue", linewidth=2.3, label=mean_label)
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.2, color="grey")
    ax.set_xlabel("False positive rate", fontsize=14)
    ax.set_ylabel("True positive rate", fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="lower right", frameon=False, fontsize=11)
    set_plot_grid(ax)
    fig.tight_layout()
    _save(fig, _fig_paths(fig_dir, stem, fig_format))


def save_importance_sd_plot(
    importance_df: pd.DataFrame,
    fig_dir: Path,
    stem: str,
    plot_data_path: Path,
    title: str,
    top_n: int,
    fig_format: str,
) -> None:
    plot_df = importance_df.sort_values("rank_within_run").head(top_n).copy()
    plot_df = plot_df.iloc[::-1].reset_index(drop=True)
    tsv(plot_df, plot_data_path)

    height = max(5.2, 0.34 * len(plot_df) + 2.0)
    fig, ax = plt.subplots(figsize=(11.0, height))
    ax.barh(
        plot_df["feature"],
        plot_df["mean_importance_score"],
        xerr=plot_df["sd_importance_score"],
        capsize=3,
        error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "black"},
    )
    ax.set_xlabel("Stability-weighted RF importance (mean +/- SD across repeats)", fontsize=13)
    ax.set_ylabel("Feature", fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(left=0.0)
    ax.set_ylim(-0.5, len(plot_df) - 0.5)
    set_plot_grid(ax)
    fig.tight_layout()
    _save(fig, _fig_paths(fig_dir, stem, fig_format))


def save_confusion_matrix_plot(
    cm_df: pd.DataFrame,
    fig_dir: Path,
    stem: str,
    plot_data_path: Path,
    title: str,
    fig_format: str,
) -> None:
    tsv(cm_df, plot_data_path)
    matrix = np.array(
        [
            [
                int(cm_df.query("true_label == @NEGATIVE_CLASS and predicted_label == @NEGATIVE_CLASS")["count"].iloc[0]),
                int(cm_df.query("true_label == @NEGATIVE_CLASS and predicted_label == @POSITIVE_CLASS")["count"].iloc[0]),
            ],
            [
                int(cm_df.query("true_label == @POSITIVE_CLASS and predicted_label == @NEGATIVE_CLASS")["count"].iloc[0]),
                int(cm_df.query("true_label == @POSITIVE_CLASS and predicted_label == @POSITIVE_CLASS")["count"].iloc[0]),
            ],
        ],
        dtype=int,
    )

    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(matrix, interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label("Count", fontsize=14)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(CLASS_LABELS, fontsize=12, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_LABELS, fontsize=12)
    ax.set_xlabel("Predicted label", fontsize=14)
    ax.set_ylabel("True label", fontsize=14)
    ax.set_title(title, fontsize=14)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=14)
    ax.set_xticks(np.arange(-0.5, 2, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 2, 1), minor=True)
    ax.grid(True, which="minor", color="grey", linewidth=0.8, alpha=0.45)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    _save(fig, _fig_paths(fig_dir, stem, fig_format))


def save_null_distribution_plot(
    null_aucs: np.ndarray,
    observed_auc: float,
    p_val: float,
    fig_dir: Path,
    stem: str,
    plot_data_path: Path,
    title: str,
    fig_format: str,
) -> None:
    if len(null_aucs) == 0:
        tsv(pd.DataFrame(columns=["bin_left", "bin_right", "bin_center", "count"]), plot_data_path)
        return
    counts, edges = np.histogram(null_aucs, bins=40)
    centers = 0.5 * (edges[:-1] + edges[1:])
    plot_df = pd.DataFrame(
        {"bin_left": edges[:-1], "bin_right": edges[1:], "bin_center": centers, "count": counts}
    )
    tsv(plot_df, plot_data_path)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.hist(null_aucs, bins=40, color="tab:blue", alpha=0.85)
    label = f"Observed AUROC = {observed_auc:.3f}"
    if not np.isnan(p_val):
        label += f"\npermutation p = {p_val:.4f}"
    ax.axvline(observed_auc, linewidth=2.0, linestyle="--", color="tab:red", label=label)
    ax.set_xlabel("AUROC under permuted labels", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.legend(frameon=False, fontsize=12)
    set_plot_grid(ax)
    fig.tight_layout()
    _save(fig, _fig_paths(fig_dir, stem, fig_format))


def fold_tables_repeated(
    details_list: Sequence[Dict[str, object]],
    seeds: Sequence[int],
    samples: Sequence[object],
    feature_names: Sequence[object],
    feature_types: Dict[str, str],
    run_id: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    param_rows = []
    sel_rows = []
    fnames = np.asarray(feature_names, dtype=object)
    for det, seed in zip(details_list, seeds):
        for fold_idx, (test_i, params, preselector) in enumerate(
            zip(det["fold_test_indices"], det["fold_params"], det["fold_preselectors"])
        ):
            row = {
                "run_id": run_id,
                "repeat_seed": int(seed),
                "fold_idx": fold_idx,
                "test_sample": samples[int(test_i)],
                "preselector": preselector,
            }
            row.update({f"param_{k}": v for k, v in params.items()})
            param_rows.append(row)

        masks = np.asarray(det["fold_masks"], dtype=bool)
        imps = np.asarray(det["fold_importances"], dtype=float)
        for fold_idx in range(masks.shape[0]):
            for idx in np.where(masks[fold_idx])[0]:
                feat = str(fnames[idx])
                sel_rows.append(
                    {
                        "run_id": run_id,
                        "repeat_seed": int(seed),
                        "fold_idx": fold_idx,
                        "feature": feat,
                        "feature_type": feature_types.get(feat, "unknown"),
                        "fold_importance": imps[fold_idx, idx],
                    }
                )
    return pd.DataFrame(param_rows), pd.DataFrame(sel_rows)


# =============================================================================
# Run orchestration
# =============================================================================

def is_run_complete(run_dir: Path, run_id: str) -> bool:
    table_dir = run_dir / "tables"
    return (
        (run_dir / "RUN_COMPLETE.ok").exists()
        and (table_dir / f"{run_id}_Summary.tsv").exists()
        and (table_dir / f"{run_id}_FeatureImportance.tsv").exists()
    )


def run_one_block(
    data: FeatureSetData,
    seeds: Sequence[int],
    out_dir: Path,
    n_jobs: int,
    n_permutations: int,
    threshold: float,
    top_n_plot: int,
    prescreen: str,
    boruta_perc: int,
    boruta_max_iter: int,
    boruta_estimator_n: int,
    fallback_max_features: int,
    param_grid: Dict[str, Sequence[object]],
    fig_format: str,
    write_fold_tables: bool,
    resume: bool,
    overwrite: bool,
) -> Dict[str, object]:
    run_name = "all_features"
    run_id = f"{data.group}_{data.feature_set}_{run_name}"
    run_dir = out_dir / data.group / data.feature_set / run_name
    table_dir = run_dir / "tables"
    fig_dir = run_dir / "figures"
    plot_data_dir = run_dir / "plot_data"
    for d in [table_dir, fig_dir, plot_data_dir]:
        d.mkdir(parents=True, exist_ok=True)

    if resume and not overwrite and is_run_complete(run_dir, run_id):
        print(f"[resume] skipping completed run: {run_id}", flush=True)
        return {"run_id": run_id, "skipped": True}

    X_run = data.X
    feature_types_run = data.feature_types

    print("\n" + "=" * 90, flush=True)
    print(f"RUN: {run_id}", flush=True)
    print(f"  X={X_run.shape}; labels={data.y.value_counts().to_dict()}; "
          f"n_repeats={len(seeds)}; n_permutations={n_permutations}", flush=True)
    print("=" * 90, flush=True)

    y_enc = (data.y.loc[X_run.index].values == POSITIVE_CLASS).astype(int)
    n_pos = int(y_enc.sum())
    n_neg = int(len(y_enc) - n_pos)
    if n_pos == 0 or n_neg == 0:
        raise ValueError(f"{run_id}: need both classes; got pos={n_pos}, neg={n_neg}")

    t0 = time.time()

    # (1) Repeated nested LOO-CV -> CI + importance SD; repeat 0 (base seed) is the
    #     "observed" model that the permutation test evaluates.
    oof_list, auc_list, details_list = repeated_nested_loo(
        X_run, y_enc, seeds=seeds, n_jobs=n_jobs, prescreen=prescreen,
        boruta_perc=boruta_perc, boruta_max_iter=boruta_max_iter,
        boruta_estimator_n=boruta_estimator_n, fallback_max_features=fallback_max_features,
        param_grid=param_grid,
    )
    roc_ci = compute_roc_ci(y_enc, oof_list)
    observed_auc = float(auc_list[0])
    base_seed = int(seeds[0])
    print(
        f"  mean AUROC={roc_ci['auc_mean']:.4f} 95% CI "
        f"[{roc_ci['auc_ci_low']:.4f}, {roc_ci['auc_ci_high']:.4f}] "
        f"(SD={roc_ci['auc_sd']:.4f}); observed (seed {base_seed}) AUROC={observed_auc:.4f}",
        flush=True,
    )

    # (2) Permutation significance of the base-seed model.
    p_val, null_aucs, n_null_ge = permutation_pvalue(
        X_run, y_enc, observed_auc=observed_auc, n_perm=n_permutations, n_jobs=n_jobs,
        prescreen=prescreen, boruta_perc=boruta_perc, boruta_max_iter=boruta_max_iter,
        boruta_estimator_n=boruta_estimator_n, fallback_max_features=fallback_max_features,
        param_grid=param_grid,
    )
    runtime_sec = float(time.time() - t0)
    print(f"  permutation p={p_val if not np.isnan(p_val) else 'NA'}; "
          f"runtime={runtime_sec:.1f}s", flush=True)

    settings = {
        "run_id": run_id,
        "group": data.group,
        "feature_set": data.feature_set,
        "run_name": run_name,
        "source_files": data.source_files,
        "label_col": LABEL_COL,
        "positive_class_encoded_1": POSITIVE_CLASS,
        "negative_class_encoded_0": NEGATIVE_CLASS,
        "ci_method": "t_based_95pct_CI_of_mean_across_seed_repeats",
        "permutation_test": "label_permutation_on_base_seed_model",
        "percentile_cutoffs": "not_run",
        "base_seed": base_seed,
        "seeds": [int(s) for s in seeds],
        "n_repeats": len(seeds),
        "n_permutations": int(n_permutations),
        "n_jobs": n_jobs,
        "threshold": threshold,
        "prescreen": prescreen,
        "boruta_available": HAS_BORUTA,
        "boruta_perc": boruta_perc,
        "boruta_max_iter": boruta_max_iter,
        "boruta_estimator_n": boruta_estimator_n,
        "fallback_max_features": fallback_max_features,
        "param_grid": param_grid,
    }
    (table_dir / f"{run_id}_Settings.json").write_text(json.dumps(settings, indent=2, default=str))

    # Mean out-of-fold probability across repeats drives predictions + confusion matrix.
    oof_matrix = np.vstack(oof_list)
    mean_proba = oof_matrix.mean(axis=0)
    sd_proba = oof_matrix.std(axis=0, ddof=1) if oof_matrix.shape[0] > 1 else np.zeros(oof_matrix.shape[1])

    pred_df, cm_df, cm_metrics = prediction_and_confusion_tables(
        samples=X_run.index, y_enc=y_enc, y_proba=mean_proba, threshold=threshold
    )
    pred_df["sd_predicted_probability_inefficiency"] = sd_proba
    for i, seed in enumerate(seeds):
        pred_df[f"predicted_probability_seed{seed}"] = oof_matrix[i]
    pred_df.insert(0, "run_id", run_id)
    pred_df.insert(1, "disease_group", data.group)
    pred_df.insert(2, "feature_set", data.feature_set)
    tsv(pred_df, table_dir / f"{run_id}_Predictions.tsv")

    cm_df.insert(0, "run_id", run_id)
    cm_df.insert(1, "disease_group", data.group)
    cm_df.insert(2, "feature_set", data.feature_set)
    tsv(cm_df, table_dir / f"{run_id}_ConfusionMatrix.tsv")

    # Per-repeat AUROC.
    per_auc_df = pd.DataFrame(
        {
            "run_id": run_id,
            "disease_group": data.group,
            "feature_set": data.feature_set,
            "repeat_index": np.arange(1, len(seeds) + 1, dtype=int),
            "seed": [int(s) for s in seeds],
            "auc": np.asarray(roc_ci["per_auc"], dtype=float),
        }
    )
    tsv(per_auc_df, table_dir / f"{run_id}_PerRepeatAUROC.tsv")

    # AUROC CI scalar summary.
    auc_ci_df = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "disease_group": data.group,
                "feature_set": data.feature_set,
                "n_repeats": int(roc_ci["n_repeats"]),
                "base_seed": base_seed,
                "seeds": ",".join(str(int(s)) for s in seeds),
                "mean_auc": roc_ci["auc_mean"],
                "sd_auc": roc_ci["auc_sd"],
                "sem_auc": roc_ci["auc_sem"],
                "t_critical_0975": roc_ci["tcrit"],
                "auc_ci95_low": roc_ci["auc_ci_low"],
                "auc_ci95_high": roc_ci["auc_ci_high"],
                "min_auc": float(np.min(roc_ci["per_auc"])),
                "max_auc": float(np.max(roc_ci["per_auc"])),
            }
        ]
    )
    tsv(auc_ci_df, table_dir / f"{run_id}_AUROC_CI.tsv")

    # Permutation tables.
    perm_df = pd.DataFrame(
        {
            "run_id": run_id,
            "disease_group": data.group,
            "feature_set": data.feature_set,
            "permutation_id": np.arange(1, len(null_aucs) + 1, dtype=int),
            "null_auc": null_aucs,
            "observed_auc": observed_auc,
            "p_value": p_val,
        }
    )
    tsv(perm_df, table_dir / f"{run_id}_Permutation.tsv")

    pval_df = pd.DataFrame(
        [
            {
                "run_id": run_id,
                "disease_group": data.group,
                "feature_set": data.feature_set,
                "base_seed": base_seed,
                "observed_auc": observed_auc,
                "n_permutations_requested": int(n_permutations),
                "n_permutations_completed": int(len(null_aucs)),
                "n_null_auc_ge_observed": int(n_null_ge),
                "p_value": p_val,
            }
        ]
    )
    tsv(pval_df, table_dir / f"{run_id}_PValue.tsv")

    # Feature importance with mean +/- SD across repeats.
    importance_df = build_repeated_importance_table(details_list, seeds, feature_types_run)
    importance_df.insert(0, "run_id", run_id)
    importance_df.insert(1, "disease_group", data.group)
    importance_df.insert(2, "feature_set", data.feature_set)
    importance_df.insert(3, "n_features_input_to_run", X_run.shape[1])
    tsv(importance_df, table_dir / f"{run_id}_FeatureImportance.tsv")

    # Optional per-fold traceability across all repeats (large tables).
    if write_fold_tables:
        fold_params_df, fold_selected_df = fold_tables_repeated(
            details_list, seeds, X_run.index, X_run.columns, feature_types_run, run_id
        )
        tsv(fold_params_df, table_dir / f"{run_id}_FoldParameters.tsv")
        tsv(fold_selected_df, table_dir / f"{run_id}_SelectedFeaturesByFold.tsv")

    # Figures + matching plot-data TSVs.
    save_roc_ci_plot(
        roc_ci=roc_ci, seeds=seeds, fig_dir=fig_dir, stem=f"{run_id}_AUROC",
        plot_data_path=plot_data_dir / f"{run_id}_AUROC.tsv",
        title=f"{data.group} {data.feature_set} {run_name}: AUROC (mean +/- 95% CI)",
        fig_format=fig_format,
    )
    save_importance_sd_plot(
        importance_df=importance_df, fig_dir=fig_dir, stem=f"{run_id}_ImportantScore",
        plot_data_path=plot_data_dir / f"{run_id}_ImportantScore.tsv",
        title=f"{data.group} {data.feature_set} {run_name}: importance (mean +/- SD)",
        top_n=top_n_plot, fig_format=fig_format,
    )
    save_confusion_matrix_plot(
        cm_df=cm_df, fig_dir=fig_dir, stem=f"{run_id}_ConfusionMatrix",
        plot_data_path=plot_data_dir / f"{run_id}_ConfusionMatrix.tsv",
        title=f"{data.group} {data.feature_set} {run_name}: confusion matrix (mean prob.)",
        fig_format=fig_format,
    )
    save_null_distribution_plot(
        null_aucs=null_aucs, observed_auc=observed_auc, p_val=p_val, fig_dir=fig_dir,
        stem=f"{run_id}_PermutationNullAUROC",
        plot_data_path=plot_data_dir / f"{run_id}_PermutationNullAUROC.tsv",
        title=f"{data.group} {data.feature_set} {run_name}: permutation null AUROC",
        fig_format=fig_format,
    )

    summary_row = {
        "run_id": run_id,
        "disease_group": data.group,
        "feature_set": data.feature_set,
        "run_name": run_name,
        "n_samples": int(X_run.shape[0]),
        "n_features_input_to_run": int(X_run.shape[1]),
        "n_inefficiency_positive": n_pos,
        "n_remission_negative": n_neg,
        "n_repeats": int(roc_ci["n_repeats"]),
        "base_seed": base_seed,
        "mean_auc": roc_ci["auc_mean"],
        "sd_auc": roc_ci["auc_sd"],
        "sem_auc": roc_ci["auc_sem"],
        "auc_ci95_low": roc_ci["auc_ci_low"],
        "auc_ci95_high": roc_ci["auc_ci_high"],
        "min_auc": float(np.min(roc_ci["per_auc"])),
        "max_auc": float(np.max(roc_ci["per_auc"])),
        "observed_auc": observed_auc,
        "n_permutations_completed": int(len(null_aucs)),
        "n_null_auc_ge_observed": int(n_null_ge),
        "p_value": p_val,
        "runtime_sec": runtime_sec,
        "prescreen": prescreen,
        "boruta_available": HAS_BORUTA,
    }
    summary_row.update(cm_metrics)
    tsv(pd.DataFrame([summary_row]), table_dir / f"{run_id}_Summary.tsv")

    (run_dir / "RUN_COMPLETE.ok").write_text(f"completed\t{time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    return {"run_id": run_id, "skipped": False}


def collect_global_outputs(out_dir: Path) -> None:
    table_patterns = {
        "all_runs_summary.tsv": "*_Summary.tsv",
        "all_runs_auroc_ci.tsv": "*_AUROC_CI.tsv",
        "all_runs_pvalues.tsv": "*_PValue.tsv",
        "all_runs_per_repeat_auroc.tsv": "*_PerRepeatAUROC.tsv",
        "all_runs_predictions.tsv": "*_Predictions.tsv",
        "all_runs_confusion_matrices.tsv": "*_ConfusionMatrix.tsv",
    }
    combined_dir = out_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    for out_name, pattern in table_patterns.items():
        paths = sorted(out_dir.glob(f"*/*/*/tables/{pattern}"))
        if not paths:
            continue
        frames = []
        for p in paths:
            try:
                frames.append(read_tsv(p))
            except Exception as exc:
                print(f"[collect warning] could not read {p}: {exc}", file=sys.stderr)
        if frames:
            tsv(pd.concat(frames, ignore_index=True), combined_dir / out_name)


def write_manifest(out_dir: Path, groups: Sequence[str], feature_sets: Sequence[str], data_dir: Path) -> None:
    rows = []
    inputs = get_input_files(data_dir)
    for group in groups:
        files = inputs[group]
        for feature_set in feature_sets:
            rows.append(
                {
                    "disease_group": group,
                    "feature_set": feature_set,
                    "pathway_file": files.pathway.name,
                    "pathway_file_exists": files.pathway.exists(),
                    "species_file": files.species.name,
                    "species_file_exists": files.species.exists(),
                }
            )
    (out_dir / "combined").mkdir(parents=True, exist_ok=True)
    tsv(pd.DataFrame(rows), out_dir / "combined" / "input_manifest.tsv")


def run_pipeline(args: argparse.Namespace) -> None:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    groups = args.groups
    feature_sets = args.feature_sets
    seeds = resolve_seeds(args)
    n_jobs = max(1, int(args.n_cores))

    param_grid = DEFAULT_PARAM_GRID
    if args.fast_grid:
        param_grid = {
            "n_estimators": [200],
            "max_depth": [None, 5],
            "min_samples_leaf": [1, 2],
            "max_features": ["sqrt"],
        }

    prescreen = "none" if args.no_boruta else args.prescreen
    resume = not args.no_resume

    print(f"[config] data_dir={data_dir}", flush=True)
    print(f"[config] out_dir={out_dir}", flush=True)
    print(f"[config] groups={groups}; feature_sets={feature_sets}", flush=True)
    print(f"[config] seeds={seeds[0]}..{seeds[-1]} (n_repeats={len(seeds)}); "
          f"n_permutations={args.n_permutations}", flush=True)
    print(f"[config] prescreen={prescreen}; boruta_available={HAS_BORUTA}; "
          f"n_cores={n_jobs}; fig_format={args.fig_format}", flush=True)

    write_manifest(out_dir, groups, feature_sets, data_dir)

    if args.dry_run:
        print("[dry-run] Checking input files and feature shapes only.", flush=True)
        for group in groups:
            for feature_set in feature_sets:
                load_feature_set(group, feature_set, data_dir)
        print("[dry-run] All requested inputs loaded successfully.", flush=True)
        return

    failed_runs: List[Dict[str, str]] = []
    for group in groups:
        for feature_set in feature_sets:
            try:
                data = load_feature_set(group, feature_set, data_dir)
                run_one_block(
                    data=data, seeds=seeds, out_dir=out_dir, n_jobs=n_jobs,
                    n_permutations=args.n_permutations, threshold=args.threshold,
                    top_n_plot=args.top_n_plot, prescreen=prescreen,
                    boruta_perc=args.boruta_perc, boruta_max_iter=args.boruta_max_iter,
                    boruta_estimator_n=args.boruta_estimator_n,
                    fallback_max_features=args.fallback_max_features, param_grid=param_grid,
                    fig_format=args.fig_format, write_fold_tables=args.write_fold_tables,
                    resume=resume, overwrite=args.overwrite,
                )
            except Exception as exc:
                failed_runs.append(
                    {
                        "disease_group": group,
                        "feature_set": feature_set,
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                print(f"[ERROR] {group} {feature_set}: {exc}", file=sys.stderr, flush=True)
                print(traceback.format_exc(), file=sys.stderr, flush=True)

    collect_global_outputs(out_dir)
    if failed_runs:
        tsv(pd.DataFrame(failed_runs), out_dir / "combined" / "FAILED_RUNS.tsv")
        raise SystemExit(
            f"Pipeline finished with {len(failed_runs)} failed blocks. See combined/FAILED_RUNS.tsv"
        )
    print(f"\nAll requested runs completed. Results: {out_dir.resolve()}", flush=True)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Nested-LOO random-forest MTX-response pipeline: permutation p-value "
            "on the base-seed model plus a repeated-seed t-based 95% AUROC CI and "
            "mean +/- SD feature importance. No percentile-cutoff gradient."
        )
    )
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="Directory with input TSV tables.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="Output directory for results.")
    parser.add_argument("--groups", nargs="+", default=["RA", "PsA"], choices=["RA", "PsA"])
    parser.add_argument(
        "--feature-sets", nargs="+", default=["pathway_only", "species_diversity"],
        choices=["pathway_only", "species_diversity", "pathway_diversity"],
        help="Feature sets to run (default: the two focus sets).",
    )
    parser.add_argument("--n-repeats", type=int, default=int(os.environ.get("RF_N_REPEATS", str(DEFAULT_N_REPEATS))),
                        help="Number of seed repeats for the CI (ignored if --seeds given).")
    parser.add_argument("--base-seed", type=int, default=int(os.environ.get("RF_BASE_SEED", str(DEFAULT_BASE_SEED))),
                        help="First seed; repeats use base_seed, base_seed+1, ...")
    parser.add_argument("--seeds", nargs="*", default=None, help="Explicit seed list; overrides --n-repeats/--base-seed.")
    parser.add_argument("--n-permutations", type=int, default=int(os.environ.get("RF_N_PERMUTATIONS", str(DEFAULT_N_PERMUTATIONS))),
                        help="Label permutations for the base-seed p-value (0 to skip).")
    parser.add_argument("--n-cores", type=int, default=int(os.environ.get("N_CORES", os.cpu_count() or 1)),
                        help="Parallel workers (across LOO folds / permutations).")
    parser.add_argument("--threshold", type=float, default=0.5, help="Probability threshold for the confusion matrix.")
    parser.add_argument("--top-n-plot", type=int, default=30, help="Number of features in the importance plot.")
    parser.add_argument("--boruta-perc", type=int, default=90)
    parser.add_argument("--boruta-max-iter", type=int, default=100)
    parser.add_argument("--boruta-estimator-n", type=int, default=200)
    parser.add_argument("--fallback-max-features", type=int, default=50,
                        help="Features kept by the RF-importance screen (rf_fallback / Boruta fallback).")
    parser.add_argument("--prescreen", choices=["rf_fallback", "boruta", "none"],
                        default=os.environ.get("RF_PRESCREEN", "rf_fallback"),
                        help="In-fold pre-screen. 'rf_fallback' (default) reproduces the primary models.")
    parser.add_argument("--no-boruta", action="store_true", help="Alias for --prescreen none.")
    parser.add_argument("--fig-format", choices=["svg", "png", "both"], default="svg",
                        help="Figure output format (default: svg).")
    parser.add_argument("--write-fold-tables", action="store_true",
                        help="Also write the large per-fold parameter/selection tables.")
    parser.add_argument("--fast-grid", action="store_true", help="Smaller RF tuning grid; useful for testing.")
    parser.add_argument("--dry-run", action="store_true", help="Load inputs and print shapes without running models.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite completed runs.")
    parser.add_argument("--no-resume", action="store_true", help="Do not skip completed runs.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = build_arg_parser().parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":
    main()
