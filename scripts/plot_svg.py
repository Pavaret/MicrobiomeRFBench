#!/usr/bin/env python3
"""
Regenerate the pipeline figures as SVG from the plot-ready TSV files.

The pipeline already writes SVG figures directly, but every figure also has a
matching plot-data TSV (same file stem). This script rebuilds the SVGs from those
TSVs, so you can restyle a plot (e.g. draw a +/- 1 SD band instead of the t-based
CI, or change the top-N of the importance plot) without re-running the models.

It walks a directory tree, detects the four plot types by filename suffix and
writes one SVG per file, preserving the relative folder layout under --out-dir:

    *_AUROC.tsv                 -> mean ROC + 95% CI band (+ faint per-seed curves)
    *_ImportantScore.tsv        -> top-N features, mean +/- SD bars
    *_ConfusionMatrix.tsv       -> 2x2 confusion matrix
    *_PermutationNullAUROC.tsv  -> permutation null-AUROC histogram

Usage
-----
    python plot_svg.py --plot-data-dir ../results --out-dir ../results/figures_gallery
    python plot_svg.py --plot-data-dir ../results/RA/species_diversity/all_features/plot_data \
                       --out-dir /tmp/svg --top-n-importance 20
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


NEGATIVE_CLASS = "remission"
POSITIVE_CLASS = "inefficiency"
CLASS_LABELS = [NEGATIVE_CLASS, POSITIVE_CLASS]


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


def infer_title_from_stem(stem: str, plot_type: str) -> str:
    """Turn ``RA_species_diversity_all_features_AUROC`` into a readable title."""
    base = stem[: -(len(plot_type) + 1)] if stem.endswith(f"_{plot_type}") else stem
    parts = base.split("_")
    group = parts[0] if parts else ""
    run_name = "all_features" if "all_features" in base else ""
    if run_name:
        feature_set = base.replace(f"{group}_", "", 1).replace(f"_{run_name}", "").strip("_")
    else:
        feature_set = base.replace(f"{group}_", "", 1)

    titles = {
        "AUROC": f"{group} {feature_set} {run_name}: AUROC (mean +/- 95% CI)",
        "ImportantScore": f"{group} {feature_set} {run_name}: importance (mean +/- SD)",
        "ConfusionMatrix": f"{group} {feature_set} {run_name}: confusion matrix (mean prob.)",
        "PermutationNullAUROC": f"{group} {feature_set} {run_name}: permutation null AUROC",
    }
    return titles.get(plot_type, stem).strip()


def save_svg(fig, svg_path: Path) -> None:
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Per-plot renderers
# --------------------------------------------------------------------------- #

def render_auroc(df: pd.DataFrame, svg_path: Path, title: str) -> None:
    required = {"mean_fpr", "mean_tpr", "tpr_ci95_lower", "tpr_ci95_upper"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing AUROC columns: {sorted(missing)}")

    mean_fpr = df["mean_fpr"].to_numpy(float)
    mean_tpr = df["mean_tpr"].to_numpy(float)
    lower = df["tpr_ci95_lower"].to_numpy(float)
    upper = df["tpr_ci95_upper"].to_numpy(float)
    seed_cols = [c for c in df.columns if c.startswith("tpr_seed")]
    auc_mean = float(np.trapz(mean_tpr, mean_fpr))

    fig, ax = plt.subplots(figsize=(6.2, 6.2))
    for col in seed_cols:
        ax.plot(mean_fpr, df[col].to_numpy(float), color="tab:blue", linewidth=0.8, alpha=0.25)
    ax.fill_between(mean_fpr, lower, upper, color="tab:blue", alpha=0.25,
                    linewidth=0, label="95% CI (t-based)")
    label = f"Mean AUROC = {auc_mean:.3f}"
    if seed_cols:
        label += f"\n({len(seed_cols)} seed repeats)"
    ax.plot(mean_fpr, mean_tpr, color="tab:blue", linewidth=2.3, label=label)
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
    save_svg(fig, svg_path)


def render_importance(df: pd.DataFrame, svg_path: Path, title: str, top_n: Optional[int]) -> None:
    required = {"feature", "mean_importance_score", "sd_importance_score"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing importance columns: {sorted(missing)}")

    df = df.copy()
    if "rank_within_run" in df.columns:
        df = df.sort_values("rank_within_run")
    else:
        df = df.sort_values("mean_importance_score", ascending=False)
    if top_n is not None and top_n > 0:
        df = df.head(top_n)
    df = df.iloc[::-1].reset_index(drop=True)

    height = max(5.2, 0.34 * len(df) + 2.0)
    fig, ax = plt.subplots(figsize=(11.0, height))
    ax.barh(
        df["feature"],
        df["mean_importance_score"].to_numpy(float),
        xerr=df["sd_importance_score"].to_numpy(float),
        capsize=3,
        error_kw={"elinewidth": 1.0, "capthick": 1.0, "ecolor": "black"},
    )
    ax.set_xlabel("Stability-weighted RF importance (mean +/- SD across repeats)", fontsize=13)
    ax.set_ylabel("Feature", fontsize=14)
    ax.set_title(title, fontsize=14)
    ax.set_xlim(left=0.0)
    ax.set_ylim(-0.5, len(df) - 0.5)
    set_plot_grid(ax)
    fig.tight_layout()
    save_svg(fig, svg_path)


def render_confusion(df: pd.DataFrame, svg_path: Path, title: str) -> None:
    required = {"true_label", "predicted_label", "count"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing confusion-matrix columns: {sorted(missing)}")

    def count(t: str, p: str) -> int:
        row = df[(df["true_label"] == t) & (df["predicted_label"] == p)]
        return 0 if row.empty else int(row["count"].iloc[0])

    matrix = np.array(
        [[count(NEGATIVE_CLASS, NEGATIVE_CLASS), count(NEGATIVE_CLASS, POSITIVE_CLASS)],
         [count(POSITIVE_CLASS, NEGATIVE_CLASS), count(POSITIVE_CLASS, POSITIVE_CLASS)]],
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
    save_svg(fig, svg_path)


def render_permutation_null(df: pd.DataFrame, svg_path: Path, title: str) -> None:
    required = {"bin_left", "bin_right", "count"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing permutation-null columns: {sorted(missing)}")
    if df.empty:
        return

    left = df["bin_left"].to_numpy(float)
    right = df["bin_right"].to_numpy(float)
    counts = df["count"].to_numpy(float)
    widths = right - left

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.bar(left, counts, width=widths, align="edge", color="tab:blue", alpha=0.85)
    label = None
    if {"observed_auc", "p_value"}.issubset(df.columns):
        obs = float(df["observed_auc"].iloc[0])
        pval = float(df["p_value"].iloc[0])
        ax.axvline(obs, linewidth=2.0, linestyle="--", color="tab:red",
                   label=f"Observed AUROC = {obs:.3f}\npermutation p = {pval:.4f}")
        label = True
    ax.set_xlabel("AUROC under permuted labels", fontsize=14)
    ax.set_ylabel("Count", fontsize=14)
    ax.set_title(title, fontsize=14)
    if label:
        ax.legend(frameon=False, fontsize=12)
    set_plot_grid(ax)
    fig.tight_layout()
    save_svg(fig, svg_path)


# --------------------------------------------------------------------------- #
# Detection + batch driver
# --------------------------------------------------------------------------- #

SUFFIXES = ["AUROC", "ImportantScore", "ConfusionMatrix", "PermutationNullAUROC"]


def detect_plot_type(path: Path) -> Optional[str]:
    for suffix in SUFFIXES:  # longest-specific first is not needed: names are unique
        if path.name.endswith(f"_{suffix}.tsv"):
            return suffix
    return None


def output_path_for(input_tsv: Path, input_root: Path, out_dir: Path, flatten: bool) -> Path:
    if flatten:
        return out_dir / input_tsv.with_suffix(".svg").name
    try:
        rel = input_tsv.relative_to(input_root)
    except ValueError:
        rel = Path(input_tsv.name)
    return out_dir / rel.with_suffix(".svg")


def replot_one(input_tsv: Path, input_root: Path, out_dir: Path,
               top_n_importance: Optional[int], flatten: bool) -> Optional[Path]:
    plot_type = detect_plot_type(input_tsv)
    if plot_type is None:
        return None
    df = pd.read_csv(input_tsv, sep="\t")
    svg_path = output_path_for(input_tsv, input_root, out_dir, flatten)
    title = infer_title_from_stem(input_tsv.stem, plot_type)

    if plot_type == "AUROC":
        render_auroc(df, svg_path, title)
    elif plot_type == "ImportantScore":
        render_importance(df, svg_path, title, top_n_importance)
    elif plot_type == "ConfusionMatrix":
        render_confusion(df, svg_path, title)
    elif plot_type == "PermutationNullAUROC":
        render_permutation_null(df, svg_path, title)
    return svg_path


def find_plot_data_files(root: Path) -> List[Path]:
    return sorted(root.rglob("*.tsv"))


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Regenerate pipeline figures as SVG from plot-data TSVs.")
    parser.add_argument("--plot-data-dir", required=True,
                        help="Result root or a specific plot_data folder to scan (recursively).")
    parser.add_argument("--out-dir", required=True, help="Directory for the SVG files.")
    parser.add_argument("--top-n-importance", type=int, default=None,
                        help="Top-N features for importance plots (default: as saved in the TSV).")
    parser.add_argument("--flatten", action="store_true",
                        help="Write all SVGs into --out-dir directly instead of mirroring the tree.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_root = Path(args.plot_data_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_root}")

    written, skipped = [], []
    for tsv_path in find_plot_data_files(input_root):
        svg_path = replot_one(tsv_path, input_root, out_dir, args.top_n_importance, args.flatten)
        if svg_path is None:
            skipped.append(tsv_path)
        else:
            written.append(svg_path)
            print(f"[svg] {svg_path}")

    print(f"\nSVG files written: {len(written)}")
    print(f"TSV files skipped: {len(skipped)}")
    if not written:
        print("\nNo supported plot-data files found. Expected names ending with:")
        for s in SUFFIXES:
            print(f"  *_{s}.tsv")


if __name__ == "__main__":
    main()
