
"""
Phase 6.4: Tube bootstrap and one-standard-error model selection.

Input:
    results/model_ladder_oof.csv

Outputs:
    results/tube_bootstrap_summary.csv
    results/tube_bootstrap_samples.csv
    results/model_selection_one_se.csv
    reports/model_selection_6_4.md
    reports/final_candidate.json

Only training-tube OOF predictions are used.
Validation and test data are not accessed.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

OOF_PATH = ROOT / "results/model_ladder_oof.csv"

BOOTSTRAP_SUMMARY_PATH = ROOT / "results/tube_bootstrap_summary.csv"
BOOTSTRAP_SAMPLES_PATH = ROOT / "results/tube_bootstrap_samples.csv"
SELECTION_PATH = ROOT / "results/model_selection_one_se.csv"

REPORT_PATH = ROOT / "reports/model_selection_6_4.md"
CANDIDATE_PATH = ROOT / "reports/final_candidate.json"

N_BOOTSTRAP = 2000
RANDOM_SEED = 0

# Predeclared simplicity order.
SIMPLICITY_ORDER = [
    "B0",
    "B1",
    "Polynomial_Ridge_b_med",
    "Ridge_S2_scratch",
    "SVR_RBF_S2",
    "Random_Forest_S2",
    "Gradient_Boosting_S2",
    "MLP_S2",
]


def validate_oof_data(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "image_id",
        "tube_id",
        "model",
        "actual",
        "predicted",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"OOF file is missing columns: {sorted(missing)}"
        )

    if df[list(required)].isna().any().any():
        raise ValueError("OOF file contains missing required values")

    if df.duplicated(["model", "image_id"]).any():
        raise ValueError("Duplicate image/model OOF records")

    df = df.copy()

    df["actual"] = pd.to_numeric(df["actual"], errors="raise")
    df["predicted"] = pd.to_numeric(
        df["predicted"],
        errors="raise",
    )

    df["absolute_error"] = np.abs(
        df["predicted"] - df["actual"]
    )

    expected_models = set(SIMPLICITY_ORDER)
    observed_models = set(df["model"].astype(str))

    missing_models = expected_models - observed_models
    unexpected_models = observed_models - expected_models

    if missing_models:
        raise ValueError(
            f"Missing models in OOF file: {sorted(missing_models)}"
        )

    if unexpected_models:
        raise ValueError(
            f"Unexpected models in OOF file: {sorted(unexpected_models)}"
        )

    # Every model must have predictions for exactly the same images.
    image_sets = {
        model: set(
            group["image_id"].astype(str)
        )
        for model, group in df.groupby("model")
    }

    reference_images = image_sets[SIMPLICITY_ORDER[0]]

    for model, images in image_sets.items():
        if images != reference_images:
            raise ValueError(
                f"{model} has a different OOF image set"
            )

    # Actual values and tube IDs must agree across models for each image.
    reference = (
        df[df["model"] == SIMPLICITY_ORDER[0]]
        .set_index("image_id")[["tube_id", "actual"]]
        .sort_index()
    )

    for model in SIMPLICITY_ORDER[1:]:
        current = (
            df[df["model"] == model]
            .set_index("image_id")[["tube_id", "actual"]]
            .sort_index()
        )

        if not reference["tube_id"].equals(current["tube_id"]):
            raise ValueError(
                f"Tube assignments differ for model {model}"
            )

        np.testing.assert_allclose(
            reference["actual"].to_numpy(dtype=float),
            current["actual"].to_numpy(dtype=float),
            atol=1e-12,
            rtol=0,
        )

    # If fold assignments exist, confirm they match across models.
    if "fold" in df.columns:
        reference_folds = (
            df[df["model"] == SIMPLICITY_ORDER[0]]
            .set_index("image_id")["fold"]
            .sort_index()
        )

        for model in SIMPLICITY_ORDER[1:]:
            current_folds = (
                df[df["model"] == model]
                .set_index("image_id")["fold"]
                .sort_index()
            )

            if not reference_folds.equals(current_folds):
                raise ValueError(
                    f"Outer fold assignments differ for {model}"
                )

    return df


def build_tube_table(model_frame: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate absolute errors per physical tube.

    Keep both sum and count so a bootstrap resample reproduces
    the image-pooled MAE used in the original evaluation.
    """

    per_tube = (
        model_frame.groupby("tube_id")["absolute_error"]
        .agg(error_sum="sum", image_count="count")
        .sort_index()
    )

    if len(per_tube) != 31:
        raise ValueError(
            f"Expected 31 training tubes; found {len(per_tube)}"
        )

    if (per_tube["image_count"] <= 0).any():
        raise ValueError("A tube has no OOF image predictions")

    return per_tube


def main():
    if not OOF_PATH.is_file():
        raise FileNotFoundError(OOF_PATH)

    df = pd.read_csv(OOF_PATH)
    df = validate_oof_data(df)

    models = SIMPLICITY_ORDER

    print("PHASE 6.4: TUBE BOOTSTRAP AND ONE-SELECTION")
    print("=" * 72)
    print("OOF images:", df["image_id"].nunique())
    print("Training tubes:", df["tube_id"].nunique())
    print("Models:", len(models))
    print("Bootstrap repetitions:", N_BOOTSTRAP)
    print("Random seed:", RANDOM_SEED)

    # Pooled OOF MAE for each model.
    pooled_mae = {}

    # Per-tube error totals, ordered identically for every model.
    tube_tables = {}

    reference_tubes = None

    for model in models:
        model_frame = df[df["model"] == model].copy()

        pooled_mae[model] = float(
            model_frame["absolute_error"].mean()
        )

        per_tube = build_tube_table(model_frame)

        if reference_tubes is None:
            reference_tubes = per_tube.index.tolist()
        elif per_tube.index.tolist() != reference_tubes:
            raise ValueError(
                f"Tube order/set differs for model {model}"
            )

        tube_tables[model] = per_tube

    n_tubes = len(reference_tubes)

    # Generate one common set of tube resamples.
    # Reusing the same draws across models makes later
    # paired bootstrap differences possible.
    rng = np.random.default_rng(RANDOM_SEED)

    draws = rng.integers(
        low=0,
        high=n_tubes,
        size=(N_BOOTSTRAP, n_tubes),
    )

    summary_rows = []
    sample_rows = []

    for model in models:
        per_tube = tube_tables[model]

        sums = per_tube["error_sum"].to_numpy(dtype=float)
        counts = per_tube["image_count"].to_numpy(dtype=float)

        # Bootstrap resamples tubes, not individual photographs.
        sampled_sums = sums[draws].sum(axis=1)
        sampled_counts = counts[draws].sum(axis=1)

        bootstrap_mae = sampled_sums / sampled_counts

        bootstrap_se = float(
            np.std(bootstrap_mae, ddof=1)
        )

        ci_low, ci_high = np.percentile(
            bootstrap_mae,
            [2.5, 97.5],
        )

        summary_rows.append({
            "model": model,
            "oof_mae": pooled_mae[model],
            "bootstrap_mean_mae": float(bootstrap_mae.mean()),
            "bootstrap_se": bootstrap_se,
            "bootstrap_ci_2_5": float(ci_low),
            "bootstrap_ci_97_5": float(ci_high),
            "n_bootstrap": N_BOOTSTRAP,
            "n_tubes": n_tubes,
            "n_images": int(per_tube["image_count"].sum()),
            "seed": RANDOM_SEED,
        })

        for replicate, value in enumerate(bootstrap_mae, start=1):
            sample_rows.append({
                "model": model,
                "bootstrap_replicate": replicate,
                "bootstrap_mae": float(value),
            })

    bootstrap_summary = pd.DataFrame(summary_rows)

    # Best model is the one with lowest pooled outer OOF MAE.
    best_row = bootstrap_summary.loc[
        bootstrap_summary["oof_mae"].idxmin()
    ]

    best_model = str(best_row["model"])
    best_mae = float(best_row["oof_mae"])
    best_se = float(best_row["bootstrap_se"])

    one_se_cutoff = best_mae + best_se

    bootstrap_summary["one_se_cutoff"] = one_se_cutoff

    bootstrap_summary["within_one_se"] = (
        bootstrap_summary["oof_mae"] <= one_se_cutoff
    )

    simplicity_rank = {
        model: rank
        for rank, model in enumerate(models, start=1)
    }

    bootstrap_summary["simplicity_rank"] = (
        bootstrap_summary["model"].map(simplicity_rank)
    )

    # Choose the simplest eligible model.
    eligible = bootstrap_summary[
        bootstrap_summary["within_one_se"]
    ].copy()

    selected_row = eligible.sort_values(
        ["simplicity_rank", "oof_mae"]
    ).iloc[0]

    selected_model = str(selected_row["model"])

    bootstrap_summary["selected"] = (
        bootstrap_summary["model"] == selected_model
    )

    bootstrap_summary = bootstrap_summary.sort_values(
        ["simplicity_rank"]
    ).reset_index(drop=True)

    # Save outputs.
    BOOTSTRAP_SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    bootstrap_summary.to_csv(
        BOOTSTRAP_SUMMARY_PATH,
        index=False,
    )

    pd.DataFrame(sample_rows).to_csv(
        BOOTSTRAP_SAMPLES_PATH,
        index=False,
    )

    bootstrap_summary.to_csv(
        SELECTION_PATH,
        index=False,
    )

    selection_record = {
        "selection_rule": "one_standard_error",
        "bootstrap_repetitions": N_BOOTSTRAP,
        "bootstrap_seed": RANDOM_SEED,
        "resampling_unit": "tube_id",
        "best_by_oof_mae": best_model,
        "best_oof_mae": best_mae,
        "best_bootstrap_se": best_se,
        "one_se_cutoff": one_se_cutoff,
        "selected_simplest_eligible_model": selected_model,
        "simplicity_order": models,
        "validation_used": False,
        "test_used": False,
    }

    CANDIDATE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    CANDIDATE_PATH.write_text(
        json.dumps(selection_record, indent=2),
        encoding="utf-8",
    )

    # Generate a concise human-readable report.
    report_lines = [
        "# Phase 6.4 — One-standard-error model selection",
        "",
        f"- Training images: {df['image_id'].nunique()}",
        f"- Training tubes: {n_tubes}",
        f"- Bootstrap repetitions: {N_BOOTSTRAP}",
        f"- Bootstrap seed: {RANDOM_SEED}",
        "- Resampling unit: physical tube (`tube_id`)",
        "- Validation used: no",
        "- Test used: no",
        "",
        "## Bootstrap results",
        "",
        "| Model | OOF MAE | Bootstrap SE | 95% CI | Within one SE | Selected |",
        "|---|---:|---:|---:|---|---|",
    ]

    for _, row in bootstrap_summary.iterrows():
        ci = (
            f"[{row['bootstrap_ci_2_5']:.4f}, "
            f"{row['bootstrap_ci_97_5']:.4f}]"
        )

        report_lines.append(
            f"| {row['model']} "
            f"| {row['oof_mae']:.6f} "
            f"| {row['bootstrap_se']:.6f} "
            f"| {ci} "
            f"| {'Yes' if row['within_one_se'] else 'No'} "
            f"| {'**Yes**' if row['selected'] else 'No'} |"
        )

    report_lines.extend([
        "",
        "## One-standard-error calculation",
        "",
        f"- Best model by OOF MAE: **{best_model}**",
        f"- Best OOF MAE: {best_mae:.6f}",
        f"- Bootstrap standard error of best model: {best_se:.6f}",
        f"- Cutoff: {best_mae:.6f} + {best_se:.6f} = **{one_se_cutoff:.6f}**",
        f"- Selected simplest eligible model: **{selected_model}**",
        "",
        "Selection uses the predeclared simplicity order:",
        "",
    ])

    for rank, model in enumerate(models, start=1):
        report_lines.append(f"{rank}. {model}")

    report_lines.extend([
        "",
        "## Interpretation",
        "",
        "This is a development-only model candidate selected by the "
        "one-standard-error rule. It is not final test performance. "
        "The 10 validation tubes remain reserved for candidate assessment, "
        "and the 10 test tubes remain locked.",
        "",
        "Bootstrap uncertainty is estimated by resampling physical tubes "
        "with replacement and recomputing image-pooled MAE from per-tube "
        "error sums and image counts.",
    ])

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    print("\nBOOTSTRAP SUMMARY")
    print("=" * 100)

    print(
        bootstrap_summary[
            [
                "model",
                "oof_mae",
                "bootstrap_se",
                "bootstrap_ci_2_5",
                "bootstrap_ci_97_5",
                "within_one_se",
                "selected",
            ]
        ].to_string(index=False)
    )

    print("\nONE-STANDARD-ERROR RULE")
    print("=" * 72)
    print("Best model:", best_model)
    print(f"Best OOF MAE: {best_mae:.6f}")
    print(f"Bootstrap SE: {best_se:.6f}")
    print(f"One-SE cutoff: {one_se_cutoff:.6f}")
    print("Selected simplest eligible model:", selected_model)

    print("\nOUTPUTS")
    print("=" * 72)
    print("Bootstrap summary:", BOOTSTRAP_SUMMARY_PATH.relative_to(ROOT))
    print("Bootstrap samples:", BOOTSTRAP_SAMPLES_PATH.relative_to(ROOT))
    print("One-SE table:", SELECTION_PATH.relative_to(ROOT))
    print("Report:", REPORT_PATH.relative_to(ROOT))
    print("Candidate config:", CANDIDATE_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
