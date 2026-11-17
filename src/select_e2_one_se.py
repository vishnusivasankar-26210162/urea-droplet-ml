
"""
Reapply the one-standard-error rule to all 20 E2 configurations.

Input:
    results/e2_feature_ablation_oof.csv
    data/processed/feature_set_columns.json

Outputs:
    results/e2_tube_bootstrap_summary.csv
    results/e2_tube_bootstrap_samples.csv
    results/e2_one_se_selection.csv
    reports/e2_one_se_selection.md
    reports/e2_candidate.json

Resampling unit: physical tube.
All configurations use identical bootstrap tube draws.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

OOF_PATH = ROOT / "results/e2_feature_ablation_oof.csv"
REGISTRY_PATH = ROOT / "data/processed/feature_set_columns.json"

SUMMARY_PATH = ROOT / "results/e2_tube_bootstrap_summary.csv"
SAMPLES_PATH = ROOT / "results/e2_tube_bootstrap_samples.csv"
SELECTION_PATH = ROOT / "results/e2_one_se_selection.csv"

REPORT_PATH = ROOT / "reports/e2_one_se_selection.md"
CANDIDATE_PATH = ROOT / "reports/e2_candidate.json"

N_BOOTSTRAP = 2000
SEED = 0

MODELS = ["Ridge", "SVR"]
FEATURE_SETS = ["S0", "S1", "S2", "S3", "S4"]
VARIANTS = ["colour_only", "with_device_and_lamp"]


def load_inputs():
    if not OOF_PATH.is_file():
        raise FileNotFoundError(OOF_PATH)

    if not REGISTRY_PATH.is_file():
        raise FileNotFoundError(REGISTRY_PATH)

    oof = pd.read_csv(OOF_PATH)

    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)

    required = {
        "image_id",
        "tube_id",
        "model",
        "configuration",
        "actual",
        "predicted",
    }

    missing = required - set(oof.columns)

    if missing:
        raise ValueError(
            f"E2 OOF file missing columns: {sorted(missing)}"
        )

    if oof[list(required)].isna().any().any():
        raise ValueError("E2 OOF file has missing required values")

    oof = oof.copy()
    oof["image_id"] = oof["image_id"].astype(str)
    oof["tube_id"] = oof["tube_id"].astype(str)
    oof["actual"] = pd.to_numeric(oof["actual"], errors="raise")
    oof["predicted"] = pd.to_numeric(
        oof["predicted"],
        errors="raise",
    )

    oof["absolute_error"] = np.abs(
        oof["actual"] - oof["predicted"]
    )

    if oof.duplicated(
        ["model", "configuration", "image_id"]
    ).any():
        raise ValueError("Duplicate image/configuration prediction")

    expected_pairs = {
        (model, f"{feature_set}_{variant}")
        for model in MODELS
        for feature_set in FEATURE_SETS
        for variant in VARIANTS
    }

    observed_pairs = set(
        zip(
            oof["model"].astype(str),
            oof["configuration"].astype(str),
        )
    )

    if expected_pairs != observed_pairs:
        raise ValueError(
            "E2 model/configuration combinations do not match "
            "the expected 20 combinations.\n"
            f"Missing: {sorted(expected_pairs - observed_pairs)}\n"
            f"Unexpected: {sorted(observed_pairs - expected_pairs)}"
        )

    # Every configuration must predict the same images.
    image_sets = {}

    for (model, config), group in oof.groupby(
        ["model", "configuration"]
    ):
        key = (str(model), str(config))
        image_sets[key] = set(group["image_id"])

    reference_images = next(iter(image_sets.values()))

    for key, image_ids in image_sets.items():
        if image_ids != reference_images:
            raise ValueError(
                f"Different OOF image set for {key}"
            )

    # Tube assignments and actual labels must agree across configurations.
    reference_key = ("Ridge", "S0_colour_only")

    reference = (
        oof[
            (oof["model"] == reference_key[0])
            & (oof["configuration"] == reference_key[1])
        ]
        .set_index("image_id")
        .sort_index()
    )

    for (model, config), group in oof.groupby(
        ["model", "configuration"]
    ):
        current = group.set_index("image_id").sort_index()

        if not reference["tube_id"].equals(current["tube_id"]):
            raise ValueError(
                f"Tube assignment mismatch in {model}/{config}"
            )

        np.testing.assert_allclose(
            reference["actual"].to_numpy(dtype=float),
            current["actual"].to_numpy(dtype=float),
            atol=1e-12,
            rtol=0,
        )

        if "fold" in oof.columns:
            if not reference["fold"].equals(current["fold"]):
                raise ValueError(
                    f"Fold assignment mismatch in {model}/{config}"
                )

    tube_count = reference["tube_id"].nunique()

    if tube_count != 31:
        raise ValueError(
            f"Expected 31 training tubes, found {tube_count}"
        )

    return oof, registry, reference_images, tube_count


def parse_configuration(configuration):
    parts = str(configuration).split("_", 1)

    if len(parts) != 2:
        raise ValueError(
            f"Unexpected E2 configuration name: {configuration}"
        )

    feature_set, variant = parts

    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature set: {feature_set}")

    if variant not in VARIANTS:
        raise ValueError(f"Unknown feature variant: {variant}")

    return feature_set, variant


def main():
    oof, registry, expected_images, n_tubes = load_inputs()

    print("E2 ONE-STANDARD-ERROR FOLLOW-UP")
    print("=" * 72)
    print("Configurations:", 20)
    print("OOF images:", len(expected_images))
    print("Training tubes:", n_tubes)
    print("Bootstrap repetitions:", N_BOOTSTRAP)
    print("Seed:", SEED)

    # Establish one common tube order for every configuration.
    first = oof[
        (oof["model"] == "Ridge")
        & (oof["configuration"] == "S0_colour_only")
    ]

    tube_order = sorted(first["tube_id"].unique().tolist())

    rng = np.random.default_rng(SEED)

    # One set of tube draws shared by all 20 configurations.
    draws = rng.integers(
        low=0,
        high=len(tube_order),
        size=(N_BOOTSTRAP, len(tube_order)),
    )

    summary_rows = []
    sample_rows = []

    for (model, configuration), group in oof.groupby(
        ["model", "configuration"],
        sort=False,
    ):
        model = str(model)
        configuration = str(configuration)

        feature_set, variant = parse_configuration(configuration)

        feature_columns = registry[feature_set][variant]
        n_features = len(feature_columns)

        per_tube = (
            group.groupby("tube_id")["absolute_error"]
            .agg(error_sum="sum", image_count="count")
            .reindex(tube_order)
        )

        if per_tube.isna().any().any():
            raise ValueError(
                f"Missing tube error aggregates in "
                f"{model}/{configuration}"
            )

        errors = per_tube["error_sum"].to_numpy(dtype=float)
        counts = per_tube["image_count"].to_numpy(dtype=float)

        pooled_mae = float(group["absolute_error"].mean())

        sampled_error_sums = errors[draws].sum(axis=1)
        sampled_image_counts = counts[draws].sum(axis=1)

        bootstrap_mae = (
            sampled_error_sums / sampled_image_counts
        )

        bootstrap_se = float(
            np.std(bootstrap_mae, ddof=1)
        )

        ci_low, ci_high = np.percentile(
            bootstrap_mae,
            [2.5, 97.5],
        )

        configuration_name = f"{model}__{configuration}"

        summary_rows.append({
            "configuration_name": configuration_name,
            "model": model,
            "feature_set": feature_set,
            "variant": variant,
            "n_features": n_features,
            "oof_mae": pooled_mae,
            "bootstrap_mean_mae": float(bootstrap_mae.mean()),
            "bootstrap_se": bootstrap_se,
            "bootstrap_ci_2_5": float(ci_low),
            "bootstrap_ci_97_5": float(ci_high),
            "n_bootstrap": N_BOOTSTRAP,
            "n_tubes": n_tubes,
            "n_images": len(group),
            "seed": SEED,
        })

        for replicate, value in enumerate(bootstrap_mae, start=1):
            sample_rows.append({
                "configuration_name": configuration_name,
                "model": model,
                "configuration": configuration,
                "bootstrap_replicate": replicate,
                "bootstrap_mae": float(value),
            })

    summary = pd.DataFrame(summary_rows)

    if len(summary) != 20:
        raise RuntimeError(
            f"Expected 20 configurations, got {len(summary)}"
        )

    # Best raw OOF MAE and its bootstrap standard error.
    best = summary.loc[summary["oof_mae"].idxmin()]

    best_mae = float(best["oof_mae"])
    best_se = float(best["bootstrap_se"])
    cutoff = best_mae + best_se

    summary["one_se_cutoff"] = cutoff
    summary["within_one_se"] = summary["oof_mae"] <= cutoff

    # Predeclared simplicity order:
    # 1. Ridge before RBF-SVR.
    # 2. Fewer input columns.
    # 3. Colour-only before device/lamp when counts tie.
    # 4. Lower OOF MAE as tie-breaker.
    summary["model_complexity_rank"] = (
        summary["model"].map({
            "Ridge": 0,
            "SVR": 1,
        })
    )

    summary["variant_rank"] = (
        summary["variant"].map({
            "colour_only": 0,
            "with_device_and_lamp": 1,
        })
    )

    eligible = summary[
        summary["within_one_se"]
    ].copy()

    if eligible.empty:
        raise RuntimeError("No E2 configuration is within one SE")

    selected = eligible.sort_values(
        [
            "model_complexity_rank",
            "n_features",
            "variant_rank",
            "oof_mae",
        ],
        ascending=True,
    ).iloc[0]

    selected_name = str(selected["configuration_name"])

    summary["selected"] = (
        summary["configuration_name"] == selected_name
    )

    summary = summary.sort_values(
        [
            "within_one_se",
            "model_complexity_rank",
            "n_features",
            "variant_rank",
            "oof_mae",
        ],
        ascending=[False, True, True, True, True],
    ).reset_index(drop=True)

    # Save tables.
    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(SUMMARY_PATH, index=False)

    pd.DataFrame(sample_rows).to_csv(
        SAMPLES_PATH,
        index=False,
    )

    summary.to_csv(
        SELECTION_PATH,
        index=False,
    )

    candidate = {
        "selection_rule": "one_standard_error_across_E2",
        "selected_configuration": selected_name,
        "model": str(selected["model"]),
        "feature_set": str(selected["feature_set"]),
        "variant": str(selected["variant"]),
        "feature_columns": registry[
            str(selected["feature_set"])
        ][str(selected["variant"])],
        "n_features": int(selected["n_features"]),
        "best_raw_configuration": str(best["configuration_name"]),
        "best_raw_oof_mae": best_mae,
        "best_raw_bootstrap_se": best_se,
        "one_se_cutoff": cutoff,
        "n_bootstrap": N_BOOTSTRAP,
        "seed": SEED,
        "resampling_unit": "tube_id",
        "n_tubes": n_tubes,
        "n_images": len(expected_images),
        "validation_used": False,
        "test_used": False,
    }

    CANDIDATE_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    CANDIDATE_PATH.write_text(
        json.dumps(candidate, indent=2),
        encoding="utf-8",
    )

    # Write concise report.
    lines = [
        "# E2 One-Standard-Error Selection",
        "",
        f"- Configurations compared: {len(summary)}",
        f"- Training images: {len(expected_images)}",
        f"- Training tubes: {n_tubes}",
        f"- Bootstrap repetitions: {N_BOOTSTRAP}",
        f"- Seed: {SEED}",
        "- Resampling unit: physical tube",
        "- Same bootstrap draws used for all configurations",
        "- Validation used: no",
        "- Test used: no",
        "",
        "## Results",
        "",
        "| Configuration | Features | OOF MAE | Bootstrap SE | 95% interval | Within one SE | Selected |",
        "|---|---:|---:|---:|---:|---|---|",
    ]

    for _, row in summary.iterrows():
        lines.append(
            f"| {row['configuration_name']} "
            f"| {int(row['n_features'])} "
            f"| {row['oof_mae']:.6f} "
            f"| {row['bootstrap_se']:.6f} "
            f"| [{row['bootstrap_ci_2_5']:.6f}, "
            f"{row['bootstrap_ci_97_5']:.6f}] "
            f"| {'Yes' if row['within_one_se'] else 'No'} "
            f"| {'**Yes**' if row['selected'] else 'No'} |"
        )

    lines.extend([
        "",
        "## Selection",
        "",
        f"- Best raw configuration: **{best['configuration_name']}**",
        f"- Best raw OOF MAE: {best_mae:.6f}",
        f"- Bootstrap SE of best configuration: {best_se:.6f}",
        f"- One-SE cutoff: **{cutoff:.6f}**",
        f"- Selected simplest eligible configuration: **{selected_name}**",
        "",
        "Simplicity order: Ridge before RBF-SVR; fewer features before more features; "
        "colour-only before device/lamp when counts tie; lower MAE as final tie-breaker.",
        "",
        "The selected configuration is a development candidate only. "
        "Validation and test data were not used.",
    ])

    REPORT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_PATH.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\nE2 BOOTSTRAP RESULTS")
    print("=" * 100)

    print(
        summary[
            [
                "configuration_name",
                "n_features",
                "oof_mae",
                "bootstrap_se",
                "bootstrap_ci_2_5",
                "bootstrap_ci_97_5",
                "within_one_se",
                "selected",
            ]
        ].to_string(index=False)
    )

    print("\nONE-STANDARD-ERROR DECISION")
    print("=" * 72)
    print("Best raw configuration:", best["configuration_name"])
    print(f"Best MAE: {best_mae:.6f}")
    print(f"Bootstrap SE: {best_se:.6f}")
    print(f"Cutoff: {cutoff:.6f}")
    print("Selected simplest eligible:", selected_name)

    print("\nOUTPUTS")
    print("=" * 72)
    print(SUMMARY_PATH.relative_to(ROOT))
    print(SAMPLES_PATH.relative_to(ROOT))
    print(SELECTION_PATH.relative_to(ROOT))
    print(REPORT_PATH.relative_to(ROOT))
    print(CANDIDATE_PATH.relative_to(ROOT))


if __name__ == "__main__":
    main()
