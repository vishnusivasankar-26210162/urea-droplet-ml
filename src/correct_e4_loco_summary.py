
"""
Correct Phase 6.5 E4 per-concentration R2 reporting.

Input:
    results/e4_loco_predictions.csv

Output:
    results/e4_loco_summary.csv

R2 is undefined within a held-out concentration because all actual
values in that group are constant.

The pooled E4 summary and its R2 are left unchanged.
No models are retrained.
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = ROOT / "results/e4_loco_predictions.csv"
SUMMARY_PATH = ROOT / "results/e4_loco_summary.csv"
POOLED_PATH = ROOT / "results/e4_loco_pooled_summary.csv"


def calculate_group_metrics(group):
    actual = group["actual"].to_numpy(dtype=float)
    predicted = group["predicted"].to_numpy(dtype=float)

    error = predicted - actual
    absolute_error = np.abs(error)

    if len(np.unique(actual)) != 1:
        raise ValueError(
            "Expected a constant target within each held-out "
            "concentration group."
        )

    low = actual < 0.5
    medium = (actual >= 0.5) & (actual <= 1.0)
    high = actual > 1.0

    def band_mae(mask):
        return (
            float(absolute_error[mask].mean())
            if mask.any()
            else np.nan
        )

    return {
        "n_images": len(group),
        "n_tubes": group["tube_id"].nunique(),
        "n_unique_actual": len(np.unique(actual)),
        "mae": float(absolute_error.mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "r2": np.nan,
        "r2_status": "undefined_constant_target",
        "bias_pred_minus_actual": float(error.mean()),
        "within_plus_minus_0_1": float(
            np.mean(absolute_error <= 0.1)
        ),
        "mae_low_actual_lt_0_5": band_mae(low),
        "mae_medium_0_5_to_1_0": band_mae(medium),
        "mae_high_actual_gt_1_0": band_mae(high),
    }


def main():
    if not PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(PREDICTIONS_PATH)

    predictions = pd.read_csv(PREDICTIONS_PATH)

    required = {
        "model",
        "heldout_concentration",
        "tube_id",
        "image_id",
        "actual",
        "predicted",
    }

    missing = required - set(predictions.columns)

    if missing:
        raise ValueError(
            f"E4 predictions missing columns: {sorted(missing)}"
        )

    predictions["actual"] = pd.to_numeric(
        predictions["actual"],
        errors="raise",
    )

    predictions["predicted"] = pd.to_numeric(
        predictions["predicted"],
        errors="raise",
    )

    rows = []

    for (model, concentration), group in predictions.groupby(
        ["model", "heldout_concentration"],
        sort=True,
    ):
        metrics = calculate_group_metrics(group)

        rows.append({
            "model": model,
            "heldout_concentration": concentration,
            **metrics,
        })

    summary = pd.DataFrame(rows)

    # Explicitly guarantee that all per-concentration R2 values are missing.
    if summary["r2"].notna().any():
        raise RuntimeError(
            "Per-concentration R2 should be undefined"
        )

    SUMMARY_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    print("E4 PER-CONCENTRATION SUMMARY CORRECTED")
    print("=" * 72)
    print("Model/concentration groups:", len(summary))
    print("All per-concentration R2 values: undefined")
    print("Per-concentration MAE/RMSE/bias/fraction: retained")
    print("\nSummary preview:")
    print(
        summary[
            [
                "model",
                "heldout_concentration",
                "n_images",
                "n_tubes",
                "mae",
                "rmse",
                "r2",
                "r2_status",
                "bias_pred_minus_actual",
                "within_plus_minus_0_1",
            ]
        ].to_string(index=False)
    )

    # Preserve and display the valid pooled R2 values.
    if POOLED_PATH.is_file():
        pooled = pd.read_csv(POOLED_PATH)

        print("\nPOOLED E4 SUMMARY — PRESERVED")
        print("=" * 72)

        display_columns = [
            column
            for column in [
                "model",
                "mae",
                "rmse",
                "r2",
                "bias_pred_minus_actual",
                "within_plus_minus_0_1",
            ]
            if column in pooled.columns
        ]

        print(pooled[display_columns].to_string(index=False))

    print("\nCorrected file:", SUMMARY_PATH.relative_to(ROOT))
    print("Pooled summary was not modified.")
    print("No models were retrained.")


if __name__ == "__main__":
    main()
