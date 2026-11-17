
"""
Phase 6.5 experiments E1-E5.

Usage:
    python src/run_phase6_5_experiments.py --experiment E1
    python src/run_phase6_5_experiments.py --experiment E2
    python src/run_phase6_5_experiments.py --experiment E3
    python src/run_phase6_5_experiments.py --experiment E4
    python src/run_phase6_5_experiments.py --experiment E5
    python src/run_phase6_5_experiments.py --experiment all

All experiments use training rows only.
Validation and test partitions are excluded.
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, GroupKFold

try:
    from src.phase6_experiment_utils import (
        ROOT,
        S2_COLUMNS,
        CONDITIONS,
        OUTER_SPLITS,
        DATASET_PATH,
        load_training_data,
        load_registry,
        load_saved_outer_fold_map,
        exact_saved_outer_splits,
        grouped_outer_splits,
        calculate_metrics,
        build_model,
        fit_with_inner_cv,
        get_X,
        nested_oof,
        append_metrics,
    )
except ModuleNotFoundError:
    from phase6_experiment_utils import (
        ROOT,
        S2_COLUMNS,
        CONDITIONS,
        OUTER_SPLITS,
        DATASET_PATH,
        load_training_data,
        load_registry,
        load_saved_outer_fold_map,
        exact_saved_outer_splits,
        grouped_outer_splits,
        calculate_metrics,
        build_model,
        fit_with_inner_cv,
        get_X,
        nested_oof,
        append_metrics,
    )


RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

RUN_ID = datetime.now().strftime(
    "phase6_5_%Y%m%d_%H%M%S"
)


def metric_record(experiment, configuration, model, metrics, n_images, n_tubes):
    row = {
        "run_id": RUN_ID,
        "date": "2026-11-17",
        "experiment": experiment,
        "configuration": configuration,
        "model": model,
        "evaluation_role": "train",
        "n_images": int(n_images),
        "n_tubes": int(n_tubes),
    }

    row.update(metrics)
    return row


def save_experiment_outputs(
    experiment,
    oof_frames,
    metric_rows,
    parameter_frames=None,
):
    RESULTS.mkdir(parents=True, exist_ok=True)

    oof = (
        pd.concat(oof_frames, ignore_index=True)
        if oof_frames
        else pd.DataFrame()
    )

    if not oof.empty:
        oof.to_csv(
            RESULTS / f"{experiment.lower()}_oof.csv",
            index=False,
        )

    if parameter_frames:
        params = pd.concat(
            parameter_frames,
            ignore_index=True,
        )

        params.to_csv(
            RESULTS / f"{experiment.lower()}_best_params.csv",
            index=False,
        )

    if metric_rows:
        append_metrics(metric_rows)


# ============================================================
# E1 — Random image split versus tube-grouped split
# ============================================================

def run_e1(train, fold_map):
    print("\nE1 — LEAKAGE DEMONSTRATION")
    print("=" * 72)

    from sklearn.linear_model import LinearRegression
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.base import clone

    # Same B1 feature and model as Phase 6.1.
    columns = ["b_med"]

    y = train["conc"].to_numpy(dtype=float)
    groups = train["tube_id"].astype(str).to_numpy()
    X = train[columns].apply(pd.to_numeric, errors="coerce")

    # Grouped result from Phase 6.1, not recomputed.
    old = pd.read_csv(
        RESULTS / "walking_skeleton_oof.csv"
    )

    grouped = old[
        old["model"].astype(str) == "B1"
    ].copy()

    if set(grouped["image_id"].astype(str)) != set(
        train["image_id"].astype(str)
    ):
        raise ValueError(
            "Phase 6.1 B1 OOF image IDs do not match training rows"
        )

    grouped_metrics = calculate_metrics(
        grouped["actual"],
        grouped["predicted"],
    )

    grouped_out = grouped[
        ["image_id", "tube_id", "actual", "predicted"]
    ].copy()

    grouped_out["split_type"] = "tube_grouped"

    # Random image-level KFold.
    random_cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", LinearRegression()),
    ])

    random_predictions = np.full(len(train), np.nan)
    random_fold = np.full(len(train), -1)

    overlap_rows = []

    for fold, (tr, va) in enumerate(
        random_cv.split(X, y),
        start=1,
    ):
        train_tubes = set(groups[tr])
        valid_tubes = set(groups[va])
        overlap = train_tubes & valid_tubes

        fitted = clone(pipeline)
        fitted.fit(X.iloc[tr], y[tr])

        random_predictions[va] = fitted.predict(X.iloc[va])
        random_fold[va] = fold

        overlap_rows.append({
            "fold": fold,
            "train_images": len(tr),
            "validation_images": len(va),
            "train_tubes": len(train_tubes),
            "validation_tubes": len(valid_tubes),
            "overlapping_tubes": len(overlap),
            "overlap_tube_ids": ";".join(sorted(overlap)),
        })

        print(
            f"Random fold {fold}: "
            f"{len(overlap)} tubes occur on both sides"
        )

    if np.isnan(random_predictions).any():
        raise RuntimeError("Random KFold missed some predictions")

    random_metrics = calculate_metrics(y, random_predictions)

    random_out = train[
        ["image_id", "tube_id"]
    ].copy()

    random_out["actual"] = y
    random_out["predicted"] = random_predictions
    random_out["fold"] = random_fold
    random_out["split_type"] = "random_image"

    overlap_df = pd.DataFrame(overlap_rows)

    overlap_df.to_csv(
        RESULTS / "e1_random_split_tube_overlap.csv",
        index=False,
    )

    random_out.to_csv(
        RESULTS / "e1_random_oof.csv",
        index=False,
    )

    comparison = pd.DataFrame([
        {
            "split_type": "random_image",
            **random_metrics,
        },
        {
            "split_type": "tube_grouped",
            **grouped_metrics,
        },
    ])

    comparison["mae_optimism_vs_grouped"] = (
        grouped_metrics["mae"] - comparison["mae"]
    )

    comparison.to_csv(
        RESULTS / "e1_leakage_comparison.csv",
        index=False,
    )

    metrics_rows = []

    metrics_rows.append(
        metric_record(
            "E1_leakage",
            "random_image_5fold",
            "B1",
            random_metrics,
            len(train),
            train["tube_id"].nunique(),
        )
    )

    metrics_rows.append(
        metric_record(
            "E1_leakage",
            "tube_grouped_5fold",
            "B1",
            grouped_metrics,
            len(train),
            train["tube_id"].nunique(),
        )
    )

    append_metrics(metrics_rows)

    print("\nE1 COMPARISON")
    print(comparison.to_string(index=False))
    print("\nAverage overlapping tubes per random fold:",
          overlap_df["overlapping_tubes"].mean())

    print("\nSaved:")
    print("results/e1_leakage_comparison.csv")
    print("results/e1_random_split_tube_overlap.csv")
    print("results/e1_random_oof.csv")


# ============================================================
# E2 — S0-S4 feature ablation with Ridge and SVR
# ============================================================

def run_e2(train, fold_map):
    print("\nE2 — FEATURE ABLATION S0-S4")
    print("=" * 72)

    registry = load_registry()

    oof_frames = []
    parameter_frames = []
    metric_rows = []
    summary_rows = []

    outer_splits, _ = exact_saved_outer_splits(
        train,
        fold_map,
    )

    for feature_set in ("S0", "S1", "S2", "S3", "S4"):
        for variant in (
            "colour_only",
            "with_device_and_lamp",
        ):
            columns = registry[feature_set][variant]

            for model_name in ("Ridge", "SVR"):
                label = f"{feature_set}_{variant}"

                print(
                    f"\nE2 {label} — {model_name}: "
                    f"{len(columns)} features"
                )

                oof, metrics, params = nested_oof(
                    df=train,
                    model_name=model_name,
                    columns=columns,
                    outer_splits=outer_splits,
                    experiment="E2_feature_ablation",
                    label=label,
                )

                oof_frames.append(oof)
                parameter_frames.append(pd.DataFrame(params))

                summary = {
                    "feature_set": feature_set,
                    "variant": variant,
                    "model": model_name,
                    "n_features": len(columns),
                    **metrics,
                }

                summary_rows.append(summary)

                metric_rows.append(
                    metric_record(
                        "E2_feature_ablation",
                        label,
                        model_name,
                        metrics,
                        len(train),
                        train["tube_id"].nunique(),
                    )
                )

    summary_df = pd.DataFrame(summary_rows).sort_values("mae")

    summary_df.to_csv(
        RESULTS / "e2_feature_ablation.csv",
        index=False,
    )

    save_experiment_outputs(
        "E2_feature_ablation",
        oof_frames,
        metric_rows,
        parameter_frames,
    )

    print("\nE2 RESULTS — SORTED BY MAE")
    print(summary_df.to_string(index=False))


# ============================================================
# E3 — Model ladder per condition and pooled
# ============================================================

def run_e3(train, fold_map):
    print("\nE3 — CONDITION-SPECIFIC AND POOLED MODEL LADDER")
    print("=" * 72)

    oof_frames = []
    parameter_frames = []
    metric_rows = []
    summary_rows = []

    model_specs = [
        ("B0", []),
        ("B1", ["b_med"]),
        ("Ridge_S2", S2_COLUMNS),
        ("Polynomial_Ridge_b_med", ["b_med"]),
        ("SVR_RBF_S2", S2_COLUMNS),
        ("Random_Forest_S2", S2_COLUMNS),
        ("Gradient_Boosting_S2", S2_COLUMNS),
        ("MLP_S2", S2_COLUMNS),
    ]

    experiments = []

    for condition in CONDITIONS:
        subset = train[
            train["condition"] == condition
        ].copy().reset_index(drop=True)

        if subset["tube_id"].nunique() < 5:
            print(f"Skipping {condition}: fewer than five tubes")
            continue

        experiments.append((condition, subset, False))

    # Pooled dataset with device/lamp one-hot columns.
    pooled = train.copy().reset_index(drop=True)
    experiments.append(("pooled_device_lamp", pooled, True))

    device_lamp = [
        "device_dino",
        "device_phone",
        "lamp_off",
        "lamp_on",
    ]

    for label, subset, pooled_mode in experiments:
        if pooled_mode:
            outer_splits, _ = exact_saved_outer_splits(
                subset,
                fold_map,
            )
        else:
            outer_splits = grouped_outer_splits(
                subset,
                n_splits=5,
            )

        print(
            f"\nE3 CONDITION: {label}, "
            f"{len(subset)} images, "
            f"{subset['tube_id'].nunique()} tubes"
        )

        for model_name, base_columns in model_specs:
            columns = list(base_columns)

            if pooled_mode and model_name != "B0":
                columns += device_lamp

            oof, metrics, params = nested_oof(
                df=subset,
                model_name=model_name,
                columns=columns,
                outer_splits=outer_splits,
                experiment="E3_condition_model_ladder",
                label=label,
            )

            oof_frames.append(oof)
            parameter_frames.append(pd.DataFrame(params))

            summary = {
                "condition": label,
                "model": model_name,
                "feature_columns": ";".join(columns),
                "n_images": len(subset),
                "n_tubes": subset["tube_id"].nunique(),
                **metrics,
            }

            summary_rows.append(summary)

            metric_rows.append(
                metric_record(
                    "E3_condition_model_ladder",
                    label,
                    model_name,
                    metrics,
                    len(subset),
                    subset["tube_id"].nunique(),
                )
            )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["condition", "mae"]
    )

    summary_df.to_csv(
        RESULTS / "e3_condition_model_ladder.csv",
        index=False,
    )

    save_experiment_outputs(
        "E3_condition_model_ladder",
        oof_frames,
        metric_rows,
        parameter_frames,
    )

    print("\nE3 SUMMARY")
    print(summary_df.to_string(index=False))


# ============================================================
# E4 — Leave-one-concentration-out
# ============================================================

def run_e4(train):
    print("\nE4 — LEAVE-ONE-CONCENTRATION-OUT")
    print("=" * 72)

    best_three = [
        "Random_Forest_S2",
        "Gradient_Boosting_S2",
        "SVR_RBF_S2",
    ]

    concentration_values = sorted(
        train["conc"].unique().tolist()
    )

    oof_frames = []
    parameter_rows = []
    summary_rows = []
    metric_rows = []

    for concentration in concentration_values:
        heldout = train[
            train["conc"] == concentration
        ].copy().reset_index(drop=True)

        development = train[
            train["conc"] != concentration
        ].copy().reset_index(drop=True)

        if heldout.empty or development.empty:
            continue

        if development["tube_id"].nunique() < 4:
            raise ValueError(
                "Not enough training tubes for inner CV "
                f"when holding out concentration {concentration}"
            )

        print(
            f"\nHolding out concentration {concentration}: "
            f"{len(heldout)} images, "
            f"{heldout['tube_id'].nunique()} tubes"
        )

        for model_name in best_three:
            X_train = get_X(
                development,
                S2_COLUMNS,
                model_name,
            )

            X_test = get_X(
                heldout,
                S2_COLUMNS,
                model_name,
            )

            y_train = development["conc"].to_numpy(dtype=float)
            y_test = heldout["conc"].to_numpy(dtype=float)

            groups_train = (
                development["tube_id"].astype(str).to_numpy()
            )

            fitted, params, inner_score = fit_with_inner_cv(
                model_name,
                X_train,
                y_train,
                groups_train,
            )

            predictions = fitted.predict(X_test)

            if not np.allclose(y_test, concentration):
                raise RuntimeError(
                    "LOCO held-out set contains another concentration"
                )

            result = heldout[
                [
                    "image_id",
                    "tube_id",
                    "conc",
                    "condition",
                ]
            ].copy()

            result["experiment"] = "E4_LOCO"
            result["model"] = model_name
            result["heldout_concentration"] = concentration
            result["actual"] = y_test
            result["predicted"] = predictions
            result["absolute_error"] = np.abs(
                predictions - y_test
            )

            oof_frames.append(result)

            per_conc_metrics = calculate_metrics(
                y_test,
                predictions,
            )

            summary_rows.append({
                "model": model_name,
                "heldout_concentration": concentration,
                "n_images": len(heldout),
                "n_tubes": heldout["tube_id"].nunique(),
                **per_conc_metrics,
            })

            parameter_rows.append({
                "model": model_name,
                "heldout_concentration": concentration,
                "n_train_tubes": development["tube_id"].nunique(),
                "n_heldout_tubes": heldout["tube_id"].nunique(),
                "inner_best_neg_mae": inner_score,
                "best_params": str(params),
            })

    predictions_df = pd.concat(
        oof_frames,
        ignore_index=True,
    )

    per_concentration_df = pd.DataFrame(summary_rows)

    predictions_df.to_csv(
        RESULTS / "e4_loco_predictions.csv",
        index=False,
    )

    per_concentration_df.to_csv(
        RESULTS / "e4_loco_summary.csv",
        index=False,
    )

    pd.DataFrame(parameter_rows).to_csv(
        RESULTS / "e4_loco_best_params.csv",
        index=False,
    )

    for model_name, group in predictions_df.groupby("model"):
        metrics = calculate_metrics(
            group["actual"],
            group["predicted"],
        )

        metric_rows.append(
            metric_record(
                "E4_leave_one_concentration_out",
                "S2_colour_only",
                model_name,
                metrics,
                len(group),
                group["tube_id"].nunique(),
            )
        )

    append_metrics(metric_rows)

    pooled_rows = []

    for model_name, group in predictions_df.groupby("model"):
        pooled_rows.append({
            "model": model_name,
            **calculate_metrics(
                group["actual"],
                group["predicted"],
            ),
            "n_images": len(group),
            "n_tubes": group["tube_id"].nunique(),
        })

    pooled_summary = pd.DataFrame(pooled_rows)

    pooled_summary.to_csv(
        RESULTS / "e4_loco_pooled_summary.csv",
        index=False,
    )

    print("\nE4 POOLED LOCO RESULTS")
    print(pooled_summary.to_string(index=False))

    print("\nE4 PER-CONCENTRATION RESULTS")
    print(per_concentration_df.to_string(index=False))


# ============================================================
# E5 — Cross-condition transfer
# ============================================================

def run_e5(train):
    print("\nE5 — CROSS-CONDITION TRANSFER")
    print("=" * 72)

    model_name = "SVR_RBF_S2"

    predictions_all = []
    details = []
    metric_rows = []

    matrix_rows = []

    for source_condition in CONDITIONS:
        source = train[
            train["condition"] == source_condition
        ].copy()

        source_tubes = set(source["tube_id"].astype(str))

        for target_condition in CONDITIONS:
            target = train[
                train["condition"] == target_condition
            ].copy()

            target_tubes = set(target["tube_id"].astype(str))

            # Only tubes with both source and target conditions.
            common_tubes = sorted(
                source_tubes & target_tubes
            )

            if len(common_tubes) < OUTER_SPLITS:
                print(
                    f"Skipping {source_condition} -> {target_condition}: "
                    f"only {len(common_tubes)} common tubes"
                )
                matrix_rows.append({
                    "source_condition": source_condition,
                    "target_condition": target_condition,
                    "n_common_tubes": len(common_tubes),
                    "n_target_images": 0,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "r2": np.nan,
                    "bias_pred_minus_actual": np.nan,
                    "within_plus_minus_0_1": np.nan,
                })
                continue

            source = source[
                source["tube_id"].astype(str).isin(common_tubes)
            ].copy()

            target = target[
                target["tube_id"].astype(str).isin(common_tubes)
            ].copy()

            # Create folds over unique physical tubes.
            tube_frame = pd.DataFrame({
                "tube_id": common_tubes,
            })

            tube_cv = GroupKFold(n_splits=OUTER_SPLITS)

            tube_splits = list(
                tube_cv.split(
                    np.zeros((len(tube_frame), 1)),
                    groups=tube_frame["tube_id"].to_numpy(),
                )
            )

            cell_predictions = []

            print(
                f"\n{source_condition} -> {target_condition}: "
                f"{len(common_tubes)} common tubes"
            )

            for fold, (train_tube_idx, valid_tube_idx) in enumerate(
                tube_splits,
                start=1,
            ):
                train_tubes = set(
                    tube_frame.iloc[train_tube_idx]["tube_id"]
                )
                heldout_tubes = set(
                    tube_frame.iloc[valid_tube_idx]["tube_id"]
                )

                if not train_tubes.isdisjoint(heldout_tubes):
                    raise RuntimeError(
                        "Cross-condition tube leakage detected"
                    )

                source_train = source[
                    source["tube_id"].astype(str).isin(train_tubes)
                ].copy()

                target_test = target[
                    target["tube_id"].astype(str).isin(heldout_tubes)
                ].copy()

                if source_train.empty or target_test.empty:
                    continue

                X_train = source_train[S2_COLUMNS].apply(
                    pd.to_numeric,
                    errors="coerce",
                )

                X_test = target_test[S2_COLUMNS].apply(
                    pd.to_numeric,
                    errors="coerce",
                )

                y_train = source_train["conc"].to_numpy(dtype=float)
                y_test = target_test["conc"].to_numpy(dtype=float)

                groups_train = (
                    source_train["tube_id"].astype(str).to_numpy()
                )

                fitted, params, inner_score = fit_with_inner_cv(
                    model_name,
                    X_train,
                    y_train,
                    groups_train,
                )

                predicted = fitted.predict(X_test)

                output = target_test[
                    [
                        "image_id",
                        "tube_id",
                        "conc",
                    ]
                ].copy()

                output["source_condition"] = source_condition
                output["target_condition"] = target_condition
                output["fold"] = fold
                output["actual"] = y_test
                output["predicted"] = predicted
                output["absolute_error"] = np.abs(
                    predicted - y_test
                )

                cell_predictions.append(output)

                details.append({
                    "source_condition": source_condition,
                    "target_condition": target_condition,
                    "fold": fold,
                    "n_train_tubes": len(train_tubes),
                    "n_heldout_tubes": len(heldout_tubes),
                    "n_train_images": len(source_train),
                    "n_target_images": len(target_test),
                    "inner_best_neg_mae": inner_score,
                    "best_params": str(params),
                })

            if not cell_predictions:
                raise RuntimeError(
                    f"No predictions for {source_condition}->{target_condition}"
                )

            cell = pd.concat(
                cell_predictions,
                ignore_index=True,
            )

            predictions_all.append(cell)

            metrics = calculate_metrics(
                cell["actual"],
                cell["predicted"],
            )

            matrix_rows.append({
                "source_condition": source_condition,
                "target_condition": target_condition,
                "n_common_tubes": len(common_tubes),
                "n_target_images": len(cell),
                **metrics,
            })

            metric_rows.append(
                metric_record(
                    "E5_cross_condition_transfer",
                    f"{source_condition}_to_{target_condition}",
                    model_name,
                    metrics,
                    len(cell),
                    len(common_tubes),
                )
            )

    if not predictions_all:
        raise RuntimeError("E5 produced no transfer predictions")

    pred_df = pd.concat(
        predictions_all,
        ignore_index=True,
    )

    details_df = pd.DataFrame(details)
    matrix_df = pd.DataFrame(matrix_rows)

    pred_df.to_csv(
        RESULTS / "e5_cross_condition_predictions.csv",
        index=False,
    )

    details_df.to_csv(
        RESULTS / "e5_cross_condition_details.csv",
        index=False,
    )

    matrix_df.to_csv(
        RESULTS / "e5_cross_condition_mae_matrix.csv",
        index=False,
    )

    pivot = matrix_df.pivot(
        index="source_condition",
        columns="target_condition",
        values="mae",
    )

    pivot = pivot.reindex(
        index=CONDITIONS,
        columns=CONDITIONS,
    )

    pivot.to_csv(
        RESULTS / "e5_cross_condition_mae_pivot.csv"
    )

    append_metrics(metric_rows)

    print("\nE5 MAE MATRIX")
    print(pivot.to_string())

    print("\nE5 CELL DETAILS")
    print(matrix_df.to_string(index=False))


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        choices=["E1", "E2", "E3", "E4", "E5", "all"],
        default="all",
    )

    args = parser.parse_args()

    RESULTS.mkdir(parents=True, exist_ok=True)

    train = load_training_data()
    fold_map = load_saved_outer_fold_map()

    print("PHASE 6.5 EXPERIMENTS")
    print("=" * 72)
    print("Training images:", len(train))
    print("Training tubes:", train["tube_id"].nunique())
    print("Validation rows: excluded")
    print("Test rows: excluded")
    print("Experiment:", args.experiment)

    if len(train) != 666:
        raise ValueError(
            f"Expected 666 training images, found {len(train)}"
        )

    if train["tube_id"].nunique() != 31:
        raise ValueError(
            f"Expected 31 training tubes, "
            f"found {train['tube_id'].nunique()}"
        )

    if args.experiment in {"E1", "all"}:
        run_e1(train, fold_map)

    if args.experiment in {"E2", "all"}:
        run_e2(train, fold_map)

    if args.experiment in {"E3", "all"}:
        run_e3(train, fold_map)

    if args.experiment in {"E4", "all"}:
        run_e4(train)

    if args.experiment in {"E5", "all"}:
        run_e5(train)

    print("\nPHASE 6.5 EXPERIMENT RUN COMPLETE")
    print("=" * 72)
    print("Experiment:", args.experiment)
    print("Training images:", len(train))
    print("Training tubes:", train["tube_id"].nunique())
    print("No validation or test rows were used.")


if __name__ == "__main__":
    main()
