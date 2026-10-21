"""Held-out MAE comparison for Model R and the B2 reference baseline."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.errors import ArtifactError, ContractError

_KEYS = ("split_seed", "shipment_id", "checkpoint")
_MODEL_COLUMNS = (*_KEYS, "actual_final_delay_h", "predicted_final_delay_h")
_BASELINE_COLUMNS = (*_KEYS, "baseline", "predicted_final_delay_h")
_Q_COLUMNS = (
    *_KEYS,
    "actual_final_delay_h",
    "predicted_final_lower_h",
    "predicted_final_upper_h",
    "quantiles_crossed",
)
_C_COLUMNS = (
    *_KEYS,
    "actual_is_delayed",
    "predicted_delay_probability",
)
_RELIABILITY_BINS = 10
_CLASSIFICATION_THRESHOLD = 0.5
_THRESHOLD_SWEEP = tuple(index / 20 for index in range(21))


def _validated_prediction_keys(table: pd.DataFrame) -> pd.DataFrame:
    if table[list(_KEYS)].isna().any().any():
        raise ContractError("prediction seed, shipment ID, and checkpoint are required")
    table["shipment_id"] = table["shipment_id"].astype("string")
    seeds = pd.to_numeric(table["split_seed"], errors="coerce")
    checkpoints = pd.to_numeric(table["checkpoint"], errors="coerce")
    if (
        seeds.isna().any()
        or not np.isfinite(seeds).all()
        or (seeds % 1 != 0).any()
        or checkpoints.isna().any()
        or not np.isfinite(checkpoints).all()
        or (checkpoints % 1 != 0).any()
    ):
        raise ContractError("prediction seeds and checkpoints must be finite integers")
    table["split_seed"] = seeds.astype("int64")
    table["checkpoint"] = checkpoints.astype("int64")
    if table.duplicated(list(_KEYS)).any():
        raise ContractError(
            "each prediction must be unique per seed, shipment, checkpoint"
        )
    return table


def _paired_model_r_b2_predictions(
    model_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
) -> pd.DataFrame:
    missing_model = sorted(set(_MODEL_COLUMNS) - set(model_predictions.columns))
    if missing_model:
        raise ContractError(f"Model R predictions are missing columns: {missing_model}")
    missing_baseline = sorted(
        set(_BASELINE_COLUMNS) - set(baseline_predictions.columns)
    )
    if missing_baseline:
        raise ContractError(
            f"baseline predictions are missing columns: {missing_baseline}"
        )

    model = model_predictions[list(_MODEL_COLUMNS)].copy()
    baseline = baseline_predictions.loc[
        baseline_predictions["baseline"] == "B2", list(_BASELINE_COLUMNS)
    ].copy()
    if model.empty or baseline.empty:
        raise ContractError("Model R and B2 predictions must be non-empty")

    model = _validated_prediction_keys(model).rename(
        columns={
            "actual_final_delay_h": "actual_h",
            "predicted_final_delay_h": "model_pred_h",
        }
    )
    baseline = _validated_prediction_keys(baseline).rename(
        columns={"predicted_final_delay_h": "b2_pred_h"}
    )
    joined = model.merge(baseline, on=list(_KEYS), how="outer", indicator=True)
    if not joined["_merge"].eq("both").all():
        unmatched_model = joined.loc[
            joined["_merge"] == "left_only", "shipment_id"
        ].astype(str)
        unmatched_b2 = joined.loc[
            joined["_merge"] == "right_only", "shipment_id"
        ].astype(str)
        raise ContractError(
            "Model R and B2 prediction rows do not match; "
            f"missing B2: {sorted(unmatched_model.unique())[:5]}, "
            f"missing Model R: {sorted(unmatched_b2.unique())[:5]}"
        )

    for column in ("actual_h", "model_pred_h", "b2_pred_h"):
        values = pd.to_numeric(joined[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise ContractError(f"prediction column {column} must be finite numbers")
        joined[column] = values
    return joined


def _model_r_metrics_by_seed(joined: pd.DataFrame) -> pd.DataFrame:
    joined["model_error_h"] = (joined["actual_h"] - joined["model_pred_h"]).abs()
    joined["b2_error_h"] = (joined["actual_h"] - joined["b2_pred_h"]).abs()
    joined["model_squared_error_h"] = (joined["actual_h"] - joined["model_pred_h"]) ** 2
    joined["b2_squared_error_h"] = (joined["actual_h"] - joined["b2_pred_h"]) ** 2
    by_seed = joined.groupby(["split_seed", "checkpoint"], as_index=False).agg(
        model_mae_h=("model_error_h", "mean"),
        model_median_ae_h=("model_error_h", "median"),
        model_p90_ae_h=("model_error_h", lambda errors: errors.quantile(0.9)),
        model_mse_h=("model_squared_error_h", "mean"),
        b2_mae_h=("b2_error_h", "mean"),
        b2_median_ae_h=("b2_error_h", "median"),
        b2_p90_ae_h=("b2_error_h", lambda errors: errors.quantile(0.9)),
        b2_mse_h=("b2_squared_error_h", "mean"),
    )
    by_seed["model_rmse_h"] = np.sqrt(by_seed["model_mse_h"])
    by_seed["b2_rmse_h"] = np.sqrt(by_seed["b2_mse_h"])
    return by_seed


def compare_model_r_to_b2(
    model_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Report held-out regression errors by checkpoint across split seeds."""
    by_seed = _model_r_metrics_by_seed(
        _paired_model_r_b2_predictions(model_predictions, baseline_predictions)
    )
    by_seed["model_beats_b2"] = by_seed["model_mae_h"] < by_seed["b2_mae_h"]
    summary = by_seed.groupby("checkpoint", as_index=False).agg(
        model_mae_mean_h=("model_mae_h", "mean"),
        model_mae_std_h=("model_mae_h", "std"),
        model_median_ae_mean_h=("model_median_ae_h", "mean"),
        model_median_ae_std_h=("model_median_ae_h", "std"),
        model_p90_ae_mean_h=("model_p90_ae_h", "mean"),
        model_p90_ae_std_h=("model_p90_ae_h", "std"),
        model_rmse_mean_h=("model_rmse_h", "mean"),
        model_rmse_std_h=("model_rmse_h", "std"),
        b2_mae_mean_h=("b2_mae_h", "mean"),
        b2_mae_std_h=("b2_mae_h", "std"),
        b2_median_ae_mean_h=("b2_median_ae_h", "mean"),
        b2_median_ae_std_h=("b2_median_ae_h", "std"),
        b2_p90_ae_mean_h=("b2_p90_ae_h", "mean"),
        b2_p90_ae_std_h=("b2_p90_ae_h", "std"),
        b2_rmse_mean_h=("b2_rmse_h", "mean"),
        b2_rmse_std_h=("b2_rmse_h", "std"),
        model_beats_b2_seeds=("model_beats_b2", "sum"),
        n_split_seeds=("split_seed", "nunique"),
    )
    summary["skill_score"] = np.where(
        summary["b2_mae_mean_h"] > 0,
        1.0 - summary["model_mae_mean_h"] / summary["b2_mae_mean_h"],
        np.nan,
    )
    return summary


def _validated_model_c_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(set(_C_COLUMNS) - set(predictions.columns))
    if missing:
        raise ContractError(f"Model C predictions are missing columns: {missing}")
    if predictions.empty:
        raise ContractError("Model C predictions must be non-empty")

    table = _validated_prediction_keys(predictions[list(_C_COLUMNS)].copy())
    labels = pd.to_numeric(table["actual_is_delayed"], errors="coerce")
    if (
        labels.isna().any()
        or not np.isfinite(labels).all()
        or not labels.isin((0, 1)).all()
    ):
        raise ContractError("Model C actual_is_delayed values must be binary labels")
    probabilities = pd.to_numeric(table["predicted_delay_probability"], errors="coerce")
    if (
        probabilities.isna().any()
        or not np.isfinite(probabilities).all()
        or ((probabilities < 0) | (probabilities > 1)).any()
    ):
        raise ContractError(
            "Model C predicted_delay_probability values must be between 0 and 1"
        )

    table["actual_is_delayed"] = labels.astype("int8")
    table["predicted_delay_probability"] = probabilities
    return table


def _model_c_metrics_by_seed(table: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (seed, checkpoint), rows in table.groupby(
        ["split_seed", "checkpoint"], sort=True
    ):
        labels = rows["actual_is_delayed"].to_numpy(dtype=int)
        probabilities = rows["predicted_delay_probability"].to_numpy(dtype=float)
        has_both_labels = np.unique(labels).size == 2
        if has_both_labels:
            precision, recall, _ = precision_recall_curve(labels, probabilities)
            pr_auc = auc(recall, precision)
            roc_auc = roc_auc_score(labels, probabilities)
        else:
            pr_auc = np.nan
            roc_auc = np.nan
        records.append(
            {
                "split_seed": seed,
                "checkpoint": checkpoint,
                "brier_score": brier_score_loss(labels, probabilities),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "f1_at_0_5": f1_score(
                    labels,
                    probabilities >= _CLASSIFICATION_THRESHOLD,
                    zero_division=0,
                ),
            }
        )
    return pd.DataFrame(records)


def evaluate_model_c_classification_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize held-out classification metrics by checkpoint across seeds."""
    table = _validated_model_c_predictions(predictions)
    by_seed = _model_c_metrics_by_seed(table)
    summary = by_seed.groupby("checkpoint", as_index=False).agg(
        brier_score_mean=("brier_score", "mean"),
        brier_score_std=("brier_score", "std"),
        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_std=("pr_auc", "std"),
        f1_at_0_5_mean=("f1_at_0_5", "mean"),
        f1_at_0_5_std=("f1_at_0_5", "std"),
        n_split_seeds=("split_seed", "nunique"),
        n_auc_split_seeds=("roc_auc", "count"),
    )
    return summary


def evaluate_model_c_threshold_sweep(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report precision, recall, and F1 over fixed thresholds by checkpoint."""
    table = _validated_model_c_predictions(predictions)
    records = []
    for (seed, checkpoint), rows in table.groupby(
        ["split_seed", "checkpoint"], sort=True
    ):
        labels = rows["actual_is_delayed"].to_numpy(dtype=int)
        probabilities = rows["predicted_delay_probability"].to_numpy(dtype=float)
        for threshold in _THRESHOLD_SWEEP:
            predicted = probabilities >= threshold
            records.append(
                {
                    "split_seed": seed,
                    "checkpoint": checkpoint,
                    "threshold": threshold,
                    "precision": precision_score(labels, predicted, zero_division=0),
                    "recall": recall_score(labels, predicted, zero_division=0),
                    "f1": f1_score(labels, predicted, zero_division=0),
                }
            )

    by_seed = pd.DataFrame(records)
    return by_seed.groupby(["checkpoint", "threshold"], as_index=False).agg(
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        n_split_seeds=("split_seed", "nunique"),
    )


def evaluate_model_c_reliability(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return equal-width calibration bins for held-out Model C probabilities."""
    table = _validated_model_c_predictions(predictions)
    bins = (table["predicted_delay_probability"] * _RELIABILITY_BINS).astype("int64")
    table["probability_bin"] = np.minimum(bins, _RELIABILITY_BINS - 1)
    table["bin_lower"] = table["probability_bin"] / _RELIABILITY_BINS
    table["bin_upper"] = (table["probability_bin"] + 1) / _RELIABILITY_BINS

    curve = table.groupby(["checkpoint", "probability_bin"], as_index=False).agg(
        bin_lower=("bin_lower", "first"),
        bin_upper=("bin_upper", "first"),
        mean_predicted_probability=("predicted_delay_probability", "mean"),
        observed_delay_rate=("actual_is_delayed", "mean"),
        n_predictions=("actual_is_delayed", "size"),
    )
    return curve[
        [
            "checkpoint",
            "probability_bin",
            "bin_lower",
            "bin_upper",
            "mean_predicted_probability",
            "observed_delay_rate",
            "n_predictions",
        ]
    ]


def plot_model_c_reliability(reliability: pd.DataFrame, path: Path) -> None:
    """Save the Model C reliability curve with one line per checkpoint."""
    required = {
        "checkpoint",
        "mean_predicted_probability",
        "observed_delay_rate",
    }
    missing = sorted(required - set(reliability.columns))
    if missing:
        raise ContractError(f"Model C reliability data is missing columns: {missing}")
    if reliability.empty:
        raise ContractError("Model C reliability data must be non-empty")

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ArtifactError(
            "matplotlib is required to write the reliability plot"
        ) from error

    figure, axis = plt.subplots(figsize=(7, 7))
    try:
        for checkpoint, rows in reliability.groupby("checkpoint", sort=True):
            axis.plot(
                rows["mean_predicted_probability"],
                rows["observed_delay_rate"],
                marker="o",
                label=f"Checkpoint {checkpoint}",
            )
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect")
        axis.set(
            xlim=(0, 1),
            ylim=(0, 1),
            xlabel="Mean predicted delay probability",
            ylabel="Observed delay rate",
            title="Model C reliability by checkpoint",
        )
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150)
    except OSError as error:
        raise ArtifactError(
            f"cannot write Model C reliability plot {path}: {error}"
        ) from error
    finally:
        plt.close(figure)


def evaluate_model_q_intervals(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report 80% interval coverage, width, and quantile crossings by checkpoint."""
    missing = sorted(set(_Q_COLUMNS) - set(predictions.columns))
    if missing:
        raise ContractError(f"Model Q predictions are missing columns: {missing}")
    if predictions.empty:
        raise ContractError("Model Q predictions must be non-empty")

    table = _validated_prediction_keys(predictions[list(_Q_COLUMNS)].copy())
    for column in (
        "actual_final_delay_h",
        "predicted_final_lower_h",
        "predicted_final_upper_h",
    ):
        values = pd.to_numeric(table[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise ContractError(f"Model Q prediction column {column} must be finite")
        table[column] = values
    if not pd.api.types.is_bool_dtype(table["quantiles_crossed"]):
        raise ContractError("Model Q quantiles_crossed values must be boolean")
    if table["quantiles_crossed"].isna().any():
        raise ContractError("Model Q quantiles_crossed values must be present")
    if (table["predicted_final_lower_h"] > table["predicted_final_upper_h"]).any():
        raise ContractError(
            "Model Q interval lower bounds must not exceed upper bounds"
        )

    table["covered"] = table["actual_final_delay_h"].between(
        table["predicted_final_lower_h"],
        table["predicted_final_upper_h"],
        inclusive="both",
    )
    table["interval_width_h"] = (
        table["predicted_final_upper_h"] - table["predicted_final_lower_h"]
    )
    by_seed = table.groupby(["split_seed", "checkpoint"], as_index=False).agg(
        coverage=("covered", "mean"),
        mean_width_h=("interval_width_h", "mean"),
        quantile_crossing_rate=("quantiles_crossed", "mean"),
    )
    summary = by_seed.groupby("checkpoint", as_index=False).agg(
        coverage_mean=("coverage", "mean"),
        coverage_std=("coverage", "std"),
        mean_width_h=("mean_width_h", "mean"),
        mean_width_std_h=("mean_width_h", "std"),
        quantile_crossing_rate=("quantile_crossing_rate", "mean"),
        n_split_seeds=("split_seed", "nunique"),
    )
    summary["nominal_coverage"] = 0.8
    return summary[
        [
            "checkpoint",
            "nominal_coverage",
            "coverage_mean",
            "coverage_std",
            "mean_width_h",
            "mean_width_std_h",
            "quantile_crossing_rate",
            "n_split_seeds",
        ]
    ]
