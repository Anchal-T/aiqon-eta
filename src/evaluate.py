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
    return joined.drop(columns="_merge")


def _model_r_metrics_by_seed(
    joined: pd.DataFrame,
    group_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    joined["model_error_h"] = (joined["actual_h"] - joined["model_pred_h"]).abs()
    joined["b2_error_h"] = (joined["actual_h"] - joined["b2_pred_h"]).abs()
    joined["model_squared_error_h"] = (joined["actual_h"] - joined["model_pred_h"]) ** 2
    joined["b2_squared_error_h"] = (joined["actual_h"] - joined["b2_pred_h"]) ** 2
    by_seed = joined.groupby(
        ["split_seed", "checkpoint", *group_columns], as_index=False
    ).agg(
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


def _summarize_model_r_metrics(
    by_seed: pd.DataFrame,
    group_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    by_seed = by_seed.copy()
    by_seed["model_beats_b2"] = by_seed["model_mae_h"] < by_seed["b2_mae_h"]
    aggregations = {}
    for model in ("model", "b2"):
        for metric in ("mae", "median_ae", "p90_ae", "rmse"):
            source = f"{model}_{metric}_h"
            aggregations[f"{model}_{metric}_mean_h"] = (source, "mean")
            aggregations[f"{model}_{metric}_std_h"] = (source, "std")
    aggregations.update(
        model_beats_b2_seeds=("model_beats_b2", "sum"),
        n_split_seeds=("split_seed", "nunique"),
    )
    summary = by_seed.groupby(["checkpoint", *group_columns], as_index=False).agg(
        **aggregations
    )
    summary["skill_score"] = np.where(
        summary["b2_mae_mean_h"] > 0,
        1.0 - summary["model_mae_mean_h"] / summary["b2_mae_mean_h"],
        np.nan,
    )
    return summary


def compare_model_r_to_b2(
    model_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Report held-out regression errors by checkpoint across split seeds."""
    by_seed = _model_r_metrics_by_seed(
        _paired_model_r_b2_predictions(model_predictions, baseline_predictions)
    )
    return _summarize_model_r_metrics(by_seed)


def build_evaluation_group_metadata(
    shipments: pd.DataFrame,
    truth: pd.DataFrame,
    hidden_mask: pd.DataFrame,
) -> pd.DataFrame:
    """Validate evaluation-only inputs and derive one row of cuts per shipment."""
    required_shipments = {"shipment_id", "journey_id", "hub"}
    missing_shipments = sorted(required_shipments - set(shipments.columns))
    if missing_shipments:
        raise ContractError(
            f"shipment metadata is missing columns: {missing_shipments}"
        )
    if {"shipment_id", "missed_connection"} - set(truth.columns):
        raise ContractError("truth metadata requires shipment_id and missed_connection")
    if "shipment_id" not in hidden_mask.columns:
        raise ContractError("hidden-stage metadata is missing shipment_id")

    metadata = shipments[["shipment_id", "journey_id", "hub"]].copy()
    metadata["shipment_id"] = metadata["shipment_id"].astype("string")
    if metadata[["shipment_id", "journey_id"]].isna().any().any():
        raise ContractError("shipment metadata requires shipment IDs and journey IDs")
    if metadata["shipment_id"].duplicated().any():
        raise ContractError("shipment metadata IDs must be unique")
    metadata["journey_id"] = metadata["journey_id"].astype("string")

    missed = truth[["shipment_id", "missed_connection"]].copy()
    missed["shipment_id"] = missed["shipment_id"].astype("string")
    if missed.isna().any().any() or missed["shipment_id"].duplicated().any():
        raise ContractError(
            "truth metadata IDs and missed_connection values must be unique and present"
        )
    if not missed["missed_connection"].isin([True, False]).all():
        raise ContractError("missed_connection truth values must be boolean")
    missed["missed_connection"] = missed["missed_connection"].astype(bool)

    shipment_ids = set(metadata["shipment_id"])
    if set(missed["shipment_id"]) != shipment_ids:
        raise ContractError("shipment and truth metadata IDs do not match")
    hidden_ids = hidden_mask["shipment_id"].astype("string")
    if hidden_ids.isna().any() or not set(hidden_ids).issubset(shipment_ids):
        raise ContractError("hidden-stage metadata contains an unknown shipment ID")

    metadata = metadata.merge(missed, on="shipment_id", validate="one_to_one")
    metadata["connection_type"] = np.where(
        metadata["hub"].notna(), "connecting", "direct"
    )
    metadata["missed_connection"] = np.where(
        metadata["missed_connection"], "true", "false"
    )
    metadata["hidden_stage"] = np.where(
        metadata["shipment_id"].isin(hidden_ids),
        "with_hidden_stage",
        "without_hidden_stage",
    )
    return metadata.drop(columns="hub")


def _grouped_evaluation_rows(
    table: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    required = (
        "shipment_id",
        "journey_id",
        "connection_type",
        "missed_connection",
        "hidden_stage",
    )
    missing = sorted(set(required) - set(metadata.columns))
    if missing:
        raise ContractError(f"evaluation group metadata is missing columns: {missing}")
    metadata = metadata[list(required)].copy()
    metadata["shipment_id"] = metadata["shipment_id"].astype("string")
    if metadata.isna().any().any() or metadata["shipment_id"].duplicated().any():
        raise ContractError(
            "evaluation group metadata must be complete and unique per shipment"
        )

    joined = table.merge(
        metadata,
        on="shipment_id",
        how="left",
        validate="many_to_one",
        indicator=True,
    )
    if not joined["_merge"].eq("both").all():
        missing_ids = sorted(
            joined.loc[joined["_merge"] != "both", "shipment_id"].unique()
        )
        raise ContractError(
            f"evaluation metadata is missing for predicted shipments: {missing_ids[:5]}"
        )
    joined = joined.drop(columns="_merge")

    return pd.concat(
        [
            joined[table.columns].assign(
                group_type=group_type,
                group_value=joined[column].astype("string"),
            )
            for group_type, column in (
                ("journey", "journey_id"),
                ("connection_type", "connection_type"),
                ("missed_connection", "missed_connection"),
                ("hidden_stage", "hidden_stage"),
            )
        ],
        ignore_index=True,
    )


def compare_model_r_to_b2_by_group(
    model_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    group_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Report Model R vs B2 metrics by journey and shipment subgroup."""
    joined = _paired_model_r_b2_predictions(model_predictions, baseline_predictions)
    grouped = _grouped_evaluation_rows(joined, group_metadata)
    by_seed = _model_r_metrics_by_seed(grouped, ("group_type", "group_value"))
    return _summarize_model_r_metrics(by_seed, ("group_type", "group_value"))


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


def _model_c_metrics_by_seed(
    table: pd.DataFrame,
    group_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    records = []
    grouping = ["split_seed", "checkpoint", *group_columns]
    for key, rows in table.groupby(grouping, sort=True):
        seed, checkpoint, *group_values = key
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
                **dict(zip(group_columns, group_values, strict=True)),
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


def _summarize_model_c_classification_metrics(
    by_seed: pd.DataFrame,
    group_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    return by_seed.groupby(["checkpoint", *group_columns], as_index=False).agg(
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


def evaluate_model_c_classification_metrics(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize held-out classification metrics by checkpoint across seeds."""
    by_seed = _model_c_metrics_by_seed(_validated_model_c_predictions(predictions))
    return _summarize_model_c_classification_metrics(by_seed)


def evaluate_model_c_classification_metrics_by_group(
    predictions: pd.DataFrame,
    group_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize held-out classification metrics by checkpoint and shipment cut."""
    table = _grouped_evaluation_rows(
        _validated_model_c_predictions(predictions), group_metadata
    )
    by_seed = _model_c_metrics_by_seed(table, ("group_type", "group_value"))
    return _summarize_model_c_classification_metrics(
        by_seed, ("group_type", "group_value")
    )


def _model_c_threshold_sweep_by_seed(
    table: pd.DataFrame,
    group_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    records = []
    grouping = ["split_seed", "checkpoint", *group_columns]
    for key, rows in table.groupby(grouping, sort=True):
        seed, checkpoint, *group_values = key
        labels = rows["actual_is_delayed"].to_numpy(dtype=int)
        probabilities = rows["predicted_delay_probability"].to_numpy(dtype=float)
        group_values = dict(zip(group_columns, group_values, strict=True))
        for threshold in _THRESHOLD_SWEEP:
            predicted = probabilities >= threshold
            records.append(
                {
                    "split_seed": seed,
                    "checkpoint": checkpoint,
                    **group_values,
                    "threshold": threshold,
                    "precision": precision_score(labels, predicted, zero_division=0),
                    "recall": recall_score(labels, predicted, zero_division=0),
                    "f1": f1_score(labels, predicted, zero_division=0),
                }
            )
    return pd.DataFrame(records)


def _summarize_model_c_threshold_sweep(
    by_seed: pd.DataFrame,
    group_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    return by_seed.groupby(
        ["checkpoint", *group_columns, "threshold"], as_index=False
    ).agg(
        precision_mean=("precision", "mean"),
        precision_std=("precision", "std"),
        recall_mean=("recall", "mean"),
        recall_std=("recall", "std"),
        f1_mean=("f1", "mean"),
        f1_std=("f1", "std"),
        n_split_seeds=("split_seed", "nunique"),
    )


def evaluate_model_c_threshold_sweep(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report precision, recall, and F1 over fixed thresholds by checkpoint."""
    by_seed = _model_c_threshold_sweep_by_seed(
        _validated_model_c_predictions(predictions)
    )
    return _summarize_model_c_threshold_sweep(by_seed)


def evaluate_model_c_threshold_sweep_by_group(
    predictions: pd.DataFrame,
    group_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Report threshold metrics by checkpoint and shipment cut."""
    table = _grouped_evaluation_rows(
        _validated_model_c_predictions(predictions), group_metadata
    )
    by_seed = _model_c_threshold_sweep_by_seed(table, ("group_type", "group_value"))
    return _summarize_model_c_threshold_sweep(by_seed, ("group_type", "group_value"))


def _model_c_reliability(
    table: pd.DataFrame,
    group_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    bins = (table["predicted_delay_probability"] * _RELIABILITY_BINS).astype("int64")
    table = table.copy()
    table["probability_bin"] = np.minimum(bins, _RELIABILITY_BINS - 1)
    table["bin_lower"] = table["probability_bin"] / _RELIABILITY_BINS
    table["bin_upper"] = (table["probability_bin"] + 1) / _RELIABILITY_BINS

    curve = table.groupby(
        ["checkpoint", *group_columns, "probability_bin"], as_index=False
    ).agg(
        bin_lower=("bin_lower", "first"),
        bin_upper=("bin_upper", "first"),
        mean_predicted_probability=("predicted_delay_probability", "mean"),
        observed_delay_rate=("actual_is_delayed", "mean"),
        n_predictions=("actual_is_delayed", "size"),
    )
    return curve[
        [
            "checkpoint",
            *group_columns,
            "probability_bin",
            "bin_lower",
            "bin_upper",
            "mean_predicted_probability",
            "observed_delay_rate",
            "n_predictions",
        ]
    ]


def evaluate_model_c_reliability(predictions: pd.DataFrame) -> pd.DataFrame:
    """Return equal-width calibration bins for held-out Model C probabilities."""
    return _model_c_reliability(_validated_model_c_predictions(predictions))


def evaluate_model_c_reliability_by_group(
    predictions: pd.DataFrame,
    group_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Return reliability bins by checkpoint and shipment cut."""
    table = _grouped_evaluation_rows(
        _validated_model_c_predictions(predictions), group_metadata
    )
    return _model_c_reliability(table, ("group_type", "group_value"))


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


def _validated_model_q_predictions(predictions: pd.DataFrame) -> pd.DataFrame:
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
    return table


def _model_q_metrics_by_seed(
    table: pd.DataFrame,
    group_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    table = table.copy()
    table["covered"] = table["actual_final_delay_h"].between(
        table["predicted_final_lower_h"],
        table["predicted_final_upper_h"],
        inclusive="both",
    )
    table["interval_width_h"] = (
        table["predicted_final_upper_h"] - table["predicted_final_lower_h"]
    )
    return table.groupby(
        ["split_seed", "checkpoint", *group_columns], as_index=False
    ).agg(
        coverage=("covered", "mean"),
        mean_width_h=("interval_width_h", "mean"),
        quantile_crossing_rate=("quantiles_crossed", "mean"),
    )


def _summarize_model_q_metrics(
    by_seed: pd.DataFrame,
    group_columns: tuple[str, ...] = (),
) -> pd.DataFrame:
    summary = by_seed.groupby(["checkpoint", *group_columns], as_index=False).agg(
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
            *group_columns,
            "nominal_coverage",
            "coverage_mean",
            "coverage_std",
            "mean_width_h",
            "mean_width_std_h",
            "quantile_crossing_rate",
            "n_split_seeds",
        ]
    ]


def evaluate_model_q_intervals(predictions: pd.DataFrame) -> pd.DataFrame:
    """Report 80% interval metrics by checkpoint across split seeds."""
    by_seed = _model_q_metrics_by_seed(_validated_model_q_predictions(predictions))
    return _summarize_model_q_metrics(by_seed)


def evaluate_model_q_intervals_by_group(
    predictions: pd.DataFrame,
    group_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Report 80% interval metrics by checkpoint and shipment cut."""
    table = _grouped_evaluation_rows(
        _validated_model_q_predictions(predictions), group_metadata
    )
    by_seed = _model_q_metrics_by_seed(table, ("group_type", "group_value"))
    return _summarize_model_q_metrics(by_seed, ("group_type", "group_value"))


def plot_model_r_missingness_sweep(metrics: pd.DataFrame, path: Path) -> None:
    """Plot mean held-out Model R MAE by missingness rate and checkpoint."""
    required = {
        "missing_rate",
        "checkpoint",
        "model_mae_mean_h",
        "model_mae_std_h",
    }
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ContractError(f"missingness sweep metrics are missing columns: {missing}")
    if metrics.empty:
        raise ContractError("missingness sweep metrics must be non-empty and unique")

    table = metrics.copy()
    for column in ("missing_rate", "checkpoint", "model_mae_mean_h"):
        values = pd.to_numeric(table[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise ContractError(f"missingness sweep column {column} must be finite")
        table[column] = values
    if not table["missing_rate"].between(0.0, 1.0).all():
        raise ContractError("missing rates must be between 0 and 1")
    if (table["checkpoint"] % 1 != 0).any():
        raise ContractError("missingness sweep checkpoints must be integers")

    raw_spread = table["model_mae_std_h"]
    spread = pd.to_numeric(raw_spread, errors="coerce")
    invalid_spread = raw_spread.notna() & (spread.isna() | ~np.isfinite(spread))
    if invalid_spread.any() or (spread.dropna() < 0).any():
        raise ContractError(
            "missingness sweep MAE spread must be finite and non-negative"
        )
    table["model_mae_std_h"] = spread.fillna(0)
    if table.duplicated(["missing_rate", "checkpoint"]).any():
        raise ContractError("missingness sweep metrics must be non-empty and unique")

    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ArtifactError(
            "matplotlib is required to write the missingness plot"
        ) from error

    figure, axis = plt.subplots(figsize=(8, 5))
    try:
        for checkpoint, rows in table.groupby("checkpoint", sort=True):
            rows = rows.sort_values("missing_rate")
            axis.errorbar(
                rows["missing_rate"],
                rows["model_mae_mean_h"],
                yerr=rows["model_mae_std_h"],
                marker="o",
                capsize=3,
                label=f"Checkpoint {checkpoint}",
            )
        axis.set(
            xlabel="Missing rate",
            ylabel="Held-out MAE (hours)",
            title="Model R MAE vs missingness rate",
        )
        axis.set_xticks(sorted(table["missing_rate"].unique()))
        axis.legend(title="Checkpoint")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150)
    except OSError as error:
        raise ArtifactError(
            f"cannot write missingness sweep plot {path}: {error}"
        ) from error
    finally:
        plt.close(figure)
