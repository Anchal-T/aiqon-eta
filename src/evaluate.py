"""Held-out MAE comparison for Model R and the B2 reference baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.errors import ContractError

_KEYS = ("split_seed", "shipment_id", "checkpoint")
_MODEL_COLUMNS = (*_KEYS, "actual_final_delay_h", "predicted_final_delay_h")
_BASELINE_COLUMNS = (*_KEYS, "baseline", "predicted_final_delay_h")


def compare_model_r_to_b2(
    model_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Report held-out MAE mean and spread by checkpoint across split seeds."""
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

    for table in (model, baseline):
        if table[list(_KEYS)].isna().any().any():
            raise ContractError(
                "prediction seed, shipment ID, and checkpoint are required"
            )
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
            raise ContractError(
                "prediction seeds and checkpoints must be finite integers"
            )
        table["split_seed"] = seeds.astype("int64")
        table["checkpoint"] = checkpoints.astype("int64")

    if model.duplicated(list(_KEYS)).any() or baseline.duplicated(list(_KEYS)).any():
        raise ContractError(
            "each prediction must be unique per seed, shipment, checkpoint"
        )

    model = model.rename(
        columns={
            "actual_final_delay_h": "actual_h",
            "predicted_final_delay_h": "model_pred_h",
        }
    )
    baseline = baseline.rename(columns={"predicted_final_delay_h": "b2_pred_h"})
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
    joined["model_error_h"] = (joined["actual_h"] - joined["model_pred_h"]).abs()
    joined["b2_error_h"] = (joined["actual_h"] - joined["b2_pred_h"]).abs()

    by_seed = joined.groupby(["split_seed", "checkpoint"], as_index=False).agg(
        model_mae_h=("model_error_h", "mean"),
        b2_mae_h=("b2_error_h", "mean"),
    )
    by_seed["model_beats_b2"] = by_seed["model_mae_h"] < by_seed["b2_mae_h"]
    summary = by_seed.groupby("checkpoint", as_index=False).agg(
        model_mae_mean_h=("model_mae_h", "mean"),
        model_mae_std_h=("model_mae_h", "std"),
        b2_mae_mean_h=("b2_mae_h", "mean"),
        b2_mae_std_h=("b2_mae_h", "std"),
        model_beats_b2_seeds=("model_beats_b2", "sum"),
        n_split_seeds=("split_seed", "nunique"),
    )
    summary["skill_score"] = np.where(
        summary["b2_mae_mean_h"] > 0,
        1.0 - summary["model_mae_mean_h"] / summary["b2_mae_mean_h"],
        np.nan,
    )
    return summary
