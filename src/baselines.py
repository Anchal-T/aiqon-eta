"""Held-out reference predictions for the ETA feature table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import splits
from src.errors import ContractError, SplitError

_REQUIRED_STACKED = {
    "shipment_id",
    "journey_id",
    "checkpoint",
    "final_delay_h",
    "delay_so_far",
}

PREDICTION_COLUMNS = (
    "split_seed",
    "shipment_id",
    "checkpoint",
    "baseline",
    "predicted_final_delay_h",
)


def _validated_rows(stacked: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    missing = sorted(_REQUIRED_STACKED - set(stacked.columns))
    if missing:
        raise ContractError(f"stacked table is missing columns: {missing}")
    if stacked.empty:
        raise ContractError("stacked table must be non-empty")

    rows = stacked[list(_REQUIRED_STACKED)].copy()
    if rows[["shipment_id", "journey_id", "checkpoint"]].isna().any().any():
        raise ContractError(
            "stacked shipment, journey, and checkpoint values are required"
        )
    rows["shipment_id"] = rows["shipment_id"].astype("string")
    if rows.duplicated(["shipment_id", "checkpoint"]).any():
        raise ContractError("stacked shipment/checkpoint rows must be unique")

    for column in ("final_delay_h", "delay_so_far"):
        values = pd.to_numeric(rows[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise ContractError(f"stacked {column} values must be finite numbers")
        rows[column] = values

    targets = rows[["shipment_id", "journey_id", "final_delay_h"]].drop_duplicates()
    if targets["shipment_id"].duplicated().any():
        raise ContractError("each shipment must have one journey_id and final_delay_h")
    return rows, targets


def _predict_seed(
    rows: pd.DataFrame,
    shipment_targets: pd.DataFrame,
    assignments: pd.DataFrame,
    seed: int,
) -> pd.DataFrame:
    expected_ids = set(shipment_targets["shipment_id"])
    assigned_ids = set(assignments["shipment_id"])
    if assigned_ids != expected_ids:
        missing = sorted(expected_ids - assigned_ids)
        unexpected = sorted(assigned_ids - expected_ids)
        raise ContractError(
            f"split seed {seed} shipment IDs do not match stacked data; "
            f"missing: {missing[:5]}, unexpected: {unexpected[:5]}"
        )

    train_ids = set(
        assignments.loc[assignments["outer_split"] == "train", "shipment_id"]
    )
    test_ids = set(assignments.loc[assignments["outer_split"] == "test", "shipment_id"])
    if not train_ids or not test_ids:
        raise SplitError(f"split seed {seed} must have train and test shipments")

    train_targets = shipment_targets[shipment_targets["shipment_id"].isin(train_ids)]
    journey_means = train_targets.groupby("journey_id")["final_delay_h"].mean()
    test_rows = rows[rows["shipment_id"].isin(test_ids)].copy()
    test_rows["journey_mean_h"] = test_rows["journey_id"].map(journey_means)
    if test_rows["journey_mean_h"].isna().any():
        unseen = sorted(
            test_rows.loc[test_rows["journey_mean_h"].isna(), "journey_id"]
            .astype(str)
            .unique()
        )
        raise SplitError(
            f"no outer-training journey mean for split seed {seed}: {unseen[:5]}"
        )

    predictions = []
    for name, values in (
        ("B0", pd.Series(0.0, index=test_rows.index)),
        ("B1", test_rows["journey_mean_h"]),
        ("B2", test_rows["delay_so_far"]),
    ):
        result = test_rows[["shipment_id", "checkpoint"]].copy()
        result.insert(0, "split_seed", seed)
        result["baseline"] = name
        result["predicted_final_delay_h"] = values.to_numpy()
        predictions.append(result)
    return pd.concat(predictions, ignore_index=True)


def predict(stacked: pd.DataFrame, split_manifest: pd.DataFrame) -> pd.DataFrame:
    """Predict final delay for held-out checkpoints with B0, B1, and B2."""
    rows, shipment_targets = _validated_rows(stacked)
    assignments = splits.validate_assignments(split_manifest)
    predictions = [
        _predict_seed(rows, shipment_targets, seed_assignments, int(seed))
        for seed, seed_assignments in assignments.groupby("split_seed", sort=False)
    ]
    return pd.concat(predictions, ignore_index=True).reindex(columns=PREDICTION_COLUMNS)
