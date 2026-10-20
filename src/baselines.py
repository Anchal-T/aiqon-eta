"""Held-out reference predictions for the ETA feature table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.errors import ContractError, SplitError

_REQUIRED_STACKED = {
    "shipment_id",
    "journey_id",
    "checkpoint",
    "final_delay_h",
    "delay_so_far",
}
_REQUIRED_SPLITS = {"split_seed", "shipment_id", "outer_split"}
PREDICTION_COLUMNS = (
    "split_seed",
    "shipment_id",
    "checkpoint",
    "baseline",
    "predicted_final_delay_h",
)


def predict(stacked: pd.DataFrame, split_manifest: pd.DataFrame) -> pd.DataFrame:
    """Predict final delay for each outer-test checkpoint with B0, B1, and B2.

    B1's journey means are fitted on distinct outer-training shipments only.
    The returned table contains held-out rows only, ready for evaluation.
    """
    missing_stacked = sorted(_REQUIRED_STACKED - set(stacked.columns))
    if missing_stacked:
        raise ContractError(f"stacked table is missing columns: {missing_stacked}")
    missing_splits = sorted(_REQUIRED_SPLITS - set(split_manifest.columns))
    if missing_splits:
        raise ContractError(f"split manifest is missing columns: {missing_splits}")
    if stacked.empty or split_manifest.empty:
        raise ContractError("stacked table and split manifest must be non-empty")

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

    shipment_targets = rows[
        ["shipment_id", "journey_id", "final_delay_h"]
    ].drop_duplicates()
    if shipment_targets["shipment_id"].duplicated().any():
        raise ContractError("each shipment must have one journey_id and final_delay_h")

    assignments = split_manifest[list(_REQUIRED_SPLITS)].copy()
    if assignments.isna().any().any():
        raise ContractError("split seed, shipment_id, and outer_split must be present")
    assignments["shipment_id"] = assignments["shipment_id"].astype("string")
    seeds = pd.to_numeric(assignments["split_seed"], errors="coerce")
    if seeds.isna().any() or not np.isfinite(seeds).all() or (seeds % 1 != 0).any():
        raise ContractError("split_seed values must be finite integers")
    assignments["split_seed"] = seeds.astype("int64")
    if assignments.duplicated(["split_seed", "shipment_id"]).any():
        raise ContractError("each shipment must have one split assignment per seed")
    if not assignments["outer_split"].isin(("train", "test")).all():
        raise ContractError("outer_split values must be 'train' or 'test'")

    expected_ids = set(shipment_targets["shipment_id"])
    records = []
    for seed, seed_assignments in assignments.groupby("split_seed", sort=False):
        assigned_ids = set(seed_assignments["shipment_id"])
        if assigned_ids != expected_ids:
            missing_ids = sorted(expected_ids - assigned_ids)
            unexpected_ids = sorted(assigned_ids - expected_ids)
            raise ContractError(
                f"split seed {seed} shipment IDs do not match stacked data; "
                f"missing: {missing_ids[:5]}, unexpected: {unexpected_ids[:5]}"
            )

        train_ids = set(
            seed_assignments.loc[
                seed_assignments["outer_split"] == "train", "shipment_id"
            ]
        )
        test_ids = set(
            seed_assignments.loc[
                seed_assignments["outer_split"] == "test", "shipment_id"
            ]
        )
        if not train_ids or not test_ids:
            raise SplitError(f"split seed {seed} must have train and test shipments")

        train_targets = shipment_targets[
            shipment_targets["shipment_id"].isin(train_ids)
        ]
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

        predictions = (
            ("B0", pd.Series(0.0, index=test_rows.index)),
            ("B1", test_rows["journey_mean_h"]),
            ("B2", test_rows["delay_so_far"]),
        )
        for name, values in predictions:
            result = test_rows[["shipment_id", "checkpoint"]].copy()
            result.insert(0, "split_seed", seed)
            result["baseline"] = name
            result["predicted_final_delay_h"] = values.to_numpy()
            records.append(result)

    return pd.concat(records, ignore_index=True).reindex(columns=PREDICTION_COLUMNS)
