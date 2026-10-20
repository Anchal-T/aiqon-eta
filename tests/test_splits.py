"""Tests for reproducible, shipment-level split assignments (T2)."""

from __future__ import annotations

import pandas as pd
import pytest

from src import checkpoints, config, ingest, splits


@pytest.fixture(scope="module")
def stacked() -> pd.DataFrame:
    shipments: pd.DataFrame = ingest.load()
    rows = [
        {
            "shipment_id": shipment["shipment_id"],
            "journey_id": shipment["journey_id"],
            "checkpoint": checkpoint,
        }
        for _, shipment in shipments.iterrows()
        for checkpoint in checkpoints.for_shipment(bool(shipment["is_connecting"]))
    ]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def manifest(stacked: pd.DataFrame) -> pd.DataFrame:
    return splits.create_splits(stacked)


def test_each_seed_assigns_every_shipment_once(stacked, manifest):
    expected_ids = set(stacked["shipment_id"])
    assert set(manifest["split_seed"]) == set(splits.SPLIT_SEEDS)

    for seed, assignments in manifest.groupby("split_seed"):
        assert assignments["shipment_id"].is_unique
        assert set(assignments["shipment_id"]) == expected_ids
        assert set(assignments["outer_split"]) == {"train", "test"}
        assert (
            assignments.loc[assignments["outer_split"] == "train", "fold"].notna().all()
        )
        assert (
            assignments.loc[assignments["outer_split"] == "test", "fold"].isna().all()
        )
        assert set(
            assignments.loc[assignments["outer_split"] == "train", "fold"]
        ) == set(range(config.CV_FOLDS))


def test_t2_all_checkpoint_rows_stay_on_one_side(stacked, manifest):
    expanded = stacked.merge(manifest, on="shipment_id", validate="many_to_many")
    for _, assignments in expanded.groupby(["split_seed", "shipment_id"]):
        assert assignments["outer_split"].nunique() == 1
        assert assignments["fold"].nunique(dropna=False) == 1


def test_outer_split_is_stratified_by_journey(stacked, manifest):
    shipments = stacked[["shipment_id", "journey_id"]].drop_duplicates()
    for _, assignments in manifest.groupby("split_seed"):
        by_journey = assignments.merge(shipments, on="shipment_id").groupby(
            "journey_id"
        )
        assert all(
            set(group["outer_split"]) == {"train", "test"} for _, group in by_journey
        )


def test_split_assignments_are_reproducible(stacked):
    first = splits.create_splits(stacked, seeds=(42, 43))
    second = splits.create_splits(stacked, seeds=(42, 43))
    pd.testing.assert_frame_equal(first, second)


def test_a_shipment_cannot_have_multiple_journeys(stacked):
    broken = stacked.copy()
    shipment_id = broken["shipment_id"].iloc[0]
    index = broken.index[broken["shipment_id"] == shipment_id][0]
    broken.loc[index, "journey_id"] = "different-journey"
    with pytest.raises(ValueError, match="exactly one journey_id"):
        splits.create_splits(broken, seeds=(42,))


def test_invalid_split_settings_are_rejected(stacked):
    with pytest.raises(ValueError, match="test fraction"):
        splits.create_splits(stacked, test_frac=1.0)
    with pytest.raises(ValueError, match="at least two CV folds"):
        splits.create_splits(stacked, n_folds=1)
    with pytest.raises(ValueError, match="non-empty and unique"):
        splits.create_splits(stacked, seeds=())
