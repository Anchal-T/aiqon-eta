"""Tests for reproducible, shipment-level split assignments (T2)."""

from __future__ import annotations

from dataclasses import replace

import pandas as pd
import pytest
from src.pipeline import ArtifactPaths

from src import checkpoints, config, ingest, splits


@pytest.fixture(scope="module")
def stacked() -> pd.DataFrame:
    shipments: pd.DataFrame = ingest.load(ArtifactPaths().raw_shipments)
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
    assert set(manifest["split_seed"]) == set(config.DEFAULT_RUN_SETTINGS.split_seeds)

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
        ) == set(range(config.DEFAULT_RUN_SETTINGS.cv_folds))


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
    settings = replace(config.DEFAULT_RUN_SETTINGS, seed=42, n_split_seeds=2)
    first = splits.create_splits(stacked, settings=settings)
    second = splits.create_splits(stacked, settings=settings)
    pd.testing.assert_frame_equal(first, second)


def test_a_shipment_cannot_have_multiple_journeys(stacked):
    broken = stacked.copy()
    shipment_id = broken["shipment_id"].iloc[0]
    index = broken.index[broken["shipment_id"] == shipment_id][0]
    broken.loc[index, "journey_id"] = "different-journey"
    settings = replace(config.DEFAULT_RUN_SETTINGS, seed=42, n_split_seeds=1)
    with pytest.raises(ValueError, match="exactly one journey_id"):
        splits.create_splits(broken, settings=settings)


def test_invalid_split_settings_are_rejected(stacked):
    with pytest.raises(ValueError, match="test fraction"):
        settings = replace(config.DEFAULT_RUN_SETTINGS, test_frac=1.0)
        splits.create_splits(stacked, settings=settings)
    with pytest.raises(ValueError, match="at least two CV folds"):
        settings = replace(config.DEFAULT_RUN_SETTINGS, cv_folds=1)
        splits.create_splits(stacked, settings=settings)
    with pytest.raises(ValueError, match="non-empty and unique"):
        settings = replace(config.DEFAULT_RUN_SETTINGS, n_split_seeds=0)
        splits.create_splits(stacked, settings=settings)
