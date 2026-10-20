"""Tests for checkpoint visibility and the stacked feature table."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from src import checkpoints, config, delays, features, ingest, missingness
from src.pipeline import ArtifactPaths


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return ingest.load(ArtifactPaths().raw_shipments)


@pytest.fixture(scope="module")
def stacked(raw: pd.DataFrame) -> pd.DataFrame:
    observed, _ = missingness.hide(raw)
    return features.build(observed, delays.compute(observed))


def _checkpoint_row(table: pd.DataFrame, shipment_id: str, checkpoint: int) -> Any:
    return table.loc[
        (table["shipment_id"] == shipment_id) & (table["checkpoint"] == checkpoint)
    ].iloc[0]


def test_checkpoint_visibility_and_row_count(raw, stacked):
    assert len(stacked) == sum(
        len(checkpoints.for_shipment(bool(row.is_connecting)))
        for row in raw.itertuples()
    )
    for shipment in raw.itertuples():
        rows = stacked[stacked["shipment_id"] == shipment.shipment_id]
        expected = checkpoints.for_shipment(bool(shipment.is_connecting))
        assert tuple(rows["checkpoint"]) == expected

    assert checkpoints.visible_stages(1, True) == ("departure",)
    assert checkpoints.visible_stages(4, False) == ("departure", "port_arrival")
    with pytest.raises(ValueError, match="does not apply"):
        checkpoints.visible_stages(2, False)


def test_features_are_revealed_only_when_the_checkpoint_allows(raw, stacked):
    shipment = raw.loc[raw["shipment_id"] == config.SPECIMEN_ID].iloc[0]
    c1 = _checkpoint_row(stacked, config.SPECIMEN_ID, 1)
    assert pd.notna(c1["delay_departure"])
    assert pd.isna(c1["delay_hub_arrival"])
    assert pd.isna(c1["weather_leg1"])
    assert pd.isna(c1["congestion_hub"])
    assert pd.isna(c1["congestion_port"])
    assert pd.isna(c1["port_done_missing"])

    c2 = _checkpoint_row(stacked, config.SPECIMEN_ID, 2)
    assert c2["weather_leg1"] == shipment["weather_leg1"]
    assert c2["congestion_hub"] == shipment["congestion_hub"]
    assert c2["slack_remaining"] == pytest.approx(
        c2["nominal_slack_h"] - c2["delay_hub_arrival"]
    )

    c4 = _checkpoint_row(stacked, config.SPECIMEN_ID, 4)
    assert c4["weather_leg2"] == shipment["weather_leg2"]
    assert c4["congestion_port"] == shipment["congestion_port"]
    assert c4["arrival_dow"] == shipment["arrival_dow"]
    assert c4["slot_margin"] == pytest.approx(
        config.SLOT_WINDOW_H - c4["delay_port_arrival"]
    )
    assert pd.isna(c4["port_done_missing"])


def test_direct_shipments_never_have_hub_features(raw, stacked):
    shipment_id = raw.loc[~raw["is_connecting"], "shipment_id"].iloc[0]
    rows = stacked[stacked["shipment_id"] == shipment_id]
    assert tuple(rows["checkpoint"]) == checkpoints.DIRECT_CHECKPOINTS
    assert rows["hub"].eq(-1).all()
    for column in (
        "hub_buffer_h",
        "nominal_slack_h",
        "planned_leg2_h",
        "delay_hub_arrival",
        "added_hub_arrival",
        "delay_hub_departure",
        "added_hub_departure",
        "weather_leg2",
        "congestion_hub",
        "slack_remaining",
    ):
        assert rows[column].isna().all(), column


def test_hidden_late_stages_fall_back_and_set_flags(raw):
    observed = raw.copy()
    shipment_id = config.SPECIMEN_ID
    index = observed.index[observed["shipment_id"] == shipment_id][0]
    observed.loc[index, ["act_port_done", "act_customs"]] = pd.NaT
    stacked = features.build(observed, delays.compute(observed))

    c5 = _checkpoint_row(stacked, shipment_id, 5)
    c6 = _checkpoint_row(stacked, shipment_id, 6)
    assert c5["port_done_missing"] == 1
    assert pd.isna(c5["customs_missing"])
    assert c5["delay_so_far"] == c5["delay_port_arrival"]
    assert c5["n_observed_stages"] == 4
    assert c6["port_done_missing"] == 1
    assert c6["customs_missing"] == 1
    assert c6["delay_so_far"] == c6["delay_port_arrival"]
    assert c6["n_observed_stages"] == 4


def test_observed_and_delay_mismatch_reports_shipment_ids(raw):
    delay_table = delays.compute(raw).iloc[1:]

    with pytest.raises(ingest.ContractError, match=config.SPECIMEN_ID):
        features.build(raw, delay_table)


def test_delay_threshold_comes_from_run_settings(raw):
    observed, _ = missingness.hide(raw)
    settings = replace(config.DEFAULT_RUN_SETTINGS, delay_threshold_h=1_000.0)
    table = features.build(observed, delays.compute(observed), settings=settings)
    assert not table["is_delayed"].any()


def test_targets_are_derived_from_delay_so_far(stacked):
    np.testing.assert_allclose(
        stacked["remaining_delay_h"],
        stacked["final_delay_h"] - stacked["delay_so_far"],
    )
    assert "shipment_id" not in features.FEATURE_COLUMNS
    assert "delay_departure" in features.FEATURE_COLUMNS
    assert "added_departure" not in features.FEATURE_COLUMNS
    assert not set(features.TARGET_COLUMNS) & set(features.FEATURE_COLUMNS)
    assert not any(column.endswith("delivery") for column in features.FEATURE_COLUMNS)


@pytest.mark.parametrize("checkpoint", checkpoints.CONNECTING_CHECKPOINTS)
def test_future_junk_does_not_change_features(checkpoint, raw):
    shipment_id = config.SPECIMEN_ID
    visible = set(checkpoints.visible_stages(checkpoint, True))
    baseline = features.build(raw, delays.compute(raw))

    junk = raw.copy()
    for stage in config.STAGE_ORDER:
        if stage not in visible:
            column = f"act_{stage}"
            present = junk[column].notna()
            junk.loc[present, column] += pd.Timedelta(days=7)

    environment_features = (
        "weather_leg1",
        "weather_leg2",
        "congestion_hub",
        "congestion_port",
        "arrival_dow",
    )
    visible_environment = {
        feature
        for feature in environment_features
        if checkpoints.feature_is_visible(feature, checkpoint, True)
    }
    for column in environment_features:
        if column not in visible_environment:
            if column == "arrival_dow":
                junk[column] = (junk[column] + 1) % 7
            else:
                junk[column] = junk[column] + 0.123

    future_junk = features.build(junk, delays.compute(junk))
    expected = _checkpoint_row(baseline, shipment_id, checkpoint)[
        list(features.FEATURE_COLUMNS)
    ]
    actual = _checkpoint_row(future_junk, shipment_id, checkpoint)[
        list(features.FEATURE_COLUMNS)
    ]
    pd.testing.assert_series_equal(actual, expected)
