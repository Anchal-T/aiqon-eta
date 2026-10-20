"""Tests for loading the raw shipments and checking the data contract."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, ingest


@pytest.fixture(scope="module")
def raw():
    return ingest.load()


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def test_the_raw_file_loads(raw):
    assert len(raw) == config.N_SHIPMENTS
    assert raw["shipment_id"].is_unique


def test_columns_have_the_contract_types(raw):
    for column in ingest.STRING_COLUMNS:
        assert isinstance(raw[column].dtype, pd.StringDtype), column
    for column in ingest.FLOAT_COLUMNS:
        assert raw[column].dtype == np.dtype("float64"), column
    for column in ingest.INT_COLUMNS:
        assert raw[column].dtype == np.dtype("int64"), column
    for column in ingest.TIMESTAMP_COLUMNS:
        assert pd.api.types.is_datetime64_any_dtype(raw[column]), column


def test_journey_id_is_derived_from_the_route(raw):
    expected = raw.apply(
        lambda row: ">".join(
            str(part)
            for part in (row["origin"], row["hub"], row["destination"])
            if pd.notna(part)
        ),
        axis=1,
    ).astype("string")
    pd.testing.assert_series_equal(raw["journey_id"], expected, check_names=False)


def test_is_connecting_is_the_hub(raw):
    pd.testing.assert_series_equal(
        raw["is_connecting"], raw["hub"].notna(), check_names=False
    )


def test_the_hub_columns_are_empty_for_direct_shipments(raw):
    direct = ~raw["is_connecting"]
    for stage in config.HUB_STAGES:
        assert raw.loc[direct, f"sched_{stage}"].isna().all()
        assert raw.loc[direct, f"act_{stage}"].isna().all()


# ---------------------------------------------------------------------------
# The contract checks
# ---------------------------------------------------------------------------


def test_a_schedule_out_of_stage_order_is_rejected(raw):
    broken = raw.copy()
    row = broken.index[0]
    broken.loc[row, "sched_port_arrival"] = broken.loc[row, "sched_departure"]
    with pytest.raises(ingest.ContractError, match="stage order"):
        ingest.validate(broken)


def test_a_hub_column_set_on_a_direct_shipment_is_rejected(raw):
    direct = raw.index[~raw["is_connecting"]][0]
    broken = raw.copy()
    broken.loc[direct, "sched_hub_arrival"] = broken.loc[direct, "sched_departure"]
    with pytest.raises(ingest.ContractError, match="direct shipment"):
        ingest.validate(broken)


def test_a_hub_column_missing_on_a_connecting_shipment_is_rejected(raw):
    connecting = raw.index[raw["is_connecting"]][0]
    broken = raw.copy()
    broken.loc[connecting, "act_hub_departure"] = pd.NaT
    with pytest.raises(ingest.ContractError, match="connecting shipment"):
        ingest.validate(broken)


def test_a_missing_actual_timestamp_is_rejected(raw):
    broken = raw.copy()
    broken.loc[broken.index[0], "act_port_arrival"] = pd.NaT
    with pytest.raises(ingest.ContractError, match="actual timestamps are missing"):
        ingest.validate(broken)


def test_a_non_finite_final_delay_is_rejected(raw):
    broken = raw.copy()
    broken.loc[broken.index[0], "act_delivery"] = pd.NaT
    with pytest.raises(ingest.ContractError, match="not finite"):
        ingest.validate(broken, complete_actuals=False)


def test_weather_outside_its_range_is_rejected(raw):
    broken = raw.copy()
    broken.loc[broken.index[0], "weather_leg1"] = 1.5
    with pytest.raises(ingest.ContractError, match="weather_leg1"):
        ingest.validate(broken)


def test_an_arrival_day_outside_its_range_is_rejected(raw):
    broken = raw.copy()
    broken.loc[broken.index[0], "arrival_dow"] = 9
    with pytest.raises(ingest.ContractError, match="arrival_dow"):
        ingest.validate(broken)


def test_a_wrong_journey_id_in_the_raw_file_is_rejected(raw, tmp_path):
    broken = raw.copy()
    broken.loc[broken.index[0], "journey_id"] = "Nowhere>Nowhere"
    path = tmp_path / "shipments.csv"
    broken.to_csv(path, index=False)
    with pytest.raises(ingest.ContractError, match="journey_id"):
        ingest.load(path)


def test_a_missing_column_is_rejected(raw, tmp_path):
    path = tmp_path / "shipments.csv"
    raw.drop(columns=["weather_leg1"]).to_csv(path, index=False)
    with pytest.raises(ingest.ContractError, match="missing the columns"):
        ingest.load(path)


def test_a_duplicate_shipment_id_is_rejected(raw, tmp_path):
    broken = raw.copy()
    broken.loc[broken.index[1], "shipment_id"] = broken.loc[broken.index[0], "shipment_id"]
    path = tmp_path / "shipments.csv"
    broken.to_csv(path, index=False)
    with pytest.raises(ingest.ContractError, match="not unique"):
        ingest.load(path)


def test_an_unparseable_timestamp_is_rejected(raw, tmp_path):
    broken = raw.copy()
    broken["act_delivery"] = "not-a-date"
    path = tmp_path / "shipments.csv"
    broken.to_csv(path, index=False)
    with pytest.raises(ingest.ContractError, match="cannot be parsed"):
        ingest.load(path)
