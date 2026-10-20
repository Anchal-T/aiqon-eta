"""Tests for the raw data generator.

Covers T3 and T5 from the LLD (section 7) and the data contract of section 5.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import config, generate

#: Cumulative delays the LLD quotes for S000 (section 7, T3).
LLD_DELAYS = {
    "departure": 3.7,
    "hub_arrival": 7.8,
    "hub_departure": 48.0,
    "port_arrival": 24.3,
    "port_done": 54.1,
    "customs": 69.8,
    "delivery": 58.5,
}

#: The same numbers as added delays, i.e. the LLD's worked example (section 8).
LLD_ADDED = {
    "departure": 3.7,
    "hub_arrival": 4.1,
    "hub_departure": 40.2,
    "port_arrival": -23.7,
    "port_done": 29.8,
    "customs": 15.7,
    "delivery": -11.3,
}


@pytest.fixture(scope="module")
def data():
    return generate.generate()


@pytest.fixture(scope="module")
def shipments(data):
    return data[0]


@pytest.fixture(scope="module")
def truth(data):
    return data[1]


def _delay_hours(row: pd.Series, stage: str) -> float:
    return (row[f"act_{stage}"] - row[f"sched_{stage}"]).total_seconds() / 3600.0


# ---------------------------------------------------------------------------
# T3 - the specimen reproduces the worked example
# ---------------------------------------------------------------------------


def test_specimen_delays_match_the_lld(shipments):
    row = shipments.loc[shipments.shipment_id == config.SPECIMEN_ID].iloc[0]
    for stage, expected in LLD_DELAYS.items():
        assert _delay_hours(row, stage) == pytest.approx(expected, abs=1e-3), stage


def test_specimen_added_delays_match_the_lld(shipments):
    row = shipments.loc[shipments.shipment_id == config.SPECIMEN_ID].iloc[0]
    previous = 0.0
    for stage in config.STAGE_ORDER:
        delay = _delay_hours(row, stage)
        assert delay - previous == pytest.approx(LLD_ADDED[stage], abs=1e-3), stage
        previous = delay


def test_specimen_is_the_lld_route(shipments):
    row = shipments.loc[shipments.shipment_id == config.SPECIMEN_ID].iloc[0]
    assert row.journey_id == "Chennai>Singapore>Rotterdam"
    assert row.origin == "Chennai"
    assert row.hub == "Singapore"
    assert row.destination == "Rotterdam"


# ---------------------------------------------------------------------------
# T5 - reproducible
# ---------------------------------------------------------------------------


def test_same_seed_gives_identical_data():
    first = generate.generate(seed=7, n_shipments=40)
    second = generate.generate(seed=7, n_shipments=40)
    pd.testing.assert_frame_equal(first[0], second[0])
    pd.testing.assert_frame_equal(first[1], second[1])


def test_a_different_seed_changes_the_data():
    first = generate.generate(seed=7, n_shipments=40)[0]
    second = generate.generate(seed=8, n_shipments=40)[0]
    assert not first.equals(second)


# ---------------------------------------------------------------------------
# Data contract (section 5)
# ---------------------------------------------------------------------------


def test_shipments_have_the_expected_shape(shipments):
    assert len(shipments) == config.N_SHIPMENTS
    assert list(shipments.shipment_id) == [f"S{i:03d}" for i in range(config.N_SHIPMENTS)]
    assert shipments.shipment_id.is_unique


def test_stage_columns_are_present_and_datetime(shipments):
    for stage in config.STAGE_ORDER:
        for prefix in ("sched_", "act_"):
            column = f"{prefix}{stage}"
            assert column in shipments.columns
            assert pd.api.types.is_datetime64_any_dtype(shipments[column])


def test_no_applicable_actual_timestamp_is_missing(shipments):
    for stage in config.STAGE_ORDER:
        if stage in config.HUB_STAGES:
            continue
        assert shipments[f"act_{stage}"].notna().all(), stage


def test_hub_columns_are_empty_exactly_for_direct_shipments(shipments):
    connecting = shipments.hub.notna()
    for column in ("sched_hub_arrival", "sched_hub_departure", "act_hub_arrival", "act_hub_departure"):
        assert shipments.loc[connecting, column].notna().all(), column
        assert shipments.loc[~connecting, column].isna().all(), column
    for column in ("weather_leg2", "congestion_hub", "hub_buffer_h"):
        assert shipments.loc[connecting, column].notna().all(), column
        assert shipments.loc[~connecting, column].isna().all(), column


def test_scheduled_times_are_in_stage_order(shipments):
    connecting = shipments.hub.notna()
    for stages, subset in (
        (config.STAGE_ORDER, shipments.loc[connecting]),
        (config.DIRECT_STAGES, shipments),
    ):
        diffs = subset[[f"sched_{s}" for s in stages]].diff(axis=1).iloc[:, 1:]
        assert (diffs > pd.Timedelta(0)).all().all(), stages


def test_covariates_are_in_range(shipments):
    for column in ("weather_leg1", "weather_leg2"):
        values = shipments[column].dropna()
        assert values.between(0.0, 1.0).all(), column
    assert shipments.arrival_dow.between(0, 6).all()


def test_hub_buffer_is_at_least_the_connect_time(shipments):
    buffers = shipments.hub_buffer_h.dropna()
    assert (buffers >= config.MIN_CONNECT_H).all()


def test_direct_and_connecting_mix_is_close_to_the_design(shipments):
    share = shipments.hub.notna().mean()
    assert 0.45 < share < 0.60, share


# ---------------------------------------------------------------------------
# Planted effects are really planted
# ---------------------------------------------------------------------------


def test_missed_connection_resets_the_delay_to_the_wait(shipments, truth):
    merged = shipments.merge(truth, on="shipment_id")
    connecting = merged.hub.notna()
    missed = connecting & merged.missed_connection
    caught = connecting & ~merged.missed_connection

    delays = merged.apply(lambda r: _delay_hours(r, "hub_departure"), axis=1)
    assert delays[missed].eq(config.NEXT_SAILING_WAIT_H).all()
    assert delays[caught].abs().max() < config.NEXT_SAILING_WAIT_H / 2


def test_missed_connection_happens_exactly_when_the_slack_runs_out(shipments, truth):
    merged = shipments.merge(truth, on="shipment_id")
    connecting = merged.hub.notna()
    slack = (merged.hub_buffer_h - config.MIN_CONNECT_H) - merged.apply(
        lambda r: _delay_hours(r, "hub_arrival"), axis=1
    )
    assert (merged.missed_connection == (slack < 0))[connecting].all()


def test_missed_slot_happens_exactly_when_the_window_runs_out(shipments, truth):
    merged = shipments.merge(truth, on="shipment_id")
    arrival_delay = merged.apply(lambda r: _delay_hours(r, "port_arrival"), axis=1)
    assert (merged.missed_slot == (arrival_delay > config.SLOT_WINDOW_H)).all()


def test_missing_the_berth_slot_costs_about_the_penalty(shipments, truth):
    merged = shipments.merge(truth, on="shipment_id")
    dwell_extra = merged.apply(
        lambda r: (r.act_port_done - r.act_port_arrival).total_seconds() / 3600.0
        - config.PORT_NORMS[r.destination].dwell_base_h,
        axis=1,
    )
    missed = merged.missed_slot
    assert dwell_extra[missed].mean() - dwell_extra[~missed].mean() > 12.0


def test_a_storm_shows_up_on_its_own_leg(shipments, truth):
    merged = shipments.merge(truth, on="shipment_id")
    connecting = merged.hub.notna()
    added_leg1 = merged.apply(
        lambda r: _delay_hours(r, "hub_arrival") - _delay_hours(r, "departure"), axis=1
    )
    storm = merged.storm_leg1
    assert added_leg1[connecting & storm].mean() > added_leg1[connecting & ~storm].mean() + 10.0
    assert merged.storm_leg1.sum() > 0 and merged.storm_leg2.sum() > 0
