"""Tests for the delay arithmetic, including the gap rule."""

from __future__ import annotations

import pandas as pd
import pytest
from src.pipeline import ArtifactPaths

from src import config, delays, ingest, missingness


@pytest.fixture(scope="module")
def raw():
    return ingest.load(ArtifactPaths().raw_shipments)


@pytest.fixture(scope="module")
def observed(raw):
    return missingness.hide(raw)[0]


@pytest.fixture(scope="module")
def computed(observed):
    return delays.compute(observed)


def _delay_hours(row: pd.Series, stage: str) -> float:
    return (row[f"act_{stage}"] - row[f"sched_{stage}"]).total_seconds() / 3600.0


# ---------------------------------------------------------------------------
# The plain arithmetic
# ---------------------------------------------------------------------------


def test_the_delay_of_a_stage_is_actual_minus_scheduled(computed, observed):
    for _, row in computed.iterrows():
        source = observed.loc[row.name]
        for stage in config.STAGE_ORDER:
            if pd.isna(source[f"act_{stage}"]):
                assert pd.isna(row[f"delay_{stage}"])
            else:
                assert row[f"delay_{stage}"] == pytest.approx(
                    _delay_hours(source, stage), abs=1e-9
                )


def test_the_final_delay_is_the_delay_at_delivery(computed, observed):
    for _, row in computed.iterrows():
        assert row["final_delay_h"] == pytest.approx(row["delay_delivery"], abs=1e-9)
        assert row["final_delay_h"] == pytest.approx(
            _delay_hours(observed.loc[row.name], "delivery"), abs=1e-9
        )


def test_the_added_delay_of_the_first_stage_is_the_delay_itself(computed):
    for _, row in computed.iterrows():
        assert row["added_departure"] == pytest.approx(row["delay_departure"], abs=1e-9)


def test_the_added_delay_is_the_difference_to_the_previous_known_stage(
    computed, observed
):
    for _, row in computed.iterrows():
        source = observed.loc[row.name]
        stages = config.STAGE_ORDER if source["is_connecting"] else config.DIRECT_STAGES
        previous = 0.0
        for stage in stages:
            value = row[f"delay_{stage}"]
            if pd.isna(value):
                continue
            assert row[f"added_{stage}"] == pytest.approx(value - previous, abs=1e-9)
            previous = value


# ---------------------------------------------------------------------------
# The gap rule
# ---------------------------------------------------------------------------


def test_a_hidden_stage_leaves_its_own_delay_empty(computed, observed):
    hidden_id = observed.loc[observed["act_port_done"].isna(), "shipment_id"].iloc[0]
    row = computed[computed["shipment_id"] == hidden_id].iloc[0]
    assert pd.isna(row["delay_port_done"])
    assert pd.isna(row["added_port_done"])


def test_the_next_known_stage_is_measured_against_the_last_known_one(raw):
    """The added delay of a stage after a hidden one covers a merged span."""
    broken = raw.copy()
    target = broken["shipment_id"].iloc[1]
    broken.loc[broken["shipment_id"] == target, "act_port_done"] = pd.NaT
    row = delays.compute(broken).iloc[1]

    assert bool(row["merged_customs"])
    assert row["added_customs"] == pytest.approx(
        row["delay_customs"] - row["delay_port_arrival"], abs=1e-9
    )


def test_both_late_stages_hidden_merges_the_whole_tail(raw):
    broken = raw.copy()
    target = broken["shipment_id"].iloc[1]
    broken.loc[broken["shipment_id"] == target, ["act_port_done", "act_customs"]] = (
        pd.NaT
    )
    row = delays.compute(broken).iloc[1]

    # A hidden stage has no added delay of its own, so it carries no flag.
    assert pd.isna(row["delay_port_done"])
    assert pd.isna(row["delay_customs"])
    assert not row["merged_customs"]

    # The delivery is the next stage we can see, so its added delay covers
    # port_done, customs and delivery in one number.
    assert bool(row["merged_delivery"])
    assert row["added_delivery"] == pytest.approx(
        row["delay_delivery"] - row["delay_port_arrival"], abs=1e-9
    )


def test_an_ungapped_shipment_is_not_flagged(computed, observed):
    """A shipment with nothing hidden has no merged spans."""
    complete = observed[
        observed["act_port_done"].notna() & observed["act_customs"].notna()
    ]
    rows = computed[computed["shipment_id"].isin(complete["shipment_id"])]
    for stage in config.STAGE_ORDER:
        assert not rows[f"merged_{stage}"].any(), stage


# ---------------------------------------------------------------------------
# Direct shipments
# ---------------------------------------------------------------------------


def test_a_direct_shipment_has_no_hub_delays(computed, observed):
    direct = observed[~observed["is_connecting"]]
    rows = computed[computed["shipment_id"].isin(direct["shipment_id"])]
    for stage in config.HUB_STAGES:
        assert rows[f"delay_{stage}"].isna().all(), stage
        assert not rows[f"merged_{stage}"].any(), stage


def test_a_direct_shipment_measures_the_port_arrival_against_the_departure(raw):
    broken = raw.copy()
    direct = broken.index[~broken["is_connecting"]][0]
    row = delays.compute(broken).loc[direct]
    assert row["added_port_arrival"] == pytest.approx(
        row["delay_port_arrival"] - row["delay_departure"], abs=1e-9
    )
    assert not row["merged_port_arrival"]


def test_a_connecting_shipment_measures_the_hub_arrival_against_the_departure(raw):
    row = delays.compute(raw).iloc[0]
    assert row["added_hub_arrival"] == pytest.approx(
        row["delay_hub_arrival"] - row["delay_departure"], abs=1e-9
    )


# ---------------------------------------------------------------------------
# Output shape
# ---------------------------------------------------------------------------


def test_the_delay_table_has_one_row_per_shipment(computed, observed):
    assert len(computed) == len(observed)
    assert computed["shipment_id"].is_unique
