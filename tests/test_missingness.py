"""Tests for the missing-data simulation.

Covers T4 from the LLD (section 7): the mask never touches the protected
milestones, and the hidden cells are exactly the ones the mask records.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import ingest, missingness


@pytest.fixture(scope="module")
def raw():
    return ingest.load()


@pytest.fixture(scope="module")
def hidden(raw):
    return missingness.hide(raw)


# ---------------------------------------------------------------------------
# T4
# ---------------------------------------------------------------------------


def test_the_protected_milestones_are_never_hidden(hidden):
    observed, mask = hidden
    for column in missingness.PROTECTED:
        assert observed[column].notna().all(), column
        assert column not in set(mask["column"]), column


def test_only_the_hideable_columns_are_ever_missing(hidden):
    observed, mask = hidden
    assert set(mask["column"]) <= set(missingness.HIDEABLE)
    for column in observed.columns:
        if not column.startswith("act_") or column in missingness.HIDEABLE:
            continue
        # Hub stages are empty for direct shipments because they do not apply
        # to them, not because anything was hidden.
        applicable = observed[column].notna() | observed["is_connecting"]
        missing = observed[column].isna() & applicable
        assert not missing.any(), column


def test_the_mask_records_exactly_the_hidden_cells(hidden, raw):
    observed, mask = hidden
    for _, record in mask.iterrows():
        row = observed[observed["shipment_id"] == record["shipment_id"]].iloc[0]
        assert pd.isna(row[record["column"]])
        original = raw[raw["shipment_id"] == record["shipment_id"]].iloc[0]
        assert original[record["column"]] == record["true_value"]


def test_the_mask_carries_the_columns_the_contract_asks_for(hidden):
    _, mask = hidden
    assert list(mask.columns) == ["shipment_id", "column", "true_value"]


# ---------------------------------------------------------------------------
# Behaviour of the hiding
# ---------------------------------------------------------------------------


def test_no_holes_at_rate_zero(raw):
    observed, mask = missingness.hide(raw, rate=0.0)
    assert mask.empty
    assert observed.equals(raw)


def test_the_observed_rate_is_close_to_the_requested_one(raw):
    observed, mask = missingness.hide(raw, rate=0.1)
    expected = 0.1 * len(raw) * len(missingness.HIDEABLE)
    assert abs(len(mask) - expected) < 0.15 * expected


def test_the_holes_are_reproducible(raw):
    first = missingness.hide(raw, rate=0.2)[1]
    second = missingness.hide(raw, rate=0.2)[1]
    pd.testing.assert_frame_equal(first, second)


def test_a_different_seed_gives_different_holes(raw):
    first = missingness.hide(raw, rate=0.2, seed=1)[1]
    second = missingness.hide(raw, rate=0.2, seed=2)[1]
    assert not first.equals(second)


def test_the_hidden_sets_nest_as_the_rate_grows(raw):
    """The rate sweep has to compare the same world at different rates."""
    small = missingness.hide(raw, rate=0.1)[1]
    medium = missingness.hide(raw, rate=0.2)[1]
    large = missingness.hide(raw, rate=0.3)[1]

    def keys(mask):
        return set(zip(mask["shipment_id"], mask["column"]))

    assert keys(small) <= keys(medium) <= keys(large)


def test_a_rate_outside_the_unit_interval_is_rejected(raw):
    with pytest.raises(ValueError):
        missingness.hide(raw, rate=1.5)
