"""Load the raw shipments and check them against the data contract.

The raw file is what a real system would see: a route, a plan, and the
milestones observed so far.  This module reads it, gives every column an
explicit type, derives the two fields the contract does not carry, and fails
loudly when something is off.

Two kinds of check live here.  The checks on the *raw* file demand that
every applicable actual timestamp is present.  The observed file, which
missingness.py produces, is deliberately missing some of them, so loading it
passes ``complete_actuals=False``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

RAW_PATH = Path("data/raw/shipments.csv")

STRING_COLUMNS = ("shipment_id", "journey_id", "origin", "hub", "destination")
FLOAT_COLUMNS = (
    "weather_leg1",
    "weather_leg2",
    "congestion_hub",
    "congestion_port",
    "hub_buffer_h",
)
INT_COLUMNS = ("arrival_dow",)
TIMESTAMP_COLUMNS = tuple(
    f"{prefix}_{stage}" for stage in config.STAGE_ORDER for prefix in ("sched", "act")
)

#: The columns the contract names.  ``journey_id`` and ``is_connecting`` are
#: derived, so they are not on the list.
REQUIRED_COLUMNS = (
    "shipment_id",
    "origin",
    "hub",
    "destination",
    "weather_leg1",
    "weather_leg2",
    "congestion_hub",
    "congestion_port",
    "hub_buffer_h",
    "arrival_dow",
) + TIMESTAMP_COLUMNS


class ContractError(ValueError):
    """The data breaks the contract in LLD section 5."""


#: A stage list paired with the mask of the shipments it applies to.
StageGroups = tuple[tuple[tuple[str, ...], np.ndarray], ...]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def journey_id(table: pd.DataFrame) -> pd.Series:
    """``origin > hub-or-none > destination`` (LLD section 5.3)."""
    return table.apply(
        lambda row: ">".join(
            str(part)
            for part in (row["origin"], row["hub"], row["destination"])
            if pd.notna(part)
        ),
        axis=1,
    ).astype("string")


def load(path: Path | str = RAW_PATH, *, complete_actuals: bool = True) -> pd.DataFrame:
    """Read the shipments, type them, derive the missing fields, and validate.

    ``complete_actuals`` demands that every applicable actual timestamp is
    present.  That holds for the raw file and not for the observed one.
    """
    table = pd.read_csv(path, parse_dates=list(TIMESTAMP_COLUMNS))
    _cast(table)

    derived = journey_id(table)
    if "journey_id" in table.columns:
        known = table["journey_id"].notna()
        mismatch = known & (derived != table["journey_id"])
        if mismatch.any():
            raise ContractError(
                f"journey_id does not match origin/hub/destination for "
                f"{list(table.loc[mismatch, 'shipment_id'][:5])}"
            )
    table["journey_id"] = derived
    table["is_connecting"] = table["hub"].notna()

    validate(table, complete_actuals=complete_actuals)
    return table


def _cast(table: pd.DataFrame) -> None:
    """Give every column an explicit type."""
    for column in STRING_COLUMNS:
        if column in table.columns:
            table[column] = table[column].astype("string")
    for column in FLOAT_COLUMNS:
        if column in table.columns:
            table[column] = table[column].astype("float64")
    for column in INT_COLUMNS:
        if column in table.columns:
            table[column] = table[column].astype("int64")
    for column in TIMESTAMP_COLUMNS:
        if column in table.columns:
            try:
                table[column] = pd.to_datetime(table[column])
            except (TypeError, ValueError) as error:
                raise ContractError(f"{column} holds a timestamp that cannot be parsed: {error}") from error


# ---------------------------------------------------------------------------
# The contract
# ---------------------------------------------------------------------------


def validate(table: pd.DataFrame, *, complete_actuals: bool = True) -> None:
    """Raise :class:`ContractError` if the table breaks the data contract."""
    _check_columns(table)

    connecting = table["hub"].notna().to_numpy()
    groups = ((config.STAGE_ORDER, connecting), (config.DIRECT_STAGES, ~connecting))

    _check_schedule(table, groups)
    _check_hub_columns(table)
    if complete_actuals:
        _check_actual_timestamps(table, groups)
    _check_final_delay(table)
    _check_ranges(table)


def _check_columns(table: pd.DataFrame) -> None:
    """The table carries every column the contract names, and a unique key."""
    missing = [column for column in REQUIRED_COLUMNS if column not in table.columns]
    if missing:
        raise ContractError(f"the table is missing the columns {missing}")

    duplicated = table["shipment_id"].duplicated()
    if duplicated.any():
        raise ContractError(
            f"shipment_id is not unique for {list(table.loc[duplicated, 'shipment_id'][:5])}"
        )


def _check_schedule(table: pd.DataFrame, groups: StageGroups) -> None:
    """Scheduled times are in stage order for every shipment."""
    for stages, mask in groups:
        columns = [f"sched_{stage}" for stage in stages]
        subset = table.loc[mask, columns]
        if subset.isna().any().any():
            raise ContractError(f"scheduled timestamps are missing for stages {stages}")
        gaps = subset.diff(axis=1).iloc[:, 1:]
        out_of_order = (gaps <= pd.Timedelta(0)).any(axis=1)
        if out_of_order.any():
            raise ContractError(
                f"scheduled times are not in stage order for shipments "
                f"{list(subset.index[out_of_order][:5])}"
            )


def _check_hub_columns(table: pd.DataFrame) -> None:
    """Hub columns are empty if and only if the shipment has a hub."""
    connecting = table["hub"].notna()
    columns = [
        f"{prefix}_{stage}" for prefix in ("sched", "act") for stage in config.HUB_STAGES
    ]
    for column in columns:
        if table.loc[connecting, column].isna().any():
            raise ContractError(f"{column} is missing for a connecting shipment")
        if table.loc[~connecting, column].notna().any():
            raise ContractError(f"{column} is set for a direct shipment")


def _check_actual_timestamps(table: pd.DataFrame, groups: StageGroups) -> None:
    """No applicable actual timestamp is missing."""
    for stages, mask in groups:
        columns = [f"act_{stage}" for stage in stages]
        subset = table.loc[mask, columns]
        if subset.isna().any().any():
            raise ContractError(f"actual timestamps are missing for stages {stages}")


def _check_final_delay(table: pd.DataFrame) -> None:
    """The delivery delay is a finite number."""
    delay = (table["act_delivery"] - table["sched_delivery"]).dt.total_seconds() / 3600.0
    if not np.isfinite(delay).all():
        raise ContractError("the final delay is not finite")


def _check_ranges(table: pd.DataFrame) -> None:
    """The covariates stay inside the ranges the contract gives them."""
    for column in ("weather_leg1", "weather_leg2"):
        values = table[column].dropna()
        if not values.between(0.0, 1.0).all():
            raise ContractError(f"{column} is outside [0, 1]")
    if not table["arrival_dow"].between(0, 6).all():
        raise ContractError("arrival_dow is outside [0, 6]")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Check the raw shipments against the contract.")
    parser.add_argument("--in", dest="path", type=Path, default=RAW_PATH)
    args = parser.parse_args()

    table = load(args.path)
    connecting = int(table["is_connecting"].sum())
    print(
        f"{len(table)} shipments ok: {connecting} connecting, {len(table) - connecting} direct"
    )


if __name__ == "__main__":
    main()
