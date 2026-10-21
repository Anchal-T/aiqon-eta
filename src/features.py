"""Build the checkpoint-stacked model table from observed logs and delays."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src import checkpoints, config
from src.errors import ContractError

OBSERVED_STAGES = config.STAGE_ORDER[:-1]
ROUTE_CATEGORICAL_FEATURES = ("journey_id", "origin", "hub", "destination")
CATEGORICAL_FEATURES = (
    *ROUTE_CATEGORICAL_FEATURES,
    "sched_dep_dow",
    "sched_arrival_dow",
    "arrival_dow",
    "port_done_dow",
)

_CATEGORY_VALUES = {
    "journey_id": sorted({route.journey_id for route in config.ROUTES}),
    "origin": sorted({route.origin for route in config.ROUTES}),
    "hub": sorted({route.hub for route in config.ROUTES if route.hub is not None}),
    "destination": sorted({route.destination for route in config.ROUTES}),
}
CATEGORY_CODES = {
    column: {value: code for code, value in enumerate(values)}
    for column, values in _CATEGORY_VALUES.items()
}

STAGE_FEATURE_COLUMNS = (
    "delay_departure",
    *(
        column
        for stage in OBSERVED_STAGES[1:]
        for column in (f"delay_{stage}", f"added_{stage}")
    ),
)
FEATURE_COLUMNS = (
    "checkpoint",
    "journey_id",
    "origin",
    "hub",
    "destination",
    "is_connecting",
    "n_legs",
    "planned_total_h",
    "planned_leg1_h",
    "planned_leg2_h",
    "hub_buffer_h",
    "nominal_slack_h",
    "sched_dep_dow",
    "sched_arrival_dow",
    "sched_dep_doy",
    "port_dwell_base_h",
    "port_customs_base_h",
    "port_base_congestion",
    *STAGE_FEATURE_COLUMNS,
    "delay_so_far",
    "n_observed_stages",
    "weather_leg1",
    "weather_leg2",
    "congestion_hub",
    "congestion_port",
    "arrival_dow",
    "slack_remaining",
    "slot_margin",
    "cong_x_dwell",
    "cong_x_customs",
    "port_done_dow",
    "port_done_is_weekend",
    "port_done_missing",
    "customs_missing",
)

_BASE_ABLATION_FEATURES = (
    "checkpoint",
    "journey_id",
    "origin",
    "hub",
    "destination",
    "is_connecting",
    "n_legs",
    "planned_total_h",
    "planned_leg1_h",
    "planned_leg2_h",
    "hub_buffer_h",
    "nominal_slack_h",
    "sched_dep_dow",
    "sched_arrival_dow",
    "sched_dep_doy",
    "port_dwell_base_h",
    "port_customs_base_h",
    "port_base_congestion",
    *STAGE_FEATURE_COLUMNS,
    "delay_so_far",
    "n_observed_stages",
)
_ENVIRONMENT_ABLATION_FEATURES = (
    "weather_leg1",
    "weather_leg2",
    "congestion_hub",
    "congestion_port",
    "arrival_dow",
)
_LINK_ABLATION_FEATURES = (
    "slack_remaining",
    "slot_margin",
    "cong_x_dwell",
    "cong_x_customs",
    "port_done_dow",
    "port_done_is_weekend",
    "port_done_missing",
    "customs_missing",
)
MODEL_R_ABLATIONS = {
    "A0": _BASE_ABLATION_FEATURES,
    "A1": (*_BASE_ABLATION_FEATURES, *_ENVIRONMENT_ABLATION_FEATURES),
    "A2": (
        *_BASE_ABLATION_FEATURES,
        *_ENVIRONMENT_ABLATION_FEATURES,
        *_LINK_ABLATION_FEATURES,
    ),
}
TARGET_COLUMNS = ("final_delay_h", "remaining_delay_h", "is_delayed")
OUTPUT_COLUMNS = ("shipment_id", *FEATURE_COLUMNS, *TARGET_COLUMNS)

_REQUIRED_OBSERVED = {
    "shipment_id",
    "journey_id",
    "origin",
    "hub",
    "destination",
    "is_connecting",
    "weather_leg1",
    "weather_leg2",
    "congestion_hub",
    "congestion_port",
    "hub_buffer_h",
    "arrival_dow",
    *(f"sched_{stage}" for stage in config.STAGE_ORDER),
    *(f"act_{stage}" for stage in config.STAGE_ORDER),
}
_REQUIRED_DELAYS = {
    "shipment_id",
    "final_delay_h",
    *(f"delay_{stage}" for stage in OBSERVED_STAGES),
    *(f"added_{stage}" for stage in OBSERVED_STAGES[1:]),
}


def _hours(end: pd.Timestamp, start: pd.Timestamp) -> float:
    return (end - start).total_seconds() / 3600.0


def _category_code(column: str, value: Any) -> int:
    if bool(pd.isna(value)):
        return -1
    try:
        return CATEGORY_CODES[column][str(value)]
    except KeyError as error:
        raise ContractError(f"unknown {column} category: {value}") from error


def _add_schedule_features(
    record: dict[str, object],
    shipment: Any,
    is_connecting: bool,
    norm: config.PortNorm,
) -> None:
    departure = shipment["sched_departure"]
    port_arrival = shipment["sched_port_arrival"]
    record["planned_total_h"] = _hours(shipment["sched_delivery"], departure)
    record["planned_leg1_h"] = _hours(
        shipment["sched_hub_arrival"] if is_connecting else port_arrival,
        departure,
    )
    if is_connecting:
        record["planned_leg2_h"] = _hours(port_arrival, shipment["sched_hub_departure"])
        record["hub_buffer_h"] = float(shipment["hub_buffer_h"])
        record["nominal_slack_h"] = (
            float(shipment["hub_buffer_h"]) - config.MIN_CONNECT_H
        )

    record["sched_dep_dow"] = departure.dayofweek
    record["sched_arrival_dow"] = port_arrival.dayofweek
    record["sched_dep_doy"] = departure.dayofyear
    record["port_dwell_base_h"] = norm.dwell_base_h
    record["port_customs_base_h"] = norm.customs_base_h
    record["port_base_congestion"] = norm.base_congestion


def _add_observed_delay_features(
    record: dict[str, object], delay_row: Any, visible: tuple[str, ...]
) -> float:
    known_delays = []
    for stage in OBSERVED_STAGES:
        if stage not in visible:
            continue
        record[f"delay_{stage}"] = delay_row[f"delay_{stage}"]
        if stage != "departure":
            record[f"added_{stage}"] = delay_row[f"added_{stage}"]
        value = delay_row[f"delay_{stage}"]
        if pd.notna(value):
            known_delays.append(float(value))

    delay_so_far = known_delays[-1] if known_delays else 0.0
    record["delay_so_far"] = delay_so_far
    record["n_observed_stages"] = len(known_delays)
    return delay_so_far


def _add_checkpoint_gated_features(
    record: dict[str, object],
    shipment: Any,
    delay_row: Any,
    checkpoint: int,
    is_connecting: bool,
    norm: config.PortNorm,
) -> None:
    def is_visible(feature: str) -> bool:
        return checkpoints.feature_is_visible(feature, checkpoint, is_connecting)

    if is_visible("weather_leg1"):
        record["weather_leg1"] = shipment["weather_leg1"]
    if is_visible("congestion_hub"):
        record["congestion_hub"] = shipment["congestion_hub"]
    if is_visible("slack_remaining"):
        hub_arrival_delay = delay_row["delay_hub_arrival"]
        if pd.notna(hub_arrival_delay):
            record["slack_remaining"] = record["nominal_slack_h"] - hub_arrival_delay
    if is_visible("weather_leg2"):
        record["weather_leg2"] = shipment["weather_leg2"]
    if is_visible("congestion_port"):
        record["congestion_port"] = shipment["congestion_port"]
    if is_visible("arrival_dow"):
        record["arrival_dow"] = shipment["arrival_dow"]
    if is_visible("slot_margin"):
        port_arrival_delay = delay_row["delay_port_arrival"]
        if pd.notna(port_arrival_delay):
            record["slot_margin"] = config.SLOT_WINDOW_H - port_arrival_delay
    if is_visible("cong_x_dwell"):
        record["cong_x_dwell"] = shipment["congestion_port"] * norm.dwell_base_h
    if is_visible("cong_x_customs"):
        record["cong_x_customs"] = shipment["congestion_port"] * norm.customs_base_h

    port_done = shipment["act_port_done"]
    if is_visible("port_done_missing"):
        record["port_done_missing"] = int(pd.isna(port_done))
    if pd.notna(port_done):
        if is_visible("port_done_dow"):
            record["port_done_dow"] = port_done.dayofweek
        if is_visible("port_done_is_weekend"):
            record["port_done_is_weekend"] = int(port_done.dayofweek >= 5)
    if is_visible("customs_missing"):
        record["customs_missing"] = int(pd.isna(shipment["act_customs"]))


def _add_target_features(
    record: dict[str, object],
    shipment_id: Any,
    delay_row: Any,
    delay_so_far: float,
    settings: config.RunSettings,
) -> None:
    final_delay = float(delay_row["final_delay_h"])
    if not np.isfinite(final_delay):
        raise ContractError(f"non-finite final delay for shipment {shipment_id}")
    record["final_delay_h"] = final_delay
    record["remaining_delay_h"] = final_delay - delay_so_far
    record["is_delayed"] = final_delay > settings.delay_threshold_h


def _build_feature_row(
    shipment: Any,
    delay_row: Any,
    checkpoint: int,
    settings: config.RunSettings,
) -> dict[str, object]:
    is_connecting = bool(shipment["is_connecting"])
    visible = checkpoints.visible_stages(checkpoint, is_connecting)
    destination = str(shipment["destination"])
    try:
        norm = config.PORT_NORMS[destination]
    except KeyError as error:
        raise ContractError(f"unknown destination: {destination}") from error

    record: dict[str, object] = {column: np.nan for column in FEATURE_COLUMNS}
    record["shipment_id"] = str(shipment["shipment_id"])
    record["checkpoint"] = checkpoint
    for column in ROUTE_CATEGORICAL_FEATURES:
        record[column] = _category_code(column, shipment[column])
    record["is_connecting"] = int(is_connecting)
    record["n_legs"] = 2 if is_connecting else 1

    _add_schedule_features(record, shipment, is_connecting, norm)
    delay_so_far = _add_observed_delay_features(record, delay_row, visible)
    _add_checkpoint_gated_features(
        record, shipment, delay_row, checkpoint, is_connecting, norm
    )
    _add_target_features(
        record, shipment["shipment_id"], delay_row, delay_so_far, settings
    )
    return record


def build(
    observed: pd.DataFrame,
    delay_table: pd.DataFrame,
    settings: config.RunSettings = config.DEFAULT_RUN_SETTINGS,
) -> pd.DataFrame:
    """Build the stacked feature table from contract-validated inputs.

    ``observed`` may have the expected missing actuals. ``delay_table`` must
    contain matching shipment identities and numeric delay values.
    """
    missing_observed = sorted(_REQUIRED_OBSERVED - set(observed.columns))
    if missing_observed:
        raise ContractError(f"observed table is missing columns: {missing_observed}")
    missing_delays = sorted(_REQUIRED_DELAYS - set(delay_table.columns))
    if missing_delays:
        raise ContractError(f"delay table is missing columns: {missing_delays}")
    if bool(observed["shipment_id"].isna().any()) or bool(
        observed["shipment_id"].duplicated().any()
    ):
        raise ContractError("observed shipment_id values must be present and unique")

    delay_table = delay_table.copy()
    delay_table["shipment_id"] = delay_table["shipment_id"].astype("string")
    if bool(delay_table["shipment_id"].isna().any()) or bool(
        delay_table["shipment_id"].duplicated().any()
    ):
        raise ContractError("delay shipment_id values must be present and unique")
    observed_ids = set(observed["shipment_id"].astype(str))
    delay_ids = set(delay_table["shipment_id"].astype(str))
    if observed_ids != delay_ids:
        missing_delays = sorted(observed_ids - delay_ids)
        missing_observed = sorted(delay_ids - observed_ids)
        raise ContractError(
            "observed and delay tables must contain the same shipments; "
            f"missing from delays: {missing_delays[:5]}, "
            f"missing from observed: {missing_observed[:5]}"
        )

    for column in _REQUIRED_DELAYS - {"shipment_id"}:
        values = pd.to_numeric(delay_table[column], errors="coerce")
        invalid = delay_table[column].notna() & values.isna()
        if invalid.any():
            raise ContractError(f"delay column {column} contains non-numeric values")
        delay_table[column] = values
    if not np.isfinite(delay_table["final_delay_h"]).all():
        raise ContractError("final_delay_h must be finite for every shipment")

    delay_by_id = delay_table.set_index("shipment_id")

    records = []
    for _, shipment in observed.iterrows():
        is_connecting = bool(shipment["is_connecting"])
        delay_row = delay_by_id.loc[str(shipment["shipment_id"])]
        for checkpoint in checkpoints.for_shipment(is_connecting):
            records.append(
                _build_feature_row(shipment, delay_row, checkpoint, settings)
            )
    return pd.DataFrame.from_records(records, columns=OUTPUT_COLUMNS)
