"""Turn timestamps into delay numbers.

For every shipment and every stage:

* ``delay_<stage>`` is how late the stage happened, in hours.  It is missing
  when the stage is hidden.
* ``added_<stage>`` is how much lateness that stage added on top of the last
  stage we actually know about.
* ``merged_<stage>`` says whether that added delay covers more than one
  stage, because a stage in between was hidden.

The gap rule matters.  When a stage is hidden we cannot say what it added, so
the next stage we can see is compared against the last one we could see.  Its
added delay then covers a span of the journey, which is flagged rather than
pretended away.

Direct shipments never have hub stages.  Those are not applicable, not
hidden, so they are skipped without flagging a gap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.errors import ContractError

_REQUIRED_COLUMNS = {
    "shipment_id",
    "is_connecting",
    *(f"sched_{stage}" for stage in config.STAGE_ORDER),
    *(f"act_{stage}" for stage in config.STAGE_ORDER),
}


def compute(observed: pd.DataFrame) -> pd.DataFrame:
    """Calculate delays from the timestamp columns needed by this transform."""
    missing = sorted(_REQUIRED_COLUMNS - set(observed.columns))
    if missing:
        raise ContractError(f"observed table is missing columns: {missing}")
    if (
        observed["shipment_id"].isna().any()
        or observed["shipment_id"].duplicated().any()
    ):
        raise ContractError("observed shipment_id values must be present and unique")
    if observed["is_connecting"].isna().any():
        raise ContractError("observed is_connecting values must be present")
    if observed["act_delivery"].isna().any():
        raise ContractError("actual delivery timestamps must be present")

    records = [_shipment_delays(row) for _, row in observed.iterrows()]
    return pd.DataFrame(records)


def _shipment_delays(shipment: pd.Series) -> dict:
    """The delays of one shipment, stage by stage."""
    stages = config.STAGE_ORDER if shipment["is_connecting"] else config.DIRECT_STAGES

    delays: dict[str, float] = {}
    for stage in stages:
        actual = shipment[f"act_{stage}"]
        if pd.isna(actual):
            delays[stage] = np.nan
        else:
            planned = shipment[f"sched_{stage}"]
            delays[stage] = (actual - planned).total_seconds() / 3600.0

    added: dict[str, float] = {}
    merged: dict[str, bool] = {}
    previous_delay = 0.0
    skipped = False
    for stage in stages:
        value = delays[stage]
        if np.isnan(value):
            added[stage] = np.nan
            merged[stage] = False
            skipped = True
            continue
        added[stage] = value - previous_delay
        merged[stage] = skipped
        previous_delay = value
        skipped = False

    record: dict = {"shipment_id": shipment["shipment_id"]}
    for stage in config.STAGE_ORDER:
        record[f"delay_{stage}"] = delays.get(stage, np.nan)
        record[f"added_{stage}"] = added.get(stage, np.nan)
        record[f"merged_{stage}"] = merged.get(stage, False)
    record["final_delay_h"] = delays["delivery"]
    return record
