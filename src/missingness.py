"""Simulate incomplete logs.

Only two milestones are ever hidden: ``act_port_done`` and ``act_customs``.
The departure is always there and the delivery is the target, so neither is
ever touched.

Every hidden cell is recorded in the mask together with its true value.  The
mask is what lets the evaluation measure how much error the missing data
causes, and what lets the features know that a stage is gone.

The random draws are made once per cell and thresholded by the rate, so the
hidden sets nest: the cells hidden at rate 0.1 are a subset of those hidden
at rate 0.2.  The missing-rate sweep in the evaluation then compares the
same world at different rates instead of different worlds.

The mask holds a timestamp in its ``true_value`` column.  Read it back with
``parse_dates=["true_value"]`` or the values stay strings.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config
from src.errors import ContractError

#: The only milestones that may go missing.
HIDEABLE = ("act_port_done", "act_customs")

#: Never hidden: the departure is the first thing that happens, and the
#: delivery is the target.
PROTECTED = ("act_departure", "act_delivery")


def hide(
    table: pd.DataFrame,
    settings: config.RunSettings = config.DEFAULT_RUN_SETTINGS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the observed table with holes, and the mask of what was hidden.

    The mask has one row per hidden cell: the shipment, the column and the
    value that was there.
    """
    missing = sorted({"shipment_id", *HIDEABLE} - set(table.columns))
    if missing:
        raise ContractError(f"raw table is missing columns: {missing}")
    if table["shipment_id"].isna().any() or table["shipment_id"].duplicated().any():
        raise ContractError("raw shipment_id values must be present and unique")
    if table[list(HIDEABLE)].isna().any().any():
        raise ContractError("raw hideable actual timestamps must be complete")

    # A stream of its own, so that changing the generator cannot shift the
    # holes around.
    rng = np.random.default_rng([settings.seed, 1])

    observed = table.copy()
    records: list[dict] = []
    for column in HIDEABLE:
        hidden = rng.random(len(table)) < settings.missing_rate
        for position in np.flatnonzero(hidden):
            records.append(
                {
                    "shipment_id": table["shipment_id"].iloc[position],
                    "column": column,
                    "true_value": table[column].iloc[position],
                }
            )
        observed.loc[hidden, column] = pd.NaT

    mask = pd.DataFrame(records, columns=["shipment_id", "column", "true_value"])
    return observed, mask
