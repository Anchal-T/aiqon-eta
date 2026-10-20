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

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src import config, ingest

OBSERVED_PATH = Path("data/interim/observed.csv")
MASK_PATH = Path("data/interim/hidden_mask.csv")

#: The only milestones that may go missing.
HIDEABLE = ("act_port_done", "act_customs")

#: Never hidden: the departure is the first thing that happens, and the
#: delivery is the target.
PROTECTED = ("act_departure", "act_delivery")


def hide(
    table: pd.DataFrame,
    rate: float = config.MISSING_RATE,
    seed: int = config.SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the observed table with holes, and the mask of what was hidden.

    The mask has one row per hidden cell: the shipment, the column and the
    value that was there.
    """
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"missing rate must be between 0 and 1, got {rate}")

    protected = set(HIDEABLE) & set(PROTECTED)
    if protected:
        raise ValueError(f"{sorted(protected)} must never be hidden")

    # A stream of its own, so that changing the generator cannot shift the
    # holes around.
    rng = np.random.default_rng([seed, 1])

    observed = table.copy()
    records: list[dict] = []
    for column in HIDEABLE:
        hidden = rng.random(len(table)) < rate
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


def write(observed: pd.DataFrame, mask: pd.DataFrame, out_dir: Path) -> None:
    """Write the observed table and the mask."""
    out_dir.mkdir(parents=True, exist_ok=True)
    observed.to_csv(out_dir / OBSERVED_PATH.name, index=False)
    mask.to_csv(out_dir / MASK_PATH.name, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Hide part of the observed milestones.")
    parser.add_argument("--raw", type=Path, default=ingest.RAW_PATH)
    parser.add_argument("--out", type=Path, default=Path("data/interim"))
    parser.add_argument("--rate", type=float, default=config.MISSING_RATE)
    parser.add_argument("--seed", type=int, default=config.SEED)
    args = parser.parse_args()

    table = ingest.load(args.raw)
    observed, mask = hide(table, rate=args.rate, seed=args.seed)
    write(observed, mask, args.out)
    print(
        f"hid {len(mask)} cells at rate {args.rate} -> {args.out / OBSERVED_PATH.name}"
    )


if __name__ == "__main__":
    main()
