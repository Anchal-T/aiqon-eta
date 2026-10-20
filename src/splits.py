"""Create repeatable shipment-level outer splits and grouped CV folds."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src import config

STACKED_PATH = Path("data/processed/stacked.parquet")
SPLITS_PATH = Path("data/processed/splits.csv")
SPLIT_SEEDS = tuple(range(config.SEED, config.SEED + config.N_SPLIT_SEEDS))
SPLIT_COLUMNS = ("split_seed", "shipment_id", "outer_split", "fold")


def create_splits(
    stacked: pd.DataFrame,
    *,
    seeds: Sequence[int] = SPLIT_SEEDS,
    test_frac: float = config.TEST_FRAC,
    n_folds: int = config.CV_FOLDS,
) -> pd.DataFrame:
    """Assign every shipment to one outer split and one training CV fold.

    ``fold`` identifies the held-out validation fold for training shipments;
    outer-test shipments have a null fold. All checkpoint rows for a shipment
    share the same assignment.
    """
    required = {"shipment_id", "journey_id"}
    missing = sorted(required - set(stacked.columns))
    if missing:
        raise ValueError(f"stacked table is missing columns: {missing}")
    if stacked.empty:
        raise ValueError("stacked table has no shipment rows")
    if not 0.0 < test_frac < 1.0:
        raise ValueError(f"test fraction must be between 0 and 1, got {test_frac}")
    if n_folds < 2:
        raise ValueError(f"at least two CV folds are required, got {n_folds}")

    split_seeds = tuple(seeds)
    if not split_seeds or len(set(split_seeds)) != len(split_seeds):
        raise ValueError("split seeds must be non-empty and unique")

    shipment_rows: pd.DataFrame = pd.DataFrame(
        stacked[["shipment_id", "journey_id"]].copy()
    )
    if bool(pd.isna(shipment_rows["shipment_id"]).sum()) or bool(
        pd.isna(shipment_rows["journey_id"]).sum()
    ):
        raise ValueError("shipment_id and journey_id must be present")
    shipment_rows["shipment_id"] = shipment_rows["shipment_id"].astype("string")
    shipments: pd.DataFrame = pd.DataFrame(shipment_rows.drop_duplicates())
    if shipments["shipment_id"].duplicated().any():
        raise ValueError("each shipment_id must have exactly one journey_id")

    records = []
    for seed in split_seeds:
        train, _test = train_test_split(
            shipments,
            test_size=test_frac,
            random_state=seed,
            stratify=shipments["journey_id"],
        )

        fold_for_shipment: dict[str, int] = {}
        splitter = StratifiedGroupKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=seed,
        )
        for fold, (_, validation_indices) in enumerate(
            splitter.split(
                train,
                y=train["journey_id"],
                groups=train["shipment_id"],
            )
        ):
            for shipment_id in train.iloc[validation_indices]["shipment_id"]:
                fold_for_shipment[str(shipment_id)] = fold

        train_ids = set(train["shipment_id"].astype(str))
        for shipment_id in shipments["shipment_id"].astype(str):
            is_train = shipment_id in train_ids
            records.append(
                {
                    "split_seed": seed,
                    "shipment_id": shipment_id,
                    "outer_split": "train" if is_train else "test",
                    "fold": fold_for_shipment.get(shipment_id) if is_train else None,
                }
            )

    result = pd.DataFrame.from_records(records, columns=SPLIT_COLUMNS)
    result["fold"] = result["fold"].astype("Int64")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create shipment-level train/test splits."
    )
    parser.add_argument("--stacked", type=Path, default=STACKED_PATH)
    parser.add_argument("--out", type=Path, default=SPLITS_PATH.parent)
    args = parser.parse_args()

    stacked = pd.read_parquet(args.stacked)
    splits = create_splits(stacked)
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / SPLITS_PATH.name
    splits.to_csv(output, index=False)
    print(f"wrote {len(splits)} shipment split rows -> {output}")


if __name__ == "__main__":
    main()
