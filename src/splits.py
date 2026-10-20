"""Create repeatable shipment-level outer splits and grouped CV folds."""

from __future__ import annotations

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from src import config
from src.errors import ContractError, SplitError

SPLIT_COLUMNS = ("split_seed", "shipment_id", "outer_split", "fold")


def create_splits(
    stacked: pd.DataFrame,
    settings: config.RunSettings = config.DEFAULT_RUN_SETTINGS,
) -> pd.DataFrame:
    """Assign every shipment to one outer split and one training CV fold.

    ``fold`` identifies the held-out validation fold for training shipments;
    outer-test shipments have a null fold. All checkpoint rows for a shipment
    share the same assignment.
    """
    required = {"shipment_id", "journey_id"}
    missing = sorted(required - set(stacked.columns))
    if missing:
        raise ContractError(f"stacked table is missing columns: {missing}")
    if stacked.empty:
        raise ContractError("stacked table has no shipment rows")

    shipment_rows: pd.DataFrame = pd.DataFrame(
        stacked[["shipment_id", "journey_id"]].copy()
    )
    if bool(pd.isna(shipment_rows["shipment_id"]).sum()) or bool(
        pd.isna(shipment_rows["journey_id"]).sum()
    ):
        raise ContractError("shipment_id and journey_id must be present")
    shipment_rows["shipment_id"] = shipment_rows["shipment_id"].astype("string")
    shipments: pd.DataFrame = pd.DataFrame(shipment_rows.drop_duplicates())
    if shipments["shipment_id"].duplicated().any():
        raise ContractError("each shipment_id must have exactly one journey_id")

    records = []
    for seed in settings.split_seeds:
        try:
            train, _test = train_test_split(
                shipments,
                test_size=settings.test_frac,
                random_state=seed,
                stratify=shipments["journey_id"],
            )

            fold_for_shipment: dict[str, int] = {}
            splitter = StratifiedGroupKFold(
                n_splits=settings.cv_folds,
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
        except ValueError as error:
            raise SplitError(
                f"cannot create stratified split for seed {seed}: {error}"
            ) from error
        if len(fold_for_shipment) != len(train):
            raise SplitError(
                f"CV folds did not assign every training shipment for seed {seed}"
            )

        train_ids = set(train["shipment_id"].astype(str))
        for shipment_id in shipments["shipment_id"].astype(str):
            is_train = shipment_id in train_ids
            records.append(
                {
                    "split_seed": seed,
                    "shipment_id": shipment_id,
                    "outer_split": "train" if is_train else "test",
                    "fold": fold_for_shipment[shipment_id] if is_train else None,
                }
            )

    result = pd.DataFrame.from_records(records, columns=SPLIT_COLUMNS)
    result["fold"] = result["fold"].astype("Int64")
    return result
