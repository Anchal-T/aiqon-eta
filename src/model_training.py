"""Shared validation and grouped regression training for ETA models."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config, features, splits
from src.errors import ConfigurationError, ContractError, ModelError, SplitError


def _validate_training_options(max_estimators: int, early_stopping_rounds: int) -> None:
    if max_estimators < 1:
        raise ConfigurationError("max_estimators must be at least one")
    if early_stopping_rounds < 1:
        raise ConfigurationError("early_stopping_rounds must be at least one")


def _validated_rows(
    stacked: pd.DataFrame, target_columns: tuple[str, ...]
) -> pd.DataFrame:
    required = {
        "shipment_id",
        "checkpoint",
        *target_columns,
        *features.FEATURE_COLUMNS,
    }
    missing = sorted(required - set(stacked.columns))
    if missing:
        raise ContractError(f"stacked table is missing model columns: {missing}")
    if stacked.empty:
        raise ContractError("stacked table has no model rows")

    rows = stacked[["shipment_id", *target_columns, *features.FEATURE_COLUMNS]].copy()
    if rows[["shipment_id", "checkpoint"]].isna().any().any():
        raise ContractError("model shipment IDs and checkpoints must be present")
    rows["shipment_id"] = rows["shipment_id"].astype("string")
    if rows.duplicated(["shipment_id", "checkpoint"]).any():
        raise ContractError("model shipment/checkpoint rows must be unique")

    for column in target_columns:
        values = pd.to_numeric(rows[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values).all():
            raise ContractError(f"model target {column} must contain finite values")
        rows[column] = values
    for column in features.FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(rows[column]):
            raise ContractError(f"model feature {column} must be numeric")
        values = rows[column].to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(values).any():
            raise ContractError(f"model feature {column} must be finite or missing")
    return rows


def _validated_splits(split_manifest: pd.DataFrame) -> pd.DataFrame:
    assignments = splits.validate_assignments(split_manifest)
    if "fold" not in split_manifest.columns:
        raise ContractError("split manifest is missing model columns: ['fold']")

    folds = pd.to_numeric(split_manifest["fold"], errors="coerce")
    is_training = assignments["outer_split"] == "train"
    if folds[is_training].isna().any() or folds[~is_training].notna().any():
        raise ContractError("training rows need folds and test rows must not have them")
    if not np.isfinite(folds[is_training]).all() or (folds[is_training] % 1 != 0).any():
        raise ContractError("training fold values must be finite integers")
    assignments["fold"] = folds.astype("Int64")
    return assignments


@dataclass(frozen=True)
class _FitSpec:
    model_name: str
    objective: str
    eval_metric: str
    max_estimators: int
    early_stopping_rounds: int
    alpha: float | None = None


def _estimator_options(seed: int, n_estimators: int) -> dict[str, Any]:
    return {
        "n_estimators": n_estimators,
        "num_leaves": 8,
        "min_child_samples": 15,
        "learning_rate": 0.05,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": 1,
        "verbosity": -1,
    }


def _new_regressor(seed: int, n_estimators: int, spec: _FitSpec) -> lgb.LGBMRegressor:
    options = _estimator_options(seed, n_estimators)
    options["objective"] = spec.objective
    if spec.alpha is not None:
        options["alpha"] = spec.alpha
    return lgb.LGBMRegressor(**options)


def _rows_for_seed(
    rows: pd.DataFrame, seed_assignments: pd.DataFrame, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    expected_ids = set(rows["shipment_id"])
    assigned_ids = set(seed_assignments["shipment_id"])
    if assigned_ids != expected_ids:
        missing = sorted(expected_ids - assigned_ids)
        unexpected = sorted(assigned_ids - expected_ids)
        raise ContractError(
            f"split seed {seed} shipment IDs do not match stacked data; "
            f"missing: {missing[:5]}, unexpected: {unexpected[:5]}"
        )

    assigned_rows = rows.merge(
        seed_assignments, on="shipment_id", how="left", validate="many_to_one"
    )
    training = assigned_rows[assigned_rows["outer_split"] == "train"]
    testing = assigned_rows[assigned_rows["outer_split"] == "test"]
    if training.empty or testing.empty:
        raise SplitError(f"split seed {seed} must have train and test rows")
    return training, testing


def _rows_by_seed(
    rows: pd.DataFrame, assignments: pd.DataFrame
) -> Iterator[tuple[int, pd.DataFrame, pd.DataFrame]]:
    for seed_value, seed_assignments in assignments.groupby("split_seed", sort=False):
        seed = int(seed_value)
        training, testing = _rows_for_seed(rows, seed_assignments, seed)
        yield seed, training, testing


def _validate_cv_folds(
    training: pd.DataFrame, seed: int, settings: config.RunSettings
) -> None:
    folds = training["fold"].astype(int)
    if set(folds.unique()) != set(range(settings.cv_folds)):
        raise SplitError(f"split seed {seed} does not contain all configured CV folds")


def _best_iteration(
    training: pd.DataFrame,
    seed: int,
    fold: int,
    spec: _FitSpec,
) -> int:
    fold_train = training[training["fold"] != fold]
    fold_valid = training[training["fold"] == fold]
    if fold_train.empty or fold_valid.empty:
        raise SplitError(f"split seed {seed} has an empty CV fold {fold}")

    estimator = _new_regressor(seed, spec.max_estimators, spec)
    try:
        estimator.fit(
            fold_train.loc[:, features.FEATURE_COLUMNS],
            fold_train["remaining_delay_h"],
            eval_X=fold_valid.loc[:, features.FEATURE_COLUMNS],
            eval_y=fold_valid["remaining_delay_h"],
            eval_metric=spec.eval_metric,
            categorical_feature=list(features.CATEGORICAL_FEATURES),
            callbacks=[lgb.early_stopping(spec.early_stopping_rounds, verbose=False)],
        )
    except (ValueError, lgb.basic.LightGBMError) as error:
        raise ModelError(
            f"{spec.model_name} failed for split seed {seed}, CV fold {fold}: {error}"
        ) from error
    return estimator.best_iteration_ or spec.max_estimators


def _fit_final_model(
    training: pd.DataFrame,
    seed: int,
    n_estimators: int,
    spec: _FitSpec,
) -> lgb.LGBMRegressor:
    estimator = _new_regressor(seed, n_estimators, spec)
    try:
        estimator.fit(
            training.loc[:, features.FEATURE_COLUMNS],
            training["remaining_delay_h"],
            categorical_feature=list(features.CATEGORICAL_FEATURES),
        )
    except (ValueError, lgb.basic.LightGBMError) as error:
        raise ModelError(
            f"{spec.model_name} failed to fit split seed {seed}: {error}"
        ) from error
    return estimator


def _fit_grouped_models(
    rows: pd.DataFrame,
    assignments: pd.DataFrame,
    settings: config.RunSettings,
    spec: _FitSpec,
) -> tuple[dict[int, lgb.LGBMRegressor], dict[int, pd.DataFrame]]:
    fitted_models: dict[int, lgb.LGBMRegressor] = {}
    test_rows_by_seed: dict[int, pd.DataFrame] = {}

    for seed, training, testing in _rows_by_seed(rows, assignments):
        _validate_cv_folds(training, seed, settings)
        iterations = [
            _best_iteration(training, seed, fold, spec)
            for fold in range(settings.cv_folds)
        ]
        n_estimators = max(1, int(np.median(iterations)))
        fitted_models[seed] = _fit_final_model(training, seed, n_estimators, spec)
        test_rows_by_seed[seed] = testing[
            ["shipment_id", "final_delay_h", *features.FEATURE_COLUMNS]
        ].copy()

    return fitted_models, test_rows_by_seed


def _predict_remaining(
    estimator: lgb.LGBMRegressor,
    test_rows: pd.DataFrame,
    *,
    seed: int,
    model_name: str,
) -> np.ndarray:
    try:
        remaining = estimator.predict(test_rows.loc[:, features.FEATURE_COLUMNS])
    except (ValueError, lgb.basic.LightGBMError) as error:
        raise ModelError(
            f"{model_name} prediction failed for split seed {seed}: {error}"
        ) from error
    if not np.isfinite(remaining).all():
        raise ModelError(
            f"{model_name} returned non-finite predictions for split seed {seed}"
        )
    return remaining
