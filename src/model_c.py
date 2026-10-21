"""Binary classifier for held-out shipment delay probabilities."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config, features
from src.errors import ConfigurationError, ContractError, ModelError, SplitError
from src.model_training import (
    _estimator_options,
    _rows_by_seed,
    _validated_rows,
    _validated_splits,
)

MODEL_C_PREDICTION_COLUMNS = (
    "split_seed",
    "shipment_id",
    "checkpoint",
    "actual_is_delayed",
    "predicted_delay_probability",
)
_DEFAULT_N_ESTIMATORS = 100
_PARAMETER_GRID = (
    (
        "small",
        {
            "max_depth": 3,
            "num_leaves": 4,
            "learning_rate": 0.05,
            "min_child_samples": 15,
        },
    ),
    (
        "medium",
        {
            "max_depth": 4,
            "num_leaves": 8,
            "learning_rate": 0.05,
            "min_child_samples": 15,
        },
    ),
    (
        "regularized",
        {
            "max_depth": 3,
            "num_leaves": 4,
            "learning_rate": 0.05,
            "min_child_samples": 30,
        },
    ),
    (
        "faster_learning",
        {
            "max_depth": 4,
            "num_leaves": 8,
            "learning_rate": 0.1,
            "min_child_samples": 15,
        },
    ),
)


def _validated_classification_rows(stacked: pd.DataFrame) -> pd.DataFrame:
    rows = _validated_rows(stacked, ("is_delayed",))
    if not rows["is_delayed"].isin((0, 1)).all():
        raise ContractError("model target is_delayed must contain binary labels")
    if rows["is_delayed"].nunique() != 2:
        raise ContractError("Model C requires both delayed and non-delayed rows")
    if rows.groupby("shipment_id")["is_delayed"].nunique().gt(1).any():
        raise ContractError("each shipment must have one is_delayed label")
    return rows


def _fit_classifier(
    training: pd.DataFrame, seed: int, parameters: dict[str, int | float]
) -> lgb.LGBMClassifier:
    if training["is_delayed"].nunique() != 2:
        raise SplitError(f"split seed {seed} training data needs both labels")

    options = _estimator_options(seed, _DEFAULT_N_ESTIMATORS)
    options.update(parameters)
    estimator = lgb.LGBMClassifier(objective="binary", **options)
    try:
        estimator.fit(
            training.loc[:, features.FEATURE_COLUMNS],
            training["is_delayed"].astype(int),
            categorical_feature=list(features.CATEGORICAL_FEATURES),
        )
    except (ValueError, lgb.basic.LightGBMError) as error:
        raise ModelError(f"Model C failed to fit split seed {seed}: {error}") from error
    return estimator


def _predict_delay_probability(
    estimator: lgb.LGBMClassifier, test_rows: pd.DataFrame, seed: int
) -> np.ndarray:
    try:
        probabilities = np.asarray(
            estimator.predict_proba(test_rows.loc[:, features.FEATURE_COLUMNS]),
            dtype=float,
        )
        positive_index = list(estimator.classes_).index(1)
        delayed = probabilities[:, positive_index]
    except (ValueError, lgb.basic.LightGBMError, IndexError) as error:
        raise ModelError(
            f"Model C prediction failed for split seed {seed}: {error}"
        ) from error
    if not np.isfinite(delayed).all() or ((delayed < 0) | (delayed > 1)).any():
        raise ModelError(
            f"Model C returned invalid delay probabilities for split seed {seed}"
        )
    return delayed


def _tune_classifier(
    training: pd.DataFrame, seed: int, settings: config.RunSettings
) -> tuple[dict[str, int | float], list[dict[str, object]]]:
    folds = training["fold"].astype(int)
    if set(folds.unique()) != set(range(settings.cv_folds)):
        raise SplitError(f"split seed {seed} does not contain all configured CV folds")

    results = []
    for candidate, parameters in _PARAMETER_GRID:
        fold_losses = []
        for fold in range(settings.cv_folds):
            fold_train = training[folds != fold]
            fold_valid = training[folds == fold]
            if fold_train.empty or fold_valid.empty:
                raise SplitError(f"split seed {seed} has an empty CV fold {fold}")
            estimator = _fit_classifier(fold_train, seed, parameters)
            probabilities = _predict_delay_probability(estimator, fold_valid, seed)
            labels = fold_valid["is_delayed"].to_numpy(dtype=int)
            clipped = np.clip(
                probabilities, np.finfo(float).eps, 1 - np.finfo(float).eps
            )
            row_losses = -(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))
            per_shipment = pd.DataFrame(
                {
                    "shipment_id": fold_valid["shipment_id"].to_numpy(),
                    "loss": row_losses,
                }
            )
            fold_losses.append(
                float(per_shipment.groupby("shipment_id")["loss"].mean().mean())
            )

        results.append(
            {
                "split_seed": seed,
                "candidate": candidate,
                **parameters,
                "mean_cv_log_loss": float(np.mean(fold_losses)),
                "selected": False,
            }
        )

    best_index = min(
        range(len(results)), key=lambda index: results[index]["mean_cv_log_loss"]
    )
    results[best_index]["selected"] = True
    parameters = dict(_PARAMETER_GRID[best_index][1])
    return parameters, results


class ModelC:
    """Tune a small grouped-CV grid, then fit one classifier per outer split."""

    def __init__(
        self, settings: config.RunSettings = config.DEFAULT_RUN_SETTINGS
    ) -> None:
        self.settings = settings
        self._models: dict[int, lgb.LGBMClassifier] = {}
        self._test_rows: dict[int, pd.DataFrame] = {}
        self._tuning_results = pd.DataFrame()

    def fit(self, stacked: pd.DataFrame, split_manifest: pd.DataFrame) -> ModelC:
        """Fit on outer-training shipments and retain only outer-test rows."""
        rows = _validated_classification_rows(stacked)
        assignments = _validated_splits(split_manifest)
        fitted_models = {}
        test_rows_by_seed = {}
        tuning_results = []

        for seed, training, testing in _rows_by_seed(rows, assignments):
            parameters, seed_results = _tune_classifier(training, seed, self.settings)
            tuning_results.extend(seed_results)
            fitted_models[seed] = _fit_classifier(training, seed, parameters)
            test_rows_by_seed[seed] = testing[
                ["shipment_id", "is_delayed", *features.FEATURE_COLUMNS]
            ].copy()

        self._models = fitted_models
        self._test_rows = test_rows_by_seed
        self._tuning_results = pd.DataFrame.from_records(tuning_results)
        return self

    @property
    def tuning_results(self) -> pd.DataFrame:
        """Candidate CV scores and selected settings for each outer split."""
        if not self._models:
            raise ConfigurationError(
                "Model C must be fit before reading tuning results"
            )
        return self._tuning_results.copy()

    def predict_test(self) -> pd.DataFrame:
        """Return held-out delay probabilities by split, shipment, and checkpoint."""
        if not self._models:
            raise ConfigurationError("Model C must be fit before predicting")

        predictions = []
        for seed, estimator in self._models.items():
            test_rows = self._test_rows[seed]
            probability = _predict_delay_probability(estimator, test_rows, seed)
            predictions.append(
                pd.DataFrame(
                    {
                        "split_seed": seed,
                        "shipment_id": test_rows["shipment_id"].astype(str).to_numpy(),
                        "checkpoint": test_rows["checkpoint"].to_numpy(),
                        "actual_is_delayed": test_rows["is_delayed"]
                        .astype("int8")
                        .to_numpy(),
                        "predicted_delay_probability": probability,
                    }
                )
            )

        return pd.concat(predictions, ignore_index=True).reindex(
            columns=MODEL_C_PREDICTION_COLUMNS
        )
