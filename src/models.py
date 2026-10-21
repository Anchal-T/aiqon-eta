"""Point and interval models for remaining shipment delay."""

from __future__ import annotations

from collections.abc import Mapping

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config, features
from src.errors import ConfigurationError, ContractError, ModelError, SplitError

MODEL_R_PREDICTION_COLUMNS = (
    "split_seed",
    "shipment_id",
    "checkpoint",
    "actual_final_delay_h",
    "delay_so_far",
    "predicted_remaining_delay_h",
    "predicted_final_delay_h",
)
MODEL_Q_PREDICTION_COLUMNS = (
    "split_seed",
    "shipment_id",
    "checkpoint",
    "actual_final_delay_h",
    "delay_so_far",
    "predicted_remaining_lower_h",
    "predicted_remaining_upper_h",
    "predicted_final_lower_h",
    "predicted_final_upper_h",
    "quantiles_crossed",
)
_REQUIRED_SPLITS = ("split_seed", "shipment_id", "outer_split", "fold")
_QUANTILES = (0.1, 0.9)


def _validate_training_options(max_estimators: int, early_stopping_rounds: int) -> None:
    if max_estimators < 1:
        raise ConfigurationError("max_estimators must be at least one")
    if early_stopping_rounds < 1:
        raise ConfigurationError("early_stopping_rounds must be at least one")


def _validated_rows(stacked: pd.DataFrame) -> pd.DataFrame:
    required = {
        "shipment_id",
        "checkpoint",
        "final_delay_h",
        "remaining_delay_h",
        *features.FEATURE_COLUMNS,
    }
    missing = sorted(required - set(stacked.columns))
    if missing:
        raise ContractError(f"stacked table is missing model columns: {missing}")
    if stacked.empty:
        raise ContractError("stacked table has no model rows")

    rows = stacked[
        [
            "shipment_id",
            "final_delay_h",
            "remaining_delay_h",
            *features.FEATURE_COLUMNS,
        ]
    ].copy()
    if rows[["shipment_id", "checkpoint"]].isna().any().any():
        raise ContractError("model shipment IDs and checkpoints must be present")
    rows["shipment_id"] = rows["shipment_id"].astype("string")
    if rows.duplicated(["shipment_id", "checkpoint"]).any():
        raise ContractError("model shipment/checkpoint rows must be unique")

    for column in ("final_delay_h", "remaining_delay_h"):
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
    missing = sorted(set(_REQUIRED_SPLITS) - set(split_manifest.columns))
    if missing:
        raise ContractError(f"split manifest is missing model columns: {missing}")
    if split_manifest.empty:
        raise ContractError("split manifest has no model assignments")

    assignments = split_manifest[list(_REQUIRED_SPLITS)].copy()
    if assignments[["shipment_id", "outer_split"]].isna().any().any():
        raise ContractError("split shipment IDs and outer splits must be present")
    assignments["shipment_id"] = assignments["shipment_id"].astype("string")
    seeds = pd.to_numeric(assignments["split_seed"], errors="coerce")
    if seeds.isna().any() or not np.isfinite(seeds).all() or (seeds % 1 != 0).any():
        raise ContractError("split seeds must be finite integers")
    assignments["split_seed"] = seeds.astype("int64")
    if assignments.duplicated(["split_seed", "shipment_id"]).any():
        raise ContractError("each shipment must have one split assignment per seed")
    if not assignments["outer_split"].isin(("train", "test")).all():
        raise ContractError("outer_split values must be 'train' or 'test'")

    folds = pd.to_numeric(assignments["fold"], errors="coerce")
    is_training = assignments["outer_split"] == "train"
    if folds[is_training].isna().any() or folds[~is_training].notna().any():
        raise ContractError("training rows need folds and test rows must not have them")
    if not np.isfinite(folds[is_training]).all() or (folds[is_training] % 1 != 0).any():
        raise ContractError("training fold values must be finite integers")
    assignments["fold"] = folds.astype("Int64")
    return assignments


def _new_estimator(
    seed: int,
    n_estimators: int,
    objective: str,
    alpha: float | None = None,
) -> lgb.LGBMRegressor:
    options = {
        "objective": objective,
        "n_estimators": n_estimators,
        "num_leaves": 8,
        "min_child_samples": 15,
        "learning_rate": 0.05,
        "colsample_bytree": 0.8,
        "random_state": seed,
        "n_jobs": 1,
        "verbosity": -1,
    }
    if alpha is not None:
        options["alpha"] = alpha
    return lgb.LGBMRegressor(**options)


def _fit_grouped_models(
    rows: pd.DataFrame,
    assignments: pd.DataFrame,
    settings: config.RunSettings,
    *,
    objective: str,
    eval_metric: str,
    model_name: str,
    max_estimators: int,
    early_stopping_rounds: int,
    alpha: float | None = None,
) -> tuple[dict[int, lgb.LGBMRegressor], dict[int, pd.DataFrame]]:
    expected_ids = set(rows["shipment_id"])
    fitted_models: dict[int, lgb.LGBMRegressor] = {}
    test_rows_by_seed: dict[int, pd.DataFrame] = {}

    for seed_value, seed_assignments in assignments.groupby("split_seed", sort=False):
        seed = int(seed_value)
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
        folds = training["fold"].astype(int)
        if set(folds.unique()) != set(range(settings.cv_folds)):
            raise SplitError(
                f"split seed {seed} does not contain all configured CV folds"
            )

        best_iterations = []
        for fold in range(settings.cv_folds):
            fold_train = training[training["fold"] != fold]
            fold_valid = training[training["fold"] == fold]
            if fold_train.empty or fold_valid.empty:
                raise SplitError(f"split seed {seed} has an empty CV fold {fold}")
            estimator = _new_estimator(
                seed, max_estimators, objective=objective, alpha=alpha
            )
            try:
                estimator.fit(
                    fold_train.loc[:, features.FEATURE_COLUMNS],
                    fold_train["remaining_delay_h"],
                    eval_X=fold_valid.loc[:, features.FEATURE_COLUMNS],
                    eval_y=fold_valid["remaining_delay_h"],
                    eval_metric=eval_metric,
                    categorical_feature=list(features.CATEGORICAL_FEATURES),
                    callbacks=[
                        lgb.early_stopping(early_stopping_rounds, verbose=False)
                    ],
                )
            except (ValueError, lgb.basic.LightGBMError) as error:
                raise ModelError(
                    f"{model_name} failed for split seed {seed}, CV fold {fold}: {error}"
                ) from error
            best_iterations.append(estimator.best_iteration_ or max_estimators)

        final_iterations = max(1, int(np.median(best_iterations)))
        final_model = _new_estimator(
            seed, final_iterations, objective=objective, alpha=alpha
        )
        try:
            final_model.fit(
                training.loc[:, features.FEATURE_COLUMNS],
                training["remaining_delay_h"],
                categorical_feature=list(features.CATEGORICAL_FEATURES),
            )
        except (ValueError, lgb.basic.LightGBMError) as error:
            raise ModelError(
                f"{model_name} failed to fit split seed {seed}: {error}"
            ) from error

        fitted_models[seed] = final_model
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


class ModelR:
    """Fit an L1 model per outer split and predict held-out shipments."""

    def __init__(
        self,
        settings: config.RunSettings = config.DEFAULT_RUN_SETTINGS,
        *,
        max_estimators: int = 1000,
        early_stopping_rounds: int = 50,
    ) -> None:
        _validate_training_options(max_estimators, early_stopping_rounds)
        self.settings = settings
        self.max_estimators = max_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self._models: dict[int, lgb.LGBMRegressor] = {}
        self._test_rows: dict[int, pd.DataFrame] = {}

    def fit(self, stacked: pd.DataFrame, split_manifest: pd.DataFrame) -> ModelR:
        """Validate inputs, cross-validate stopping rounds, and fit each split."""
        rows = _validated_rows(stacked)
        assignments = _validated_splits(split_manifest)
        self._models, self._test_rows = _fit_grouped_models(
            rows,
            assignments,
            self.settings,
            objective="regression_l1",
            eval_metric="l1",
            model_name="Model R",
            max_estimators=self.max_estimators,
            early_stopping_rounds=self.early_stopping_rounds,
        )
        return self

    def predict_test(self) -> pd.DataFrame:
        """Return final-delay and remaining-delay predictions for outer-test rows."""
        if not self._models:
            raise ConfigurationError("Model R must be fit before predicting")

        predictions = []
        for seed, estimator in self._models.items():
            test_rows = self._test_rows[seed]
            remaining = _predict_remaining(
                estimator, test_rows, seed=seed, model_name="Model R"
            )
            predictions.append(
                pd.DataFrame(
                    {
                        "split_seed": seed,
                        "shipment_id": test_rows["shipment_id"].astype(str).to_numpy(),
                        "checkpoint": test_rows["checkpoint"].to_numpy(),
                        "actual_final_delay_h": test_rows["final_delay_h"].to_numpy(),
                        "delay_so_far": test_rows["delay_so_far"].to_numpy(),
                        "predicted_remaining_delay_h": remaining,
                        "predicted_final_delay_h": (
                            test_rows["delay_so_far"].to_numpy() + remaining
                        ),
                    }
                )
            )

        return pd.concat(predictions, ignore_index=True).reindex(
            columns=MODEL_R_PREDICTION_COLUMNS
        )

    @property
    def fitted_models(self) -> Mapping[int, lgb.LGBMRegressor]:
        """Fitted final estimators, keyed by outer split seed."""
        return dict(self._models)


class ModelQ:
    """Fit 10th/90th percentile models and return held-out ETA intervals."""

    def __init__(
        self,
        settings: config.RunSettings = config.DEFAULT_RUN_SETTINGS,
        *,
        max_estimators: int = 1000,
        early_stopping_rounds: int = 50,
    ) -> None:
        _validate_training_options(max_estimators, early_stopping_rounds)
        self.settings = settings
        self.max_estimators = max_estimators
        self.early_stopping_rounds = early_stopping_rounds
        self._models: dict[float, dict[int, lgb.LGBMRegressor]] = {}
        self._test_rows: dict[int, pd.DataFrame] = {}

    def fit(self, stacked: pd.DataFrame, split_manifest: pd.DataFrame) -> ModelQ:
        """Fit both quantiles using the same grouped outer splits and CV folds."""
        rows = _validated_rows(stacked)
        assignments = _validated_splits(split_manifest)
        models_by_quantile = {}
        test_rows_by_seed = {}
        for alpha in _QUANTILES:
            fitted, test_rows = _fit_grouped_models(
                rows,
                assignments,
                self.settings,
                objective="quantile",
                eval_metric="quantile",
                model_name=f"Model Q (q={alpha})",
                max_estimators=self.max_estimators,
                early_stopping_rounds=self.early_stopping_rounds,
                alpha=alpha,
            )
            models_by_quantile[alpha] = fitted
            if not test_rows_by_seed:
                test_rows_by_seed = test_rows
        self._models = models_by_quantile
        self._test_rows = test_rows_by_seed
        return self

    def predict_test(self) -> pd.DataFrame:
        """Return ordered 80% prediction intervals for outer-test rows."""
        if not self._models:
            raise ConfigurationError("Model Q must be fit before predicting")

        predictions = []
        lower_models = self._models[0.1]
        upper_models = self._models[0.9]
        for seed, lower_model in lower_models.items():
            test_rows = self._test_rows[seed]
            lower_raw = _predict_remaining(
                lower_model, test_rows, seed=seed, model_name="Model Q q=0.1"
            )
            upper_raw = _predict_remaining(
                upper_models[seed], test_rows, seed=seed, model_name="Model Q q=0.9"
            )
            crossed = lower_raw > upper_raw
            lower = np.minimum(lower_raw, upper_raw)
            upper = np.maximum(lower_raw, upper_raw)
            delay_so_far = test_rows["delay_so_far"].to_numpy()
            predictions.append(
                pd.DataFrame(
                    {
                        "split_seed": seed,
                        "shipment_id": test_rows["shipment_id"].astype(str).to_numpy(),
                        "checkpoint": test_rows["checkpoint"].to_numpy(),
                        "actual_final_delay_h": test_rows["final_delay_h"].to_numpy(),
                        "delay_so_far": delay_so_far,
                        "predicted_remaining_lower_h": lower,
                        "predicted_remaining_upper_h": upper,
                        "predicted_final_lower_h": delay_so_far + lower,
                        "predicted_final_upper_h": delay_so_far + upper,
                        "quantiles_crossed": crossed,
                    }
                )
            )

        return pd.concat(predictions, ignore_index=True).reindex(
            columns=MODEL_Q_PREDICTION_COLUMNS
        )
