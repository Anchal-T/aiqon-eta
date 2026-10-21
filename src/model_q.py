"""Quantile-regression model for remaining shipment delay intervals."""

from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from src import config
from src.errors import ConfigurationError
from src.model_training import (
    _fit_grouped_models,
    _FitSpec,
    _predict_remaining,
    _validate_training_options,
    _validated_rows,
    _validated_splits,
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

_QUANTILES = (0.1, 0.9)


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
        rows = _validated_rows(stacked, ("final_delay_h", "remaining_delay_h"))
        assignments = _validated_splits(split_manifest)
        models_by_quantile = {}
        test_rows_by_seed = {}
        for alpha in _QUANTILES:
            fitted, test_rows = _fit_grouped_models(
                rows,
                assignments,
                self.settings,
                _FitSpec(
                    model_name=f"Model Q (q={alpha})",
                    objective="quantile",
                    eval_metric="quantile",
                    max_estimators=self.max_estimators,
                    early_stopping_rounds=self.early_stopping_rounds,
                    alpha=alpha,
                ),
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
