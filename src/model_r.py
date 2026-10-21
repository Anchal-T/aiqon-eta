"""Point-regression model for remaining shipment delay."""

from __future__ import annotations

from collections.abc import Mapping

import lightgbm as lgb
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

MODEL_R_PREDICTION_COLUMNS = (
    "split_seed",
    "shipment_id",
    "checkpoint",
    "actual_final_delay_h",
    "delay_so_far",
    "predicted_remaining_delay_h",
    "predicted_final_delay_h",
)


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
        rows = _validated_rows(stacked, ("final_delay_h", "remaining_delay_h"))
        assignments = _validated_splits(split_manifest)
        self._models, self._test_rows = _fit_grouped_models(
            rows,
            assignments,
            self.settings,
            _FitSpec(
                model_name="Model R",
                objective="regression_l1",
                eval_metric="l1",
                max_estimators=self.max_estimators,
                early_stopping_rounds=self.early_stopping_rounds,
            ),
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
