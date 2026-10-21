"""Model R feature-set ablations from the LLD."""

from __future__ import annotations

import pandas as pd

from src import config, features
from src.evaluate import compare_model_r_to_b2
from src.model_r import ModelR


def evaluate_model_r_ablations(
    stacked: pd.DataFrame,
    split_manifest: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
    full_model_predictions: pd.DataFrame,
    settings: config.RunSettings,
) -> pd.DataFrame:
    """Compare A0/A1 against the already-fitted full A2 Model R."""
    results = []
    for variant, columns in features.MODEL_R_ABLATIONS.items():
        predictions = (
            full_model_predictions
            if variant == "A2"
            else ModelR(settings, feature_columns=columns)
            .fit(stacked, split_manifest)
            .predict_test()
        )
        metrics = compare_model_r_to_b2(predictions, baseline_predictions)
        metrics.insert(0, "variant", variant)
        results.append(metrics)
    return pd.concat(results, ignore_index=True)
