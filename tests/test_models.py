from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, features
from src.errors import ContractError
from src.evaluate import compare_model_r_to_b2, evaluate_model_c_classification_metrics
from src.model_c import ModelC
from src.model_r import ModelR


def _small_model_tables():
    settings = config.RunSettings(seed=17, n_shipments=12, cv_folds=2, n_split_seeds=1)
    stacked_rows = []
    assignments = []
    for number in range(settings.n_shipments):
        shipment_id = f"S{number:03d}"
        is_test = number >= 8
        assignments.append(
            {
                "split_seed": 17,
                "shipment_id": shipment_id,
                "outer_split": "test" if is_test else "train",
                "fold": None if is_test else number // 4,
            }
        )
        for checkpoint in range(2):
            final_delay = 2.0 + (number % 5) * 1.5 + checkpoint
            delay_so_far = checkpoint * (number % 3)
            row: dict[str, object] = {
                column: 0.0 for column in features.FEATURE_COLUMNS
            }
            row.update(
                {
                    "shipment_id": shipment_id,
                    "checkpoint": checkpoint,
                    "delay_so_far": delay_so_far,
                    "final_delay_h": final_delay,
                    "remaining_delay_h": final_delay - delay_so_far,
                    "is_delayed": int(final_delay > settings.delay_threshold_h),
                }
            )
            stacked_rows.append(row)
    return settings, pd.DataFrame(stacked_rows), pd.DataFrame(assignments)


def test_model_r_rejects_infinite_features_at_fit_boundary():
    row: dict[str, object] = {column: 0.0 for column in features.FEATURE_COLUMNS}
    row.update(
        {
            "shipment_id": "S1",
            "final_delay_h": 0.0,
            "remaining_delay_h": 0.0,
            "delay_so_far": float("inf"),
        }
    )

    with pytest.raises(ContractError, match="delay_so_far must be finite or missing"):
        ModelR().fit(pd.DataFrame([row]), pd.DataFrame())


def test_model_c_tunes_grouped_folds_and_predicts_outer_test_rows_only():
    settings, stacked, assignments = _small_model_tables()
    expected_test_ids = {f"S{number:03d}" for number in range(8, 12)}
    model = ModelC(settings).fit(stacked, assignments)

    predictions = model.predict_test()
    tuning = model.tuning_results

    assert len(predictions) == len(expected_test_ids) * 2
    assert set(predictions["shipment_id"]) == expected_test_ids
    assert set(predictions["split_seed"]) == {17}
    probabilities = predictions["predicted_delay_probability"].to_numpy()
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()
    assert len(tuning) == 4
    assert tuning["selected"].sum() == 1
    assert np.isfinite(tuning["mean_cv_log_loss"]).all()


def test_model_r_feature_ablation_sets_are_nested_and_complete():
    a0, a1, a2 = (set(features.MODEL_R_ABLATIONS[name]) for name in ("A0", "A1", "A2"))

    assert a0 < a1 < a2
    assert features.MODEL_R_ABLATIONS["A2"] == features.FEATURE_COLUMNS


def test_same_seed_model_metrics_are_reproducible():
    settings, stacked, assignments = _small_model_tables()
    first_r = (
        ModelR(settings, max_estimators=12, early_stopping_rounds=2)
        .fit(stacked, assignments)
        .predict_test()
    )
    second_r = (
        ModelR(settings, max_estimators=12, early_stopping_rounds=2)
        .fit(stacked, assignments)
        .predict_test()
    )
    pd.testing.assert_frame_equal(first_r, second_r)

    baseline = first_r[["split_seed", "shipment_id", "checkpoint"]].copy()
    baseline["baseline"] = "B2"
    baseline["predicted_final_delay_h"] = first_r["delay_so_far"].to_numpy()
    pd.testing.assert_frame_equal(
        compare_model_r_to_b2(first_r, baseline),
        compare_model_r_to_b2(second_r, baseline),
    )

    first_c = ModelC(settings).fit(stacked, assignments)
    second_c = ModelC(settings).fit(stacked, assignments)
    first_predictions = first_c.predict_test()
    second_predictions = second_c.predict_test()
    pd.testing.assert_frame_equal(first_predictions, second_predictions)
    pd.testing.assert_frame_equal(
        evaluate_model_c_classification_metrics(first_predictions),
        evaluate_model_c_classification_metrics(second_predictions),
    )


def test_model_c_rejects_non_binary_targets_at_fit_boundary():
    row: dict[str, object] = {column: 0.0 for column in features.FEATURE_COLUMNS}
    row.update(
        {
            "shipment_id": "S1",
            "is_delayed": 0.5,
        }
    )

    with pytest.raises(ContractError, match="is_delayed must contain binary labels"):
        ModelC().fit(pd.DataFrame([row]), pd.DataFrame())
