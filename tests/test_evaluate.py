from __future__ import annotations

import pandas as pd
import pytest

from src.errors import ContractError
from src.evaluate import (
    compare_model_r_to_b2,
    evaluate_model_c_classification_metrics,
    evaluate_model_c_reliability,
    evaluate_model_c_threshold_sweep,
    evaluate_model_q_intervals,
)


def test_model_r_comparison_reports_held_out_mae_by_checkpoint():
    model = pd.DataFrame(
        [
            (1, "A", 4, 10.0, 12.0),
            (1, "B", 4, 20.0, 20.0),
            (2, "C", 4, 5.0, 6.0),
            (2, "D", 4, 15.0, 13.0),
        ],
        columns=(
            "split_seed",
            "shipment_id",
            "checkpoint",
            "actual_final_delay_h",
            "predicted_final_delay_h",
        ),
    )
    baselines = pd.DataFrame(
        [
            (1, "A", 4, "B2", 14.0),
            (1, "B", 4, "B2", 16.0),
            (2, "C", 4, "B2", 5.0),
            (2, "D", 4, "B2", 10.0),
        ],
        columns=(
            "split_seed",
            "shipment_id",
            "checkpoint",
            "baseline",
            "predicted_final_delay_h",
        ),
    )

    comparison = compare_model_r_to_b2(model, baselines).iloc[0]

    assert comparison["model_mae_mean_h"] == pytest.approx(1.25)
    assert comparison["b2_mae_mean_h"] == pytest.approx(3.25)
    assert comparison["model_beats_b2_seeds"] == 2
    assert comparison["n_split_seeds"] == 2


def test_model_q_interval_metrics_report_coverage_width_and_crossings():
    predictions = pd.DataFrame(
        [
            (1, "A", 4, 10.0, 8.0, 12.0, False),
            (1, "B", 4, 20.0, 12.0, 18.0, False),
            (2, "C", 4, 5.0, 4.0, 7.0, True),
            (2, "D", 4, 15.0, 16.0, 20.0, False),
        ],
        columns=(
            "split_seed",
            "shipment_id",
            "checkpoint",
            "actual_final_delay_h",
            "predicted_final_lower_h",
            "predicted_final_upper_h",
            "quantiles_crossed",
        ),
    )

    metrics = evaluate_model_q_intervals(predictions).iloc[0]

    assert metrics["nominal_coverage"] == pytest.approx(0.8)
    assert metrics["coverage_mean"] == pytest.approx(0.5)
    assert metrics["mean_width_h"] == pytest.approx(4.25)
    assert metrics["quantile_crossing_rate"] == pytest.approx(0.25)
    assert metrics["n_split_seeds"] == 2


def test_model_c_reliability_reports_observed_rate_by_checkpoint_and_bin():
    predictions = pd.DataFrame(
        [
            (1, "A", 2, 0, 0.05),
            (1, "B", 2, 1, 0.09),
            (1, "C", 2, 1, 0.15),
            (1, "D", 2, 1, 0.85),
            (1, "E", 2, 0, 1.0),
            (2, "A", 3, 1, 0.95),
        ],
        columns=(
            "split_seed",
            "shipment_id",
            "checkpoint",
            "actual_is_delayed",
            "predicted_delay_probability",
        ),
    )

    curve = evaluate_model_c_reliability(predictions)

    assert curve["checkpoint"].tolist() == [2, 2, 2, 2, 3]
    assert curve["probability_bin"].tolist() == [0, 1, 8, 9, 9]
    assert curve["mean_predicted_probability"].tolist() == pytest.approx(
        [0.07, 0.15, 0.85, 1.0, 0.95]
    )
    assert curve["observed_delay_rate"].tolist() == [0.5, 1.0, 1.0, 0.0, 1.0]
    assert curve["n_predictions"].tolist() == [2, 1, 1, 1, 1]


def test_model_c_metrics_and_threshold_sweep_summarize_split_seeds():
    predictions = pd.DataFrame(
        [
            (1, "A", 2, 0, 0.1),
            (1, "B", 2, 1, 0.8),
            (1, "C", 2, 0, 0.4),
            (1, "D", 2, 1, 0.7),
            (2, "E", 2, 0, 0.2),
            (2, "F", 2, 1, 0.6),
            (2, "G", 2, 0, 0.3),
            (2, "H", 2, 1, 0.1),
        ],
        columns=(
            "split_seed",
            "shipment_id",
            "checkpoint",
            "actual_is_delayed",
            "predicted_delay_probability",
        ),
    )

    metrics = evaluate_model_c_classification_metrics(predictions).iloc[0]
    threshold = evaluate_model_c_threshold_sweep(predictions)
    at_half = threshold.loc[
        (threshold["checkpoint"] == 2) & (threshold["threshold"] == 0.5)
    ].iloc[0]

    assert metrics["brier_score_mean"] == pytest.approx(0.175)
    assert metrics["roc_auc_mean"] == pytest.approx(0.75)
    assert metrics["pr_auc_mean"] == pytest.approx(41 / 48)
    assert metrics["f1_at_0_5_mean"] == pytest.approx(5 / 6)
    assert metrics["n_split_seeds"] == 2
    assert at_half["precision_mean"] == pytest.approx(1.0)
    assert at_half["recall_mean"] == pytest.approx(0.75)
    assert at_half["f1_mean"] == pytest.approx(5 / 6)


def test_model_c_auc_metrics_are_missing_when_a_seed_has_one_label():
    predictions = pd.DataFrame(
        [(1, "A", 2, 0, 0.1), (1, "B", 2, 0, 0.2)],
        columns=(
            "split_seed",
            "shipment_id",
            "checkpoint",
            "actual_is_delayed",
            "predicted_delay_probability",
        ),
    )

    metrics = evaluate_model_c_classification_metrics(predictions).iloc[0]

    assert pd.isna(metrics["roc_auc_mean"])
    assert pd.isna(metrics["pr_auc_mean"])
    assert metrics["n_auc_split_seeds"] == 0


def test_model_c_reliability_rejects_non_binary_labels_and_invalid_probabilities():
    predictions = pd.DataFrame(
        [(1, "A", 2, 2, 0.5)],
        columns=(
            "split_seed",
            "shipment_id",
            "checkpoint",
            "actual_is_delayed",
            "predicted_delay_probability",
        ),
    )

    with pytest.raises(ContractError, match="binary labels"):
        evaluate_model_c_reliability(predictions)

    predictions["actual_is_delayed"] = 1
    predictions["predicted_delay_probability"] = 1.1
    with pytest.raises(ContractError, match="values must be between 0 and 1"):
        evaluate_model_c_reliability(predictions)
