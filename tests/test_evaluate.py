from __future__ import annotations

import pandas as pd
import pytest

from src.errors import ContractError
from src.evaluate import (
    build_evaluation_group_metadata,
    compare_model_r_to_b2,
    compare_model_r_to_b2_by_group,
    evaluate_model_c_classification_metrics,
    evaluate_model_c_classification_metrics_by_group,
    evaluate_model_c_reliability,
    evaluate_model_c_reliability_by_group,
    evaluate_model_c_threshold_sweep,
    evaluate_model_c_threshold_sweep_by_group,
    evaluate_model_q_intervals,
    evaluate_model_q_intervals_by_group,
)


def _group_metadata() -> pd.DataFrame:
    shipments = pd.DataFrame(
        [
            ("A", "J1", None),
            ("B", "J1", "HUB"),
            ("C", "J2", "HUB"),
            ("D", "J2", "HUB"),
        ],
        columns=("shipment_id", "journey_id", "hub"),
    )
    truth = pd.DataFrame(
        [(shipment, shipment == "D") for shipment in "ABCD"],
        columns=("shipment_id", "missed_connection"),
    )
    hidden_mask = pd.DataFrame(
        [("B", "act_customs"), ("D", "act_port_done"), ("D", "act_customs")],
        columns=("shipment_id", "column"),
    )
    return build_evaluation_group_metadata(shipments, truth, hidden_mask)


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
    assert comparison["model_median_ae_mean_h"] == pytest.approx(1.25)
    assert comparison["b2_median_ae_mean_h"] == pytest.approx(3.25)
    assert comparison["model_p90_ae_mean_h"] == pytest.approx(1.85)
    assert comparison["b2_p90_ae_mean_h"] == pytest.approx(4.25)
    assert comparison["model_rmse_mean_h"] == pytest.approx((2**0.5 + 2.5**0.5) / 2)
    assert comparison["b2_rmse_mean_h"] == pytest.approx((16**0.5 + 12.5**0.5) / 2)
    assert comparison["model_beats_b2_seeds"] == 2
    assert comparison["n_split_seeds"] == 2


def test_model_r_group_metrics_report_all_cuts_by_checkpoint_across_seeds():
    predictions = pd.DataFrame(
        [
            (seed, shipment, 4, actual, actual + errors[shipment])
            for seed, errors in (
                (1, {"A": 2.0, "B": 1.0, "C": 1.0, "D": 4.0}),
                (2, {"A": 1.0, "B": 2.0, "C": 4.0, "D": 1.0}),
            )
            for shipment, actual in zip("ABCD", (10.0, 20.0, 30.0, 40.0))
        ],
        columns=(
            "split_seed",
            "shipment_id",
            "checkpoint",
            "actual_final_delay_h",
            "predicted_final_delay_h",
        ),
    )
    baseline_predictions = pd.DataFrame(
        [
            (seed, shipment, 4, "B2", actual + errors[shipment])
            for seed, errors in (
                (1, {"A": 4.0, "B": 2.0, "C": 3.0, "D": 1.0}),
                (2, {"A": 2.0, "B": 4.0, "C": 1.0, "D": 3.0}),
            )
            for shipment, actual in zip("ABCD", (10.0, 20.0, 30.0, 40.0))
        ],
        columns=(
            "split_seed",
            "shipment_id",
            "checkpoint",
            "baseline",
            "predicted_final_delay_h",
        ),
    )
    metrics = compare_model_r_to_b2_by_group(
        predictions, baseline_predictions, _group_metadata()
    )

    assert set(metrics["group_type"]) == {
        "journey",
        "connection_type",
        "missed_connection",
        "hidden_stage",
    }
    assert set(
        metrics.loc[metrics["group_type"] == "connection_type", "group_value"]
    ) == {"direct", "connecting"}
    assert set(
        metrics.loc[metrics["group_type"] == "missed_connection", "group_value"]
    ) == {"true", "false"}
    assert set(metrics.loc[metrics["group_type"] == "hidden_stage", "group_value"]) == {
        "with_hidden_stage",
        "without_hidden_stage",
    }
    journey = metrics.loc[
        (metrics["group_type"] == "journey") & (metrics["group_value"] == "J1")
    ].iloc[0]
    assert journey["model_mae_mean_h"] == pytest.approx(1.5)
    assert journey["b2_mae_mean_h"] == pytest.approx(3.0)
    assert journey["n_split_seeds"] == 2


def test_model_r_group_metrics_reject_metadata_missing_prediction_shipments():
    predictions = pd.DataFrame(
        [(1, "A", 4, 10.0, 11.0)],
        columns=(
            "split_seed",
            "shipment_id",
            "checkpoint",
            "actual_final_delay_h",
            "predicted_final_delay_h",
        ),
    )
    baseline_predictions = pd.DataFrame(
        [(1, "A", 4, "B2", 12.0)],
        columns=(
            "split_seed",
            "shipment_id",
            "checkpoint",
            "baseline",
            "predicted_final_delay_h",
        ),
    )
    metadata = build_evaluation_group_metadata(
        pd.DataFrame([("B", "J1", None)], columns=("shipment_id", "journey_id", "hub")),
        pd.DataFrame([("B", False)], columns=("shipment_id", "missed_connection")),
        pd.DataFrame(columns=("shipment_id", "column")),
    )

    with pytest.raises(ContractError, match="metadata is missing"):
        compare_model_r_to_b2_by_group(predictions, baseline_predictions, metadata)


def test_model_q_group_metrics_report_interval_quality_by_cut():
    predictions = pd.DataFrame(
        [
            (1, "A", 4, 10.0, 8.0, 12.0, False),
            (1, "B", 4, 20.0, 22.0, 24.0, False),
            (1, "C", 4, 30.0, 28.0, 32.0, False),
            (1, "D", 4, 40.0, 35.0, 45.0, True),
            (2, "A", 4, 10.0, 9.0, 11.0, False),
            (2, "B", 4, 20.0, 19.0, 23.0, False),
            (2, "C", 4, 30.0, 29.0, 31.0, False),
            (2, "D", 4, 40.0, 38.0, 42.0, False),
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

    metrics = evaluate_model_q_intervals_by_group(predictions, _group_metadata())
    journey = metrics.loc[
        (metrics["group_type"] == "journey") & (metrics["group_value"] == "J1")
    ].iloc[0]

    assert journey["coverage_mean"] == pytest.approx(0.75)
    assert journey["mean_width_h"] == pytest.approx(3.0)
    assert journey["n_split_seeds"] == 2
    assert set(metrics["group_type"]) == {
        "journey",
        "connection_type",
        "missed_connection",
        "hidden_stage",
    }


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


def test_model_c_group_metrics_cover_classification_sweep_and_reliability():
    predictions = pd.DataFrame(
        [
            (1, "A", 4, 0, 0.1),
            (1, "B", 4, 1, 0.8),
            (1, "C", 4, 1, 0.7),
            (1, "D", 4, 0, 0.2),
            (2, "A", 4, 0, 0.2),
            (2, "B", 4, 1, 0.7),
            (2, "C", 4, 1, 0.8),
            (2, "D", 4, 0, 0.1),
        ],
        columns=(
            "split_seed",
            "shipment_id",
            "checkpoint",
            "actual_is_delayed",
            "predicted_delay_probability",
        ),
    )
    metadata = _group_metadata()

    classification = evaluate_model_c_classification_metrics_by_group(
        predictions, metadata
    )
    j1 = classification.loc[
        (classification["group_type"] == "journey")
        & (classification["group_value"] == "J1")
    ].iloc[0]
    threshold_sweep = evaluate_model_c_threshold_sweep_by_group(predictions, metadata)
    j1_at_half = threshold_sweep.loc[
        (threshold_sweep["group_type"] == "journey")
        & (threshold_sweep["group_value"] == "J1")
        & (threshold_sweep["threshold"] == 0.5)
    ].iloc[0]
    reliability = evaluate_model_c_reliability_by_group(predictions, metadata)

    assert j1["brier_score_mean"] == pytest.approx(0.045)
    assert j1["roc_auc_mean"] == pytest.approx(1.0)
    assert j1["f1_at_0_5_mean"] == pytest.approx(1.0)
    assert j1_at_half["precision_mean"] == pytest.approx(1.0)
    assert j1_at_half["recall_mean"] == pytest.approx(1.0)
    assert set(reliability["group_type"]) == {
        "journey",
        "connection_type",
        "missed_connection",
        "hidden_stage",
    }


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
