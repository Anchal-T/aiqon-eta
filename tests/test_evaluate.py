from __future__ import annotations

import pandas as pd
import pytest

from src.evaluate import compare_model_r_to_b2, evaluate_model_q_intervals


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
