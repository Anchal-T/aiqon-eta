from __future__ import annotations

import numpy as np
import pandas as pd

from src import config, features
from src.explain import (
    build_model_r_explanations,
    plot_model_r_dependence,
    plot_model_r_shap_summary,
    plot_model_r_waterfall,
)
from src.model_r import ModelR


def _model_inputs() -> tuple[
    config.RunSettings, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame
]:
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
        for checkpoint in (2, 4):
            row: dict[str, object] = {
                column: 0.0 for column in features.FEATURE_COLUMNS
            }
            hub_arrival_delay = 1.0 + number / 10
            delay_so_far = hub_arrival_delay + (0.5 if checkpoint == 4 else 0)
            final_delay = 5.0 + number / 5
            row.update(
                {
                    "shipment_id": shipment_id,
                    "checkpoint": checkpoint,
                    "is_connecting": 1,
                    "delay_departure": 0.5 + number / 20,
                    "delay_hub_arrival": hub_arrival_delay,
                    "added_hub_arrival": hub_arrival_delay - (0.5 + number / 20),
                    "delay_so_far": delay_so_far,
                    "remaining_delay_h": final_delay - delay_so_far,
                    "final_delay_h": final_delay,
                    "slack_remaining": 4.0 - number / 3,
                    "slot_margin": 2.0 - number / 4,
                }
            )
            if checkpoint == 2:
                for stage in ("hub_departure", "port_arrival", "port_done", "customs"):
                    row[f"delay_{stage}"] = np.nan
                    row[f"added_{stage}"] = np.nan
                row["slot_margin"] = np.nan
                row["congestion_port"] = np.nan
                row["arrival_dow"] = np.nan
                row["cong_x_dwell"] = np.nan
                row["cong_x_customs"] = np.nan
                row["port_done_dow"] = np.nan
                row["port_done_is_weekend"] = np.nan
                row["port_done_missing"] = np.nan
                row["customs_missing"] = np.nan
            else:
                row.update(
                    {
                        "delay_hub_departure": hub_arrival_delay + 1.0,
                        "added_hub_departure": 1.0,
                        "delay_port_arrival": hub_arrival_delay + 0.5,
                        "added_port_arrival": -0.5,
                        "delay_port_done": np.nan,
                        "added_port_done": np.nan,
                        "delay_customs": np.nan,
                        "added_customs": np.nan,
                        "port_done_dow": np.nan,
                        "port_done_is_weekend": np.nan,
                        "port_done_missing": np.nan,
                        "customs_missing": np.nan,
                    }
                )
            stacked_rows.append(row)

    delays = pd.DataFrame(
        [
            {
                "shipment_id": f"S{number:03d}",
                **{f"merged_{stage}": False for stage in config.STAGE_ORDER},
            }
            for number in range(settings.n_shipments)
        ]
    )
    truth = pd.DataFrame(
        [
            {
                "shipment_id": f"S{number:03d}",
                "missed_connection": number % 2 == 0,
                "missed_slot": number % 3 == 0,
                "storm_leg1": number % 4 == 0,
                "storm_leg2": number % 4 == 1,
            }
            for number in range(settings.n_shipments)
        ]
    )
    return (
        settings,
        pd.DataFrame(stacked_rows),
        pd.DataFrame(assignments),
        delays,
        truth,
    )


def test_model_r_explanations_are_additive_and_use_visible_stage_facts():
    settings, stacked, assignments, delays, truth = _model_inputs()
    model = ModelR(settings, max_estimators=10, early_stopping_rounds=2).fit(
        stacked, assignments
    )

    explanations, recovery = build_model_r_explanations(model, delays, truth)

    assert len(explanations) == 8
    assert not {
        "missed_connection",
        "missed_slot",
        "storm_leg1",
        "storm_leg2",
    } & set(explanations.columns)
    shap_columns = [
        column for column in explanations if column.startswith("shap_group_")
    ]
    np.testing.assert_allclose(
        explanations["base_value"] + explanations[shap_columns].sum(axis=1),
        explanations["predicted_remaining_delay_h"],
        rtol=1e-5,
        atol=1e-5,
    )
    fact_columns = [f"fact_added_{stage}_h" for stage in config.STAGE_ORDER[:-1]]
    np.testing.assert_allclose(
        explanations[fact_columns].fillna(0).sum(axis=1),
        explanations["delay_so_far"],
    )
    checkpoint_two = explanations.loc[explanations["checkpoint"] == 2]
    assert checkpoint_two["fact_added_hub_departure_h"].isna().all()
    assert set(recovery["check"]) == {
        "cliff_recovery_c2",
        "slot_recovery_c4",
        "storm_leg1_fact_top",
        "storm_leg2_fact_top",
    }


def test_model_r_explanation_plots_render(tmp_path):
    settings, stacked, assignments, delays, truth = _model_inputs()
    model = ModelR(settings, max_estimators=10, early_stopping_rounds=2).fit(
        stacked, assignments
    )
    explanations, _ = build_model_r_explanations(model, delays, truth)
    outputs = (
        tmp_path / "summary.png",
        tmp_path / "slack.png",
        tmp_path / "slot.png",
        tmp_path / "waterfall.png",
    )

    plot_model_r_shap_summary(explanations, outputs[0])
    plot_model_r_dependence(explanations, "slack_remaining", outputs[1])
    plot_model_r_dependence(explanations, "slot_margin", outputs[2])
    plot_model_r_waterfall(explanations.iloc[0], outputs[3])

    assert all(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n") for path in outputs)
