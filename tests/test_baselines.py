from __future__ import annotations

import pandas as pd

from src.baselines import predict


def test_baselines_predict_outer_test_rows_without_leaking_test_targets():
    stacked = pd.DataFrame(
        [
            ("S1", 0, 0, 10.0, 0.0),
            ("S1", 0, 1, 10.0, 2.0),
            ("S2", 0, 0, 20.0, -3.0),
            ("S3", 0, 0, 100.0, 7.0),
            ("S3", 0, 1, 100.0, 8.0),
            ("S4", 1, 0, 40.0, 4.0),
            ("S5", 1, 0, 90.0, -2.0),
        ],
        columns=(
            "shipment_id",
            "journey_id",
            "checkpoint",
            "final_delay_h",
            "delay_so_far",
        ),
    )
    splits = pd.DataFrame(
        [
            (42, "S1", "train"),
            (42, "S2", "train"),
            (42, "S3", "test"),
            (42, "S4", "train"),
            (42, "S5", "test"),
        ],
        columns=("split_seed", "shipment_id", "outer_split"),
    )

    predictions = predict(stacked, splits)

    assert set(predictions["shipment_id"]) == {"S3", "S5"}
    assert set(predictions["baseline"]) == {"B0", "B1", "B2"}
    b0 = predictions[predictions["baseline"] == "B0"]
    assert b0["predicted_final_delay_h"].eq(0.0).all()
    b1 = predictions[predictions["baseline"] == "B1"].set_index("shipment_id")
    assert b1.loc["S3", "predicted_final_delay_h"].eq(15.0).all()
    assert b1.loc["S5", "predicted_final_delay_h"] == 40.0
    b2 = predictions[predictions["baseline"] == "B2"].set_index(
        ["shipment_id", "checkpoint"]
    )
    assert b2.loc[("S3", 0), "predicted_final_delay_h"] == 7.0
    assert b2.loc[("S3", 1), "predicted_final_delay_h"] == 8.0
    assert b2.loc[("S5", 0), "predicted_final_delay_h"] == -2.0
