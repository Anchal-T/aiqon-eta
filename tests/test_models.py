from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.model_c import ModelC
from src.model_r import ModelR

from src import features
from src.errors import ContractError


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


def test_model_c_returns_probabilities_for_outer_test_checkpoints_only():
    stacked_rows = []
    assignments = []
    expected_test_ids = {f"S{number:03d}" for number in range(8, 12)}

    for number in range(12):
        shipment_id = f"S{number:03d}"
        outer_split = "train" if number < 8 else "test"
        fold = number // 4 if outer_split == "train" else None
        label = number % 2
        assignments.append(
            {
                "split_seed": 17,
                "shipment_id": shipment_id,
                "outer_split": outer_split,
                "fold": fold,
            }
        )
        for checkpoint in range(2):
            row: dict[str, object] = {
                column: 0.0 for column in features.FEATURE_COLUMNS
            }
            row.update(
                {
                    "shipment_id": shipment_id,
                    "checkpoint": checkpoint,
                    "is_delayed": label,
                }
            )
            stacked_rows.append(row)

    predictions = (
        ModelC()
        .fit(pd.DataFrame(stacked_rows), pd.DataFrame(assignments))
        .predict_test()
    )

    assert len(predictions) == len(expected_test_ids) * 2
    assert set(predictions["shipment_id"]) == expected_test_ids
    assert set(predictions["split_seed"]) == {17}
    probabilities = predictions["predicted_delay_probability"].to_numpy()
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


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
