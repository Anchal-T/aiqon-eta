from __future__ import annotations

import pandas as pd
import pytest

from src import features
from src.errors import ContractError
from src.models import ModelR


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
