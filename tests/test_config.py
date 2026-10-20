from __future__ import annotations

from dataclasses import replace

import pytest

from src.config import DEFAULT_RUN_SETTINGS
from src.errors import ConfigurationError


def test_default_settings_preserve_current_run_values():
    assert DEFAULT_RUN_SETTINGS.seed == 42
    assert DEFAULT_RUN_SETTINGS.n_shipments == 300
    assert DEFAULT_RUN_SETTINGS.split_seeds == (42, 43, 44, 45, 46)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("seed", -1, "non-negative"),
        ("n_shipments", 0, "at least one"),
        ("missing_rate", 1.1, "between 0 and 1"),
        ("delay_threshold_h", -1.0, "non-negative"),
        ("test_frac", 1.0, "test fraction"),
        ("cv_folds", 1, "at least two CV folds"),
        ("n_split_seeds", 0, "non-empty and unique"),
    ],
)
def test_invalid_run_settings_are_rejected(field, value, message):
    with pytest.raises(ConfigurationError, match=message):
        replace(DEFAULT_RUN_SETTINGS, **{field: value})


def test_configuration_errors_remain_value_errors():
    assert issubclass(ConfigurationError, ValueError)
