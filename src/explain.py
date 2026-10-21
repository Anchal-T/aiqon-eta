"""Explain held-out Model R predictions with observed facts and TreeSHAP."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src import config, features
from src.errors import ArtifactError, ContractError, ModelError
from src.model_r import ModelR

FEATURE_GROUPS = {
    "departure": ("delay_departure",),
    "leg1": (
        "planned_leg1_h",
        "weather_leg1",
        "delay_hub_arrival",
        "added_hub_arrival",
    ),
    "hub": (
        "hub_buffer_h",
        "nominal_slack_h",
        "congestion_hub",
        "slack_remaining",
        "delay_hub_departure",
        "added_hub_departure",
    ),
    "leg2": ("planned_leg2_h", "weather_leg2"),
    "port_arrival_slot": (
        "delay_port_arrival",
        "added_port_arrival",
        "arrival_dow",
        "slot_margin",
    ),
    "dwell_congestion": (
        "port_dwell_base_h",
        "port_base_congestion",
        "congestion_port",
        "cong_x_dwell",
        "delay_port_done",
        "added_port_done",
        "port_done_dow",
        "port_done_is_weekend",
        "port_done_missing",
    ),
    "customs": (
        "port_customs_base_h",
        "cong_x_customs",
        "delay_customs",
        "added_customs",
        "customs_missing",
    ),
    "route_plan": (
        "checkpoint",
        "journey_id",
        "origin",
        "hub",
        "destination",
        "is_connecting",
        "n_legs",
        "planned_total_h",
        "sched_dep_dow",
        "sched_arrival_dow",
        "sched_dep_doy",
        "delay_so_far",
        "n_observed_stages",
    ),
}

_GROUP_LABELS = {
    "departure": "Departure",
    "leg1": "Leg 1",
    "hub": "Hub",
    "leg2": "Leg 2",
    "port_arrival_slot": "Port arrival and slot",
    "dwell_congestion": "Dwell and congestion",
    "customs": "Customs",
    "route_plan": "Route and plan",
}
_GROUP_COLUMNS = {group: f"shap_group_{group}_h" for group in FEATURE_GROUPS}
_FACT_STAGES = config.STAGE_ORDER[:-1]
_MERGED_COLUMNS = tuple(f"merged_{stage}" for stage in _FACT_STAGES)
_TRUTH_COLUMNS = (
    "missed_connection",
    "missed_slot",
    "storm_leg1",
    "storm_leg2",
)


def _validated_table(
    table: pd.DataFrame, required: tuple[str, ...], name: str
) -> pd.DataFrame:
    missing = sorted(set(required) - set(table.columns))
    if missing:
        raise ContractError(f"{name} is missing columns: {missing}")
    if table.empty or table["shipment_id"].isna().any():
        raise ContractError(f"{name} must contain shipment IDs")
    result = table[list(required)].copy()
    result["shipment_id"] = result["shipment_id"].astype("string")
    if result["shipment_id"].duplicated().any():
        raise ContractError(f"{name} shipment IDs must be unique")
    return result


def _validated_delays(delays: pd.DataFrame) -> pd.DataFrame:
    table = _validated_table(delays, ("shipment_id", *_MERGED_COLUMNS), "delay facts")
    for column in _MERGED_COLUMNS:
        if table[column].isna().any() or not table[column].isin((True, False)).all():
            raise ContractError(f"delay fact {column} must contain boolean values")
        table[column] = table[column].astype(bool)
    return table.set_index("shipment_id")


def _validated_truth(truth: pd.DataFrame) -> pd.DataFrame:
    table = _validated_table(truth, ("shipment_id", *_TRUTH_COLUMNS), "truth table")
    for column in _TRUTH_COLUMNS:
        if table[column].isna().any() or not table[column].isin((True, False)).all():
            raise ContractError(f"truth column {column} must contain boolean values")
        table[column] = table[column].astype(bool)
    return table


def _group_contributions(values: np.ndarray) -> dict[str, np.ndarray]:
    feature_index = {name: index for index, name in enumerate(features.FEATURE_COLUMNS)}
    assigned = [name for group in FEATURE_GROUPS.values() for name in group]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(feature_index):
        raise ModelError("Model R feature groups must cover each feature exactly once")
    return {
        group: values[:, [feature_index[name] for name in columns]].sum(axis=1)
        for group, columns in FEATURE_GROUPS.items()
    }


def _fact_columns(row: pd.Series, delay_row: pd.Series) -> dict[str, object]:
    facts: dict[str, object] = {}
    for stage in _FACT_STAGES:
        source = "delay_departure" if stage == "departure" else f"added_{stage}"
        value = row[source]
        facts[f"fact_added_{stage}_h"] = value
        facts[f"fact_merged_{stage}"] = (
            bool(delay_row[f"merged_{stage}"]) if pd.notna(value) else np.nan
        )
    return facts


def _truth_recovery_metrics(
    explanations: pd.DataFrame, truth: pd.DataFrame
) -> pd.DataFrame:
    joined = explanations.merge(
        truth, on="shipment_id", how="left", validate="many_to_one"
    )
    if joined[list(_TRUTH_COLUMNS)].isna().any().any():
        raise ContractError("truth table is missing explained shipments")

    cliff = joined.loc[(joined["checkpoint"] == 2) & (joined["is_connecting"] == 1)]
    cliff_labels = cliff["missed_connection"].astype(int)
    cliff_auc = (
        roc_auc_score(cliff_labels, cliff["shap_group_hub_h"].abs())
        if cliff_labels.nunique() == 2
        else np.nan
    )

    slot = joined.loc[joined["checkpoint"] == 4]
    missed_slot_magnitude = slot.loc[
        slot["missed_slot"], "shap_group_port_arrival_slot_h"
    ].abs()
    kept_slot_magnitude = slot.loc[
        ~slot["missed_slot"], "shap_group_port_arrival_slot_h"
    ].abs()
    slot_difference = (
        missed_slot_magnitude.mean() - kept_slot_magnitude.mean()
        if not missed_slot_magnitude.empty and not kept_slot_magnitude.empty
        else np.nan
    )

    rows = [
        _recovery_row(
            "cliff_recovery_c2",
            "ROC AUC of absolute hub SHAP for missed connection",
            cliff_auc,
            0.8,
            len(cliff),
            int(cliff_labels.sum()),
            int((cliff_labels == 0).sum()),
        ),
        _recovery_row(
            "slot_recovery_c4",
            "mean absolute slot-group SHAP difference (missed - kept)",
            slot_difference,
            0.0,
            len(slot),
            int(slot["missed_slot"].sum()),
            int((~slot["missed_slot"]).sum()),
        ),
    ]
    for storm, check in (
        ("storm_leg1", "storm_leg1_fact_top"),
        ("storm_leg2", "storm_leg2_fact_top"),
    ):
        fraction, count = _storm_fact_top_fraction(joined, storm)
        rows.append(
            _recovery_row(
                check, "fraction with storm-leg fact ranked top", fraction, 0.5, count
            )
        )
    return pd.DataFrame.from_records(rows)


def _recovery_row(
    check: str,
    measure: str,
    value: float,
    threshold: float,
    n_samples: int,
    n_positive: int = 0,
    n_negative: int = 0,
) -> dict[str, object]:
    finite = np.isfinite(value)
    return {
        "check": check,
        "measure": measure,
        "value": value,
        "threshold": threshold,
        "passed": bool(finite and value > threshold),
        "n_samples": n_samples,
        "n_positive": n_positive,
        "n_negative": n_negative,
    }


def _storm_fact_top_fraction(
    joined: pd.DataFrame, storm_column: str
) -> tuple[float, int]:
    storm_rows = joined.loc[joined[storm_column]]
    if storm_column == "storm_leg1":
        storm_rows = storm_rows.loc[
            ((storm_rows["is_connecting"] == 1) & (storm_rows["checkpoint"] == 2))
            | ((storm_rows["is_connecting"] == 0) & (storm_rows["checkpoint"] == 4))
        ]
    else:
        storm_rows = storm_rows.loc[
            (storm_rows["is_connecting"] == 1) & (storm_rows["checkpoint"] == 4)
        ]

    top_hits = []
    for _, row in storm_rows.iterrows():
        stages = (
            config.STAGE_ORDER[:-1]
            if row["is_connecting"]
            else config.DIRECT_STAGES[:-1]
        )
        facts = {
            stage: row[f"fact_added_{stage}_h"]
            for stage in stages
            if pd.notna(row[f"fact_added_{stage}_h"])
        }
        target = (
            "port_arrival"
            if storm_column == "storm_leg2" or not row["is_connecting"]
            else "hub_arrival"
        )
        if target in facts and facts:
            top_hits.append(facts[target] >= max(facts.values()))
    return (float(np.mean(top_hits)), len(top_hits)) if top_hits else (np.nan, 0)


def _tree_shap_for_seed(
    estimator: lgb.LGBMRegressor, seed: int, rows: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    matrix = rows.loc[:, features.FEATURE_COLUMNS]
    try:
        # LightGBM's pred_contrib uses TreeSHAP; the last column is its base value.
        contributions = np.asarray(
            estimator.booster_.predict(matrix, pred_contrib=True), dtype=float
        )
        remaining = np.asarray(estimator.predict(matrix), dtype=float)
    except (ValueError, lgb.basic.LightGBMError) as error:
        raise ModelError(
            f"Model R explanation failed for split seed {seed}: {error}"
        ) from error

    expected_shape = (len(rows), len(features.FEATURE_COLUMNS) + 1)
    if contributions.shape != expected_shape or not np.isfinite(contributions).all():
        raise ModelError(
            f"Model R returned invalid TreeSHAP values for split seed {seed}"
        )

    base_values = contributions[:, -1]
    feature_values = contributions[:, :-1]
    grouped = _group_contributions(feature_values)
    if not np.allclose(
        base_values + sum(grouped.values()), remaining, rtol=1e-5, atol=1e-5
    ):
        raise ModelError(
            f"Model R TreeSHAP values do not sum to predictions for split seed {seed}"
        )
    return base_values, feature_values, remaining, grouped


def build_model_r_explanations(
    model: ModelR, delays: pd.DataFrame, truth: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return grouped held-out TreeSHAP rows and synthetic-truth checks."""
    delay_rows = _validated_delays(delays)
    truth_rows = _validated_truth(truth)
    fitted_models = model.fitted_models
    test_rows = model.held_out_features
    if set(fitted_models) != set(test_rows):
        raise ModelError("Model R estimators and held-out rows have different seeds")

    records = []
    for seed, rows in test_rows.items():
        base_values, feature_values, remaining, grouped = _tree_shap_for_seed(
            fitted_models[seed], seed, rows
        )
        for position, (_, row) in enumerate(rows.iterrows()):
            shipment_id = str(row["shipment_id"])
            if shipment_id not in delay_rows.index:
                raise ContractError(f"delay facts are missing shipment {shipment_id}")
            delay_row = delay_rows.loc[shipment_id]
            fact_data = _fact_columns(row, delay_row)
            fact_total = sum(
                float(value)
                for key, value in fact_data.items()
                if key.startswith("fact_added_") and pd.notna(value)
            )
            if not np.isclose(fact_total, row["delay_so_far"], rtol=1e-9, atol=1e-9):
                raise ContractError(
                    f"visible stage facts do not sum to delay_so_far for {shipment_id}"
                )

            record: dict[str, object] = {
                "split_seed": seed,
                "shipment_id": shipment_id,
                "checkpoint": int(row["checkpoint"]),
                "is_connecting": int(row["is_connecting"]),
                "delay_so_far": float(row["delay_so_far"]),
                "slack_remaining": row["slack_remaining"],
                "slot_margin": row["slot_margin"],
                "base_value": float(base_values[position]),
                "predicted_remaining_delay_h": float(remaining[position]),
                "predicted_final_delay_h": float(
                    row["delay_so_far"] + remaining[position]
                ),
                "shap_slack_remaining_h": float(
                    feature_values[
                        position, features.FEATURE_COLUMNS.index("slack_remaining")
                    ]
                ),
                "shap_slot_margin_h": float(
                    feature_values[
                        position, features.FEATURE_COLUMNS.index("slot_margin")
                    ]
                ),
                **{
                    _GROUP_COLUMNS[group]: float(values[position])
                    for group, values in grouped.items()
                },
                **fact_data,
            }
            records.append(record)

    explanations = pd.DataFrame.from_records(records)
    truth_metrics = _truth_recovery_metrics(explanations, truth_rows)
    return explanations, truth_metrics


def _plotting():
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise ArtifactError(
            "matplotlib is required to write explanation plots"
        ) from error
    return plt


def _save_figure(figure, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150)
    except OSError as error:
        raise ArtifactError(f"cannot write explanation plot {path}: {error}") from error


def plot_model_r_shap_summary(explanations: pd.DataFrame, path: Path) -> None:
    """Plot mean absolute grouped SHAP values across held-out predictions."""
    if explanations.empty or any(
        column not in explanations for column in _GROUP_COLUMNS.values()
    ):
        raise ContractError("explanation table is empty or missing grouped SHAP values")
    values = explanations[list(_GROUP_COLUMNS.values())].abs().mean().sort_values()
    labels = [
        _GROUP_LABELS[column.removeprefix("shap_group_").removesuffix("_h")]
        for column in values.index
    ]
    plt = _plotting()
    figure, axis = plt.subplots(figsize=(8, 5))
    try:
        axis.barh(labels, values.to_numpy())
        axis.set(
            xlabel="Mean absolute SHAP contribution (hours)",
            title="Model R importance by feature group",
        )
        axis.grid(axis="x", alpha=0.25)
        figure.tight_layout()
        _save_figure(figure, path)
    finally:
        plt.close(figure)


def plot_model_r_dependence(
    explanations: pd.DataFrame, feature: str, path: Path
) -> None:
    """Plot a feature against its individual TreeSHAP contribution."""
    shap_column = {
        "slack_remaining": "shap_slack_remaining_h",
        "slot_margin": "shap_slot_margin_h",
    }.get(feature)
    if shap_column is None:
        raise ContractError(f"unsupported dependence feature: {feature}")
    if feature not in explanations or shap_column not in explanations:
        raise ContractError(f"explanation table is missing {feature} values")
    values = explanations[[feature, shap_column]].apply(pd.to_numeric, errors="coerce")
    values = values[np.isfinite(values[feature]) & np.isfinite(values[shap_column])]
    if values.empty:
        raise ContractError(f"no held-out predictions have visible {feature}")

    plt = _plotting()
    figure, axis = plt.subplots(figsize=(7, 5))
    try:
        axis.scatter(values[feature], values[shap_column], alpha=0.45, s=14)
        axis.axhline(0, color="black", linewidth=0.8)
        axis.set(
            xlabel=f"{feature} (hours)",
            ylabel="SHAP contribution (hours)",
            title=f"Model R dependence: {feature}",
        )
        axis.grid(alpha=0.25)
        figure.tight_layout()
        _save_figure(figure, path)
    finally:
        plt.close(figure)


def plot_model_r_waterfall(row: pd.Series, path: Path) -> None:
    """Plot one selected held-out prediction from observed facts to final delay."""
    required = {
        "shipment_id",
        "split_seed",
        "checkpoint",
        "delay_so_far",
        "base_value",
        "predicted_remaining_delay_h",
        "predicted_final_delay_h",
        *_GROUP_COLUMNS.values(),
        *(f"fact_added_{stage}_h" for stage in _FACT_STAGES),
        *(f"fact_merged_{stage}" for stage in _FACT_STAGES),
    }
    missing = sorted(required - set(row.index))
    if missing:
        raise ContractError(f"explanation row is missing columns: {missing}")

    items = []
    for stage in _FACT_STAGES:
        value = row[f"fact_added_{stage}_h"]
        if pd.notna(value):
            merged = row[f"fact_merged_{stage}"]
            label = stage.replace("_", " ") + (
                " (merged)" if pd.notna(merged) and bool(merged) else ""
            )
            items.append((label, float(value)))
    facts_total = sum(value for label, value in items)
    if not np.isclose(facts_total, row["delay_so_far"], rtol=1e-9, atol=1e-9):
        raise ContractError("observed facts do not sum to delay_so_far")

    items.append(("Model baseline", float(row["base_value"])))
    items.extend(
        (_GROUP_LABELS[group], float(row[column]))
        for group, column in _GROUP_COLUMNS.items()
    )
    if not np.isclose(
        sum(value for _, value in items),
        row["predicted_final_delay_h"],
        rtol=1e-5,
        atol=1e-5,
    ):
        raise ContractError("waterfall values do not sum to predicted final delay")

    plt = _plotting()
    figure, axis = plt.subplots(figsize=(max(10, len(items) * 0.65), 5))
    try:
        current = 0.0
        for index, (label, value) in enumerate(items):
            next_value = current + value
            axis.bar(
                index,
                abs(value),
                bottom=min(current, next_value),
                color="tab:blue" if value >= 0 else "tab:orange",
            )
            current = next_value
        total_index = len(items)
        axis.bar(total_index, abs(current), bottom=min(0.0, current), color="tab:green")
        axis.set_xticks(
            range(total_index + 1),
            [label for label, _ in items] + ["Predicted final"],
            rotation=35,
            ha="right",
        )
        axis.set_ylabel("Delay contribution (hours)")
        axis.set_title(
            f"{row['shipment_id']} · split {int(row['split_seed'])} · C{int(row['checkpoint'])}"
        )
        axis.axhline(0, color="black", linewidth=0.8)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        _save_figure(figure, path)
    finally:
        plt.close(figure)
