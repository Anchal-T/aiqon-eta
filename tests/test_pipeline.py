from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config, generate
from src.errors import ArtifactError, ConfigurationError
from src.pipeline import ArtifactPaths, EtaPipeline, main


def _temporary_paths(tmp_path: Path) -> ArtifactPaths:
    return ArtifactPaths(
        raw_shipments=tmp_path / "raw" / "shipments.csv",
        truth=tmp_path / "raw" / "truth.csv",
        observed=tmp_path / "interim" / "observed.csv",
        hidden_mask=tmp_path / "interim" / "hidden_mask.csv",
        delays=tmp_path / "interim" / "delays.csv",
        stacked=tmp_path / "processed" / "stacked.parquet",
        splits=tmp_path / "processed" / "splits.csv",
        baseline_predictions=tmp_path / "processed" / "baseline_predictions.csv",
        model_r_predictions=tmp_path / "outputs" / "predictions" / "model_r.csv",
        model_r_metrics=tmp_path / "outputs" / "metrics" / "model_r_vs_b2.csv",
        model_r_group_metrics=tmp_path / "outputs" / "metrics" / "model_r_by_group.csv",
        model_q_predictions=tmp_path / "outputs" / "predictions" / "model_q.csv",
        model_q_metrics=tmp_path / "outputs" / "metrics" / "model_q_coverage.csv",
        model_q_group_metrics=tmp_path / "outputs" / "metrics" / "model_q_by_group.csv",
        model_c_predictions=tmp_path / "outputs" / "predictions" / "model_c.csv",
        model_c_reliability=tmp_path
        / "outputs"
        / "metrics"
        / "model_c_reliability.csv",
        model_c_reliability_plot=tmp_path
        / "outputs"
        / "metrics"
        / "model_c_reliability.png",
        model_c_classification_metrics=tmp_path
        / "outputs"
        / "metrics"
        / "model_c_classification_metrics.csv",
        model_c_threshold_sweep=tmp_path
        / "outputs"
        / "metrics"
        / "model_c_threshold_sweep.csv",
        model_c_reliability_by_group=tmp_path
        / "outputs"
        / "metrics"
        / "model_c_reliability_by_group.csv",
        model_c_classification_metrics_by_group=tmp_path
        / "outputs"
        / "metrics"
        / "model_c_classification_metrics_by_group.csv",
        model_c_threshold_sweep_by_group=tmp_path
        / "outputs"
        / "metrics"
        / "model_c_threshold_sweep_by_group.csv",
        model_r_missingness_sweep=tmp_path
        / "outputs"
        / "metrics"
        / "model_r_missingness_sweep.csv",
        model_r_missingness_sweep_plot=tmp_path
        / "outputs"
        / "metrics"
        / "model_r_missingness_sweep.png",
        model_r_explanations=tmp_path
        / "outputs"
        / "predictions"
        / "model_r_explanations.csv",
        model_r_explanation_recovery=tmp_path
        / "outputs"
        / "metrics"
        / "model_r_explanation_recovery.csv",
        model_r_shap_summary=tmp_path
        / "outputs"
        / "metrics"
        / "model_r_shap_summary.png",
        model_r_slack_dependence=tmp_path
        / "outputs"
        / "metrics"
        / "model_r_slack_dependence.png",
        model_r_slot_dependence=tmp_path
        / "outputs"
        / "metrics"
        / "model_r_slot_dependence.png",
        model_r_explanation_waterfall=tmp_path
        / "outputs"
        / "metrics"
        / "model_r_explanation_waterfall.png",
    )


def test_run_to_evaluation_executes_each_stage_once(monkeypatch):
    pipeline = EtaPipeline()
    calls = []
    monkeypatch.setattr(pipeline, "build_data", lambda: calls.append("data"))
    monkeypatch.setattr(pipeline, "build_features", lambda: calls.append("features"))
    monkeypatch.setattr(pipeline, "build_splits", lambda: calls.append("splits"))
    monkeypatch.setattr(pipeline, "build_baselines", lambda: calls.append("baselines"))
    monkeypatch.setattr(pipeline, "build_model_r", lambda: calls.append("model_r"))
    monkeypatch.setattr(
        pipeline, "build_evaluation", lambda: calls.append("evaluation")
    )

    pipeline.run_to("evaluation")

    assert calls == ["data", "features", "splits", "baselines", "model_r", "evaluation"]


def test_model_q_target_skips_baseline_and_point_model(monkeypatch):
    pipeline = EtaPipeline()
    calls = []
    monkeypatch.setattr(pipeline, "build_data", lambda: calls.append("data"))
    monkeypatch.setattr(pipeline, "build_features", lambda: calls.append("features"))
    monkeypatch.setattr(pipeline, "build_splits", lambda: calls.append("splits"))
    monkeypatch.setattr(pipeline, "build_baselines", lambda: calls.append("baselines"))
    monkeypatch.setattr(pipeline, "build_model_r", lambda: calls.append("model_r"))
    monkeypatch.setattr(
        pipeline, "build_evaluation", lambda: calls.append("evaluation")
    )
    monkeypatch.setattr(pipeline, "build_model_q", lambda: calls.append("model_q"))

    pipeline.run_to("model_q")

    assert calls == ["data", "features", "splits", "model_q"]


def test_model_c_target_skips_baseline_and_regression_models(monkeypatch):
    pipeline = EtaPipeline()
    calls = []
    for stage in ("data", "features", "splits", "model_c"):
        monkeypatch.setattr(
            pipeline,
            f"build_{stage}",
            lambda stage=stage: calls.append(stage),
        )

    pipeline.run_to("model_c")

    assert calls == ["data", "features", "splits", "model_c"]


def test_model_c_evaluation_target_builds_only_its_prerequisites(monkeypatch):
    pipeline = EtaPipeline()
    calls = []
    for stage in ("data", "features", "splits", "model_c", "model_c_evaluation"):
        method = (
            "build_model_c_evaluation"
            if stage == "model_c_evaluation"
            else f"build_{stage}"
        )
        monkeypatch.setattr(pipeline, method, lambda stage=stage: calls.append(stage))

    pipeline.run_to("classification_evaluation")

    assert calls == ["data", "features", "splits", "model_c", "model_c_evaluation"]


def test_run_all_executes_shared_prerequisites_once(monkeypatch):
    pipeline = EtaPipeline()
    calls = []
    for stage in (
        "data",
        "features",
        "splits",
        "baselines",
        "model_r",
        "evaluation",
        "model_q",
        "interval_evaluation",
        "model_c",
        "model_c_evaluation",
        "missingness_sweep",
        "explanations",
    ):
        monkeypatch.setattr(
            pipeline,
            f"build_{stage}",
            lambda stage=stage: calls.append(stage),
        )

    pipeline.run_all()

    assert calls == [
        "data",
        "features",
        "splits",
        "baselines",
        "model_r",
        "evaluation",
        "model_q",
        "interval_evaluation",
        "model_c",
        "model_c_evaluation",
        "missingness_sweep",
        "explanations",
    ]


def test_explanations_target_builds_model_r_once(monkeypatch):
    pipeline = EtaPipeline()
    calls = []
    for stage in ("data", "features", "splits", "model_r", "explanations"):
        method = "build_explanations" if stage == "explanations" else f"build_{stage}"
        monkeypatch.setattr(pipeline, method, lambda stage=stage: calls.append(stage))

    pipeline.run_to("explanations")

    assert calls == ["data", "features", "splits", "model_r", "explanations"]


def test_missingness_sweep_runs_fixed_rates_with_same_seeds_and_writes_outputs(
    tmp_path, monkeypatch
):
    paths = _temporary_paths(tmp_path)
    settings = replace(config.DEFAULT_RUN_SETTINGS, seed=17, n_split_seeds=3)
    pipeline = EtaPipeline(settings, paths)
    calls = []
    plotted = []

    def fake_run_to(run, target):
        assert target == "evaluation"
        calls.append((run.settings, run.paths))
        run.paths.model_r_metrics.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {
                "checkpoint": [1],
                "model_mae_mean_h": [5.0 + run.settings.missing_rate],
                "model_mae_std_h": [0.5],
                "b2_mae_mean_h": [6.0],
            }
        ).to_csv(run.paths.model_r_metrics, index=False)

    monkeypatch.setattr(EtaPipeline, "run_to", fake_run_to)
    monkeypatch.setattr(
        "src.pipeline.evaluate.plot_model_r_missingness_sweep",
        lambda metrics, path: plotted.append((metrics.copy(), path)),
    )

    pipeline.build_missingness_sweep()

    rates = [0.0, 0.1, 0.2, 0.3]
    assert [run_settings.missing_rate for run_settings, _ in calls] == rates
    assert all(run_settings.seed == settings.seed for run_settings, _ in calls)
    assert all(
        run_settings.split_seeds == settings.split_seeds for run_settings, _ in calls
    )
    assert all(not run_paths.model_r_metrics.exists() for _, run_paths in calls)
    result = pd.read_csv(paths.model_r_missingness_sweep)
    assert result["missing_rate"].tolist() == rates
    assert result["b2_mae_mean_h"].tolist() == [6.0] * len(rates)
    assert plotted[0][1] == paths.model_r_missingness_sweep_plot
    assert plotted[0][0]["missing_rate"].tolist() == rates


def test_missingness_sweep_cli_uses_out_directory(monkeypatch, capsys):
    output_dir = Path("custom/output")
    calls = []
    monkeypatch.setattr(
        EtaPipeline,
        "run_to",
        lambda self, target: calls.append(
            (
                target,
                self.paths.model_r_missingness_sweep,
                self.paths.model_r_missingness_sweep_plot,
            )
        ),
    )

    assert main(["eval-missingness", "--out", str(output_dir)]) == 0

    assert calls == [
        (
            "missingness_sweep",
            output_dir / "model_r_missingness_sweep.csv",
            output_dir / "model_r_missingness_sweep.png",
        )
    ]
    assert "eval-missingness pipeline completed" in capsys.readouterr().out


def test_missingness_sweep_cli_rejects_rate_override(capsys):
    assert main(["eval-missingness", "--rate", "0.2"]) == 1
    assert "fixed" in capsys.readouterr().err


def test_explain_cli_builds_artifacts_and_selected_waterfall(monkeypatch):
    output_dir = Path("custom/output")
    calls = []
    monkeypatch.setattr(
        EtaPipeline,
        "run_to",
        lambda self, target: calls.append(("run", target)),
    )
    monkeypatch.setattr(
        EtaPipeline,
        "write_model_r_waterfall",
        lambda self, shipment_id, split_seed, checkpoint: calls.append(
            (
                "waterfall",
                shipment_id,
                split_seed,
                checkpoint,
                self.paths.model_r_explanation_waterfall,
            )
        ),
    )

    assert (
        main(
            [
                "explain",
                "--shipment-id",
                "S000",
                "--split-seed",
                "42",
                "--checkpoint",
                "4",
                "--out",
                str(output_dir),
            ]
        )
        == 0
    )

    assert calls == [
        ("run", "explanations"),
        (
            "waterfall",
            "S000",
            42,
            4,
            output_dir / "model_r_explanation_waterfall.png",
        ),
    ]


def test_explain_cli_requires_all_waterfall_keys(capsys):
    assert main(["explain", "--shipment-id", "S000"]) == 1
    assert "provided together" in capsys.readouterr().err


def test_run_all_writes_expected_artifacts_in_temporary_paths(tmp_path):
    paths = _temporary_paths(tmp_path)

    pipeline = EtaPipeline(paths=paths)
    pipeline.run_all()

    assert all(
        path.is_file()
        for path in (
            paths.raw_shipments,
            paths.truth,
            paths.observed,
            paths.hidden_mask,
            paths.delays,
            paths.stacked,
            paths.splits,
            paths.baseline_predictions,
            paths.model_r_predictions,
            paths.model_r_metrics,
            paths.model_r_group_metrics,
            paths.model_q_predictions,
            paths.model_q_metrics,
            paths.model_q_group_metrics,
            paths.model_c_predictions,
            paths.model_c_reliability,
            paths.model_c_reliability_plot,
            paths.model_c_classification_metrics,
            paths.model_c_threshold_sweep,
            paths.model_c_reliability_by_group,
            paths.model_c_classification_metrics_by_group,
            paths.model_c_threshold_sweep_by_group,
            paths.model_r_missingness_sweep,
            paths.model_r_missingness_sweep_plot,
            paths.model_r_explanations,
            paths.model_r_explanation_recovery,
            paths.model_r_shap_summary,
            paths.model_r_slack_dependence,
            paths.model_r_slot_dependence,
        )
    )
    assert (
        len(pd.read_csv(paths.raw_shipments)) == config.DEFAULT_RUN_SETTINGS.n_shipments
    )
    assert len(pd.read_csv(paths.splits)) == (
        config.DEFAULT_RUN_SETTINGS.n_shipments
        * config.DEFAULT_RUN_SETTINGS.n_split_seeds
    )
    assert set(pd.read_csv(paths.baseline_predictions)["baseline"]) == {
        "B0",
        "B1",
        "B2",
    }
    predictions = pd.read_csv(paths.model_r_predictions)
    np.testing.assert_allclose(
        predictions["predicted_final_delay_h"],
        predictions["delay_so_far"] + predictions["predicted_remaining_delay_h"],
        rtol=1e-12,
        atol=1e-12,
    )
    metrics = pd.read_csv(paths.model_r_metrics)
    assert set(metrics["checkpoint"]) == set(range(7))
    assert metrics["model_rmse_mean_h"].ge(0).all()
    assert metrics["b2_p90_ae_mean_h"].ge(0).all()
    sweep = pd.read_csv(paths.model_r_missingness_sweep)
    assert set(sweep["missing_rate"]) == {0.0, 0.1, 0.2, 0.3}
    assert set(sweep["checkpoint"]) == set(range(7))
    assert sweep["b2_mae_mean_h"].ge(0).all()
    assert paths.model_r_missingness_sweep_plot.read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )
    explanations = pd.read_csv(paths.model_r_explanations)
    selected = explanations.iloc[0]
    pipeline.write_model_r_waterfall(
        str(selected["shipment_id"]),
        int(selected["split_seed"]),
        int(selected["checkpoint"]),
    )
    assert paths.model_r_explanation_waterfall.read_bytes().startswith(
        b"\x89PNG\r\n\x1a\n"
    )
    shap_columns = [
        column for column in explanations if column.startswith("shap_group_")
    ]
    np.testing.assert_allclose(
        explanations["base_value"] + explanations[shap_columns].sum(axis=1),
        explanations["predicted_remaining_delay_h"],
        rtol=1e-5,
        atol=1e-5,
    )
    recovery = pd.read_csv(paths.model_r_explanation_recovery)
    assert set(recovery["check"]) == {
        "cliff_recovery_c2",
        "slot_recovery_c4",
        "storm_leg1_fact_top",
        "storm_leg2_fact_top",
    }
    for plot_path in (
        paths.model_r_shap_summary,
        paths.model_r_slack_dependence,
        paths.model_r_slot_dependence,
    ):
        assert plot_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    grouped_metrics = pd.read_csv(paths.model_r_group_metrics)
    assert set(grouped_metrics["checkpoint"]) == set(range(7))
    assert set(grouped_metrics["group_type"]) == {
        "journey",
        "connection_type",
        "missed_connection",
        "hidden_stage",
    }
    intervals = pd.read_csv(paths.model_q_predictions)
    assert (
        intervals["predicted_final_lower_h"] <= intervals["predicted_final_upper_h"]
    ).all()
    np.testing.assert_allclose(
        intervals["predicted_final_lower_h"],
        intervals["delay_so_far"] + intervals["predicted_remaining_lower_h"],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        intervals["predicted_final_upper_h"],
        intervals["delay_so_far"] + intervals["predicted_remaining_upper_h"],
        rtol=1e-12,
        atol=1e-12,
    )
    interval_metrics = pd.read_csv(paths.model_q_metrics)
    assert set(interval_metrics["checkpoint"]) == set(range(7))
    assert (interval_metrics["mean_width_h"] >= 0).all()
    grouped_intervals = pd.read_csv(paths.model_q_group_metrics)
    assert set(grouped_intervals["checkpoint"]) == set(range(7))
    assert set(grouped_intervals["group_type"]) == {
        "journey",
        "connection_type",
        "missed_connection",
        "hidden_stage",
    }
    classification = pd.read_csv(paths.model_c_predictions)
    probabilities = classification["predicted_delay_probability"]
    assert classification["actual_is_delayed"].isin((0, 1)).all()
    assert np.isfinite(probabilities).all()
    assert probabilities.between(0, 1).all()
    reliability = pd.read_csv(paths.model_c_reliability)
    assert set(reliability["checkpoint"]) == set(range(7))
    assert reliability["observed_delay_rate"].between(0, 1).all()
    assert reliability["n_predictions"].gt(0).all()
    assert paths.model_c_reliability_plot.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    classification_metrics = pd.read_csv(paths.model_c_classification_metrics)
    assert set(classification_metrics["checkpoint"]) == set(range(7))
    assert classification_metrics["brier_score_mean"].between(0, 1).all()
    assert classification_metrics["f1_at_0_5_mean"].between(0, 1).all()
    grouped_classification = pd.read_csv(paths.model_c_classification_metrics_by_group)
    assert set(grouped_classification["group_type"]) == {
        "journey",
        "connection_type",
        "missed_connection",
        "hidden_stage",
    }
    grouped_reliability = pd.read_csv(paths.model_c_reliability_by_group)
    assert grouped_reliability["observed_delay_rate"].between(0, 1).all()
    grouped_threshold_sweep = pd.read_csv(paths.model_c_threshold_sweep_by_group)
    assert grouped_threshold_sweep["threshold"].nunique() == 21
    threshold_sweep = pd.read_csv(paths.model_c_threshold_sweep)
    assert set(threshold_sweep["checkpoint"]) == set(range(7))
    assert threshold_sweep["threshold"].nunique() == 21
    assert threshold_sweep["f1_mean"].between(0, 1).all()


def test_custom_evaluation_predictions_require_matching_metadata():
    pipeline = EtaPipeline()
    with pytest.raises(ConfigurationError, match="explicit.*metadata"):
        pipeline.build_evaluation(model_path=Path("custom/model_r.csv"))
    with pytest.raises(ConfigurationError, match="explicit.*metadata"):
        pipeline.build_interval_evaluation(model_q_path=Path("custom/model_q.csv"))
    with pytest.raises(ConfigurationError, match="explicit.*metadata"):
        pipeline.build_model_c_evaluation(model_c_path=Path("custom/model_c.csv"))


def test_eval_cli_passes_explicit_predictions_and_metadata(monkeypatch):
    paths = [
        Path("custom/model_r.csv"),
        Path("custom/baselines.csv"),
        Path("custom/shipments.csv"),
        Path("custom/truth.csv"),
        Path("custom/hidden_mask.csv"),
    ]
    calls = []
    monkeypatch.setattr(
        EtaPipeline,
        "build_evaluation",
        lambda self, *inputs: calls.append((inputs, self.paths.model_r_group_metrics)),
    )
    monkeypatch.setattr(
        EtaPipeline,
        "run_to",
        lambda self, target: pytest.fail("explicit inputs must skip prerequisites"),
    )

    assert (
        main(
            [
                "eval",
                "--model-r",
                str(paths[0]),
                "--baselines",
                str(paths[1]),
                "--shipments",
                str(paths[2]),
                "--truth",
                str(paths[3]),
                "--hidden-mask",
                str(paths[4]),
                "--out",
                "custom/output",
            ]
        )
        == 0
    )
    assert calls == [(tuple(paths), Path("custom/output/model_r_by_group.csv"))]


def test_artifact_read_error_includes_path_and_cause(tmp_path):
    missing = tmp_path / "missing.parquet"
    pipeline = EtaPipeline(paths=replace(_temporary_paths(tmp_path), stacked=missing))

    with pytest.raises(ArtifactError, match=str(missing)) as error:
        pipeline.build_splits()

    assert isinstance(error.value.__cause__, FileNotFoundError)


def test_build_data_uses_explicit_raw_artifact_without_generating(
    tmp_path, monkeypatch
):
    settings = replace(config.DEFAULT_RUN_SETTINGS, n_shipments=20)
    shipments, _ = generate.generate(settings)
    raw_path = tmp_path / "input" / "shipments.csv"
    raw_path.parent.mkdir()
    shipments.to_csv(raw_path, index=False)
    paths = _temporary_paths(tmp_path)
    pipeline = EtaPipeline(settings, paths)
    monkeypatch.setattr(
        "src.pipeline.generate.generate",
        lambda *_: pytest.fail("explicit raw input must skip generation"),
    )

    pipeline.build_data(raw_path)

    assert not paths.raw_shipments.exists()
    assert paths.observed.is_file()
    assert paths.delays.is_file()


def test_data_cli_starts_from_explicit_raw_input(monkeypatch, capsys):
    raw = Path("existing/shipments.csv")
    calls = []
    monkeypatch.setattr(
        EtaPipeline,
        "build_data",
        lambda self, raw_path=None: calls.append(raw_path),
    )
    monkeypatch.setattr(
        EtaPipeline,
        "run_to",
        lambda self, target: pytest.fail("explicit raw input must skip generation"),
    )

    assert main(["data", "--raw", str(raw)]) == 0
    assert calls == [raw]
    assert "data pipeline completed" in capsys.readouterr().out


def test_features_cli_starts_from_explicit_inputs(monkeypatch, capsys):
    observed = Path("existing/observed.csv")
    delays = Path("existing/delays.csv")
    calls = []
    monkeypatch.setattr(
        EtaPipeline,
        "build_data",
        lambda self: pytest.fail("explicit inputs must skip data generation"),
    )
    monkeypatch.setattr(
        EtaPipeline,
        "build_features",
        lambda self, observed_path=None, delays_path=None: calls.append(
            (observed_path, delays_path)
        ),
    )

    assert main(["features", "--observed", str(observed), "--delays", str(delays)]) == 0
    assert calls == [(observed, delays)]
    assert "features pipeline completed" in capsys.readouterr().out


def test_model_c_cli_starts_from_explicit_inputs(monkeypatch):
    stacked = Path("existing/stacked.parquet")
    split_manifest = Path("existing/splits.csv")
    output_dir = Path("custom/output")
    calls = []
    monkeypatch.setattr(
        EtaPipeline,
        "build_model_c",
        lambda self, stacked_path=None, splits_path=None: calls.append(
            (stacked_path, splits_path, self.paths.model_c_predictions)
        ),
    )
    monkeypatch.setattr(
        EtaPipeline,
        "run_to",
        lambda self, target: pytest.fail("explicit inputs must skip prerequisites"),
    )

    assert (
        main(
            [
                "model-c",
                "--stacked",
                str(stacked),
                "--splits",
                str(split_manifest),
                "--out",
                str(output_dir),
            ]
        )
        == 0
    )
    assert calls == [(stacked, split_manifest, output_dir / "model_c.csv")]


def test_model_c_evaluation_cli_starts_from_explicit_predictions(monkeypatch):
    predictions = Path("existing/model_c.csv")
    metadata_paths = (
        Path("existing/shipments.csv"),
        Path("existing/truth.csv"),
        Path("existing/hidden_mask.csv"),
    )
    output_dir = Path("custom/output")
    calls = []
    monkeypatch.setattr(
        EtaPipeline,
        "build_model_c_evaluation",
        lambda self, *inputs: calls.append(
            (
                inputs,
                self.paths.model_c_reliability,
                self.paths.model_c_reliability_plot,
                self.paths.model_c_classification_metrics,
                self.paths.model_c_threshold_sweep,
                self.paths.model_c_reliability_by_group,
                self.paths.model_c_classification_metrics_by_group,
                self.paths.model_c_threshold_sweep_by_group,
            )
        ),
    )
    monkeypatch.setattr(
        EtaPipeline,
        "run_to",
        lambda self, target: pytest.fail("explicit inputs must skip prerequisites"),
    )

    assert (
        main(
            [
                "eval-c",
                "--model-c",
                str(predictions),
                "--shipments",
                str(metadata_paths[0]),
                "--truth",
                str(metadata_paths[1]),
                "--hidden-mask",
                str(metadata_paths[2]),
                "--out",
                str(output_dir),
            ]
        )
        == 0
    )
    assert calls == [
        (
            (predictions, *metadata_paths),
            output_dir / "model_c_reliability.csv",
            output_dir / "model_c_reliability.png",
            output_dir / "model_c_classification_metrics.csv",
            output_dir / "model_c_threshold_sweep.csv",
            output_dir / "model_c_reliability_by_group.csv",
            output_dir / "model_c_classification_metrics_by_group.csv",
            output_dir / "model_c_threshold_sweep_by_group.csv",
        )
    ]


def test_model_q_evaluation_cli_passes_matching_metadata(monkeypatch):
    inputs = (
        Path("existing/model_q.csv"),
        Path("existing/shipments.csv"),
        Path("existing/truth.csv"),
        Path("existing/hidden_mask.csv"),
    )
    output_dir = Path("custom/output")
    calls = []
    monkeypatch.setattr(
        EtaPipeline,
        "build_interval_evaluation",
        lambda self, *paths: calls.append(
            (paths, self.paths.model_q_metrics, self.paths.model_q_group_metrics)
        ),
    )
    monkeypatch.setattr(
        EtaPipeline,
        "run_to",
        lambda self, target: pytest.fail("explicit inputs must skip prerequisites"),
    )

    assert (
        main(
            [
                "eval-q",
                "--model-q",
                str(inputs[0]),
                "--shipments",
                str(inputs[1]),
                "--truth",
                str(inputs[2]),
                "--hidden-mask",
                str(inputs[3]),
                "--out",
                str(output_dir),
            ]
        )
        == 0
    )
    assert calls == [
        (
            inputs,
            output_dir / "model_q_coverage.csv",
            output_dir / "model_q_by_group.csv",
        )
    ]


def test_splits_cli_starts_from_explicit_stacked_input(monkeypatch):
    stacked = Path("existing/stacked.parquet")
    calls = []
    monkeypatch.setattr(
        EtaPipeline,
        "build_data",
        lambda self: pytest.fail("explicit input must skip data generation"),
    )
    monkeypatch.setattr(
        EtaPipeline,
        "build_splits",
        lambda self, stacked_path=None: calls.append(stacked_path),
    )

    assert main(["splits", "--stacked", str(stacked)]) == 0
    assert calls == [stacked]
