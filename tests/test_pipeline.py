from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src import config, generate
from src.errors import ArtifactError
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
        model_q_predictions=tmp_path / "outputs" / "predictions" / "model_q.csv",
        model_q_metrics=tmp_path / "outputs" / "metrics" / "model_q_coverage.csv",
        model_c_predictions=tmp_path / "outputs" / "predictions" / "model_c.csv",
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
    ]


def test_run_all_writes_expected_artifacts_in_temporary_paths(tmp_path):
    paths = _temporary_paths(tmp_path)

    EtaPipeline(paths=paths).run_all()

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
            paths.model_q_predictions,
            paths.model_q_metrics,
            paths.model_c_predictions,
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
    classification = pd.read_csv(paths.model_c_predictions)
    probabilities = classification["predicted_delay_probability"]
    assert classification["actual_is_delayed"].isin((0, 1)).all()
    assert np.isfinite(probabilities).all()
    assert probabilities.between(0, 1).all()


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
