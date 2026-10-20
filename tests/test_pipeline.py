from __future__ import annotations

from dataclasses import replace
from pathlib import Path

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
    )


def test_run_to_splits_executes_each_stage_once(monkeypatch):
    pipeline = EtaPipeline()
    calls = []
    monkeypatch.setattr(pipeline, "build_data", lambda: calls.append("data"))
    monkeypatch.setattr(pipeline, "build_features", lambda: calls.append("features"))
    monkeypatch.setattr(pipeline, "build_splits", lambda: calls.append("splits"))

    pipeline.run_to("splits")

    assert calls == ["data", "features", "splits"]


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
        )
    )
    assert (
        len(pd.read_csv(paths.raw_shipments)) == config.DEFAULT_RUN_SETTINGS.n_shipments
    )
    assert len(pd.read_csv(paths.splits)) == (
        config.DEFAULT_RUN_SETTINGS.n_shipments
        * config.DEFAULT_RUN_SETTINGS.n_split_seeds
    )


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
