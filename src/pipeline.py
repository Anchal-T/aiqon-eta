"""Artifact I/O and orchestration for the ETA pipeline."""

from __future__ import annotations

import argparse
import csv
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa

from src import (
    baselines,
    config,
    delays,
    evaluate,
    features,
    generate,
    ingest,
    missingness,
    model_c,
    model_q,
    model_r,
    splits,
)
from src.errors import ArtifactError, ConfigurationError, PipelineError

_PipelineStageName = Literal[
    "data",
    "features",
    "splits",
    "baselines",
    "model_r",
    "evaluation",
    "model_q",
    "model_c",
    "interval_evaluation",
    "classification_evaluation",
]


@dataclass(frozen=True)
class ArtifactPaths:
    """Locations of the files produced and consumed by a pipeline run."""

    raw_shipments: Path = Path("data/raw/shipments.csv")
    truth: Path = Path("data/raw/truth.csv")
    observed: Path = Path("data/interim/observed.csv")
    hidden_mask: Path = Path("data/interim/hidden_mask.csv")
    delays: Path = Path("data/interim/delays.csv")
    stacked: Path = Path("data/processed/stacked.parquet")
    splits: Path = Path("data/processed/splits.csv")
    baseline_predictions: Path = Path("data/processed/baseline_predictions.csv")
    model_r_predictions: Path = Path("outputs/predictions/model_r.csv")
    model_r_metrics: Path = Path("outputs/metrics/model_r_vs_b2.csv")
    model_q_predictions: Path = Path("outputs/predictions/model_q.csv")
    model_q_metrics: Path = Path("outputs/metrics/model_q_coverage.csv")
    model_c_predictions: Path = Path("outputs/predictions/model_c.csv")
    model_c_reliability: Path = Path("outputs/metrics/model_c_reliability.csv")
    model_c_reliability_plot: Path = Path("outputs/metrics/model_c_reliability.png")
    model_c_classification_metrics: Path = Path(
        "outputs/metrics/model_c_classification_metrics.csv"
    )
    model_c_threshold_sweep: Path = Path("outputs/metrics/model_c_threshold_sweep.csv")


@dataclass(frozen=True)
class _PipelineStage:
    builder: str
    prerequisites: tuple[_PipelineStageName, ...] = ()


_PIPELINE_STAGES: dict[_PipelineStageName, _PipelineStage] = {
    "data": _PipelineStage("build_data"),
    "features": _PipelineStage("build_features", ("data",)),
    "splits": _PipelineStage("build_splits", ("features",)),
    "baselines": _PipelineStage("build_baselines", ("splits",)),
    "model_r": _PipelineStage("build_model_r", ("splits",)),
    "evaluation": _PipelineStage("build_evaluation", ("baselines", "model_r")),
    "model_q": _PipelineStage("build_model_q", ("splits",)),
    "model_c": _PipelineStage("build_model_c", ("splits",)),
    "interval_evaluation": _PipelineStage("build_interval_evaluation", ("model_q",)),
    "classification_evaluation": _PipelineStage(
        "build_model_c_evaluation", ("model_c",)
    ),
}


@dataclass(frozen=True)
class _CliTarget:
    stage: _PipelineStageName | None
    path_args: tuple[str, ...] = ()
    explicit_method: str | None = None
    explicit_inputs: tuple[str, ...] = ()
    output_fields: tuple[str, ...] = ()


_CLI_TARGETS = {
    "all": _CliTarget(stage=None),
    "data": _CliTarget(
        stage="data",
        path_args=("raw", "out"),
        explicit_method="build_data",
        explicit_inputs=("raw",),
        output_fields=("raw_shipments", "truth"),
    ),
    "features": _CliTarget(
        stage="features",
        path_args=("observed", "delays", "out"),
        explicit_method="build_features",
        explicit_inputs=("observed", "delays"),
        output_fields=("stacked",),
    ),
    "splits": _CliTarget(
        stage="splits",
        path_args=("stacked", "out"),
        explicit_method="build_splits",
        explicit_inputs=("stacked",),
        output_fields=("splits",),
    ),
    "baselines": _CliTarget(
        stage="baselines",
        path_args=("stacked", "splits", "out"),
        explicit_method="build_baselines",
        explicit_inputs=("stacked", "splits"),
        output_fields=("baseline_predictions",),
    ),
    "model-r": _CliTarget(
        stage="model_r",
        path_args=("stacked", "splits", "out"),
        explicit_method="build_model_r",
        explicit_inputs=("stacked", "splits"),
        output_fields=("model_r_predictions",),
    ),
    "eval": _CliTarget(
        stage="evaluation",
        path_args=("model-r", "baselines", "out"),
        explicit_method="build_evaluation",
        explicit_inputs=("model_r", "baselines"),
        output_fields=("model_r_metrics",),
    ),
    "model-q": _CliTarget(
        stage="model_q",
        path_args=("stacked", "splits", "out"),
        explicit_method="build_model_q",
        explicit_inputs=("stacked", "splits"),
        output_fields=("model_q_predictions",),
    ),
    "model-c": _CliTarget(
        stage="model_c",
        path_args=("stacked", "splits", "out"),
        explicit_method="build_model_c",
        explicit_inputs=("stacked", "splits"),
        output_fields=("model_c_predictions",),
    ),
    "eval-q": _CliTarget(
        stage="interval_evaluation",
        path_args=("model-q", "out"),
        explicit_method="build_interval_evaluation",
        explicit_inputs=("model_q",),
        output_fields=("model_q_metrics",),
    ),
    "eval-c": _CliTarget(
        stage="classification_evaluation",
        path_args=("model-c", "out"),
        explicit_method="build_model_c_evaluation",
        explicit_inputs=("model_c",),
        output_fields=(
            "model_c_reliability",
            "model_c_reliability_plot",
            "model_c_classification_metrics",
            "model_c_threshold_sweep",
        ),
    ),
}


class EtaPipeline:
    """Own one run's settings, artifact paths, and stage order."""

    def __init__(
        self,
        settings: config.RunSettings = config.DEFAULT_RUN_SETTINGS,
        paths: ArtifactPaths | None = None,
    ) -> None:
        self.settings = settings
        self.paths = paths if paths is not None else ArtifactPaths()

    def build_data(self, raw_path: Path | None = None) -> None:
        """Generate or load raw data, then simulate missingness and delays."""
        source = raw_path if raw_path is not None else self.paths.raw_shipments
        if raw_path is None:
            shipments, truth = generate.generate(self.settings)
            _write_artifact(shipments, self.paths.raw_shipments)
            _write_artifact(truth, self.paths.truth)

        raw = ingest.load(source)
        observed, mask = missingness.hide(raw, self.settings)
        _write_artifact(observed, self.paths.observed)
        _write_artifact(mask, self.paths.hidden_mask)
        _write_artifact(delays.compute(observed), self.paths.delays)

    def build_features(
        self,
        observed_path: Path | None = None,
        delays_path: Path | None = None,
    ) -> None:
        """Build features from the selected observed and delay artifacts."""
        observed = ingest.load(
            observed_path if observed_path is not None else self.paths.observed,
            complete_actuals=False,
        )
        delay_table = _read_artifact(
            delays_path if delays_path is not None else self.paths.delays
        )
        stacked = features.build(observed, delay_table, self.settings)
        _write_artifact(stacked, self.paths.stacked)

    def build_splits(self, stacked_path: Path | None = None) -> None:
        """Build splits from the selected stacked feature artifact."""
        stacked = _read_artifact(
            stacked_path if stacked_path is not None else self.paths.stacked
        )
        _write_artifact(splits.create_splits(stacked, self.settings), self.paths.splits)

    def build_baselines(
        self,
        stacked_path: Path | None = None,
        splits_path: Path | None = None,
    ) -> None:
        """Generate held-out baseline predictions from features and splits."""
        stacked = _read_artifact(
            stacked_path if stacked_path is not None else self.paths.stacked
        )
        split_manifest = _read_artifact(
            splits_path if splits_path is not None else self.paths.splits
        )
        predictions = baselines.predict(stacked, split_manifest)
        _write_artifact(predictions, self.paths.baseline_predictions)

    def build_model_r(
        self,
        stacked_path: Path | None = None,
        splits_path: Path | None = None,
    ) -> None:
        """Fit Model R and write its outer-test predictions."""
        stacked = _read_artifact(
            stacked_path if stacked_path is not None else self.paths.stacked
        )
        split_manifest = _read_artifact(
            splits_path if splits_path is not None else self.paths.splits
        )
        model = model_r.ModelR(self.settings).fit(stacked, split_manifest)
        _write_artifact(model.predict_test(), self.paths.model_r_predictions)

    def build_evaluation(
        self,
        model_path: Path | None = None,
        baselines_path: Path | None = None,
    ) -> None:
        """Compare Model R's held-out MAE with B2 by checkpoint."""
        model_predictions = _read_artifact(
            model_path if model_path is not None else self.paths.model_r_predictions
        )
        baseline_predictions = _read_artifact(
            baselines_path
            if baselines_path is not None
            else self.paths.baseline_predictions
        )
        metrics = evaluate.compare_model_r_to_b2(
            model_predictions, baseline_predictions
        )
        _write_artifact(metrics, self.paths.model_r_metrics)

    def build_model_q(
        self,
        stacked_path: Path | None = None,
        splits_path: Path | None = None,
    ) -> None:
        """Fit Model Q and write its outer-test prediction intervals."""
        stacked = _read_artifact(
            stacked_path if stacked_path is not None else self.paths.stacked
        )
        split_manifest = _read_artifact(
            splits_path if splits_path is not None else self.paths.splits
        )
        model = model_q.ModelQ(self.settings).fit(stacked, split_manifest)
        _write_artifact(model.predict_test(), self.paths.model_q_predictions)

    def build_model_c(
        self,
        stacked_path: Path | None = None,
        splits_path: Path | None = None,
    ) -> None:
        """Fit Model C and write its outer-test delay probabilities."""
        stacked = _read_artifact(
            stacked_path if stacked_path is not None else self.paths.stacked
        )
        split_manifest = _read_artifact(
            splits_path if splits_path is not None else self.paths.splits
        )
        model = model_c.ModelC().fit(stacked, split_manifest)
        _write_artifact(model.predict_test(), self.paths.model_c_predictions)

    def build_interval_evaluation(self, model_q_path: Path | None = None) -> None:
        """Evaluate held-out Model Q coverage and interval width by checkpoint."""
        predictions = _read_artifact(
            model_q_path if model_q_path is not None else self.paths.model_q_predictions
        )
        metrics = evaluate.evaluate_model_q_intervals(predictions)
        _write_artifact(metrics, self.paths.model_q_metrics)

    def build_model_c_evaluation(self, model_c_path: Path | None = None) -> None:
        """Report and plot the held-out Model C reliability curve."""
        predictions = _read_artifact(
            model_c_path if model_c_path is not None else self.paths.model_c_predictions
        )
        reliability = evaluate.evaluate_model_c_reliability(predictions)
        metrics = evaluate.evaluate_model_c_classification_metrics(predictions)
        threshold_sweep = evaluate.evaluate_model_c_threshold_sweep(predictions)
        evaluate.plot_model_c_reliability(
            reliability, self.paths.model_c_reliability_plot
        )
        _write_artifact(reliability, self.paths.model_c_reliability)
        _write_artifact(metrics, self.paths.model_c_classification_metrics)
        _write_artifact(threshold_sweep, self.paths.model_c_threshold_sweep)

    def run_to(self, target: _PipelineStageName) -> None:
        """Run the target's prerequisites once, without unrelated branches."""
        self._run_targets((target,))

    def _run_targets(self, targets: tuple[_PipelineStageName, ...]) -> None:
        completed: set[str] = set()

        def build(stage: _PipelineStageName) -> None:
            stage_spec = _PIPELINE_STAGES.get(stage)
            if stage_spec is None:
                raise ConfigurationError(f"unknown pipeline target: {stage}")
            if stage in completed:
                return
            for prerequisite in stage_spec.prerequisites:
                build(prerequisite)
            getattr(self, stage_spec.builder)()
            completed.add(stage)

        for target in targets:
            build(target)

    def run_all(self) -> None:
        """Run every implemented model and evaluation branch once."""
        self._run_targets(
            ("evaluation", "interval_evaluation", "classification_evaluation")
        )


def _write_artifact(table: pd.DataFrame, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".csv":
            table.to_csv(path, index=False)
        elif path.suffix == ".parquet":
            table.to_parquet(path, index=False)
        else:
            raise ArtifactError(f"unsupported artifact format: {path}")
    except (
        OSError,
        UnicodeError,
        ImportError,
        ValueError,
        csv.Error,
        pa.ArrowException,
    ) as error:
        raise ArtifactError(f"cannot write artifact {path}: {error}") from error


def _read_artifact(path: Path) -> pd.DataFrame:
    try:
        if path.suffix == ".csv":
            return pd.read_csv(path)
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        raise ArtifactError(f"unsupported artifact format: {path}")
    except (
        OSError,
        UnicodeError,
        ImportError,
        ValueError,
        csv.Error,
        pd.errors.ParserError,
        pa.ArrowException,
    ) as error:
        raise ArtifactError(f"cannot read artifact {path}: {error}") from error


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build ETA pipeline artifacts.")
    subparsers = parser.add_subparsers(dest="target", required=True)

    for target, spec in _CLI_TARGETS.items():
        command = subparsers.add_parser(target)
        command.add_argument("--seed", type=int)
        command.add_argument("--n-shipments", type=int)
        command.add_argument("--rate", type=float)
        for argument in spec.path_args:
            command.add_argument(f"--{argument}", type=Path)

    return parser


def _run_target(pipeline: EtaPipeline, args: argparse.Namespace) -> None:
    spec = _CLI_TARGETS[args.target]
    if spec.stage is None:
        pipeline.run_all()
    elif spec.explicit_method and any(
        getattr(args, name) is not None for name in spec.explicit_inputs
    ):
        inputs = (getattr(args, name) for name in spec.explicit_inputs)
        getattr(pipeline, spec.explicit_method)(*inputs)
    else:
        pipeline.run_to(spec.stage)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; expected pipeline failures are concise and non-zero."""
    args = _parser().parse_args(argv)
    try:
        settings_updates = {
            key: value
            for key, value in (
                ("seed", args.seed),
                ("n_shipments", args.n_shipments),
                ("missing_rate", args.rate),
            )
            if value is not None
        }
        settings = replace(config.DEFAULT_RUN_SETTINGS, **settings_updates)
        paths = ArtifactPaths()

        target_spec = _CLI_TARGETS[args.target]
        if (
            target_spec.output_fields
            and getattr(args, "out", None) is not None
            and (args.target != "data" or args.raw is None)
        ):
            paths = replace(
                paths,
                **{
                    field: args.out / getattr(paths, field).name
                    for field in target_spec.output_fields
                },
            )

        pipeline = EtaPipeline(settings, paths)
        _run_target(pipeline, args)
        print(f"{args.target} pipeline completed")
        return 0
    except PipelineError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
