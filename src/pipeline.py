"""Artifact I/O and orchestration for the ETA pipeline."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Sequence

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
    models,
    splits,
)
from src.errors import ArtifactError, ConfigurationError, PipelineError


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
        model = models.ModelR(self.settings).fit(stacked, split_manifest)
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
        model = models.ModelQ(self.settings).fit(stacked, split_manifest)
        _write_artifact(model.predict_test(), self.paths.model_q_predictions)

    def build_interval_evaluation(self, model_q_path: Path | None = None) -> None:
        """Evaluate held-out Model Q coverage and interval width by checkpoint."""
        predictions = _read_artifact(
            model_q_path if model_q_path is not None else self.paths.model_q_predictions
        )
        metrics = evaluate.evaluate_model_q_intervals(predictions)
        _write_artifact(metrics, self.paths.model_q_metrics)

    def run_to(
        self,
        target: Literal[
            "data",
            "features",
            "splits",
            "baselines",
            "model_r",
            "evaluation",
            "model_q",
            "interval_evaluation",
        ],
    ) -> None:
        """Run the target's prerequisites once, without unrelated branches."""
        self._run_targets((target,))

    def _run_targets(self, targets: tuple[str, ...]) -> None:
        builders = {
            "data": self.build_data,
            "features": self.build_features,
            "splits": self.build_splits,
            "baselines": self.build_baselines,
            "model_r": self.build_model_r,
            "evaluation": self.build_evaluation,
            "model_q": self.build_model_q,
            "interval_evaluation": self.build_interval_evaluation,
        }
        prerequisites = {
            "data": (),
            "features": ("data",),
            "splits": ("features",),
            "baselines": ("splits",),
            "model_r": ("splits",),
            "evaluation": ("baselines", "model_r"),
            "model_q": ("splits",),
            "interval_evaluation": ("model_q",),
        }
        completed: set[str] = set()

        def build(stage: str) -> None:
            try:
                build_stage = builders[stage]
                stage_prerequisites = prerequisites[stage]
            except KeyError as error:
                raise ConfigurationError(f"unknown pipeline target: {stage}") from error
            if stage in completed:
                return
            for prerequisite in stage_prerequisites:
                build(prerequisite)
            build_stage()
            completed.add(stage)

        for target in targets:
            build(target)

    def run_all(self) -> None:
        """Run every implemented model and evaluation branch once."""
        self._run_targets(("evaluation", "interval_evaluation"))


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

    for target in (
        "all",
        "data",
        "features",
        "splits",
        "baselines",
        "model-r",
        "eval",
        "model-q",
        "eval-q",
    ):
        command = subparsers.add_parser(target)
        command.add_argument("--seed", type=int)
        command.add_argument("--n-shipments", type=int)
        command.add_argument("--rate", type=float)
        if target == "data":
            command.add_argument("--raw", type=Path)
            command.add_argument("--out", type=Path)
        elif target == "features":
            command.add_argument("--observed", type=Path)
            command.add_argument("--delays", type=Path)
            command.add_argument("--out", type=Path)
        elif target == "splits":
            command.add_argument("--stacked", type=Path)
            command.add_argument("--out", type=Path)
        elif target == "baselines":
            command.add_argument("--stacked", type=Path)
            command.add_argument("--splits", type=Path)
            command.add_argument("--out", type=Path)
        elif target == "model-r":
            command.add_argument("--stacked", type=Path)
            command.add_argument("--splits", type=Path)
            command.add_argument("--out", type=Path)
        elif target == "eval":
            command.add_argument("--model-r", type=Path)
            command.add_argument("--baselines", type=Path)
            command.add_argument("--out", type=Path)
        elif target == "model-q":
            command.add_argument("--stacked", type=Path)
            command.add_argument("--splits", type=Path)
            command.add_argument("--out", type=Path)
        elif target == "eval-q":
            command.add_argument("--model-q", type=Path)
            command.add_argument("--out", type=Path)

    return parser


def _run_target(pipeline: EtaPipeline, args: argparse.Namespace) -> None:
    default_actions = {
        "all": pipeline.run_all,
        "data": lambda: pipeline.run_to("data"),
        "features": lambda: pipeline.run_to("features"),
        "splits": lambda: pipeline.run_to("splits"),
        "baselines": lambda: pipeline.run_to("baselines"),
        "model-r": lambda: pipeline.run_to("model_r"),
        "eval": lambda: pipeline.run_to("evaluation"),
        "model-q": lambda: pipeline.run_to("model_q"),
        "eval-q": lambda: pipeline.run_to("interval_evaluation"),
    }
    explicit_actions = {
        "data": (
            getattr(args, "raw", None) is not None,
            lambda: pipeline.build_data(args.raw),
        ),
        "features": (
            getattr(args, "observed", None) is not None
            or getattr(args, "delays", None) is not None,
            lambda: pipeline.build_features(args.observed, args.delays),
        ),
        "splits": (
            getattr(args, "stacked", None) is not None,
            lambda: pipeline.build_splits(args.stacked),
        ),
        "baselines": (
            getattr(args, "stacked", None) is not None
            or getattr(args, "splits", None) is not None,
            lambda: pipeline.build_baselines(args.stacked, args.splits),
        ),
        "model-r": (
            getattr(args, "stacked", None) is not None
            or getattr(args, "splits", None) is not None,
            lambda: pipeline.build_model_r(args.stacked, args.splits),
        ),
        "eval": (
            getattr(args, "model_r", None) is not None
            or getattr(args, "baselines", None) is not None,
            lambda: pipeline.build_evaluation(args.model_r, args.baselines),
        ),
        "model-q": (
            getattr(args, "stacked", None) is not None
            or getattr(args, "splits", None) is not None,
            lambda: pipeline.build_model_q(args.stacked, args.splits),
        ),
        "eval-q": (
            getattr(args, "model_q", None) is not None,
            lambda: pipeline.build_interval_evaluation(args.model_q),
        ),
    }
    use_explicit_input, explicit_action = explicit_actions.get(
        args.target, (False, None)
    )
    if use_explicit_input and explicit_action is not None:
        explicit_action()
    else:
        default_actions[args.target]()


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

        if args.target == "data" and args.out is not None and args.raw is None:
            paths = replace(
                paths,
                raw_shipments=args.out / paths.raw_shipments.name,
                truth=args.out / paths.truth.name,
            )
        elif args.target == "features" and args.out is not None:
            paths = replace(paths, stacked=args.out / paths.stacked.name)
        elif args.target == "splits" and args.out is not None:
            paths = replace(paths, splits=args.out / paths.splits.name)
        elif args.target == "baselines" and args.out is not None:
            paths = replace(
                paths,
                baseline_predictions=args.out / paths.baseline_predictions.name,
            )
        elif args.target == "model-r" and args.out is not None:
            paths = replace(
                paths,
                model_r_predictions=args.out / paths.model_r_predictions.name,
            )
        elif args.target == "eval" and args.out is not None:
            paths = replace(
                paths,
                model_r_metrics=args.out / paths.model_r_metrics.name,
            )
        elif args.target == "model-q" and args.out is not None:
            paths = replace(
                paths,
                model_q_predictions=args.out / paths.model_q_predictions.name,
            )
        elif args.target == "eval-q" and args.out is not None:
            paths = replace(
                paths,
                model_q_metrics=args.out / paths.model_q_metrics.name,
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
