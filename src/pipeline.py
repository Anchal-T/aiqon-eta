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

from src import config, delays, features, generate, ingest, missingness, splits
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

    def run_to(self, target: Literal["data", "features", "splits"]) -> None:
        """Run the default prerequisite chain through ``target`` once."""
        stages = (
            ("data", self.build_data),
            ("features", self.build_features),
            ("splits", self.build_splits),
        )
        targets = tuple(name for name, _ in stages)
        try:
            last_stage = targets.index(target)
        except ValueError as error:
            raise ConfigurationError(f"unknown pipeline target: {target}") from error

        for _, build_stage in stages[: last_stage + 1]:
            build_stage()

    def run_all(self) -> None:
        """Run the complete data, feature, and split pipeline once."""
        self.run_to("splits")


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

    for target in ("all", "data", "features", "splits"):
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

    return parser


def _run_target(pipeline: EtaPipeline, args: argparse.Namespace) -> None:
    default_actions = {
        "all": pipeline.run_all,
        "data": lambda: pipeline.run_to("data"),
        "features": lambda: pipeline.run_to("features"),
        "splits": lambda: pipeline.run_to("splits"),
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
    }
    use_explicit_input, explicit_action = explicit_actions.get(
        args.target, (False, None)
    )
    (explicit_action if use_explicit_input else default_actions[args.target])()


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

        pipeline = EtaPipeline(settings, paths)
        _run_target(pipeline, args)
        print(f"{args.target} pipeline completed")
        return 0
    except PipelineError as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
