"""Artifact I/O and orchestration for the ETA pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal

import pandas as pd
import pyarrow as pa

from src import (
    ablations,
    baselines,
    config,
    delays,
    evaluate,
    explain,
    features,
    generate,
    ingest,
    missingness,
    model_c,
    model_q,
    model_r,
    splits,
)
from src.errors import ArtifactError, ConfigurationError, ContractError, PipelineError

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
    "missingness_sweep",
    "explanations",
    "ablations",
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
    model_r_group_metrics: Path = Path("outputs/metrics/model_r_by_group.csv")
    model_r_comparison_plot: Path = Path("outputs/metrics/model_r_vs_baselines.png")
    model_q_predictions: Path = Path("outputs/predictions/model_q.csv")
    model_q_metrics: Path = Path("outputs/metrics/model_q_coverage.csv")
    model_q_group_metrics: Path = Path("outputs/metrics/model_q_by_group.csv")
    model_c_predictions: Path = Path("outputs/predictions/model_c.csv")
    model_c_tuning: Path = Path("outputs/metrics/model_c_tuning.csv")
    model_c_reliability: Path = Path("outputs/metrics/model_c_reliability.csv")
    model_c_reliability_plot: Path = Path("outputs/metrics/model_c_reliability.png")
    model_c_classification_metrics: Path = Path(
        "outputs/metrics/model_c_classification_metrics.csv"
    )
    model_c_threshold_sweep: Path = Path("outputs/metrics/model_c_threshold_sweep.csv")
    model_c_reliability_by_group: Path = Path(
        "outputs/metrics/model_c_reliability_by_group.csv"
    )
    model_c_classification_metrics_by_group: Path = Path(
        "outputs/metrics/model_c_classification_metrics_by_group.csv"
    )
    model_c_threshold_sweep_by_group: Path = Path(
        "outputs/metrics/model_c_threshold_sweep_by_group.csv"
    )
    model_r_missingness_sweep: Path = Path(
        "outputs/metrics/model_r_missingness_sweep.csv"
    )
    model_r_missingness_sweep_plot: Path = Path(
        "outputs/metrics/model_r_missingness_sweep.png"
    )
    model_r_explanations: Path = Path("outputs/predictions/model_r_explanations.csv")
    model_r_explanation_recovery: Path = Path(
        "outputs/metrics/model_r_explanation_recovery.csv"
    )
    model_r_shap_summary: Path = Path("outputs/metrics/model_r_shap_summary.png")
    model_r_slack_dependence: Path = Path(
        "outputs/metrics/model_r_slack_dependence.png"
    )
    model_r_slot_dependence: Path = Path("outputs/metrics/model_r_slot_dependence.png")
    model_r_explanation_waterfall: Path = Path(
        "outputs/metrics/model_r_explanation_waterfall.png"
    )
    model_r_ablation_metrics: Path = Path("outputs/metrics/model_r_ablations.csv")
    run_manifest: Path = Path("outputs/run_manifest.json")


@dataclass(frozen=True)
class _PipelineStage:
    builder: str
    prerequisites: tuple[_PipelineStageName, ...] = ()


_MISSINGNESS_SWEEP_RATES = (0.0, 0.1, 0.2, 0.3)


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
    "missingness_sweep": _PipelineStage("build_missingness_sweep"),
    "explanations": _PipelineStage("build_explanations", ("model_r",)),
    "ablations": _PipelineStage("build_ablations", ("baselines", "model_r")),
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
        path_args=(
            "model-r",
            "baselines",
            "shipments",
            "truth",
            "hidden-mask",
            "out",
        ),
        explicit_method="build_evaluation",
        explicit_inputs=(
            "model_r",
            "baselines",
            "shipments",
            "truth",
            "hidden_mask",
        ),
        output_fields=(
            "model_r_metrics",
            "model_r_group_metrics",
            "model_r_comparison_plot",
        ),
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
        output_fields=("model_c_predictions", "model_c_tuning"),
    ),
    "eval-q": _CliTarget(
        stage="interval_evaluation",
        path_args=("model-q", "shipments", "truth", "hidden-mask", "out"),
        explicit_method="build_interval_evaluation",
        explicit_inputs=("model_q", "shipments", "truth", "hidden_mask"),
        output_fields=("model_q_metrics", "model_q_group_metrics"),
    ),
    "eval-c": _CliTarget(
        stage="classification_evaluation",
        path_args=("model-c", "shipments", "truth", "hidden-mask", "out"),
        explicit_method="build_model_c_evaluation",
        explicit_inputs=("model_c", "shipments", "truth", "hidden_mask"),
        output_fields=(
            "model_c_reliability",
            "model_c_reliability_plot",
            "model_c_classification_metrics",
            "model_c_threshold_sweep",
            "model_c_reliability_by_group",
            "model_c_classification_metrics_by_group",
            "model_c_threshold_sweep_by_group",
        ),
    ),
    "eval-missingness": _CliTarget(
        stage="missingness_sweep",
        path_args=("out",),
        output_fields=(
            "model_r_missingness_sweep",
            "model_r_missingness_sweep_plot",
        ),
    ),
    "explain": _CliTarget(
        stage="explanations",
        path_args=("out",),
        output_fields=(
            "model_r_explanations",
            "model_r_explanation_recovery",
            "model_r_shap_summary",
            "model_r_slack_dependence",
            "model_r_slot_dependence",
            "model_r_explanation_waterfall",
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
        self._model_r_model: model_r.ModelR | None = None

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
        self._model_r_model = model

    def _load_evaluation_group_metadata(
        self,
        paths: tuple[Path | None, Path | None, Path | None],
        custom_predictions: bool,
    ) -> pd.DataFrame:
        if custom_predictions and any(path is None for path in paths):
            raise ConfigurationError(
                "custom predictions require explicit --shipments, --truth, "
                "and --hidden-mask metadata"
            )
        if any(path is not None for path in paths) and any(
            path is None for path in paths
        ):
            raise ConfigurationError(
                "evaluation metadata must include shipments, truth, and hidden mask"
            )
        shipments_path, truth_path, hidden_mask_path = paths
        return evaluate.build_evaluation_group_metadata(
            ingest.load(shipments_path or self.paths.raw_shipments),
            _read_artifact(truth_path or self.paths.truth),
            _read_artifact(hidden_mask_path or self.paths.hidden_mask),
        )

    def build_evaluation(
        self,
        model_path: Path | None = None,
        baselines_path: Path | None = None,
        shipments_path: Path | None = None,
        truth_path: Path | None = None,
        hidden_mask_path: Path | None = None,
    ) -> None:
        """Report checkpoint and subgroup metrics for Model R vs B2."""
        metadata = self._load_evaluation_group_metadata(
            (shipments_path, truth_path, hidden_mask_path),
            model_path is not None or baselines_path is not None,
        )
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
        evaluate.plot_model_r_mae_comparison(
            model_predictions,
            baseline_predictions,
            self.paths.model_r_comparison_plot,
        )
        grouped_metrics = evaluate.compare_model_r_to_b2_by_group(
            model_predictions, baseline_predictions, metadata
        )
        _write_artifact(metrics, self.paths.model_r_metrics)
        _write_artifact(grouped_metrics, self.paths.model_r_group_metrics)

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
        model = model_c.ModelC(self.settings).fit(stacked, split_manifest)
        _write_artifact(model.predict_test(), self.paths.model_c_predictions)
        _write_artifact(model.tuning_results, self.paths.model_c_tuning)

    def build_interval_evaluation(
        self,
        model_q_path: Path | None = None,
        shipments_path: Path | None = None,
        truth_path: Path | None = None,
        hidden_mask_path: Path | None = None,
    ) -> None:
        """Evaluate held-out Model Q intervals by checkpoint and shipment cut."""
        metadata = self._load_evaluation_group_metadata(
            (shipments_path, truth_path, hidden_mask_path), model_q_path is not None
        )
        predictions = _read_artifact(
            model_q_path if model_q_path is not None else self.paths.model_q_predictions
        )
        metrics = evaluate.evaluate_model_q_intervals(predictions)
        grouped_metrics = evaluate.evaluate_model_q_intervals_by_group(
            predictions, metadata
        )
        _write_artifact(metrics, self.paths.model_q_metrics)
        _write_artifact(grouped_metrics, self.paths.model_q_group_metrics)

    def build_model_c_evaluation(
        self,
        model_c_path: Path | None = None,
        shipments_path: Path | None = None,
        truth_path: Path | None = None,
        hidden_mask_path: Path | None = None,
    ) -> None:
        """Report Model C metrics, reliability, and thresholds by shipment cut."""
        metadata = self._load_evaluation_group_metadata(
            (shipments_path, truth_path, hidden_mask_path), model_c_path is not None
        )
        predictions = _read_artifact(
            model_c_path if model_c_path is not None else self.paths.model_c_predictions
        )
        reliability = evaluate.evaluate_model_c_reliability(predictions)
        metrics = evaluate.evaluate_model_c_classification_metrics(predictions)
        threshold_sweep = evaluate.evaluate_model_c_threshold_sweep(predictions)
        grouped_reliability = evaluate.evaluate_model_c_reliability_by_group(
            predictions, metadata
        )
        grouped_metrics = evaluate.evaluate_model_c_classification_metrics_by_group(
            predictions, metadata
        )
        grouped_threshold_sweep = evaluate.evaluate_model_c_threshold_sweep_by_group(
            predictions, metadata
        )
        evaluate.plot_model_c_reliability(
            reliability, self.paths.model_c_reliability_plot
        )
        _write_artifact(reliability, self.paths.model_c_reliability)
        _write_artifact(metrics, self.paths.model_c_classification_metrics)
        _write_artifact(threshold_sweep, self.paths.model_c_threshold_sweep)
        _write_artifact(grouped_reliability, self.paths.model_c_reliability_by_group)
        _write_artifact(
            grouped_metrics, self.paths.model_c_classification_metrics_by_group
        )
        _write_artifact(
            grouped_threshold_sweep, self.paths.model_c_threshold_sweep_by_group
        )

    def build_missingness_sweep(self) -> None:
        """Compare Model R with B2 at the fixed missingness rates."""
        results = []
        for rate in _MISSINGNESS_SWEEP_RATES:
            with TemporaryDirectory(prefix=f"eta-missingness-{rate:g}-") as directory:
                root = Path(directory)
                paths = ArtifactPaths(
                    raw_shipments=root / "shipments.csv",
                    truth=root / "truth.csv",
                    observed=root / "observed.csv",
                    hidden_mask=root / "hidden_mask.csv",
                    delays=root / "delays.csv",
                    stacked=root / "stacked.parquet",
                    splits=root / "splits.csv",
                    baseline_predictions=root / "baseline_predictions.csv",
                    model_r_predictions=root / "model_r.csv",
                    model_r_metrics=root / "model_r_metrics.csv",
                    model_r_group_metrics=root / "model_r_group_metrics.csv",
                    model_r_comparison_plot=root / "model_r_comparison.png",
                )
                EtaPipeline(replace(self.settings, missing_rate=rate), paths).run_to(
                    "evaluation"
                )
                results.append(
                    _read_artifact(paths.model_r_metrics).assign(missing_rate=rate)
                )

        metrics = pd.concat(results, ignore_index=True)
        evaluate.plot_model_r_missingness_sweep(
            metrics, self.paths.model_r_missingness_sweep_plot
        )
        _write_artifact(metrics, self.paths.model_r_missingness_sweep)

    def build_explanations(self) -> None:
        """Write held-out Model R facts, grouped SHAP, and recovery checks."""
        if self._model_r_model is None:
            raise ConfigurationError("Model R must be fit before explaining")
        explanations, recovery = explain.build_model_r_explanations(
            self._model_r_model,
            _read_artifact(self.paths.delays),
            _read_artifact(self.paths.truth),
        )
        explain.plot_model_r_shap_summary(explanations, self.paths.model_r_shap_summary)
        explain.plot_model_r_dependence(
            explanations, "slack_remaining", self.paths.model_r_slack_dependence
        )
        explain.plot_model_r_dependence(
            explanations, "slot_margin", self.paths.model_r_slot_dependence
        )
        _write_artifact(explanations, self.paths.model_r_explanations)
        _write_artifact(recovery, self.paths.model_r_explanation_recovery)

    def write_model_r_waterfall(
        self, shipment_id: str, split_seed: int, checkpoint: int
    ) -> None:
        """Write one requested held-out prediction's fact/SHAP waterfall."""
        table = _read_artifact(self.paths.model_r_explanations)
        selected = table.loc[
            (table["shipment_id"].astype("string") == shipment_id)
            & pd.to_numeric(table["split_seed"], errors="coerce").eq(split_seed)
            & pd.to_numeric(table["checkpoint"], errors="coerce").eq(checkpoint)
        ]
        if selected.empty:
            raise ConfigurationError(
                f"no held-out explanation for {shipment_id}, split {split_seed}, "
                f"checkpoint C{checkpoint}"
            )
        if len(selected) != 1:
            raise ContractError("held-out explanation keys must be unique")
        explain.plot_model_r_waterfall(
            selected.iloc[0], self.paths.model_r_explanation_waterfall
        )

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

    def build_ablations(self) -> None:
        """Compare Model R's baseline, environment, and link feature sets."""
        result = ablations.evaluate_model_r_ablations(
            _read_artifact(self.paths.stacked),
            _read_artifact(self.paths.splits),
            _read_artifact(self.paths.baseline_predictions),
            _read_artifact(self.paths.model_r_predictions),
            self.settings,
        )
        _write_artifact(result, self.paths.model_r_ablation_metrics)

    def run_all(self) -> None:
        """Run every implemented model and evaluation branch once."""
        self._run_targets(
            (
                "evaluation",
                "interval_evaluation",
                "classification_evaluation",
                "missingness_sweep",
                "explanations",
                "ablations",
            )
        )
        self._write_run_manifest()

    def _write_run_manifest(self) -> None:
        """Record the run settings, source revision, and headline metrics."""
        try:
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=Path(__file__).resolve().parents[1],
                check=False,
                capture_output=True,
                text=True,
            )
        except OSError:
            git_hash = None
            git_dirty = None
        else:
            git_hash = revision.stdout.strip() if revision.returncode == 0 else None
            git_dirty = bool(status.stdout.strip()) if status.returncode == 0 else None
        metrics = {
            "model_r_vs_b2": _metric_records(
                self.paths.model_r_metrics,
                ("checkpoint", "model_mae_mean_h", "b2_mae_mean_h", "skill_score"),
            ),
            "model_q_intervals": _metric_records(
                self.paths.model_q_metrics,
                ("checkpoint", "coverage_mean", "mean_width_h"),
            ),
            "model_c_classification": _metric_records(
                self.paths.model_c_classification_metrics,
                (
                    "checkpoint",
                    "brier_score_mean",
                    "roc_auc_mean",
                    "pr_auc_mean",
                    "f1_at_0_5_mean",
                ),
            ),
            "model_r_ablations": _metric_records(
                self.paths.model_r_ablation_metrics,
                ("variant", "checkpoint", "model_mae_mean_h", "skill_score"),
            ),
        }
        manifest = {
            "seed": self.settings.seed,
            "config": asdict(self.settings),
            "git_hash": git_hash,
            "git_dirty": git_dirty,
            "metrics_summary": metrics,
        }
        try:
            self.paths.run_manifest.parent.mkdir(parents=True, exist_ok=True)
            self.paths.run_manifest.write_text(
                json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n",
                encoding="utf-8",
            )
        except (OSError, TypeError, ValueError) as error:
            raise ArtifactError(
                f"cannot write run manifest {self.paths.run_manifest}: {error}"
            ) from error


def _metric_records(path: Path, columns: tuple[str, ...]) -> list[dict[str, object]]:
    table = _read_artifact(path)
    missing = sorted(set(columns) - set(table.columns))
    if missing:
        raise ContractError(f"metric artifact {path} is missing columns: {missing}")
    return json.loads(table.loc[:, list(columns)].to_json(orient="records"))


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
        if target == "explain":
            command.add_argument("--shipment-id")
            command.add_argument("--split-seed", type=int)
            command.add_argument("--checkpoint", type=int)

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
        if args.target == "explain" and args.shipment_id is not None:
            pipeline.write_model_r_waterfall(
                args.shipment_id, args.split_seed, args.checkpoint
            )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; expected pipeline failures are concise and non-zero."""
    args = _parser().parse_args(argv)
    try:
        if args.target == "eval-missingness" and args.rate is not None:
            raise ConfigurationError(
                "eval-missingness uses fixed rates 0, 0.1, 0.2, and 0.3; "
                "--rate is not supported"
            )
        if args.target == "explain":
            selection = (args.shipment_id, args.split_seed, args.checkpoint)
            if any(value is not None for value in selection) and any(
                value is None for value in selection
            ):
                raise ConfigurationError(
                    "--shipment-id, --split-seed, and --checkpoint must be provided together"
                )
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
