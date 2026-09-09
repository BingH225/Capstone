"""Command-line fitting and evaluation for reproducible policy manifests."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .calibration import calibration_diagnostics, sigmoid
from .contracts import FEATURE_NAMES, InferenceInput
from .evaluation import policy_metrics
from .policy import ReliabilityPolicy, ReliabilityPolicyConfig
from .temporal import TemporalPolicyConfig


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _subject_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number} is not a JSON object")
            rows.append(value)
        return rows
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError("input files must use .jsonl or .csv")


def _parse_features(row: Mapping[str, Any]) -> tuple[float, ...]:
    if "features" in row and row["features"] not in (None, ""):
        raw = row["features"]
        if isinstance(raw, str):
            raw = json.loads(raw)
        values = tuple(float(value) for value in raw)
    else:
        values = tuple(float(row[name]) for name in FEATURE_NAMES)
    if len(values) != len(FEATURE_NAMES):
        raise ValueError(f"expected {len(FEATURE_NAMES)} features")
    return values


def _subject_ids(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    return {
        str(row["subject_id"])
        for row in rows
        if row.get("subject_id") not in (None, "")
    }


def _validate_subjects(
    reference_rows: Sequence[Mapping[str, Any]],
    calibration_rows: Sequence[Mapping[str, Any]],
    *,
    allow_overlap: bool,
    allow_missing: bool,
) -> tuple[set[str], set[str]]:
    reference_ids = _subject_ids(reference_rows)
    calibration_ids = _subject_ids(calibration_rows)
    if not allow_missing and (
        len(reference_ids) == 0
        or len(calibration_ids) == 0
        or any(row.get("subject_id") in (None, "") for row in reference_rows)
        or any(row.get("subject_id") in (None, "") for row in calibration_rows)
    ):
        raise ValueError(
            "every fitting row requires subject_id; use --allow-missing-subject-ids only for a documented non-subject protocol"
        )
    overlap = reference_ids & calibration_ids
    if overlap and not allow_overlap:
        raise ValueError(
            "reference/calibration subject leakage detected: " + ", ".join(sorted(overlap))
        )
    return reference_ids, calibration_ids


def _extract_predictions(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    has_probabilities = all(row.get("raw_probability") not in (None, "") for row in rows)
    has_logits = all(row.get("raw_logit") not in (None, "") for row in rows)
    if has_probabilities == has_logits:
        raise ValueError(
            "calibration/evaluation rows must contain exactly one complete raw_probability or raw_logit column"
        )
    if has_probabilities:
        return np.asarray([float(row["raw_probability"]) for row in rows]), None
    return None, np.asarray([float(row["raw_logit"]) for row in rows])


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def fit_command(args: argparse.Namespace) -> int:
    reference_path = Path(args.reference).resolve()
    calibration_path = Path(args.calibration).resolve()
    reference_rows = _read_rows(reference_path)
    calibration_rows = _read_rows(calibration_path)
    reference_ids, calibration_ids = _validate_subjects(
        reference_rows,
        calibration_rows,
        allow_overlap=args.allow_subject_overlap,
        allow_missing=args.allow_missing_subject_ids,
    )
    reference_features = [_parse_features(row) for row in reference_rows]
    calibration_features = [_parse_features(row) for row in calibration_rows]
    labels = [int(row["label"]) for row in calibration_rows]
    probabilities, logits = _extract_predictions(calibration_rows)
    config = ReliabilityPolicyConfig(
        policy_version=args.policy_version,
        expected_model_id=args.model_id,
        calibrator=args.calibrator,
        conformal_alpha=args.alpha,
        minimum_confidence=args.minimum_confidence,
        ood_threshold_quantile=args.ood_quantile,
        minimum_external_quality=args.minimum_external_quality,
        quality_soft_z_limit=args.quality_soft_z_limit,
        quality_hard_z_limit=args.quality_hard_z_limit,
        require_baseline_version=args.require_baseline_version,
        temporal=TemporalPolicyConfig(
            window_size=args.window_size,
            required_elevated=args.required_elevated,
            threshold_on=args.threshold_on,
            threshold_off=args.threshold_off,
            cooldown_seconds=args.cooldown_seconds,
        ),
    )
    provenance = {
        "reference_file": reference_path.name,
        "reference_sha256": _file_sha256(reference_path),
        "reference_rows": len(reference_rows),
        "reference_subject_hashes": sorted(_subject_hash(value) for value in reference_ids),
        "calibration_file": calibration_path.name,
        "calibration_sha256": _file_sha256(calibration_path),
        "calibration_rows": len(calibration_rows),
        "calibration_subject_hashes": sorted(
            _subject_hash(value) for value in calibration_ids
        ),
        "subject_overlap_allowed": bool(args.allow_subject_overlap),
        "missing_subject_ids_allowed": bool(args.allow_missing_subject_ids),
    }
    policy = ReliabilityPolicy.fit(
        reference_features=reference_features,
        calibration_labels=labels,
        calibration_features=calibration_features,
        calibration_probabilities=probabilities,
        calibration_logits=logits,
        config=config,
        data_provenance=provenance,
    )
    output = policy.save(args.output)
    print(output)
    return 0


def _assert_evaluation_subject_separation(
    policy: ReliabilityPolicy,
    rows: Sequence[Mapping[str, Any]],
    *,
    allow_overlap: bool,
    allow_missing: bool,
) -> None:
    ids = _subject_ids(rows)
    if not allow_missing and (
        not ids or any(row.get("subject_id") in (None, "") for row in rows)
    ):
        raise ValueError("every evaluation row requires subject_id")
    if allow_overlap:
        return
    evaluation_hashes = {_subject_hash(value) for value in ids}
    fitted_hashes = set(policy.data_provenance.get("reference_subject_hashes", []))
    fitted_hashes.update(policy.data_provenance.get("calibration_subject_hashes", []))
    overlap = evaluation_hashes & fitted_hashes
    if overlap:
        raise ValueError("evaluation subject leakage detected against policy provenance")


def evaluate_command(args: argparse.Namespace) -> int:
    policy = ReliabilityPolicy.load(args.policy)
    data_path = Path(args.data).resolve()
    rows = _read_rows(data_path)
    _assert_evaluation_subject_separation(
        policy,
        rows,
        allow_overlap=args.allow_subject_overlap,
        allow_missing=args.allow_missing_subject_ids,
    )
    probabilities, logits = _extract_predictions(rows)
    raw_probabilities = (
        probabilities
        if probabilities is not None
        else sigmoid(np.asarray(logits, dtype=np.float64))
    )
    labels = [int(row["label"]) for row in rows]
    default_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    decisions = []
    for index, row in enumerate(rows):
        timestamp = row.get("timestamp") or (
            default_start + timedelta(seconds=index)
        ).isoformat()
        inference = InferenceInput(
            features=_parse_features(row),
            raw_probability=(
                float(row["raw_probability"])
                if row.get("raw_probability") not in (None, "")
                else None
            ),
            raw_logit=(
                float(row["raw_logit"])
                if row.get("raw_logit") not in (None, "")
                else None
            ),
            timestamp=timestamp,
            model_id=str(row.get("model_id") or policy.config.expected_model_id),
            session_id=str(row.get("session_id") or row.get("subject_id") or "evaluation"),
            external_quality_score=(
                float(row["external_quality_score"])
                if row.get("external_quality_score") not in (None, "")
                else None
            ),
            input_source=(str(row["input_source"]) if row.get("input_source") not in (None, "") else None),
            baseline_version=(str(row["baseline_version"]) if row.get("baseline_version") not in (None, "") else None),
            signal_quality=(
                json.loads(row["signal_quality"])
                if isinstance(row.get("signal_quality"), str) and row.get("signal_quality")
                else dict(row.get("signal_quality") or {})
            ),
            top_drivers=(
                tuple(json.loads(row["top_drivers"]))
                if isinstance(row.get("top_drivers"), str) and row.get("top_drivers")
                else tuple(row.get("top_drivers") or ())
            ),
        )
        decisions.append(policy.evaluate(inference))
    calibrated = policy.calibrator.predict_from_probabilities(raw_probabilities)
    report = {
        "policy": str(Path(args.policy).resolve()),
        "data_file": data_path.name,
        "data_sha256": _file_sha256(data_path),
        "calibration_before": calibration_diagnostics(labels, raw_probabilities),
        "calibration_after": calibration_diagnostics(labels, calibrated),
        "policy_metrics": policy_metrics(labels, decisions),
        "decisions": [decision.to_dict() for decision in decisions],
    }
    _write_json(Path(args.output), report)
    print(Path(args.output).resolve())
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="smartstress-reliability",
        description="Fit and evaluate the SmartStress post-DNN reliability policy.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser("fit", help="Fit a checksummed policy manifest")
    fit_parser.add_argument("--reference", required=True)
    fit_parser.add_argument("--calibration", required=True)
    fit_parser.add_argument("--output", required=True)
    fit_parser.add_argument("--policy-version", default="physio-rel-v1")
    fit_parser.add_argument("--model-id", default="wesad_attention_v1")
    fit_parser.add_argument(
        "--calibrator", choices=["identity", "temperature", "platt"], default="temperature"
    )
    fit_parser.add_argument("--alpha", type=float, default=0.1)
    fit_parser.add_argument("--minimum-confidence", type=float, default=0.6)
    fit_parser.add_argument("--ood-quantile", type=float, default=0.99)
    fit_parser.add_argument("--minimum-external-quality", type=float, default=0.5)
    fit_parser.add_argument("--quality-soft-z-limit", type=float, default=6.0)
    fit_parser.add_argument("--quality-hard-z-limit", type=float, default=12.0)
    fit_parser.add_argument("--require-baseline-version", action="store_true")
    fit_parser.add_argument("--window-size", type=int, default=5)
    fit_parser.add_argument("--required-elevated", type=int, default=3)
    fit_parser.add_argument("--threshold-on", type=float, default=0.65)
    fit_parser.add_argument("--threshold-off", type=float, default=0.45)
    fit_parser.add_argument("--cooldown-seconds", type=int, default=900)
    fit_parser.add_argument("--allow-subject-overlap", action="store_true")
    fit_parser.add_argument("--allow-missing-subject-ids", action="store_true")
    fit_parser.set_defaults(handler=fit_command)

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a held-out dataset")
    eval_parser.add_argument("--policy", required=True)
    eval_parser.add_argument("--data", required=True)
    eval_parser.add_argument("--output", required=True)
    eval_parser.add_argument("--allow-subject-overlap", action="store_true")
    eval_parser.add_argument("--allow-missing-subject-ids", action="store_true")
    eval_parser.set_defaults(handler=evaluate_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
