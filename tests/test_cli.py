from __future__ import annotations

import json

import numpy as np
import pytest

from smartstress_policy_reliability.cli import main


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_cli_fit_evaluate_and_subject_leakage_guard(tmp_path) -> None:
    rng = np.random.default_rng(7)
    reference = [
        {"subject_id": "reference-subject", "features": row.tolist()}
        for row in rng.normal(size=(80, 12))
    ]
    calibration = []
    for index, row in enumerate(rng.normal(size=(60, 12))):
        label = index % 2
        calibration.append(
            {
                "subject_id": "calibration-subject",
                "features": row.tolist(),
                "label": label,
                "raw_logit": (-1.0 if label == 0 else 1.0) + float(rng.normal(0, 0.2)),
            }
        )
    evaluation = []
    for index, row in enumerate(rng.normal(size=(12, 12))):
        label = index % 2
        evaluation.append(
            {
                "subject_id": "held-out-subject",
                "session_id": "held-out-session",
                "timestamp": f"2026-01-01T00:00:{index:02d}Z",
                "features": row.tolist(),
                "label": label,
                "raw_logit": -1.2 if label == 0 else 1.2,
            }
        )
    reference_path = tmp_path / "reference.jsonl"
    calibration_path = tmp_path / "calibration.jsonl"
    evaluation_path = tmp_path / "evaluation.jsonl"
    policy_path = tmp_path / "policy.json"
    report_path = tmp_path / "report.json"
    _write_jsonl(reference_path, reference)
    _write_jsonl(calibration_path, calibration)
    _write_jsonl(evaluation_path, evaluation)

    assert main(
        [
            "fit",
            "--reference",
            str(reference_path),
            "--calibration",
            str(calibration_path),
            "--output",
            str(policy_path),
        ]
    ) == 0
    assert policy_path.exists()
    assert main(
        [
            "evaluate",
            "--policy",
            str(policy_path),
            "--data",
            str(evaluation_path),
            "--output",
            str(report_path),
        ]
    ) == 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert len(report["decisions"]) == len(evaluation)
    assert "calibration_before" in report and "calibration_after" in report
    assert "policy_metrics" in report

    leaked = [dict(row, subject_id="calibration-subject") for row in evaluation]
    leaked_path = tmp_path / "leaked.jsonl"
    _write_jsonl(leaked_path, leaked)
    with pytest.raises(ValueError, match="leakage"):
        main(
            [
                "evaluate",
                "--policy",
                str(policy_path),
                "--data",
                str(leaked_path),
                "--output",
                str(report_path),
            ]
        )
