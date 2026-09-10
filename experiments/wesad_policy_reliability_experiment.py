"""Real WESAD LOSO experiment for the post-DNN Policy Reliability module.

The existing cross-validation script selected the best epoch using the held-out
subject. This experiment deliberately uses epoch 49 for every fold so the
reported out-of-fold predictions are not selected on the evaluation subject.
For each outer subject, the other 14 subjects are deterministically divided into
disjoint policy-reference and policy-calibration groups.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from smartstress_policy_reliability import (  # noqa: E402
    FEATURE_NAMES,
    InferenceInput,
    ReliabilityPolicy,
    ReliabilityPolicyConfig,
    ReliabilityState,
    TemporalPolicyConfig,
)
from smartstress_policy_reliability.calibration import calibration_metrics  # noqa: E402


SUBJECTS = (
    "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9", "S10", "S11",
    "S13", "S14", "S15", "S16", "S17",
)
METHODS = ("raw_0.5", "calibrated", "quality_ood", "selective", "full_temporal")


class ClassifierDNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.nnECG = nn.Sequential(
            nn.Linear(12, 128), nn.BatchNorm1d(128), nn.Dropout(0.5), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.Dropout(0.5), nn.LeakyReLU(0.2),
            nn.Linear(64, 16), nn.BatchNorm1d(16), nn.Dropout(0.5), nn.LeakyReLU(0.2),
            nn.Linear(16, 4), nn.BatchNorm1d(4), nn.Dropout(0.5), nn.LeakyReLU(0.2),
            nn.Linear(4, 1), nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.nnECG(inputs)


class ClassifierDNNAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_dim = 32
        self.embedding = nn.Linear(1, self.embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.embed_dim, num_heads=4, batch_first=True
        )
        self.nnECG = nn.Sequential(
            nn.Linear(12 * self.embed_dim, 128), nn.BatchNorm1d(128), nn.Dropout(0.5), nn.LeakyReLU(0.2),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.Dropout(0.5), nn.LeakyReLU(0.2),
            nn.Linear(64, 16), nn.BatchNorm1d(16), nn.Dropout(0.5), nn.LeakyReLU(0.2),
            nn.Linear(16, 4), nn.BatchNorm1d(4), nn.Dropout(0.5), nn.LeakyReLU(0.2),
            nn.Linear(4, 1), nn.Sigmoid(),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        batch_size = inputs.size(0)
        embedded = self.embedding(inputs.view(batch_size, 12, 1))
        attended, _ = self.attention(embedded, embedded, embedded)
        return self.nnECG(attended.reshape(batch_size, -1))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_subject(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    features = np.asarray(payload["features"], dtype=np.float32)
    original_labels = np.asarray(payload["label"], dtype=np.int64)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"{path} does not contain the required 12 features")
    if original_labels.shape != (features.shape[0],):
        raise ValueError(f"{path} labels are not aligned")
    if not np.isfinite(features).all():
        raise ValueError(f"{path} contains non-finite features")
    return features, (original_labels == 2).astype(np.int64)


def checkpoint_path(model_repo: Path, fold: int, model_name: str, epoch: int) -> Path:
    suffix = "Attention" if model_name == "attention" else "DNN"
    return model_repo / "Models_CrossVal_Full" / f"fold_{fold}_{suffix}" / f"epoch_{epoch}.pth"


def build_model(model_name: str) -> nn.Module:
    if model_name == "attention":
        return ClassifierDNNAttention()
    if model_name == "dnn":
        return ClassifierDNN()
    raise ValueError(f"unknown model: {model_name}")


def predict(
    model_name: str,
    checkpoint: Path,
    features: np.ndarray,
    *,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model = build_model(model_name).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    outputs: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start : start + batch_size]).to(device)
            values = model(batch).reshape(-1).float().cpu().numpy()
            outputs.append(values)
    probabilities = np.concatenate(outputs).astype(np.float64)
    if probabilities.shape != (features.shape[0],) or not np.isfinite(probabilities).all():
        raise RuntimeError(f"invalid predictions from {checkpoint}")
    return probabilities


def deterministic_policy_split(eval_subject: str, seed: int) -> tuple[list[str], list[str]]:
    candidates = [subject for subject in SUBJECTS if subject != eval_subject]
    ordered = sorted(
        candidates,
        key=lambda subject: hashlib.sha256(f"{seed}:{eval_subject}:{subject}".encode()).hexdigest(),
    )
    midpoint = len(ordered) // 2
    return sorted(ordered[:midpoint]), sorted(ordered[midpoint:])


def confusion_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=np.int64)
    predictions = np.asarray(predictions, dtype=np.int64)
    if labels.shape != predictions.shape or labels.size == 0:
        return {
            "rows": int(labels.size), "accuracy": 0.0, "precision": 0.0,
            "recall": 0.0, "f1": 0.0, "tp": 0, "tn": 0, "fp": 0, "fn": 0,
        }
    tp = int(np.sum((labels == 1) & (predictions == 1)))
    tn = int(np.sum((labels == 0) & (predictions == 0)))
    fp = int(np.sum((labels == 0) & (predictions == 1)))
    fn = int(np.sum((labels == 1) & (predictions == 0)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "rows": int(labels.size),
        "accuracy": (tp + tn) / labels.size,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    labels = np.asarray(labels, dtype=np.int64)
    scores = np.asarray(scores, dtype=np.float64)
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if not positives.size or not negatives.size:
        return None
    ordered = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(ordered, dtype=np.float64)
    ranks[ordered] = np.arange(1, len(scores) + 1, dtype=np.float64)
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    for group, count in enumerate(counts):
        if count > 1:
            indices = np.where(inverse == group)[0]
            ranks[indices] = np.mean(ranks[indices])
    rank_sum = float(np.sum(ranks[labels == 1]))
    return (rank_sum - positives.size * (positives.size + 1) / 2) / (positives.size * negatives.size)


def method_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    predictions: np.ndarray,
    accepted: np.ndarray,
    *,
    notifications: int,
    false_notifications: int,
    window_seconds: float,
) -> dict[str, Any]:
    accepted = np.asarray(accepted, dtype=bool)
    kept_labels = labels[accepted]
    kept_predictions = predictions[accepted]
    kept_scores = scores[accepted]
    result: dict[str, Any] = confusion_metrics(kept_labels, kept_predictions)
    result.update(
        {
            "total_rows": int(labels.size),
            "coverage": float(np.mean(accepted)),
            "abstained_rows": int(np.sum(~accepted)),
            "selective_risk": 1.0 - float(result["accuracy"]) if kept_labels.size else None,
            "auroc": auc(kept_labels, kept_scores),
            "notifications": int(notifications),
            "false_notifications": int(false_notifications),
            "observation_hours": labels.size * window_seconds / 3600.0,
            "false_notifications_per_hour": (
                false_notifications / (labels.size * window_seconds / 3600.0)
                if labels.size and window_seconds > 0 else None
            ),
        }
    )
    if kept_labels.size:
        result["calibration"] = calibration_metrics(kept_labels, kept_scores)
    return result


def flatten(parts: Iterable[np.ndarray]) -> np.ndarray:
    arrays = list(parts)
    return np.concatenate(arrays) if arrays else np.asarray([])


def fit_policy(
    model_name: str,
    eval_subject: str,
    reference_subjects: Sequence[str],
    calibration_subjects: Sequence[str],
    features: Mapping[str, np.ndarray],
    labels: Mapping[str, np.ndarray],
    probabilities: Mapping[str, np.ndarray],
    *,
    alpha: float,
    minimum_confidence: float,
    ood_quantile: float,
    seed: int,
) -> ReliabilityPolicy:
    reference = flatten(features[subject] for subject in reference_subjects)
    cal_features = flatten(features[subject] for subject in calibration_subjects)
    cal_labels = flatten(labels[subject] for subject in calibration_subjects).astype(np.int64)
    cal_probabilities = flatten(probabilities[subject] for subject in calibration_subjects)
    model_id = f"wesad_{model_name}_loso_epoch49"
    config = ReliabilityPolicyConfig(
        policy_version=f"wesad-policy-crossfit-v1-{model_name}-{eval_subject}",
        expected_model_id=model_id,
        calibrator="temperature",
        conformal_alpha=alpha,
        minimum_confidence=minimum_confidence,
        ood_threshold_quantile=ood_quantile,
        temporal=TemporalPolicyConfig(
            window_size=5,
            required_elevated=3,
            threshold_on=0.65,
            threshold_off=0.45,
            cooldown_seconds=900,
        ),
    )
    return ReliabilityPolicy.fit(
        reference_features=reference,
        calibration_labels=cal_labels,
        calibration_features=cal_features,
        calibration_probabilities=cal_probabilities,
        config=config,
        data_provenance={
            "protocol": "subject-wise outer LOSO with disjoint OOF reference/calibration subjects",
            "seed": seed,
            "evaluation_subject": eval_subject,
            "reference_subjects": list(reference_subjects),
            "calibration_subjects": list(calibration_subjects),
            "checkpoint_epoch": 49,
        },
    )


def evaluate_subject(
    model_name: str,
    subject: str,
    policy: ReliabilityPolicy,
    subject_features: np.ndarray,
    subject_labels: np.ndarray,
    raw_probabilities: np.ndarray,
    *,
    window_seconds: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    calibrated = policy.calibrator.predict_from_probabilities(raw_probabilities)
    raw_predictions = (raw_probabilities >= 0.5).astype(np.int64)
    calibrated_predictions = (calibrated >= 0.5).astype(np.int64)
    all_rows = np.ones(len(subject_labels), dtype=bool)

    quality_ood_mask = np.zeros(len(subject_labels), dtype=bool)
    selective_mask = np.zeros(len(subject_labels), dtype=bool)
    for index, row in enumerate(subject_features):
        quality = policy.quality_profile.evaluate(row)
        if quality.valid and not policy.ood_detector.evaluate(row).is_ood:
            quality_ood_mask[index] = True
            selective_mask[index] = policy.selective_predictor.evaluate(calibrated[index]).accepted

    full_predictions = np.zeros(len(subject_labels), dtype=np.int64)
    full_mask = np.zeros(len(subject_labels), dtype=bool)
    notifications = np.zeros(len(subject_labels), dtype=bool)
    state_counts: dict[str, int] = {}
    reason_counts: dict[str, int] = {}
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    for index, (row, probability) in enumerate(zip(subject_features, raw_probabilities)):
        decision = policy.evaluate(
            InferenceInput(
                features=row,
                raw_probability=float(probability),
                timestamp=start + timedelta(seconds=index * window_seconds),
                model_id=policy.config.expected_model_id,
                session_id=f"{model_name}:{subject}",
                input_source="WESAD-processed-20s-window",
            )
        )
        state_counts[decision.reliability_state.value] = state_counts.get(decision.reliability_state.value, 0) + 1
        for reason in decision.reason_codes:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if decision.reliability_state in {ReliabilityState.RELIABLE_LOW, ReliabilityState.RELIABLE_ELEVATED}:
            full_mask[index] = True
            full_predictions[index] = int(decision.reliability_state == ReliabilityState.RELIABLE_ELEVATED)
        notifications[index] = decision.proactive_notification

    methods = {
        "raw_0.5": method_metrics(
            subject_labels, raw_probabilities, raw_predictions, all_rows,
            notifications=int(np.sum(raw_predictions == 1)),
            false_notifications=int(np.sum((raw_predictions == 1) & (subject_labels == 0))),
            window_seconds=window_seconds,
        ),
        "calibrated": method_metrics(
            subject_labels, calibrated, calibrated_predictions, all_rows,
            notifications=int(np.sum(calibrated_predictions == 1)),
            false_notifications=int(np.sum((calibrated_predictions == 1) & (subject_labels == 0))),
            window_seconds=window_seconds,
        ),
        "quality_ood": method_metrics(
            subject_labels, calibrated, calibrated_predictions, quality_ood_mask,
            notifications=int(np.sum((calibrated_predictions == 1) & quality_ood_mask)),
            false_notifications=int(np.sum((calibrated_predictions == 1) & (subject_labels == 0) & quality_ood_mask)),
            window_seconds=window_seconds,
        ),
        "selective": method_metrics(
            subject_labels, calibrated, calibrated_predictions, selective_mask,
            notifications=int(np.sum((calibrated_predictions == 1) & selective_mask)),
            false_notifications=int(np.sum((calibrated_predictions == 1) & (subject_labels == 0) & selective_mask)),
            window_seconds=window_seconds,
        ),
        "full_temporal": method_metrics(
            subject_labels, calibrated, full_predictions, full_mask,
            notifications=int(np.sum(notifications)),
            false_notifications=int(np.sum(notifications & (subject_labels == 0))),
            window_seconds=window_seconds,
        ),
    }
    diagnostics = {"state_counts": state_counts, "reason_counts": reason_counts}
    arrays = {
        "labels": subject_labels,
        "raw": raw_probabilities,
        "calibrated": calibrated,
        "raw_predictions": raw_predictions,
        "calibrated_predictions": calibrated_predictions,
        "quality_ood_mask": quality_ood_mask,
        "selective_mask": selective_mask,
        "full_mask": full_mask,
        "full_predictions": full_predictions,
        "notifications": notifications,
    }
    return methods, diagnostics, arrays


def corruption_audit(policy: ReliabilityPolicy, features: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    count = min(100, len(features))
    rejected = {"nonfinite": 0, "extreme": 0, "feature_order": 0}
    start = datetime(2027, 1, 1, tzinfo=timezone.utc)
    for index in range(count):
        base = features[index].astype(np.float64)
        cases = {
            "nonfinite": (np.where(np.arange(len(base)) == 0, np.nan, base), FEATURE_NAMES),
            "extreme": (base * 50.0, FEATURE_NAMES),
            "feature_order": (base, tuple(reversed(FEATURE_NAMES))),
        }
        for offset, (name, (row, order)) in enumerate(cases.items()):
            decision = policy.evaluate(
                InferenceInput(
                    features=row,
                    raw_probability=float(probabilities[index]),
                    timestamp=start + timedelta(seconds=index * 10 + offset),
                    model_id=policy.config.expected_model_id,
                    session_id=f"corruption:{name}:{index}",
                    feature_names=order,
                )
            )
            rejected[name] += int(
                decision.reliability_state in {ReliabilityState.DATA_INVALID, ReliabilityState.OOD_OR_UNCERTAIN}
            )
    return {
        "rows_per_corruption": count,
        "rejected": rejected,
        "rejection_rates": {name: value / count for name, value in rejected.items()},
    }


def aggregate_model(
    model_name: str,
    per_subject: Sequence[Mapping[str, Any]],
    arrays_by_subject: Mapping[str, Mapping[str, np.ndarray]],
    *,
    window_seconds: float,
) -> dict[str, Any]:
    labels = flatten(arrays_by_subject[subject]["labels"] for subject in SUBJECTS)
    raw = flatten(arrays_by_subject[subject]["raw"] for subject in SUBJECTS)
    calibrated = flatten(arrays_by_subject[subject]["calibrated"] for subject in SUBJECTS)
    output: dict[str, Any] = {
        "model": model_name,
        "subjects": len(SUBJECTS),
        "rows": int(labels.size),
        "calibration_raw": calibration_metrics(labels, raw),
        "calibration_after": calibration_metrics(labels, calibrated),
        "methods": {},
    }
    for method in METHODS:
        if method == "raw_0.5":
            scores = raw
            predictions = flatten(arrays_by_subject[s]["raw_predictions"] for s in SUBJECTS)
            accepted = np.ones(labels.size, dtype=bool)
            notifications = int(np.sum(predictions == 1))
            false_notifications = int(np.sum((predictions == 1) & (labels == 0)))
        elif method == "calibrated":
            scores = calibrated
            predictions = flatten(arrays_by_subject[s]["calibrated_predictions"] for s in SUBJECTS)
            accepted = np.ones(labels.size, dtype=bool)
            notifications = int(np.sum(predictions == 1))
            false_notifications = int(np.sum((predictions == 1) & (labels == 0)))
        elif method == "quality_ood":
            scores = calibrated
            predictions = flatten(arrays_by_subject[s]["calibrated_predictions"] for s in SUBJECTS)
            accepted = flatten(arrays_by_subject[s]["quality_ood_mask"] for s in SUBJECTS).astype(bool)
            notifications = int(np.sum((predictions == 1) & accepted))
            false_notifications = int(np.sum((predictions == 1) & (labels == 0) & accepted))
        elif method == "selective":
            scores = calibrated
            predictions = flatten(arrays_by_subject[s]["calibrated_predictions"] for s in SUBJECTS)
            accepted = flatten(arrays_by_subject[s]["selective_mask"] for s in SUBJECTS).astype(bool)
            notifications = int(np.sum((predictions == 1) & accepted))
            false_notifications = int(np.sum((predictions == 1) & (labels == 0) & accepted))
        else:
            scores = calibrated
            predictions = flatten(arrays_by_subject[s]["full_predictions"] for s in SUBJECTS)
            accepted = flatten(arrays_by_subject[s]["full_mask"] for s in SUBJECTS).astype(bool)
            notification_mask = flatten(arrays_by_subject[s]["notifications"] for s in SUBJECTS).astype(bool)
            notifications = int(np.sum(notification_mask))
            false_notifications = int(np.sum(notification_mask & (labels == 0)))
        metrics = method_metrics(
            labels, scores, predictions, accepted,
            notifications=notifications,
            false_notifications=false_notifications,
            window_seconds=window_seconds,
        )
        subject_values = [row["methods"][method] for row in per_subject]
        metrics["macro_subject_mean"] = {
            key: float(np.mean([float(value[key]) for value in subject_values if value[key] is not None]))
            for key in ("coverage", "accuracy", "precision", "recall", "f1", "selective_risk")
        }
        output["methods"][method] = metrics
    raw_false = int(output["methods"]["raw_0.5"]["false_notifications"])
    full_false = int(output["methods"]["full_temporal"]["false_notifications"])
    subject_coverages = {
        str(row["subject"]): float(row["methods"]["full_temporal"]["coverage"])
        for row in per_subject
    }
    adequately_covered = [
        row for row in per_subject
        if float(row["methods"]["full_temporal"]["coverage"]) >= 0.50
    ]
    output["subject_heterogeneity"] = {
        "full_policy_coverage_by_subject": subject_coverages,
        "coverage_min": float(min(subject_coverages.values())),
        "coverage_median": float(np.median(list(subject_coverages.values()))),
        "coverage_max": float(max(subject_coverages.values())),
        "subjects_below_50_percent_coverage": sorted(
            subject for subject, coverage in subject_coverages.items() if coverage < 0.50
        ),
        "subjects_at_least_50_percent_coverage": len(adequately_covered),
        "risk_improved_at_least_50_percent_coverage": sum(
            row["methods"]["full_temporal"]["selective_risk"]
            < row["methods"]["raw_0.5"]["selective_risk"]
            for row in adequately_covered
        ),
        "f1_improved_at_least_50_percent_coverage": sum(
            row["methods"]["full_temporal"]["f1"] > row["methods"]["raw_0.5"]["f1"]
            for row in adequately_covered
        ),
    }
    output["policy_effect"] = {
        "coverage_change": output["methods"]["full_temporal"]["coverage"] - 1.0,
        "accepted_risk_change": output["methods"]["full_temporal"]["selective_risk"] - output["methods"]["raw_0.5"]["selective_risk"],
        "accepted_f1_change": output["methods"]["full_temporal"]["f1"] - output["methods"]["raw_0.5"]["f1"],
        "naive_per_window_false_dispatch_reduction": (1.0 - full_false / raw_false) if raw_false else None,
        "ece_change_after_calibration": output["calibration_after"]["ece_15"] - output["calibration_raw"]["ece_15"],
    }
    return output


def write_oof_csv(path: Path, predictions: Mapping[str, Mapping[str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "subject_id", "window_index", "label", "raw_probability"))
        for model_name in sorted(predictions):
            for subject in SUBJECTS:
                values = predictions[model_name][subject]
                for index, (label, probability) in enumerate(zip(values["labels"], values["probabilities"])):
                    writer.writerow((model_name, subject, index, int(label), f"{float(probability):.10f}"))


def write_per_subject_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = (
        "model", "subject", "method", "rows", "coverage", "accuracy", "precision",
        "recall", "f1", "auroc", "selective_risk", "notifications", "false_notifications",
        "false_notifications_per_hour",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            for method in METHODS:
                metrics = row["methods"][method]
                writer.writerow({
                    "model": row["model"], "subject": row["subject"], "method": method,
                    **{field: metrics.get(field) for field in fields if field not in {"model", "subject", "method"}},
                })


def generate_plots(output_dir: Path, summaries: Mapping[str, Mapping[str, Any]]) -> None:
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    markers = {"attention": "o", "dnn": "s"}
    for model_name, summary in summaries.items():
        coverages = [summary["methods"][method]["coverage"] for method in METHODS]
        risks = [summary["methods"][method]["selective_risk"] for method in METHODS]
        axes[0].plot(coverages, risks, marker=markers[model_name], label=model_name)
        for method, x, y in zip(METHODS, coverages, risks):
            axes[0].annotate(method.replace("_", "\n"), (x, y), fontsize=7)
    axes[0].set_xlabel("Coverage")
    axes[0].set_ylabel("Accepted-sample error (selective risk)")
    axes[0].set_title("Coverage–risk trade-off")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    labels = list(summaries)
    x = np.arange(len(labels))
    width = 0.35
    axes[1].bar(x - width / 2, [summaries[m]["calibration_raw"]["ece_15"] for m in labels], width, label="raw")
    axes[1].bar(x + width / 2, [summaries[m]["calibration_after"]["ece_15"] for m in labels], width, label="temperature-scaled")
    axes[1].set_xticks(x, labels)
    axes[1].set_ylabel("ECE (15 bins; lower is better)")
    axes[1].set_title("Cross-fitted calibration")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_dir / "policy_tradeoff.png", dpi=180)
    plt.close(figure)


def markdown_report(payload: Mapping[str, Any]) -> str:
    lines = [
        "# WESAD Post-DNN Policy Reliability Experiment",
        "",
        f"Generated: `{payload['generated_at']}`",
        "",
        "## Protocol",
        "",
        "- 15-subject WESAD leave-one-subject-out evaluation.",
        "- Fixed `epoch_49` checkpoint for every DNN fold; the held-out subject is not used to select an epoch.",
        "- For each outer subject, the other 14 subjects are deterministically split into 7 reference and 7 calibration subjects.",
        "- Calibration uses only out-of-fold predictions from non-evaluation subjects.",
        "- Original preprocessing provides chronological 20-second windows with a nominal 1-second step; alert rates use retained-window exposure.",
        "- Main comparison: raw threshold → calibration → quality/OOD → selective prediction → full temporal policy.",
        "",
        "![Policy trade-off](policy_tradeoff.png)",
        "",
    ]
    for model_name, summary in payload["models"].items():
        title = "Attention DNN" if model_name == "attention" else "Standard DNN"
        lines.extend([
            f"## {title}",
            "",
            f"Rows: `{summary['rows']}`; subjects: `{summary['subjects']}`.",
            "",
            "| Method | Coverage | Accuracy* | Precision* | Recall* | F1* | Selective risk | False dispatches/alerts† |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ])
        for method in METHODS:
            metrics = summary["methods"][method]
            lines.append(
                f"| {method} | {metrics['coverage']:.4f} | {metrics['accuracy']:.4f} | "
                f"{metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} | "
                f"{metrics['selective_risk']:.4f} | {metrics['false_notifications']} |"
            )
        effect = summary["policy_effect"]
        lines.extend([
            "",
            "`*` Metrics after abstention are computed only on released RELIABLE samples; coverage must always be reported alongside them.",
            "",
            f"- Raw ECE: `{summary['calibration_raw']['ece_15']:.4f}`; calibrated ECE: `{summary['calibration_after']['ece_15']:.4f}`.",
            f"- Full-policy coverage change: `{effect['coverage_change']:+.4f}`.",
            f"- Full-policy accepted-risk change: `{effect['accepted_risk_change']:+.4f}`.",
            f"- Full-policy accepted-F1 change: `{effect['accepted_f1_change']:+.4f}`.",
            f"- Naive per-window false-dispatch reduction after temporal event coalescing: `{effect['naive_per_window_false_dispatch_reduction']:.2%}`.",
            f"- Subject coverage min/median/max: `{summary['subject_heterogeneity']['coverage_min']:.4f}` / "
            f"`{summary['subject_heterogeneity']['coverage_median']:.4f}` / "
            f"`{summary['subject_heterogeneity']['coverage_max']:.4f}`.",
            f"- Subjects below 50% coverage: `{', '.join(summary['subject_heterogeneity']['subjects_below_50_percent_coverage']) or 'none'}`.",
            f"- Among subjects with at least 50% coverage, accepted risk improved for "
            f"`{summary['subject_heterogeneity']['risk_improved_at_least_50_percent_coverage']}/"
            f"{summary['subject_heterogeneity']['subjects_at_least_50_percent_coverage']}` and accepted F1 improved for "
            f"`{summary['subject_heterogeneity']['f1_improved_at_least_50_percent_coverage']}/"
            f"{summary['subject_heterogeneity']['subjects_at_least_50_percent_coverage']}`.",
            "",
        ])
    lines.extend([
        "## Controlled corruption audit",
        "",
        "| Model | Non-finite | Extreme scale | Wrong feature order |",
        "|---|---:|---:|---:|",
    ])
    for model_name, audit in payload["corruption_audits"].items():
        rates = audit["rejection_rates"]
        lines.append(f"| {model_name} | {rates['nonfinite']:.2%} | {rates['extreme']:.2%} | {rates['feature_order']:.2%} |")
    lines.extend([
        "",
        "## Interpretation constraints",
        "",
        "- The policy is a selective decision layer, so higher accepted-sample accuracy at lower coverage is not an all-sample accuracy improvement.",
        "- † For pre-temporal methods, this column counts positive windows under a deliberately naive dispatch-on-every-window baseline. For the full policy it counts debounced notification events. The reduction therefore measures operational coalescing plus rejection, not a like-for-like classifier false-positive reduction.",
        "- Subject-level heterogeneity is material: a subject rejected almost entirely by the OOD gate demonstrates safe abstention but also poor usability, and must not be hidden by pooled metrics.",
        "- Window timestamps were unavailable in the processed JSON; retained chronological rows were treated as one-second steps for alert-rate normalization.",
        "- Raw-signal quality metrics cannot be reconstructed from the 12-feature JSON and were therefore not evaluated here.",
        "- Results apply to the frozen LOSO epoch-49 checkpoints and the documented thresholds; threshold tuning on evaluation subjects was not performed.",
        "",
    ])
    return "\n".join(lines)


def run(args: argparse.Namespace) -> dict[str, Any]:
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    model_repo = Path(args.model_repo).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    requested_models = tuple(item.strip() for item in args.models.split(",") if item.strip())
    if not requested_models or set(requested_models) - {"attention", "dnn"}:
        raise ValueError("--models must contain attention and/or dnn")

    data_files = {subject: model_repo / "Data_Processed" / f"WESADECG_{subject}.json" for subject in SUBJECTS}
    for path in data_files.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    data = {subject: load_subject(path) for subject, path in data_files.items()}
    features = {subject: value[0] for subject, value in data.items()}
    labels = {subject: value[1] for subject, value in data.items()}

    oof: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    checkpoint_files: dict[str, dict[str, Path]] = {}
    for model_name in requested_models:
        oof[model_name] = {}
        checkpoint_files[model_name] = {}
        for fold, subject in enumerate(SUBJECTS):
            checkpoint = checkpoint_path(model_repo, fold, model_name, args.epoch)
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            print(f"[{model_name}] OOF inference {subject} using {checkpoint.name}", flush=True)
            probabilities = predict(
                model_name, checkpoint, features[subject], device=device, batch_size=args.batch_size
            )
            oof[model_name][subject] = {"labels": labels[subject], "probabilities": probabilities}
            checkpoint_files[model_name][subject] = checkpoint
    write_oof_csv(output_dir / "oof_predictions.csv", oof)

    summaries: dict[str, Any] = {}
    per_subject_rows: list[dict[str, Any]] = []
    corruption_audits: dict[str, Any] = {}
    split_manifest: dict[str, Any] = {}
    for model_name in requested_models:
        model_probabilities = {subject: oof[model_name][subject]["probabilities"] for subject in SUBJECTS}
        arrays_by_subject: dict[str, Mapping[str, np.ndarray]] = {}
        model_rows: list[dict[str, Any]] = []
        model_corruption: list[dict[str, Any]] = []
        for subject in SUBJECTS:
            reference_subjects, calibration_subjects = deterministic_policy_split(subject, args.seed)
            split_manifest[f"{model_name}:{subject}"] = {
                "reference_subjects": reference_subjects,
                "calibration_subjects": calibration_subjects,
                "evaluation_subject": subject,
            }
            print(f"[{model_name}] policy outer fold {subject}", flush=True)
            policy = fit_policy(
                model_name, subject, reference_subjects, calibration_subjects,
                features, labels, model_probabilities,
                alpha=args.alpha,
                minimum_confidence=args.minimum_confidence,
                ood_quantile=args.ood_quantile,
                seed=args.seed,
            )
            policy.save(output_dir / "policies" / model_name / f"{subject}.json")
            methods, diagnostics, arrays = evaluate_subject(
                model_name, subject, policy, features[subject], labels[subject],
                model_probabilities[subject], window_seconds=args.window_seconds,
            )
            row = {
                "model": model_name,
                "subject": subject,
                "reference_subjects": reference_subjects,
                "calibration_subjects": calibration_subjects,
                "methods": methods,
                "diagnostics": diagnostics,
            }
            model_rows.append(row)
            per_subject_rows.append(row)
            arrays_by_subject[subject] = arrays
            model_corruption.append(corruption_audit(policy, features[subject], model_probabilities[subject]))
        summaries[model_name] = aggregate_model(
            model_name, model_rows, arrays_by_subject, window_seconds=args.window_seconds
        )
        corruption_audits[model_name] = {
            "rows_per_corruption": sum(item["rows_per_corruption"] for item in model_corruption),
            "rejected": {
                name: sum(item["rejected"][name] for item in model_corruption)
                for name in ("nonfinite", "extreme", "feature_order")
            },
        }
        total = corruption_audits[model_name]["rows_per_corruption"]
        corruption_audits[model_name]["rejection_rates"] = {
            name: value / total for name, value in corruption_audits[model_name]["rejected"].items()
        }

    source_manifest = {
        "data": {subject: {"path": str(path), "sha256": sha256_file(path)} for subject, path in data_files.items()},
        "checkpoints": {
            model_name: {
                subject: {"path": str(path), "sha256": sha256_file(path)}
                for subject, path in paths.items()
            }
            for model_name, paths in checkpoint_files.items()
        },
    }
    payload: dict[str, Any] = {
        "schema_version": "wesad-policy-experiment-v1",
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "device": str(device),
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
        "config": {
            "subjects": list(SUBJECTS),
            "models": list(requested_models),
            "checkpoint_epoch": args.epoch,
            "seed": args.seed,
            "conformal_alpha": args.alpha,
            "minimum_confidence": args.minimum_confidence,
            "ood_quantile": args.ood_quantile,
            "window_seconds": args.window_seconds,
        },
        "policy_splits": split_manifest,
        "models": summaries,
        "corruption_audits": corruption_audits,
        "source_manifest": source_manifest,
    }
    atomic_json(output_dir / "summary.json", payload)
    write_per_subject_csv(output_dir / "per_subject.csv", per_subject_rows)
    generate_plots(output_dir, summaries)
    (output_dir / "report.md").write_text(markdown_report(payload), encoding="utf-8")
    return payload


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Run real WESAD LOSO post-DNN reliability ablations")
    value.add_argument("--model-repo", default=r"D:\NUS\BMI5101\smart-stress-model")
    value.add_argument("--output-dir", default=str(REPO_ROOT / "reports" / "wesad_policy_reliability"))
    value.add_argument("--models", default="attention,dnn")
    value.add_argument("--epoch", type=int, default=49)
    value.add_argument("--batch-size", type=int, default=4096)
    value.add_argument("--device")
    value.add_argument("--seed", type=int, default=5101)
    value.add_argument("--alpha", type=float, default=0.10)
    value.add_argument("--minimum-confidence", type=float, default=0.60)
    value.add_argument("--ood-quantile", type=float, default=0.99)
    value.add_argument("--window-seconds", type=float, default=1.0)
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    result = run(args)
    for model_name, summary in result["models"].items():
        effect = summary["policy_effect"]
        print(
            f"{model_name}: coverage={summary['methods']['full_temporal']['coverage']:.4f} "
            f"risk_delta={effect['accepted_risk_change']:+.4f} "
            f"naive_false_dispatch_reduction={effect['naive_per_window_false_dispatch_reduction']:.2%}"
        )
    print(Path(args.output_dir).resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
