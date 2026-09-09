"""Metrics for calibration, selective prediction, and final policy decisions."""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Sequence

import numpy as np

from .calibration import calibration_metrics
from .contracts import ReliabilityDecision, ReliabilityState


def discrimination_metrics(labels: Iterable[int], probabilities: Iterable[float]) -> dict[str, float]:
    y = np.asarray(list(labels), dtype=np.int64)
    p = np.asarray(list(probabilities), dtype=np.float64)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("labels and probabilities must be non-empty and aligned")
    predicted = (p >= 0.5).astype(np.int64)
    tp = int(np.sum((predicted == 1) & (y == 1)))
    tn = int(np.sum((predicted == 0) & (y == 0)))
    fp = int(np.sum((predicted == 1) & (y == 0)))
    fn = int(np.sum((predicted == 0) & (y == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    positives = p[y == 1]
    negatives = p[y == 0]
    if positives.size and negatives.size:
        comparisons = positives[:, None] - negatives[None, :]
        auroc = float(np.mean(comparisons > 0) + 0.5 * np.mean(comparisons == 0))
    else:
        auroc = 0.0
    return {
        "accuracy": (tp + tn) / y.size,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auroc": auroc,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def policy_metrics(
    labels: Sequence[int], decisions: Sequence[ReliabilityDecision]
) -> dict[str, object]:
    if len(labels) != len(decisions) or not labels:
        raise ValueError("labels and decisions must be non-empty and aligned")
    state_counts: dict[str, int] = {}
    accepted_indices: list[int] = []
    accepted_predictions: list[int] = []
    calibrated_indices: list[int] = []
    calibrated_probabilities: list[float] = []
    conformal_hits = 0
    conformal_rows = 0
    singleton_rows = 0
    proactive_notifications = 0
    false_proactive_notifications = 0
    session_times: dict[str, list[datetime]] = {}
    for index, decision in enumerate(decisions):
        key = decision.reliability_state.value
        state_counts[key] = state_counts.get(key, 0) + 1
        if decision.calibrated_probability is not None:
            calibrated_indices.append(index)
            calibrated_probabilities.append(decision.calibrated_probability)
            conformal_rows += 1
            expected_label = "stress" if labels[index] == 1 else "nonstress"
            conformal_hits += int(expected_label in decision.prediction_set)
            singleton_rows += int(len(decision.prediction_set) == 1)
        if decision.proactive_notification:
            proactive_notifications += 1
            false_proactive_notifications += int(labels[index] == 0)
        try:
            timestamp = datetime.fromisoformat(decision.timestamp.replace("Z", "+00:00"))
            session_times.setdefault(decision.session_id, []).append(timestamp)
        except ValueError:
            pass
        if decision.reliability_state in {
            ReliabilityState.RELIABLE_LOW,
            ReliabilityState.RELIABLE_ELEVATED,
        }:
            accepted_indices.append(index)
            accepted_predictions.append(
                int(decision.reliability_state == ReliabilityState.RELIABLE_ELEVATED)
            )
    coverage = len(accepted_indices) / len(labels)
    if accepted_indices:
        errors = [
            int(prediction != labels[index])
            for prediction, index in zip(accepted_predictions, accepted_indices)
        ]
        selective_risk = float(np.mean(errors))
    else:
        selective_risk = 0.0
    result: dict[str, object] = {
        "rows": len(labels),
        "state_counts": state_counts,
        "coverage": coverage,
        "selective_risk": selective_risk,
        "conformal_empirical_coverage": (
            conformal_hits / conformal_rows if conformal_rows else 0.0
        ),
        "conformal_singleton_rate": (
            singleton_rows / conformal_rows if conformal_rows else 0.0
        ),
        "proactive_notifications": proactive_notifications,
    }
    total_hours = sum(
        max((max(times) - min(times)).total_seconds(), 0.0) / 3600.0
        for times in session_times.values()
        if times
    )
    result["observation_hours"] = total_hours
    result["false_proactive_alerts_per_hour"] = (
        false_proactive_notifications / total_hours if total_hours > 0 else None
    )
    if calibrated_indices:
        calibrated_labels = [labels[index] for index in calibrated_indices]
        result["calibration"] = calibration_metrics(
            calibrated_labels, calibrated_probabilities
        )
        probabilities = np.asarray(calibrated_probabilities)
        accepted_labels = np.asarray([labels[index] for index in calibrated_indices])
        confidence = np.maximum(probabilities, 1.0 - probabilities)
        errors = ((probabilities >= 0.5).astype(int) != accepted_labels).astype(float)
        order = np.argsort(-confidence)
        cumulative_risk = np.cumsum(errors[order]) / np.arange(1, len(order) + 1)
        coverages = np.arange(1, len(order) + 1) / len(order)
        result["aurc"] = float(np.trapz(cumulative_risk, coverages))
    return result
