"""Split-conformal and confidence-based selective classification."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import numpy as np

from .contracts import ReasonCode, SelectiveResult


@dataclass(frozen=True)
class BinarySelectivePredictor:
    """Binary split-conformal prediction set with an optional confidence floor."""

    alpha: float
    nonconformity_quantile: float
    minimum_confidence: float = 0.0

    @classmethod
    def fit(
        cls,
        calibrated_probabilities: Iterable[float],
        labels: Iterable[int],
        *,
        alpha: float = 0.1,
        minimum_confidence: float = 0.0,
    ) -> "BinarySelectivePredictor":
        probabilities = np.asarray(list(calibrated_probabilities), dtype=np.float64)
        targets = np.asarray(list(labels), dtype=np.int64)
        if probabilities.ndim != 1 or probabilities.size < 20:
            raise ValueError("at least 20 calibration probabilities are required")
        if targets.shape != probabilities.shape or not np.isin(targets, [0, 1]).all():
            raise ValueError("labels must be aligned binary values")
        if not np.isfinite(probabilities).all() or np.any((probabilities < 0) | (probabilities > 1)):
            raise ValueError("probabilities must be finite and within [0, 1]")
        if not 0.0 < alpha < 1.0:
            raise ValueError("alpha must be between 0 and 1")
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError("minimum_confidence must be within [0, 1]")
        true_class_probabilities = np.where(targets == 1, probabilities, 1.0 - probabilities)
        scores = 1.0 - true_class_probabilities
        n = scores.size
        rank = min(n, math.ceil((n + 1) * (1.0 - alpha)))
        quantile = float(np.partition(scores, rank - 1)[rank - 1])
        return cls(
            alpha=float(alpha),
            nonconformity_quantile=quantile,
            minimum_confidence=float(minimum_confidence),
        )

    def evaluate(self, probability: float) -> SelectiveResult:
        value = float(probability)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("probability must be finite and within [0, 1]")
        candidates: list[str] = []
        if 1.0 - (1.0 - value) <= self.nonconformity_quantile:
            candidates.append("nonstress")
        if 1.0 - value <= self.nonconformity_quantile:
            candidates.append("stress")
        confidence = max(value, 1.0 - value)
        reasons: list[str] = []
        if len(candidates) != 1:
            reasons.append(ReasonCode.CONFORMAL_AMBIGUOUS.value)
        if confidence < self.minimum_confidence:
            reasons.append(ReasonCode.CONFIDENCE_LOW.value)
        accepted = len(candidates) == 1 and confidence >= self.minimum_confidence
        if accepted:
            reasons.append(ReasonCode.SELECTIVE_SINGLETON.value)
        return SelectiveResult(
            prediction_set=tuple(candidates),
            confidence=confidence,
            accepted=accepted,
            reasons=tuple(reasons),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "binary_split_conformal",
            "alpha": self.alpha,
            "nonconformity_quantile": self.nonconformity_quantile,
            "minimum_confidence": self.minimum_confidence,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BinarySelectivePredictor":
        if payload.get("kind") != "binary_split_conformal":
            raise ValueError("unsupported selective predictor kind")
        return cls(
            alpha=float(payload["alpha"]),
            nonconformity_quantile=float(payload["nonconformity_quantile"]),
            minimum_confidence=float(payload["minimum_confidence"]),
        )
