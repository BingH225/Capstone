"""Interpretable out-of-distribution screening in the DNN feature space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import FEATURE_NAMES, OODResult, ReasonCode


@dataclass(frozen=True)
class RobustDiagonalOODDetector:
    """Robust diagonal Mahalanobis detector fitted only on reference-domain data."""

    feature_names: tuple[str, ...]
    center: tuple[float, ...]
    scale: tuple[float, ...]
    threshold: float
    threshold_quantile: float

    @classmethod
    def fit(
        cls,
        reference_features: Iterable[Sequence[float]],
        *,
        feature_names: Sequence[str] = FEATURE_NAMES,
        threshold_quantile: float = 0.99,
    ) -> "RobustDiagonalOODDetector":
        matrix = np.asarray(list(reference_features), dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(feature_names):
            raise ValueError(
                f"reference_features must have shape [n, {len(feature_names)}]"
            )
        if matrix.shape[0] < 20:
            raise ValueError("at least 20 reference rows are required for OOD fitting")
        if not np.isfinite(matrix).all():
            raise ValueError("reference_features must contain only finite values")
        if not 0.5 < threshold_quantile < 1.0:
            raise ValueError("threshold_quantile must be between 0.5 and 1.0")
        center = np.median(matrix, axis=0)
        mad = np.median(np.abs(matrix - center), axis=0)
        fallback = np.std(matrix, axis=0, ddof=1)
        scale = 1.4826 * mad
        scale = np.where(scale > 1e-8, scale, fallback)
        scale = np.where(scale > 1e-8, scale, 1.0)
        distances = np.sqrt(np.mean(((matrix - center) / scale) ** 2, axis=1))
        threshold = float(np.quantile(distances, threshold_quantile, method="higher"))
        return cls(
            feature_names=tuple(feature_names),
            center=tuple(float(value) for value in center),
            scale=tuple(float(value) for value in scale),
            threshold=threshold,
            threshold_quantile=float(threshold_quantile),
        )

    def score(self, features: Sequence[float]) -> float:
        if len(features) != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} features")
        values = np.asarray(features, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("features must contain only finite values")
        center = np.asarray(self.center, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        return float(np.sqrt(np.mean(((values - center) / scale) ** 2)))

    def evaluate(self, features: Sequence[float]) -> OODResult:
        score = self.score(features)
        is_ood = score > self.threshold
        return OODResult(
            is_ood=is_ood,
            score=score,
            threshold=self.threshold,
            reasons=(
                ReasonCode.OOD_HIGH.value if is_ood else ReasonCode.IN_DOMAIN.value,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "robust_diagonal_mahalanobis",
            "feature_names": list(self.feature_names),
            "center": list(self.center),
            "scale": list(self.scale),
            "threshold": self.threshold,
            "threshold_quantile": self.threshold_quantile,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RobustDiagonalOODDetector":
        if payload.get("kind") != "robust_diagonal_mahalanobis":
            raise ValueError("unsupported OOD detector kind")
        return cls(
            feature_names=tuple(str(value) for value in payload["feature_names"]),
            center=tuple(float(value) for value in payload["center"]),
            scale=tuple(float(value) for value in payload["scale"]),
            threshold=float(payload["threshold"]),
            threshold_quantile=float(payload["threshold_quantile"]),
        )
