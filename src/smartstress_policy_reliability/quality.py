"""Feature-level quality screening for the frozen 12-feature DNN contract."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .contracts import FEATURE_NAMES, QualityResult, ReasonCode


@dataclass(frozen=True)
class FeatureQualityProfile:
    """Robust reference profile used to reject malformed or extreme inputs."""

    feature_names: tuple[str, ...]
    median: tuple[float, ...]
    robust_scale: tuple[float, ...]
    soft_z_limit: float = 6.0
    hard_z_limit: float = 12.0
    minimum_external_quality: float = 0.5
    minimum_window_seconds: float = 20.0
    minimum_rpeak_success: float = 0.8
    maximum_abnormal_rr_fraction: float = 0.2
    minimum_valid_fraction: float = 0.8
    maximum_flatline_fraction: float = 0.05
    maximum_saturation_fraction: float = 0.05

    @classmethod
    def fit(
        cls,
        reference_features: Iterable[Sequence[float]],
        *,
        feature_names: Sequence[str] = FEATURE_NAMES,
        soft_z_limit: float = 6.0,
        hard_z_limit: float = 12.0,
        minimum_external_quality: float = 0.5,
    ) -> "FeatureQualityProfile":
        matrix = np.asarray(list(reference_features), dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[1] != len(feature_names):
            raise ValueError(
                f"reference_features must have shape [n, {len(feature_names)}]"
            )
        if matrix.shape[0] < 10:
            raise ValueError("at least 10 reference rows are required")
        if not np.isfinite(matrix).all():
            raise ValueError("reference_features must contain only finite values")
        if not 0.0 <= minimum_external_quality <= 1.0:
            raise ValueError("minimum_external_quality must be within [0, 1]")
        if not 0.0 < soft_z_limit < hard_z_limit:
            raise ValueError("require 0 < soft_z_limit < hard_z_limit")
        median = np.median(matrix, axis=0)
        mad = np.median(np.abs(matrix - median), axis=0)
        fallback = np.std(matrix, axis=0, ddof=1)
        robust_scale = 1.4826 * mad
        robust_scale = np.where(robust_scale > 1e-8, robust_scale, fallback)
        robust_scale = np.where(robust_scale > 1e-8, robust_scale, 1.0)
        return cls(
            feature_names=tuple(feature_names),
            median=tuple(float(value) for value in median),
            robust_scale=tuple(float(value) for value in robust_scale),
            soft_z_limit=float(soft_z_limit),
            hard_z_limit=float(hard_z_limit),
            minimum_external_quality=float(minimum_external_quality),
        )

    def evaluate(
        self,
        features: Sequence[float],
        *,
        external_quality_score: float | None = None,
        signal_quality: Mapping[str, float] | None = None,
    ) -> QualityResult:
        reasons: list[str] = []
        try:
            feature_count = len(features)
        except TypeError:
            feature_count = -1
        if feature_count != len(self.feature_names):
            return QualityResult(
                valid=False,
                score=0.0,
                max_robust_z=None,
                reasons=(ReasonCode.FEATURE_COUNT_MISMATCH.value,),
            )
        try:
            values = np.asarray(features, dtype=np.float64)
        except (TypeError, ValueError):
            return QualityResult(
                valid=False,
                score=0.0,
                max_robust_z=None,
                reasons=(ReasonCode.NONFINITE_FEATURE.value,),
            )
        if not np.isfinite(values).all():
            return QualityResult(
                valid=False,
                score=0.0,
                max_robust_z=None,
                reasons=(ReasonCode.NONFINITE_FEATURE.value,),
            )
        if external_quality_score is not None:
            try:
                external = float(external_quality_score)
            except (TypeError, ValueError):
                external = float("nan")
            if not np.isfinite(external) or not 0.0 <= external <= 1.0:
                return QualityResult(
                    valid=False,
                    score=0.0,
                    max_robust_z=None,
                    reasons=(ReasonCode.EXTERNAL_QUALITY_LOW.value,),
                )
            if external < self.minimum_external_quality:
                return QualityResult(
                    valid=False,
                    score=external,
                    max_robust_z=None,
                    reasons=(ReasonCode.EXTERNAL_QUALITY_LOW.value,),
                )
        else:
            external = 1.0

        signal_score = 1.0
        if signal_quality:
            try:
                evidence = {key: float(value) for key, value in signal_quality.items()}
            except (AttributeError, TypeError, ValueError):
                evidence = {"window_seconds": float("nan")}
            if not all(np.isfinite(value) for value in evidence.values()):
                return QualityResult(
                    valid=False,
                    score=0.0,
                    max_robust_z=None,
                    reasons=(ReasonCode.SIGNAL_QUALITY_INVALID.value,),
                )
            checks = (
                ("window_seconds", lambda value: value >= self.minimum_window_seconds, ReasonCode.SIGNAL_WINDOW_TOO_SHORT),
                ("sampling_rate_hz", lambda value: value > 0.0, ReasonCode.SAMPLING_RATE_INVALID),
                ("rpeak_success", lambda value: value >= self.minimum_rpeak_success, ReasonCode.RPEAK_QUALITY_LOW),
                ("abnormal_rr_fraction", lambda value: 0.0 <= value <= self.maximum_abnormal_rr_fraction, ReasonCode.ABNORMAL_RR_HIGH),
                ("valid_fraction", lambda value: self.minimum_valid_fraction <= value <= 1.0, ReasonCode.VALID_SEGMENT_LOW),
                ("flatline_fraction", lambda value: 0.0 <= value <= self.maximum_flatline_fraction, ReasonCode.FLATLINE_HIGH),
                ("saturation_fraction", lambda value: 0.0 <= value <= self.maximum_saturation_fraction, ReasonCode.SATURATION_HIGH),
            )
            failed = [reason.value for key, predicate, reason in checks if key in evidence and not predicate(evidence[key])]
            if failed:
                return QualityResult(
                    valid=False,
                    score=0.0,
                    max_robust_z=None,
                    reasons=tuple(failed),
                )
            bounded_values = [
                evidence[key]
                for key in ("rpeak_success", "valid_fraction")
                if key in evidence
            ]
            signal_score = min(bounded_values, default=1.0)

        median = np.asarray(self.median, dtype=np.float64)
        scale = np.asarray(self.robust_scale, dtype=np.float64)
        robust_z = np.abs(values - median) / scale
        max_z = float(np.max(robust_z))
        if max_z >= self.hard_z_limit:
            reasons.append(ReasonCode.FEATURE_OUTLIER_EXTREME.value)
            return QualityResult(
                valid=False,
                score=0.0,
                max_robust_z=max_z,
                reasons=tuple(reasons),
            )

        # A smooth score avoids pretending that all in-range windows are equally clean.
        feature_score = float(np.exp(-0.5 * (max_z / self.soft_z_limit) ** 2))
        score = float(np.clip(min(feature_score, external, signal_score), 0.0, 1.0))
        reasons.append(ReasonCode.QUALITY_OK.value)
        return QualityResult(
            valid=True,
            score=score,
            max_robust_z=max_z,
            reasons=tuple(reasons),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "feature_names": list(self.feature_names),
            "median": list(self.median),
            "robust_scale": list(self.robust_scale),
            "soft_z_limit": self.soft_z_limit,
            "hard_z_limit": self.hard_z_limit,
            "minimum_external_quality": self.minimum_external_quality,
            "minimum_window_seconds": self.minimum_window_seconds,
            "minimum_rpeak_success": self.minimum_rpeak_success,
            "maximum_abnormal_rr_fraction": self.maximum_abnormal_rr_fraction,
            "minimum_valid_fraction": self.minimum_valid_fraction,
            "maximum_flatline_fraction": self.maximum_flatline_fraction,
            "maximum_saturation_fraction": self.maximum_saturation_fraction,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureQualityProfile":
        return cls(
            feature_names=tuple(str(value) for value in payload["feature_names"]),
            median=tuple(float(value) for value in payload["median"]),
            robust_scale=tuple(float(value) for value in payload["robust_scale"]),
            soft_z_limit=float(payload["soft_z_limit"]),
            hard_z_limit=float(payload["hard_z_limit"]),
            minimum_external_quality=float(payload["minimum_external_quality"]),
            minimum_window_seconds=float(payload.get("minimum_window_seconds", 20.0)),
            minimum_rpeak_success=float(payload.get("minimum_rpeak_success", 0.8)),
            maximum_abnormal_rr_fraction=float(payload.get("maximum_abnormal_rr_fraction", 0.2)),
            minimum_valid_fraction=float(payload.get("minimum_valid_fraction", 0.8)),
            maximum_flatline_fraction=float(payload.get("maximum_flatline_fraction", 0.05)),
            maximum_saturation_fraction=float(payload.get("maximum_saturation_fraction", 0.05)),
        )
