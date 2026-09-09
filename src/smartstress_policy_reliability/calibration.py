"""Binary post-hoc calibrators and calibration metrics."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Mapping

import numpy as np


EPSILON = 1e-7


def clip_probabilities(values: Iterable[float] | np.ndarray) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=np.float64), EPSILON, 1.0 - EPSILON)


def probabilities_to_logits(values: Iterable[float] | np.ndarray) -> np.ndarray:
    probabilities = clip_probabilities(values)
    return np.log(probabilities) - np.log1p(-probabilities)


def sigmoid(values: Iterable[float] | np.ndarray) -> np.ndarray:
    logits = np.asarray(values, dtype=np.float64)
    output = np.empty_like(logits)
    positive = logits >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-logits[positive]))
    exp_values = np.exp(logits[~positive])
    output[~positive] = exp_values / (1.0 + exp_values)
    return output


def _validated_labels(labels: Iterable[int] | np.ndarray) -> np.ndarray:
    result = np.asarray(labels, dtype=np.int64)
    if result.ndim != 1 or result.size < 2:
        raise ValueError("labels must be a one-dimensional array with at least two rows")
    if not np.isin(result, [0, 1]).all():
        raise ValueError("labels must contain only 0 and 1")
    if np.unique(result).size < 2:
        raise ValueError("calibration requires both classes")
    return result


class BinaryCalibrator:
    kind = "base"

    def predict_from_logits(self, logits: Iterable[float] | np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def predict_from_probabilities(
        self, probabilities: Iterable[float] | np.ndarray
    ) -> np.ndarray:
        return self.predict_from_logits(probabilities_to_logits(probabilities))

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError


@dataclass(frozen=True)
class IdentityCalibrator(BinaryCalibrator):
    kind = "identity"

    @classmethod
    def fit(cls, logits: Iterable[float], labels: Iterable[int]) -> "IdentityCalibrator":
        logits_array = np.asarray(logits, dtype=np.float64)
        labels_array = _validated_labels(labels)
        if logits_array.shape != labels_array.shape or not np.isfinite(logits_array).all():
            raise ValueError("logits must be finite and aligned with labels")
        return cls()

    def predict_from_logits(self, logits: Iterable[float] | np.ndarray) -> np.ndarray:
        return sigmoid(logits)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind}


@dataclass(frozen=True)
class TemperatureScaler(BinaryCalibrator):
    temperature: float
    kind = "temperature"

    @classmethod
    def fit(
        cls,
        logits: Iterable[float] | np.ndarray,
        labels: Iterable[int] | np.ndarray,
    ) -> "TemperatureScaler":
        logits_array = np.asarray(logits, dtype=np.float64)
        labels_array = _validated_labels(labels)
        if logits_array.shape != labels_array.shape or not np.isfinite(logits_array).all():
            raise ValueError("logits must be finite and aligned with labels")

        def objective(log_temperature: float) -> float:
            probabilities = sigmoid(logits_array / math.exp(log_temperature))
            return negative_log_likelihood(labels_array, probabilities)

        left, right = math.log(0.05), math.log(20.0)
        ratio = (math.sqrt(5.0) - 1.0) / 2.0
        c = right - ratio * (right - left)
        d = left + ratio * (right - left)
        fc, fd = objective(c), objective(d)
        for _ in range(100):
            if fc < fd:
                right, d, fd = d, c, fc
                c = right - ratio * (right - left)
                fc = objective(c)
            else:
                left, c, fc = c, d, fd
                d = left + ratio * (right - left)
                fd = objective(d)
        return cls(temperature=float(math.exp((left + right) / 2.0)))

    def predict_from_logits(self, logits: Iterable[float] | np.ndarray) -> np.ndarray:
        return sigmoid(np.asarray(logits, dtype=np.float64) / self.temperature)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "temperature": self.temperature}


@dataclass(frozen=True)
class PlattScaler(BinaryCalibrator):
    slope: float
    intercept: float
    kind = "platt"

    @classmethod
    def fit(
        cls,
        logits: Iterable[float] | np.ndarray,
        labels: Iterable[int] | np.ndarray,
        l2: float = 1e-4,
    ) -> "PlattScaler":
        x = np.asarray(logits, dtype=np.float64)
        y = _validated_labels(labels).astype(np.float64)
        if x.shape != y.shape or not np.isfinite(x).all():
            raise ValueError("logits must be finite and aligned with labels")
        slope, intercept = 1.0, 0.0
        for _ in range(100):
            probabilities = sigmoid(slope * x + intercept)
            residual = probabilities - y
            weights = np.maximum(probabilities * (1.0 - probabilities), 1e-9)
            gradient = np.array(
                [np.sum(residual * x) + l2 * slope, np.sum(residual)],
                dtype=np.float64,
            )
            hessian = np.array(
                [
                    [np.sum(weights * x * x) + l2, np.sum(weights * x)],
                    [np.sum(weights * x), np.sum(weights)],
                ],
                dtype=np.float64,
            )
            step = np.linalg.solve(hessian, gradient)
            slope -= float(step[0])
            intercept -= float(step[1])
            if float(np.linalg.norm(step)) < 1e-9:
                break
        return cls(slope=slope, intercept=intercept)

    def predict_from_logits(self, logits: Iterable[float] | np.ndarray) -> np.ndarray:
        values = np.asarray(logits, dtype=np.float64)
        return sigmoid(self.slope * values + self.intercept)

    def to_dict(self) -> dict[str, Any]:
        return {"kind": self.kind, "slope": self.slope, "intercept": self.intercept}


def calibrator_from_dict(payload: Mapping[str, Any]) -> BinaryCalibrator:
    kind = payload.get("kind")
    if kind == "identity":
        return IdentityCalibrator()
    if kind == "temperature":
        return TemperatureScaler(temperature=float(payload["temperature"]))
    if kind == "platt":
        return PlattScaler(
            slope=float(payload["slope"]), intercept=float(payload["intercept"])
        )
    raise ValueError(f"Unsupported calibrator kind: {kind!r}")


def fit_calibrator(
    kind: str,
    logits: Iterable[float] | np.ndarray,
    labels: Iterable[int] | np.ndarray,
) -> BinaryCalibrator:
    normalized = kind.strip().lower()
    if normalized == "identity":
        return IdentityCalibrator.fit(logits, labels)
    if normalized == "temperature":
        return TemperatureScaler.fit(logits, labels)
    if normalized == "platt":
        return PlattScaler.fit(logits, labels)
    raise ValueError("calibrator must be one of: identity, temperature, platt")


def negative_log_likelihood(labels: Iterable[int], probabilities: Iterable[float]) -> float:
    y = np.asarray(labels, dtype=np.float64)
    p = clip_probabilities(probabilities)
    if y.shape != p.shape:
        raise ValueError("labels and probabilities must have the same shape")
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log1p(-p)))


def brier_score(labels: Iterable[int], probabilities: Iterable[float]) -> float:
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.shape != p.shape:
        raise ValueError("labels and probabilities must have the same shape")
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    labels: Iterable[int], probabilities: Iterable[float], bins: int = 15
) -> float:
    if bins < 2:
        raise ValueError("bins must be at least 2")
    y = np.asarray(labels, dtype=np.int64)
    p = np.asarray(probabilities, dtype=np.float64)
    if y.shape != p.shape or p.size == 0:
        raise ValueError("labels and probabilities must be non-empty and aligned")
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.minimum(np.digitize(p, edges[1:-1], right=True), bins - 1)
    error = 0.0
    for index in range(bins):
        mask = indices == index
        if not np.any(mask):
            continue
        error += float(np.mean(mask)) * abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))
    return error


def calibration_metrics(labels: Iterable[int], probabilities: Iterable[float]) -> dict[str, float]:
    return {
        "nll": negative_log_likelihood(labels, probabilities),
        "brier": brier_score(labels, probabilities),
        "ece_15": expected_calibration_error(labels, probabilities, bins=15),
    }


def calibration_diagnostics(
    labels: Iterable[int], probabilities: Iterable[float], bins: int = 10
) -> dict[str, Any]:
    """Return scalar calibration metrics plus slope/intercept and plot-ready bins."""
    y = np.asarray(list(labels), dtype=np.int64)
    if y.ndim != 1 or y.size < 2 or not np.isin(y, [0, 1]).all():
        raise ValueError("labels must be an aligned binary vector with at least two rows")
    p = np.asarray(list(probabilities), dtype=np.float64)
    if p.shape != y.shape or not np.isfinite(p).all() or np.any((p < 0) | (p > 1)):
        raise ValueError("probabilities must be finite, within [0, 1], and aligned")
    result: dict[str, Any] = dict(calibration_metrics(y, p))
    if np.unique(y).size == 2:
        platt = PlattScaler.fit(probabilities_to_logits(p), y)
        result["calibration_slope"] = platt.slope
        result["calibration_intercept"] = platt.intercept
    else:
        result["calibration_slope"] = None
        result["calibration_intercept"] = None
    edges = np.linspace(0.0, 1.0, bins + 1)
    indices = np.minimum(np.digitize(p, edges[1:-1], right=True), bins - 1)
    diagram = []
    for index in range(bins):
        mask = indices == index
        if np.any(mask):
            diagram.append(
                {
                    "bin_lower": float(edges[index]),
                    "bin_upper": float(edges[index + 1]),
                    "count": int(np.sum(mask)),
                    "mean_confidence": float(np.mean(p[mask])),
                    "empirical_rate": float(np.mean(y[mask])),
                }
            )
    result["reliability_diagram_bins"] = diagram
    return result
