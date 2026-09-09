from __future__ import annotations

import numpy as np

from smartstress_policy_reliability.calibration import (
    PlattScaler,
    TemperatureScaler,
    calibrator_from_dict,
    calibration_metrics,
    negative_log_likelihood,
    sigmoid,
)


def test_temperature_scaling_improves_overconfident_predictions() -> None:
    logits = np.array([-8, -6, -4, -2, 2, 4, 6, 8] * 10, dtype=float)
    labels = np.array([0, 0, 0, 1, 0, 1, 1, 1] * 10)
    fitted = TemperatureScaler.fit(logits, labels)
    assert fitted.temperature > 1.0
    assert negative_log_likelihood(labels, fitted.predict_from_logits(logits)) < (
        negative_log_likelihood(labels, sigmoid(logits))
    )


def test_platt_serialization_round_trip() -> None:
    logits = np.linspace(-3.0, 3.0, 60)
    labels = (logits + np.sin(logits) > 0).astype(int)
    fitted = PlattScaler.fit(logits, labels)
    restored = calibrator_from_dict(fitted.to_dict())
    np.testing.assert_allclose(
        restored.predict_from_logits(logits), fitted.predict_from_logits(logits)
    )


def test_calibration_metrics_are_finite() -> None:
    result = calibration_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert set(result) == {"nll", "brier", "ece_15"}
    assert all(np.isfinite(value) and value >= 0 for value in result.values())
