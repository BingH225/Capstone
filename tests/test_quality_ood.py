from __future__ import annotations

import numpy as np

from smartstress_policy_reliability import FeatureQualityProfile, RobustDiagonalOODDetector
from smartstress_policy_reliability.contracts import ReasonCode


def test_quality_rejects_bad_shape_nonfinite_external_and_extreme(reference_matrix) -> None:
    profile = FeatureQualityProfile.fit(reference_matrix)
    assert ReasonCode.FEATURE_COUNT_MISMATCH.value in profile.evaluate([0.0]).reasons
    nonfinite = np.zeros(12)
    nonfinite[3] = np.nan
    assert ReasonCode.NONFINITE_FEATURE.value in profile.evaluate(nonfinite).reasons
    assert ReasonCode.EXTERNAL_QUALITY_LOW.value in profile.evaluate(
        np.zeros(12), external_quality_score="bad"
    ).reasons
    assert ReasonCode.FEATURE_OUTLIER_EXTREME.value in profile.evaluate(
        np.full(12, 100.0)
    ).reasons
    signal = profile.evaluate(
        np.zeros(12), signal_quality={"window_seconds": 10, "sampling_rate_hz": 700}
    )
    assert not signal.valid
    assert ReasonCode.SIGNAL_WINDOW_TOO_SHORT.value in signal.reasons


def test_ood_detector_and_round_trip(reference_matrix) -> None:
    detector = RobustDiagonalOODDetector.fit(reference_matrix, threshold_quantile=0.95)
    assert not detector.evaluate(np.median(reference_matrix, axis=0)).is_ood
    assert detector.evaluate(np.full(12, 20.0)).is_ood
    restored = RobustDiagonalOODDetector.from_dict(detector.to_dict())
    assert restored.score(reference_matrix[0]) == detector.score(reference_matrix[0])
