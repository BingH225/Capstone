from __future__ import annotations

import numpy as np
import pytest

from smartstress_policy_reliability import (
    BinarySelectivePredictor,
    FEATURE_NAMES,
    FeatureQualityProfile,
    IdentityCalibrator,
    ReliabilityPolicy,
    ReliabilityPolicyConfig,
    RobustDiagonalOODDetector,
    TemporalPolicyConfig,
)


@pytest.fixture
def reference_matrix() -> np.ndarray:
    rng = np.random.default_rng(5101)
    return rng.normal(0.0, 1.0, size=(240, len(FEATURE_NAMES)))


@pytest.fixture
def deterministic_policy(reference_matrix: np.ndarray) -> ReliabilityPolicy:
    config = ReliabilityPolicyConfig(
        calibrator="identity",
        minimum_confidence=0.6,
        quality_hard_z_limit=30.0,
        temporal=TemporalPolicyConfig(
            window_size=3,
            required_elevated=2,
            threshold_on=0.65,
            threshold_off=0.45,
            cooldown_seconds=60,
        ),
    )
    detector = RobustDiagonalOODDetector.fit(reference_matrix)
    # Use an explicit threshold to make policy precedence tests deterministic.
    detector = RobustDiagonalOODDetector(
        feature_names=detector.feature_names,
        center=detector.center,
        scale=detector.scale,
        threshold=6.0,
        threshold_quantile=detector.threshold_quantile,
    )
    return ReliabilityPolicy(
        config=config,
        calibrator=IdentityCalibrator(),
        quality_profile=FeatureQualityProfile.fit(
            reference_matrix, hard_z_limit=30.0
        ),
        ood_detector=detector,
        selective_predictor=BinarySelectivePredictor(
            alpha=0.1, nonconformity_quantile=0.4, minimum_confidence=0.6
        ),
        fitted_at="2026-08-29T00:00:00Z",
        data_provenance={"fixture": True},
    )
