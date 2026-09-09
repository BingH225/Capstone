from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from smartstress_policy_reliability import BinarySelectivePredictor, TemporalPolicy, TemporalPolicyConfig
from smartstress_policy_reliability.contracts import ReliabilityState


def test_conformal_singleton_and_abstention() -> None:
    probabilities = np.array([0.05] * 20 + [0.95] * 20)
    labels = np.array([0] * 20 + [1] * 20)
    predictor = BinarySelectivePredictor.fit(probabilities, labels, alpha=0.1)
    assert predictor.evaluate(0.95).prediction_set == ("stress",)
    assert predictor.evaluate(0.05).prediction_set == ("nonstress",)
    assert not predictor.evaluate(0.5).accepted


def test_temporal_k_of_m_hysteresis_cooldown_and_session_isolation() -> None:
    policy = TemporalPolicy(
        TemporalPolicyConfig(
            window_size=3,
            required_elevated=2,
            threshold_on=0.65,
            threshold_off=0.45,
            cooldown_seconds=60,
        )
    )
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    assert policy.evaluate(session_id="a", probability=0.8, timestamp=start).state == ReliabilityState.MONITOR
    active = policy.evaluate(
        session_id="a", probability=0.9, timestamp=start + timedelta(seconds=1)
    )
    assert active.state == ReliabilityState.RELIABLE_ELEVATED
    assert active.proactive_notification
    cooldown = policy.evaluate(
        session_id="a", probability=0.7, timestamp=start + timedelta(seconds=2)
    )
    assert cooldown.state == ReliabilityState.RELIABLE_ELEVATED
    assert not cooldown.proactive_notification
    assert policy.evaluate(
        session_id="a", probability=0.4, timestamp=start + timedelta(seconds=3)
    ).state == ReliabilityState.RELIABLE_LOW
    assert policy.evaluate(
        session_id="b", probability=0.8, timestamp=start
    ).state == ReliabilityState.MONITOR
    assert policy.evaluate(
        session_id="c", probability=0.55, timestamp=start
    ).state == ReliabilityState.MONITOR
    with pytest.raises(ValueError, match="non-decreasing"):
        policy.evaluate(session_id="a", probability=0.8, timestamp=start)
