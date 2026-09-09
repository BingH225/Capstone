from __future__ import annotations

import json

import numpy as np

from smartstress_policy_reliability import (
    Action,
    InferenceInput,
    ReliabilityPolicy,
    ReliabilityState,
    apply_reliability_to_physio_state,
)


def _input(probability: float, timestamp: str, **overrides) -> InferenceInput:
    payload = {
        "features": np.zeros(12),
        "raw_probability": probability,
        "timestamp": timestamp,
        "model_id": "wesad_attention_v1",
        "session_id": "s1",
    }
    payload.update(overrides)
    return InferenceInput(**payload)


def test_policy_precedence_and_fail_safe_contract(deterministic_policy) -> None:
    bad_model = deterministic_policy.evaluate(
        _input(0.9, "2026-01-01T00:00:00Z", model_id="wrong")
    )
    assert bad_model.reliability_state == ReliabilityState.DATA_INVALID
    assert Action.SUPPORT not in bad_model.allowed_actions
    bad_time = deterministic_policy.evaluate(_input(0.9, "not-a-time"))
    assert bad_time.reliability_state == ReliabilityState.DATA_INVALID
    bad_quality = deterministic_policy.evaluate(
        _input(0.9, "2026-01-01T00:00:00Z", external_quality_score=0.1)
    )
    assert bad_quality.reliability_state == ReliabilityState.DATA_INVALID
    bad_order = deterministic_policy.evaluate(
        _input(0.9, "2026-01-01T00:00:00Z", feature_names=tuple(reversed([
            "mean_hr", "std_hr", "tinn", "hrv_index", "nn50", "pnn50",
            "mean_hrv", "std_hrv", "rmssd", "fft_mean", "fft_std", "sum_psd",
        ])))
    )
    assert bad_order.reliability_state == ReliabilityState.DATA_INVALID
    ood = deterministic_policy.evaluate(
        _input(0.9, "2026-01-01T00:00:01Z", features=np.full(12, 10.0))
    )
    assert ood.reliability_state == ReliabilityState.OOD_OR_UNCERTAIN
    uncertain = deterministic_policy.evaluate(_input(0.5, "2026-01-01T00:00:02Z"))
    assert uncertain.reliability_state == ReliabilityState.OOD_OR_UNCERTAIN


def test_policy_temporal_actions_and_out_of_order(deterministic_policy) -> None:
    first = deterministic_policy.evaluate(_input(0.9, "2026-01-01T00:00:00Z"))
    assert first.reliability_state == ReliabilityState.MONITOR
    second = deterministic_policy.evaluate(_input(0.9, "2026-01-01T00:00:01Z"))
    assert second.reliability_state == ReliabilityState.RELIABLE_ELEVATED
    assert second.proactive_notification
    assert Action.PROPOSE_DRY_RUN in second.allowed_actions
    third = deterministic_policy.evaluate(_input(0.9, "2026-01-01T00:00:02Z"))
    assert third.reliability_state == ReliabilityState.RELIABLE_ELEVATED
    assert not third.proactive_notification
    invalid = deterministic_policy.evaluate(_input(0.9, "2025-12-31T23:59:59Z"))
    assert invalid.reliability_state == ReliabilityState.DATA_INVALID


def test_policy_hysteresis_band_is_monitor_not_reliable_low(deterministic_policy) -> None:
    decision = deterministic_policy.evaluate(_input(0.61, "2026-01-01T00:00:00Z"))
    assert decision.reliability_state == ReliabilityState.MONITOR


def test_manifest_round_trip_and_tamper_detection(deterministic_policy, tmp_path) -> None:
    path = deterministic_policy.save(tmp_path / "policy.json")
    restored = ReliabilityPolicy.load(path)
    assert restored.manifest() == deterministic_policy.manifest()
    envelope = json.loads(path.read_text(encoding="utf-8"))
    envelope["manifest"]["config"]["policy_version"] = "tampered"
    path.write_text(json.dumps(envelope), encoding="utf-8")
    try:
        ReliabilityPolicy.load(path)
    except RuntimeError as exc:
        assert "checksum" in str(exc)
    else:
        raise AssertionError("tampered manifest was accepted")


def test_physio_adapter_is_non_mutating_and_fail_safe(deterministic_policy) -> None:
    state = {
        "physio_features": [0.0] * 12,
        "current_stress_prob": 0.9,
        "physio_timestamp": "2026-01-01T00:00:00Z",
        "physio_model_id": "wesad_attention_v1",
        "session_id": "adapter",
        "physio_input_source": "ecg_upload",
        "physio_baseline_version": "baseline-v1",
        "physio_top_drivers": [{"feature": "rmssd", "direction": "up"}],
    }
    original = dict(state)
    updates = apply_reliability_to_physio_state(state, deterministic_policy)
    assert state == original
    assert not updates["stress_detected"]  # first high window is only MONITOR
    assert updates["physio_reliability_state"] == "MONITOR"
    assert updates["physio_audit_event"]["details"]["input_source"] == "ecg_upload"
    assert updates["physio_audit_event"]["details"]["baseline_version"] == "baseline-v1"


def test_session_rejects_baseline_version_mixing(deterministic_policy) -> None:
    first = deterministic_policy.evaluate(
        _input(0.2, "2026-01-01T00:00:00Z", baseline_version="baseline-v1")
    )
    assert first.reliability_state == ReliabilityState.RELIABLE_LOW
    mixed = deterministic_policy.evaluate(
        _input(0.2, "2026-01-01T00:00:01Z", baseline_version="baseline-v2")
    )
    assert mixed.reliability_state == ReliabilityState.DATA_INVALID
