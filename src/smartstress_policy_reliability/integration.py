"""Adapters for the existing SmartStress PhysioSense state contract."""

from __future__ import annotations

from typing import Any, Mapping

from .contracts import FEATURE_NAMES, InferenceInput, ReliabilityState
from .policy import ReliabilityPolicy


def inference_from_physio_state(
    state: Mapping[str, Any], *, session_id: str | None = None
) -> InferenceInput:
    timestamps = state.get("stress_timestamps") or []
    timestamp = state.get("physio_timestamp") or (timestamps[-1] if timestamps else None)
    return InferenceInput(
        features=tuple(state.get("physio_features") or ()),
        feature_names=tuple(state.get("physio_feature_names") or FEATURE_NAMES),
        raw_probability=state.get("current_stress_prob"),
        timestamp=timestamp,
        model_id=str(state.get("physio_model_id") or "wesad_attention_v1"),
        session_id=session_id or str(state.get("session_id") or "default"),
        external_quality_score=state.get("physio_quality_score"),
        input_source=state.get("physio_input_source"),
        baseline_version=state.get("physio_baseline_version"),
        signal_quality=dict(state.get("physio_signal_quality") or {}),
        top_drivers=tuple(state.get("physio_top_drivers") or ()),
        metadata={
            "adapter": "physiosense_state_v1",
        },
    )


def apply_reliability_to_physio_state(
    state: Mapping[str, Any],
    policy: ReliabilityPolicy,
    *,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Return LangGraph-compatible updates without mutating the input state."""
    decision = policy.evaluate(inference_from_physio_state(state, session_id=session_id))
    reliable_elevated = decision.reliability_state == ReliabilityState.RELIABLE_ELEVATED
    return {
        "calibrated_stress_prob": decision.calibrated_probability,
        "stress_detected": reliable_elevated,
        "physio_reliability": decision.to_dict(),
        "physio_reliability_state": decision.reliability_state.value,
        "physio_allowed_actions": [action.value for action in decision.allowed_actions],
        "physio_policy_version": decision.policy_version,
        "physio_audit_event": decision.audit_event(),
    }
