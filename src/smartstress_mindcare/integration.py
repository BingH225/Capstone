"""Adapters for the existing SmartStress LangGraph state contract."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

from .contracts import Evidence, MindCareRequest, PhysioContext, PhysioReliabilityState, UserProfile, WrapperDecision
from .wrapper import MindCareReliabilityWrapper


def _message_content(value: Any) -> str:
    if isinstance(value, Mapping):
        return str(value.get("content", ""))
    return str(getattr(value, "content", ""))


def _latest_user_text(state: Mapping[str, Any]) -> str:
    for message in reversed(state.get("conversation_history") or []):
        role = message.get("role") if isinstance(message, Mapping) else getattr(message, "type", None)
        if role in {"user", "human"} or message.__class__.__name__ == "HumanMessage":
            content = _message_content(message).strip()
            if content:
                return content
    value = str(state.get("user_text") or "").strip()
    if not value:
        raise ValueError("MindCare requires a current user message")
    return value


def _physio_context(state: Mapping[str, Any]) -> PhysioContext:
    payload = state.get("physio_reliability")
    if not isinstance(payload, Mapping):
        return PhysioContext(
            reliability_state=PhysioReliabilityState.DATA_INVALID,
            reason_codes=("PHYSIO_RELIABILITY_MISSING",),
            allowed_actions=("retry_sensor", "continue_by_text"),
        )
    return PhysioContext.from_dict(payload)


def _evidence(state: Mapping[str, Any]) -> tuple[Evidence, ...]:
    structured = state.get("evidence_refs") or []
    if structured:
        return tuple(Evidence.from_dict(item) for item in structured)
    result = []
    for index, chunk in enumerate(state.get("rag_context") or []):
        text = str(chunk)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
        result.append(
            Evidence(
                evidence_id=f"legacy-rag:{digest}",
                source="legacy-rag-context",
                chunk=text,
                retrieval_score=0.5,
                license="runtime-source-metadata-unavailable",
                version=f"legacy-{index}",
            )
        )
    return tuple(result)


def request_from_smartstress_state(
    state: Mapping[str, Any], *, session_id: str | None = None
) -> MindCareRequest:
    """Build the wrapper request; name retained as the project-facing module entry."""
    preferences = dict(state.get("user_preferences") or {})
    return MindCareRequest(
        user_text=_latest_user_text(state),
        physio_context=_physio_context(state),
        user_profile=UserProfile(
            archetype=str(preferences.get("archetype") or "anonymous_adult"),
            language=str(preferences.get("language") or "en"),
            tone=str(preferences.get("tone") or "brief"),
            available_minutes=(int(preferences["available_minutes"]) if preferences.get("available_minutes") is not None else None),
            preferences=preferences,
            disallowed_interventions=tuple(preferences.get("disallowed_interventions") or ()),
        ),
        evidence=_evidence(state),
        consent_response=state.get("human_confirmation_response"),
        awaiting_confirmation=bool(state.get("awaiting_human_confirmation")),
        locale=str(preferences.get("locale") or "en-SG"),
        crisis_resources=tuple(state.get("crisis_resources") or ()),
        session_id=session_id or str(state.get("session_id") or "default"),
        current_stressor=(str(state["current_stressor"]) if state.get("current_stressor") else None),
        allowed_tools=tuple(state.get("mindcare_allowed_tools") or ()),
    )


def apply_mindcare_to_state(
    state: Mapping[str, Any],
    wrapper: MindCareReliabilityWrapper,
    *,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Return LangGraph-compatible updates without causing external side effects."""
    result = wrapper.run(request_from_smartstress_state(state, session_id=session_id))
    audit_trail = list(state.get("audit_trail") or [])
    audit_trail.append(result.audit_event())
    policy_versions = dict(state.get("policy_versions") or {})
    policy_versions["mindcare"] = result.wrapper_version
    updates: dict[str, Any] = {
        "mindcare_reliability": result.to_dict(),
        "mindcare_response": result.response,
        "evidence_refs": [item.__dict__.copy() for item in result.evidence],
        "allowed_actions": list(result.allowed_actions),
        "policy_versions": policy_versions,
        "safety_escalation": result.decision == WrapperDecision.ESCALATE,
        "external_side_effects": False,
        "tool_execution_mode": "dry_run",
        "suggested_action": None,
        "awaiting_human_confirmation": False,
        "audit_trail": audit_trail,
    }
    if result.proposed_action is not None and result.decision == WrapperDecision.RELEASE:
        updates["suggested_action"] = result.proposed_action.__dict__.copy()
        updates["awaiting_human_confirmation"] = True
    return updates
