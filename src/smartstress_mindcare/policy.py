"""Deterministic pre-generation policy for MindCare."""

from __future__ import annotations

import re

from .contracts import MindCareRequest, PhysioReliabilityState, PolicyTarget
from .validation import contains_crisis


_MEDICAL_REQUEST = re.compile(
    r"\b(?:diagnos(?:e|is)|what disorder|what medication|which medicine|change my dose|stop taking|clinical treatment plan)\b",
    re.IGNORECASE,
)
_INJECTION = re.compile(
    r"(?:ignore (?:all |the )?(?:previous|system) instructions|reveal (?:the )?system prompt|developer message|bypass (?:safety|policy))",
    re.IGNORECASE,
)


def normalize_consent(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = " ".join(value.casefold().strip().split()).rstrip(".!?")
    if normalized in {"yes", "y", "sure", "ok", "okay", "confirm", "proceed"}:
        return "yes"
    if normalized in {"no", "n", "nope", "no thanks", "do not proceed"}:
        return "no"
    if normalized in {"cancel", "stop", "never mind", "nevermind"}:
        return "cancel"
    if normalized in {"refine", "change it", "modify", "edit"}:
        return "refine"
    return None


def is_prompt_injection(text: str) -> bool:
    return bool(_INJECTION.search(text))


def is_medical_authority_request(text: str) -> bool:
    return bool(_MEDICAL_REQUEST.search(text))


def expected_policy_target(request: MindCareRequest) -> PolicyTarget:
    if contains_crisis(request.user_text):
        return PolicyTarget.ESCALATE
    if is_prompt_injection(request.user_text) or is_medical_authority_request(request.user_text):
        return PolicyTarget.ABSTAIN
    if request.awaiting_confirmation:
        consent = normalize_consent(request.consent_response or request.user_text)
        if consent == "yes":
            return PolicyTarget.CONFIRM
        if consent in {"no", "refine"}:
            return PolicyTarget.REFINE
        if consent == "cancel":
            return PolicyTarget.MONITOR
        return PolicyTarget.ASK
    state = request.physio_context.reliability_state
    if state in {PhysioReliabilityState.DATA_INVALID, PhysioReliabilityState.OOD_OR_UNCERTAIN}:
        return PolicyTarget.ASK
    if state == PhysioReliabilityState.MONITOR:
        return PolicyTarget.MONITOR
    if state == PhysioReliabilityState.RELIABLE_ELEVATED:
        if request.current_stressor and "propose_dry_run" in request.physio_context.allowed_actions:
            return PolicyTarget.PROPOSE
        return PolicyTarget.SUPPORT
    return PolicyTarget.SUPPORT
