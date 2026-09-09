"""Strict JSON-safe contracts for MindCare data, training and inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import json
import math
from typing import Any, Mapping, Sequence


class PolicyTarget(str, Enum):
    SUPPORT = "support"
    ASK = "ask"
    ABSTAIN = "abstain"
    ESCALATE = "escalate"
    PROPOSE = "propose"
    CONFIRM = "confirm"
    REFINE = "refine"
    MONITOR = "monitor"


class WrapperDecision(str, Enum):
    RELEASE = "RELEASE"
    REWRITE = "REWRITE"
    ABSTAIN = "ABSTAIN"
    ESCALATE = "ESCALATE"


class PhysioReliabilityState(str, Enum):
    DATA_INVALID = "DATA_INVALID"
    OOD_OR_UNCERTAIN = "OOD_OR_UNCERTAIN"
    MONITOR = "MONITOR"
    RELIABLE_LOW = "RELIABLE_LOW"
    RELIABLE_ELEVATED = "RELIABLE_ELEVATED"


class ProvenanceType(str, Enum):
    HUMAN = "human"
    SYNTHETIC = "synthetic"
    TRANSFORMED = "transformed"


class ReviewStatus(str, Enum):
    UNREVIEWED = "unreviewed"
    AUTO_REVIEWED = "auto_reviewed"
    HUMAN_REVIEWED = "human_reviewed"
    REJECTED = "rejected"


class SafetyLabel(str, Enum):
    SAFE = "safe"
    CRISIS = "crisis"
    SELF_HARM = "self_harm"
    HARM_TO_OTHERS = "harm_to_others"
    MEDICATION = "medication"
    DIAGNOSIS = "diagnosis"
    MINOR = "minor"


ALLOWED_ACTIONS = {
    "monitor",
    "retry_sensor",
    "continue_by_text",
    "ask_user",
    "support",
    "propose_dry_run",
}


def _require_nonempty(name: str, value: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")


def _unit_interval(name: str, value: float) -> None:
    if not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
        raise ValueError(f"{name} must be finite and within [0, 1]")


@dataclass(frozen=True)
class Message:
    role: str
    content: str
    tool_name: str | None = None

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant", "tool"}:
            raise ValueError(f"unsupported message role: {self.role!r}")
        _require_nonempty("message.content", self.content)
        if self.role == "tool" and not self.tool_name:
            raise ValueError("tool messages require tool_name")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Message":
        return cls(
            role=str(payload["role"]),
            content=str(payload["content"]),
            tool_name=(str(payload["tool_name"]) if payload.get("tool_name") else None),
        )


@dataclass(frozen=True)
class PhysioContext:
    reliability_state: PhysioReliabilityState
    raw_probability: float | None = None
    calibrated_probability: float | None = None
    reason_codes: tuple[str, ...] = ()
    allowed_actions: tuple[str, ...] = ("monitor", "continue_by_text")
    top_drivers: tuple[Mapping[str, Any], ...] = ()
    model_id: str | None = None
    policy_version: str | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("raw_probability", self.raw_probability),
            ("calibrated_probability", self.calibrated_probability),
        ):
            if value is not None:
                _unit_interval(name, value)
        unknown = set(self.allowed_actions) - ALLOWED_ACTIONS
        if unknown:
            raise ValueError(f"unknown allowed actions: {sorted(unknown)}")
        if self.reliability_state == PhysioReliabilityState.DATA_INVALID and self.calibrated_probability is not None:
            raise ValueError("DATA_INVALID cannot expose a calibrated probability")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "PhysioContext":
        return cls(
            reliability_state=PhysioReliabilityState(str(payload["reliability_state"])),
            raw_probability=(float(payload["raw_probability"]) if payload.get("raw_probability") is not None else None),
            calibrated_probability=(float(payload["calibrated_probability"]) if payload.get("calibrated_probability") is not None else None),
            reason_codes=tuple(str(value) for value in payload.get("reason_codes", ())),
            allowed_actions=tuple(str(value) for value in payload.get("allowed_actions", ("monitor", "continue_by_text"))),
            top_drivers=tuple(dict(value) for value in payload.get("top_drivers", ())),
            model_id=(str(payload["model_id"]) if payload.get("model_id") else None),
            policy_version=(str(payload["policy_version"]) if payload.get("policy_version") else None),
        )


@dataclass(frozen=True)
class UserProfile:
    archetype: str = "anonymous_adult"
    language: str = "en"
    tone: str = "brief"
    available_minutes: int | None = None
    preferences: Mapping[str, Any] = field(default_factory=dict)
    disallowed_interventions: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty("user_profile.archetype", self.archetype)
        _require_nonempty("user_profile.language", self.language)
        _require_nonempty("user_profile.tone", self.tone)
        if self.available_minutes is not None and self.available_minutes < 0:
            raise ValueError("available_minutes cannot be negative")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "UserProfile":
        return cls(
            archetype=str(payload.get("archetype", "anonymous_adult")),
            language=str(payload.get("language", "en")),
            tone=str(payload.get("tone", "brief")),
            available_minutes=(int(payload["available_minutes"]) if payload.get("available_minutes") is not None else None),
            preferences=dict(payload.get("preferences", {})),
            disallowed_interventions=tuple(str(value) for value in payload.get("disallowed_interventions", ())),
        )


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    source: str
    chunk: str
    retrieval_score: float
    license: str
    version: str = "v1"

    def __post_init__(self) -> None:
        _require_nonempty("evidence_id", self.evidence_id)
        _require_nonempty("evidence.source", self.source)
        _require_nonempty("evidence.chunk", self.chunk)
        _require_nonempty("evidence.license", self.license)
        _unit_interval("retrieval_score", self.retrieval_score)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Evidence":
        return cls(
            evidence_id=str(payload.get("evidence_id") or payload.get("id") or ""),
            source=str(payload.get("source", "")),
            chunk=str(payload.get("chunk") or payload.get("content") or ""),
            retrieval_score=float(payload.get("retrieval_score", payload.get("score", 0.0))),
            license=str(payload.get("license", "")),
            version=str(payload.get("version", "v1")),
        )


@dataclass(frozen=True)
class ActionTarget:
    tool_name: str
    tool_input: Mapping[str, Any]
    execution_mode: str = "dry_run"
    requires_confirmation: bool = True

    def __post_init__(self) -> None:
        _require_nonempty("tool_name", self.tool_name)
        if self.execution_mode != "dry_run":
            raise ValueError("only dry_run action targets are allowed")
        if not self.requires_confirmation:
            raise ValueError("all action targets require explicit confirmation")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ActionTarget":
        return cls(
            tool_name=str(payload["tool_name"]),
            tool_input=dict(payload.get("tool_input", {})),
            execution_mode=str(payload.get("execution_mode", "dry_run")),
            requires_confirmation=bool(payload.get("requires_confirmation", True)),
        )


@dataclass(frozen=True)
class QualityLabels:
    safety: SafetyLabel = SafetyLabel.SAFE
    groundedness: float = 1.0
    empathy: float = 1.0
    personalization: float = 1.0
    consent: float = 1.0
    conciseness: float = 1.0

    def __post_init__(self) -> None:
        for name in ("groundedness", "empathy", "personalization", "consent", "conciseness"):
            _unit_interval(name, getattr(self, name))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "QualityLabels":
        return cls(
            safety=SafetyLabel(str(payload.get("safety", "safe"))),
            groundedness=float(payload.get("groundedness", 1.0)),
            empathy=float(payload.get("empathy", 1.0)),
            personalization=float(payload.get("personalization", 1.0)),
            consent=float(payload.get("consent", 1.0)),
            conciseness=float(payload.get("conciseness", 1.0)),
        )


@dataclass(frozen=True)
class Provenance:
    type: ProvenanceType
    source_id: str
    source: str
    license: str
    generator_version: str | None = None
    prompt_version: str | None = None
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED
    reviewer_id_hash: str | None = None

    def __post_init__(self) -> None:
        _require_nonempty("provenance.source_id", self.source_id)
        _require_nonempty("provenance.source", self.source)
        _require_nonempty("provenance.license", self.license)
        if self.type == ProvenanceType.SYNTHETIC and not self.generator_version:
            raise ValueError("synthetic examples require generator_version")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "Provenance":
        return cls(
            type=ProvenanceType(str(payload["type"])),
            source_id=str(payload["source_id"]),
            source=str(payload["source"]),
            license=str(payload["license"]),
            generator_version=(str(payload["generator_version"]) if payload.get("generator_version") else None),
            prompt_version=(str(payload["prompt_version"]) if payload.get("prompt_version") else None),
            review_status=ReviewStatus(str(payload.get("review_status", "unreviewed"))),
            reviewer_id_hash=(str(payload["reviewer_id_hash"]) if payload.get("reviewer_id_hash") else None),
        )


@dataclass(frozen=True)
class DialogueExample:
    conversation_id: str
    turn_id: str
    messages: tuple[Message, ...]
    physio_context: PhysioContext
    user_profile: UserProfile
    retrieval_context: tuple[Evidence, ...]
    policy_target: PolicyTarget
    action_target: ActionTarget | None
    labels: QualityLabels
    provenance: Provenance
    split_group: str
    scenario: str

    def __post_init__(self) -> None:
        _require_nonempty("conversation_id", self.conversation_id)
        _require_nonempty("turn_id", self.turn_id)
        _require_nonempty("split_group", self.split_group)
        _require_nonempty("scenario", self.scenario)
        roles = [message.role for message in self.messages]
        if "user" not in roles or "assistant" not in roles:
            raise ValueError("SFT examples require user and assistant messages")
        if self.action_target is not None and self.policy_target not in {PolicyTarget.PROPOSE, PolicyTarget.CONFIRM}:
            raise ValueError("action_target is only valid for propose/confirm samples")
        if self.labels.safety in {SafetyLabel.CRISIS, SafetyLabel.SELF_HARM, SafetyLabel.HARM_TO_OTHERS} and self.policy_target != PolicyTarget.ESCALATE:
            raise ValueError("crisis/harm examples must target escalation")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["physio_context"]["reliability_state"] = self.physio_context.reliability_state.value
        payload["policy_target"] = self.policy_target.value
        payload["labels"]["safety"] = self.labels.safety.value
        payload["provenance"]["type"] = self.provenance.type.value
        payload["provenance"]["review_status"] = self.provenance.review_status.value
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DialogueExample":
        action = payload.get("action_target")
        return cls(
            conversation_id=str(payload["conversation_id"]),
            turn_id=str(payload["turn_id"]),
            messages=tuple(Message.from_dict(value) for value in payload["messages"]),
            physio_context=PhysioContext.from_dict(payload["physio_context"]),
            user_profile=UserProfile.from_dict(payload.get("user_profile", {})),
            retrieval_context=tuple(Evidence.from_dict(value) for value in payload.get("retrieval_context", ())),
            policy_target=PolicyTarget(str(payload["policy_target"])),
            action_target=ActionTarget.from_dict(action) if action else None,
            labels=QualityLabels.from_dict(payload.get("labels", {})),
            provenance=Provenance.from_dict(payload["provenance"]),
            split_group=str(payload["split_group"]),
            scenario=str(payload["scenario"]),
        )

    def sft_row(self) -> dict[str, Any]:
        return {
            "messages": [asdict(message) for message in self.messages],
            "conversation_id": self.conversation_id,
            "turn_id": self.turn_id,
            "split_group": self.split_group,
            "scenario": self.scenario,
        }

    def grpo_row(self) -> dict[str, Any]:
        prompt = [asdict(message) for message in self.messages if message.role != "assistant"]
        return {
            "prompt": prompt,
            "policy_target": self.policy_target.value,
            "allowed_actions": list(self.physio_context.allowed_actions),
            "allowed_tools": ([self.action_target.tool_name] if self.action_target else []),
            "physio_state": self.physio_context.reliability_state.value,
            "evidence_ids": [evidence.evidence_id for evidence in self.retrieval_context],
            "evidence_chunks": [evidence.chunk for evidence in self.retrieval_context],
            "language": self.user_profile.language,
            "tone": self.user_profile.tone,
            "disallowed_interventions": list(self.user_profile.disallowed_interventions),
            "safety_label": self.labels.safety.value,
            "scenario": self.scenario,
            "split_group": self.split_group,
            "current_stressor": "labeled-stressor" if self.policy_target == PolicyTarget.PROPOSE else None,
            "awaiting_confirmation": self.policy_target in {PolicyTarget.CONFIRM, PolicyTarget.REFINE},
            "consent_response": (
                "yes" if self.policy_target == PolicyTarget.CONFIRM else
                "no" if self.policy_target == PolicyTarget.REFINE else None
            ),
        }


@dataclass(frozen=True)
class MindCareCandidate:
    response: str
    policy_target: PolicyTarget
    evidence_ids: tuple[str, ...] = ()
    proposed_action: ActionTarget | None = None
    uncertainty_acknowledged: bool = False

    def __post_init__(self) -> None:
        _require_nonempty("candidate.response", self.response)

    @classmethod
    def parse(cls, value: str | Mapping[str, Any]) -> "MindCareCandidate":
        payload = json.loads(value) if isinstance(value, str) else dict(value)
        expected = {"response", "policy_target", "evidence_ids", "proposed_action", "uncertainty_acknowledged"}
        unknown = set(payload) - expected
        if unknown:
            raise ValueError(f"candidate contains unknown fields: {sorted(unknown)}")
        action = payload.get("proposed_action")
        return cls(
            response=str(payload["response"]),
            policy_target=PolicyTarget(str(payload["policy_target"])),
            evidence_ids=tuple(str(item) for item in payload.get("evidence_ids", ())),
            proposed_action=ActionTarget.from_dict(action) if action else None,
            uncertainty_acknowledged=bool(payload.get("uncertainty_acknowledged", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["policy_target"] = self.policy_target.value
        return payload


@dataclass(frozen=True)
class MindCareRequest:
    user_text: str
    physio_context: PhysioContext
    user_profile: UserProfile = field(default_factory=UserProfile)
    evidence: tuple[Evidence, ...] = ()
    consent_response: str | None = None
    awaiting_confirmation: bool = False
    locale: str = "en-SG"
    crisis_resources: tuple[str, ...] = ()
    session_id: str = "default"
    current_stressor: str | None = None
    allowed_tools: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty("user_text", self.user_text)
        _require_nonempty("session_id", self.session_id)


@dataclass(frozen=True)
class VerificationResult:
    passed: bool
    fixable: bool
    scores: Mapping[str, float]
    reason_codes: tuple[str, ...]


@dataclass(frozen=True)
class MindCareReliabilityResult:
    decision: WrapperDecision
    response: str | None
    policy_target: PolicyTarget
    scores: Mapping[str, float]
    evidence: tuple[Evidence, ...]
    allowed_actions: tuple[str, ...]
    reason_codes: tuple[str, ...]
    model_version: str
    wrapper_version: str
    rewrite_count: int
    proposed_action: ActionTarget | None = None
    session_id: str = "default"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["decision"] = self.decision.value
        payload["policy_target"] = self.policy_target.value
        return payload

    def audit_event(self) -> dict[str, Any]:
        return {
            "node": "mindcare_reliability",
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "summary": f"MindCare reliability decision: {self.decision.value}",
            "details": {
                "session_id": self.session_id,
                "policy_target": self.policy_target.value,
                "reason_codes": list(self.reason_codes),
                "scores": dict(self.scores),
                "evidence_ids": [item.evidence_id for item in self.evidence],
                "allowed_actions": list(self.allowed_actions),
                "model_version": self.model_version,
                "wrapper_version": self.wrapper_version,
                "rewrite_count": self.rewrite_count,
            },
        }
