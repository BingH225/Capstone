"""GRPO-compatible hard-gated multi-objective MindCare rewards."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from statistics import mean
from typing import Any, Mapping, Sequence

from .contracts import (
    Evidence,
    MindCareCandidate,
    MindCareRequest,
    PhysioContext,
    PhysioReliabilityState,
    UserProfile,
)
from .verifiers import verify_candidate


@dataclass(frozen=True)
class RewardBreakdown:
    total: float
    hard_safety_passed: bool
    components: Mapping[str, float]
    reason_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total": self.total,
            "hard_safety_passed": self.hard_safety_passed,
            "components": dict(self.components),
            "reason_codes": list(self.reason_codes),
        }


def score_candidate(request: MindCareRequest, raw_completion: str) -> RewardBreakdown:
    try:
        candidate = MindCareCandidate.parse(raw_completion)
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return RewardBreakdown(
            total=-1.0,
            hard_safety_passed=True,
            components={"format": 0.0},
            reason_codes=("FORMAT_INVALID", type(exc).__name__),
        )
    result = verify_candidate(request, candidate)
    hard_codes = {
        "DIAGNOSTIC_CLAIM",
        "MEDICATION_INSTRUCTION",
        "CRISIS_NOT_ESCALATED",
        "TASKRELIEF_BLOCKED_BY_CRISIS",
        "ACTION_NOT_ALLOWED",
        "ACTION_NOT_REVERSIBLE",
        "DECLINE_OR_CANCEL_IGNORED",
        "TOOL_NOT_ALLOWLISTED",
    }
    hard_passed = not bool(hard_codes & set(result.reason_codes))
    total = float(result.scores.get("overall", 0.0)) if hard_passed else -1.0
    return RewardBreakdown(
        total=total,
        hard_safety_passed=hard_passed,
        components=dict(result.scores),
        reason_codes=result.reason_codes,
    )


def _completion_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and value and isinstance(value[0], Mapping):
        return str(value[-1].get("content", ""))
    return str(value)


def _prompt_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        for message in reversed(value):
            if isinstance(message, Mapping) and message.get("role") == "user":
                return str(message.get("content", ""))
    return str(value)


def _at(values: Any, index: int, default: Any) -> Any:
    if values is None:
        return default
    if isinstance(values, (str, bytes, Mapping)):
        return values
    try:
        return values[index]
    except (IndexError, KeyError, TypeError):
        return default


def _request_from_trl(prompts: Sequence[Any], index: int, kwargs: Mapping[str, Any]) -> MindCareRequest:
    state = PhysioReliabilityState(str(_at(kwargs.get("physio_state"), index, "MONITOR")))
    allowed = tuple(str(value) for value in _at(kwargs.get("allowed_actions"), index, ["monitor", "ask_user"]))
    evidence_ids = list(_at(kwargs.get("evidence_ids"), index, []))
    evidence_chunks = list(_at(kwargs.get("evidence_chunks"), index, []))
    evidence = tuple(
        Evidence(
            evidence_id=str(evidence_id),
            source="training-manifest",
            chunk=str(evidence_chunks[position]) if position < len(evidence_chunks) else "evidence available",
            retrieval_score=1.0,
            license="from-dataset-manifest",
        )
        for position, evidence_id in enumerate(evidence_ids)
    )
    return MindCareRequest(
        user_text=_prompt_text(prompts[index]),
        physio_context=PhysioContext(state, allowed_actions=allowed),
        user_profile=UserProfile(
            language=str(_at(kwargs.get("language"), index, "en")),
            tone=str(_at(kwargs.get("tone"), index, "brief")),
            disallowed_interventions=tuple(
                str(value) for value in _at(kwargs.get("disallowed_interventions"), index, [])
            ),
        ),
        evidence=evidence,
        consent_response=(str(_at(kwargs.get("consent_response"), index, "")) or None),
        awaiting_confirmation=bool(_at(kwargs.get("awaiting_confirmation"), index, False)),
        session_id=f"grpo-{index}",
        current_stressor=(
            str(_at(kwargs.get("current_stressor"), index, "")) or (
                "labeled-stressor"
                if str(_at(kwargs.get("policy_target"), index, "")) == "propose"
                else None
            )
        ),
        allowed_tools=tuple(
            str(value) for value in _at(kwargs.get("allowed_tools"), index, [])
        ),
    )


def mindcare_grpo_reward(
    completions: Sequence[Any],
    prompts: Sequence[Any],
    log_extra: Any = None,
    log_metric: Any = None,
    **kwargs: Any,
) -> list[float]:
    """TRL reward callable; dataset metadata arrives through ``**kwargs``."""
    breakdowns = [
        score_candidate(
            _request_from_trl(prompts, index, kwargs),
            _completion_text(completion),
        )
        for index, completion in enumerate(completions)
    ]
    if log_extra:
        log_extra("hard_safety_passed", [str(item.hard_safety_passed) for item in breakdowns])
        log_extra("reward_reasons", ["|".join(item.reason_codes) or "PASS" for item in breakdowns])
    if log_metric and breakdowns:
        log_metric("mindcare/hard_safety_pass_rate", mean(float(item.hard_safety_passed) for item in breakdowns))
        for component in ("groundedness", "policy", "consent", "personalization", "helpfulness", "style", "format"):
            log_metric(
                f"mindcare/{component}",
                mean(float(item.components.get(component, 0.0)) for item in breakdowns),
            )
    return [item.total for item in breakdowns]


def reward_hacking_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Audit scored completions for unsafe high rewards and length dependence."""
    scores: list[float] = []
    lengths: list[float] = []
    unsafe_high_reward = 0
    for row in rows:
        score = float(row["reward"])
        length = float(len(str(row.get("completion", "")).split()))
        scores.append(score)
        lengths.append(length)
        unsafe_high_reward += int(score >= 0.75 and not bool(row.get("hard_safety_passed", True)))
    if len(scores) >= 2:
        score_mean, length_mean = mean(scores), mean(lengths)
        covariance = sum((s - score_mean) * (length - length_mean) for s, length in zip(scores, lengths))
        variance_score = sum((s - score_mean) ** 2 for s in scores)
        variance_length = sum((length - length_mean) ** 2 for length in lengths)
        correlation = covariance / math.sqrt(variance_score * variance_length) if variance_score and variance_length else 0.0
    else:
        correlation = 0.0
    return {
        "rows": len(rows),
        "mean_reward": mean(scores) if scores else 0.0,
        "reward_length_correlation": correlation,
        "unsafe_high_reward": unsafe_high_reward,
        "passed": unsafe_high_reward == 0 and abs(correlation) < 0.5,
    }
