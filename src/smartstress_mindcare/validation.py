"""Dataset quality gates and cross-split leakage checks."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Mapping, Sequence

from .contracts import (
    DialogueExample,
    MindCareCandidate,
    PhysioReliabilityState,
    PolicyTarget,
    ReviewStatus,
    SafetyLabel,
)
from .privacy import content_fingerprint, jaccard_similarity, token_shingles


_DIAGNOSIS = re.compile(r"\b(?:you (?:have|are suffering from)|diagnos(?:e|ed|is)|mental disorder)\b", re.IGNORECASE)
_MEDICATION = re.compile(r"\b(?:start|stop|increase|decrease|take)\s+(?:your\s+)?(?:dose|medication|antidepressant|ssri)\b", re.IGNORECASE)
_CRISIS = re.compile(r"\b(?:suicid(?:e|al)|kill myself|end my life|hurt myself|self[- ]?harm|hurt someone|kill someone)\b", re.IGNORECASE)
_SAFE_CLINICAL_REFUSAL = re.compile(
    r"\b(?:i\s+)?(?:cannot|can't|won't|do not|am not able to)\s+(?:diagnos\w*|recommend\s+medication|prescribe\w*)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    message: str
    fatal: bool = True


@dataclass(frozen=True)
class DatasetValidationReport:
    valid: bool
    examples: int
    issues: tuple[ValidationIssue, ...]
    scenario_counts: Mapping[str, int]
    policy_counts: Mapping[str, int]
    review_counts: Mapping[str, int]

    def to_dict(self) -> dict[str, object]:
        return {
            "valid": self.valid,
            "examples": self.examples,
            "issues": [issue.__dict__ for issue in self.issues],
            "scenario_counts": dict(self.scenario_counts),
            "policy_counts": dict(self.policy_counts),
            "review_counts": dict(self.review_counts),
        }


def contains_crisis(text: str) -> bool:
    return bool(_CRISIS.search(text))


def contains_unsafe_clinical_claim(text: str) -> bool:
    without_refusals = _SAFE_CLINICAL_REFUSAL.sub("", text)
    return bool(_DIAGNOSIS.search(without_refusals) or _MEDICATION.search(without_refusals))


def validate_example(example: DialogueExample) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    user_text = " ".join(message.content for message in example.messages if message.role == "user")
    assistant_text = next(
        (message.content for message in reversed(example.messages) if message.role == "assistant"),
        "",
    )
    try:
        candidate = MindCareCandidate.parse(assistant_text)
    except (KeyError, TypeError, ValueError) as exc:
        return (ValidationIssue("SCHEMA_INVALID", str(exc)),)
    if candidate.policy_target != example.policy_target:
        issues.append(ValidationIssue("POLICY_TARGET_MISMATCH", "assistant target differs from label"))
    if contains_crisis(user_text) and example.policy_target != PolicyTarget.ESCALATE:
        issues.append(ValidationIssue("CRISIS_NOT_ESCALATED", "crisis language must target escalation"))
    if contains_unsafe_clinical_claim(candidate.response) and example.policy_target != PolicyTarget.ESCALATE:
        issues.append(ValidationIssue("UNSAFE_CLINICAL_CLAIM", "diagnosis or medication instruction detected"))
    evidence_ids = {item.evidence_id for item in example.retrieval_context}
    if not set(candidate.evidence_ids).issubset(evidence_ids):
        issues.append(ValidationIssue("UNKNOWN_EVIDENCE_ID", "candidate cites evidence outside retrieval context"))
    if not evidence_ids and candidate.evidence_ids:
        issues.append(ValidationIssue("UNGROUNDED_CITATION", "candidate cites evidence when retrieval is empty"))
    if candidate.proposed_action is not None:
        if "propose_dry_run" not in example.physio_context.allowed_actions:
            issues.append(ValidationIssue("ACTION_NOT_ALLOWED", "physio policy does not allow a proposal"))
        if example.policy_target not in {PolicyTarget.PROPOSE, PolicyTarget.CONFIRM}:
            issues.append(ValidationIssue("ACTION_TARGET_INVALID", "action attached to non-action policy target"))
    state = example.physio_context.reliability_state
    target = example.policy_target
    allowed_by_state = {
        PhysioReliabilityState.DATA_INVALID: {PolicyTarget.ASK, PolicyTarget.ABSTAIN},
        PhysioReliabilityState.OOD_OR_UNCERTAIN: {PolicyTarget.ASK, PolicyTarget.ABSTAIN},
        PhysioReliabilityState.MONITOR: {PolicyTarget.MONITOR, PolicyTarget.ASK},
        PhysioReliabilityState.RELIABLE_LOW: {PolicyTarget.MONITOR, PolicyTarget.ASK, PolicyTarget.SUPPORT},
        PhysioReliabilityState.RELIABLE_ELEVATED: {PolicyTarget.SUPPORT, PolicyTarget.ASK, PolicyTarget.PROPOSE, PolicyTarget.CONFIRM},
    }
    if target not in {PolicyTarget.ESCALATE, PolicyTarget.ABSTAIN, PolicyTarget.REFINE, PolicyTarget.CONFIRM} and target not in allowed_by_state[state]:
        issues.append(ValidationIssue("PHYSIO_POLICY_MISMATCH", f"{target.value} is incompatible with {state.value}"))
    high_risk = example.labels.safety != SafetyLabel.SAFE or contains_crisis(user_text)
    if high_risk and example.provenance.review_status != ReviewStatus.HUMAN_REVIEWED:
        issues.append(ValidationIssue("HIGH_RISK_REVIEW_REQUIRED", "high-risk examples require human review"))
    if example.provenance.license.strip().upper() in {"UNKNOWN", "NONE", "UNLICENSED"}:
        issues.append(ValidationIssue("LICENSE_UNRESOLVED", "training export requires a resolved source license"))
    return tuple(issues)


def validate_dataset(
    examples: Sequence[DialogueExample],
    *,
    splits: Mapping[str, Sequence[DialogueExample]] | None = None,
    minimum_normal_human_review_rate: float = 0.2,
    near_duplicate_threshold: float = 0.9,
) -> DatasetValidationReport:
    issues: list[ValidationIssue] = []
    scenario_counts: dict[str, int] = {}
    policy_counts: dict[str, int] = {}
    review_counts: dict[str, int] = {}
    ids: set[tuple[str, str]] = set()
    normal_count = normal_reviewed = 0
    if not examples:
        issues.append(ValidationIssue("EMPTY_DATASET", "dataset must contain at least one example"))
    for example in examples:
        key = (example.conversation_id, example.turn_id)
        if key in ids:
            issues.append(ValidationIssue("DUPLICATE_ID", f"duplicate sample ID: {key}"))
        ids.add(key)
        issues.extend(validate_example(example))
        scenario_counts[example.scenario] = scenario_counts.get(example.scenario, 0) + 1
        policy_counts[example.policy_target.value] = policy_counts.get(example.policy_target.value, 0) + 1
        status = example.provenance.review_status.value
        review_counts[status] = review_counts.get(status, 0) + 1
        if example.labels.safety == SafetyLabel.SAFE:
            normal_count += 1
            normal_reviewed += int(example.provenance.review_status == ReviewStatus.HUMAN_REVIEWED)
    if normal_count and normal_reviewed / normal_count < minimum_normal_human_review_rate:
        issues.append(
            ValidationIssue(
                "NORMAL_REVIEW_RATE_LOW",
                f"normal human-review rate {normal_reviewed / normal_count:.3f} is below {minimum_normal_human_review_rate:.3f}",
            )
        )
    if splits:
        required_splits = {"train", "validation", "test"}
        missing_splits = required_splits - set(splits)
        for split_name in sorted(missing_splits):
            issues.append(ValidationIssue("SPLIT_MISSING", f"required split {split_name} is missing"))
        for split_name in sorted(required_splits & set(splits)):
            if not splits[split_name]:
                issues.append(ValidationIssue("SPLIT_EMPTY", f"required split {split_name} is empty"))
        group_owner: dict[str, str] = {}
        fingerprints: dict[str, str] = {}
        shingles: dict[str, frozenset[tuple[str, ...]]] = {}
        for split_name, rows in splits.items():
            for example in rows:
                owner = group_owner.setdefault(example.split_group, split_name)
                if owner != split_name:
                    issues.append(ValidationIssue("GROUP_LEAKAGE", f"group {example.split_group} spans {owner}/{split_name}"))
                text = " ".join(message.content for message in example.messages if message.role == "user")
                fingerprint = content_fingerprint(text)
                previous = fingerprints.setdefault(fingerprint, split_name)
                if previous != split_name:
                    issues.append(ValidationIssue("EXACT_DUPLICATE_LEAKAGE", f"duplicate content spans {previous}/{split_name}"))
                shingles[f"{split_name}:{example.conversation_id}:{example.turn_id}"] = token_shingles(text)
        items = list(shingles.items())
        for index, (left_key, left_shingles) in enumerate(items):
            left_split = left_key.split(":", 1)[0]
            for right_key, right_shingles in items[index + 1 :]:
                right_split = right_key.split(":", 1)[0]
                if left_split != right_split and jaccard_similarity(left_shingles, right_shingles) >= near_duplicate_threshold:
                    issues.append(ValidationIssue("NEAR_DUPLICATE_LEAKAGE", f"near duplicate spans {left_key}/{right_key}"))
    return DatasetValidationReport(
        valid=not any(issue.fatal for issue in issues),
        examples=len(examples),
        issues=tuple(issues),
        scenario_counts=scenario_counts,
        policy_counts=policy_counts,
        review_counts=review_counts,
    )
