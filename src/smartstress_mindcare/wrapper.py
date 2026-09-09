"""Generate-verify-rewrite MindCare Reliability Wrapper."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Callable, Sequence

from .contracts import (
    Evidence,
    MindCareCandidate,
    MindCareReliabilityResult,
    MindCareRequest,
    PolicyTarget,
    WrapperDecision,
)
from .policy import expected_policy_target, is_prompt_injection
from .validation import contains_crisis
from .verifiers import parse_and_verify


Generator = Callable[[list[dict[str, str]], str], str]
Retriever = Callable[[str, int], Sequence[Evidence]]


@dataclass(frozen=True)
class WrapperConfig:
    wrapper_version: str = "mindcare-rel-v1"
    model_version: str = "base+sft"
    max_rewrites: int = 2
    retrieval_k: int = 3
    minimum_retrieval_score: float = 0.5

    def __post_init__(self) -> None:
        if self.max_rewrites < 0 or self.max_rewrites > 3:
            raise ValueError("max_rewrites must be within [0, 3]")
        if self.retrieval_k < 0:
            raise ValueError("retrieval_k cannot be negative")
        if not 0.0 <= self.minimum_retrieval_score <= 1.0:
            raise ValueError("minimum_retrieval_score must be within [0, 1]")


class MindCareReliabilityWrapper:
    def __init__(self, generator: Generator, retriever: Retriever | None = None, config: WrapperConfig | None = None) -> None:
        self.generator = generator
        self.retriever = retriever
        self.config = config or WrapperConfig()

    def _result(
        self,
        *,
        decision: WrapperDecision,
        response: str | None,
        target: PolicyTarget,
        evidence: Sequence[Evidence],
        allowed_actions: Sequence[str],
        reasons: Sequence[str],
        scores: dict[str, float] | None = None,
        rewrites: int = 0,
        candidate: MindCareCandidate | None = None,
        session_id: str = "default",
    ) -> MindCareReliabilityResult:
        return MindCareReliabilityResult(
            decision=decision,
            response=response,
            policy_target=target,
            scores=dict(scores or {}),
            evidence=tuple(evidence),
            allowed_actions=tuple(allowed_actions),
            reason_codes=tuple(dict.fromkeys(reasons)),
            model_version=self.config.model_version,
            wrapper_version=self.config.wrapper_version,
            rewrite_count=rewrites,
            proposed_action=(candidate.proposed_action if candidate else None),
            session_id=session_id,
        )

    def _crisis_result(self, request: MindCareRequest) -> MindCareReliabilityResult:
        resources = " ".join(request.crisis_resources)
        response = (
            "I’m concerned about your immediate safety. SmartStress cannot provide crisis care. "
            "Please contact local emergency services now and, if possible, reach a trusted person who can stay with you."
        )
        if resources:
            response += " Local resources configured for your region: " + resources
        reasons = ["CRISIS_PRECHECK", "TASKRELIEF_BLOCKED"]
        if not resources:
            reasons.append("LOCAL_CRISIS_RESOURCE_MISSING")
        return self._result(
            decision=WrapperDecision.ESCALATE,
            response=response,
            target=PolicyTarget.ESCALATE,
            evidence=(),
            allowed_actions=(),
            reasons=reasons,
            scores={"safety": 1.0, "overall": 1.0},
            session_id=request.session_id,
        )

    def _retrieve(self, request: MindCareRequest) -> tuple[tuple[Evidence, ...], tuple[str, ...]]:
        if request.evidence:
            evidence = request.evidence
        elif self.retriever is None or self.config.retrieval_k == 0:
            return (), ("RETRIEVAL_UNAVAILABLE",)
        else:
            try:
                evidence = tuple(self.retriever(request.user_text, self.config.retrieval_k))
            except Exception:
                return (), ("RETRIEVAL_FAILURE",)
        unresolved = {"", "unknown", "none", "unlicensed", "runtime-source-metadata-unavailable"}
        accepted = tuple(
            item for item in evidence
            if item.retrieval_score >= self.config.minimum_retrieval_score
            and item.license.strip().casefold() not in unresolved
        )
        if not accepted:
            if any(item.license.strip().casefold() in unresolved for item in evidence):
                return (), ("EVIDENCE_LICENSE_UNRESOLVED",)
            return (), ("EVIDENCE_INSUFFICIENT",)
        filtered = len(accepted) != len(evidence)
        return accepted, (("EVIDENCE_FILTERED",) if filtered else ())

    @staticmethod
    def _system_prompt(request: MindCareRequest, target: PolicyTarget, evidence: Sequence[Evidence]) -> str:
        context = {
            "required_policy_target": target.value,
            "physio_state": request.physio_context.reliability_state.value,
            "allowed_actions": list(request.physio_context.allowed_actions),
            "allowed_tools": list(request.allowed_tools),
            "language": request.user_profile.language,
            "tone": request.user_profile.tone,
            "disallowed_interventions": list(request.user_profile.disallowed_interventions),
            "evidence": [
                {"evidence_id": item.evidence_id, "source": item.source, "chunk": item.chunk}
                for item in evidence
            ],
        }
        return (
            "You are MindCare, a non-clinical stress-support assistant. Output exactly one JSON object with keys "
            "response, policy_target, evidence_ids, proposed_action, uncertainty_acknowledged. "
            "Cite only supplied evidence IDs. Never diagnose or advise medication. Only propose dry-run, reversible actions "
            "that require explicit confirmation. Respect declines, cancellations and blocked interventions.\nCONTEXT:\n"
            + json.dumps(context, ensure_ascii=False, sort_keys=True)
        )

    def run(self, request: MindCareRequest) -> MindCareReliabilityResult:
        if contains_crisis(request.user_text):
            return self._crisis_result(request)
        target = expected_policy_target(request)
        if is_prompt_injection(request.user_text):
            return self._result(
                decision=WrapperDecision.ABSTAIN,
                response="I can’t follow instructions that bypass the safety or consent policy. I can still help with a brief, non-clinical stress check-in.",
                target=PolicyTarget.ABSTAIN,
                evidence=(),
                allowed_actions=("continue_by_text",),
                reasons=("PROMPT_INJECTION_BLOCKED",),
                scores={"safety": 1.0, "overall": 1.0},
                session_id=request.session_id,
            )
        evidence, retrieval_reasons = self._retrieve(request)
        effective_request = MindCareRequest(
            user_text=request.user_text,
            physio_context=request.physio_context,
            user_profile=request.user_profile,
            evidence=evidence,
            consent_response=request.consent_response,
            awaiting_confirmation=request.awaiting_confirmation,
            locale=request.locale,
            crisis_resources=request.crisis_resources,
            session_id=request.session_id,
            current_stressor=request.current_stressor,
            allowed_tools=request.allowed_tools,
        )
        messages = [{"role": "user", "content": request.user_text}]
        feedback: tuple[str, ...] = ()
        for attempt in range(self.config.max_rewrites + 1):
            system_prompt = self._system_prompt(effective_request, target, evidence)
            if feedback:
                system_prompt += "\nREWRITE_REQUIRED: " + ", ".join(feedback)
            try:
                raw = self.generator(messages, system_prompt)
            except Exception:
                return self._result(
                    decision=WrapperDecision.ABSTAIN,
                    response="I’m unable to generate a reliable response right now. You can continue by text later or seek support from a trusted person.",
                    target=PolicyTarget.ABSTAIN,
                    evidence=evidence,
                    allowed_actions=("continue_by_text",),
                    reasons=(*retrieval_reasons, "GENERATION_FAILURE"),
                    rewrites=attempt,
                    session_id=request.session_id,
                )
            candidate, verification = parse_and_verify(effective_request, raw)
            if verification.passed and candidate is not None:
                actions = tuple(request.physio_context.allowed_actions)
                if candidate.proposed_action is None:
                    actions = tuple(action for action in actions if action != "propose_dry_run")
                return self._result(
                    decision=WrapperDecision.RELEASE,
                    response=candidate.response,
                    target=candidate.policy_target,
                    evidence=evidence,
                    allowed_actions=actions,
                    reasons=(*retrieval_reasons, "VERIFIERS_PASSED"),
                    scores=dict(verification.scores),
                    rewrites=attempt,
                    candidate=candidate,
                    session_id=request.session_id,
                )
            feedback = verification.reason_codes
            if not verification.fixable:
                return self._result(
                    decision=WrapperDecision.ABSTAIN,
                    response="I can’t safely release that response. I can offer a brief check-in without diagnosis, medication advice, or external action.",
                    target=PolicyTarget.ABSTAIN,
                    evidence=evidence,
                    allowed_actions=("continue_by_text", "ask_user"),
                    reasons=(*retrieval_reasons, *feedback, "HARD_VERIFIER_FAILURE"),
                    scores=dict(verification.scores),
                    rewrites=attempt,
                    session_id=request.session_id,
                )
        return self._result(
            decision=WrapperDecision.ABSTAIN,
            response="I couldn’t produce a response that passed the reliability checks. Could you rephrase what kind of support would help?",
            target=PolicyTarget.ABSTAIN,
            evidence=evidence,
            allowed_actions=("continue_by_text", "ask_user"),
            reasons=(*retrieval_reasons, *feedback, "REWRITE_LIMIT_REACHED"),
            rewrites=self.config.max_rewrites,
            session_id=request.session_id,
        )
