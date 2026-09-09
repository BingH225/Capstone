"""SmartStress MindCare data, training and reliability module."""

from .contracts import (
    ActionTarget,
    DialogueExample,
    Evidence,
    Message,
    MindCareCandidate,
    MindCareReliabilityResult,
    MindCareRequest,
    PhysioContext,
    PhysioReliabilityState,
    PolicyTarget,
    Provenance,
    QualityLabels,
    UserProfile,
    WrapperDecision,
)
from .dataset import audit_source, export_dataset, load_canonical_dataset, load_source_records, records_to_examples
from .integration import apply_mindcare_to_state, request_from_smartstress_state
from .policy import expected_policy_target
from .rewards import mindcare_grpo_reward, reward_hacking_audit, score_candidate
from .validation import validate_dataset, validate_example
from .wrapper import MindCareReliabilityWrapper, WrapperConfig

__all__ = [
    "ActionTarget",
    "DialogueExample",
    "Evidence",
    "Message",
    "MindCareCandidate",
    "MindCareReliabilityResult",
    "MindCareReliabilityWrapper",
    "MindCareRequest",
    "PhysioContext",
    "PhysioReliabilityState",
    "PolicyTarget",
    "Provenance",
    "QualityLabels",
    "UserProfile",
    "WrapperConfig",
    "WrapperDecision",
    "apply_mindcare_to_state",
    "audit_source",
    "expected_policy_target",
    "export_dataset",
    "load_canonical_dataset",
    "load_source_records",
    "mindcare_grpo_reward",
    "records_to_examples",
    "request_from_smartstress_state",
    "reward_hacking_audit",
    "score_candidate",
    "validate_dataset",
    "validate_example",
]
