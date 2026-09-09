"""Minimal LangGraph-facing example; generator and retriever stay application-owned."""

from smartstress_mindcare.integration import apply_mindcare_to_state
from smartstress_mindcare.wrapper import MindCareReliabilityWrapper, WrapperConfig


def model_generator(messages: list[dict[str, str]], system_prompt: str) -> str:
    """Call the pinned SFT/SFT+GRPO adapter and return its unmodified text."""
    raise NotImplementedError("bind the deployed model client here with an application timeout")


def structured_retriever(query: str, k: int):
    """Return Sequence[Evidence] with source, license, version and retrieval score."""
    return ()


wrapper = MindCareReliabilityWrapper(
    generator=model_generator,
    retriever=structured_retriever,
    config=WrapperConfig(
        model_version="mindcare-sft-v1",
        wrapper_version="mindcare-rel-v1",
        max_rewrites=2,
    ),
)


def mindcare_reliability_node(state: dict) -> dict:
    """Place after physio_reliability; this function never executes an external action."""
    return apply_mindcare_to_state(state, wrapper)
