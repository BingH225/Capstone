"""Minimal LangGraph/PhysioSense adapter example."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from smartstress_policy_reliability import (
    ReliabilityPolicy,
    apply_reliability_to_physio_state,
)


POLICY = ReliabilityPolicy.load(Path("artifacts/physio_reliability_policy.json"))


def physio_reliability_node(state: Mapping[str, Any]) -> dict[str, Any]:
    """Run immediately after the frozen WESAD DNN node."""
    return apply_reliability_to_physio_state(state, POLICY)
