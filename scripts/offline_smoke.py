"""Reproduce data/wrapper and optional real-model/API checks without credentials."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import patch

from smartstress_mindcare.cli import main as mindcare_main
from smartstress_mindcare.contracts import (
    MindCareCandidate, MindCareRequest, PhysioContext, PhysioReliabilityState,
    PolicyTarget, WrapperDecision,
)
from smartstress_mindcare.wrapper import MindCareReliabilityWrapper


def backend_smoke(directory: Path) -> None:
    from fastapi.testclient import TestClient

    # Configure the isolated checkpoint store before importing the graph API.
    os.environ["SMARTSTRESS_DB_PATH"] = str(directory / "smoke.db")
    from smartstress_langgraph.examples.sample_data import DEMO_STRESS_FEATURES
    from smartstress_langgraph.physio.model import WesadAttentionPredictor
    from smartstress_langgraph.server import app

    prediction = WesadAttentionPredictor().predict(DEMO_STRESS_FEATURES)
    if not 0 <= prediction.probability <= 1:
        raise RuntimeError("Invalid bundled-model probability")
    with (
        patch("smartstress_langgraph.nodes.mind_care_node._generate_chat",
              return_value="[OFFLINE DEMO] That sounds difficult. What would help most right now?"),
        patch("smartstress_langgraph.nodes.mind_care_node._retrieve_context", return_value=[]),
        TestClient(app) as client,
    ):
        client.get("/health").raise_for_status()
        started = client.post("/api/start_session", json={
            "user": {"user_id": "offline-smoke", "session_id": "isolated", "traits": {}},
        })
        started.raise_for_status()
        continued = client.post("/api/continue_session", json={
            "session_handle": started.json()["handle"],
            "user_message": {"role": "user", "content": "I feel overloaded today."},
        })
        continued.raise_for_status()
        view = continued.json()["view"]
        if "[OFFLINE DEMO]" not in view["conversation_history"][-1]["content"]:
            raise RuntimeError("Session API did not return the isolated demo response")
        if view["external_side_effects"]:
            raise RuntimeError("Offline session reported external side effects")
    from smartstress_langgraph.api import APP
    APP.checkpointer.conn.close()  # Windows cannot delete an open SQLite file.
    print(f"Real bundled model OK: probability={prediction.probability:.6f}; session API OK.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", action="store_true", help="Also test installed backend and bundled DNN")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="smartstress-smoke-") as temporary:
        directory = Path(temporary)
        dataset = directory / "dataset"
        if mindcare_main(["demo-data", "--output-dir", str(dataset)]):
            raise RuntimeError("Synthetic dataset export failed")
        if mindcare_main(["validate", "--canonical-dir", str(dataset / "canonical"),
                          "--output", str(directory / "validation.json")]):
            raise RuntimeError("Synthetic dataset validation failed")
        candidate = MindCareCandidate(
            response="That sounds difficult. What would help most right now?",
            policy_target=PolicyTarget.SUPPORT,
        )
        wrapper = MindCareReliabilityWrapper(lambda *_: json.dumps(candidate.to_dict()))
        result = wrapper.run(MindCareRequest(
            user_text="I feel overloaded today.",
            physio_context=PhysioContext(
                PhysioReliabilityState.RELIABLE_LOW,
                allowed_actions=("monitor", "continue_by_text", "ask_user", "support"),
            ),
        ))
        if result.decision != WrapperDecision.RELEASE:
            raise RuntimeError(f"Wrapper smoke failed: {result.reason_codes}")
        print("Synthetic data export/validation and deterministic wrapper release OK.")
        if args.backend:
            backend_smoke(directory)
    print("Offline smoke passed; no cloud model, database or external action was used.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
