"""Regression checks for configuration and real graph construction after migration."""

import importlib

from smartstress_langgraph.config import _parse_dotenv


def test_shell_environment_has_priority_over_dotenv(monkeypatch, tmp_path):
    monkeypatch.setenv("SMARTSTRESS_PORT", "8123")
    dotenv = tmp_path / ".env"
    dotenv.write_text("SMARTSTRESS_PORT=8999\n", encoding="utf-8")
    _parse_dotenv(dotenv)
    import os
    assert os.environ["SMARTSTRESS_PORT"] == "8123"


def test_graph_builds_after_node_submodules_are_imported():
    # Importing a submodule installs a module-valued package attribute. Graph
    # construction must import the function directly rather than that attribute.
    importlib.import_module("smartstress_langgraph.nodes.physio_sense_node")
    importlib.import_module("smartstress_langgraph.nodes.mind_care_node")
    from smartstress_langgraph.graph import build_workflow_graph
    graph = build_workflow_graph().compile()
    assert "physio_sense" in graph.nodes
    assert "mind_care" in graph.nodes
