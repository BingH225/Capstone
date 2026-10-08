"""Credential-free integrity and syntax checks for a fresh Git checkout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    required = (
        "pyproject.toml", ".env.example", "apps/web/package-lock.json",
        "src/smartstress_policy_reliability/__init__.py",
        "src/smartstress_mindcare/__init__.py",
        "src/smartstress_langgraph/server.py",
        "research/physio/scripts/run_stressid_adaptation.py",
        "docs/DEVELOPMENT.md", "docs/source_import_manifest.json",
    )
    for relative in required:
        if not (ROOT / relative).is_file():
            raise RuntimeError(f"Missing checkout file: {relative}")
    model_dir = ROOT / "src/smartstress_langgraph/physio/artifacts"
    manifest = json.loads((model_dir / "wesad_attention_v1.json").read_text(encoding="utf-8"))
    checkpoint = model_dir / manifest["checkpoint"]["filename"]
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if digest.lower() != manifest["checkpoint"]["sha256"].lower():
        raise RuntimeError("Bundled runtime checkpoint does not match its manifest")
    count = 0
    for directory in ("src", "scripts", "research", "experiments", "tools", "tests"):
        for file in (ROOT / directory).rglob("*.py"):
            compile(file.read_text(encoding="utf-8-sig"), str(file), "exec")
            count += 1
    print(f"Checkout OK: {count} Python files compile; bundled model SHA-256 verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
