"""Shared preflight and reproducibility helpers for SFT and GRPO."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
from typing import Any, Mapping


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit(directory: str | Path = ".") -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=directory, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} must contain a JSON object")
        rows.append(value)
    if not rows:
        raise ValueError(f"{path} is empty")
    return rows


def require_hub_configuration(push_to_hub: bool, hub_model_id: str | None) -> None:
    if not push_to_hub:
        return
    if not hub_model_id or "/" not in hub_model_id:
        raise ValueError("--push-to-hub requires --hub-model-id in owner/repository form")
    if not os.getenv("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN with write permission is required for Hub persistence")


def require_release_manifest(path: str | Path) -> Mapping[str, Any]:
    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    manifest = envelope.get("manifest")
    if not isinstance(manifest, dict) or not manifest.get("validation", {}).get("valid"):
        raise ValueError("dataset manifest has not passed the release gate")
    expected = str(envelope.get("manifest_sha256", ""))
    canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    actual = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if expected != actual:
        raise RuntimeError("dataset manifest checksum mismatch")
    return manifest


def require_manifest_artifact(
    manifest: Mapping[str, Any], path: str | Path, artifact_name: str
) -> None:
    artifact = manifest.get("artifacts", {}).get(artifact_name)
    if not isinstance(artifact, Mapping):
        raise ValueError(f"dataset manifest does not contain {artifact_name}")
    if int(artifact.get("rows", 0)) <= 0:
        raise ValueError(f"dataset artifact {artifact_name} is empty")
    if str(artifact.get("sha256", "")) != sha256_file(path):
        raise RuntimeError(f"dataset artifact checksum mismatch: {artifact_name}")


def require_adapter_manifest(adapter_dir: str | Path, run_manifest_path: str | Path) -> Mapping[str, Any]:
    envelope = json.loads(Path(run_manifest_path).read_text(encoding="utf-8"))
    manifest = envelope.get("manifest")
    if not isinstance(manifest, Mapping) or manifest.get("stage") != "sft":
        raise ValueError("SFT adapter run manifest is missing or has the wrong stage")
    canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if hashlib.sha256(canonical.encode("utf-8")).hexdigest() != envelope.get("manifest_sha256"):
        raise RuntimeError("SFT run manifest checksum mismatch")
    expected = manifest.get("metrics", {}).get("adapter_checksums")
    if not isinstance(expected, Mapping) or not expected:
        raise ValueError("SFT run manifest does not contain adapter checksums")
    actual = adapter_checksums(adapter_dir)
    for name, checksum in expected.items():
        if actual.get(str(name)) != checksum:
            raise RuntimeError(f"SFT adapter checksum mismatch: {name}")
    return manifest


def write_run_manifest(
    output_dir: str | Path,
    *,
    stage: str,
    config: Any,
    input_files: Mapping[str, str | Path],
    metrics: Mapping[str, Any] | None = None,
) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config_payload = asdict(config) if is_dataclass(config) else dict(config)
    payload = {
        "schema_version": "mindcare-training-run-v1",
        "run_id": f"{stage}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "stage": stage,
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_commit": git_commit(),
        "config": config_payload,
        "inputs": {
            name: {"path": str(Path(path).resolve()), "sha256": sha256_file(path)}
            for name, path in input_files.items()
        },
        "metrics": dict(metrics or {}),
    }
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    envelope = {"manifest": payload, "manifest_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}
    destination = output / "run_manifest.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(envelope, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination


def adapter_checksums(output_dir: str | Path) -> dict[str, str]:
    extensions = {".safetensors", ".json", ".model"}
    root = Path(output_dir)
    return {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix in extensions and path.name != "run_manifest.json"
    }
