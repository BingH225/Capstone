import json

import pytest

from smartstress_mindcare.dataset import SourceAudit, export_dataset
from smartstress_mindcare.grpo import GRPOSettings, preflight_grpo_data
from smartstress_mindcare.scenarios import demo_examples
from smartstress_mindcare.sft import SFTSettings, preflight_sft_data
from smartstress_mindcare.training_common import require_manifest_artifact, require_release_manifest


def _release(tmp_path):
    examples = demo_examples()
    audit = SourceAudit("fixtures", "f" * 64, 12, 0, 0, {}, "CC-BY-4.0", True)
    output = tmp_path / "release"
    export_dataset(
        {"train": examples[:8], "validation": examples[8:10], "test": examples[10:]},
        output,
        source_audits=(audit,),
    )
    return output


def test_training_preflights_and_manifest_artifact_binding(tmp_path):
    output = _release(tmp_path)
    manifest = require_release_manifest(output / "manifest.json")
    train = output / "sft" / "train.jsonl"
    validation = output / "sft" / "validation.jsonl"
    require_manifest_artifact(manifest, train, "sft/train.jsonl")
    assert preflight_sft_data(train, validation) == {"train": 8, "validation": 2}
    assert preflight_grpo_data(output / "grpo" / "train.jsonl") == 8

    train.write_text(train.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        require_manifest_artifact(manifest, train, "sft/train.jsonl")


def test_training_configuration_guards():
    assert SFTSettings(rank=16, base_model_revision="commit-hash").target_modules == "all-linear"
    with pytest.raises(ValueError, match="rank"):
        SFTSettings(rank=4)
    assert GRPOSettings(num_generations=4, per_device_batch_size=1, gradient_accumulation_steps=8)
    with pytest.raises(ValueError, match="divisible"):
        GRPOSettings(num_generations=3, per_device_batch_size=1, gradient_accumulation_steps=8)
