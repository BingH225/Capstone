import json

import pytest

from smartstress_mindcare.dataset import (
    SourceAudit,
    SourceRecord,
    export_dataset,
    records_to_examples,
)
from smartstress_mindcare.privacy import deidentify
from smartstress_mindcare.scenarios import demo_examples
from smartstress_mindcare.validation import validate_dataset


def _audit(rows: int = 12) -> SourceAudit:
    return SourceAudit(
        source_file="test.jsonl",
        sha256="0" * 64,
        rows=rows,
        empty_rows=0,
        exact_duplicate_questions=0,
        pii_counts={"email": 0, "url": 0, "phone": 0, "handle": 0},
        license="CC-BY-4.0",
        license_resolved=True,
    )


def test_demo_scenarios_cover_s1_to_s12_and_include_multiturn():
    examples = demo_examples()
    assert [item.scenario.split("_", 1)[0] for item in examples] == [f"S{i}" for i in range(1, 13)]
    assert len(examples[8].messages) == 5
    report = validate_dataset(
        examples,
        splits={"train": examples[:8], "validation": examples[8:10], "test": examples[10:]},
    )
    assert report.valid, report.to_dict()


def test_export_writes_checksum_manifest_and_excludes_test_from_grpo(tmp_path):
    examples = demo_examples()
    output = tmp_path / "release"
    envelope = export_dataset(
        {"train": examples[:8], "validation": examples[8:10], "test": examples[10:]},
        output,
        source_audits=(_audit(),),
    )
    assert envelope["manifest"]["validation"]["valid"] is True
    assert (output / "manifest.json").exists()
    grpo_rows = [json.loads(line) for line in (output / "grpo" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(grpo_rows) == 8
    assert {row["split_group"] for row in grpo_rows}.isdisjoint(
        {item.split_group for item in examples[8:]}
    )


def test_release_rejects_empty_split(tmp_path):
    examples = demo_examples()
    with pytest.raises(ValueError, match="SPLIT_EMPTY"):
        export_dataset(
            {"train": examples, "validation": [], "test": []},
            tmp_path,
            source_audits=(_audit(),),
        )


def test_duplicate_family_is_assigned_to_one_split():
    records = [
        SourceRecord("1", "I feel overwhelmed by several deadlines today", "Start with one small task and pause.", "fixture", "CC-BY-4.0", split_group="family"),
        SourceRecord("2", "I feel overwhelmed by several deadlines today!", "Choose the most urgent manageable step.", "fixture", "CC-BY-4.0", split_group="family"),
    ]
    _, assignments = records_to_examples(records)
    assert assignments["1"] == assignments["2"]


def test_deidentify_removes_direct_identifiers():
    cleaned = deidentify("Email me@site.com or call +65 8123 4567; see https://example.com @person")
    assert cleaned == "Email [EMAIL] or call [PHONE]; see [URL] [HANDLE]"
