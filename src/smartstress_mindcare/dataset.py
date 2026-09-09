"""Leakage-resistant MindCare dataset construction and artifact manifests."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence

from .contracts import (
    DialogueExample,
    Message,
    MindCareCandidate,
    PhysioContext,
    PhysioReliabilityState,
    PolicyTarget,
    Provenance,
    ProvenanceType,
    QualityLabels,
    ReviewStatus,
    SafetyLabel,
    UserProfile,
)
from .privacy import content_fingerprint, deidentify, jaccard_similarity, pii_counts, token_shingles
from .validation import contains_crisis, validate_dataset


SYSTEM_PROMPT = (
    "You are MindCare, a non-clinical stress-support assistant. Return one strict JSON object "
    "with response, policy_target, evidence_ids, proposed_action, and uncertainty_acknowledged. "
    "Never diagnose or prescribe. Respect allowed actions, explicit consent, user preferences, "
    "and evidence limits. Crisis or harm indicators require escalation to localized human help."
)


@dataclass(frozen=True)
class SourceRecord:
    source_id: str
    question: str
    answer: str
    source: str
    license: str
    topic: str = "general"
    language: str = "en"
    review_status: ReviewStatus = ReviewStatus.UNREVIEWED
    reviewer_id_hash: str | None = None
    split_group: str | None = None


@dataclass(frozen=True)
class SourceAudit:
    source_file: str
    sha256: str
    rows: int
    empty_rows: int
    exact_duplicate_questions: int
    pii_counts: Mapping[str, int]
    license: str
    license_resolved: bool

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class _UnionFind:
    def __init__(self, values: Iterable[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: str, right: str) -> None:
        first, second = self.find(left), self.find(right)
        if first != second:
            low, high = sorted((first, second))
            self.parent[high] = low


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            csv.field_size_limit(min(sys.maxsize, 2**31 - 1))
            return [dict(row) for row in csv.DictReader(handle)]
    raise ValueError("source must be .csv or .jsonl")


def load_source_records(
    path: str | Path,
    *,
    source: str,
    license_name: str,
) -> list[SourceRecord]:
    source_path = Path(path)
    records: list[SourceRecord] = []
    source_id_counts: dict[str, int] = {}
    for index, row in enumerate(_read_rows(source_path)):
        base_source_id = str(row.get("source_id") or row.get("questionID") or row.get("id") or index)
        occurrence = source_id_counts.get(base_source_id, 0)
        source_id_counts[base_source_id] = occurrence + 1
        source_id = base_source_id if occurrence == 0 else f"{base_source_id}:answer-{occurrence + 1}"
        question = str(row.get("question") or row.get("questionText") or "")
        answer = str(row.get("answer") or row.get("answerText") or "")
        topic = str(row.get("topic") or row.get("topics") or "general")
        review_value = str(row.get("review_status") or "unreviewed")
        records.append(
            SourceRecord(
                source_id=source_id,
                question=question,
                answer=answer,
                source=source,
                license=license_name,
                topic=topic,
                language=str(row.get("language") or "en"),
                review_status=ReviewStatus(review_value),
                reviewer_id_hash=(str(row["reviewer_id_hash"]) if row.get("reviewer_id_hash") else None),
                split_group=str(row.get("split_group") or base_source_id),
            )
        )
    return records


def audit_source(path: str | Path, *, source: str, license_name: str) -> SourceAudit:
    source_path = Path(path)
    rows = _read_rows(source_path)
    seen: set[str] = set()
    duplicate_count = empty_count = 0
    counts = {"email": 0, "url": 0, "phone": 0, "handle": 0}
    for row in rows:
        question = str(row.get("question") or row.get("questionText") or "")
        answer = str(row.get("answer") or row.get("answerText") or "")
        if not question.strip() or not answer.strip():
            empty_count += 1
        fingerprint = content_fingerprint(question)
        duplicate_count += int(fingerprint in seen)
        seen.add(fingerprint)
        for name, value in pii_counts(question + "\n" + answer).items():
            counts[name] += value
    resolved = license_name.strip().upper() not in {"", "UNKNOWN", "NONE", "UNLICENSED"}
    return SourceAudit(
        source_file=source_path.name,
        sha256=file_sha256(source_path),
        rows=len(rows),
        empty_rows=empty_count,
        exact_duplicate_questions=duplicate_count,
        pii_counts=counts,
        license=license_name,
        license_resolved=resolved,
    )


def cluster_records(records: Sequence[SourceRecord], *, threshold: float = 0.9) -> dict[str, str]:
    ordered = sorted(records, key=lambda item: item.source_id)
    union = _UnionFind(item.source_id for item in ordered)
    by_group: dict[str, str] = {}
    shingle_map = {item.source_id: token_shingles(item.question) for item in ordered}
    for record in ordered:
        group = record.split_group or record.source_id
        if group in by_group:
            union.union(record.source_id, by_group[group])
        else:
            by_group[group] = record.source_id
    for index, left in enumerate(ordered):
        for right in ordered[index + 1 :]:
            if jaccard_similarity(shingle_map[left.source_id], shingle_map[right.source_id]) >= threshold:
                union.union(left.source_id, right.source_id)
    return {item.source_id: union.find(item.source_id) for item in ordered}


def _split_for_group(group: str, *, seed: int) -> str:
    bucket = int(hashlib.sha256(f"{seed}:{group}".encode("utf-8")).hexdigest()[:8], 16) % 100
    if bucket < 80:
        return "train"
    if bucket < 90:
        return "validation"
    return "test"


def _safety_label(question: str) -> SafetyLabel:
    return SafetyLabel.CRISIS if contains_crisis(question) else SafetyLabel.SAFE


def _physio_for_target(target: PolicyTarget) -> PhysioContext:
    if target == PolicyTarget.ESCALATE:
        return PhysioContext(PhysioReliabilityState.MONITOR, allowed_actions=("monitor", "continue_by_text"))
    return PhysioContext(
        PhysioReliabilityState.RELIABLE_LOW,
        raw_probability=0.2,
        calibrated_probability=0.2,
        reason_codes=("SYNTHETIC_CONTEXT",),
        allowed_actions=("monitor", "continue_by_text", "ask_user", "support"),
        model_id="synthetic-for-dialogue-data",
        policy_version="physio-rel-v1",
    )


def records_to_examples(
    records: Sequence[SourceRecord], *, seed: int = 5101, near_duplicate_threshold: float = 0.9
) -> tuple[dict[str, list[DialogueExample]], dict[str, str]]:
    clusters = cluster_records(records, threshold=near_duplicate_threshold)
    splits: dict[str, list[DialogueExample]] = {"train": [], "validation": [], "test": []}
    assignments: dict[str, str] = {}
    for record in sorted(records, key=lambda item: item.source_id):
        question, answer = deidentify(record.question), deidentify(record.answer)
        if len(question) < 10 or len(answer) < 10:
            continue
        cluster = clusters[record.source_id]
        split = _split_for_group(cluster, seed=seed)
        assignments[record.source_id] = split
        safety = _safety_label(question)
        target = PolicyTarget.ESCALATE if safety != SafetyLabel.SAFE else PolicyTarget.SUPPORT
        candidate = MindCareCandidate(
            response=answer,
            policy_target=target,
            evidence_ids=(),
            proposed_action=None,
            uncertainty_acknowledged=False,
        )
        assistant = json.dumps(candidate.to_dict(), ensure_ascii=False, sort_keys=True)
        example = DialogueExample(
            conversation_id=f"{record.source}:{record.source_id}",
            turn_id="turn-001",
            messages=(
                Message("system", SYSTEM_PROMPT),
                Message("user", question),
                Message("assistant", assistant),
            ),
            physio_context=_physio_for_target(target),
            user_profile=UserProfile(language=record.language),
            retrieval_context=(),
            policy_target=target,
            action_target=None,
            labels=QualityLabels(safety=safety),
            provenance=Provenance(
                type=ProvenanceType.TRANSFORMED,
                source_id=record.source_id,
                source=record.source,
                license=record.license,
                prompt_version="mindcare-schema-v1",
                review_status=record.review_status,
                reviewer_id_hash=record.reviewer_id_hash,
            ),
            split_group=f"cluster:{cluster}",
            scenario=f"source:{record.topic.casefold().replace(' ', '_')}",
        )
        splits[split].append(example)
    return splits, assignments


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    _atomic_write(path, "".join(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def export_dataset(
    splits: Mapping[str, Sequence[DialogueExample]],
    output_dir: str | Path,
    *,
    source_audits: Sequence[SourceAudit],
    seed: int = 5101,
    minimum_normal_human_review_rate: float = 0.2,
) -> dict[str, Any]:
    output = Path(output_dir)
    all_examples = [item for rows in splits.values() for item in rows]
    report = validate_dataset(
        all_examples,
        splits=splits,
        minimum_normal_human_review_rate=minimum_normal_human_review_rate,
    )
    if not report.valid:
        codes = sorted({issue.code for issue in report.issues})
        raise ValueError("dataset release gate failed: " + ", ".join(codes))
    artifacts: dict[str, dict[str, Any]] = {}
    for split_name, rows in splits.items():
        canonical_path = output / "canonical" / f"{split_name}.jsonl"
        sft_path = output / "sft" / f"{split_name}.jsonl"
        _write_jsonl(canonical_path, (item.to_dict() for item in rows))
        _write_jsonl(sft_path, (item.sft_row() for item in rows))
        for path in (canonical_path, sft_path):
            artifacts[str(path.relative_to(output)).replace("\\", "/")] = {
                "rows": len(rows),
                "sha256": file_sha256(path),
            }
    grpo_path = output / "grpo" / "train.jsonl"
    _write_jsonl(grpo_path, (item.grpo_row() for item in splits.get("train", ())))
    artifacts["grpo/train.jsonl"] = {
        "rows": len(splits.get("train", ())),
        "sha256": file_sha256(grpo_path),
    }
    manifest = {
        "schema_version": "mindcare-data-v1",
        "created_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "seed": seed,
        "source_audits": [audit.to_dict() for audit in source_audits],
        "artifacts": artifacts,
        "validation": report.to_dict(),
        "invariants": {
            "split_before_training": True,
            "group_isolation": True,
            "near_duplicate_isolation": True,
            "test_excluded_from_grpo": True,
            "raw_pii_not_exported": True,
        },
    }
    canonical = json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    envelope = {"manifest": manifest, "manifest_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest()}
    _atomic_write(output / "manifest.json", json.dumps(envelope, indent=2, ensure_ascii=False) + "\n")
    data_card = (
        "# MindCare Dataset Card\n\n"
        "This artifact is for non-clinical stress-support research. It must not be used for diagnosis, medication, crisis counselling, or autonomous external actions.\n\n"
        f"- Schema: `mindcare-data-v1`\n- Seed: `{seed}`\n- Examples: `{len(all_examples)}`\n"
        "- Splitting: source/duplicate-family group split before SFT, GRPO, retrieval, or evaluation construction.\n"
        "- Privacy: deterministic HTML/Unicode cleanup and email/URL/phone/handle redaction; residual PII still requires human review.\n"
        "- Safety: all crisis, harm, medication, diagnosis, and minor-related examples require human review.\n"
        "- Known limitation: pattern-based filters are not clinical or privacy guarantees; hidden evaluation must be stored separately from training artifacts.\n"
        "- Deletion: remove the source record and all artifacts sharing its split_group, then rebuild and issue a new manifest.\n"
    )
    _atomic_write(output / "DATA_CARD.md", data_card)
    return envelope


def load_canonical_dataset(path: str | Path) -> list[DialogueExample]:
    return [DialogueExample.from_dict(row) for row in _read_rows(Path(path))]
