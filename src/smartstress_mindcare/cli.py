"""CLI for MindCare data construction, validation and evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .dataset import (
    SourceAudit,
    audit_source,
    export_dataset,
    load_canonical_dataset,
    load_source_records,
    records_to_examples,
)
from .evaluation import compare_reports, evaluate_completions
from .rewards import reward_hacking_audit
from .scenarios import demo_examples
from .validation import validate_dataset, validate_example


def _write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(destination)


def _report_envelope(schema: str, report: Mapping[str, Any]) -> dict[str, Any]:
    canonical = json.dumps(dict(report), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "schema": schema,
        "report": dict(report),
        "report_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }


def audit_command(args: argparse.Namespace) -> int:
    audit = audit_source(args.input, source=args.source, license_name=args.license)
    _write_json(args.output, audit.to_dict())
    print(Path(args.output).resolve())
    return 0


def build_command(args: argparse.Namespace) -> int:
    audit = audit_source(args.input, source=args.source, license_name=args.license)
    if not audit.license_resolved:
        raise ValueError("source license is unresolved; audit may be saved but training export is blocked")
    records = load_source_records(
        args.input,
        source=args.source,
        license_name=args.license,
    )
    splits, _ = records_to_examples(
        records,
        seed=args.seed,
        near_duplicate_threshold=args.near_duplicate_threshold,
    )
    quarantine = []
    filtered: dict[str, list[Any]] = {"train": [], "validation": [], "test": []}
    for split_name, examples in splits.items():
        for example in examples:
            issues = validate_example(example)
            if issues:
                quarantine.append({
                    "conversation_id": example.conversation_id,
                    "turn_id": example.turn_id,
                    "split_group": example.split_group,
                    "issue_codes": [issue.code for issue in issues],
                })
            else:
                filtered[split_name].append(example)
    if quarantine:
        _write_json(
            Path(args.output_dir) / "quarantine_report.json",
            {"rows": len(quarantine), "items": quarantine},
        )
    export_dataset(
        filtered,
        args.output_dir,
        source_audits=(audit,),
        seed=args.seed,
        minimum_normal_human_review_rate=args.minimum_normal_human_review_rate,
    )
    print(Path(args.output_dir).resolve())
    return 0


def demo_command(args: argparse.Namespace) -> int:
    examples = demo_examples()
    splits = {
        "train": examples[:8],
        "validation": examples[8:10],
        "test": examples[10:],
    }
    audit = SourceAudit(
        source_file="authored-demo-fixtures",
        sha256=hashlib.sha256(b"smartstress-mindcare-demo-v1").hexdigest(),
        rows=len(examples),
        empty_rows=0,
        exact_duplicate_questions=0,
        pii_counts={"email": 0, "url": 0, "phone": 0, "handle": 0},
        license="CC-BY-4.0",
        license_resolved=True,
    )
    export_dataset(
        splits,
        args.output_dir,
        source_audits=(audit,),
        minimum_normal_human_review_rate=0.2,
    )
    print(Path(args.output_dir).resolve())
    return 0


def validate_command(args: argparse.Namespace) -> int:
    root = Path(args.canonical_dir)
    splits = {
        name: load_canonical_dataset(root / f"{name}.jsonl")
        for name in ("train", "validation", "test")
    }
    examples = [item for rows in splits.values() for item in rows]
    report = validate_dataset(
        examples,
        splits=splits,
        minimum_normal_human_review_rate=args.minimum_normal_human_review_rate,
    )
    _write_json(args.output, report.to_dict())
    print(Path(args.output).resolve())
    return 0 if report.valid else 2


def evaluate_command(args: argparse.Namespace) -> int:
    examples = load_canonical_dataset(args.canonical)
    completions = [
        str(json.loads(line)["completion"])
        for line in Path(args.completions).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = evaluate_completions(examples, completions)
    report["evaluated_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    _write_json(args.output, _report_envelope("mindcare-evaluation-v1", report))
    print(Path(args.output).resolve())
    return 0 if report["gate_passed"] else 2


def reward_audit_command(args: argparse.Namespace) -> int:
    rows = [
        json.loads(line)
        for line in Path(args.scored_completions).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    report = reward_hacking_audit(rows)
    report["audited_at"] = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    _write_json(args.output, _report_envelope("mindcare-reward-audit-v1", report))
    print(Path(args.output).resolve())
    return 0 if report["passed"] and report["rows"] >= 20 else 2


def compare_command(args: argparse.Namespace) -> int:
    reports: dict[str, Mapping[str, Any]] = {}
    for name, path in (("base", args.base), ("sft", args.sft), ("grpo", args.grpo)):
        envelope = json.loads(Path(path).read_text(encoding="utf-8"))
        report = envelope.get("report")
        if not isinstance(report, Mapping):
            raise ValueError(f"{name} report is not a MindCare evaluation envelope")
        reports[name] = report
    comparison = compare_reports(reports)
    _write_json(args.output, _report_envelope("mindcare-model-comparison-v1", comparison))
    print(Path(args.output).resolve())
    return 0 if comparison["deploy_grpo"] else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="smartstress-mindcare", description="Build and validate SmartStress MindCare artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit-source", help="Audit a CSV/JSONL source without exporting training text")
    audit.add_argument("--input", required=True)
    audit.add_argument("--source", required=True)
    audit.add_argument("--license", default="UNKNOWN")
    audit.add_argument("--output", required=True)
    audit.set_defaults(handler=audit_command)

    build = subparsers.add_parser("build", help="Build release-gated canonical, SFT and GRPO datasets")
    build.add_argument("--input", required=True)
    build.add_argument("--source", required=True)
    build.add_argument("--license", required=True)
    build.add_argument("--output-dir", required=True)
    build.add_argument("--seed", type=int, default=5101)
    build.add_argument("--near-duplicate-threshold", type=float, default=0.9)
    build.add_argument("--minimum-normal-human-review-rate", type=float, default=0.2)
    build.set_defaults(handler=build_command)

    demo = subparsers.add_parser("demo-data", help="Export twelve DEMO-only S1-S12 fixtures")
    demo.add_argument("--output-dir", required=True)
    demo.set_defaults(handler=demo_command)

    validate = subparsers.add_parser("validate", help="Validate canonical train/validation/test splits")
    validate.add_argument("--canonical-dir", required=True)
    validate.add_argument("--output", required=True)
    validate.add_argument("--minimum-normal-human-review-rate", type=float, default=0.2)
    validate.set_defaults(handler=validate_command)

    evaluate = subparsers.add_parser("evaluate", help="Evaluate aligned model completions")
    evaluate.add_argument("--canonical", required=True)
    evaluate.add_argument("--completions", required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(handler=evaluate_command)

    reward_audit = subparsers.add_parser(
        "reward-audit",
        help="Gate GRPO on scored adversarial/normal completions",
    )
    reward_audit.add_argument("--scored-completions", required=True)
    reward_audit.add_argument("--output", required=True)
    reward_audit.set_defaults(handler=reward_audit_command)

    compare = subparsers.add_parser(
        "compare",
        help="Compare Base, SFT and SFT+GRPO evaluation envelopes",
    )
    compare.add_argument("--base", required=True)
    compare.add_argument("--sft", required=True)
    compare.add_argument("--grpo", required=True)
    compare.add_argument("--output", required=True)
    compare.set_defaults(handler=compare_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
