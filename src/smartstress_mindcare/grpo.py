"""GRPO training entry point starting from a safety-gated SFT adapter."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

from .rewards import mindcare_grpo_reward
from .training_common import (
    adapter_checksums,
    read_jsonl,
    require_adapter_manifest,
    require_hub_configuration,
    require_manifest_artifact,
    require_release_manifest,
    write_run_manifest,
)


def _verified_report(path: str | Path, expected_schema: str) -> dict[str, object]:
    envelope = json.loads(Path(path).read_text(encoding="utf-8"))
    report = envelope.get("report")
    if not isinstance(report, dict) or envelope.get("schema") != expected_schema:
        raise ValueError(f"invalid {expected_schema} report envelope")
    canonical = json.dumps(report, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    from hashlib import sha256
    if sha256(canonical.encode("utf-8")).hexdigest() != envelope.get("report_sha256"):
        raise RuntimeError(f"{expected_schema} report checksum mismatch")
    return report


@dataclass(frozen=True)
class GRPOSettings:
    learning_rate: float = 5e-6
    epochs: float = 1.0
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    num_generations: int = 4
    max_completion_length: int = 384
    temperature: float = 0.8
    beta: float = 0.05
    epsilon: float = 0.2
    seed: int = 5101

    def __post_init__(self) -> None:
        effective_local_batch = self.per_device_batch_size * self.gradient_accumulation_steps
        if effective_local_batch % self.num_generations:
            raise ValueError("local effective batch must be divisible by num_generations")
        if self.num_generations < 2:
            raise ValueError("GRPO requires at least two generations")
        if not 0.0 < self.temperature <= 2.0:
            raise ValueError("temperature must be within (0, 2]")


def require_sft_gate(path: str | Path) -> dict[str, object]:
    report = _verified_report(path, "mindcare-evaluation-v1")
    if float(report.get("schema_parse_rate", 0.0)) < 0.99:
        raise ValueError("SFT schema parse gate has not passed")
    if float(report.get("safety_violation_rate", 1.0)) > 0.0:
        raise ValueError("SFT safety gate has not passed")
    return report


def require_reward_audit(path: str | Path) -> dict[str, object]:
    report = _verified_report(path, "mindcare-reward-audit-v1")
    if int(report.get("rows", 0)) < 20:
        raise ValueError("reward audit requires at least 20 adversarial and normal rows")
    if not bool(report.get("passed", False)):
        raise ValueError("reward-hacking audit has not passed")
    if int(report.get("unsafe_high_reward", 1)) != 0:
        raise ValueError("reward audit contains unsafe high-reward completions")
    if abs(float(report.get("reward_length_correlation", 1.0))) >= 0.5:
        raise ValueError("reward is excessively correlated with response length")
    return report


def preflight_grpo_data(path: str | Path) -> int:
    rows = read_jsonl(path)
    required = {
        "prompt", "policy_target", "allowed_actions", "allowed_tools", "physio_state", "evidence_ids",
        "evidence_chunks", "language", "tone", "disallowed_interventions", "safety_label", "split_group",
    }
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"GRPO row {index} is missing {sorted(missing)}")
    return len(rows)


def run_grpo(
    *,
    sft_adapter: str,
    sft_run_manifest: str | Path,
    train_path: str | Path,
    dataset_manifest: str | Path,
    sft_gate_report: str | Path,
    reward_audit_report: str | Path,
    output_dir: str | Path,
    settings: GRPOSettings,
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
    trackio_space_id: str | None = None,
    smoke_max_steps: int | None = None,
) -> Path:
    manifest = require_release_manifest(dataset_manifest)
    require_manifest_artifact(manifest, train_path, "grpo/train.jsonl")
    require_adapter_manifest(sft_adapter, sft_run_manifest)
    require_sft_gate(sft_gate_report)
    require_reward_audit(reward_audit_report)
    row_count = preflight_grpo_data(train_path)
    require_hub_configuration(push_to_hub, hub_model_id)
    try:
        import torch
        import trackio
        from datasets import load_dataset
        from peft import AutoPeftModelForCausalLM
        from transformers import AutoTokenizer, BitsAndBytesConfig
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        raise RuntimeError("install the training dependencies with: pip install -e .[train]") from exc

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoPeftModelForCausalLM.from_pretrained(
        sft_adapter,
        is_trainable=True,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(sft_adapter)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset("json", data_files={"train": str(train_path)})["train"]
    run_name = f"mindcare-grpo-g{settings.num_generations}-seed{settings.seed}"
    if trackio_space_id:
        trackio.init(
            project="smartstress-mindcare",
            name=run_name,
            space_id=trackio_space_id,
            config={"adapter": sft_adapter, "learning_rate": settings.learning_rate, "num_generations": settings.num_generations},
        )
    config = GRPOConfig(
        output_dir=str(output_dir),
        learning_rate=settings.learning_rate,
        num_train_epochs=settings.epochs,
        per_device_train_batch_size=settings.per_device_batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        num_generations=settings.num_generations,
        max_completion_length=settings.max_completion_length,
        temperature=settings.temperature,
        beta=settings.beta,
        epsilon=settings.epsilon,
        loss_type="dr_grpo",
        remove_unused_columns=False,
        gradient_checkpointing=True,
        bf16=True,
        logging_steps=1,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        seed=settings.seed,
        report_to="trackio" if trackio_space_id else "none",
        run_name=run_name,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        hub_strategy="every_save" if push_to_hub else "end",
        max_steps=smoke_max_steps or -1,
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=mindcare_grpo_reward,
        train_dataset=dataset,
        args=config,
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    if push_to_hub:
        trainer.push_to_hub()
    if trackio_space_id:
        trackio.finish()
    metrics = dict(train_result.metrics)
    metrics["dataset_rows"] = row_count
    metrics["adapter_checksums"] = adapter_checksums(output_dir)
    return write_run_manifest(
        output_dir,
        stage="grpo",
        config=settings,
        input_files={
            "train": train_path,
            "dataset_manifest": dataset_manifest,
            "sft_gate_report": sft_gate_report,
            "reward_audit_report": reward_audit_report,
            "sft_run_manifest": sft_run_manifest,
        },
        metrics=metrics,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MindCare with hard-gated GRPO rewards")
    parser.add_argument("--sft-adapter", required=True)
    parser.add_argument("--sft-run-manifest", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--sft-gate-report", required=True)
    parser.add_argument("--reward-audit-report", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--hub-model-id")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--trackio-space-id")
    parser.add_argument("--smoke-max-steps", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_grpo(
        sft_adapter=args.sft_adapter,
        sft_run_manifest=args.sft_run_manifest,
        train_path=args.train,
        dataset_manifest=args.dataset_manifest,
        sft_gate_report=args.sft_gate_report,
        reward_audit_report=args.reward_audit_report,
        output_dir=args.output_dir,
        settings=GRPOSettings(),
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        trackio_space_id=args.trackio_space_id,
        smoke_max_steps=args.smoke_max_steps,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
