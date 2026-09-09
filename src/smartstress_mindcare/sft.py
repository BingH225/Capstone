"""Executable QLoRA/LoRA SFT entry point for MindCare."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Sequence

from .contracts import MindCareCandidate
from .training_common import (
    adapter_checksums,
    read_jsonl,
    require_hub_configuration,
    require_manifest_artifact,
    require_release_manifest,
    sha256_file,
    write_run_manifest,
)


@dataclass(frozen=True)
class SFTSettings:
    base_model: str = "Qwen/Qwen2.5-3B-Instruct"
    base_model_revision: str = "main"
    base_model_license: str = "Apache-2.0"
    rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: str = "all-linear"
    max_length: int = 2048
    learning_rate: float = 1e-4
    epochs: float = 2.0
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 32
    warmup_ratio: float = 0.05
    seed: int = 5101

    def __post_init__(self) -> None:
        if not self.base_model_license.strip() or self.base_model_license.upper() == "UNKNOWN":
            raise ValueError("base model license must be resolved before training")
        if not self.base_model_revision.strip():
            raise ValueError("base model revision cannot be empty")
        if self.rank not in {8, 16, 32}:
            raise ValueError("rank must be one of 8, 16, 32")
        if self.max_length < 512:
            raise ValueError("max_length is too short for the MindCare schema")


def preflight_sft_data(train_path: str | Path, validation_path: str | Path) -> dict[str, int]:
    if sha256_file(train_path) == sha256_file(validation_path):
        raise ValueError("train and validation datasets are identical")
    result = {}
    for name, path in (("train", train_path), ("validation", validation_path)):
        rows = read_jsonl(path)
        for index, row in enumerate(rows):
            messages = row.get("messages")
            if not isinstance(messages, list) or not messages:
                raise ValueError(f"{name} row {index} lacks messages")
            assistants = [message for message in messages if message.get("role") == "assistant"]
            if not assistants:
                raise ValueError(f"{name} row {index} lacks an assistant target")
            MindCareCandidate.parse(assistants[-1].get("content", ""))
        result[name] = len(rows)
    return result


def run_sft(
    *,
    train_path: str | Path,
    validation_path: str | Path,
    dataset_manifest: str | Path,
    output_dir: str | Path,
    settings: SFTSettings,
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
    trackio_space_id: str | None = None,
    smoke_max_steps: int | None = None,
) -> Path:
    manifest = require_release_manifest(dataset_manifest)
    require_manifest_artifact(manifest, train_path, "sft/train.jsonl")
    require_manifest_artifact(manifest, validation_path, "sft/validation.jsonl")
    counts = preflight_sft_data(train_path, validation_path)
    require_hub_configuration(push_to_hub, hub_model_id)
    try:
        import torch
        import trackio
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError("install the training dependencies with: pip install -e .[train]") from exc

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        settings.base_model,
        revision=settings.base_model_revision,
        quantization_config=quantization,
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False
    tokenizer = AutoTokenizer.from_pretrained(settings.base_model, revision=settings.base_model_revision)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dataset = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(validation_path)},
    )
    run_name = f"mindcare-sft-r{settings.rank}-seed{settings.seed}"
    if trackio_space_id:
        trackio.init(
            project="smartstress-mindcare",
            name=run_name,
            space_id=trackio_space_id,
            config={"model": settings.base_model, "rank": settings.rank, "learning_rate": settings.learning_rate},
        )
    trainer_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=settings.epochs,
        per_device_train_batch_size=settings.per_device_batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        learning_rate=settings.learning_rate,
        warmup_ratio=settings.warmup_ratio,
        lr_scheduler_type="cosine",
        max_length=settings.max_length,
        assistant_only_loss=True,
        gradient_checkpointing=True,
        bf16=True,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        logging_steps=10,
        seed=settings.seed,
        report_to="trackio" if trackio_space_id else "none",
        run_name=run_name,
        push_to_hub=push_to_hub,
        hub_model_id=hub_model_id,
        hub_strategy="every_save" if push_to_hub else "end",
        max_steps=smoke_max_steps or -1,
    )
    lora = LoraConfig(
        r=settings.rank,
        lora_alpha=settings.lora_alpha,
        lora_dropout=settings.lora_dropout,
        target_modules=settings.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=lora,
        args=trainer_config,
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    if push_to_hub:
        trainer.push_to_hub()
    if trackio_space_id:
        trackio.finish()
    metrics = dict(train_result.metrics)
    metrics["dataset_rows"] = counts
    metrics["adapter_checksums"] = adapter_checksums(output_dir)
    return write_run_manifest(
        output_dir,
        stage="sft",
        config=settings,
        input_files={"train": train_path, "validation": validation_path, "dataset_manifest": dataset_manifest},
        metrics=metrics,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the MindCare QLoRA/LoRA SFT adapter")
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument(
        "--base-model-revision",
        required=True,
        help="Immutable Hub commit hash or an explicitly accepted model revision",
    )
    parser.add_argument("--base-model-license", default="Apache-2.0")
    parser.add_argument("--rank", type=int, choices=[8, 16, 32], default=16)
    parser.add_argument("--hub-model-id")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--trackio-space-id")
    parser.add_argument("--smoke-max-steps", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = SFTSettings(
        base_model=args.base_model,
        base_model_revision=args.base_model_revision,
        base_model_license=args.base_model_license,
        rank=args.rank,
    )
    result = run_sft(
        train_path=args.train,
        validation_path=args.validation,
        dataset_manifest=args.dataset_manifest,
        output_dir=args.output_dir,
        settings=settings,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        trackio_space_id=args.trackio_space_id,
        smoke_max_steps=args.smoke_max_steps,
    )
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
