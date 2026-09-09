# SmartStress Capstone

This repository contains two reliability boundaries for the SmartStress capstone system:

1. `smartstress_policy_reliability`: converts frozen-DNN outputs into calibrated, selective, temporally stable physiological states.
2. `smartstress_mindcare`: constructs governed dialogue data, trains LoRA/QLoRA + GRPO adapters, and wraps every MindCare response in deterministic safety, grounding, policy, consent, and action checks.

The full project plan is in [`docs/SmartStress_Capstone_Detailed_Project_Plan.md`](docs/SmartStress_Capstone_Detailed_Project_Plan.md). The executable Module 2 guide is in [`MINDCARE_MODULE.md`](MINDCARE_MODULE.md).

## Install and test

```powershell
python -m pip install -e ".[test]"
python -m pytest -q -p no:cacheprovider
```

Training dependencies are optional and should only be installed on the GPU environment:

```powershell
python -m pip install -e ".[train]"
```

## Module 2 quick check

The included S1–S12 fixtures are synthetic, human-reviewed test fixtures. They are not a substitute for the production corpus.

```powershell
smartstress-mindcare demo-data --output-dir artifacts/mindcare-demo
smartstress-mindcare validate `
  --canonical-dir artifacts/mindcare-demo/canonical `
  --output artifacts/mindcare-demo/validation.json
```

Local CounselChat files are intentionally **not** released as training data. Their repository-local audits record unresolved licensing plus duplicate and PII counts under `audits/`. A resolved source license and the human-review gates in `MINDCARE_MODULE.md` are required before `smartstress-mindcare build` can export production data.

## Safety boundary

SmartStress is non-clinical. It must not diagnose, prescribe, provide crisis counselling, or perform an external action without explicit confirmation. Crisis language bypasses generation and TaskRelief. Model text is never sent directly to the UI or tools; the MindCare reliability wrapper decides `RELEASE`, `ABSTAIN`, or `ESCALATE` and emits an auditable state update.
