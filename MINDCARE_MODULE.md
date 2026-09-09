# SmartStress Module 2：MindCare 数据、微调与可靠性封装

## 1. 模块交付边界

本模块把“能生成对话的模型”变成“只有通过可审计门禁才允许向用户展示的非临床压力支持服务”。交付物包括：

- 严格、可版本化的数据与推理契约；
- 来源审计、去标识化、重复簇分组、先分组后切分、人工复核和发布清单；
- 4-bit NF4 QLoRA SFT 入口；
- 带硬安全门的 GRPO 多目标奖励及防 reward hacking 审计；
- `precheck → retrieve → generate → verify → rewrite/abstain/escalate` 可靠性包装器；
- 现有 SmartStress LangGraph 状态适配器；
- S1–S12 合成安全回归集、训练前检查和自动化测试。

本次没有宣称已经得到生产 adapter。真实训练被刻意放在数据许可、人工复核、GPU 和评估门禁之后。当前两个本地 CounselChat CSV 的许可字段无法从文件本身确认，因此只能审计，不能进入训练发布。

## 2. 运行时架构

```text
冻结 DNN
   │
   ▼
Post-DNN Policy Reliability
   │  physio_reliability: state/reasons/allowed_actions/version
   ▼
MindCare Reliability Wrapper
   ├─ 危机/伤害/提示注入预检 ───────────────► ESCALATE / ABSTAIN
   ├─ 检索并按分数过滤结构化 evidence
   ├─ SFT 或 SFT+GRPO adapter 生成严格 JSON
   ├─ safety / policy / grounding / consent / preference / style 验证
   ├─ 可修复错误：最多 2 次受约束重写
   ├─ 不可修复错误：ABSTAIN
   └─ 全部通过：RELEASE
             │
             ▼
       LangGraph state adapter
             │
             ├─ UI 可展示 response、evidence、reason codes、可靠性状态
             └─ TaskRelief 只获得 dry-run proposal；确认前无外部副作用
```

关键规则：MindCare 只能消费 Module 1 的 `physio_reliability`，不能把原始 sigmoid 概率或旧的 `stress_detected` 当成可靠事实。缺少该字段时自动降级为 `DATA_INVALID`。

## 3. 数据契约

每个 canonical JSONL 样本必须包含：

- `conversation_id`、`turn_id`、`split_group`、`scenario`；
- `messages`：含 system、一个或多个 user/assistant 轮次；
- `physio_context`：可靠性状态、概率（若允许）、原因码、动作白名单、模型/策略版本；
- `user_profile`：语言、语气、可用时间、偏好和禁止干预；
- `retrieval_context`：`evidence_id/source/chunk/retrieval_score/license/version`；
- `policy_target`：`support/ask/abstain/escalate/propose/confirm/refine/monitor`；
- 可选 `action_target`：仅允许 `dry_run`，且必须 `requires_confirmation=true`；
- `labels`：安全、groundedness、empathy、personalization、consent、conciseness；
- `provenance`：human/synthetic/transformed、source ID、来源、许可、生成/提示版本、复核状态和匿名 reviewer ID。

assistant 目标本身必须是严格 JSON：

```json
{
  "response": "面向用户的非临床支持文本",
  "policy_target": "support",
  "evidence_ids": [],
  "proposed_action": null,
  "uncertainty_acknowledged": false
}
```

未知字段、未知 evidence ID、诊断/药物指令、越过动作白名单、在不确定传感状态下不承认不确定性，都会被发布门禁拦截。

## 4. 对话数据构建流程

### 4.1 冻结来源

对每个原始文件记录：文件 SHA-256、行数、空行、精确重复问题、PII 命中数、来源名和许可。许可为 `UNKNOWN/NONE/UNLICENSED` 时，`audit-source` 可以运行，但 `build` 必须失败。

```powershell
smartstress-mindcare audit-source `
  --input path/to/source.csv `
  --source owner/dataset `
  --license UNKNOWN `
  --output audits/source.audit.json
```

### 4.2 去标识化与质量处理

流水线进行 Unicode NFC、HTML 清理、空白标准化以及 email、URL、phone、handle 替换。规则匹配不是隐私保证，所有高风险内容仍需人工复核。问题或回答不足 10 个字符的记录不会进入候选集。

### 4.3 重复簇与切分

先按原始 `questionID/split_group` 合并同族记录，再按问题 3-gram Jaccard 相似度聚合近重复。对簇 ID 和固定 seed 做哈希，得到约 80/10/10 的 train/validation/test。任何变换、SFT/GRPO 导出和评估都在切分之后发生，从机制上阻止同问题不同回答或近重复跨集合泄漏。

### 4.4 场景与多轮构造

S1–S12 必须覆盖：低风险、可靠高压力、不确定、无效信号、明确 stressor、有证据 RAG、无证据/超范围、TaskRelief 提案、拒绝/修改同意、危机、个性化限制、诊断/药物边界。多轮样本保留同一个 conversation family；例如 S9 包含“提出 20 分钟计划 → 用户拒绝并要求缩短 → 系统停止并询问修改”的完整上下文。

生产构造建议保留 2–6 个对话 turn，并混合：

- 原始人类问题 + 经许可的人类回答转写；
- 专家按场景模板创作；
- 模型生成候选后由人工接受、修改或拒绝；
- 对危机、伤害、药物、诊断、未成年人样本做 100% 人工复核；
- 其他安全样本至少 20% 人工抽检。

### 4.5 发布

```powershell
smartstress-mindcare build `
  --input path/to/licensed.csv `
  --source owner/dataset `
  --license "SPDX-or-reviewed-license" `
  --output-dir artifacts/mindcare-data-v1
```

发布会生成：

- `canonical/{train,validation,test}.jsonl`；
- `sft/{train,validation,test}.jsonl`；
- `grpo/train.jsonl`，不会包含 validation/test；
- `DATA_CARD.md`；
- `manifest.json`，包含每个文件的行数和 SHA-256、来源审计、验证报告及不变量。

三个 split 任意一个为空、存在跨 split 重复、复核率不足、许可未解决或单条样本不安全，均不得发布。

## 5. QLoRA/LoRA SFT

默认基线为 Qwen2.5-3B-Instruct，但模型 ID、不可变 revision 和许可都写入运行清单。配置采用 4-bit NF4 + double quant、BF16、`target_modules="all-linear"`、LoRA rank 16（支持 8/16/32）、assistant-only loss、最大长度 2048、学习率 `1e-4`。先用 `--smoke-max-steps` 做小规模过拟合/格式检查，再进行完整训练。

```powershell
python -m smartstress_mindcare.sft `
  --train artifacts/mindcare-data-v1/sft/train.jsonl `
  --validation artifacts/mindcare-data-v1/sft/validation.jsonl `
  --dataset-manifest artifacts/mindcare-data-v1/manifest.json `
  --output-dir artifacts/adapters/mindcare-sft-v1 `
  --base-model Qwen/Qwen2.5-3B-Instruct `
  --base-model-revision <immutable-hub-commit> `
  --base-model-license Apache-2.0 `
  --rank 16 `
  --smoke-max-steps 20
```

训练入口会再次核对 train/validation 文件是否与数据清单的哈希完全一致，拒绝空文件、同一文件充当训练和验证、非 JSON assistant 目标、未解决模型许可和错误 Hub 配置。输出目录包含 adapter、tokenizer 与 checksummed `run_manifest.json`。若启用 `--push-to-hub`，必须同时提供 `--hub-model-id owner/repo` 和有写权限的 `HF_TOKEN`。

SFT 完整训练后，要在独立的隐藏测试集上生成每行 `{"completion":"..."}` 的 JSONL，并运行：

```powershell
smartstress-mindcare evaluate `
  --canonical hidden-test/canonical.jsonl `
  --completions artifacts/eval/sft.completions.jsonl `
  --output artifacts/eval/sft.report.json
```

进入 GRPO 的最低门槛是 schema parse rate ≥ 0.99 且安全违规数为 0。

## 6. GRPO 微调

GRPO 从已通过门禁的 SFT adapter 继续训练，不从 base model 直接开始。奖励组件为 groundedness、policy state、consent、personalization、helpfulness、style 和 format；诊断、药物指令、危机未升级、危机时 TaskRelief、越权或不可逆动作、忽略拒绝等为硬失败，最终 reward 固定为 `-1`，不能被其他分数抵消。

在正式 GRPO 前，必须对至少 20 条正常和对抗 completion 运行 reward audit：

```powershell
smartstress-mindcare reward-audit `
  --scored-completions artifacts/eval/reward-audit.rows.jsonl `
  --output artifacts/eval/reward-audit.report.json
```

要求：unsafe high reward 为 0，reward 与长度的绝对相关系数 < 0.5。然后运行：

```powershell
python -m smartstress_mindcare.grpo `
  --sft-adapter artifacts/adapters/mindcare-sft-v1 `
  --sft-run-manifest artifacts/adapters/mindcare-sft-v1/run_manifest.json `
  --train artifacts/mindcare-data-v1/grpo/train.jsonl `
  --dataset-manifest artifacts/mindcare-data-v1/manifest.json `
  --sft-gate-report artifacts/eval/sft.report.json `
  --reward-audit-report artifacts/eval/reward-audit.report.json `
  --output-dir artifacts/adapters/mindcare-grpo-v1 `
  --smoke-max-steps 20
```

默认使用 4 个 generations、temperature 0.8、KL beta 0.05、clip epsilon 0.2、`dr_grpo` loss 和 `remove_unused_columns=false`，以便 reward function 能接收数据集附加字段。有效本地 batch 必须能被 generation 数整除。训练完成后用相同隐藏集分别评估 Base、SFT、SFT+GRPO，并用 `smartstress-mindcare compare` 决定是否部署 GRPO；若收益未提高或安全变差，回退到 `SFT + wrapper`。

## 7. 可靠性包装器

应用层必须注入两个函数：

- `generator(messages, system_prompt) -> strict_json_string`；
- 可选 `retriever(query, k) -> Sequence[Evidence]`。

包装器行为：

1. 危机/伤害词在生成前直接 `ESCALATE`，不调用模型或 TaskRelief。
2. 提示注入在生成前 `ABSTAIN`。
3. evidence 只保留达到最低 retrieval score 的结构化项。
4. 生成严格 JSON；格式错误或可修复验证失败最多重写两次。
5. 硬安全失败立即 `ABSTAIN`。
6. 只有所有 verifier 无原因码且总分达标才 `RELEASE`。
7. action 必须来自 physio 白名单、为 dry-run、可逆、要求确认；包装器本身不执行工具。
8. 每次结果返回模型/包装器版本、reason codes、分项分数、evidence、动作白名单、重写次数和带 UTC 时间的 audit event。

## 8. 接入现有 LangGraph 与 UI

示例见 `examples/mindcare_langgraph_node.py`。在现有图中用新的可靠性节点替换直接的 `mind_care_node` 输出路径：

```text
physio_sense → physio_reliability → mindcare_reliability → meta_reflective_orchestrator
```

适配器读取现有 `conversation_history/user_preferences/rag_context/evidence_refs/current_stressor/human_confirmation_response/mindcare_allowed_tools`，写回：

- `mindcare_reliability`；
- `mindcare_response`；
- `evidence_refs`、`allowed_actions`、`policy_versions`；
- `safety_escalation`；
- `suggested_action` 与 `awaiting_human_confirmation`；
- `external_side_effects=false`、`tool_execution_mode=dry_run`；
- 追加后的 `audit_trail`。

每次调用都会先清空旧的 `suggested_action`，只有本轮通过 `RELEASE` 的新 proposal 才重新设置，避免 stale action 被误执行。
`mindcare_allowed_tools` 默认是空集合；应用必须按环境显式传入允许的工具名，不能依赖模型自己声明工具权限。

前端应消费这些结构化字段，而不是从文本猜状态。最低 UI 改造包括：可靠性状态条、信号不确定/不可用提示、证据抽屉、reason-code 调试面板、危机升级卡、TaskRelief 的确认/修改/取消按钮、模型与策略版本、重试/继续文字输入按钮。危机卡隐藏普通 action；未确认 proposal 只显示预览；`ABSTAIN` 不显示被拦截的模型原文。

## 9. 验收与上线门槛

- 数据：许可明确、哈希固定、三 split 非空、无 group/精确/近重复泄漏、高风险 100% 和普通 ≥20% 人工复核。
- SFT：严格 JSON parse ≥99%，隐藏测试安全违规 0，run manifest 和 adapter checksum 完整。
- GRPO：SFT gate 通过；reward audit ≥20 条且通过；安全不劣于 SFT；平均 reward 有提升。
- Wrapper：危机和提示注入生成前拦截；硬失败不重写；可修复错误最多两次；任何外部动作都必须先显式确认。
- 系统：Module 1 缺失时 fail-safe；每轮审计可追溯；UI 只展示 wrapper 释放内容；所有自动化测试通过。

## 10. 当前已知限制

- 本地 CounselChat 文件许可未确认；`audits/*.audit.json` 只证明具体文件哈希和静态统计，不能证明可用于训练。
- PII 和临床风险检查为规则型防线，不能替代人工复核和专业伦理审查。
- SHA-256 清单用于完整性与可复现检查，不是发布者身份的数字签名。
- 真正上线前仍需使用目标部署模型、目标 GPU、独立隐藏测试集和本地化危机资源做端到端演练。
