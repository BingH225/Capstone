# SmartStress Post-DNN Policy Reliability

该包实现冻结版 WESAD DNN 之后、MindCare Agent 和 UI 之前的可靠性决策层。它不重新训练 DNN，也不把单次 `sigmoid >= 0.5` 当作可直接执行的压力判断，而是把模型输出转换为可校准、可拒识、可追踪、可审计的策略决策。

## 决策流水线

```text
12维生理特征 + DNN raw probability/logit
  -> 输入/模型版本校验
  -> 概率校准（identity / temperature / Platt）
  -> 特征质量门控
  -> 鲁棒对角 Mahalanobis OOD 检测
  -> split-conformal prediction set + confidence abstention
  -> session级 k-of-m + hysteresis + cooldown
  -> versioned ReliabilityDecision
```

优先级是 fail-safe 的：`DATA_INVALID` 高于 `OOD_OR_UNCERTAIN`，二者均高于时序判定。只有 `RELIABLE_ELEVATED` 可以开放 `support` 和 `propose_dry_run`；`propose_dry_run` 只表示允许下游向用户提出预览，不表示允许自动发送消息或执行外部动作。

## 固定输入契约

特征顺序必须严格等于：

```text
mean_hr, std_hr, tinn, hrv_index, nn50, pnn50,
mean_hrv, std_hrv, rmssd, fft_mean, fft_std, sum_psd
```

每条推理输入必须提供：

- 上述 12 维有限数值特征；
- `raw_probability` 或 `raw_logit`，且只能提供一个；
- 与 manifest 相同的 `model_id`，默认 `wesad_attention_v1`；
- `session_id` 和非递减的 ISO-8601 `timestamp`；
- 可选的 `[0, 1]` 外部信号质量分数。
- 可选的固定 `feature_names`、`input_source`、`baseline_version`、SHAP `top_drivers`；
- 若有原始 ECG 质量证据，可在 `signal_quality` 提供 `window_seconds`、`sampling_rate_hz`、`rpeak_success`、`abnormal_rr_fraction`、`valid_fraction`、`flatline_fraction`、`saturation_fraction`。提供后即按 manifest 中的门限 fail closed。

输出 `ReliabilityDecision` 包含原始/校准概率、质量分、OOD 分数及阈值、conformal prediction set、可靠性状态、动作白名单、reason codes、policy/model 版本、时间、session 和诊断字段。

## 数据准备与无泄漏拟合

CLI 接受 `.jsonl` 或 `.csv`。特征可放在 JSON 数组字段 `features` 中，也可使用 12 个独立同名列。

Reference 数据只用于拟合质量/OOD 分布，至少 20 行：

```json
{"subject_id":"S01","features":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]}
```

Calibration 数据至少 20 行，必须同时包含两类标签，并提供完整 `raw_probability` 或 `raw_logit`：

```json
{"subject_id":"S11","features":[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],"label":1,"raw_logit":1.42}
```

默认要求 reference、calibration 和 evaluation 都有 `subject_id`，并拒绝 reference/calibration 重叠以及 evaluation 与拟合 provenance 重叠。只有在已有独立、书面记录的非 subject-wise 协议时才应使用两个 `--allow-*` 逃生参数。

```powershell
python -m smartstress_policy_reliability fit `
  --reference data/reference.jsonl `
  --calibration data/calibration.jsonl `
  --output artifacts/physio_reliability_policy.json `
  --calibrator temperature `
  --alpha 0.10 `
  --minimum-confidence 0.60 `
  --window-size 5 `
  --required-elevated 3 `
  --threshold-on 0.65 `
  --threshold-off 0.45 `
  --cooldown-seconds 900
```

个体基线参与特征标准化时应加入 `--require-baseline-version`。同一 session 首次通过输入校验后会绑定该版本；后续缺失或更换版本会输出 `DATA_INVALID / BASELINE_VERSION_MISMATCH`，必须显式 reset session 后才能切换。

manifest 使用规范化 JSON 的 SHA-256 校验和，记录配置、12 维特征顺序、各拟合组件、输入文件哈希、行数和经过哈希的 subject IDs。加载时会拒绝未同步更新校验和的损坏或改动；这提供完整性检查而不是发布者身份认证，正式发布仍应使用受保护的制品仓库或额外数字签名。

## 独立评估

```powershell
python -m smartstress_policy_reliability evaluate `
  --policy artifacts/physio_reliability_policy.json `
  --data data/held_out_test.jsonl `
  --output artifacts/held_out_reliability_report.json
```

报告同时保存校准前后 NLL/Brier/ECE、校准斜率/截距、可直接绘图的 reliability bins、conformal empirical coverage、singleton rate、risk-coverage AURC、状态计数、有效 coverage、selective risk、主动提醒数和 false proactive alerts/hour，以及逐条完整决策。

真实 WESAD subject-wise held-out 实验现已完成，入口为
`experiments/wesad_policy_reliability_experiment.py`，结果见
`reports/wesad_policy_reliability/report.md`。该实验固定使用每折 `epoch_49`
checkpoint，避免依据 held-out 受试者选择 epoch；每个外层折再把其余受试者分成互斥的
reference/calibration 两组。报告中的选择后 Accuracy/F1 必须始终和 coverage 一起解读，
不能表述为全样本分类性能提升。

建议门槛必须依据 held-out 结果冻结，而不是直接采用默认值：

- 校准后 NLL/Brier 至少不劣于校准前，并检查 reliability diagram；
- conformal empirical coverage 接近 `1-alpha`，同时报告有效 coverage 和 selective risk；
- OOD/质量异常样本不会进入支持或执行候选动作；
- 对 k-of-m、hysteresis、cooldown、跨 session 隔离和乱序时间做回放测试；
- manifest 与 DNN checkpoint/model ID 成对版本化。

## PhysioSense / LangGraph 接入

现有状态字段可直接适配：

```python
from smartstress_policy_reliability import (
    ReliabilityPolicy,
    apply_reliability_to_physio_state,
)

policy = ReliabilityPolicy.load("artifacts/physio_reliability_policy.json")

def physio_reliability_node(state):
    return apply_reliability_to_physio_state(state, policy)
```

该节点读取 `physio_features`、`current_stress_prob`、`physio_model_id`、`physio_quality_score`、`physio_timestamp`/`stress_timestamps` 和 `session_id`，返回：

- `calibrated_stress_prob`；
- `physio_reliability` 完整审计对象；
- `physio_reliability_state`；
- `physio_allowed_actions`；
- `physio_policy_version`；
- `physio_audit_event`，结构与现有 `audit_trail` 的 timestamp/node/summary/details 约定一致且不复制原始特征；
- 兼容字段 `stress_detected`，仅在 `RELIABLE_ELEVATED` 时为 `True`。

接线时应把该节点放在 frozen DNN 后，并让 MindCare policy 和 UI 只消费这里生成的状态与动作白名单，禁止继续读取旧的 `current_stress_prob >= 0.5` 作为直接执行依据。完整示例见 `examples/physiosense_reliability_node.py`。

## UI 状态语义

| 状态 | UI 建议 | 允许动作 |
|---|---|---|
| `DATA_INVALID` | 显示传感器/输入不可用，允许重试或改用文字 | `retry_sensor`, `continue_by_text` |
| `OOD_OR_UNCERTAIN` | 明确显示“不确定”，请求确认或继续监测 | `monitor`, `ask_user`, `continue_by_text` |
| `MONITOR` | 显示趋势仍在确认中，不主动提醒 | `monitor`, `ask_user` |
| `RELIABLE_LOW` | 正常低风险状态 | `monitor`, `continue_by_text` |
| `RELIABLE_ELEVATED` | 展示支持选项及可撤销 dry-run | `monitor`, `ask_user`, `support`, `propose_dry_run` |

## 安全边界

本模块是工程可靠性与研究原型，不是医疗器械、诊断工具或紧急服务。压力概率不能解释为疾病概率。高风险文本、自伤风险、医疗症状和紧急情形必须由独立的文本安全策略及人工/紧急资源流程处理；生理模块不得覆盖它们。任何外部消息、预约或联系人操作仍需单独生成 dry-run、展示目标与内容并取得用户明确确认。

## 开发验证

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

测试覆盖校准、质量、OOD、conformal abstention、k-of-m、hysteresis、cooldown、session 隔离、失效安全优先级、manifest 防篡改、CLI 数据泄漏拦截和 PhysioSense 适配。
