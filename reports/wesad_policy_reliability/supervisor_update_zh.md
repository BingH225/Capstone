# DNN 后 Policy Reliability 模块阶段汇报

## 一句话结论

模块已经完成可运行版本，并已在真实 WESAD 数据上完成 15 位受试者的
LOSO（leave-one-subject-out）实验。它的主要价值不是把所有窗口的 DNN
准确率直接抬高，而是在约保留 84% 窗口的条件下，拒绝不可靠输入、降低已放行样本的错误风险，
并把连续阳性窗口合并为少量可执行提醒。实验也暴露出一个必须解决的问题：S6
几乎被 OOD 门全部拒绝，说明当前跨人群分布门在安全性上有效、但个别受试者的可用性不足。

## 已完成内容

Policy Reliability 位于冻结的生理 DNN/Attention-DNN 与 MindCare Agent 之间，已实现：

1. 12 维特征契约、有限值和量纲范围检查；
2. temperature calibration；
3. 基于训练参考受试者的 robust diagonal Mahalanobis OOD gate；
4. split conformal 与最低置信度联合拒识；
5. `3-of-5`、双阈值 hysteresis 和 15 分钟 cooldown；
6. `RELIABLE_LOW`、`RELIABLE_ELEVATED`、`OOD_OR_UNCERTAIN`、`DATA_INVALID`
   等显式状态、reason code、policy manifest 和审计记录；
7. LangGraph/PhysioSense 接口适配，以及保存、加载和损坏校验。

模块测试目前为 33 项通过，其中 Policy Reliability 19 项、MindCare 14 项。

## 实验设计

- 数据：WESAD 15 位受试者，共 43,457 个已处理窗口；stress 为正类，其余状态为负类。
- 基模型：Standard DNN 与 Attention DNN。
- 外层评估：每次完整留出 1 位受试者。
- DNN 权重：固定读取每折 `epoch_49.pth`，没有用测试受试者选择最佳 epoch，避免原训练脚本中的测试集选 epoch 泄漏。
- Policy 拟合：每个外层折把其余 14 位受试者确定性分成 7 位 reference 与 7 位 calibration；评估受试者不参与任何 Policy 拟合或调参。
- 消融顺序：raw threshold → calibration → quality/OOD → selective prediction → temporal policy。
- 固定参数：conformal `alpha=0.10`、最低置信度 `0.60`、OOD 分位数 `0.99`、
  `3-of-5`、on/off 阈值 `0.65/0.45`、cooldown `900s`。

## 核心结果

下表的 Accuracy/F1 是“已放行窗口”的指标，因此必须同时报告 Coverage；不能把它说成全样本准确率提升。

| 模型 | 方法 | Coverage | Accuracy | F1 | Selective risk |
|---|---|---:|---:|---:|---:|
| Attention DNN | 原始阈值 | 100.00% | 90.72% | 80.56% | 9.28% |
| Attention DNN | 完整 Policy | 84.44% | 93.63% | 85.70% | 6.37% |
| Standard DNN | 原始阈值 | 100.00% | 90.33% | 80.12% | 9.67% |
| Standard DNN | 完整 Policy | 83.77% | 93.45% | 85.66% | 6.55% |

相对原始阈值：

- Attention DNN：coverage 减少 15.56 个百分点，已放行样本错误率下降 2.90
  个百分点，F1 增加 5.15 个百分点；
- Standard DNN：coverage 减少 16.23 个百分点，已放行样本错误率下降 3.13
  个百分点，F1 增加 5.53 个百分点；
- 受试者 coverage 中位数分别为 93.43% 和 90.23%，但两种模型的 S6 coverage
  都只有 0.035%，几乎完全被 OOD gate 拒绝；
- 在 coverage 至少 50% 的 14 位受试者中，Attention DNN 的 accepted risk 在
  14/14 位上改善、F1 在 12/14 位上改善；Standard DNN 分别为 12/14 和 10/14；
- 三类受控异常——非有限值、极端量纲、错误特征顺序——各 1,500 次检查均实现
  100% 拒绝。

## 校准和提醒结果应如何解释

temperature scaling 没有改变 0.5 分类标签，因为该变换保持 0.5 决策边界。
Standard DNN 的 ECE 从 5.99% 降到 5.47%，但 Attention DNN 从 7.12% 升到
7.28%，NLL/Brier 也没有改善。因此目前不能声称“校准稳定有效”；下一步应比较
identity、temperature 与 Platt，并只按非测试受试者上的指标选择校准器。

完整时序策略将误触发从 Attention 的 2,716 个“假阳性窗口”压缩为 19 个
“假提醒事件”，Standard DNN 从 3,012 个压缩为 22 个，对应约 99.3% 的下降。
这个数字包含拒识、连续窗口合并和 cooldown，比较的两端口径并不相同，适合说明
“相对每个阳性窗口都推送的朴素方案，提醒负担显著下降”，不能说成分类器假阳性率下降 99.3%。

## 建议向导师直接这样汇报

> 第一部分已从设计稿推进到可运行和可复现的实现，并完成了真实 WESAD LOSO
> 实验。我们没有用 held-out subject 选择 epoch；每个测试受试者也完全不参与
> Policy 的 reference、calibration 或阈值拟合。结果显示，这个模块体现的是
> selective reliability，而不是无条件提升 DNN：Attention 模型在保留 84.44%
> 窗口时，已放行样本 F1 从 80.56% 提升到 85.70%，错误率从 9.28% 降到
> 6.37%；Standard DNN 的趋势相近。受控坏输入可以 100% 拒绝，时序门能显著
> 合并重复提醒。不过 S6 几乎全被 OOD gate 拒绝，且 Attention 的温度校准没有
> 改善 ECE。这两个负结果说明模块已经能安全 abstain，但还需要做阈值敏感性、
> 校准器选择和个体基线适配，才能冻结部署阈值。

## 下一步实验

1. 只在非评估受试者上做嵌套选择，比较 identity/temperature/Platt；
2. 报告 OOD quantile、最低置信度与 conformal alpha 的 coverage-risk 敏感性曲线；
3. 针对 S6 分解各特征的 OOD 距离贡献，测试少量个体 neutral baseline 是否恢复 coverage；
4. 从原始时间戳重建连续时间轴，按 episode 统一比较 raw 与 Policy 的 event-level
   alert precision、false alerts/hour 和 detection delay；
5. 在 StressID 上做完全不调阈值的外部验证，检验跨数据集拒识和风险控制。

## 复现入口与证据

- 实验脚本：`experiments/wesad_policy_reliability_experiment.py`
- 完整英文报告：`reports/wesad_policy_reliability/report.md`
- 聚合结果：`reports/wesad_policy_reliability/summary.json`
- 逐受试者结果：`reports/wesad_policy_reliability/per_subject.csv`
- OOF 预测：`reports/wesad_policy_reliability/oof_predictions.csv`
- 每折 Policy manifest：`reports/wesad_policy_reliability/policies/`
- 图：`reports/wesad_policy_reliability/policy_tradeoff.png`

复现命令：

```powershell
conda run --no-capture-output -n torch python experiments\wesad_policy_reliability_experiment.py
```
