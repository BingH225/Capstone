# SmartStress Capstone

这是 Capstone 的统一开发仓库。生理可靠性、MindCare 数据与微调、原有 Agent 后端、React 前端和生理研究脚本都已收进本仓库，无需另外克隆旧的三个仓库，也没有 Git submodule。

## 仓库结构

```text
Capstone/
├─ src/
│  ├─ smartstress_policy_reliability/  # DNN 后校准、质量/OOD、拒识和时间策略
│  ├─ smartstress_mindcare/            # 数据、SFT/GRPO、奖励、回复验证与包装器
│  └─ smartstress_langgraph/           # 迁入的 PhysioSense/RAG/编排/API
│     └─ physio/artifacts/             # 随代码发布的 S17 checkpoint + SHA-256 manifest
├─ apps/web/                          # React + Vite 前端及 npm lockfile
├─ research/physio/                   # WESAD/StressID 预处理、训练、评估和适配
├─ experiments/                       # WESAD 可靠性实验及历史 RAG 实验代码
├─ tools/rag/                         # 本地语料转换、TiDB 导入工具
├─ tests/                             # 可靠性和迁入后端测试
├─ scripts/                           # 仓库自检、离线 smoke
├─ examples/                          # 两个可靠性节点的接入示例
├─ reports/wesad_policy_reliability/   # 已冻结的真实 WESAD 结果
├─ docs/                              # 计划、开发交接、迁移来源
├─ data/                              # 本地数据约定；真实数据默认不提交
└─ .github/workflows/ci.yml           # Python 检查、后端测试、前端构建
```

## 新环境快速开始

推荐 Python 3.11、Node.js 22、Git。先克隆默认 `main` 分支：

```bash
git clone https://github.com/BingH225/Capstone.git
cd Capstone
python -m venv .venv
```

Windows PowerShell：

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS：

```bash
source .venv/bin/activate
```

如果 PowerShell 不允许激活，可以直接使用 `.\.venv\Scripts\python.exe` 执行下面的 Python 命令。

### 只开发可靠性和数据模块

不需要 GPU、API key 或外部数据库：

```bash
python -m pip install -c requirements/validated-constraints.txt -e ".[test]"
python scripts/check_checkout.py
python -m pytest -q --ignore=tests/agent
python scripts/offline_smoke.py
```

### 开发完整后端

CPU 环境可先安装 CPU PyTorch，避免下载 GPU 运行库；有 GPU 时按目标 CUDA 环境安装 PyTorch。

```bash
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -c requirements/validated-constraints.txt -e ".[backend,research,test]"
python -m pytest -q
python scripts/offline_smoke.py --backend
```

已验证的关键依赖版本记录在 `requirements/validated-constraints.txt`。上述安装命令默认使用该约束文件；GPU 训练依赖单独安装。

离线 smoke 使用明确标记的本地测试回复，并调用随仓库保存的真实 DNN 权重；不调用云模型或 TiDB，不把测试回复当成微调模型的效果。

## 启动后端和前端

将 `.env.example` 复制为 `.env`。`.env` 和所有 API key 都不提交；shell 环境变量优先于 `.env`。实时 Gemini 对话需要设置 `GOOGLE_API_KEY`，模型名可由环境变量覆盖。未配置 TiDB 时检索返回空结果；未配置模型时现有 Agent 使用其降级回复。

仓库根目录启动 API：

```bash
smartstress-server
# 或 python server.py
```

默认地址 `http://127.0.0.1:8000`；Swagger 在 `/docs`，健康检查在 `/health`。

另一个终端启动前端：

```bash
cd apps/web
npm ci
npm run dev
```

打开 Vite 输出的本地地址。开发模式通过 `/api` 代理访问后端；可用 `VITE_API_BASE_URL` 显式指定另一 API origin。

构建前端：

```bash
cd apps/web
npm run build
```

回到仓库根目录重新启动后端，它会自动托管 `apps/web/dist`。Python wheel 包含后端包和小型 DNN checkpoint；React 源码和研究脚本通过 Git 仓库交付。

## 数据、模型与实验

- 已包含：运行 PhysioSense 所需的冻结 S17 checkpoint、校验 manifest、测试 fixture，以及真实 WESAD 可靠性报告。
- 按本地配置提供：WESAD/StressID 原始数据、全体 LOSO checkpoints、正式 MindCare 语料和训练 adapter；参见 [数据说明](data/README.md)。
- CounselChat 数据许可尚未确认。仓库保留来源审计和代码；未将原始语料或其派生测试问题发布为可训练数据。
- 运行 SFT/GRPO 前在目标 GPU 环境安装 `python -m pip install -e ".[train]"`，并遵守数据、评估和 adapter 门禁；参见 [MindCare 模块](MINDCARE_MODULE.md)。

真实 WESAD 复现（准备所需数据/权重后）：

```bash
python experiments/wesad_policy_reliability_experiment.py --model-repo research/physio --output-dir artifacts/wesad-policy-rerun
```

已冻结报告在 `reports/`；新实验写进 `artifacts/`，避免覆盖原始证据。

## 当前阶段与下一步

两个可靠性模块和工程入口已完成，WESAD 真实实验已完成。正式 MindCare 训练尚未进行。迁入的 `smartstress_langgraph` 保留原有工作流，两个新可靠性模块的完整接线和前端状态改造仍是下一阶段工作。本次迁移没有将它们声明为已经接好。

建议先建立独立评估集及旗舰 API/未微调小模型基线，再按错误类型构建训练数据和开展 LoRA；根据实测收益决定是否加入 GRPO。

- [开发交接与待办](docs/DEVELOPMENT.md)
- [迁移范围与来源](docs/REPOSITORY_MIGRATION.md)
- [详细项目计划](docs/SmartStress_Capstone_Detailed_Project_Plan.md)
- [生理可靠性模块](POLICY_RELIABILITY.md)
- [WESAD 实验报告](reports/wesad_policy_reliability/report.md)

SmartStress 是非临床学术原型；所有 TaskRelief 动作保持明确确认和 dry-run。
