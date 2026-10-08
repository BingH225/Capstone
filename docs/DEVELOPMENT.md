# Developer handoff

## Development baseline

The authoritative development location is this repository. Install its root Python package in editable mode, and use `apps/web` for the UI. Upstream repositories are provenance references, not runtime dependencies. Source commit IDs and import-time file hashes are in `source_import_manifest.json`.

Python 3.11 and Node 22 are the CI targets. Run the instructions in the root README from the checkout root. Fast reliability/data tests need only `.[test]`; the complete inherited agent suite needs `.[backend,research,test]`. Training is an optional GPU extra.

Use `python scripts/check_checkout.py` for checkout integrity and `python scripts/offline_smoke.py --backend` for a credential-free API/model smoke. The latter injects clearly identified deterministic test responses and disables retrieval only inside that process.

## Where to edit

| Work | Location |
|---|---|
| Physiological decision contract, calibration, quality/OOD, abstention, temporal policy | `src/smartstress_policy_reliability/` |
| Dialogue schema, split/export, SFT/GRPO, reward, wrapper | `src/smartstress_mindcare/` |
| Runtime graph, API schema, agent nodes, checkpointing | `src/smartstress_langgraph/` |
| React UI and API client | `apps/web/src/` |
| Frozen-model feature extraction and inference | `src/smartstress_langgraph/physio/` |
| WESAD/StressID research and training | `research/physio/` |
| Reliability evidence generation | `experiments/wesad_policy_reliability_experiment.py` |
| Immutable existing results | `reports/wesad_policy_reliability/` |

## Next tasks, in order

1. Freeze an independent MindCare evaluation set and establish flagship API plus untuned-small-model baselines with equal RAG, policy context and wrapper inputs. The latest discussion suggested 200–300 evaluation scenarios before large-scale synthetic training data.
2. Resolve corpus provenance/licenses, human review and duplicate-family isolation; generate governed SFT/GRPO exports. S1–S12 fixtures validate mechanics, not training quality.
3. Run SFT and compare against both baselines. Add GRPO only if independent evaluation supports a measurable benefit. No production adapter is currently included.
4. Integrate the new boundaries into `graph.py`: `physio_sense -> physio_reliability -> mindcare_reliability -> orchestrator`. Extend `state.py`, `io_models.py` and `api.py` together. Start from the adapters in each reliability package and the examples; persist per-session temporal state with explicit isolation.
5. Update React to consume `physio_reliability`, `mindcare_reliability`, evidence, reasons and action permissions. Replace/label mock dashboards and demo forecasts. Preserve confirmation/refinement/cancel paths.
6. Investigate S6 OOD rejection, compare calibrators on non-test subjects, unify event-level alert metrics and conduct external StressID validation.

## Important open boundaries

- The inherited runtime still calls the original MindCare node. Standalone reliability adapters exist, but consolidation does not automatically change the research pipeline.
- Old UI charts are mock data. Chat now starts a legal text-only session; it does not fabricate a heart-rate sensor payload.
- RAG tools and historical RAG experiments are retained for research. Production train/evaluation/index splits must be rebuilt under the governed protocol; they are not a shortcut around licensing or leakage controls.
- Some legacy physiological scripts select epochs using held-out data. Preserve them as historical implementations; use the fixed-epoch reliability experiment for the documented result, and design nested selection for new training claims.
- Reported selective metrics must include coverage. S17 runtime-checkpoint performance is not the 15-subject LOSO average.

## Verification and artifact rules

Run `python -m pytest -q` and `npm ci && npm run build` (the npm commands from `apps/web`) after related changes. CI runs these on Linux. Add focused tests for new policy/runtime behavior. New experiment outputs go to `artifacts/`; publish only reviewed results and their manifests into `reports/`.

Keep `.env`, databases, raw/processed corpora, checkpoints and GPU adapters outside Git. The deliberately bundled 267 KB S17 checkpoint is the runtime exception, protected by its SHA-256 manifest and packaged into wheels. Never copy another checkout's `.git`, `.venv`, `node_modules`, secret files or submodules into this repository.
