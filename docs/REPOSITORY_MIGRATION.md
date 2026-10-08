# Repository consolidation — 2026-10-08

## Imported sources

This private Capstone repository now contains the source needed to continue development without sibling repositories:

| Source | Destination |
|---|---|
| `BingH225/smart-stress-agent`, package and server | `src/smartstress_langgraph/` |
| Original Agent tests and golden fixture | `tests/agent/` |
| Original RAG experiment Python scripts | `experiments/rag/` |
| Corpus conversion/ingestion utilities | `tools/rag/` |
| `cecildelakers/smart-stress-ui`, React sources, public fixtures, npm lockfile | `apps/web/` |
| `BingH225/smart-stress-model`, preprocessing/training/evaluation/adaptation | `research/physio/` |
| Current Capstone reliability implementation, tests, plans and reports | Existing root `src/`, `tests/`, `docs/`, `reports/` |

`source_import_manifest.json` records upstream URLs, branches, exact commits, original paths and pre-migration SHA-256 hashes. Imported files may then be intentionally adapted; these source hashes describe provenance rather than current-file integrity. Two locally available LOSO training scripts were previously excluded from the model repository's tracked files; their import hashes are also recorded.

## Portability changes

- Root `pyproject.toml` owns installable Python packages, optional backend/research/training/test dependencies and CLI entry points.
- The model manifest and small S17 checkpoint are Python package data, included in built wheels.
- Runtime configuration uses environment variables/root `.env`; shell settings take priority. Gemini defaults to its official endpoint, with an explicit optional gateway setting.
- SQLite runtime data defaults to ignored `artifacts/runtime/`; frontend hosting defaults to this checkout's `apps/web/dist`.
- Research CLI defaults use repository-relative paths. WESAD reliability evaluation no longer defaults to a Windows sibling checkout.
- Vite proxies `/api` to the local backend. Text-only startup omits the obsolete `values: {hr:75}` payload that the typed backend rejects.
- The unused browser-side Dify key client was removed. LLM credentials are backend configuration.
- The former standalone server remains accessible through `python server.py` and the `smartstress-server` entry point.
- Fixed graph node imports so prior submodule imports cannot turn a node function into a module-valued package attribute.
- CI and offline smoke support clean checkouts without cloud credentials or external corpora.

## Exclusions

The import intentionally excludes upstream Git metadata/submodule checkout, credentials, source `.env` files, local databases, raw/processed participant data, full checkpoint archives, build caches, machine-specific HPC submission scripts and unresolved-license CounselChat corpus/question exports. Original repositories are unchanged.

The project plan's references to old local paths are retained as historical provenance. Active code and new developer instructions use this repository's paths. See `data/README.md` for external assets and `DEVELOPMENT.md` for remaining integration work.

No new public license is inferred from upstream README badges; original authorship and provenance are retained. A redistribution license should be reviewed before any public release.
