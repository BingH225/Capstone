# Historical RAG experiment code

The original Agent's A/B query generation, evaluation and report scripts are retained here for reference. Install the root backend/research dependencies and run scripts with explicit paths. Optional BERTScore evaluation additionally needs `bert-score` installed on the evaluation environment.

The original `test_queries.json` is not imported: its CounselChat provenance/license and overlap with RAG documents were unresolved. Generate a newly reviewed, isolated evaluation set before running these experiments. New MindCare evaluation should follow `docs/DEVELOPMENT.md`, not reuse old corpus answers as hidden tests.
