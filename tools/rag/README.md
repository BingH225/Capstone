# RAG utilities

After a source has been licensed and its train/evaluation split reviewed, place local CSV inputs under `data/mindcare/`. The converter writes to ignored `artifacts/rag_docs/counselchat/`; the TiDB importer reads that directory. Both import the root-installed Python package. Passing a relative `--csv` path to the converter resolves against the working directory.

These are historical utilities; they do not perform the governed MindCare production release checks. Use `smartstress-mindcare build` for training data and keep held-out questions/answers out of the RAG index. TiDB ingestion requires explicit backend credentials and is a separate manual operation.
