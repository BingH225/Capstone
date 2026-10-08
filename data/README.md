# Local data and external assets

A Git clone includes every required source package and the small frozen runtime model. It does not redistribute research participant datasets or unresolved-license dialogue corpora. Tests and offline smoke work without those assets.

## Physiological experiments

Suggested layout (ignored by Git):

```text
data/WESAD/S2/S2.pkl ... S17/S17.pkl
data/StressID/labels.csv and dataset-specific signal files
research/physio/Data_Processed/WESADECG_S2.json ... WESADECG_S17.json
research/physio/Data_Processed_StressID/STRESSIDECG_*.json
research/physio/Models_CrossVal_Full/fold_0_Attention/epoch_49.pth ...
research/physio/Models_CrossVal_Full/fold_0_DNN/epoch_49.pth ...
```

Obtain raw WESAD/StressID through their dataset owners and applicable access conditions. Import existing processed JSON/full LOSO checkpoints from your authorized research storage, or regenerate using the preserved research scripts. Do not replace the runtime S17 manifest/checkpoint with full-fold weights.

The 15-subject policy experiment needs 15 `WESADECG_<subject>.json` files and fixed epoch-49 weights for the selected architectures. Subject order is explicit in the experiment script; fold-to-subject mapping must match it. Its `--model-repo` can point to another authorized local asset root with the same layout. The script reports a missing file before attempting an incomplete evaluation.

Legacy model scripts load `.env` in their working directory. Copy `research/physio/.env.example` to `.env` in that directory and run legacy training scripts from `research/physio`. Use relative paths or paths configured for the current machine. Legacy scripts which concatenate directory strings need trailing `/` as shown in the template. Their historical feature definitions/epoch selection are not a validated replacement for a new nested training protocol.

## MindCare and RAG

Raw licensed dialogue sources may be stored under `data/mindcare/`. Governed exports, adapters and run manifests belong under `artifacts/`. Keep API credentials in root `.env` or environment variables.

CounselChat audit reports remain under `audits/`, with unresolved licenses unchanged. The migration does not make those files eligible for training. Follow the production data gates in `MINDCARE_MODULE.md`.

## Reproducibility distinction

Source/tests/demo are clone-ready. Rerunning participant-data experiments requires the corresponding locally supplied datasets and full checkpoints; GPU training additionally requires hardware and a pinned base model. Existing aggregate results and per-fold policies are included as inspectable evidence, not as a substitute for those inputs.
