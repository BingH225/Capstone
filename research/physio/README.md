# Physiological research sources

Migrated WESAD/StressID research scripts live here. Runtime inference remains in `src/smartstress_langgraph/physio`; do not maintain a second runtime package here.

Install root `.[research]` and see `../../data/README.md` for asset layout. Run legacy scripts from this directory with `.env` copied from `.env.example`. CLI scripts accept explicit input/output paths; use `--help` before preparing new data.

`Cross_validation_ablation.py` and `Cross_validation_ablation_100hz.py` preserve the full historical LOSO training implementations. They may select epochs on held-out subjects; use nested non-test selection for new scientific claims. The capstone reliability experiment uses fixed epoch 49.

The stepwise StressID evaluator and subject-aware adaptation/asset scripts are retained. Account-specific PBS/SSH submission automation is excluded; invoke Python on your GPU environment using explicit local input paths.

Existing follow-up plans in `docs/` document prior research decisions. They are not proof of a completed external reliability experiment.
