# SEAGRAMMAR pipeline v3 — how to run

Self-contained copy of the v3 ML pipeline (CSV-driven). Directory layout is preserved,
so scripts run as-is from this folder.

## Setup
```
pip install -r requirements.txt        # scikit-learn, pandas, numpy, rdkit, matplotlib
```

## Run order
```
# 1) build the training table from data/inputs/*.csv  ->  data/ml_training_dataset_v3.csv + DATA_CARD_v3.json
python3 build_dataset_v3.py

# 2) train + honest evaluation (leave-compound-out)   ->  visualizations/model_v3/
python3 train_model_v2.py

# 3) isotonic calibration                             ->  visualizations/ml_upgrade/
python3 ml_upgrade/calibrate_eval.py

# 4) baselines + pre-registered tests A/B/C           ->  visualizations/ml_upgrade/baselines_tests_report.json
python3 ml_upgrade/baselines_tests.py

# 5) figures + pipeline diagram
python3 ml_upgrade/make_figures.py
python3 ml_upgrade/make_pipeline_diagram.py

# 6) honest evaluation: PR-AUC+lift (primary), ROC-AUC (secondary), LCO vs LFO, cluster-bootstrap CI
python3 ml_upgrade/honest_eval.py            # tune bootstrap: HONEST_EVAL_NBOOT=2000 (default)

# optional) LC-MS peaks -> observations.csv (real-label path), demo mode:
python3 msms_to_observations.py --demo
# aux: RF vs NaiveBayes, ROC curves
python3 test_bayes.py
python3 visualize_roc_v2.py
```

## Notes for the coder
- Data dir is auto-detected (`data/`); override with env `SEAGRAMMAR_DATA=/path/to/data`.
- `ml_upgrade/*.py` all import the shared `ml_upgrade/_core.py` (pure-numpy fallbacks when
  sklearn/rdkit are absent; RandomForest + Morgan FP when present).
- **Labels are SIMULATED (noisy-OR)** — all metrics are a positive control (rule recovery),
  not biology. See `SEAGRAMMAR_Algorithm_Spec.md` and `data/README_dataset_v3.md`.
- Evaluation gate: a label is scored only if it has >=8 positive rows AND its substructure
  spans >=3 distinct compounds (protects leave-compound-out). Report **PR-AUC + lift** primary,
  ROC-AUC secondary, and both leave-compound-out and leave-family-out in parallel.
