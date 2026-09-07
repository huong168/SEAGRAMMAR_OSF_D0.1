#!/usr/bin/env bash
# Reproduce the registered run (v4b: presence bits plus additive match counts).
# Requires: pip install -r framework/requirements.txt  (RDKit is NOT needed here)
set -euo pipefail
cd "$(dirname "$0")/pilot/run_v4b_registered" 2>/dev/null || true
cat <<'EOF'
The deposited run_v4b_registered/ directory contains the completed run: dataset,
logs, per-compound and per-class results, and figures.

To re-execute from scratch, copy framework/pipeline/ and framework/rules/ into a
working tree laid out as the pipeline expects, then:

  export SEAGRAMMAR_VOCAB=counts_plus
  export SEAGRAMMAR_FEATURE_TABLE=_features_counts_plus.csv
  cd scripts
  python make_panel_inputs.py
  python build_dataset_v3.py
  python train_model_v2.py
  python ml_upgrade/calibrate_eval.py
  python ml_upgrade/baselines_tests.py
  HONEST_EVAL_NBOOT=500 python ml_upgrade/honest_eval.py
  python pilot_sealed_eval.py
  python pilot_figure_table.py

To regenerate the feature tables from SMILES (needs rdkit==2026.03.6):
  python framework/vocabulary/make_count_features.py <path-to>/data/inputs
EOF
