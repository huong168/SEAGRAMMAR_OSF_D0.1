# Baselines

Three baselines, all scored on exactly the same splits and labels as the rule model. See
`../protocol/protocol.md` §6 for why each is here.

| File | Provides |
|---|---|
| `baselines_tests.py` | Tanimoto-kNN (k = 5), frequency, and the public/terrestrial-trained baseline; plus the reusability, compositionality and predictivity test harness |
| `_core.py` | the shared primitives: pure-numpy logistic regression fallback, PAV isotonic calibration, grouped K-fold, Brier, Jensen–Shannon, weighted Jaccard, cosine, Tanimoto, paired cluster bootstrap |
| `honest_eval.py` | leave-compound-out vs leave-family-out in parallel, PR-AUC + lift primary, ROC-AUC secondary, cluster-bootstrap CIs |
| `calibrate_eval.py` | out-of-fold isotonic calibration and per-label reliability curves |

**The Tanimoto-kNN fingerprint is boolean.** `_core.tanimoto()` casts to `bool`, so the similarity
baseline reads substructure *presence* whether the feature table stores presence or match counts.
That is deliberate: it holds the baseline fixed across vocabulary versions, so any change in the
rule-vs-similarity comparison is attributable to the rule model. It is also why the baseline scores
identically (0.0810 sealed-6 Brier) in all three deposited runs.

**On simulated labels the similarity baseline is an equally-informed competitor**, because it reads
the same substructure bits the noisy-OR generator used. Do not read it as a weak baseline until real
LC-MS labels exist.
