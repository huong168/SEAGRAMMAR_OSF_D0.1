# Results of the registered run

**All labels are simulated** (noisy-OR from the rule priors). Every number below is rule
recovery, not biology. Registered variant: **v4b, presence bits plus additive match counts**.
Seed 42. Python 3.11.15 / scikit-learn 1.8.0 / pandas 3.0.2 / numpy 2.4.4, no RDKit on the
evaluation path. Confidence intervals from 500 cluster-bootstrap resamples over compounds.

The v3 column reproduces the 17/08/2026 run character for character, so the columns differ by
vocabulary alone: the simulated labels are bit-identical across all three.

## Headline

| Metric | v3 presence | v4a counts | **v4b registered** |
|---|---|---|---|
| Rule recovery, leave-compound-out — macro PR-AUC | 0.366 | 0.345 | **0.367** |
| Rule recovery, leave-compound-out — lift over prevalence | 8.7× | 7.7× | **8.1×** |
| Rule recovery, leave-compound-out — ROC-AUC | 0.851 | 0.843 | **0.854** |
| Rule recovery, leave-family-out — macro PR-AUC | 0.229 | 0.245 | **0.266** |
| Rule recovery, leave-family-out — lift over prevalence | 5.8× | 6.6× | **6.9×** |
| Rule recovery, leave-family-out — ROC-AUC | 0.705 | 0.681 | **0.691** |
| **Scaffold dependence** (LCO − LFO PR-AUC gap) | 0.137 | 0.100 | **0.101** |
| Sealed-6 multi-label Brier — rule model | 0.0795 | 0.0757 | **0.0762** |
| Sealed-6 multi-label Brier — Tanimoto-kNN baseline | 0.0810 | 0.0810 | **0.0810** |
| Sealed-6 multi-label Brier — frequency baseline | 0.0842 | 0.0842 | **0.0842** |
| DMSP abstention — mean p(no reaction) | 0.693 | 0.683 | **0.700** |
| Marine training advantage over public-only | 0.0302 | 0.0313 | **0.0318** |
| Public-trained error penalty on marine substrates | 43% | 46% | **48%** |

## The flagship claim: desulfation on unseen sulfate scaffolds

Trained on the two sulfated training compounds alone (zosteric acid, apigenin 7-sulfate), how
well does the model rank **which strains** desulfate a sulfate scaffold it has never seen?
The similarity baseline scores one number per compound, so within a compound it is constant:
ROC-AUC 0.50 by construction.

| Sealed compound | v3 | v4a | **v4b registered** | similarity baseline |
|---|---|---|---|---|
| Luteolin 7-sulfate (ROC-AUC) | 0.764 | 0.743 | **0.764** | 0.50 |
| rac-Hesperetin 7-O-sulfate (ROC-AUC) | 0.767 | 0.767 | **0.767** | 0.50 |

H02 is the harder case and the more informative one: it is a flavanone, so `flavone_core = 0`,
and the prediction rests on the aryl sulfate substructure with no scaffold support.

## Per sealed compound (multi-label Brier, rule model)

| ID | Compound | v3 | v4a | **v4b registered** | Tanimoto baseline |
|---|---|---|---|---|---|
| H01 | Luteolin 7-sulfate | 0.0881 | 0.0844 | **0.0911** | 0.0768 |
| H02 | rac-Hesperetin 7-O-sulfate | 0.0608 | 0.063 | **0.0639** | 0.067 |
| H03 | Sinapic acid | 0.0808 | 0.079 | **0.0819** | 0.0846 |
| H04 | Dehydrodivanillin (divanillin) | 0.0746 | 0.0684 | **0.0694** | 0.0849 |
| H05 | Chicoric acid | 0.1554 | 0.141 | **0.1388** | 0.0895 |
| H06 | DMSP (dimethylsulfoniopropionate) | 0.0175 | 0.0183 | **0.0124** | 0.0833 |

The two compounds the vocabulary change was made for improve most: **divanillin** (H04)
0.0746 → 0.0694 and **chicoric acid** (H05) 0.1554 → 0.1388. Chicoric acid remains the worst of
the six and still loses to the similarity baseline (0.1388 vs 0.0895); it is a diester scored on
a fresh stochastic draw, and a confident model is penalised there. **DMSP** (H06), the abstention
control, improves from 0.0175 to 0.0124 — the best of the six, which is the right place for a
model to be at its best.

## Per reaction class on the sealed six (registered run)

| Reaction class | n positive rows | PR-AUC | Lift over prevalence | ROC-AUC | Mean p (true +) | Mean p (true −) |
|---|---|---|---|---|---|---|
| O_demethylation | 145 | 0.666 | 3.47 | 0.929 | 0.834 | 0.18 |
| aldehyde_oxidation | 69 | 0.648 | 7.1 | 0.973 | 0.741 | 0.081 |
| desulfation | 56 | 0.402 | 5.43 | 0.933 | 0.604 | 0.093 |
| ester_hydrolysis | 70 | 0.657 | 7.09 | 0.969 | 0.935 | 0.066 |
| ring_cleavage | 48 | 0.316 | 4.98 | 0.926 | 0.753 | 0.185 |
| none | 329 | 0.826 | 1.9 | 0.836 | 0.608 | 0.306 |

`decarboxylation` and `deglycosylation` do not appear: the sealed six contain no substrate for
them. That gap is declared in `protocol/protocol.md` §8, not discovered afterwards.

## What must not be claimed from this

- **The aggregate sealed-6 Brier comparison against the similarity baseline does not resolve.**
  v3 presence (control): Δ = +0.0014, 95% CI [-0.0307, +0.0327], P(rule better) = 0.56.
  v4a counts (replace): Δ = +0.0052, 95% CI [-0.0232, +0.0335], P(rule better) = 0.64.
  v4b presence+counts (REGISTERED): Δ = +0.0046, 95% CI [-0.0231, +0.0361], P(rule better) = 0.60.
  Six compounds cannot decide it. This is exactly why the registered endpoint is
  **within-compound strain ranking** and not this comparison (`protocol/protocol.md` §4).
- **On simulated labels the similarity baseline reads the same substructure bits that generated
  the labels**, so it is an equally-informed competitor, not a strawman. It only becomes a weak
  baseline once real LC-MS labels exist.
- **Compositionality (JSD = 0.0079) is a machinery demonstration** on labels generated by the
  very chain being scored. It shows the code computes the statistic; it is not evidence.
- **Reusability fails on the flagship desulfation pair for a mechanical reason**: under
  leave-compound-out the shared-probability term is zero, so the weighted Jaccard is zero. That
  is a property of the split, not of the biology.
