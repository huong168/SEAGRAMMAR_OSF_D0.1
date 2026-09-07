# SEAGRAMMAR evaluation protocol (registered, frozen 2026-09-07)

This document fixes how the model will be evaluated, **before any experimental data exist**. It is
part of registration D0.1. Nothing here may be changed once the registration is public; changes to
acquisition settings only are filed as a registered update at M2 (see §7).

---

## 1. The object being evaluated

A rule set of the form `σ ⊢ {(ρ, p)} | c` — substructure σ, under strain capability context c,
licenses reaction ρ with probability p. Prediction is **multi-label per (compound, strain)**: for
each of the 11 reaction classes plus an explicit `none` outcome, a calibrated probability.

`none` is a scored label, not an absence of prediction. A model that cannot say "nothing happens" is
not being tested on the thing that matters most for a natural-products screen.

## 2. The three splits, run in parallel

| Split | What is withheld | What it measures |
|---|---|---|
| **Leave-compound-out (LCO)** | one compound at a time, all its rows | can the rule set be recovered from structure when this molecule is unseen |
| **Leave-family-out (LFO)** | a whole scaffold family at once | how much the model leans on scaffold identity rather than on the substructure |
| **Sealed 20 → 6** | the six registered sealed compounds, permanently | prospective predictivity, the decisive test |

Grouping unit for cross-validation and for every confidence interval is the **compound**
(`cv_group = compound_id`), never the row. Replicates and strains within a compound are not
independent and must not be resampled as if they were.

The **LCO − LFO gap** is reported as a first-class result, not a footnote. It is the honest measure
of scaffold dependence, and the proposal's reusability claim lives or dies on it being small.

## 3. Metrics

**Primary:** PR-AUC (average precision) together with **lift over prevalence**. Reaction labels are
rare and unbalanced, so PR-AUC and lift are what a reader can act on.

**Secondary:** ROC-AUC (threshold-free), reported alongside, never alone.

**Calibration:** multi-label Brier, before and after isotonic (pool-adjacent-violators) calibration
fitted out-of-fold. Reliability curves per label are deposited with each run.

**Reusability (Test 1):** weighted-Jaccard and cosine between the strain-probability profiles a rule
produces on two chemically distinct substrates carrying the same substructure. Threshold: both
shared-probability terms > 0.5.

**Compositionality (Test 2):** Jensen–Shannon divergence between the predicted and observed
distribution over the steps of a cascade. Threshold: JSD < 0.05.

**Predictivity (Test 3):** see §4.

**Confidence intervals:** cluster bootstrap over compound groups, 2000 resamples by default (500 in
the deposited pilot runs, for runtime; this widens intervals slightly and is stated in each log).

## 4. The pre-registered predictivity endpoint

> **The endpoint is within-compound strain ranking on the sealed compounds:** for each sealed
> compound and each reaction class with positive rows, how well the model orders **which strains**
> perform the reaction. Reported as ROC-AUC and PR-AUC within the compound, against the
> structure-similarity baseline.

**What is explicitly NOT the endpoint: an aggregate multi-label Brier comparison against the
similarity baseline across the six sealed compounds.** This is registered as a negative
specification because the pilot showed it cannot resolve: at n = 6 clusters the observed difference
was ΔBrier ≈ 0.001 with 95% CI [−0.031, +0.033]. Six compounds cannot decide that comparison, and a
"win" on it would be noise. Registering the endpoint this way removes the temptation to report it
after the fact.

The choice also reflects what the similarity baseline can and cannot do: Tanimoto-kNN assigns one
number per compound, so **within a compound it cannot rank strains at all** (ROC-AUC 0.50 by
construction). The endpoint is the question the rule model exists to answer.

## 5. Admission gate for a rule

A rule is admitted once it is observed in **≥ 2 distinct substrates and ≥ 5 distinct strains
(≥ 10 independent substrate × strain observations)**.

The gate is stated in **observations**, not in compound counts, on purpose. A compound-count gate
would exclude chemically narrow but biologically central rules — desulfation is trained on only two
sulfated substrates — by construction rather than by evidence. Features below the threshold produce
no scored rule and are reported as open questions, not as negative results.

## 6. Baselines

Every predictivity claim is reported against all three:

| Baseline | What it is | Why it is here |
|---|---|---|
| **Tanimoto-kNN** | k = 5 nearest training compounds by substructure-bit Tanimoto; label rate averaged | the honest competitor: if structural similarity alone suffices, the rule set adds nothing |
| **Frequency** | per-label training prevalence | the floor; beats a badly calibrated model more often than people expect |
| **Public/terrestrial-trained** | the same model fitted only on public land-plant rows, scored on marine substrates | the secondary success criterion: does marine training earn its cost |

The similarity baseline is computed on the **presence** of substructures (boolean), so it is
invariant to the match-count refinement described in `CHANGELOG.md`. This is deliberate: it keeps
the baseline fixed across vocabulary versions, so any change in the comparison is attributable to
the rule model alone.

## 7. What is fixed here and what is filed at M2

**Fixed by this registration, unchangeable:** the rule schema; the substructure vocabulary; the
identity of the 20 training and 6 sealed compounds; the three splits; the metrics; the admission
gate; the predictivity endpoint; the baselines.

**Filed at M2 as a registered update, with the schema and sealed identities unchanged:** LC-HRMS/MS
acquisition and annotation settings. These cannot honestly be fixed before the instrument time is
booked, and pretending otherwise would be the kind of retrofit this registration exists to prevent.

## 8. Declared limitations, registered rather than discovered later

1. **The sealed six exercise 9 of 11 reaction classes.** No glycoside and no decarboxylation
   substrate is in the sealed set, so those two classes have zero positive rows there.
2. **The pilot labels are simulated.** Every number in `pilot/` is rule recovery on noisy-OR labels
   drawn from the priors, not biology. Real predictive numbers require LC-MS labels.
3. **The similarity baseline is an equally-informed competitor on simulated labels**, because it
   reads the same substructure bits that generated them. It is a strawman only after real labels
   arrive, not before.
4. **Assembled, not natural.** Community-level transfer is tested on communities assembled from the
   host's isolates at 2, 5 and 10 members, not on natural sediment communities.
