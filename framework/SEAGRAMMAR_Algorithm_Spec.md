# SEAGRAMMAR — Algorithm & Rule Specification (for full Python build)

**Purpose.** This document specifies, in implementation-ready detail, the rule-learning and
evaluation framework behind SEAGRAMMAR so a developer can build/harden the full Python
package. It describes the data model, how chemical "reference" knowledge is applied to the
data, every algorithm (with pseudocode), how to run mock/simulated and real data, the
anti-leakage validation discipline, and a build roadmap.

> **Ghi chú (VI).** Đây là bản đặc tả kỹ thuật để bàn giao cho người code Python. Phần "before
> the fellowship" (khung mô phỏng) đã chạy được end-to-end; mục 9 liệt kê đúng những phần cần
> *full build / hardening*. Khung suy luận **đóng băng** trước khi có funding; chỉ **dữ liệu** thay đổi
> (nhãn mô phỏng → nhãn LC-HRMS/MS thật).

**Status legend:** ✅ implemented in current prototype · 🟡 partial/prototype · 🔭 to build for full version.

---

## 0. Scientific object: the rule

A **rule** maps a chemical substructure σ to a probability distribution over reaction
classes ρ, conditioned on context c (here c = aerobic, fixed by design):

```
σ ⊢ { (ρ, p) }            with ρ_none an explicit, first-class outcome
```

- **σ (substructure)** — a chemically meaningful moiety detected from structure (e.g.
  `aryl_sulfate_ester`, `catechol`, `guaiacyl`).
- **ρ (reaction class)** — one of 11 EC-aligned classes + `none`.
- **p** — probability the rule fires (in some strains/conditions, not others).
- **Multi-label** — one substrate may undergo several reactions simultaneously.

The model is **not** a memoriser of reactions; it is a *compression* of many observed
transformations into a small reusable rule set, judged by three pre-registered tests
(§7). The whole point is generalisation to **unseen** compounds.

---

## 1. Repository / module map

```
build_dataset_v3.py        ✅ CSV-driven dataset generator (sim labels + experimental override)
train_model_v2.py          ✅ multi-label baseline trainer, leave-compound-out CV
test_bayes.py              ✅ model comparison (RandomForest vs BernoulliNB)
msms_to_observations.py    ✅ LC-MS/MS peak tables -> observations.csv (real labels)
data/
  features_rdkit.py        ✅ SMARTS -> 0/1 substructure features (RDKit)
  reaction_rules.py        ✅ expert-prior rule table (σ,ρ)->p + EC + PMID; noisy-OR
  compounds_real.py        ✅ 20-compound panel with real SMILES + provenance
  strains_real.py          ✅ strain table + enzyme capabilities
  inputs/
    compounds.csv          ✅ panel (compound_id, name, pillar, class_id, scaffold, smiles, provenance, origin)
    strains.csv            ✅ strains (strain_id, name, genus, family, phylum, class_tax, role, capabilities, habitat)
    reaction_rules.csv     ✅ rules (reaction, substructure, enzyme, prior_prob, EC, refs)
    observations.csv       ✅ REAL labels (compound_id, strain_id, reaction, label, weight, confidence_level, ...)
    ms2_reaction_map.csv   ✅ Δm map for suspect screening (reaction, delta_mz, type, ppm_tol)
    lcms_samples.csv       ✅ LC-MS sample sheet (demo)
    lcms_features.csv      ✅ LC-MS peak table (demo)
  ml_training_dataset_v3.csv  ✅ generated training table
  DATA_CARD_v3.json           ✅ machine-readable data card
ml_upgrade/                  ✅ UPGRADE package (calibration, baselines, tests, figures)
  _core.py                   ✅ shared primitives + engine auto-fallback (sklearn/rdkit ↔ numpy)
  calibrate_eval.py          ✅ Upgrade 1: isotonic calibration + first-class no-reaction
  baselines_tests.py         ✅ Upgrade 2: baselines + tests A/B/C1/C2
  make_figures.py            ✅ render 5 PNG figures from saved outputs
  make_pipeline_diagram.py   ✅ render the pipeline flow diagram (PNG/SVG)
```

**Dependencies (current):** `scikit-learn>=1.2`, `pandas>=1.5`, `numpy>=1.23`, `rdkit`,
`matplotlib`. (RDKit is required by `features_rdkit.py` and `msms_to_observations.py`.)
The `ml_upgrade/` package degrades gracefully: with sklearn+rdkit it uses RandomForest +
Morgan fingerprints; without them it falls back to a pure-numpy logistic regression +
substructure-bit Tanimoto (so it runs anywhere, identical interface).

### 1b. Pipeline at a glance

`data → model → calibration → tests → figures` (see
`visualizations/ml_upgrade/SEAGRAMMAR_pipeline_flow.png`):

1. **DATA** — input CSVs + `features_rdkit` + `reaction_rules` → `build_dataset_v3.py` →
   `ml_training_dataset_v3.csv`; `msms_to_observations.py` injects REAL labels (override).
2. **MODEL** — `train_model_v2.py` (the original baseline): RF binary-relevance,
   leave-compound-out, reports F1/PR-AUC/ROC-AUC. `_core.py` provides the engine.
3. **CALIBRATION** — `calibrate_eval.py`: isotonic (group-safe) + first-class no-reaction →
   calibrated `p`, Brier, reliability curves. *(This is what `train_model_v2.py` lacked.)*
4. **TESTS** — `baselines_tests.py`: baselines (Tanimoto-kNN / frequency / public-only) +
   A compositionality (JSD), B reusability (weighted Jaccard), C1/C2 productivity
   (Brier + paired bootstrap).
5. **FIGURES** — `make_figures.py`: 5 PNGs for the preliminary-data section.

**Relationship to the old code.** `train_model_v2.py` measures *discrimination* (F1/AUC).
The upgrade adds *trustworthy probabilities* (calibration + no-reaction) and *evidence of
beating baselines* (the three tests) — neither existed before. `train_model_v2.py` is kept
as-is (the discrimination baseline); the new science lives in `ml_upgrade/`.

---

## 2. Data model & schemas

### 2.1 `compounds.csv` — the substrate panel
| column | meaning |
|---|---|
| `compound_id` | stable key (C01…) |
| `compound_name` | trivial name |
| `pillar` | chemical family group (Cinnamic, Hydroxycinnamic, …) |
| `class_id` | integer family id (used for leave-family-out CV) |
| `scaffold` | Bemis–Murcko scaffold tag (used for scaffold-disjoint CV) |
| `smiles` | **canonical SMILES** — the single source of truth; features derive from it |
| `provenance` | ChEMBL/PubChem id (audit trail) |
| `origin` | `seagrass_marine` etc. (metadata, **not** a feature) |

### 2.2 `strains.csv` — the bacterial panel
| column | meaning |
|---|---|
| `strain_id` | stable key (S01…) |
| `genus`, `family`, `phylum`, `class_tax` | taxonomy (phylum used as covariate) |
| `role` | `primary` (1.0) / `expansion` (0.7) — evidence tier |
| `capabilities` | `;`-separated enzyme capabilities (arylsulfatase;esterase;…) |
| `habitat` | `marine` / `terrestrial` (terrestrial = down-weighted augmentation) |

### 2.3 `reaction_rules.csv` — the expert-prior rule table
`reaction, substructure, enzyme, prior_prob, EC, refs`

`prior_prob` is an **expert prior anchored to enzymology** (EC class + PMID literature),
**not** a measured rate. It exists only to generate *simulated* labels with realistic
aleatoric noise so the ML must re-estimate p(σ,ρ). Each row carries a real EC number and
PMID for mechanistic traceability.

### 2.4 `observations.csv` — REAL experimental labels (the during-fellowship input)
`compound_id, strain_id, reaction, label, weight, source, ref, precursor_mz, delta_mz,
msms_match_score, annotation_tool, confidence_level, conversion_pct`

When a row exists for `(compound, strain, reaction)`, its `label` **overrides** the
simulated label and the record is tagged `evidence='experimental'` with high weight.
`confidence_level` follows Schymanski 1–5 (1 = reference standard … 5 = m/z only).

### 2.5 `ms2_reaction_map.csv` — mass-shift map for suspect screening
`reaction, delta_mz_monoisotopic, formula_change, type, ppm_tol, note` where `type ∈
{shift, split, needs_ms2}`. Key marine-specific entry: `desulfation, -79.956820, -SO3,
shift, 8` (the neutral-loss-of-SO₃ diagnostic). `ring_cleavage` is `needs_ms2` → never
auto-labelled, always routed to manual review.

### 2.6 Reaction-class vocabulary (11 + none)
`desulfation, ester_hydrolysis, O_demethylation, aromatic_hydroxylation,
side_chain_cleavage, aldehyde_oxidation, decarboxylation, methylation, deglycosylation,
ring_cleavage, reduction` + `none`.

---

## 3. Applying the chemical "reference" to the data

Two reference layers turn raw structures into model-ready signal.

### 3.1 Structure → substructure features (RDKit/SMARTS) ✅
For every compound, compute a binary vector over a fixed SMARTS dictionary (16 patterns,
see `features_rdkit.py`). 100% reproducible from SMILES.

```python
mol = Chem.MolFromSmiles(smiles)
features[name] = int(mol.HasSubstructMatch(Chem.MolFromSmarts(SMARTS[name])))
```

Representative patterns:
| feature σ | SMARTS | note |
|---|---|---|
| `aryl_sulfate_ester` | `cOS(=O)(=O)[OX2H,OX1-]` | marine-distinctive; triggers desulfation |
| `catechol` | `c([OX2H])c[OX2H]` | ortho-diol; ring-cleavage substrate |
| `gentisate_25_diol` | `[OX2H]c1ccc([OX2H])cc1C(=O)[OX2H1]` | **negative control** (para-diol, NOT catechol) |
| `guaiacyl` | `c([OX2H])c[OX2][CH3]` | O-demethylation |
| `ester_bond` | `[CX3](=O)[OX2H0][#6]` | depside/ester hydrolysis |
| `flavone_core` | `O=c1cc(-c2ccccc2)oc2ccccc12` | flavone ring cleavage |

**Degenerate-feature handling (anti-leakage):** drop constant features (e.g.
`aromatic_ring` always 1) and design-redundant ones (`reducible_double_bond` ≡
`vinyl_aromatic`). Collinear pairs that arise only because the panel is small
(`phenol≡monophenol_para`, `guaiacyl≡aromatic_methoxy`) are **kept and flagged** in the
data card, not silently dropped.

### 3.2 (Structure × strain) → reaction prior (noisy-OR) ✅
A rule fires only if the compound carries σ **and** the strain has capability ρ. Multiple
rules reaching the same reaction combine by **noisy-OR**:

```
P(reaction) = 1 − Π_over_firing_rules (1 − p_rule)
```

```python
def activation_probs(feats, caps, rules, reactions):
    probs = {r: 0.0 for r in reactions}
    for reaction, sub, cap, p, ec, refs in rules:
        if feats.get(sub, 0) == 1 and cap in caps:
            probs[reaction] = 1 - (1 - probs[reaction]) * (1 - p)
    return probs
```

This produces *simulated* per-reaction firing probabilities used only in the before-phase.

---

## 4. Label generation & experimental override (`build_dataset_v3.py`) ✅

For each (strain × compound × replicate):

```
1. feats   = RDKit features for compound        (real, from SMILES)
2. caps    = strain capabilities                 (from strains.csv)
3. probs   = activation_probs(feats, caps, ...)  (noisy-OR prior)
4. labels  = { r: Bernoulli(probs[r]) }          (simulated, seed=42)
5. IF observations.csv has (compound, strain, r):
        labels[r] = real label                   (OVERRIDE)
        evidence  = 'experimental'; weight = high (>=5, or by confidence tier)
6. labels['none'] = 1 if sum(reaction labels)==0 else 0
7. emit row: metadata + used_features + y_<reaction>...
```

**Weighting (single mechanism — no double counting):**
```
marine primary    -> 1.0
marine expansion  -> 0.7
terrestrial ref   -> 0.5      (augmentation only; never dominates marine signal)
experimental      -> >=5.0, or Schymanski tier: {1:8, 2:6, 3:4, 4:2, 5:1}
```

**Output:** `ml_training_dataset_v3.csv` (rows = strains × compounds × replicates) +
`DATA_CARD_v3.json`. Alongside the 12 `y_<label>` columns, the table also emits 12
per-label weight columns `w_<label>` (per-label evidence weighting; **not** used as
features). The dataset is **append-only**: add rows to the input CSVs and re-run;
no code change needed to ingest real data.

> Current mock scale: 32 compounds × 42 strains × 3 replicates = 4,032 rows. Reaction labels are
> **simulated** from expert priors — the honest status is "engineering feasibility, not a
> biological result" (see proposal §1.1.7).

---

## 5. From LC-HRMS/MS to real labels (`msms_to_observations.py`) ✅/🟡

Converts MZmine/XCMS long-format peak tables into `observations.csv`. Three detection modes:

**(A) Suspect screening** (most reliable, no networking needed):
```
for each reaction with known Δm:
    expected_mz = substrate_ion_mz(SMILES, polarity) + Δm
    if a peak matches expected_mz within ppm AND appears in Tn(live) but NOT control:
        emit observation(label=1, confidence=2, conversion_pct=area-based)
```
`substrate_ion_mz = ExactMolWt(mol) ± proton` (neg/pos). Desulfation uses Δm = −79.95682
(−SO₃), ppm_tol = 8.

**(B) Substrate depletion** (catches transformations with *unknown* Δm):
```
if substrate area at Tn(live) < 50% of control AND no suspect matched:
    route to review (reaction='unknown_transformation')   # activity flag, not a labelled reaction
```

**(C) Review queue** (no false auto-labels):
- new peaks not matching any Δm, and any `needs_ms2` reaction (e.g. `ring_cleavage`),
  are written to `review_queue.csv` with evidence for **manual** MS² annotation
  (SIRIUS/CFM-ID). **Never** auto-labelled.

**Censoring:** products below detection are recorded as "not observed" (censored), **not**
confirmed absences — must propagate into scoring (§7).

> 🟡 To harden: isotope/adduct disambiguation, RT consistency for isobaric cases
> (aldehyde_oxidation vs aromatic_hydroxylation both +O), `split`-type fragment matching
> for ester hydrolysis (depside → 2 acids), and a proper MS² annotation hand-off.

---

## 6. Model & training protocol (`train_model_v2.py`) ✅

**Estimator.** Multi-label via **binary relevance** (one classifier per reaction class).
Default `RandomForestClassifier(n_estimators=50, class_weight='balanced')`; `BernoulliNB`
benchmarked in `test_bayes.py`. Keep models **interpretable** (RF feature importance maps
back to substructures); use anything heavier only if it clearly beats this, then explain
post-hoc.

**Input features X.** 16 substructure columns (RDKit) + one-hot `strain_phylum` (+ one-hot
`habitat` covariate). **Never** use as features: `source, evidence, strain_id, compound_id,
cv_group, provenance, smiles, weight, replicate_id`.

**Sample weights.** Pass the per-label `w_<label>` column to `sample_weight` when present
(falls back to the single `weight` column for backward compatibility). Per-label weights
ensure that an experimental observation for one reaction up-weights **only that label**, not
the still-simulated labels sharing the same row.

**Cross-validation (anti-leakage — critical).**
```
GroupKFold(groups=cv_group)      # cv_group = compound_id  -> LEAVE-COMPOUND-OUT
```
This guarantees no identical feature vector sits in both train and test. The prototype
benchmark shows a random KFold inflates mean F1 by ~+70% versus GroupKFold — so random
splits are forbidden for reporting.

Also support: **scaffold-disjoint** (`groups=scaffold`) and **leave-family-out**
(`groups=class_id`), and **leave-strain-out** for the strain generalisation axis.

**Metrics.** Per-label **F1, PR-AUC (average precision), ROC-AUC, precision, recall,
support**, plus macro averages. **Never** report accuracy (all-zero baseline ≈ 90% here).
Labels with fewer than `MIN_POS` (=8) positive rows **or** whose σ occurs in fewer than
`MIN_COMPOUNDS` (=3 in the mock; target 10 in the funded panel) distinct compounds are not
evaluated (declared, not hidden). The **compound-count gate is the one that actually protects
leave-compound-out** — a label can clear `MIN_POS` on row count yet still live on only 1–2
compounds (e.g. desulfation, deglycosylation), which cannot generalise across held-out compounds.

**Rare-reaction caveat.** Under leave-compound-out, a reaction whose σ occurs in only 1–2
compounds (desulfation, deglycosylation in the mock panel) is **not evaluable** — needs a
larger panel or scaffold-level grouping. This is the **minimum-support gate**: σ must
appear in ≥ N substrates (target N=10 in the funded panel) to admit a rule.

Pseudocode:
```python
for reaction in reactions:
    y = labels[reaction]
    n_comp = n_distinct_compounds(y == 1)
    if y.sum() < MIN_POS or n_comp < MIN_COMPOUNDS: skip   # compound-count gate
    w = per_label_weight(reaction)                          # w_<label>, fallback to weight
    oof = leave_compound_out_predict(X, y, groups=cv_group, sample_weight=w)
    report f1/pr_auc/roc_auc(y, oof)
```

---

## 7. The three pre-registered tests 🔭 (design — to build)

These are the scientific acceptance criteria. They are **specified** in the proposal
(Methods Appendix) but only partially coded; this is the core of the full build.

**Two-tier rule confidence.** High = ≥1 ¹³C-confirmed substrate in the σ class; Medium =
otherwise. Report metrics per tier.

### 7.1 Compositionality (A) — do chained atomic rules predict cascade endpoints?
- Chain atomic rules to predict the **end-product distribution** of a multi-step cascade
  (e.g. ferulic→vanillin→vanillic→protocatechuate; zosteric→p-coumaric→…).
- Score = **Jensen–Shannon divergence** between predicted and observed end-product
  distributions. Acceptance threshold pre-registered, calibrated from replicate spread + a
  known-kinetics literature cascade (rosmarinic→caffeic+DHPL→protocatechuate) + cell-free
  controls.
```
pred_dist = compose(atomic_rules, start=substrate)   # propagate probabilities through DAG
score     = 1 - JSD(pred_dist, observed_dist)
accept    = JSD < tau_A
```

### 7.2 Reusability (B) — does a rule transfer across related substrates?
- Compare predicted vs observed reaction profiles on **matched substrate pairs**
  (zosteric acid ↔ apigenin 7-sulfate for desulfation; vanillic ↔ syringic for
  demethylation) by **weighted-Jaccard / cosine** similarity, MS parameters frozen.

### 7.3 Productivity (C) — predict the fate of compounds never seen in training
- The decisive test. Sealed predictions for the 6 held-out compounds (incl. the correct
  **no-reaction** call on non-aromatic DMSP) registered on OSF *before* unblinding.
- Score = **multi-label Brier** vs (i) a structure-similarity **Tanimoto-kNN** baseline
  and (ii) a frequency baseline; below-detection products **censored**; significance by
  **paired bootstrap** (95% CI of the difference excluding 0). The rule model must beat the
  similarity baseline, not merely randomness.
```
brier_rules    = mean((p_rules - y)^2)      # over labels, censored entries masked
brier_baseline = mean((p_tanimoto_knn - y)^2)
delta, ci      = paired_bootstrap(brier_baseline - brier_rules, B=10000)
win            = ci.low > 0
```

**Primary success criterion (overall).** On a scaffold-disjoint hold-out, structure-aware
rules beat the similarity baseline **and** project-generated marine data adds power over a
public-data-only model — i.e. prediction comes from learned biochemical logic, not naïve
similarity or prior databases.

---

## 8. Baselines (must be implemented to make the claims meaningful) 🔭

| baseline | purpose |
|---|---|
| **Tanimoto-kNN** (Morgan FP) | "naïve molecular similarity" — the bar to beat |
| **Frequency / prevalence** | per-label base rate; sanity floor |
| **Public-data-only** | model trained only on down-weighted public reactions (BioTransformer/KEGG/BRENDA) → shows marine data adds value |
| **enviRule / enviPath** 🔭 | external automatic rule extractor — *comparator only* (post-funding); positions our calibrated, community-level, no-reaction model against the best existing rule system |

Public reaction DBs are kept but **down-weighted** (clinical/terrestrial bias); project
marine data weighted up. Report prediction error **separately per bacterial group** so
uneven learning is visible.

---

## 9. Build roadmap for the full Python package 🔭

What exists is a working **before-phase prototype**. **Update:** `ml_upgrade/` now provides
prototype implementations of calibration + first-class no-reaction (item 2–3), the three
tests A/B/C (item 4), and the baselines incl. public-only (item 6). The full version
*hardens and packages* these. Remaining/again items:

1. **Package + CLI.** `seagrammar/` package; `seagrammar build|train|evaluate|predict`;
   config via YAML; pin versions; Dockerfile + Snakemake for reproducibility (Zenodo DOI).
2. **Probability calibration.** Wrap each binary-relevance classifier in isotonic/Platt
   calibration (`CalibratedClassifierCV`) so `p` in σ ⊢ {(ρ,p)} is a real probability —
   required for Brier scoring and the no-reaction outcome.
3. **First-class `no-reaction`.** Treat ρ_none as a calibrated prediction and score it
   (esp. the DMSP boundary case), not just `1 − Σ`.
4. **The three tests (§7)** as a tested module: JSD compositionality over a reaction DAG,
   weighted-Jaccard reusability, multi-label Brier productivity with **censoring** + paired
   bootstrap CIs.
5. **Cascade engine.** Represent reactions as a DAG; propagate probabilities; expose
   `compose()` for Test A and for the "discovery filter" (rank plausible unannotated
   products for metabolomics).
6. **Baselines module** (§8), incl. public-data-only and enviRule comparator hooks.
7. **Sealed-prediction workflow.** Freeze model + emit OSF-ready sealed predictions for the
   6 held-out compounds; hash + timestamp; never re-fit on held-out.
8. **¹³C confidence tagging.** Join isotope-confirmation flags onto rules → High/Medium
   tiers in all reports.
9. **Robust LC-MS ingestion** (§5 hardening) + MS² annotation hand-off.
10. **Test suite + CI.** Unit tests for featurization, noisy-OR, leakage guards (assert no
    compound spans train/test), metric correctness on synthetic fixtures.

**Frozen vs during-fellowship (do not blur):** architecture, evaluation protocol,
baselines, and the sealed held-out predictions are **frozen before** funding. During the
fellowship **only the data change** — simulated labels are replaced by real LC-HRMS/MS
reaction calls via `observations.csv`. The inferential machinery does not move.

---

## 10. How to run (current prototype)

```bash
pip install -r requirements.txt          # + rdkit (conda-forge or pip rdkit-pypi)

# 1) (mock or real) build the training table from the input CSVs
python3 build_dataset_v3.py
#    -> data/ml_training_dataset_v3.csv, data/DATA_CARD_v3.json

# 2) train + honest evaluation (leave-compound-out)
python3 train_model_v2.py
#    -> visualizations/model_v3/{per_label_metrics.csv, macro_summary.json, *.png}

# 3) model comparison (RF vs Naive Bayes)
python3 test_bayes.py

# 4) when real LC-MS arrives: peaks -> observations.csv, then re-run step 1
python3 msms_to_observations.py --demo
python3 msms_to_observations.py \
    --samples data/inputs/lcms_samples.csv \
    --features data/inputs/lcms_features.csv

# 5) UPGRADES (run in this order so figures pick up real numbers)
cd ml_upgrade
python3 calibrate_eval.py        # calibration + no-reaction  -> calibration_*.csv/json
python3 baselines_tests.py       # baselines + tests A/B/C1/C2 -> baselines_tests_report.json
python3 make_figures.py          # 5 PNG figures
python3 make_pipeline_diagram.py # pipeline flow diagram (PNG/SVG)
```

To **add real data**: append rows to `data/inputs/observations.csv` (or generate them via
`msms_to_observations.py`) and re-run `build_dataset_v3.py`. Experimental labels override
simulated ones automatically and are up-weighted.

---

## 11. Honesty / scope notes (carry into any write-up)

- Mock reaction labels are **simulated from expert priors** (EC/PMID-anchored), not
  measured. Real parts of the prototype: compound structures (SMILES), cheminformatics
  features (RDKit), strain taxonomy, and the enzymology behind each rule.
- **Aerobic only**; the oxygen context axis is fixed (the σ ⊢ ρ | c extension to c=redox is
  future work).
- Rules are **not** claimed universal; the salinity-gradient transfer test classifies each
  rule as universal / context-dependent / community-emergent. Failed transfer is an
  informative result, not a bug.
- Below-detection ≠ absence (censoring), throughout.
