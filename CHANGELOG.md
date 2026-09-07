# CHANGELOG — from the 17/08/2026 pilot to the registered version

## Registered vocabulary: v4b (presence bits **plus** additive match counts)

### Why the vocabulary changed at all

The 17/08/2026 pilot encoded each of 17 substructures as **presence or absence**. Its own
interpretation flagged the consequence: two of the six sealed compounds could not test what they had
been chosen to test.

- **Dehydrodivanillin (H04)** had the same feature vector as **vanillin (C13)**, so the
  monomer → dimer question was invisible to the model.
- **Chicoric acid (H05)** had the same vector as **caftaric acid (C05)**, so the two-sequential-
  hydrolyses question was invisible too.

Re-checking with RDKit before registration showed the problem is larger than those two pairs.
Under presence/absence, **15 of the 37 panel compounds fall into 7 groups with identical feature
vectors**:

```
caftaric = rosmarinic = chicoric = chlorogenic
vanillin = divanillin
luteolin = quercetin
apigenin = kaempferol
luteolin 7-O-glucoside = quercetin 3-O-glucoside
naringenin = tyrosol
catechin = hydroxytyrosol
```

The last two are the clearest sign that presence/absence is too coarse: a flavanone and a
phenylethanoid are not the same molecule to an enzyme. With match counts there are **0 colliding
vectors**.

### What v4b is, and why "plus" rather than "instead of"

v4b keeps the 17 presence bits and **adds** 17 integer `n_<feature>` match-count columns
(30 features after constant and redundant-by-design columns are dropped). Two reasons, both settled
without reference to any result:

1. It is what the protocol says. The proposal's own wording is that "a substructure match-count
   feature is **added** before the OSF registration".
2. It is a strict superset of the v3 representation, so nothing the presence vocabulary could
   express is lost.

A variant that **replaces** the bits with counts (v4a) was also run. It is deposited in full at
`pilot/run_v4a_counts_replace/` — see the transparency note below.

### Two defects fixed while making the change

1. **`build_dataset_v3.py` tested `feats[σ] == 1`.** Under a count vocabulary a value of 2 would
   have silently failed the rule, so divanillin (`aromatic_aldehyde = 2`) would have stopped being
   an aldehyde-oxidation substrate — the simulated ground truth itself would have moved. Changed to
   `> 0`. It is a no-op on the v3 binary table, which is why the control run reproduces exactly and
   the simulated labels are **bit-identical across all three vocabulary variants**.
2. **`data/inputs/observations.csv` shipped a draft placeholder row** —
   `C19,S09,desulfation,1,5.0,experimental,PMID:XXXXXXX_...` — a template row carrying a
   placeholder PMID. It never affected a result: `build_dataset_v3.py` drops any row whose `ref`
   contains `XXXX` by design, and every run reports `n_experimental_rows_overridden: 0`. It has
   nonetheless been removed from this deposit, including from the untouched reference copy, because
   a row marked `experimental` with an invented citation should not be published at all, filtered or
   not. The slot ships as `framework/rules/observations_EMPTY_TEMPLATE.csv`, header only, documented
   in `framework/rules/observations_README.md`.

### Verification that the hand-derived v3 table was correct

The 17/08 run could not install RDKit and used a hand-derived lookup table for the sealed compounds
and arbutin. Recomputing every one of the 37 compounds with RDKit from SMILES gives
**zero disagreements** with that table on presence. The v3 features were right; they were just too
coarse.

### Environment parity

The registered run uses the same environment as 17/08: Python 3.11.15, scikit-learn 1.8.0,
pandas 3.0.2, numpy 2.4.4, matplotlib 3.10.9, **no RDKit on the evaluation path**. Re-running the v3
vocabulary in that environment reproduces the 17/08 logs **character for character**
(`pilot/run_v3_presence_control/logs/`). Every difference between the runs is therefore attributable
to the vocabulary and to nothing else.

---

## Transparency note: three variants were measured before the seal

All three vocabulary variants were run and scored, including on the sealed six, before this
registration was created. **All three are deposited in full** — datasets, logs, per-compound
results, figures — at:

```
pilot/run_v3_presence_control/     v3, presence/absence (the 17/08 vocabulary)
pilot/run_v4a_counts_replace/      v4a, presence replaced by counts
pilot/run_v4b_registered/          v4b, presence plus counts  <-- REGISTERED
```

The registered variant was chosen on the two a-priori grounds given above — the protocol's own
wording, and strict representational superiority — not on sealed-set performance. Depositing all
three removes the file drawer: a reader can see exactly what was measured and judge the choice
independently.

This note exists because the honest failure mode here is obvious and worth naming: measuring three
variants against the sealed set and keeping the best one would hollow out the seal. The mitigation
is disclosure plus a selection rule that does not read the results.
