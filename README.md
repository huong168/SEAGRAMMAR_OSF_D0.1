# SEAGRAMMAR — frozen framework, evaluation protocol, baselines and sealed benchmark

**Registration D0.1 of the SEAGRAMMAR project.** Everything in this deposit was built and frozen
**before** the fellowship it belongs to began, and before any experimental data exist. That is the
whole point of registering it: when the LC-HRMS/MS measurements arrive, only the data will be new.
The rule schema, the evaluation protocol, the baselines and the identity of the six sealed test
compounds are fixed here, in public, with a timestamp.

Author: Huong Pham (ORCID 0000-0002-6896-133X)
Frozen: 2026-09-07

---

## What the project is testing

Sediment bacteria in the eelgrass (*Zostera marina*) rhizosphere meet a chemically diverse set of
plant metabolites. The hypothesis is that they transform them according to **biochemical rules keyed
on recurring substructures** rather than on whole molecules — rules reusable enough to predict
reactions on metabolites that played no part in deriving them.

A rule is written `σ ⊢ {(ρ, p)} | c`: substructure σ, under strain capability context c, licenses
reaction ρ with probability p.

Three falsifiable tests, in the order used throughout:

| Test | Question |
|---|---|
| **Reusability** | Does the same rule predict a reaction when a recurring substructure appears in a chemically different metabolite? |
| **Compositionality** | Do individual rules combine to predict multi-step transformations, including cascades enabled by cross-feeding? |
| **Predictivity** (decisive) | Can rules learned from known transformations correctly anticipate the fate of metabolites that played no part in deriving them? |

---

## What is in this deposit

```
panel.csv                      26 compounds: 20 training + 6 SEALED, with CAS
reaction_classes.csv           the 11 reaction classes and the scaffolds each is trained on
protocol/protocol.md           evaluation protocol: splits, metrics, admission gate, endpoint
framework/                     the frozen framework
  vocabulary/                  substructure vocabulary (SMARTS) + the three feature tables
  rules/                       rule priors, strain capability profiles, SMILES, empty data slot
  pipeline/                    the code that builds the dataset, fits and scores
  orig_reference_v3_untouched/ verbatim copy of the source pipeline, for diffing
  requirements.txt             pinned versions of the registered run
  Dockerfile                   container recipe reproducing the registered run
baselines/                     Tanimoto-kNN, frequency, and public/terrestrial-trained baselines
pilot/                         in-silico pilot: all three vocabulary variants, full workspaces
RESULTS_registered_run.md      the numbers, and the three-way comparison
CHANGELOG.md                   what changed from the 17/08/2026 version, and why
RUN.sh                         one command to reproduce the registered run
```

## The sealed set

Six compounds (`panel_role = sealed` in `panel.csv`) are held back. No model in this project may be
fitted on them, and their identities cannot change after this registration. Publishing them is
deliberate: the seal exists to stop **the author** from swapping the test set after seeing data, not
to keep anyone else from knowing what it is. At the M20 seal, anyone can check the six compounds
scored against the six registered here.

Two of the six carry a self-declared limitation: the sealed set contains no glycoside and no
decarboxylation substrate, so it exercises **9 of the 11** reaction classes. That gap is registered,
not discovered later.

## The labels in the pilot are simulated

**Every metric in `pilot/` comes from labels drawn by noisy-OR from the rule priors in
`framework/rules/reaction_rules.csv`. Nothing in this deposit is a measurement.** The pilot is a
rule-recovery positive control: it shows the machinery recovers a generating rule set from structure
alone, and that the evaluation harness runs end to end on the exact panel proposed. It says nothing
about whether the biology behaves this way. `framework/rules/observations_EMPTY_TEMPLATE.csv` is the
slot where real LC-HRMS/MS labels will override simulated ones; it is empty, and shipped empty on
purpose.

## Reproducing

```
pip install -r framework/requirements.txt   # or: docker build -f framework/Dockerfile .
bash RUN.sh
```

The pipeline runs **without RDKit**, reading the shipped feature table
(`framework/vocabulary/features_registered_v4b.csv`). That is how the registered numbers were
produced, and it is what makes them reproducible on a machine where RDKit will not install. RDKit is
needed only to regenerate the feature tables from SMILES
(`framework/vocabulary/make_count_features.py`), and doing so reproduces the shipped tables exactly.

Seeded throughout (SEED = 42). Registered environment: Python 3.11.15, scikit-learn 1.8.0,
pandas 3.0.2, numpy 2.4.4, matplotlib 3.10.9.

## Licence

Code: MIT (`LICENSE`). Documentation, protocol and data tables: CC BY 4.0.

## Related deposits

Code is additionally archived with its own DOI via Zenodo from the project's GitHub release; the
identifiers are listed in `CITATION.cff` once minted. LC-HRMS/MS acquisition and annotation settings
are **not** part of this registration: they are fixed at M2 and filed then as a registered update,
with the rule schema and the sealed identities unchanged from this DOI.
# SEAGRAMMAR_OSF_D0.1
# SEAGRAMMAR_OSF_D0.1
