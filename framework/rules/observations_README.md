# observations — the slot for real labels

`observations_EMPTY_TEMPLATE.csv` is the single place where real LC-HRMS/MS labels override the
simulated ones. It ships **empty (header only)** and must stay empty in this registration: no
measurement exists yet.

One row per `(compound_id, strain_id, reaction)`. `label` is 0/1. `weight` may be given directly or
derived from `confidence_level` on the Schymanski scale (1 = reference standard … 5 = m/z only).

`build_dataset_v3.py` drops any row whose `ref` contains the string `XXXX`, so template rows cannot
silently enter a dataset. The 2026-08-17 workspace carried one such template row
(`C19,S09,desulfation,… PMID:XXXXXXX_…`); it was filtered by that rule and never affected any
result, and it has been removed from this deposit so that no fabricated observation is published,
filtered or not.
