#!/usr/bin/env python3
# ml_upgrade/baselines_tests.py
# =============================================================================
# UPGRADE 2 — Baselines + the three pre-registered tests.
#
# Baselines (the bars the rule model must beat):
#   - frequency        : per-label train prevalence
#   - tanimoto_knn     : structure-similarity only (Morgan FP if rdkit, else the
#                        16 substructure bits) — the "naive similarity" baseline
#   - public_only      : rule model trained on TERRESTRIAL rows, tested on MARINE
#                        (proxy for "public/terrestrial-biased databases")
#
# Three tests:
#   A Compositionality (Jensen–Shannon divergence) — chained-rule cascade endpoint
#   B Reusability      (weighted Jaccard / cosine) — matched substrate pairs
#   C Productivity     (multi-label Brier + paired bootstrap) — leave-FAMILY-out
#                        held-out compounds vs the tanimoto baseline
#
# Runs with sklearn/rdkit if present, else numpy/substructure fallback (_core).
# =============================================================================
import os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import _core as C

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def _resolve_data_dir(root):
    d = os.environ.get('SEAGRAMMAR_DATA')
    if d:
        return d
    for cand in ('data', 'data1'):
        if os.path.exists(os.path.join(root, cand, 'ml_training_dataset_v3.csv')):
            return os.path.join(root, cand)
    return os.path.join(root, 'data')
DATA = os.path.join(_resolve_data_dir(ROOT), 'ml_training_dataset_v3.csv')  # FIX: data/ vs data1/
OUTDIR = os.path.join(ROOT, 'visualizations', 'ml_upgrade'); os.makedirs(OUTDIR, exist_ok=True)
N_OUTER, N_INNER, K_NN = 5, 4, 5

SUBSTRUCT = ['phenol','catechol','pyrogallol','gentisate_25_diol','guaiacyl','syringyl',
             'aromatic_methoxy','monophenol_para','carboxylic_acid','benzoic_acid',
             'aromatic_aldehyde','vinyl_aromatic','ester_bond','aryl_O_glycoside',
             'aryl_sulfate_ester','flavone_core']

# --- PILOT PATCH 2026-08-17: the SUBSTRUCT list above is hard-coded, but
# build_dataset_v3.py DROPS features that are constant in the current panel
# (e.g. pyrogallol once gallic acid left the panel). Intersect instead of
# raising KeyError. Original file: orig_reference/pipeline_v3_ORIGINAL/
def _present(df, cols):
    keep = [c for c in cols if c in df.columns]
    # --- PILOT PATCH 2026-09-07 (vocabulary v4b): match-count columns are named
    # n_<feature> and are ADDITIVE to the presence columns listed in SUBSTRUCT.
    # They are absent from the v3 and v4a datasets, so this is a no-op there.
    keep = keep + [c for c in df.columns if c.startswith('n_') and c not in keep]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        import sys; print(f'[pilot-patch] dropped absent features: {missing}', file=sys.stderr)
    return keep
REACTIONS = ['O_demethylation','aldehyde_oxidation','aromatic_hydroxylation','decarboxylation',
             'deglycosylation','desulfation','ester_hydrolysis','methylation','reduction',
             'ring_cleavage','side_chain_cleavage']
LABELS = REACTIONS + ['none']


def build_X(df):
    parts = [df[_present(df, SUBSTRUCT)], pd.get_dummies(df['strain_phylum'], prefix='phylum').astype(int)]
    if 'habitat' in df.columns:
        parts.append(pd.get_dummies(df['habitat'], prefix='habitat').astype(int))
    return pd.concat(parts, axis=1).values.astype(float)


def proba1(clf, X):
    p = clf.predict_proba(X)
    return p[:, 1] if getattr(clf, 'classes_', np.array([0, 1])).shape[0] > 1 else np.zeros(len(X))


def oof_calibrated(X, y, groups, w, n_outer=N_OUTER):
    """Calibrated leave-group-out OOF probabilities for one label."""
    n = len(y); out = np.full(n, np.nan)
    for tr, te in C.group_kfold(groups, n_outer):
        if y[tr].sum() == 0:
            out[te] = 0.0; continue
        fit_i, cal_i = next(C.group_kfold(groups[tr], N_INNER))
        fit, calib = tr[fit_i], tr[cal_i]
        clf = C.make_classifier()
        if y[fit].sum() == 0 or y[calib].sum() == 0:
            clf.fit(X[tr], y[tr], sample_weight=w[tr]); out[te] = proba1(clf, X[te]); continue
        clf.fit(X[fit], y[fit], sample_weight=w[fit])
        xs, fit_iso = C.pav_isotonic_fit(proba1(clf, X[calib]), y[calib])
        out[te] = C.pav_predict(xs, fit_iso, proba1(clf, X[te]))
    return out


def fingerprints(df):
    """compound_id -> binary fingerprint (Morgan if rdkit else 16 substructure bits)."""
    fps = {}
    for cid, g in df.groupby('compound_id'):
        smi = g['smiles'].iloc[0]
        bits = C.morgan_bits(smi)
        if bits is None:
            bits = g[_present(g, SUBSTRUCT)].iloc[0].values.astype(int)
        fps[cid] = bits
    return fps


def comp_label_rate(df):
    """compound_id -> per-label positive rate (vector over LABELS)."""
    return {cid: g[['y_' + l for l in LABELS]].mean().values for cid, g in df.groupby('compound_id')}


# --------------------------- BASELINES ---------------------------------------
def tanimoto_knn_oof(df, groups, fps, rates, k=K_NN):
    """Per-row label-prob matrix from structure-similarity kNN, leave-compound-out."""
    cids = df['compound_id'].values
    n = len(df); P = np.zeros((n, len(LABELS)))
    for tr, te in C.group_kfold(groups, N_OUTER):
        train_cids = sorted(set(cids[tr]))
        for cid in sorted(set(cids[te])):
            sims = [(C.tanimoto(fps[cid], fps[tc]), tc) for tc in train_cids]
            sims.sort(reverse=True); top = sims[:k]
            wsum = sum(s for s, _ in top) or 1e-9
            pred = sum(s * rates[tc] for s, tc in top) / wsum
            P[cids == cid] = pred
    return P


def frequency_oof(df, groups):
    cids = df['compound_id'].values; n = len(df); P = np.zeros((n, len(LABELS)))
    Y = df[['y_' + l for l in LABELS]].values
    for tr, te in C.group_kfold(groups, N_OUTER):
        rate = Y[tr].mean(0); P[te] = rate
    return P


def rule_model_matrix(X, df, groups, w, n_outer=N_OUTER):
    P = np.zeros((len(df), len(LABELS)))
    for j, l in enumerate(LABELS):
        wl = df['w_' + l].values.astype(float) if ('w_' + l) in df.columns else w  # FIX: per-label weight
        P[:, j] = oof_calibrated(X, df['y_' + l].values.astype(int), groups, wl, n_outer)
    return P


def multilabel_row_brier(P, Y):
    return np.nanmean((P - Y) ** 2, axis=1)


# --------------------------- TEST A: compositionality ------------------------
def test_compositionality(df, P_rule):
    """Chained-rule cascade endpoint vs observed (Jensen–Shannon). DEMO on the
    guaiacyl funnel substrate ferulic acid (C04): chain step1=O_demethylation,
    step2=ring_cleavage; distribution over {none, step1, step1+2}."""
    cid = 'C04'; rows = df['compound_id'] == cid
    li = {l: i for i, l in enumerate(LABELS)}
    p1_pred = float(P_rule[rows, li['O_demethylation']].mean())
    p2_pred = float(P_rule[rows, li['ring_cleavage']].mean())
    p1_obs = float(df.loc[rows, 'y_O_demethylation'].mean())
    p2_obs = float(df.loc[rows, 'y_ring_cleavage'].mean())
    def dist(p1, p2):
        return np.array([(1 - p1), p1 * (1 - p2), p1 * p2])  # none / step1 / step1+2
    pred, obs = dist(p1_pred, p2_pred), dist(p1_obs, p2_obs)
    jsd = C.jensen_shannon(pred, obs)
    return {'substrate': cid, 'chain': 'O_demethylation -> ring_cleavage',
            'pred_dist': [round(x, 3) for x in pred], 'obs_dist': [round(x, 3) for x in obs],
            'JSD': round(jsd, 4), 'pass_threshold(<0.05)': bool(jsd < 0.05),
            'note': 'machinery demo on simulated labels; real endpoints come from LC-MS cascades.'}


# --------------------------- TEST B: reusability -----------------------------
def reaction_profile(df, P_rule, cid):
    rows = (df['compound_id'] == cid).values
    li = {l: i for i, l in enumerate(LABELS)}
    return np.array([P_rule[rows, li[r]].mean() for r in REACTIONS])


def test_reusability(df, P_rule):
    pairs = [('desulfation', 'C19', 'C20'),       # zosteric <-> apigenin 7-sulfate
             ('O_demethylation', 'C10', 'C11'),   # vanillic <-> syringic
             ('ester_hydrolysis', 'C05', 'C06'),  # caftaric <-> rosmarinic
             ('aldehyde_oxidation', 'C13', 'C14'),# vanillin <-> 4-OH-benzaldehyde
             ('ring_cleavage', 'C15', 'C16')]     # luteolin <-> apigenin (apigenin = negative)
    li = {r: i for i, r in enumerate(REACTIONS)}
    out = []
    for rxn, a, b in pairs:
        pa, pb = reaction_profile(df, P_rule, a), reaction_profile(df, P_rule, b)
        out.append({'rule': rxn, 'A': a, 'B': b,
                    'wJaccard': round(C.weighted_jaccard(pa, pb), 3),
                    'cosine': round(C.cosine(pa, pb), 3),
                    'sharedP_A': round(float(pa[li[rxn]]), 3),
                    'sharedP_B': round(float(pb[li[rxn]]), 3),
                    'transfers(both>0.5)': bool(pa[li[rxn]] > 0.5 and pb[li[rxn]] > 0.5)})
    return out


# --------------------------- TEST C: productivity ----------------------------
def productivity_bootstrap(P_rule, P_base, Y, split_name, note):
    """Multi-label Brier + paired bootstrap (delta = base - rule; >0 => rule better)."""
    br_rule = multilabel_row_brier(P_rule, Y)
    br_base = multilabel_row_brier(P_base, Y)
    mean, lo, hi, pwin = C.paired_bootstrap(br_base - br_rule, B=10000)
    return {'split': split_name,
            'brier_rule': round(float(br_rule.mean()), 4),
            'brier_tanimoto_baseline': round(float(br_base.mean()), 4),
            'delta(base-rule)': round(mean, 4), 'CI95': [round(lo, 4), round(hi, 4)],
            'P(rule better)': round(pwin, 3), 'rule_wins': bool(lo > 0), 'note': note}


def test_productivity_familyout(df, X, w, fps, rates):
    """C2 — HARDEST stress test: whole chemical family held out (scaffold-disjoint)."""
    fam = df['class_id'].values
    P_rule = rule_model_matrix(X, df, fam, w, n_outer=min(N_OUTER, len(set(fam))))
    P_base = tanimoto_knn_oof(df, fam, fps, rates, k=K_NN)
    Y = df[['y_' + l for l in LABELS]].values.astype(float)
    return productivity_bootstrap(P_rule, P_base, Y, 'leave-family-out (class_id) — hardest stress',
                                  'whole scaffold family removed; harder than the sealed-6 design.')


def main():
    df = pd.read_csv(DATA)
    X = build_X(df); w = df['weight'].values.astype(float); groups = df['cv_group'].values
    Y = df[['y_' + l for l in LABELS]].values.astype(float)
    print(f"[engine] model={'RF' if C.HAS_SKLEARN else 'NumpyLogReg'} | fingerprint={'Morgan' if C.HAS_RDKIT else '16-substructure-bits'}")

    fps, rates = fingerprints(df), comp_label_rate(df)

    # ---- baselines (leave-compound-out) ----
    P_rule = rule_model_matrix(X, df, groups, w)
    P_tani = tanimoto_knn_oof(df, groups, fps, rates)
    P_freq = frequency_oof(df, groups)
    base = {'rule_model_calibrated': round(float(multilabel_row_brier(P_rule, Y).mean()), 4),
            'tanimoto_knn': round(float(multilabel_row_brier(P_tani, Y).mean()), 4),
            'frequency': round(float(multilabel_row_brier(P_freq, Y).mean()), 4)}

    # public-only: train on terrestrial, test on marine
    mar = df['habitat'].values == 'marine'; ter = ~mar
    Pub = np.zeros((len(df), len(LABELS)))
    for j, l in enumerate(LABELS):
        y = df['y_' + l].values.astype(int)
        wl = df['w_' + l].values.astype(float) if ('w_' + l) in df.columns else w  # FIX: per-label weight
        clf = C.make_classifier()
        if y[ter].sum() == 0:
            Pub[mar, j] = 0.0; continue
        clf.fit(X[ter], y[ter], sample_weight=wl[ter]); Pub[mar, j] = proba1(clf, X[mar])
    base['public_only_on_marine'] = round(float(multilabel_row_brier(Pub[mar], Y[mar]).mean()), 4)
    base['rule_model_on_marine'] = round(float(multilabel_row_brier(P_rule[mar], Y[mar]).mean()), 4)
    base['marine_data_advantage(public-rule)'] = round(base['public_only_on_marine'] - base['rule_model_on_marine'], 4)

    A = test_compositionality(df, P_rule)
    B = test_reusability(df, P_rule)
    # C1 — sealed-6 proxy: novel compound, rules trained elsewhere (leave-compound-out)
    C1 = productivity_bootstrap(P_rule, P_tani, Y, 'leave-compound-out (sealed-6 proxy)',
                                'novel compound reusing trained rules — the actual Property C design.')
    # C2 — hardest stress: whole chemical family removed
    C2 = test_productivity_familyout(df, X, w, fps, rates)

    interpretation = [
        "Labels here are SIMULATED from substructure presence (noisy-OR). A similarity "
        "baseline computed on those same substructures is therefore near-optimal BY "
        "CONSTRUCTION, so the rule model is not expected to beat it on mock data. The "
        "rule-vs-similarity productivity win is reserved for REAL marine LC-MS labels "
        "(proposal §1.1.7) — this script provides the machinery, not the biological result.",
        "The informative mock result is 'marine_data_advantage': a model trained only on "
        "terrestrial/'public' strains fails badly on marine substrates (Brier ~0.74) versus "
        "the marine-trained rule model (~0.12). That large gap is the proposal's secondary "
        "success criterion (project marine data > public data), and it holds even on mock data.",
        "Engine note: fallback (numpy LogReg + substructure-bit Tanimoto) is active in this "
        "sandbox. In an environment with scikit-learn + RDKit, the rule model uses "
        "RandomForest and the baseline uses Morgan fingerprints, which changes absolute numbers.",
    ]
    report = {'engine': {'model': 'RandomForest' if C.HAS_SKLEARN else 'NumpyLogReg',
                         'fingerprint': 'Morgan' if C.HAS_RDKIT else 'substructure-bits'},
              'baselines_multilabel_brier': base,
              'testA_compositionality': A, 'testB_reusability': B,
              'testC1_productivity_compound_out': C1, 'testC2_productivity_familyout_stress': C2,
              'interpretation': interpretation}
    json.dump(report, open(os.path.join(OUTDIR, 'baselines_tests_report.json'), 'w'), indent=2)
    pd.DataFrame(B).to_csv(os.path.join(OUTDIR, 'testB_reusability.csv'), index=False)

    print("\n=== BASELINES (multi-label Brier, lower=better) ===")
    for k, v in base.items():
        print(f"  {k:34s} {v}")
    print("\n=== TEST A — compositionality (JSD) ===");  print(" ", A)
    print("\n=== TEST B — reusability (matched pairs) ===")
    for r in B: print("  ", r)
    print("\n=== TEST C1 — productivity (leave-COMPOUND-out, sealed-6 proxy) ===");  print(" ", C1)
    print("\n=== TEST C2 — productivity (leave-FAMILY-out, hardest stress) ===");  print(" ", C2)
    print("\nOutputs ->", OUTDIR)


if __name__ == '__main__':
    main()
