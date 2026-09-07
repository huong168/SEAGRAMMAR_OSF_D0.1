#!/usr/bin/env python3
# ml_upgrade/honest_eval.py
# =============================================================================
# UPGRADE 4 — "Honest evaluation" module.
#
# Addresses four methodological concerns about the current prototype numbers:
#
#   (1) POSITIVE-CONTROL FRAMING. Labels are SIMULATED from a noisy-OR over
#       substructures. Any metric here therefore measures how well the pipeline
#       RECOVERS ITS OWN GENERATING RULE (identifiability / positive control) —
#       NOT a biological result. Every number this script prints is tagged as
#       such. Real predictive numbers only exist once LC-MS labels replace the
#       simulated ones (proposal §1.1.7).
#
#   (2) LCO vs LFO IN PARALLEL. leave-compound-out (LCO, groups=compound_id)
#       still leaks signal because same-scaffold compounds have near-identical
#       feature vectors. leave-family-out (LFO, groups=class_id) is the honest
#       generalisation test. We report BOTH side by side, plus the LCO-LFO GAP
#       — the gap itself quantifies scaffold leakage.
#
#   (3) CLUSTER-BOOTSTRAP 95% CI. Confidence intervals are computed by resampling
#       whole COMPOUND groups with replacement (not rows), respecting the
#       replicate dependence. With 32 compounds and rare labels the CIs are wide
#       BY DESIGN — that honesty is the point, and it motivates a larger panel.
#
#   (4) PR-AUC + LIFT as the PRIMARY metric, ROC-AUC secondary. Under heavy
#       class imbalance (all-zero baseline ~90%) ROC-AUC flatters; PR-AUC (average
#       precision) discriminates better. Raw PR-AUC is not comparable across
#       labels (its floor = prevalence), so we report LIFT = PR-AUC / prevalence.
#       ROC-AUC is kept as a threshold-free, prevalence-comparable secondary.
#
# Runs with scikit-learn (RandomForest) when present, else the numpy fallback in
# _core.py (NumpyLogReg). Metric math (AP / ROC / bootstrap) is pure numpy and
# deterministic, so it is identical in both environments.
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


DATA = os.path.join(_resolve_data_dir(ROOT), 'ml_training_dataset_v3.csv')
OUTDIR = os.path.join(ROOT, 'visualizations', 'ml_upgrade'); os.makedirs(OUTDIR, exist_ok=True)

SEED, N_OUTER, N_INNER = 42, 5, 4
MIN_POS, MIN_COMPOUNDS = 8, 3        # same gate as train_model_v2.py (patched)
N_BOOT = int(os.environ.get('HONEST_EVAL_NBOOT', '2000'))   # cluster-bootstrap resamples (env-tunable)

SUBSTRUCT = ['phenol', 'catechol', 'pyrogallol', 'gentisate_25_diol', 'guaiacyl', 'syringyl',
             'aromatic_methoxy', 'monophenol_para', 'carboxylic_acid', 'benzoic_acid',
             'aromatic_aldehyde', 'vinyl_aromatic', 'ester_bond', 'aryl_O_glycoside',
             'aryl_sulfate_ester', 'flavone_core']
REACTIONS = ['O_demethylation', 'aldehyde_oxidation', 'aromatic_hydroxylation', 'decarboxylation',
             'deglycosylation', 'desulfation', 'ester_hydrolysis', 'methylation', 'reduction',
             'ring_cleavage', 'side_chain_cleavage']


# --------------------------- feature matrix ----------------------------------
def build_X(df):
    parts = [df[SUBSTRUCT], pd.get_dummies(df['strain_phylum'], prefix='phylum').astype(int)]
    if 'habitat' in df.columns:
        parts.append(pd.get_dummies(df['habitat'], prefix='habitat').astype(int))
    return pd.concat(parts, axis=1).values.astype(float)


def proba1(clf, X):
    p = clf.predict_proba(X)
    return p[:, 1] if getattr(clf, 'classes_', np.array([0, 1])).shape[0] > 1 else np.zeros(len(X))


# --------------------------- deterministic metrics ---------------------------
def _rankdata_avg(a):
    """Average ranks (1-based) with tie handling — for the AUC rank statistic."""
    a = np.asarray(a, float)
    order = np.argsort(a, kind='mergesort')
    ranks = np.empty(len(a), float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0     # average of ranks i..j (1-based)
        i = j + 1
    return ranks


def roc_auc(y, p):
    """ROC-AUC via the Mann-Whitney U rank statistic (ties handled)."""
    y = np.asarray(y, int); p = np.asarray(p, float)
    npos = int(y.sum()); nneg = int((y == 0).sum())
    if npos == 0 or nneg == 0:
        return np.nan
    r = _rankdata_avg(p)
    return float((r[y == 1].sum() - npos * (npos + 1) / 2.0) / (npos * nneg))


def average_precision(y, p):
    """Average precision (area under precision-recall, step convention = sklearn)."""
    y = np.asarray(y, int); p = np.asarray(p, float)
    P = int(y.sum())
    if P == 0:
        return np.nan
    order = np.argsort(-p, kind='mergesort')
    ys = y[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / P
    rec_prev = np.concatenate([[0.0], rec[:-1]])
    return float(np.sum((rec - rec_prev) * prec))


# --------------------------- calibrated OOF ----------------------------------
def oof_calibrated(X, y, w, groups):
    """Calibrated leave-GROUP-out OOF probabilities for one label."""
    n = len(y); out = np.full(n, np.nan)
    n_groups = len(set(groups.tolist()))
    n_outer = max(2, min(N_OUTER, n_groups))
    for tr, te in C.group_kfold(groups, n_outer):
        if y[tr].sum() == 0:
            out[te] = 0.0; continue
        n_in = max(2, min(N_INNER, len(set(groups[tr].tolist()))))
        fit_i, cal_i = next(C.group_kfold(groups[tr], n_in))
        fit, calib = tr[fit_i], tr[cal_i]
        clf = C.make_classifier()
        if y[fit].sum() == 0 or y[calib].sum() == 0:
            clf.fit(X[tr], y[tr], sample_weight=w[tr]); out[te] = proba1(clf, X[te]); continue
        clf.fit(X[fit], y[fit], sample_weight=w[fit])
        xs, fit_iso = C.pav_isotonic_fit(proba1(clf, X[calib]), y[calib])
        out[te] = C.pav_predict(xs, fit_iso, proba1(clf, X[te]))
    return out


# --------------------------- cluster bootstrap -------------------------------
def cluster_bootstrap_ci(metric_fn, y, p, clusters, B=N_BOOT, seed=SEED):
    """95% CI by resampling whole COMPOUND clusters with replacement.
    Skips resamples where the metric is undefined (all-pos / all-neg)."""
    y = np.asarray(y); p = np.asarray(p); clusters = np.asarray(clusters)
    uniq = np.array(sorted(set(clusters.tolist())))
    rows_by_c = {c: np.where(clusters == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(B):
        samp = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([rows_by_c[c] for c in samp])
        v = metric_fn(y[rows], p[rows])
        if not np.isnan(v):
            vals.append(v)
    if len(vals) < 0.5 * B:
        return (np.nan, np.nan, len(vals))
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return (float(lo), float(hi), len(vals))


# --------------------------- per-label evaluation ----------------------------
def evaluate_label(df, X, label):
    y = df['y_' + label].values.astype(int)
    w = df['w_' + label].values.astype(float) if ('w_' + label) in df.columns else df['weight'].values.astype(float)
    clusters = df['compound_id'].values
    prev = float(y.mean())
    n_pos = int(y.sum())
    n_comp = int(df.loc[y == 1, 'compound_id'].nunique())

    row = {'label': label, 'prevalence': round(prev, 4), 'n_pos': n_pos, 'n_compounds': n_comp}
    if n_pos < MIN_POS or n_comp < MIN_COMPOUNDS:
        row['gate'] = f'EXCLUDED (n_compounds={n_comp}<{MIN_COMPOUNDS} or n_pos<{MIN_POS}) — not evaluable under leave-compound-out'
        return row
    row['gate'] = 'evaluated'

    for scheme, gcol in [('lco', 'cv_group'), ('lfo', 'class_id')]:
        groups = df[gcol].values
        p = oof_calibrated(X, y, w, groups)
        m = ~np.isnan(p)
        ap = average_precision(y[m], p[m])
        roc = roc_auc(y[m], p[m])
        ap_lo, ap_hi, _ = cluster_bootstrap_ci(average_precision, y[m], p[m], clusters[m])
        roc_lo, roc_hi, _ = cluster_bootstrap_ci(roc_auc, y[m], p[m], clusters[m])
        lift = ap / prev if prev > 0 else np.nan
        row[f'pr_auc_{scheme}'] = round(ap, 3)
        row[f'pr_auc_{scheme}_ci'] = [round(ap_lo, 3), round(ap_hi, 3)]
        row[f'lift_{scheme}'] = round(lift, 2)                 # PR-AUC / prevalence
        row[f'roc_auc_{scheme}'] = round(roc, 3)
        row[f'roc_auc_{scheme}_ci'] = [round(roc_lo, 3), round(roc_hi, 3)]

    row['pr_auc_gap_lco_minus_lfo'] = round(row['pr_auc_lco'] - row['pr_auc_lfo'], 3)
    row['roc_auc_gap_lco_minus_lfo'] = round(row['roc_auc_lco'] - row['roc_auc_lfo'], 3)
    return row


# --------------------------- figure ------------------------------------------
def make_figure(rows_eval):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    labels = [r['label'] for r in rows_eval]
    y = np.arange(len(labels))
    lco = np.array([r['pr_auc_lco'] for r in rows_eval])
    lfo = np.array([r['pr_auc_lfo'] for r in rows_eval])
    lco_err = np.array([[r['pr_auc_lco'] - r['pr_auc_lco_ci'][0] for r in rows_eval],
                        [r['pr_auc_lco_ci'][1] - r['pr_auc_lco'] for r in rows_eval]])
    lfo_err = np.array([[r['pr_auc_lfo'] - r['pr_auc_lfo_ci'][0] for r in rows_eval],
                        [r['pr_auc_lfo_ci'][1] - r['pr_auc_lfo'] for r in rows_eval]])
    prev = np.array([r['prevalence'] for r in rows_eval])

    fig, ax = plt.subplots(figsize=(11, 6))
    h = 0.36
    ax.barh(y + h / 2, lco, h, xerr=lco_err, color='#31a354', ecolor='#555', capsize=2,
            label='PR-AUC — leave-compound-out (optimistic)')
    ax.barh(y - h / 2, lfo, h, xerr=lfo_err, color='#2c7fb8', ecolor='#555', capsize=2,
            label='PR-AUC — leave-family-out (honest)')
    ax.scatter(prev, y, color='k', zorder=5, s=18, label='prevalence (random baseline = PR-AUC floor)')
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('PR-AUC (average precision)'); ax.set_xlim(0, 1)
    ax.legend(loc='lower right', fontsize=8)
    ax.set_title('Honest evaluation (POSITIVE CONTROL on simulated labels)\n'
                 'PR-AUC with 95% cluster-bootstrap CI · LCO vs LFO · black dot = prevalence floor')
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, 'fig_honest_eval.png'), dpi=130); plt.close()


# --------------------------- main --------------------------------------------
def main():
    df = pd.read_csv(DATA)
    X = build_X(df)
    print(f"[engine] model={'RandomForest' if C.HAS_SKLEARN else 'NumpyLogReg (fallback)'}  "
          f"| rows={len(df)}  compounds={df['compound_id'].nunique()}  families={df['class_id'].nunique()}")
    print(f"[framing] labels are SIMULATED (noisy-OR) -> every number below is a POSITIVE CONTROL "
          f"(rule-recovery), NOT a biological result.\n")

    rows = [evaluate_label(df, X, r) for r in REACTIONS]
    ev = [r for r in rows if r.get('gate') == 'evaluated']

    # macro over evaluated labels
    def macro(key):
        vals = [r[key] for r in ev if key in r and not (isinstance(r[key], float) and np.isnan(r[key]))]
        return round(float(np.mean(vals)), 3) if vals else None

    summary = {
        'framing': 'POSITIVE CONTROL on simulated (noisy-OR) labels — measures rule recovery, '
                   'not biology. Real predictive numbers require LC-MS labels (proposal §1.1.7).',
        'primary_metric': 'PR-AUC (average precision) + lift over prevalence',
        'secondary_metric': 'ROC-AUC (threshold-free)',
        'n_labels_evaluated': len(ev),
        'n_labels_excluded_by_gate': len(rows) - len(ev),
        'macro_pr_auc_lco': macro('pr_auc_lco'), 'macro_pr_auc_lfo': macro('pr_auc_lfo'),
        'macro_lift_lco': macro('lift_lco'), 'macro_lift_lfo': macro('lift_lfo'),
        'macro_roc_auc_lco': macro('roc_auc_lco'), 'macro_roc_auc_lfo': macro('roc_auc_lfo'),
        'macro_pr_auc_gap_lco_minus_lfo': macro('pr_auc_gap_lco_minus_lfo'),
        'note_ci': f'95% CI from {N_BOOT} cluster-bootstrap resamples over compound groups; '
                   f'wide CIs reflect only {df["compound_id"].nunique()} compounds (honest, motivates a larger panel).',
    }

    # outputs
    flat = []
    for r in rows:
        rr = dict(r)
        for k in list(rr):
            if k.endswith('_ci') and isinstance(rr[k], list):
                rr[k] = f'[{rr[k][0]}, {rr[k][1]}]'
        flat.append(rr)
    pd.DataFrame(flat).to_csv(os.path.join(OUTDIR, 'honest_eval_per_label.csv'), index=False)
    json.dump({'summary': summary, 'per_label': rows},
              open(os.path.join(OUTDIR, 'honest_eval_summary.json'), 'w'), indent=2)
    if ev:
        make_figure(ev)

    # console
    print(f"{'label':22s} {'prev':>5s} {'PR_LCO':>16s} {'PR_LFO':>16s} {'lift_LFO':>8s} {'gap':>6s}")
    for r in ev:
        print(f"{r['label']:22s} {r['prevalence']:.2f} "
              f"{r['pr_auc_lco']:.2f} {str(r['pr_auc_lco_ci']):>11s} "
              f"{r['pr_auc_lfo']:.2f} {str(r['pr_auc_lfo_ci']):>11s} "
              f"{r['lift_lfo']:>8.1f} {r['pr_auc_gap_lco_minus_lfo']:>6.2f}")
    for r in rows:
        if r.get('gate') != 'evaluated':
            print(f"{r['label']:22s}  -- {r['gate']}")
    print("\nMACRO:", json.dumps({k: v for k, v in summary.items() if k.startswith('macro')}, indent=0))
    print("Outputs ->", OUTDIR, "(honest_eval_per_label.csv, honest_eval_summary.json, fig_honest_eval.png)")


if __name__ == '__main__':
    main()
