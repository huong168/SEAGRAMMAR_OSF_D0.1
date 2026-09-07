#!/usr/bin/env python3
# ml_upgrade/calibrate_eval.py
# =============================================================================
# UPGRADE 1 — Probability calibration + first-class "no-reaction".
#
# The rule σ ⊢ {(ρ,p)} needs p to be a *real* probability (for Brier scoring and
# the no-reaction call). Tree/score outputs are not calibrated. We add group-safe
# isotonic calibration and treat y_none as its own calibrated label.
#
# Method (leave-compound-out, no leakage):
#   outer GroupKFold(compound):
#     inner GroupKFold(compound) splits outer-train into FIT / CALIB
#     fit classifier on FIT; fit isotonic(raw_score -> label) on CALIB
#   apply classifier then isotonic to the outer TEST fold -> calibrated OOF prob.
#
# Runs with sklearn RandomForest if available, else numpy logistic (see _core).
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
N_OUTER, N_INNER, MIN_POS = 5, 4, 12

SUBSTRUCT = ['phenol','catechol','pyrogallol','gentisate_25_diol','guaiacyl','syringyl',
             'aromatic_methoxy','monophenol_para','carboxylic_acid','benzoic_acid',
             'aromatic_aldehyde','vinyl_aromatic','ester_bond','aryl_O_glycoside',
             'aryl_sulfate_ester','flavone_core']


def build_X(df):
    parts = [df[SUBSTRUCT], pd.get_dummies(df['strain_phylum'], prefix='phylum').astype(int)]
    if 'habitat' in df.columns:
        parts.append(pd.get_dummies(df['habitat'], prefix='habitat').astype(int))
    return pd.concat(parts, axis=1).values.astype(float)


def proba1(clf, X):
    p = clf.predict_proba(X)
    return p[:, 1] if getattr(clf, 'classes_', np.array([0, 1])).shape[0] > 1 else np.zeros(len(X))


def calibrated_oof(X, y, groups, w):
    n = len(y); raw = np.full(n, np.nan); cal = np.full(n, np.nan)
    for tr, te in C.group_kfold(groups, N_OUTER):
        if y[tr].sum() == 0:
            continue
        fit_i, cal_i = next(C.group_kfold(groups[tr], N_INNER))
        fit, calib = tr[fit_i], tr[cal_i]
        clf = C.make_classifier()
        if y[fit].sum() == 0 or y[calib].sum() == 0:
            clf.fit(X[tr], y[tr], sample_weight=w[tr])
            p = proba1(clf, X[te]); raw[te] = p; cal[te] = p; continue
        clf.fit(X[fit], y[fit], sample_weight=w[fit])
        s_cal = proba1(clf, X[calib])
        xs, fitted = C.pav_isotonic_fit(s_cal, y[calib])
        s_te = proba1(clf, X[te])
        raw[te] = s_te; cal[te] = C.pav_predict(xs, fitted, s_te)
    return raw, cal


def reliability(y, p, bins=10):
    m = ~np.isnan(p); y, p = y[m], p[m]; edges = np.linspace(0, 1, bins + 1); rows = []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        sel = (p >= lo) & ((p < hi) if i < bins - 1 else (p <= hi))
        if sel.sum() == 0:
            continue
        rows.append({'bin_lo': round(lo, 2), 'bin_hi': round(hi, 2),
                     'mean_pred': round(float(p[sel].mean()), 4),
                     'obs_freq': round(float(y[sel].mean()), 4), 'n': int(sel.sum())})
    return pd.DataFrame(rows)


def main():
    df = pd.read_csv(DATA)
    X = build_X(df); w = df['weight'].values.astype(float); groups = df['cv_group'].values
    ycols = [c for c in df.columns if c.startswith('y_')]
    print(f"[engine] classifier = {'sklearn RandomForest' if C.HAS_SKLEARN else 'numpy LogisticRegression (fallback)'}")
    rows = []
    for col in ycols:
        y = df[col].values.astype(int); pos = int(y.sum()); label = col[2:]
        if pos < MIN_POS:
            rows.append({'label': label, 'support': pos, 'note': 'too rare'}); continue
        wl = df['w_' + label].values.astype(float) if ('w_' + label) in df.columns else w  # FIX: per-label weight
        raw, cal = calibrated_oof(X, y, groups, wl)
        b_raw, b_cal = C.brier(y, raw), C.brier(y, cal)
        rows.append({'label': label, 'support': pos, 'brier_raw': round(b_raw, 4),
                     'brier_calibrated': round(b_cal, 4), 'improvement': round(b_raw - b_cal, 4)})
        reliability(y, cal).to_csv(os.path.join(OUTDIR, f'reliability_{label}.csv'), index=False)
        tag = '  <-- NO-REACTION (first-class)' if label == 'none' else ''
        print(f"  {label:24s} n={pos:4d}  Brier raw={b_raw:.4f} -> cal={b_cal:.4f}  (Δ={b_raw-b_cal:+.4f}){tag}")
    res = pd.DataFrame(rows); res.to_csv(os.path.join(OUTDIR, 'calibration_metrics.csv'), index=False)
    ev = res.dropna(subset=['brier_calibrated'])
    summary = {'engine': 'RandomForest' if C.HAS_SKLEARN else 'NumpyLogReg',
               'n_labels_evaluated': int(len(ev)),
               'macro_brier_raw': round(float(ev['brier_raw'].mean()), 4),
               'macro_brier_calibrated': round(float(ev['brier_calibrated'].mean()), 4),
               'macro_improvement': round(float(ev['improvement'].mean()), 4),
               'no_reaction_brier_calibrated': (round(float(ev.loc[ev.label == 'none', 'brier_calibrated'].iloc[0]), 4)
                                                if (ev.label == 'none').any() else None),
               'note': 'leave-compound-out, group-safe; mock/simulated labels (engineering feasibility).'}
    json.dump(summary, open(os.path.join(OUTDIR, 'calibration_summary.json'), 'w'), indent=2)
    print("\nMACRO:", summary); print("Outputs ->", OUTDIR)


if __name__ == '__main__':
    main()
