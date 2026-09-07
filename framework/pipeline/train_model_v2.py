#!/usr/bin/env python3
# train_model_v2.py
# =============================================================================
# Baseline training cho ml_training_dataset_v2 — multi-label, CHỐNG RÒ RỈ.
#
# Nguyên tắc (theo README_dataset_v2.md):
#   • Feature đầu vào = 16 substructure (RDKit) + one-hot strain_phylum.
#     KHÔNG dùng: source, strain_id, compound_id, cv_group, smiles, weight, replicate_id.
#   • Multi-label = binary relevance (một classifier / nhãn), class_weight='balanced'.
#   • Đánh giá = GroupKFold(groups=cv_group) -> leave-compound-out, không rò rỉ.
#   • sample_weight = cột weight (evidence tier).
#   • Metric = F1, PR-AUC (average precision), ROC-AUC, support — TỪNG NHÃN + macro.
#     KHÔNG dùng accuracy.
#   • So sánh đối chứng: KFold ngẫu nhiên (rò rỉ) để định lượng mức thổi phồng.
# =============================================================================
import os, json, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, KFold
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score, precision_score, recall_score

ROOT = os.path.dirname(os.path.abspath(__file__))
def _resolve_data_dir(root):
    d = os.environ.get('SEAGRAMMAR_DATA')
    if d:
        return d
    for cand in ('data', 'data1'):
        if os.path.exists(os.path.join(root, cand, 'ml_training_dataset_v3.csv')):
            return os.path.join(root, cand)
    return os.path.join(root, 'data')
DATA = os.path.join(_resolve_data_dir(ROOT), 'ml_training_dataset_v3.csv')  # FIX: data/ vs data1/
OUTDIR = os.path.join(ROOT, 'visualizations', 'model_v3'); os.makedirs(OUTDIR, exist_ok=True)
# FIX: cổng chặn theo SỐ HỢP CHẤT mang σ (đúng ý spec §6: σ phải ở >=N substrate;
# mục tiêu N=10 ở panel funded). MIN_POS chỉ đếm dòng nên không bảo vệ leave-compound-out.
SEED, N_SPLITS, MIN_POS, MIN_COMPOUNDS = 42, 5, 8, 3
N_TREES, N_TREES_IMP = 50, 60

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
def build_X(df):
    # feature = 16 substructure (RDKit) + one-hot phylum + one-hot habitat (covariate).
    # habitat cho phép model học hiệu ứng môi trường sống (biển vs cạn) thay vì bị nhiễu.
    parts = [df[_present(df, SUBSTRUCT)], pd.get_dummies(df['strain_phylum'], prefix='phylum').astype(int)]
    if 'habitat' in df.columns:
        parts.append(pd.get_dummies(df['habitat'], prefix='habitat').astype(int))
    X = pd.concat(parts, axis=1)
    return X, list(X.columns)


def cv_metrics(X, y, groups, w, splitter, use_groups):
    """Trả về dict metric trung bình qua các fold."""
    oof_pred = np.zeros(len(y)); oof_proba = np.zeros(len(y)); filled = np.zeros(len(y), bool)
    it = splitter.split(X, y, groups) if use_groups else splitter.split(X)
    for tr, te in it:
        if y[tr].sum() == 0:        # fold train không có positive -> bỏ qua
            continue
        clf = RandomForestClassifier(n_estimators=N_TREES, random_state=SEED,
                                     class_weight='balanced', n_jobs=-1)
        clf.fit(X[tr], y[tr], sample_weight=w[tr])
        oof_pred[te] = clf.predict(X[te])
        oof_proba[te] = clf.predict_proba(X[te])[:, 1] if clf.classes_.shape[0] > 1 else 0
        filled[te] = True
    m = filled   # FIX: bỏ np.isin(arange,arange) luôn-True (code chết)
    yt, yp, ypr = y[m], oof_pred[m].astype(int), oof_proba[m]
    out = {'f1': f1_score(yt, yp, zero_division=0),
           'precision': precision_score(yt, yp, zero_division=0),
           'recall': recall_score(yt, yp, zero_division=0),
           'pr_auc': average_precision_score(yt, ypr) if yt.sum() > 0 else np.nan,
           'roc_auc': roc_auc_score(yt, ypr) if 0 < yt.sum() < len(yt) else np.nan}
    return out


def main():
    df = pd.read_csv(DATA)
    X_df, feat_names = build_X(df)
    X = X_df.values.astype(float)
    w = df['weight'].values.astype(float)
    groups = df['cv_group'].values
    ycols = [c for c in df.columns if c.startswith('y_') and c != 'y_none']

    gkf = GroupKFold(n_splits=N_SPLITS)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    rows, importances = [], []
    for col in ycols:
        y = df[col].values.astype(int); pos = int(y.sum())
        n_comp = int(df.loc[y == 1, 'compound_id'].nunique())   # FIX: số hợp chất mang σ
        if pos < MIN_POS or n_comp < MIN_COMPOUNDS:
            rows.append({'label': col[2:], 'support': pos, 'n_compounds': n_comp,
                         'note': f'σ chỉ ở {n_comp} hợp chất (<{MIN_COMPOUNDS}) — không đánh giá được (leave-compound-out)'})
            print(f"  {col[2:]:24s} n={pos:4d}  BỎ QUA (σ chỉ ở {n_comp} hợp chất)", flush=True); continue
        wl = df['w_' + col[2:]].values.astype(float) if ('w_' + col[2:]) in df.columns else w  # FIX: per-label weight
        g = cv_metrics(X, y, groups, wl, gkf, True)        # TRUNG THỰC (leave-compound-out)
        rows.append({'label': col[2:], 'support': pos, 'n_compounds': n_comp,
                     'f1_group': round(g['f1'], 3), 'pr_auc_group': round(g['pr_auc'], 3),
                     'roc_auc_group': round(g['roc_auc'], 3),
                     'precision_group': round(g['precision'], 3), 'recall_group': round(g['recall'], 3)})
        print(f"  {col[2:]:24s} n={pos:4d}  F1={g['f1']:.3f}  PR-AUC={g['pr_auc']:.3f}", flush=True)
        # feature importance (fit toàn bộ)
        clf = RandomForestClassifier(n_estimators=N_TREES_IMP, random_state=SEED, class_weight='balanced', n_jobs=-1)
        clf.fit(X, y, sample_weight=wl)   # FIX: per-label weight
        importances.append(pd.Series(clf.feature_importances_, index=feat_names, name=col[2:]))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUTDIR, 'per_label_metrics.csv'), index=False)
    imp = pd.concat(importances, axis=1)
    imp.to_csv(os.path.join(OUTDIR, 'feature_importance.csv'))

    ev = res.dropna(subset=['f1_group'])
    macro = {'macro_f1_group': round(ev['f1_group'].mean(), 3),
             'macro_pr_auc_group': round(ev['pr_auc_group'].mean(), 3),
             'macro_roc_auc_group': round(ev['roc_auc_group'].mean(), 3),
             'n_labels_evaluated': int(len(ev))}
    json.dump(macro, open(os.path.join(OUTDIR, 'macro_summary.json'), 'w'), indent=2)

    # ---- CHART 1: per-label F1 (GroupKFold trung thực) ----
    fig, ax = plt.subplots(figsize=(11, 6)); o = ev.sort_values('f1_group')
    yy = np.arange(len(o))
    ax.barh(yy, o['f1_group'], 0.6, color='#31a354', label='GroupKFold (leave-compound-out)')
    for i, (_, r) in enumerate(o.iterrows()): ax.text(0.005, i, f"n={r['support']}", va='center', fontsize=8, color='white')
    ax.set_yticks(yy); ax.set_yticklabels(o['label']); ax.set_xlabel('F1'); ax.legend(loc='lower right')
    ax.set_title(f"Baseline RandomForest — F1 từng nhãn (đánh giá trung thực)\nmacro-F1={macro['macro_f1_group']} | macro PR-AUC={macro['macro_pr_auc_group']}")
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, '1_per_label_f1.png'), dpi=130); plt.close()

    # ---- CHART 2: PR-AUC vs baseline prevalence ----
    fig, ax = plt.subplots(figsize=(11, 6)); o = ev.sort_values('pr_auc_group')
    base = [df['y_' + l].mean() for l in o['label']]
    yy = np.arange(len(o))
    ax.barh(yy, o['pr_auc_group'], 0.6, color='#2c7fb8', label='PR-AUC (GroupKFold)')
    ax.scatter(base, yy, color='k', zorder=3, label='prevalence (baseline ngẫu nhiên)')
    ax.set_yticks(yy); ax.set_yticklabels(o['label']); ax.set_xlabel('PR-AUC'); ax.legend(loc='lower right')
    ax.set_title('PR-AUC vs baseline prevalence — điểm trên vạch đen = học được tín hiệu thật')
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, '2_pr_auc.png'), dpi=130); plt.close()

    # ---- CHART 3: feature importance heatmap ----
    fig, ax = plt.subplots(figsize=(12, 8))
    im = ax.imshow(imp.values, aspect='auto', cmap='viridis')
    ax.set_xticks(range(imp.shape[1])); ax.set_xticklabels(imp.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(imp.shape[0])); ax.set_yticklabels(imp.index, fontsize=8)
    fig.colorbar(im, ax=ax, label='importance'); ax.set_title('Feature importance theo nhãn (RandomForest)')
    plt.tight_layout(); plt.savefig(os.path.join(OUTDIR, '3_feature_importance.png'), dpi=130); plt.close()

    print(res.to_string(index=False)); print("\nMACRO:", macro)
    print("Outputs ->", OUTDIR)


if __name__ == '__main__':
    main()
