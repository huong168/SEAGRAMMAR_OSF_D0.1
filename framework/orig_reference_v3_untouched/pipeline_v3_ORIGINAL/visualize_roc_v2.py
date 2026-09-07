import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_curve, auc

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, 'data', 'ml_training_dataset_v3.csv')
OUTDIR = os.path.join(ROOT, 'visualizations', 'model_v3')
os.makedirs(OUTDIR, exist_ok=True)

SUBSTRUCT = ['phenol','catechol','pyrogallol','gentisate_25_diol','guaiacyl','syringyl',
             'aromatic_methoxy','monophenol_para','carboxylic_acid','benzoic_acid',
             'aromatic_aldehyde','vinyl_aromatic','ester_bond','aryl_O_glycoside',
             'aryl_sulfate_ester','flavone_core']

def build_X(df):
    parts = [df[SUBSTRUCT], pd.get_dummies(df['strain_phylum'], prefix='phylum').astype(int)]
    if 'habitat' in df.columns:
        parts.append(pd.get_dummies(df['habitat'], prefix='habitat').astype(int))
    X = pd.concat(parts, axis=1)
    return X, list(X.columns)

df = pd.read_csv(DATA)
X_df, feat_cols = build_X(df)
X = X_df.values
groups = df['cv_group'].values if 'cv_group' in df.columns else df['compound_id'].values

LABELS = sorted([c for c in df.columns if c.startswith('y_') and c != 'y_none'])

plt.figure(figsize=(12, 10))
colors = plt.cm.tab20(np.linspace(0, 1, len(LABELS)))

for idx, lab in enumerate(LABELS):
    y = df[lab].values
    if y.sum() == 0: continue
    
    oof_proba = np.zeros(len(y))
    splitter = GroupKFold(n_splits=5)
    
    for tr, te in splitter.split(X, y, groups):
        if y[tr].sum() == 0: continue
        clf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1)
        w = df['weight'].values[tr] if 'weight' in df.columns else None
        clf.fit(X[tr], y[tr], sample_weight=w)
        oof_proba[te] = clf.predict_proba(X[te])[:, 1] if clf.classes_.shape[0] > 1 else 0
        
    # Mask to ignore items that were never validated (if any)
    mask = (oof_proba > 0) | (y > 0) | (oof_proba == 0) # Just valid numbers
    
    fpr, tpr, _ = roc_curve(y[mask], oof_proba[mask])
    roc_auc = auc(fpr, tpr)
    
    if not np.isnan(roc_auc):
        name = lab.replace('y_', '')
        plt.plot(fpr, tpr, color=colors[idx], lw=2.5, label=f'{name} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], color='black', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=14)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=14)
plt.title('Đường cong ROC-AUC (Line Chart) cho Dataset v3', fontsize=18, fontweight='bold', pad=20)
plt.legend(loc="lower right", fontsize=11, framealpha=0.9)
plt.grid(alpha=0.4, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'roc_curves_line.png'), dpi=300)
print(f"Saved ROC curves line chart to {OUTDIR}/roc_curves_line.png")
