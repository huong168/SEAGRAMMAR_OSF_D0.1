import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

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

rf_aucs = {}
nb_aucs = {}

for lab in LABELS:
    y = df[lab].values
    if y.sum() == 0: continue
    
    rf_oof = np.zeros(len(y))
    nb_oof = np.zeros(len(y))
    
    splitter = GroupKFold(n_splits=5)
    for tr, te in splitter.split(X, y, groups):
        if y[tr].sum() == 0: continue
        w = df['weight'].values[tr] if 'weight' in df.columns else None
        
        # Random Forest
        rf = RandomForestClassifier(n_estimators=50, random_state=42, class_weight='balanced', n_jobs=-1)
        rf.fit(X[tr], y[tr], sample_weight=w)
        rf_oof[te] = rf.predict_proba(X[te])[:, 1] if rf.classes_.shape[0] > 1 else 0
        
        # Bernoulli Naive Bayes 
        nb = BernoulliNB()
        nb.fit(X[tr], y[tr], sample_weight=w)
        nb_oof[te] = nb.predict_proba(X[te])[:, 1] if nb.classes_.shape[0] > 1 else 0
        
    mask = (rf_oof > 0) | (y > 0) | (rf_oof == 0)
    
    try:
        rf_auc = roc_auc_score(y[mask], rf_oof[mask])
        nb_auc = roc_auc_score(y[mask], nb_oof[mask])
        name = lab.replace('y_', '')
        rf_aucs[name] = rf_auc
        nb_aucs[name] = nb_auc
    except ValueError:
        pass

print("=== SO SÁNH ROC-AUC: Random Forest vs Bernoulli Naive Bayes ===")
labels_found = list(rf_aucs.keys())
for name in labels_found:
    print(f"{name:25s} | RF: {rf_aucs[name]:.3f} | Naive Bayes: {nb_aucs[name]:.3f}")

macro_rf = np.mean(list(rf_aucs.values()))
macro_nb = np.mean(list(nb_aucs.values()))
print("-" * 50)
print(f"{'MACRO AVERAGE':25s} | RF: {macro_rf:.3f} | Naive Bayes: {macro_nb:.3f}")

# Plot
plt.figure(figsize=(14, 8))
x = np.arange(len(labels_found))
width = 0.35

plt.bar(x - width/2, [rf_aucs[l] for l in labels_found], width, label=f'Random Forest (Macro={macro_rf:.3f})', color='#2ca02c')
plt.bar(x + width/2, [nb_aucs[l] for l in labels_found], width, label=f'Bernoulli Naive Bayes (Macro={macro_nb:.3f})', color='#ff7f0e')

plt.ylabel('ROC-AUC Score', fontsize=12)
plt.title('So sánh thuật toán: Random Forest vs Bernoulli Naive Bayes (Dataset v3)', fontsize=16, fontweight='bold', pad=20)
plt.xticks(x, labels_found, rotation=45, ha='right', fontsize=11)
plt.ylim(0.5, 1.05)
plt.legend(fontsize=12)
plt.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUTDIR, 'bayes_vs_rf.png'), dpi=300)
print(f"Saved comparison chart to {OUTDIR}/bayes_vs_rf.png")
