#!/usr/bin/env python3
"""PILOT (NEW FILE, 2026-08-17) — explicit sealed-6 evaluation.

The shipped pipeline only has a leave-compound-out *proxy* for the sealed set.
Here the split is the real proposal design: train on the 20 marine training
compounds, predict the 6 sealed compounds that the model has never seen.
Baselines: Tanimoto-kNN on substructure bits, and per-label train prevalence.
ALL LABELS ARE SIMULATED (noisy-OR from data/inputs/reaction_rules.csv).
"""
import os, sys, json
import numpy as np, pandas as pd
ROOT=os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0,os.path.join(ROOT,'ml_upgrade'))
import _core as C
from sklearn.ensemble import RandomForestClassifier

PILOT=os.path.dirname(ROOT)
DF=pd.read_csv(os.path.join(ROOT,'data','ml_training_dataset_v3.csv'))
PANEL=pd.read_csv(os.path.join(PILOT,'data','inputs','compounds.csv'))
role=dict(zip(PANEL.compound_id,PANEL.panel_role)); name=dict(zip(PANEL.compound_id,PANEL.compound_name))
DF['panel_role']=DF.compound_id.map(role)
RES=os.path.join(PILOT,'results'); FIG=os.path.join(PILOT,'figures')
os.makedirs(RES,exist_ok=True); os.makedirs(FIG,exist_ok=True)

SUB=[c for c in ['phenol','catechol','pyrogallol','gentisate_25_diol','guaiacyl','syringyl',
 'aromatic_methoxy','monophenol_para','carboxylic_acid','benzoic_acid','aromatic_aldehyde',
 'vinyl_aromatic','ester_bond','aryl_O_glycoside','aryl_sulfate_ester','flavone_core'] if c in DF.columns]
# --- PILOT PATCH 2026-09-07 (vocabulary v4b): additive match-count columns.
SUB = SUB + [c for c in DF.columns if c.startswith('n_') and c not in SUB]
LAB=[c for c in DF.columns if c.startswith('y_')]
def build_X(df):
    return pd.concat([df[SUB],pd.get_dummies(df['strain_phylum'],prefix='phylum').astype(int),
                      pd.get_dummies(df['role'],prefix='role').astype(int)],axis=1)

X_all=build_X(DF); X_all=X_all.reindex(sorted(X_all.columns),axis=1)
tr=(DF.panel_role=='training').values; se=(DF.panel_role=='sealed').values
Xtr,Xse=X_all[tr].values.astype(float),X_all[se].values.astype(float)
w=DF['weight'].values.astype(float)
print(f'train rows {tr.sum()} ({DF[tr].compound_id.nunique()} compounds) | '
      f'sealed rows {se.sum()} ({DF[se].compound_id.nunique()} compounds) | labels {len(LAB)}')

# ---------------- models ------------------------------------------------------
P_rule=np.zeros((se.sum(),len(LAB))); P_freq=np.zeros_like(P_rule)
for j,l in enumerate(LAB):
    y=DF[l].values.astype(int)
    ytr=y[tr]
    P_freq[:,j]=ytr.mean()
    if ytr.sum()<3 or ytr.sum()==len(ytr):
        P_rule[:,j]=ytr.mean(); continue
    clf=RandomForestClassifier(n_estimators=400,min_samples_leaf=3,class_weight='balanced',
                               random_state=42,n_jobs=-1).fit(Xtr,ytr,sample_weight=w[tr])
    P_rule[:,j]=clf.predict_proba(Xse)[:,list(clf.classes_).index(1)] if 1 in clf.classes_ else 0.0

# Tanimoto-kNN baseline on substructure bits (compound level)
bits={cid:g[SUB].iloc[0].values.astype(int) for cid,g in DF.groupby('compound_id')}
rate={}
for l in LAB:
    for cid,g in DF[tr].groupby('compound_id'): rate.setdefault(l,{})[cid]=g[l].mean()
tr_cids=sorted(DF[tr].compound_id.unique())
def tanimoto(a,b):
    inter=np.logical_and(a,b).sum(); uni=np.logical_or(a,b).sum()
    return inter/uni if uni else 0.0
P_tani=np.zeros_like(P_rule); K=5
se_cids=DF[se].compound_id.values
nn_report={}
for i,cid in enumerate(se_cids):
    sims=sorted(((tanimoto(bits[cid],bits[t]),t) for t in tr_cids),reverse=True)[:K]
    tot=sum(s for s,_ in sims) or 1.0
    nn_report[cid]=[(t,round(s,3)) for s,t in sims]
    for j,l in enumerate(LAB):
        P_tani[i,j]=sum(s*rate[l][t] for s,t in sims)/tot
Y=DF[se][LAB].values.astype(float)

def brier_rows(P,Y): return ((P-Y)**2).mean(axis=1)
res={}
for nm,P in [('rule_model',P_rule),('tanimoto_knn',P_tani),('frequency',P_freq)]:
    res[nm]=round(float(brier_rows(P,Y).mean()),4)
# paired cluster bootstrap by compound
rng=np.random.default_rng(42); cl=se_cids; uc=np.unique(cl); d=[]
br=brier_rows(P_rule,Y); bb=brier_rows(P_tani,Y)
for _ in range(2000):
    pick=rng.choice(uc,len(uc),replace=True)
    idx=np.concatenate([np.where(cl==c)[0] for c in pick])
    d.append(bb[idx].mean()-br[idx].mean())
d=np.array(d)
res['delta_tanimoto_minus_rule']=round(float(d.mean()),4)
res['CI95']=[round(float(np.percentile(d,2.5)),4),round(float(np.percentile(d,97.5)),4)]
res['P(rule better)']=round(float((d>0).mean()),3)
res['rule_wins']=bool(res['CI95'][0]>0)

# ---------------- per-compound / per-class ------------------------------------
rows=[]
for i,cid in enumerate(se_cids):
    pass
per_cmp=[]
for cid in DF[se].compound_id.unique():
    m=(se_cids==cid)
    per_cmp.append(dict(compound_id=cid,compound_name=name[cid],n_rows=int(m.sum()),
        brier_rule=round(float(brier_rows(P_rule[m],Y[m]).mean()),4),
        brier_tanimoto=round(float(brier_rows(P_tani[m],Y[m]).mean()),4),
        nearest_training=' ; '.join(f'{name[t]} ({s})' for t,s in nn_report[cid])))
per_cmp=pd.DataFrame(per_cmp); per_cmp.to_csv(os.path.join(RES,'sealed_per_compound.csv'),index=False)

per_lab=[]
for j,l in enumerate(LAB):
    y=Y[:,j]; p=P_rule[:,j]
    prev=float(y.mean())
    ap=C.__dict__.get('average_precision')
    try:
        from sklearn.metrics import average_precision_score,roc_auc_score
        apv=float(average_precision_score(y,p)) if 0<y.sum()<len(y) else np.nan
        auc=float(roc_auc_score(y,p)) if 0<y.sum()<len(y) else np.nan
    except Exception: apv,auc=np.nan,np.nan
    per_lab.append(dict(label=l.replace('y_',''),prevalence_sealed=round(prev,3),
        n_pos=int(y.sum()),PR_AUC=None if np.isnan(apv) else round(apv,3),
        lift=None if (np.isnan(apv) or prev==0) else round(apv/prev,2),
        ROC_AUC=None if np.isnan(auc) else round(auc,3),
        mean_conf_pos=round(float(p[y==1].mean()),3) if y.sum() else None,
        mean_conf_neg=round(float(p[y==0].mean()),3) if (y==0).sum() else None))
per_lab=pd.DataFrame(per_lab); per_lab.to_csv(os.path.join(RES,'sealed_per_label.csv'),index=False)

# DMSP abstention
h6=(se_cids=='H06')
dmsp=dict(mean_p_any_reaction=round(float(P_rule[h6][:,[k for k,l in enumerate(LAB) if l!='y_none']].max(axis=1).mean()),4),
          mean_p_none=round(float(P_rule[h6][:,LAB.index('y_none')].mean()),4),
          true_none_rate=round(float(Y[h6][:,LAB.index('y_none')].mean()),4))
res['DMSP_abstention']=dmsp
json.dump({'brier':res,'note':'SIMULATED labels (noisy-OR). Engineering/rule-recovery result, not biology.'},
          open(os.path.join(RES,'sealed_eval_summary.json'),'w'),indent=2)
np.save(os.path.join(RES,'P_rule_sealed.npy'),P_rule); np.save(os.path.join(RES,'Y_sealed.npy'),Y)
np.save(os.path.join(RES,'P_tani_sealed.npy'),P_tani)
pd.DataFrame(P_rule,columns=[l.replace('y_','p_') for l in LAB]).assign(compound_id=se_cids)\
  .to_csv(os.path.join(RES,'sealed_predictions_raw.csv'),index=False)
print(json.dumps(res,indent=2))
print(); print(per_cmp.to_string(index=False)); print(); print(per_lab.to_string(index=False))
