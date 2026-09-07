#!/usr/bin/env python3
"""PILOT (NEW FILE) — desulfation drill-down + the single proposal figure + table."""
import os,sys,json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=os.path.dirname(os.path.abspath(__file__)); PILOT=os.path.dirname(ROOT)
RES=os.path.join(PILOT,'results'); FIG=os.path.join(PILOT,'figures')
DF=pd.read_csv(os.path.join(ROOT,'data','ml_training_dataset_v3.csv'))
PANEL=pd.read_csv(os.path.join(PILOT,'data','inputs','compounds.csv'))
name=dict(zip(PANEL.compound_id,PANEL.compound_name)); role=dict(zip(PANEL.compound_id,PANEL.panel_role))
DF['panel_role']=DF.compound_id.map(role)
se=DF[DF.panel_role=='sealed'].reset_index(drop=True)
LAB=[c for c in DF.columns if c.startswith('y_')]
P=np.load(os.path.join(RES,'P_rule_sealed.npy')); T=np.load(os.path.join(RES,'P_tani_sealed.npy'))
Y=np.load(os.path.join(RES,'Y_sealed.npy'))
cid=se.compound_id.values
BLUE='#4472C4'; ORANGE='#ED7D31'; GREY='#8C8C8C'; RED='#C00000'; GREEN='#548235'

# ---- desulfation drill-down -------------------------------------------------
j=LAB.index('y_desulfation'); rows=[]
for c in ['H01','H02']:
    m=cid==c; y=Y[m,j]; p=P[m,j]; t=T[m,j]
    rows.append(dict(compound_id=c,compound=name[c],n_strains=int(m.sum()),
      true_positive_rate=round(float(y.mean()),3),
      mean_p_rule=round(float(p.mean()),3), mean_p_tanimoto=round(float(t.mean()),3),
      ROC_AUC_rule=round(float(roc_auc_score(y,p)),3) if 0<y.sum()<len(y) else None,
      ROC_AUC_tanimoto=round(float(roc_auc_score(y,t)),3) if 0<y.sum()<len(y) else None,
      PR_AUC_rule=round(float(average_precision_score(y,p)),3) if 0<y.sum()<len(y) else None))
des=pd.DataFrame(rows); des.to_csv(os.path.join(RES,'sealed_desulfation_drilldown.csv'),index=False)
print(des.to_string(index=False))

# ---- compact proposal table --------------------------------------------------
per_lab=pd.read_csv(os.path.join(RES,'sealed_per_label.csv'))
summ=json.load(open(os.path.join(RES,'sealed_eval_summary.json')))['brier']
he=json.load(open(os.path.join(ROOT,'visualizations','ml_upgrade','honest_eval_summary.json')))
bt=json.load(open(os.path.join(ROOT,'visualizations','ml_upgrade','baselines_tests_report.json')))
tab=per_lab[per_lab.label.isin(['desulfation','ester_hydrolysis','O_demethylation','aldehyde_oxidation','ring_cleavage','none'])]\
    [['label','n_pos','PR_AUC','lift','ROC_AUC','mean_conf_pos','mean_conf_neg']]
tab.columns=['Reaction class','n positive rows','PR-AUC','Lift over prevalence','ROC-AUC','Mean p (true +)','Mean p (true −)']
tab.to_csv(os.path.join(RES,'TABLE_proposal_sealed_per_class.csv'),index=False)
print(); print(tab.to_string(index=False))

# ================= MAIN FIGURE ==================================================
fig=plt.figure(figsize=(13.4,8.6))
gs=fig.add_gridspec(2,3,height_ratios=[1,1],hspace=0.52,wspace=0.30,
                    left=0.115,right=0.985,top=0.855,bottom=0.115)

# (A) rule recovery LCO vs LFO
ax=fig.add_subplot(gs[0,0])
hl=pd.read_csv(os.path.join(ROOT,'visualizations','ml_upgrade','honest_eval_per_label.csv'))
hl=hl.sort_values('pr_auc_lco',ascending=True).tail(8)
yp=np.arange(len(hl))
ax.barh(yp-0.2,hl.pr_auc_lco,0.4,color=BLUE,label='leave-compound-out')
ax.barh(yp+0.2,hl.pr_auc_lfo,0.4,color=ORANGE,label='leave-family-out')
ax.plot(hl.prevalence,yp,'k.',ms=6,label='prevalence floor')
ax.set_yticks(yp); ax.set_yticklabels([l.replace('_',' ') for l in hl.label],fontsize=8)
ax.set_xlabel('PR-AUC',fontsize=9); ax.set_title('A  Rules are recoverable from structure\n(20-compound panel, held-out splits)',fontsize=9.5,loc='left')
ax.legend(fontsize=7,loc='lower right',frameon=False); ax.tick_params(labelsize=8)

# (B) sealed-6 per compound Brier
ax=fig.add_subplot(gs[0,1])
pc=pd.read_csv(os.path.join(RES,'sealed_per_compound.csv'))
x=np.arange(len(pc))
ax.bar(x-0.2,pc.brier_rule,0.4,color=BLUE,label='rule model')
ax.bar(x+0.2,pc.brier_tanimoto,0.4,color=GREY,label='Tanimoto-kNN baseline')
ax.set_xticks(x); ax.set_xticklabels([c.replace('H0','H') for c in pc.compound_id],fontsize=8)
ax.set_ylabel('multi-label Brier (lower better)',fontsize=8.5)
ax.set_title('B  Six sealed compounds, never seen in training\n(H6 = DMSP negative control)',fontsize=9.5,loc='left')
ax.legend(fontsize=7,frameon=False); ax.tick_params(labelsize=8)
ax.annotate('rule model\nabstains correctly',xy=(5-0.2,pc.brier_rule.iloc[5]),xytext=(3.4,0.125),
            fontsize=7.5,color=GREEN,arrowprops=dict(arrowstyle='->',color=GREEN,lw=1.2))

# (C) desulfation transfers to unseen sulfate scaffolds
ax=fig.add_subplot(gs[0,2])
m1=cid=='H01'; m2=cid=='H02'
for k,(m,lab,col) in enumerate([(m1,'H1 luteolin 7-sulfate\n(flavone)',BLUE),(m2,'H2 hesperetin 7-O-sulfate\n(flavanone, no flavone core)',GREEN)]):
    y=Y[m,j]; p=P[m,j]
    ax.scatter(np.full((y==1).sum(),k-0.14)+np.random.default_rng(1).normal(0,0.035,(y==1).sum()),p[y==1],
               s=13,color=col,alpha=.75,label='true: reacts' if k==0 else None)
    ax.scatter(np.full((y==0).sum(),k+0.14)+np.random.default_rng(2).normal(0,0.035,(y==0).sum()),p[y==0],
               s=13,facecolors='none',edgecolors=col,alpha=.55,label='true: no reaction' if k==0 else None)
ax.set_xticks([0,1]); ax.set_xticklabels(['H1','H2'],fontsize=8)
ax.set_ylabel('predicted p(desulfation)',fontsize=8.5); ax.set_ylim(-0.05,1.05)
ax.set_title('C  Desulfation transfers to unseen sulfates\nrule ROC-AUC 0.76 / 0.77 vs similarity 0.50',fontsize=9.5,loc='left')
ax.legend(fontsize=7,frameon=False,loc='upper right'); ax.tick_params(labelsize=8)
ax.text(0.5,-0.215,'similarity baseline is constant per compound (ROC 0.50):\nit cannot say WHICH strain reacts',transform=ax.transAxes,ha='center',fontsize=7.2,color=GREEN)

# (D) DMSP abstention
ax=fig.add_subplot(gs[1,0])
h6=cid=='H06'; jn=LAB.index('y_none')
others=[k for k,l in enumerate(LAB) if l!='y_none']
ax.hist(P[h6][:,others].max(axis=1),bins=18,color=RED,alpha=.75,label='max p(any reaction)')
ax.hist(P[h6][:,jn],bins=18,color=GREEN,alpha=.6,label='p(no reaction)')
ax.set_xlabel('predicted probability',fontsize=8.5); ax.set_ylabel('strain × replicate rows',fontsize=8.5)
ax.set_title('D  Calibrated abstention on DMSP\n(carries no trained substructure)',fontsize=9.5,loc='left')
ax.legend(fontsize=7,frameon=False); ax.tick_params(labelsize=8)

# (E) marine vs public-trained
ax=fig.add_subplot(gs[1,1])
b=bt['baselines_multilabel_brier']
keys=[('rule_model_on_marine','marine-trained\nrule model',BLUE),('public_only_on_marine','trained on public /\nterrestrial rows only',RED)]
vals=[b[k] for k,_,_ in keys]
ax.bar([0,1],vals,0.5,color=[c for _,_,c in keys])
for i,v in enumerate(vals): ax.text(i,v+0.002,f'{v:.4f}',ha='center',fontsize=8)
ax.set_xticks([0,1]); ax.set_xticklabels([l for _,l,_ in keys],fontsize=7.5)
ax.set_ylabel('Brier on marine rows',fontsize=8.5); ax.set_ylim(0,max(vals)*1.28)
ax.set_title(f'E  Marine data advantage = {b["marine_data_advantage(public-rule)"]:.4f}\n(secondary success criterion)',fontsize=9.5,loc='left')
ax.tick_params(labelsize=8)

# (F) the honest negative: underpowered vs similarity
ax=fig.add_subplot(gs[1,2])
lo,hi=summ['CI95']; dm=summ['delta_tanimoto_minus_rule']
ax.axvline(0,color='k',lw=1,ls='--')
ax.errorbar([dm],[0],xerr=[[dm-lo],[hi-dm]],fmt='o',color=BLUE,capsize=5,ms=7,lw=1.8)
ax.set_yticks([]); ax.set_xlabel('Brier(similarity) − Brier(rule model)\n> 0 favours the rule model',fontsize=8.5)
ax.set_title(f'F  Underpowered at n = 6 sealed compounds\nP(rule better) = {summ["P(rule better)"]:.2f}, CI spans zero',fontsize=9.5,loc='left')
ax.tick_params(labelsize=8); ax.set_xlim(min(lo,-0.005)*1.25,max(hi,0.005)*1.25)

fig.suptitle('SEAGRAMMAR — in silico pilot on the final 20 + 6 compound panel',fontsize=14,y=0.965,x=0.045,ha='left')
fig.text(0.045,0.925,'Rule-learning framework (WP0) run end-to-end on the finalised panel: 20 training compounds, 6 sealed compounds, 42 strains, 3 replicates.',
         fontsize=9,ha='left',color='#444444')
fig.text(0.045,0.038,
 'SIMULATED DATA — labels are drawn by noisy-OR from the pre-registered rule priors, not measured. These panels demonstrate that the pipeline recovers a known rule set,\n'
 'calibrates its confidence and abstains correctly; they are an engineering positive control and carry NO biological claim. Panel F is the informative negative result.',
 fontsize=8,ha='left',color=RED)
fig.savefig(os.path.join(FIG,'FIG_proposal_pilot_20plus6.png'),dpi=300,facecolor='white')
fig.savefig(os.path.join(FIG,'FIG_proposal_pilot_20plus6.pdf'),facecolor='white')
print('\nfigure ->',os.path.join(FIG,'FIG_proposal_pilot_20plus6.png'))
