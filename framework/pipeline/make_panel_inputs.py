#!/usr/bin/env python3
"""Build the 20+6 proposal panel as data/inputs/compounds.csv, plus an offline
RDKit-free feature table.  NEW FILE - no original script is modified."""
import csv, os, json
ROOT=os.path.dirname(os.path.abspath(__file__)); PILOT=os.path.dirname(ROOT)
SRC=os.path.join(ROOT,'data','inputs','compounds.csv')
FEAT_SRC=os.path.join(PILOT,'data','inputs','_features_from_rdkit_ORIGINAL.csv')
OUT_DIR=os.path.join(PILOT,'data','inputs'); os.makedirs(OUT_DIR,exist_ok=True)

FEATURE_NAMES=['phenol','catechol','pyrogallol','gentisate_25_diol','guaiacyl','syringyl',
 'aromatic_methoxy','monophenol_para','carboxylic_acid','benzoic_acid','aromatic_aldehyde',
 'vinyl_aromatic','ester_bond','aryl_O_glycoside','aryl_sulfate_ester','flavone_core',
 'reducible_double_bond']

orig={r['compound_id']:r for r in csv.DictReader(open(SRC,encoding='utf-8'))}
rdk={r['compound_id']:r for r in csv.DictReader(open(FEAT_SRC,encoding='utf-8'))}

# reducible_double_bond = cinnamoyl C=C conjugated to a carbonyl (dropped later as
# redundant-by-design, but FEATURE_NAMES still expects it)
RDB={'C01','C02','C03','C04','C05','C06','C19','T01','T02','H03','H05'}

# ---- the 6 sealed compounds (features hand-derived, each justified) ----------
SEALED=[
 dict(compound_id='H01',compound_name='Luteolin 7-sulfate',pillar='Sulfated flavone (Zostera)',
      class_id='5',scaffold='flavone_sulfate',
      smiles='O=c1cc(-c2ccc(O)c(O)c2)oc2cc(OS(=O)(=O)O)cc(O)c12',
      provenance='PubChem:CID_luteolin-7-sulfate',origin='sealed_holdout',
      feats=dict(phenol=1,catechol=1,monophenol_para=1,aryl_sulfate_ester=1,flavone_core=1),
      basis='luteolin (C15) B-ring catechol retained + 7-O-sulfate as in apigenin 7-sulfate (C20)'),
 dict(compound_id='H02',compound_name='rac-Hesperetin 7-O-sulfate',pillar='Sulfated flavanone (analogue)',
      class_id='8',scaffold='flavanone_sulfate',
      smiles='COc1ccc(cc1O)C1CC(=O)c2c(O)cc(OS(=O)(=O)O)cc2O1',
      provenance='CAS:675139-61-4',origin='sealed_holdout',
      feats=dict(phenol=1,guaiacyl=1,aromatic_methoxy=1,aryl_sulfate_ester=1),
      basis='flavanone -> flavone_core=0 (cf. naringenin T06=0); 3-OH/4-OMe pair -> guaiacyl=1 (cf. diosmetin C17); '
            '7-OH replaced by sulfate so no para-substituted free phenol remains -> monophenol_para=0'),
 dict(compound_id='H03',compound_name='Sinapic acid',pillar='Hydroxycinnamic (syringyl)',
      class_id='0',scaffold='cinnamic',smiles='COc1cc(/C=C/C(=O)O)cc(OC)c1O',
      provenance='PubChem:637775',origin='sealed_holdout',
      feats=dict(phenol=1,guaiacyl=1,syringyl=1,aromatic_methoxy=1,monophenol_para=1,
                 carboxylic_acid=1,vinyl_aromatic=1),
      basis='identical to the RDKit vector already computed for T01 in the original dataset'),
 dict(compound_id='H04',compound_name='Dehydrodivanillin (divanillin)',pillar='5,5-biaryl dimer of vanillin',
      class_id='3',scaffold='benzaldehyde_dimer',
      smiles='COc1cc(C=O)cc(-c2cc(C=O)cc(OC)c2O)c1O',
      provenance='CAS:2092-49-1',origin='sealed_holdout',
      feats=dict(phenol=1,guaiacyl=1,aromatic_methoxy=1,monophenol_para=1,aromatic_aldehyde=1),
      basis='each half is a vanillin unit -> vector IDENTICAL to vanillin (C13). The 17-feature '
            'vocabulary has no dimer/biaryl descriptor, so the monomer->dimer question is INVISIBLE here'),
 dict(compound_id='H05',compound_name='Chicoric acid',pillar='Dicaffeoyl tartrate ester',
      class_id='1',scaffold='depside_ester',
      smiles='O=C(O)C(OC(=O)/C=C/c1ccc(O)c(O)c1)C(OC(=O)/C=C/c1ccc(O)c(O)c1)C(=O)O',
      provenance='CAS:70831-56-0',origin='sealed_holdout',
      feats=dict(phenol=1,catechol=1,monophenol_para=1,carboxylic_acid=1,vinyl_aromatic=1,ester_bond=1),
      basis='vector IDENTICAL to caftaric acid (C05). Mono- vs di-ester is not representable in a '
            'presence/absence vocabulary, so the "two sequential hydrolyses" question is INVISIBLE here'),
 dict(compound_id='H06',compound_name='DMSP (dimethylsulfoniopropionate)',pillar='Non-aromatic osmolyte (negative control)',
      class_id='9',scaffold='sulfonium',smiles='C[S+](C)CCC(=O)[O-]',
      provenance='CAS:4337-33-1',origin='sealed_holdout',
      feats={},
      basis='no aromatic ring; carboxylate written as the zwitterion so carboxylic_acid SMARTS '
            '([CX3](=O)[OX2H1]) does not match -> ALL 17 features zero. Correct prediction = abstain'),
]

# ---- assemble compounds.csv -------------------------------------------------
rows=[]; feat_rows=[]; notes=[]
def add(cid,name,pillar,class_id,scaffold,smiles,prov,origin,role,feats,basis):
    rows.append(dict(compound_id=cid,compound_name=name,pillar=pillar,class_id=class_id,
                     scaffold=scaffold,smiles=smiles,provenance=prov,origin=origin,panel_role=role))
    fr={'compound_id':cid,'smiles':smiles}
    for f in FEATURE_NAMES: fr[f]=int(feats.get(f,0))
    fr['reducible_double_bond']=int(cid in RDB)
    feat_rows.append(fr); notes.append({'compound_id':cid,'compound_name':name,'feature_basis':basis})

# 20 training (C12 gallic acid -> arbutin, per the panel decision of 17/08)
ARBUTIN=dict(compound_id='C12',compound_name='Arbutin (hydroquinone b-D-glucoside)',
  pillar='Aryl glucoside (2nd deglycosylation scaffold)',class_id='2',scaffold='aryl_glucoside',
  smiles='OC[C@H]1O[C@@H](Oc2ccc(O)cc2)[C@H](O)[C@@H](O)[C@@H]1O',provenance='CAS:497-76-7',
  feats=dict(phenol=1,monophenol_para=1,aryl_O_glycoside=1),
  basis='para-diol so catechol=0; no COOH so gentisate_25_diol SMARTS cannot match; '
        'aryl_O_glycoside=1 as for luteolin 7-O-glucoside (C18)')
for cid in [f'C{i:02d}' for i in range(1,21)]:
    if cid=='C12':
        a=ARBUTIN
        add(a['compound_id'],a['compound_name'],a['pillar'],a['class_id'],a['scaffold'],
            a['smiles'],a['provenance'],'seagrass_marine','training',a['feats'],a['basis'])
        continue
    o=orig[cid]; f={k:int(rdk[cid][k]) for k in FEATURE_NAMES if k in rdk[cid]}
    add(cid,o['compound_name'],o['pillar'],o['class_id'],o['scaffold'],o['smiles'],
        o['provenance'],'seagrass_marine','training',f,'RDKit-computed in the original v3 dataset (authoritative)')
# 6 sealed
for s in SEALED:
    add(s['compound_id'],s['compound_name'],s['pillar'],s['class_id'],s['scaffold'],
        s['smiles'],s['provenance'],s['origin'],'sealed',s['feats'],s['basis'])
# terrestrial augmentation, needed for the public_only baseline (T01 dropped: now H03)
for cid in [f'T{i:02d}' for i in range(2,13)]:
    o=orig[cid]; f={k:int(rdk[cid][k]) for k in FEATURE_NAMES if k in rdk[cid]}
    add(cid,o['compound_name'],o['pillar'],o['class_id'],o['scaffold'],o['smiles'],
        o['provenance'],'terrestrial_plant','public_augmentation',f,'RDKit-computed in the original v3 dataset')

with open(os.path.join(OUT_DIR,'compounds.csv'),'w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=['compound_id','compound_name','pillar','class_id','scaffold','smiles','provenance','origin','panel_role']); w.writeheader(); w.writerows(rows)
with open(os.path.join(OUT_DIR,'_features_offline.csv'),'w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=['compound_id','smiles']+FEATURE_NAMES); w.writeheader(); w.writerows(feat_rows)
json.dump(notes,open(os.path.join(OUT_DIR,'_feature_provenance.json'),'w'),indent=2)
print(f'compounds.csv: {len(rows)} rows  '
      f"({sum(r['panel_role']=='training' for r in rows)} training, "
      f"{sum(r['panel_role']=='sealed' for r in rows)} sealed, "
      f"{sum(r['panel_role']=='public_augmentation' for r in rows)} public augmentation)")
