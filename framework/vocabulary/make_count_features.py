#!/usr/bin/env python3
"""make_count_features.py  -- SEAGRAMMAR vocabulary v4 (2026-09-07)

Computes the substructure vocabulary as INTEGER MATCH COUNTS instead of
presence/absence, using RDKit directly from SMILES.

Why: with a presence/absence vocabulary, divanillin (H04) has the same feature
vector as vanillin (C13), and chicoric acid (H05) the same as caftaric acid
(C05).  The monomer -> dimer and two-sequential-hydrolyses questions those two
sealed compounds were chosen to ask are therefore invisible to the learner.
Counting matches makes them representable.  Presence is still recoverable as
count > 0, so no information is lost relative to v3.

Run with a Python that HAS rdkit.  The pipeline itself is run WITHOUT rdkit so
that the only thing that changes between the v3 control run and the v4
treatment run is the content of the feature table.
"""
import csv, json, os, sys
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

SMARTS = {
    'phenol': 'c[OX2H]',
    'catechol': 'c([OX2H])c[OX2H]',
    'pyrogallol': 'c([OX2H])c([OX2H])c[OX2H]',
    'gentisate_25_diol': '[OX2H]c1ccc([OX2H])cc1C(=O)[OX2H1]',
    'guaiacyl': 'c([OX2H])c[OX2][CH3]',
    'syringyl': 'c([OX2][CH3])c([OX2H])c[OX2][CH3]',
    'aromatic_methoxy': 'c[OX2][CH3]',
    'monophenol_para': '[OX2H]c1ccc([!#1])cc1',
    'carboxylic_acid': '[CX3](=O)[OX2H1]',
    'benzoic_acid': 'c[CX3](=O)[OX2H1]',
    'aromatic_aldehyde': 'c[CX3H1]=O',
    'vinyl_aromatic': 'c/[CH]=[CH]/[#6]',
    'ester_bond': '[CX3](=O)[OX2H0][#6]',
    'aryl_O_glycoside': 'c[OX2][CH]1O[CH][CH][CH][CH]1',
    'aryl_sulfate_ester': 'cOS(=O)(=O)[OX2H,OX1-]',
    'flavone_core': 'O=c1cc(-c2ccccc2)oc2ccccc12',
    'reducible_double_bond': 'c/[CH]=[CH]/[CX3](=O)',
}
FEATURE_NAMES = list(SMARTS.keys())

HERE = os.path.dirname(os.path.abspath(__file__))
PILOT = os.path.dirname(HERE) if os.path.basename(HERE) == 'scripts' else HERE
IN = os.path.join(PILOT, 'data', 'inputs')


def main():
    indir = sys.argv[1] if len(sys.argv) > 1 else IN
    patts = {}
    for name, sma in SMARTS.items():
        p = Chem.MolFromSmarts(sma)
        if p is None:
            raise SystemExit(f'bad SMARTS for {name}: {sma}')
        patts[name] = p

    comps = list(csv.DictReader(open(os.path.join(indir, 'compounds.csv'), encoding='utf-8')))

    old = {}
    old_path = os.path.join(indir, '_features_offline.csv')
    if os.path.exists(old_path):
        for r in csv.DictReader(open(old_path, encoding='utf-8')):
            old[r['compound_id']] = {n: int(r[n]) for n in FEATURE_NAMES if n in r}

    count_rows, bin_rows, prov, diffs = [], [], [], []
    for c in comps:
        cid, smi = c['compound_id'], c['smiles']
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise SystemExit(f'invalid SMILES for {cid}: {smi}')
        counts = {n: len(mol.GetSubstructMatches(patts[n], uniquify=True)) for n in FEATURE_NAMES}
        binar = {n: int(counts[n] > 0) for n in FEATURE_NAMES}

        row_c = {'compound_id': cid, 'smiles': smi}; row_c.update(counts)
        row_b = {'compound_id': cid, 'smiles': smi}; row_b.update(binar)
        count_rows.append(row_c); bin_rows.append(row_b)

        multi = {n: v for n, v in counts.items() if v > 1}
        prov.append({'compound_id': cid, 'compound_name': c['compound_name'],
                     'panel_role': c.get('panel_role', ''), 'smiles': smi,
                     'counts': counts, 'features_with_multiplicity': multi,
                     'method': 'RDKit GetSubstructMatches(uniquify=True), SMARTS unchanged from v3'})

        if cid in old:
            for n in FEATURE_NAMES:
                if n in old[cid] and old[cid][n] != binar[n]:
                    diffs.append({'compound_id': cid, 'compound_name': c['compound_name'],
                                  'feature': n, 'v3_hand_derived': old[cid][n],
                                  'v4_rdkit_presence': binar[n], 'v4_rdkit_count': counts[n]})

    fields = ['compound_id', 'smiles'] + FEATURE_NAMES
    for path, rows in ((os.path.join(indir, '_features_counts.csv'), count_rows),
                       (os.path.join(indir, '_features_binary_rdkit.csv'), bin_rows)):
        with open(path, 'w', newline='', encoding='utf-8') as fh:
            w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(rows)

    # v4b: presence bits AND additive n_<feature> match counts in one table
    plus_fields = ['compound_id', 'smiles'] + FEATURE_NAMES + ['n_' + n for n in FEATURE_NAMES]
    plus_rows = []
    for b, c in zip(bin_rows, count_rows):
        r = {'compound_id': b['compound_id'], 'smiles': b['smiles']}
        r.update({n: b[n] for n in FEATURE_NAMES})
        r.update({'n_' + n: c[n] for n in FEATURE_NAMES})
        plus_rows.append(r)
    with open(os.path.join(indir, '_features_counts_plus.csv'), 'w', newline='', encoding='utf-8') as fh:
        w = csv.DictWriter(fh, fieldnames=plus_fields); w.writeheader(); w.writerows(plus_rows)

    json.dump({'vocabulary_version': 'v4-counts-2026-09-07',
               'n_compounds': len(comps),
               'smarts': SMARTS,
               'per_compound': prov,
               'qc_presence_disagreements_vs_v3_hand_derived': diffs},
              open(os.path.join(indir, '_features_v4_provenance.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)

    print(f'{len(comps)} compounds -> _features_counts.csv, _features_binary_rdkit.csv')
    print(f'presence disagreements vs the v3 hand-derived table: {len(diffs)}')
    for d in diffs:
        print("  {compound_id:5s} {compound_name:34.34s} {feature:20s} v3={v3_hand_derived} "
              "rdkit={v4_rdkit_presence} (count={v4_rdkit_count})".format(**d))
    print('')
    print('compounds whose vector now carries multiplicity (count > 1):')
    for p in prov:
        if p['features_with_multiplicity']:
            print(f"  {p['compound_id']:5s} {p['compound_name']:34.34s} {p['features_with_multiplicity']}")


if __name__ == '__main__':
    main()
