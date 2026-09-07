# features_rdkit.py  -- SEAGRAMMAR vocabulary v4 (2026-09-07)
# ---------------------------------------------------------------------------
# v4 change: the 17 substructure descriptors are INTEGER MATCH COUNTS, not
# presence/absence.  Presence is recoverable as count > 0, so v4 is a strict
# refinement of v3 -- nothing the v3 vocabulary could express is lost.
#
# Rationale (see results/PILOT_INTERPRETATION.md sect. 3 of the 17/08 run):
# under presence/absence, 15 of the 37 panel compounds fall into 7 groups with
# identical feature vectors -- including divanillin == vanillin and
# chicoric acid == caftaric acid == rosmarinic acid == chlorogenic acid.  The
# monomer->dimer and sequential-hydrolysis questions two of the six sealed
# compounds were chosen to ask were therefore invisible to the learner.
# With counts there are zero colliding vectors.
#
# Public API is unchanged: FEATURE_NAMES, compute_features(smiles) -> dict.
#   * RDKit present  -> len(GetSubstructMatches(patt, uniquify=True))
#   * RDKit absent   -> lookup table keyed by SMILES, selected by
#                       $SEAGRAMMAR_FEATURE_TABLE (default _features_counts.csv;
#                       set it to _features_offline.csv to reproduce the v3
#                       presence/absence control run).
# Earlier versions: features_rdkit.py.orig (v3 original, RDKit-only)
#                   features_rdkit.py.v3shim (v3 offline shim, 17/08 run)
# ---------------------------------------------------------------------------
import os, csv

SMARTS = {
    'phenol':'c[OX2H]', 'catechol':'c([OX2H])c[OX2H]', 'pyrogallol':'c([OX2H])c([OX2H])c[OX2H]',
    'gentisate_25_diol':'[OX2H]c1ccc([OX2H])cc1C(=O)[OX2H1]', 'guaiacyl':'c([OX2H])c[OX2][CH3]',
    'syringyl':'c([OX2][CH3])c([OX2H])c[OX2][CH3]', 'aromatic_methoxy':'c[OX2][CH3]',
    'monophenol_para':'[OX2H]c1ccc([!#1])cc1', 'carboxylic_acid':'[CX3](=O)[OX2H1]',
    'benzoic_acid':'c[CX3](=O)[OX2H1]', 'aromatic_aldehyde':'c[CX3H1]=O',
    'vinyl_aromatic':'c/[CH]=[CH]/[#6]', 'ester_bond':'[CX3](=O)[OX2H0][#6]',
    'aryl_O_glycoside':'c[OX2][CH]1O[CH][CH][CH][CH]1',
    'aryl_sulfate_ester':'cOS(=O)(=O)[OX2H,OX1-]',
    'flavone_core':'O=c1cc(-c2ccccc2)oc2ccccc12',
    'reducible_double_bond':'c/[CH]=[CH]/[CX3](=O)',
}
FEATURE_NAMES = list(SMARTS.keys())

# 'counts' (v4, default) or 'binary' (v3 control). Only affects the RDKit path;
# the offline path is chosen by which table $SEAGRAMMAR_FEATURE_TABLE points at.
# 'binary'      v3  : 17 presence bits
# 'counts'      v4a : the same 17 names, values = match counts
# 'counts_plus' v4b : 17 presence bits PLUS 17 additive n_<feature> count columns
VOCAB_MODE = os.environ.get('SEAGRAMMAR_VOCAB', 'counts').strip().lower()
TABLE_NAME = os.environ.get('SEAGRAMMAR_FEATURE_TABLE', '_features_counts.csv').strip()
if VOCAB_MODE == 'counts_plus':
    FEATURE_NAMES = FEATURE_NAMES + ['n_' + n for n in FEATURE_NAMES]

try:
    from rdkit import Chem
    _P = {n: Chem.MolFromSmarts(s) for n, s in SMARTS.items()}
    assert not [n for n, p in _P.items() if p is None]
    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False

_TABLE = {}
if not HAS_RDKIT:
    _here = os.path.dirname(os.path.abspath(__file__))
    for cand in (os.path.join(_here, 'inputs', TABLE_NAME),
                 os.path.abspath(os.path.join(_here, '..', '..', 'data', 'inputs', TABLE_NAME))):
        if os.path.exists(cand):
            with open(cand, encoding='utf-8') as fh:
                for r in csv.DictReader(fh):
                    _TABLE[r['smiles'].strip()] = {n: int(r[n]) for n in FEATURE_NAMES}
            _TABLE_PATH = cand
            break
    else:
        _TABLE_PATH = None


def compute_features(smiles):
    if HAS_RDKIT:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"invalid SMILES: {smiles}")
        base = list(SMARTS)
        if VOCAB_MODE == 'binary':
            return {n: int(mol.HasSubstructMatch(_P[n])) for n in base}
        cnt = {n: len(mol.GetSubstructMatches(_P[n], uniquify=True)) for n in base}
        if VOCAB_MODE == 'counts_plus':
            out = {n: int(cnt[n] > 0) for n in base}
            out.update({'n_' + n: cnt[n] for n in base})
            return out
        return cnt
    key = smiles.strip()
    if key not in _TABLE:
        raise ValueError(
            f"RDKit unavailable and SMILES not in the offline feature table {TABLE_NAME!r}: "
            f"{smiles!r}. Regenerate with scripts/make_count_features.py or install rdkit.")
    return dict(_TABLE[key])
