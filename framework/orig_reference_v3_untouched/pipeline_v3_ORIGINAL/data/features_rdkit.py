# features_rdkit.py
# =============================================================================
# Tính đặc trưng substructure THẬT từ SMILES bằng RDKit/SMARTS.
# Thay thế 19 cột feature hand-code trong pipeline cũ.
# Mỗi feature = có/không khớp một mẫu SMARTS -> tái lập 100% từ cấu trúc.
# =============================================================================
from rdkit import Chem

# (tên_feature, SMARTS). Mỗi feature ánh xạ tới một moiety hóa học có ý nghĩa
# enzymology (xem reaction_rules.py để biết feature nào kích hoạt phản ứng nào).
SMARTS = {
    'phenol':                'c[OX2H]',
    'catechol':              'c([OX2H])c[OX2H]',                       # ortho-diol
    'pyrogallol':            'c([OX2H])c([OX2H])c[OX2H]',              # 1,2,3-triol
    'gentisate_25_diol':     '[OX2H]c1ccc([OX2H])cc1C(=O)[OX2H1]',     # 2,5-diOH benzoic (para-diphenol, KHÔNG catechol)
    'guaiacyl':              'c([OX2H])c[OX2][CH3]',                   # 4-OH/3-OMe kề nhau
    'syringyl':              'c([OX2][CH3])c([OX2H])c[OX2][CH3]',      # OH kẹp giữa 2 OMe
    'aromatic_methoxy':      'c[OX2][CH3]',
    'monophenol_para':       '[OX2H]c1ccc([!#1])cc1',                  # phenol para-thế (hydroxyl hoá được)
    'carboxylic_acid':       '[CX3](=O)[OX2H1]',
    'benzoic_acid':          'c[CX3](=O)[OX2H1]',                      # COOH gắn trực tiếp nhân thơm
    'aromatic_aldehyde':     'c[CX3H1]=O',
    'vinyl_aromatic':        'c/[CH]=[CH]/[#6]',                       # cinnamoyl C=C
    'ester_bond':            '[CX3](=O)[OX2H0][#6]',                   # ester (acyl-O-C)
    'aryl_O_glycoside':      'c[OX2][CH]1O[CH][CH][CH][CH]1',          # đường O-glycoside thơm
    'aryl_sulfate_ester':    'cOS(=O)(=O)[OX2H,OX1-]',                 # ester sulfate thơm
    'flavone_core':          'O=c1cc(-c2ccccc2)oc2ccccc12',           # khung flavone
}

# 'reducible_double_bond' = đồng nghĩa hoá học với vinyl_aromatic (nối đôi cinnamoyl
# khử được); tính riêng để giữ tên cũ, nhưng sẽ tự loại nếu trùng phương sai.
SMARTS['reducible_double_bond'] = 'c/[CH]=[CH]/[CX3](=O)'

FEATURE_NAMES = list(SMARTS.keys())
_PATTERNS = {name: Chem.MolFromSmarts(smarts) for name, smarts in SMARTS.items()}
_BAD = [n for n, p in _PATTERNS.items() if p is None]
assert not _BAD, f"SMARTS lỗi: {_BAD}"


def compute_features(smiles):
    """Trả về dict {feature_name: 0/1} tính từ SMILES bằng RDKit."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"SMILES không hợp lệ: {smiles}")
    return {name: int(mol.HasSubstructMatch(_PATTERNS[name])) for name in FEATURE_NAMES}
