#!/usr/bin/env python3
# build_dataset_v3.py
# =============================================================================
# SEAGRASSGRAMMAR — Generator v3 (CSV-DRIVEN + GHI ĐÈ DỮ LIỆU THỰC NGHIỆM)
# -----------------------------------------------------------------------------
# Mục tiêu: BỔ SUNG DỮ LIỆU SAU NÀY mà KHÔNG cần sửa code.
#   • Toàn bộ input nằm trong data/inputs/*.csv — chỉ cần thêm dòng rồi chạy lại.
#       - compounds.csv       : thêm hợp chất mới (cần SMILES; feature do RDKit tính)
#       - strains.csv         : thêm chủng mới (+ năng lực enzyme)
#       - reaction_rules.csv  : thêm/sửa luật (σ,ρ)->prior
#       - observations.csv    : DỮ LIỆU THỰC NGHIỆM THẬT -> GHI ĐÈ nhãn mô phỏng
#   • Nhãn mô phỏng sinh từ prior (noisy-OR). Nếu observations.csv có quan sát
#     thật cho (compound, strain, reaction) thì giá trị thật THAY THẾ nhãn mô phỏng,
#     đánh dấu evidence='experimental' + weight cao -> mô hình ưu tiên dữ liệu thật.
#   • Vẫn giữ mọi tính chất chống rò rỉ của v2 (cv_group, replicate_id, 1 cơ chế weight,
#     feature RDKit không hằng số, source chỉ là metadata).
#
# Chạy:  python3 build_dataset_v3.py
# =============================================================================
import os, csv, json, random, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
def _resolve_data_dir(root):
    d = os.environ.get('SEAGRAMMAR_DATA')
    if d:
        return d
    for cand in ('data', 'data1'):
        if os.path.isdir(os.path.join(root, cand, 'inputs')) or \
           os.path.exists(os.path.join(root, cand, 'ml_training_dataset_v3.csv')):
            return os.path.join(root, cand)
    return os.path.join(root, 'data')
OUT = _resolve_data_dir(ROOT)   # FIX: auto-detect data/ vs data1/ (or $SEAGRAMMAR_DATA)
IN = os.path.join(OUT, 'inputs')
sys.path.insert(0, OUT)
from features_rdkit import compute_features, FEATURE_NAMES   # tái dùng module RDKit

SEED = 42
N_REPLICATES = 3
# Weight theo TIER (habitat, role) — 1 cơ chế duy nhất. Dữ liệu cạn (off-target
# habitat) bị down-weight: dùng để AUGMENT, không lấn át tín hiệu biển.
def tier_weight(habitat, role):
    if habitat == 'marine':
        return 1.0 if role == 'primary' else 0.7
    return 0.5   # terrestrial/reference: augmentation
EXPERIMENTAL_WEIGHT_DEFAULT = 5.0   # dữ liệu thật mạnh hơn mô phỏng
# weight theo độ tin cậy Schymanski (1=chuẩn đối chiếu ... 5=chỉ có m/z)
CONFIDENCE_WEIGHT = {'1': 8.0, '2': 6.0, '3': 4.0, '4': 2.0, '5': 1.0}
REDUNDANT_BY_DESIGN = ['reducible_double_bond', 'n_reducible_double_bond']   # bản sao của vinyl_aromatic (+ v4b count twin)


def obs_weight(o):
    """Ưu tiên weight ghi tay; nếu không có thì suy từ confidence_level (LC-MS)."""
    w = o.get('weight')
    if w not in (None, ''):
        return float(w)
    cl = str(o.get('confidence_level', '')).strip()
    return CONFIDENCE_WEIGHT.get(cl, EXPERIMENTAL_WEIGHT_DEFAULT)


# ----------------------------- đọc input CSV -----------------------------
def read_csv(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def load_inputs():
    compounds = read_csv(os.path.join(IN, 'compounds.csv'))
    strains = read_csv(os.path.join(IN, 'strains.csv'))
    rules_raw = read_csv(os.path.join(IN, 'reaction_rules.csv'))
    rules = [(r['reaction'], r['substructure'], r['enzyme'], float(r['prior_prob']),
              r['EC'], [x for x in r['refs'].split(';') if x]) for r in rules_raw]
    reactions = sorted({r[0] for r in rules})
    obs_path = os.path.join(IN, 'observations.csv')
    observations = read_csv(obs_path) if os.path.exists(obs_path) else []
    # bỏ dòng mẫu/placeholder (ref chứa 'XXXX' hoặc reaction rỗng)
    observations = [o for o in observations if o.get('reaction') and 'XXXX' not in o.get('ref', '')]
    return compounds, strains, rules, reactions, observations


def activation_probs(feats, caps, rules, reactions):
    probs = {r: 0.0 for r in reactions}
    for reaction, sub, cap, p, _ec, _refs in rules:
        # --- PILOT PATCH 2026-09-07 (vocabulary v4): the rule prior is keyed on the
        # PRESENCE of substructure sigma. With the v4 count vocabulary a feature value
        # of 2 must still fire the rule, so `== 1` becomes `> 0`. This is a no-op for the
        # v3 binary table, so the control run is unaffected and the simulated labels are
        # bit-identical between the v3 and v4 runs (same seed, same call order).
        if feats.get(sub, 0) > 0 and cap in caps:
            probs[reaction] = 1.0 - (1.0 - probs[reaction]) * (1.0 - p)
    return probs


def build():
    rng = random.Random(SEED)
    compounds, strains, rules, reactions, observations = load_inputs()
    label_cols = reactions + ['none']

    # feature thật cho mỗi hợp chất (RDKit từ SMILES)
    comp_feats, comp_info = {}, {}
    for c in compounds:
        comp_feats[c['compound_id']] = compute_features(c['smiles'])
        comp_info[c['compound_id']] = c

    # loại feature suy biến: hằng số + trùng-thiết-kế
    mat = {f: [comp_feats[c['compound_id']][f] for c in compounds] for f in FEATURE_NAMES}
    constant = [f for f in FEATURE_NAMES if len(set(mat[f])) == 1]
    use_features = [f for f in FEATURE_NAMES if f not in constant and f not in REDUNDANT_BY_DESIGN]

    # index observations: (compound, strain, reaction) -> dict
    obs_idx = {}
    for o in observations:
        key = (o['compound_id'], o['strain_id'], o['reaction'])
        obs_idx[key] = o
    n_obs_applied = 0

    records = []
    for s in strains:
        caps = set(x for x in s['capabilities'].split(';') if x)
        role = s['role']; habitat = s.get('habitat', 'marine')
        w_base = tier_weight(habitat, role)
        source = ('project_primary' if role == 'primary' else 'project_expansion') \
            if habitat == 'marine' else 'terrestrial_reference'
        for c in compounds:
            cid = c['compound_id']; feats = comp_feats[cid]; origin = c.get('origin', 'seagrass_marine')
            probs = activation_probs(feats, caps, rules, reactions)
            for rep in range(1, N_REPLICATES + 1):
                labels = {r: (1 if rng.random() < probs[r] else 0) for r in reactions}
                evidence = 'simulated'; rec_w = w_base; rec_src = source
                wl = {r: w_base for r in reactions}   # FIX: trọng số THEO TỪNG NHÃN
                # --- GHI ĐÈ bằng quan sát thực nghiệm nếu có ---
                applied = False
                for r in reactions:
                    key = (cid, s['strain_id'], r)
                    if key in obs_idx:
                        o = obs_idx[key]
                        labels[r] = int(o['label']); wl[r] = obs_weight(o); applied = True
                if applied:
                    # FIX: chỉ up-weight NHÃN có bằng chứng thực nghiệm (per-label);
                    # KHÔNG up-weight cả dòng (tránh nâng khống nhãn vẫn còn simulated).
                    ws = [wl[r] for r in reactions if (cid, s['strain_id'], r) in obs_idx]
                    rec_w = max(ws) if ws else EXPERIMENTAL_WEIGHT_DEFAULT
                    evidence = 'experimental'; rec_src = 'experimental'; n_obs_applied += 1
                labels['none'] = 1 if sum(labels[r] for r in reactions) == 0 else 0
                wl['none'] = w_base   # 'none' theo tier cơ bản

                rec = {'source': rec_src, 'evidence': evidence, 'compound_id': cid,
                       'compound_name': c['compound_name'], 'smiles': c['smiles'],
                       'provenance': c['provenance'], 'origin': origin, 'scaffold': c['scaffold'],
                       'pillar': c['pillar'], 'class_id': c['class_id'],
                       'strain_id': s['strain_id'], 'strain_name': s['strain_name'],
                       'strain_genus': s['genus'], 'strain_phylum': s['phylum'], 'habitat': habitat,
                       'role': role, 'weight': rec_w, 'replicate_id': rep, 'cv_group': cid}
                for f in use_features:
                    rec[f] = feats[f]
                for r in label_cols:
                    rec['y_' + r] = labels[r]
                for r in label_cols:
                    rec['w_' + r] = wl[r]        # FIX: cột trọng số per-label
                records.append(rec)

    meta_cols = ['source', 'evidence', 'compound_id', 'compound_name', 'smiles', 'provenance',
                 'origin', 'scaffold', 'pillar', 'class_id', 'strain_id', 'strain_name', 'strain_genus',
                 'strain_phylum', 'habitat', 'role', 'weight', 'replicate_id', 'cv_group']
    header = meta_cols + use_features + ['y_' + r for r in label_cols] + ['w_' + r for r in label_cols]
    out_csv = os.path.join(OUT, 'ml_training_dataset_v3.csv')
    with open(out_csv, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(header)
        for r in records:
            w.writerow([r[c] for c in meta_cols] + [r[f] for f in use_features] +
                       [r['y_' + l] for l in label_cols] + [r['w_' + l] for l in label_cols])

    card = {
        'name': 'SEAGRASSGRAMMAR ML dataset v3 (CSV-driven)',
        'n_rows': len(records), 'n_compounds': len(compounds), 'n_strains': len(strains),
        'n_reactions': len(reactions), 'n_replicates': N_REPLICATES,
        'features_used': use_features, 'features_dropped_constant': constant,
        'features_dropped_redundant_by_design': REDUNDANT_BY_DESIGN,
        'labels': label_cols, 'n_experimental_rows_overridden': n_obs_applied,
        'inputs': 'data/inputs/{compounds,strains,reaction_rules,observations}.csv',
        'splitting': 'GroupKFold theo cv_group (=compound_id).',
        'weighting': 'weight (per-row, =max) + w_<label> (per-label): primary=1.0, expansion=0.7, experimental=>=5.0.',
        'per_label_weight_cols': ['w_' + r for r in label_cols],
        'NOT_features': ['source', 'evidence', 'strain_id', 'compound_id', 'cv_group',
                         'provenance', 'smiles', 'weight', 'replicate_id'],
    }
    json.dump(card, open(os.path.join(OUT, 'DATA_CARD_v3.json'), 'w', encoding='utf-8'),
              ensure_ascii=False, indent=2)

    print(f">> {out_csv}")
    print(f"   {len(records)} dòng | {len(compounds)} hợp chất × {len(strains)} chủng × {N_REPLICATES} replicate")
    print(f"   feature dùng: {len(use_features)} (bỏ hằng số: {constant or 'không'})")
    print(f"   nhãn: {len(label_cols)} | quan sát thực nghiệm áp dụng: {n_obs_applied} dòng")
    print(f"   evidence breakdown: " + str({e: sum(1 for r in records if r['evidence'] == e) for e in set(r['evidence'] for r in records)}))


if __name__ == '__main__':
    build()
