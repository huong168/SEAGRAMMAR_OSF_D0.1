#!/usr/bin/env python3
# msms_to_observations.py
# =============================================================================
# Chuyển bảng peak LC-MS/MS -> observations.csv (nhãn THẬT) cho pipeline.
# Dựng sẵn để CHẠY NGAY khi có dữ liệu. Chạy demo: python3 msms_to_observations.py --demo
# Chạy thật:  python3 msms_to_observations.py --samples data/inputs/lcms_samples.csv \
#                                             --features data/inputs/lcms_features.csv
#
# ---------------------------------------------------------------------------
# HỢP ĐỒNG ĐẦU VÀO (xuất từ MZmine/XCMS, định dạng long):
#
#   lcms_samples.csv   (1 dòng / mẫu LC-MS):
#     sample_id, compound_id, strain_id, replicate, timepoint, condition, polarity
#       timepoint : T0 | Tn
#       condition : live | control      (control = heat-killed/abiotic)
#       polarity  : neg | pos
#
#   lcms_features.csv  (1 dòng / peak phát hiện trong 1 mẫu):
#     sample_id, feature_id, mz, rt_min, area
#
#   (tuỳ chọn) lcms_msms.csv: sample_id, feature_id, frag_mz, frag_intensity
#       -> để annotate sản phẩm unknown sau bằng SIRIUS/CFM-ID (KHÔNG dùng auto-label).
#
# ---------------------------------------------------------------------------
# BA CHẾ ĐỘ PHÁT HIỆN (giải quyết đúng các ca khó):
#   (A) SUSPECT SCREENING — chắc chắn nhất, không cần networking:
#       với mỗi cơ chất tính trước m/z sản phẩm kỳ vọng = m/z cơ chất + Δm (ms2_reaction_map).
#       Khớp trong ppm + xuất hiện ở Tn(live) mà không ở control -> nhãn=1, confidence 2.
#   (B) SUBSTRATE DEPLETION — bắt cả chất bị bẻ/degrade mà Δm KHÔNG biết:
#       cơ chất giảm mạnh ở Tn(live) so với control -> "có chuyển hoá" (activity flag).
#       Nếu không khớp suspect nào -> đẩy review (reaction='unknown_transformation').
#   (C) REVIEW QUEUE — peak mới không khớp + phản ứng needs_ms2 (ring_cleavage):
#       KHÔNG auto-label; ghi vào review_queue.csv kèm bằng chứng để annotate thủ công.
# =============================================================================
import os, csv, argparse, sys
ROOT = os.path.dirname(os.path.abspath(__file__))
IN = os.path.join(ROOT, 'data', 'inputs')
sys.path.insert(0, os.path.join(ROOT, 'data'))
from rdkit import Chem
from rdkit.Chem import Descriptors

PROTON = 1.007276
DEPLETION_FRAC = 0.5      # cơ chất Tn < 50% control -> coi là bị tiêu thụ
ADDUCT = {'neg': -PROTON, 'pos': +PROTON}


def load_csv(path):
    with open(path, encoding='utf-8') as f:
        return list(csv.DictReader(f))


def substrate_ion_mz(smiles, polarity):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return Descriptors.ExactMolWt(m) + ADDUCT[polarity]


def load_reaction_map():
    rows = load_csv(os.path.join(IN, 'ms2_reaction_map.csv'))
    shifts, needs_ms2, split = [], [], []
    for r in rows:
        if r['type'] == 'shift':
            shifts.append((r['reaction'], float(r['delta_mz_monoisotopic']), float(r['ppm_tol'])))
        elif r['type'] == 'needs_ms2':
            needs_ms2.append(r['reaction'])
        elif r['type'] == 'split':
            split.append(r['reaction'])
    return shifts, needs_ms2, split


def within_ppm(obs, exp, ppm):
    return abs(obs - exp) / exp * 1e6 <= ppm


def run(samples_csv, features_csv, compounds_csv, out_dir):
    samples = load_csv(samples_csv)
    feats = load_csv(features_csv)
    compounds = {c['compound_id']: c for c in load_csv(compounds_csv)}
    shifts, needs_ms2, split_rx = load_reaction_map()

    # index feature theo sample
    feat_by_sample = {}
    for f in feats:
        feat_by_sample.setdefault(f['sample_id'], []).append(
            {'mz': float(f['mz']), 'rt': float(f['rt_min']), 'area': float(f['area']),
             'feature_id': f['feature_id']})

    # gom mẫu theo (compound, strain) -> timepoint/condition
    groups = {}
    for s in samples:
        key = (s['compound_id'], s['strain_id'])
        groups.setdefault(key, []).append(s)

    observations, review = [], []
    for (cid, sid), smps in groups.items():
        comp = compounds.get(cid)
        if comp is None:
            continue
        polarity = smps[0].get('polarity', 'neg')
        sub_mz = substrate_ion_mz(comp['smiles'], polarity)
        if sub_mz is None:
            continue

        tn_live = [x for s in smps if s['timepoint'] == 'Tn' and s['condition'] == 'live'
                   for x in feat_by_sample.get(s['sample_id'], [])]
        controls = [x for s in smps if s['condition'] == 'control'
                    for x in feat_by_sample.get(s['sample_id'], [])]
        if not tn_live:
            continue

        def area_near(feat_list, mz):
            return max([f['area'] for f in feat_list if within_ppm(f['mz'], mz, 10)] or [0.0])

        matched_feat_ids, any_reaction = set(), False

        # ---- (A) SUSPECT SCREENING ----
        for reaction, delta, ppm in shifts:
            exp = sub_mz + delta
            hit = [f for f in tn_live if within_ppm(f['mz'], exp, ppm)]
            ctrl_hit = [f for f in controls if within_ppm(f['mz'], exp, ppm)]
            if hit and not ctrl_hit:
                any_reaction = True
                for f in hit:
                    matched_feat_ids.add(f['feature_id'])
                prod_area = max(f['area'] for f in hit)
                sub_area_tn = area_near(tn_live, sub_mz)
                conv = prod_area / (prod_area + sub_area_tn) if (prod_area + sub_area_tn) else None
                observations.append({
                    'compound_id': cid, 'strain_id': sid, 'reaction': reaction, 'label': 1,
                    'weight': 5.0, 'source': 'lcms_suspect', 'ref': '',
                    'precursor_mz': round(exp, 5), 'delta_mz': round(delta, 5),
                    'msms_match_score': '', 'annotation_tool': 'suspect_screening',
                    'confidence_level': 2, 'conversion_pct': round(conv * 100, 1) if conv else ''})

        # ---- (B) SUBSTRATE DEPLETION ----
        sub_tn = area_near(tn_live, sub_mz)
        sub_ctrl = area_near(controls, sub_mz)
        depleted = sub_ctrl > 0 and sub_tn < DEPLETION_FRAC * sub_ctrl
        if depleted and not any_reaction:
            # có chuyển hoá nhưng KHÔNG khớp suspect -> review (chất bị bẻ/unknown)
            review.append({
                'compound_id': cid, 'strain_id': sid, 'reaction': 'unknown_transformation',
                'evidence': 'substrate_depletion', 'detail': f'sub Tn={sub_tn:.0f} vs control={sub_ctrl:.0f}',
                'confidence_level': 4, 'action': 'annotate sản phẩm (SIRIUS/CFM-ID) hoặc xác nhận ring_cleavage'})

        # ---- (C) PEAK MỚI KHÔNG KHỚP + needs_ms2 -> REVIEW ----
        for f in tn_live:
            if f['feature_id'] in matched_feat_ids:
                continue
            if within_ppm(f['mz'], sub_mz, 10):
                continue   # chính là cơ chất
            in_ctrl = any(within_ppm(f['mz'], f2['mz'], 10) for f2 in controls)
            if not in_ctrl:
                review.append({
                    'compound_id': cid, 'strain_id': sid, 'reaction': 'unmatched_product',
                    'evidence': f"new_peak mz={f['mz']:.4f} rt={f['rt']:.2f}", 'detail': 'không khớp Δm nào',
                    'confidence_level': 5, 'action': 'có thể là cleavage/unknown -> annotate MS2'})
        for rx in needs_ms2:
            if depleted:
                review.append({
                    'compound_id': cid, 'strain_id': sid, 'reaction': rx,
                    'evidence': 'needs_ms2 + depletion', 'detail': 'Δm không xác định cho mở vòng',
                    'confidence_level': 4, 'action': 'xác nhận bằng MS2/chuẩn đối chiếu'})

    os.makedirs(out_dir, exist_ok=True)
    obs_cols = ['compound_id', 'strain_id', 'reaction', 'label', 'weight', 'source', 'ref',
                'precursor_mz', 'delta_mz', 'msms_match_score', 'annotation_tool',
                'confidence_level', 'conversion_pct']
    with open(os.path.join(out_dir, 'observations_auto.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=obs_cols); w.writeheader(); w.writerows(observations)
    rev_cols = ['compound_id', 'strain_id', 'reaction', 'evidence', 'detail', 'confidence_level', 'action']
    with open(os.path.join(out_dir, 'review_queue.csv'), 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=rev_cols); w.writeheader(); w.writerows(review)

    print(f">> observations_auto.csv : {len(observations)} nhãn tự động (suspect screening)")
    print(f">> review_queue.csv      : {len(review)} mục cần review (depletion/unknown/needs_ms2)")
    print(f"   -> kiểm tra & dán dòng đạt vào data/inputs/observations.csv, rồi chạy build_dataset_v3.py")
    return observations, review


# --------------------------- DEMO TỔNG HỢP (chạy ngay) ---------------------------
def make_demo(tmp):
    """Sinh dữ liệu LC-MS giả lập để CHỨNG MINH pipeline chạy đúng:
       - C04 Ferulic acid + S08: O_demethylation (xuất hiện sản phẩm -14.016) -> auto-label
       - C12 Gallic acid + S15: bị tiêu thụ mạnh, KHÔNG có sản phẩm khớp -> review (cleavage/unknown)
    """
    os.makedirs(tmp, exist_ok=True)
    from rdkit.Chem import Descriptors as D
    fer = D.ExactMolWt(Chem.MolFromSmiles('COc1cc(/C=C/C(=O)O)ccc1O')) - PROTON  # [M-H]-
    gal = D.ExactMolWt(Chem.MolFromSmiles('O=C(O)c1cc(O)c(O)c(O)c1')) - PROTON
    samples = [
        ('s1', 'C04', 'S08', 1, 'T0', 'live', 'neg'),
        ('s2', 'C04', 'S08', 1, 'Tn', 'live', 'neg'),
        ('s3', 'C04', 'S08', 1, 'Tn', 'control', 'neg'),
        ('s4', 'C12', 'S15', 1, 'Tn', 'live', 'neg'),
        ('s5', 'C12', 'S15', 1, 'Tn', 'control', 'neg'),
    ]
    feats = [
        ('s1', 'f1', fer, 5.2, 1e6),
        ('s2', 'f2', fer, 5.2, 4e5),               # cơ chất còn lại
        ('s2', 'f3', fer - 14.015650, 4.8, 6e5),   # SẢN PHẨM demethylation
        ('s3', 'f4', fer, 5.2, 1e6),               # control: chỉ cơ chất
        ('s4', 'f5', gal, 3.1, 1e5),               # Tn live: cơ chất gần như biến mất
        ('s4', 'f6', 191.0193, 2.4, 5e5),          # peak lạ (vd muconate-ish) không khớp Δm
        ('s5', 'f7', gal, 3.1, 1e6),               # control: cơ chất đầy đủ
    ]
    with open(os.path.join(tmp, 'lcms_samples.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['sample_id', 'compound_id', 'strain_id', 'replicate', 'timepoint', 'condition', 'polarity']); w.writerows(samples)
    with open(os.path.join(tmp, 'lcms_features.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['sample_id', 'feature_id', 'mz', 'rt_min', 'area']); w.writerows(feats)
    return os.path.join(tmp, 'lcms_samples.csv'), os.path.join(tmp, 'lcms_features.csv')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--samples'); ap.add_argument('--features')
    ap.add_argument('--compounds', default=os.path.join(IN, 'compounds.csv'))
    ap.add_argument('--out', default=os.path.join(ROOT, 'data', 'lcms_out'))
    ap.add_argument('--demo', action='store_true')
    a = ap.parse_args()
    if a.demo or not (a.samples and a.features):
        print(">> CHẾ ĐỘ DEMO (dữ liệu giả lập chứng minh pipeline)\n")
        tmp = os.path.join(ROOT, 'data', 'lcms_demo')
        s, fe = make_demo(tmp)
        run(s, fe, a.compounds, os.path.join(ROOT, 'data', 'lcms_demo_out'))
    else:
        run(a.samples, a.features, a.compounds, a.out)
