#!/usr/bin/env python3
# ml_upgrade/make_pipeline_diagram.py
# Renders the SEAGRAMMAR pipeline flow (data -> model -> calibration -> tests -> figures)
# to a single PNG (+ SVG) for the spec / proposal.
import os
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'visualizations', 'ml_upgrade'); os.makedirs(OUT, exist_ok=True)

STAGES = [
    ('1 · DATA', '#2c7fb8', '#e8f1f8'),
    ('2 · MODEL', '#6b6b6b', '#ededed'),
    ('3 · CALIBRATION', '#2ca02c', '#e7f4e7'),
    ('4 · TESTS', '#ff7f0e', '#fdeedd'),
    ('5 · FIGURES', '#7b3fa0', '#efe6f5'),
]
# file boxes per stage column: (title, subtitle)
BOXES = {
    0: [('data/inputs/*.csv', 'compounds · strains · rules\nobservations · ms2_map'),
        ('features_rdkit.py', 'SMILES → 16 σ (SMARTS)'),
        ('reaction_rules', 'noisy-OR expert prior\n(EC + PMID)'),
        ('msms_to_observations.py', 'LC-MS/MS → observations\n(REAL labels)'),
        ('build_dataset_v3.py', '→ ml_training_dataset_v3.csv\n(sim labels; real overrides)')],
    1: [('_core.py', 'engine auto-select:\nRF+Morgan  /  numpy fallback'),
        ('train_model_v2.py  (old)', 'RF binary-relevance\nleave-compound-out\n→ F1 / PR-AUC / ROC-AUC')],
    2: [('calibrate_eval.py', 'isotonic PAV (group-safe)\n+ no-reaction first-class\n→ calibrated p, Brier\n+ reliability curves')],
    3: [('baselines_tests.py', 'baselines:\nTanimoto-kNN · freq · public-only\n\nA compositionality (JSD)\nB reusability (w-Jaccard)\nC1/C2 productivity\n(Brier + paired bootstrap)\n→ report.json')],
    4: [('make_figures.py', 'reads CSV/JSON →\n5 PNG\n(calibration, reliability,\nbaselines, reusability,\nproductivity)')],
}

fig, ax = plt.subplots(figsize=(17, 9))
ax.set_xlim(0, 17); ax.set_ylim(0, 10); ax.axis('off')
colw, x0, gap = 3.1, 0.25, 0.30
top = 9.0

centers = []
for i, (name, edge, fill) in enumerate(STAGES):
    x = x0 + i * (colw + gap); cx = x + colw / 2; centers.append((x, cx))
    # header band
    ax.add_patch(FancyBboxPatch((x, top), colw, 0.6, boxstyle='round,pad=0.02,rounding_size=0.08',
                                fc=edge, ec=edge))
    ax.text(cx, top + 0.3, name, color='white', ha='center', va='center', fontsize=12, fontweight='bold')
    # file boxes
    y = top - 0.5
    for title, sub in BOXES[i]:
        nlines = sub.count('\n') + 1
        h = 0.62 + 0.30 * nlines
        y -= h + 0.28
        ax.add_patch(FancyBboxPatch((x, y), colw, h, boxstyle='round,pad=0.03,rounding_size=0.06',
                                    fc=fill, ec=edge, lw=1.6))
        ax.text(cx, y + h - 0.26, title, ha='center', va='center', fontsize=10, fontweight='bold', color='#222')
        ax.text(cx, y + (h - 0.5) / 2, sub, ha='center', va='center', fontsize=8.2, color='#333')

# big stage-to-stage arrows
def arrow(x1, x2, y, color='#444'):
    ax.add_patch(FancyArrowPatch((x1, y), (x2, y), arrowstyle='-|>', mutation_scale=22,
                                 lw=2.4, color=color))
ymid = 4.6
for i in range(4):
    xa = centers[i][0] + colw
    xb = centers[i + 1][0]
    arrow(xa + 0.02, xb - 0.02, ymid, STAGES[i + 1][1])

# feedback note: real data injection
ax.text(centers[0][1], 0.85, '↑ real LC-MS labels override\nsimulated ones (append-only)',
        ha='center', va='center', fontsize=8, style='italic', color='#2c7fb8')

# frozen banner
ax.add_patch(FancyBboxPatch((x0, 0.05), colw * 5 + gap * 4, 0.55,
                            boxstyle='round,pad=0.02,rounding_size=0.05', fc='#fff7e6', ec='#d9a441', lw=1.4))
ax.text((x0 + (colw * 5 + gap * 4) / 2), 0.32,
        'FROZEN before the fellowship: architecture · evaluation protocol · baselines · sealed predictions.   '
        'DURING: only the data change (simulated → real LC-HRMS/MS).',
        ha='center', va='center', fontsize=9.5, color='#8a5a00', fontweight='bold')

ax.set_title('SEAGRAMMAR — rule-learning pipeline:  data → model → calibration → tests → figures',
             fontsize=14, fontweight='bold', pad=12)
fig.tight_layout()
for ext in ('png', 'svg'):
    p = os.path.join(OUT, f'SEAGRAMMAR_pipeline_flow.{ext}')
    fig.savefig(p, dpi=140, bbox_inches='tight'); print('->', os.path.relpath(p, ROOT))
plt.close(fig)
