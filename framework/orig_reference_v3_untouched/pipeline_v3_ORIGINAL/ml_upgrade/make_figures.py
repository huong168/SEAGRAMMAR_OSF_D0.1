#!/usr/bin/env python3
# ml_upgrade/make_figures.py
# =============================================================================
# Render figures from the saved outputs of calibrate_eval.py + baselines_tests.py.
# Reads visualizations/ml_upgrade/*.{csv,json} (whatever engine produced them) and
# writes publication-style PNGs for the proposal's preliminary-data section.
#
# Run AFTER the two compute scripts:  python3 make_figures.py
# =============================================================================
import os, json, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, 'visualizations', 'ml_upgrade')
plt.rcParams.update({'figure.dpi': 130, 'font.size': 10, 'axes.grid': True,
                     'grid.alpha': 0.3, 'axes.axisbelow': True})
GREEN, BLUE, ORANGE, GREY = '#2ca02c', '#2c7fb8', '#ff7f0e', '#888888'


def _save(fig, name):
    p = os.path.join(D, name); fig.tight_layout(); fig.savefig(p); plt.close(fig)
    print("  ->", os.path.relpath(p, ROOT))


def fig_calibration_brier():
    f = os.path.join(D, 'calibration_metrics.csv')
    if not os.path.exists(f):
        return
    df = pd.read_csv(f).dropna(subset=['brier_calibrated']).sort_values('brier_calibrated')
    macro = {}
    mp = os.path.join(D, 'calibration_summary.json')
    if os.path.exists(mp):
        macro = json.load(open(mp))
    y = np.arange(len(df)); h = 0.38
    fig, ax = plt.subplots(figsize=(10, 6.5))
    ax.barh(y + h/2, df['brier_raw'], h, color=GREY, label='raw (uncalibrated)')
    ax.barh(y - h/2, df['brier_calibrated'], h, color=GREEN, label='isotonic-calibrated')
    ax.set_yticks(y); ax.set_yticklabels(df['label'])
    for i, (_, r) in enumerate(df.iterrows()):
        ax.text(0.002, i, f"n={int(r['support'])}", va='center', fontsize=7, color='k')
    ttl = 'Probability calibration — Brier per reaction class (lower = better)'
    if macro:
        ttl += f"\nmacro raw {macro.get('macro_brier_raw')} → calibrated {macro.get('macro_brier_calibrated')}"
    ax.set_xlabel('Brier score'); ax.set_title(ttl); ax.legend(loc='lower right')
    _save(fig, 'fig_calibration_brier.png')


def fig_reliability():
    files = sorted(glob.glob(os.path.join(D, 'reliability_*.csv')))
    files = [f for f in files if pd.read_csv(f).shape[0] >= 3][:6]
    if not files:
        return
    n = len(files); cols = 3; rows = int(np.ceil(n/cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 3.4*rows), squeeze=False)
    for ax in axes.flat:
        ax.set_visible(False)
    for k, f in enumerate(files):
        ax = axes.flat[k]; ax.set_visible(True)
        d = pd.read_csv(f); lab = os.path.basename(f)[len('reliability_'):-4]
        ax.plot([0, 1], [0, 1], '--', color=GREY, lw=1)
        ax.plot(d['mean_pred'].to_numpy(), d['obs_freq'].to_numpy(), 'o-', color=BLUE, ms=4)
        ax.set_title(lab, fontsize=9); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel('mean predicted p'); ax.set_ylabel('observed freq')
    fig.suptitle('Reliability curves (calibrated) — points on diagonal = well-calibrated', y=1.02)
    _save(fig, 'fig_reliability.png')


def _report():
    p = os.path.join(D, 'baselines_tests_report.json')
    return json.load(open(p)) if os.path.exists(p) else None


def fig_baselines():
    r = _report()
    if not r:
        return
    b = r['baselines_multilabel_brier']
    keys = ['rule_model_calibrated', 'tanimoto_knn', 'frequency',
            'rule_model_on_marine', 'public_only_on_marine']
    keys = [k for k in keys if k in b]
    vals = [b[k] for k in keys]
    colors = [GREEN if 'rule' in k else (ORANGE if 'public' in k else BLUE) for k in keys]
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(keys))
    ax.bar(x, vals, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.002, f"{v:.3f}", ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels([k.replace('_', '\n') for k in keys], fontsize=8)
    ax.set_ylabel('multi-label Brier (lower = better)')
    adv = b.get('marine_data_advantage(public-rule)')
    ax.set_title('Baselines — rule model vs naive similarity / frequency / public-only'
                 + (f'\nmarine-data advantage (public − rule) = {adv}' if adv is not None else ''))
    _save(fig, 'fig_baselines.png')


def fig_reusability():
    r = _report()
    if not r or 'testB_reusability' not in r:
        return
    B = pd.DataFrame(r['testB_reusability'])
    x = np.arange(len(B)); h = 0.38
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - h/2, B['wJaccard'], h, color=GREEN, label='weighted Jaccard')
    ax.bar(x + h/2, B['cosine'], h, color=BLUE, label='cosine')
    lab = [f"{r_['rule']}\n{r_['A']}↔{r_['B']}" for _, r_ in B.iterrows()]
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=8)
    ax.axhline(0.5, color=GREY, ls='--', lw=1)
    ax.set_ylabel('profile similarity'); ax.set_ylim(0, 1.05)
    ax.set_title('Test B — reusability across matched substrate pairs\n(ring_cleavage luteolin↔apigenin is the negative control)')
    ax.legend(loc='lower right')
    _save(fig, 'fig_reusability.png')


def fig_productivity():
    r = _report()
    if not r:
        return
    items = [('C1 sealed-6 proxy\n(leave-compound-out)', r.get('testC1_productivity_compound_out')),
             ('C2 hardest stress\n(leave-family-out)', r.get('testC2_productivity_familyout_stress'))]
    items = [(n, d) for n, d in items if d]
    if not items:
        return
    fig, ax = plt.subplots(figsize=(8, 5)); x = np.arange(len(items)); h = 0.38
    rule = [d['brier_rule'] for _, d in items]
    base = [d['brier_tanimoto_baseline'] for _, d in items]
    ax.bar(x - h/2, rule, h, color=GREEN, label='rule model')
    ax.bar(x + h/2, base, h, color=BLUE, label='Tanimoto baseline')
    for i, (_, d) in enumerate(items):
        win = 'rule wins' if d['rule_wins'] else 'tie/again'
        ax.text(i, max(rule[i], base[i]) + 0.004,
                f"Δ={d['delta(base-rule)']:+.3f}\nP(rule>base)={d['P(rule better)']}\n{win}",
                ha='center', fontsize=8)
    ax.set_xticks(x); ax.set_xticklabels([n for n, _ in items], fontsize=8)
    ax.set_ylabel('multi-label Brier (lower = better)')
    ax.set_title('Test C — productivity: rule vs similarity (paired bootstrap)')
    ax.legend(loc='upper left')
    _save(fig, 'fig_productivity.png')


def main():
    print("Rendering figures into", os.path.relpath(D, ROOT))
    fig_calibration_brier(); fig_reliability(); fig_baselines()
    fig_reusability(); fig_productivity()
    print("Done.")


if __name__ == '__main__':
    main()
