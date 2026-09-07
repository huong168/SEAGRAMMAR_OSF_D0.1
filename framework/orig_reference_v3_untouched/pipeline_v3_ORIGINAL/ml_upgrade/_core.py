#!/usr/bin/env python3
# ml_upgrade/_core.py
# =============================================================================
# Pure-numpy primitives so the upgrade runs WITHOUT sklearn/rdkit (fallback),
# and is swapped for sklearn/rdkit automatically when those are installed.
#   - LogisticRegression (numpy, L2, sample_weight, class-balanced)
#   - pav_isotonic (calibration)  - GroupKFold (numpy)
#   - metrics: brier, jensen_shannon, weighted_jaccard, cosine
#   - tanimoto_matrix (binary fingerprints)  - paired_bootstrap
# =============================================================================
import numpy as np

# ---- optional sklearn / rdkit detection -------------------------------------
try:
    from sklearn.ensemble import RandomForestClassifier  # noqa
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs           # noqa
    HAS_RDKIT = True
except Exception:
    HAS_RDKIT = False


# ----------------------------- classifier ------------------------------------
class NumpyLogReg:
    """L2-regularised logistic regression, sample weights + class balancing."""
    def __init__(self, l2=1.0, lr=0.5, epochs=300, balanced=True, seed=42):
        self.l2, self.lr, self.epochs, self.balanced, self.seed = l2, lr, epochs, balanced, seed
        self.w = None; self.b = 0.0; self.classes_ = np.array([0, 1])

    def fit(self, X, y, sample_weight=None):
        X = np.asarray(X, float); y = np.asarray(y, float)
        n, d = X.shape
        sw = np.ones(n) if sample_weight is None else np.asarray(sample_weight, float)
        if self.balanced:
            npos, nneg = max(y.sum(), 1), max((1 - y).sum(), 1)
            cw = np.where(y == 1, n / (2 * npos), n / (2 * nneg))
            sw = sw * cw
        mu, sd = X.mean(0), X.std(0) + 1e-9
        self._mu, self._sd = mu, sd
        Xs = (X - mu) / sd
        rng = np.random.default_rng(self.seed)
        self.w = rng.normal(0, 0.01, d); self.b = 0.0
        swn = sw / sw.sum()
        for _ in range(self.epochs):
            z = Xs @ self.w + self.b
            p = 1 / (1 + np.exp(-z))
            g = (p - y) * swn
            gw = Xs.T @ g + self.l2 * self.w / n
            gb = g.sum()
            self.w -= self.lr * gw; self.b -= self.lr * gb
        return self

    def predict_proba(self, X):
        Xs = (np.asarray(X, float) - self._mu) / self._sd
        p = 1 / (1 + np.exp(-(Xs @ self.w + self.b)))
        return np.column_stack([1 - p, p])


def make_classifier():
    """RandomForest if sklearn present, else numpy logistic regression."""
    if HAS_SKLEARN:
        return RandomForestClassifier(n_estimators=60, random_state=42,
                                      class_weight='balanced', n_jobs=-1)
    return NumpyLogReg()


# ----------------------------- calibration -----------------------------------
def pav_isotonic_fit(x, y, w=None):
    """Pool-adjacent-violators isotonic regression. Returns (xs, ys) step points."""
    order = np.argsort(x, kind='mergesort')
    xs = np.asarray(x, float)[order]; ys = np.asarray(y, float)[order]
    ws = (np.ones_like(xs) if w is None else np.asarray(w, float)[order])
    # merge ties on x
    val = ys.copy(); wt = ws.copy(); blocks = list(range(len(xs)))
    i = 0
    level_y = list(val); level_w = list(wt); level_x = list(xs)
    # iterative PAV
    yv = list(val); wv = list(wt)
    k = 0
    out_y = []; out_w = []
    for j in range(len(yv)):
        cy, cw = yv[j], wv[j]
        out_y.append(cy); out_w.append(cw)
        while len(out_y) > 1 and out_y[-2] > out_y[-1]:
            y2, w2 = out_y.pop(), out_w.pop()
            y1, w1 = out_y.pop(), out_w.pop()
            ny = (y1 * w1 + y2 * w2) / (w1 + w2)
            out_y.append(ny); out_w.append(w1 + w2)
    # expand back to per-point fitted values
    fitted = []
    bi = 0
    counts = []
    # recompute block sizes by re-running PAV with index tracking
    out_y2, out_w2, out_n = [], [], []
    for j in range(len(yv)):
        out_y2.append(yv[j]); out_w2.append(wv[j]); out_n.append(1)
        while len(out_y2) > 1 and out_y2[-2] > out_y2[-1]:
            y2, w2, n2 = out_y2.pop(), out_w2.pop(), out_n.pop()
            y1, w1, n1 = out_y2.pop(), out_w2.pop(), out_n.pop()
            ny = (y1 * w1 + y2 * w2) / (w1 + w2)
            out_y2.append(ny); out_w2.append(w1 + w2); out_n.append(n1 + n2)
    for yval, nval in zip(out_y2, out_n):
        fitted.extend([yval] * nval)
    fitted = np.clip(np.array(fitted), 0, 1)
    return xs, fitted


def pav_predict(xs, fitted, xq):
    """Step/linear interpolation of isotonic fit at query points."""
    xq = np.asarray(xq, float)
    return np.clip(np.interp(xq, xs, fitted), 0, 1)


# ----------------------------- CV --------------------------------------------
def group_kfold(groups, n_splits=5, seed=42, shuffle=False):
    # FIX: shuffle=False mặc định -> tất định; nhất quán reproducibility giữa các script
    # (train_model_v2 dùng sklearn.GroupKFold không shuffle). Đặt shuffle=True nếu cần ngẫu nhiên.
    groups = np.asarray(groups)
    uniq = np.array(sorted(set(groups.tolist())))
    if shuffle:
        rng = np.random.default_rng(seed); rng.shuffle(uniq)
    folds = np.array_split(uniq, n_splits)
    idx = np.arange(len(groups))
    for f in folds:
        te = np.isin(groups, f)
        yield idx[~te], idx[te]


# ----------------------------- metrics ---------------------------------------
def brier(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    m = ~np.isnan(p)
    return float(np.mean((p[m] - y[m]) ** 2))


def jensen_shannon(P, Q, eps=1e-12):
    P = np.asarray(P, float) + eps; Q = np.asarray(Q, float) + eps
    P /= P.sum(); Q /= Q.sum(); M = 0.5 * (P + Q)
    kl = lambda A, B: np.sum(A * np.log2(A / B))
    return float(0.5 * kl(P, M) + 0.5 * kl(Q, M))   # in [0,1] with log2


def weighted_jaccard(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    denom = np.maximum(a, b).sum()
    return float(np.minimum(a, b).sum() / denom) if denom > 0 else 0.0


def cosine(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


# ----------------------------- fingerprints ----------------------------------
def morgan_bits(smiles, n_bits=2048, radius=2):
    if not HAS_RDKIT:
        return None
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=int); DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def tanimoto(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    inter = np.logical_and(a, b).sum(); union = np.logical_or(a, b).sum()
    return float(inter / union) if union > 0 else 0.0


# ----------------------------- bootstrap -------------------------------------
def paired_bootstrap(diff, B=10000, seed=42):
    """diff = baseline_loss - rule_loss per item; positive => rule better.
    Returns (mean, ci_low, ci_high, p_win=fraction of resamples with mean>0)."""
    diff = np.asarray(diff, float); n = len(diff)
    rng = np.random.default_rng(seed)
    means = np.empty(B)
    for i in range(B):
        s = rng.integers(0, n, n)
        means[i] = diff[s].mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(diff.mean()), float(lo), float(hi), float((means > 0).mean())
