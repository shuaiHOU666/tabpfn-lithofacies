"""
lith_tabpfn.py
==============
Core functions for lithology / facies classification with TabPFN feature enhancement
and SHAP-guided feature selection (the six steps of the TabPFN lithology patent):

  S1  clean + standardize logs            -> clean_logs(), Scaler
  S2  TabPFN out-of-fold embeddings       -> tabpfn_embeddings()
  S3  concatenate [X*, E]                 -> np.hstack in run_lithology.py
  S4  train RF / SVM / XGBoost / MLP      -> make_models(), fit_predict()
  S5  SHAP importance (global, per class) -> shap_importance()
  S6  SHAP recursive feature elimination  -> shap_rfe()

Kept deliberately simple: no heavy tuning, sensible defaults, works on any table
with a label column and a well column (CSV or LAS).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier


# ----------------------------------------------------------------------------- S1 data
def load_tables(paths, well_col="WELL"):
    """Read one or more CSV / LAS files into one DataFrame.
    For LAS files the well name is taken from the header (WELL) or the file name."""
    frames = []
    for p in paths:
        p = Path(p)
        if p.suffix.lower() == ".las":
            import lasio
            las = lasio.read(str(p))
            df = las.df().reset_index()
            name = str(las.well.WELL.value).strip() if "WELL" in las.well else ""
            df[well_col] = name or p.stem
        else:
            df = pd.read_csv(p)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def clean_logs(df, features, label, well=None, depth=None):
    """Drop unlabelled rows, fill gaps by depth interpolation inside each well,
    then fill what is left with the median."""
    df = df.dropna(subset=[label]).copy()
    if depth and depth in df:
        df = df.sort_values([well, depth] if well else [depth])
    if well and well in df:
        df[features] = df.groupby(well)[features].transform(
            lambda s: s.interpolate(limit_direction="both"))
    df[features] = df[features].fillna(df[features].median())
    return df.reset_index(drop=True)


class Scaler:
    """Clip outliers to the training 0.5-99.5 percentiles, then Z-score (fit on train only)."""

    def fit(self, X):
        self.lo, self.hi = np.nanpercentile(X, [0.5, 99.5], axis=0)
        Xc = np.clip(X, self.lo, self.hi)
        self.mu, self.sd = Xc.mean(0), Xc.std(0) + 1e-9
        return self

    def transform(self, X):
        return (np.clip(X, self.lo, self.hi) - self.mu) / self.sd


# ----------------------------------------------------------------------------- S2 TabPFN
def make_tabpfn(version="v2", device="auto", n_estimators=1, seed=0):
    """TabPFN classifier. 'v2' is open (no licence click-through) and is the default;
    'latest' uses the newest model in your installed tabpfn (may need a HuggingFace login)."""
    import torch
    from tabpfn import TabPFNClassifier
    # float32: on CPU/GPU autocast may give bfloat16 embeddings, which numpy cannot read
    kw = dict(device=device, n_estimators=n_estimators, random_state=seed,
              ignore_pretraining_limits=True, inference_precision=torch.float32)
    if version == "latest":
        return TabPFNClassifier(**kw)
    from tabpfn.constants import ModelVersion
    return TabPFNClassifier.create_default_for_version(ModelVersion.V2, **kw)


def _embed(clf, X):
    E = np.asarray(clf.get_embeddings(X, data_source="test"))
    return E[0] if E.ndim == 3 else E          # (n_estimators, n, d) -> first member


def _folds(y, n_fold, seed, groups=None):
    """Folds for out-of-fold embeddings. With `groups` (well names) whole wells are held out,
    so a training row is embedded without its own well in the context - the same situation
    as a blind well. Otherwise stratified folds (plain K-fold if a class is too small)."""
    if groups is not None and len(np.unique(groups)) >= 2:
        return GroupKFold(min(n_fold, len(np.unique(groups)))).split(y, y, groups)
    if np.bincount(y).min() >= n_fold:
        return StratifiedKFold(n_fold, shuffle=True, random_state=seed).split(y, y)
    return KFold(n_fold, shuffle=True, random_state=seed).split(y)


def tabpfn_embeddings(Xtr, ytr, Xte, n_fold=5, seed=0, quick=False, groups=None, **tabpfn_kw):
    """Out-of-fold TabPFN embeddings.
    Each training row is embedded by a model that never saw its label (support = other
    folds, query = this fold); test rows use the full training set as support.
    Returns (E_train, E_test, model_or_None, test_proba)."""
    if quick:                                   # stand-in for a fast dry run, NOT for results
        return _quick_embeddings(Xtr, ytr, Xte, n_fold, seed, groups)
    Etr = None
    for k, (sup, qry) in enumerate(_folds(ytr, n_fold, seed, groups), 1):
        print(f"  TabPFN fold {k}/{n_fold}: support {len(sup)}, query {len(qry)}")
        clf = make_tabpfn(seed=seed, **tabpfn_kw).fit(Xtr[sup], ytr[sup])
        e = _embed(clf, Xtr[qry])
        if Etr is None:
            Etr = np.zeros((len(Xtr), e.shape[1]), np.float32)
        Etr[qry] = e
    print("  TabPFN full fit for the test set")
    clf = make_tabpfn(seed=seed, **tabpfn_kw).fit(Xtr, ytr)
    return Etr, _embed(clf, Xte), clf, clf.predict_proba(Xte)


def _quick_embeddings(Xtr, ytr, Xte, n_fold, seed, groups=None):
    """Out-of-fold random-forest class probabilities, used only to test the pipeline
    without downloading TabPFN weights."""
    rf = lambda: RandomForestClassifier(200, min_samples_leaf=3, n_jobs=-1, random_state=seed)
    K = ytr.max() + 1
    Etr = np.zeros((len(Xtr), K), np.float32)
    for sup, qry in _folds(ytr, n_fold, seed, groups):
        m = rf().fit(Xtr[sup], ytr[sup])
        Etr[np.ix_(qry, m.classes_)] = m.predict_proba(Xtr[qry])
    m = rf().fit(Xtr, ytr)
    Ete = np.zeros((len(Xte), K), np.float32)
    Ete[:, m.classes_] = m.predict_proba(Xte)
    return Etr, Ete, None, Ete


# ----------------------------------------------------------------------------- S4 models
def make_models(seed=0):
    return {
        "RF": RandomForestClassifier(300, class_weight="balanced", n_jobs=-1, random_state=seed),
        "SVM": SVC(C=10, gamma="scale", class_weight="balanced", random_state=seed),
        "XGBoost": make_xgb(seed),
        "MLP": MLPClassifier((128, 64), alpha=1e-3, max_iter=800, early_stopping=True,
                             random_state=seed),
    }


def make_xgb(seed=0, n_estimators=300):
    return XGBClassifier(n_estimators=n_estimators, max_depth=5, learning_rate=0.05,
                         subsample=0.8, colsample_bytree=0.8, random_state=seed,
                         n_jobs=-1, eval_metric="mlogloss")


def fit_predict(model, Xtr, ytr, Xte):
    """Fit with balanced class weights (where the learner supports it) and predict.
    Labels are re-encoded inside, so a fold that misses a class still works."""
    le = LabelEncoder().fit(ytr)
    y = le.transform(ytr)
    if isinstance(model, XGBClassifier):
        model.fit(Xtr, y, sample_weight=compute_sample_weight("balanced", y))
    else:
        model.fit(Xtr, y)
    return le.inverse_transform(model.predict(Xte))


# ----------------------------------------------------------------------------- S5 SHAP
def shap_importance(X, y, max_rows=2000, seed=0):
    """SHAP values of an XGBoost model on X.
    Returns global importance I_j (mean |phi| over samples and classes, shape (f,)) and
    class-wise importance I_jk (mean |phi_jk| over samples of class k, shape (f, K))."""
    import shap
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), min(max_rows, len(X)), replace=False)
    m = make_xgb(seed, n_estimators=200)
    m.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    sv = shap.TreeExplainer(m).shap_values(X[idx])
    sv = np.stack(sv, -1) if isinstance(sv, list) else np.asarray(sv)
    if sv.ndim == 2:                                   # binary case -> (n, f, 1)
        sv = sv[..., None]
    a = np.abs(sv)                                     # (n, f, K)
    glob = a.mean(axis=(0, 2))
    yy = y[idx]
    cls = np.stack([a[yy == k, :, k].mean(0) if (yy == k).any() else np.zeros(X.shape[1])
                    for k in range(a.shape[2])], 1)
    return glob, cls


# ----------------------------------------------------------------------------- S6 SHAP-RFE
def cv_macro_f1(X, y, groups=None, n_splits=5, seed=0):
    """Macro-F1 of XGBoost under grouped (by well) cross-validation."""
    if groups is not None and len(np.unique(groups)) >= 3:
        splits = GroupKFold(min(n_splits, len(np.unique(groups)))).split(X, y, groups)
    else:
        splits = StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y)
    true, pred = [], []
    for tr, va in splits:
        pred.append(fit_predict(make_xgb(seed, 150), X[tr], y[tr], X[va]))
        true.append(y[va])
    return f1_score(np.concatenate(true), np.concatenate(pred), average="macro")


def shap_rfe(X, y, names, groups=None, drop_frac=0.25, min_features=3, seed=0, tol=0.005):
    """Recursive feature elimination: score, recompute SHAP on the remaining features,
    drop the weakest `drop_frac`, repeat. Picks the smallest set within `tol` of the best."""
    keep = np.arange(X.shape[1])
    hist = []
    while True:
        score = cv_macro_f1(X[:, keep], y, groups, seed=seed)
        hist.append(dict(n_features=len(keep), macro_f1=score,
                         features="|".join(names[i] for i in keep)))
        print(f"  SHAP-RFE: {len(keep):4d} features -> CV macro-F1 {score:.3f}")
        if len(keep) <= min_features:
            break
        imp, _ = shap_importance(X[:, keep], y, seed=seed)
        n_drop = max(1, int(round(len(keep) * drop_frac)))
        n_drop = min(n_drop, len(keep) - min_features)
        keep = keep[np.sort(np.argsort(imp)[n_drop:])]
    hist = pd.DataFrame(hist)
    best = hist.macro_f1.max()
    chosen = hist[hist.macro_f1 >= best - tol].sort_values("n_features").iloc[0]
    sel = [names.index(f) for f in chosen.features.split("|")]
    return np.array(sel), hist
