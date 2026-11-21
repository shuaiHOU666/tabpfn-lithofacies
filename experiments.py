"""
experiments.py
==============
Experiments on when TabPFN embeddings help, using any table with a label and a well column.

  lowo    leave-one-well-out; each well held out in turn
  nwells  train on k randomly chosen wells (k in --k, --repeats draws), test on all other wells

Every run compares, for RF / SVM / XGBoost / MLP:
  Raw          standardized logs
  Raw+TabPFN   logs + TabPFN embeddings (well-grouped out-of-fold for the training wells)
and TabPFN used directly as a classifier.

--norm global : Z-score with training-well statistics (as in run_lithology.py)
--norm well   : each well standardized with its own median / IQR (uses the logs only, no labels;
                legitimate for a blind well). Columns in --no-norm (flags, positions) stay global.

Examples
  python experiments.py lowo   --data data/facies_vectors.csv --label Facies --well "Well Name" \
         --features GR ILD_log10 DeltaPHI PHIND PE NM_M RELPOS --no-norm NM_M RELPOS --norm well --out results_exp/hug_lowo_well
  python experiments.py nwells --data data/facies_vectors.csv --label Facies --well "Well Name" \
         --features GR ILD_log10 DeltaPHI PHIND PE NM_M RELPOS --k 2 4 6 8 --repeats 3 --out results_exp/hug_nwells
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

import lith_tabpfn as lt


def per_well_norm(df, feats, well):
    """Robust per-well standardization: (x - median_well) / IQR_well."""
    out = df.copy()
    for f in feats:
        g = df.groupby(well)[f]
        med = g.transform("median")
        iqr = g.transform(lambda s: s.quantile(0.75) - s.quantile(0.25)).replace(0, np.nan)
        out[f] = ((df[f] - med) / iqr).fillna(0.0)
    return out


def evaluate(dtr, dte, feats, label, well, norm="global", no_norm=(), device="auto", seed=0,
             with_embeddings=True):
    """Fit everything on dtr, score on dte. Returns (rows, predictions)."""
    le = LabelEncoder().fit(dtr[label])
    dte = dte[dte[label].isin(le.classes_)].reset_index(drop=True)
    ytr, yte = le.transform(dtr[label]), le.transform(dte[label])
    if norm == "well":
        nf = [f for f in feats if f not in no_norm]
        dtr, dte = per_well_norm(dtr, nf, well), per_well_norm(dte, nf, well)
    sc = lt.Scaler().fit(dtr[feats].to_numpy(float))
    Xtr, Xte = sc.transform(dtr[feats].to_numpy(float)), sc.transform(dte[feats].to_numpy(float))
    sets = {"Raw": (Xtr, Xte)}
    if with_embeddings:
        Etr, Ete, _, _ = lt.tabpfn_embeddings(Xtr, ytr, Xte, n_fold=5, seed=seed,
                                              groups=dtr[well].to_numpy(), device=device)
        sets["Raw+TabPFN"] = (np.hstack([Xtr, Etr]), np.hstack([Xte, Ete]))
    preds = {}
    for sname, (A, B) in sets.items():
        for mname, model in lt.make_models(seed).items():
            preds[(mname, sname)] = lt.fit_predict(model, A, ytr, B)
    clf = lt.make_tabpfn(device=device, n_estimators=4, seed=seed).fit(Xtr, ytr)
    preds[("TabPFN classifier", "Raw")] = clf.predict(Xte)
    rows, P = [], []
    for (m, s), p in preds.items():
        rows.append(dict(model=m, set=s, n_test=len(yte), accuracy=accuracy_score(yte, p),
                         macro_f1=f1_score(yte, p, average="macro")))
        P.append(pd.DataFrame({"well": dte[well].to_numpy(), "true": dte[label].to_numpy(),
                               "pred": le.inverse_transform(p), "model": m, "set": s}))
    return rows, pd.concat(P)


def load(a):
    df = pd.read_csv(a.data)
    if a.wells:
        df = df[df[a.well].isin(a.wells)]
    df = df.dropna(subset=a.features, how="all")
    df = lt.clean_logs(df, a.features, a.label, a.well, a.depth if a.depth in df else None)
    vc = df[a.label].value_counts()
    return df[df[a.label].isin(vc[vc >= a.min_class].index)].reset_index(drop=True)


def pooled(P):
    return P.groupby(["model", "set"]).apply(lambda g: pd.Series(dict(
        accuracy=accuracy_score(g.true, g.pred), macro_f1=f1_score(g.true, g.pred, average="macro"))),
        include_groups=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["lowo", "nwells"])
    ap.add_argument("--data", required=True)
    ap.add_argument("--label", default="FACIES")
    ap.add_argument("--well", default="WELL")
    ap.add_argument("--depth", default="DEPTH")
    ap.add_argument("--features", nargs="+", required=True)
    ap.add_argument("--wells", nargs="*", default=None)
    ap.add_argument("--test-exclude", nargs="*", default=["Recruit F9"], help="wells never used as test wells")
    ap.add_argument("--norm", choices=["global", "well"], default="global")
    ap.add_argument("--no-norm", nargs="*", default=[])
    ap.add_argument("--min-class", type=int, default=20)
    ap.add_argument("--k", nargs="+", type=int, default=[2, 4, 6, 8])
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    df = load(a)
    wells = list(df[a.well].unique())
    test_wells = [w for w in wells if w not in a.test_exclude]
    print(f"{len(df)} samples, {len(wells)} wells, classes {sorted(df[a.label].unique())}")
    rows, preds, t0 = [], [], time.time()

    if a.mode == "lowo":
        for w in test_wells:
            te = (df[a.well] == w).to_numpy()
            r, P = evaluate(df[~te], df[te], a.features, a.label, a.well, a.norm, a.no_norm, a.device, a.seed)
            for x in r:
                x.update(well=w)
            rows += r; preds.append(P)
            print(f"  {w}: done ({time.time() - t0:.0f} s)")
            pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)
    else:
        rng = np.random.default_rng(a.seed)
        trainable = test_wells + [w for w in wells if w in a.test_exclude]
        for k in a.k:
            for rep in range(a.repeats):
                tr = list(rng.choice(test_wells, size=min(k, len(test_wells) - 1), replace=False))
                tr += [w for w in wells if w in a.test_exclude]           # pseudo-wells always train
                trm = df[a.well].isin(tr).to_numpy()
                tem = ~trm & ~df[a.well].isin(a.test_exclude).to_numpy()
                r, P = evaluate(df[trm], df[tem], a.features, a.label, a.well, a.norm, a.no_norm,
                                a.device, a.seed + rep)
                for x in r:
                    x.update(k=k, repeat=rep, train_wells="|".join(map(str, tr)))
                P["k"], P["repeat"] = k, rep
                rows += r; preds.append(P)
                print(f"  k={k} rep={rep}: done ({time.time() - t0:.0f} s)")
                pd.DataFrame(rows).to_csv(out / "runs.csv", index=False)
        _ = trainable
    R = pd.DataFrame(rows)
    P = pd.concat(preds)
    P.to_csv(out / "predictions.csv", index=False)
    if a.mode == "lowo":
        pooled(P).round(4).to_csv(out / "pooled.csv")
        print(pooled(P).round(3))
    else:
        s = R.groupby(["k", "model", "set"])[["accuracy", "macro_f1"]].agg(["mean", "std"]).round(3)
        s.to_csv(out / "summary.csv")
        print(s)
    print(f"Done in {time.time() - t0:.0f} s -> {out}")


if __name__ == "__main__":
    main()
