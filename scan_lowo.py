"""
scan_lowo.py - leave-one-well-out on the SCAN table (data/scan_facies.csv)

Each well is held out in turn. Compares original logs (Raw), original + TabPFN embeddings
(well-grouped out-of-fold) and TabPFN used directly as a classifier.
python scan_lowo.py --data data/scan_facies.csv --device cpu   ->  results_scan/lowo.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

import lith_tabpfn as lt

a = argparse.ArgumentParser()
a.add_argument("--data", default="data/scan_facies.csv")
a.add_argument("--features", nargs="+", default=["GR", "LOG_RDEP", "RHOB", "NPHI", "DT"])
a.add_argument("--wells", nargs="*", default=None, help="use only these wells")
a.add_argument("--min-class", type=int, default=20, help="drop classes with fewer samples than this")
a.add_argument("--device", default="auto")
a.add_argument("--out", default="results_scan")
a.add_argument("--seed", type=int, default=0)
a = a.parse_args()
out = Path(a.out); out.mkdir(exist_ok=True)

df = pd.read_csv(a.data)
if a.wells:
    df = df[df.WELL.isin(a.wells)]
df = df.dropna(subset=a.features, how="all")
df = lt.clean_logs(df, a.features, "FACIES", "WELL", "DEPTH")
keep = df.FACIES.value_counts()
df = df[df.FACIES.isin(keep[keep >= a.min_class].index)].reset_index(drop=True)
print(df.groupby(["WELL", "FACIES"]).size().unstack(fill_value=0))

rows, preds = [], []
for w in df.WELL.unique():
    te = (df.WELL == w).to_numpy()
    dtr, dte = df[~te].reset_index(drop=True), df[te].reset_index(drop=True)
    le = LabelEncoder().fit(dtr.FACIES)
    dte = dte[dte.FACIES.isin(le.classes_)].reset_index(drop=True)
    ytr, yte = le.transform(dtr.FACIES), le.transform(dte.FACIES)
    sc = lt.Scaler().fit(dtr[a.features].to_numpy(float))
    Xtr, Xte = sc.transform(dtr[a.features].to_numpy(float)), sc.transform(dte[a.features].to_numpy(float))
    print(f"\nBlind {w}: train {len(Xtr)} ({dtr.WELL.nunique()} wells), test {len(Xte)}")
    Etr, Ete, _, _ = lt.tabpfn_embeddings(Xtr, ytr, Xte, n_fold=5, seed=a.seed,
                                          groups=dtr.WELL.to_numpy(), device=a.device)
    res = {}
    for sname, (A, B) in {"Raw": (Xtr, Xte), "Raw+TabPFN": (np.hstack([Xtr, Etr]), np.hstack([Xte, Ete]))}.items():
        for mname, model in lt.make_models(a.seed).items():
            p = lt.fit_predict(model, A, ytr, B)
            res[(mname, sname)] = p
    clf = lt.make_tabpfn(device=a.device, n_estimators=4, seed=a.seed).fit(Xtr, ytr)
    res[("TabPFN classifier", "Raw")] = clf.predict(Xte)
    for (mname, sname), p in res.items():
        rows.append(dict(well=w, model=mname, set=sname, n_test=len(yte), accuracy=accuracy_score(yte, p),
                         macro_f1=f1_score(yte, p, average="macro")))
        print(f"  {mname:18s} {sname:11s} acc {rows[-1]['accuracy']:.3f}  macro-F1 {rows[-1]['macro_f1']:.3f}")
        pr = dte[["WELL", "DEPTH", "FACIES"]].copy(); pr["model"], pr["set"] = mname, sname
        pr["pred"] = le.inverse_transform(p); preds.append(pr)
    pd.DataFrame(rows).to_csv(out / "lowo.csv", index=False)
    pd.concat(preds).to_csv(out / "lowo_predictions.csv", index=False)

r = pd.DataFrame(rows)
# pooled over all held-out samples (wells differ a lot in size)
P = pd.concat(preds)
pooled = P.groupby(["model", "set"]).apply(lambda g: pd.Series(dict(
    accuracy=accuracy_score(g.FACIES, g.pred), macro_f1=f1_score(g.FACIES, g.pred, average="macro"))))
print("\nPer-well mean:\n", r.groupby(["model", "set"])[["accuracy", "macro_f1"]].mean().round(3))
print("\nPooled over all wells:\n", pooled.round(3))
pooled.round(4).to_csv(out / "lowo_pooled.csv")
