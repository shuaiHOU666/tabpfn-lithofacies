"""
lowo.py - leave-one-well-out check on Hugoton-Panoma (robustness of the two-blind-well result)

Each real well is held out in turn; the rest train. Compares original features with
original + TabPFN embeddings (well-grouped out-of-fold) for the four classifiers.
No SHAP-RFE here, to keep it fast.  python lowo.py --device cpu   -> results/lowo.csv
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

import lith_tabpfn as lt
from run_lithology import DEMO_NAMES

HERE = Path(__file__).resolve().parent
FEATS = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS"]

a = argparse.ArgumentParser()
a.add_argument("--device", default="auto")
a.add_argument("--seed", type=int, default=0)
a = a.parse_args()

df = lt.load_tables([HERE / "data" / "facies_vectors.csv"], well_col="Well Name")
df = lt.clean_logs(df, FEATS, "Facies", "Well Name", "Depth")
df["Facies"] = df["Facies"].map(DEMO_NAMES)
wells = [w for w in df["Well Name"].unique() if w != "Recruit F9"]   # F9 is a training-only pseudo-well

rows = []
for w in wells:
    te = (df["Well Name"] == w).to_numpy()
    dtr, dte = df[~te].reset_index(drop=True), df[te].reset_index(drop=True)
    le = LabelEncoder().fit(dtr.Facies)
    dte = dte[dte.Facies.isin(le.classes_)].reset_index(drop=True)
    ytr, yte = le.transform(dtr.Facies), le.transform(dte.Facies)
    sc = lt.Scaler().fit(dtr[FEATS].to_numpy(float))
    Xtr, Xte = sc.transform(dtr[FEATS].to_numpy(float)), sc.transform(dte[FEATS].to_numpy(float))
    print(f"Blind well {w}: train {len(Xtr)}, test {len(Xte)}")
    Etr, Ete, _, _ = lt.tabpfn_embeddings(Xtr, ytr, Xte, n_fold=5, seed=a.seed,
                                          groups=dtr["Well Name"].to_numpy(), device=a.device)
    for sname, (A, B) in {"Raw": (Xtr, Xte), "Raw+TabPFN": (np.hstack([Xtr, Etr]), np.hstack([Xte, Ete]))}.items():
        for mname, model in lt.make_models(a.seed).items():
            p = lt.fit_predict(model, A, ytr, B)
            rows.append(dict(well=w, model=mname, set=sname, accuracy=accuracy_score(yte, p),
                             macro_f1=f1_score(yte, p, average="macro")))
    pd.DataFrame(rows).to_csv(HERE / "results" / "lowo.csv", index=False)

r = pd.DataFrame(rows)
print(r.groupby(["model", "set"])[["accuracy", "macro_f1"]].agg(["mean", "std"]).round(3))
