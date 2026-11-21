"""
ablation.py  -  Table 3 of the paper (run after run_lithology.py --demo)

Uses the best classifier from results/metrics.csv and compares, on the blind wells:
  Original features only
  TabPFN embeddings only
  Original + embeddings, vanilla extraction (training rows embedded with their own labels in context)
  Original + embeddings, out-of-fold by random rows
  Original + embeddings, out-of-fold by well (the proposed method)
  Original + embeddings, SHAP-RFE selected (features from results/rfe_history.csv)

python ablation.py --device cpu
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
BLIND = ["SHANKLE", "NEWBY"]

a = argparse.ArgumentParser()
a.add_argument("--device", default="auto")
a.add_argument("--results", default="results")
a.add_argument("--seed", type=int, default=0)
a = a.parse_args()
res = HERE / a.results

df = lt.load_tables([HERE / "data" / "facies_vectors.csv"], well_col="Well Name")
df = lt.clean_logs(df, FEATS, "Facies", "Well Name", "Depth")
df["Facies"] = df["Facies"].map(DEMO_NAMES)
te = df["Well Name"].isin(BLIND).to_numpy()
dtr, dte = df[~te].reset_index(drop=True), df[te].reset_index(drop=True)
le = LabelEncoder().fit(dtr.Facies)
ytr, yte = le.transform(dtr.Facies), le.transform(dte.Facies)
sc = lt.Scaler().fit(dtr[FEATS].to_numpy(float))
Xtr, Xte = sc.transform(dtr[FEATS].to_numpy(float)), sc.transform(dte[FEATS].to_numpy(float))

m = pd.read_csv(res / "metrics.csv")
m = m[(m.model != "TabPFN classifier") & (m.set != "Raw")]
best = m.sort_values("macro_f1").iloc[-1].model
print("Best classifier:", best)

groups = dtr["Well Name"].to_numpy()
print("Out-of-fold embeddings, by well ...")
Etr, Ete, _, _ = lt.tabpfn_embeddings(Xtr, ytr, Xte, n_fold=5, seed=a.seed, groups=groups, device=a.device)
print("Out-of-fold embeddings, by random rows ...")
Etr_r, Ete_r, _, _ = lt.tabpfn_embeddings(Xtr, ytr, Xte, n_fold=5, seed=a.seed, device=a.device)
print("Vanilla embeddings (training rows see their own labels) ...")
clf = lt.make_tabpfn(device=a.device, seed=a.seed).fit(Xtr, ytr)
E = np.asarray(clf.get_embeddings(Xtr, data_source="train"))
Etr_v = E[0] if E.ndim == 3 else E
Ete_v = lt._embed(clf, Xte)

names = FEATS + [f"E{i:03d}" for i in range(Etr.shape[1])]
h = pd.read_csv(res / "rfe_history.csv")
chosen = h[h.macro_f1 >= h.macro_f1.max() - 0.005].sort_values("n_features").iloc[0]
sel = [names.index(f) for f in chosen.features.split("|")]

XtrE, XteE = np.hstack([Xtr, Etr]), np.hstack([Xte, Ete])
configs = {
    "Original features only": (Xtr, Xte),
    "TabPFN embeddings only": (Etr, Ete),
    "Original + embeddings, vanilla extraction": (np.hstack([Xtr, Etr_v]), np.hstack([Xte, Ete_v])),
    "Original + embeddings, out-of-fold by random rows": (np.hstack([Xtr, Etr_r]), np.hstack([Xte, Ete_r])),
    "Original + embeddings, out-of-fold by well": (XtrE, XteE),
    "Original + embeddings, SHAP-RFE selected": (XtrE[:, sel], XteE[:, sel]),
}
rows = []
for name, (A, B) in configs.items():
    model = lt.make_models(a.seed)[best]
    p_tr = lt.fit_predict(model, A, ytr, A)
    p = lt.fit_predict(lt.make_models(a.seed)[best], A, ytr, B)
    rows.append(dict(configuration=name, n_features=A.shape[1],
                     train_accuracy=accuracy_score(ytr, p_tr),
                     blind_accuracy=accuracy_score(yte, p), blind_macro_f1=f1_score(yte, p, average="macro")))
    print(f"  {name:48s} train {rows[-1]['train_accuracy']:.3f}  blind acc {rows[-1]['blind_accuracy']:.3f}"
          f"  macro-F1 {rows[-1]['blind_macro_f1']:.3f}")
out = pd.DataFrame(rows)
out.insert(0, "classifier", best)
out.to_csv(res / "ablation.csv", index=False)
print("Saved", res / "ablation.csv")
