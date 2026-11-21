"""
run_lithology.py
================
End-to-end run: TabPFN feature enhancement + SHAP feature selection for lithology /
facies classification, evaluated on blind wells.

Examples
--------
# 1) Public Hugoton-Panoma demo (downloads facies_vectors.csv on first run)
python run_lithology.py --demo

# 2) Fast dry run without TabPFN weights (checks the pipeline only, numbers are NOT TabPFN)
python run_lithology.py --demo --quick

# 3) RP synthetic carbonate facies (make the data first with make_rp_synthetic.py)
python run_lithology.py --data data/rp_synthetic_facies.csv --label FACIES --well WELL ^
       --depth DEPTH --features GR AI VPVS RHOB --blind W07 W08 --out results_rp

# 4) Your own wells: CSV or LAS files with a facies column
python run_lithology.py --data my_wells.csv --label FACIES --well WELL --depth DEPTH ^
       --features GR RHOB NPHI DT PEF --blind WELL-A

Outputs go to --out (default results/): metrics.csv, predictions.csv, shap_*.csv,
rfe_history.csv and figures/*.png.
"""
from __future__ import annotations

import argparse
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import lith_tabpfn as lt

HERE = Path(__file__).resolve().parent
DEMO_URL = "https://raw.githubusercontent.com/seg/2016-ml-contest/master/facies_vectors.csv"
DEMO_NAMES = {1: "1-SS", 2: "2-CSiS", 3: "3-FSiS", 4: "4-SiSh", 5: "5-MS", 6: "6-WS", 7: "7-D", 8: "8-PS", 9: "9-BS"}

# categorical colours (fixed order) + grey for a 9th class
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300",
           "#6250d6", "#e34948", "#8c8c8c", "#5b4a3a"]
SET_COLORS = {"Raw": "#8c8c8c", "Raw+TabPFN": "#2a78d6", "Selected": "#eb6834"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": "#888", "axes.titlesize": 10})


def get_args():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument("--demo", action="store_true", help="use the public Hugoton-Panoma dataset")
    a.add_argument("--data", nargs="+", help="CSV or LAS file(s)")
    a.add_argument("--label", default="Facies")
    a.add_argument("--well", default="Well Name")
    a.add_argument("--depth", default="Depth")
    a.add_argument("--features", nargs="+",
                   default=["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS"])
    a.add_argument("--blind", nargs="*", default=["SHANKLE", "NEWBY"],
                   help="wells held out for testing; empty -> random 20%% split")
    a.add_argument("--quick", action="store_true",
                   help="stand-in embedder (no TabPFN download) to test the pipeline")
    a.add_argument("--tabpfn-version", default="v2", choices=["v2", "latest"])
    a.add_argument("--device", default="auto", help="auto / cpu / cuda")
    a.add_argument("--n-fold", type=int, default=5, help="out-of-fold splits for embeddings")
    a.add_argument("--oof", default="well", choices=["well", "random"],
                   help="out-of-fold split for training embeddings: by well (default) or random rows")
    a.add_argument("--no-rfe", action="store_true", help="skip SHAP feature selection")
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--out", default="results")
    return a.parse_args()


# ----------------------------------------------------------------------------- plots
def fig_tsne(Xraw, E, y, names, path, seed=0, n=1500):
    idx = np.random.default_rng(seed).choice(len(y), min(n, len(y)), replace=False)
    fig, axs = plt.subplots(1, 2, figsize=(9, 4))
    for ax, Z, t in zip(axs, [Xraw, E], ["(a) Standardized original features",
                                         "(b) TabPFN embeddings"]):
        P = TSNE(2, random_state=seed, init="pca", perplexity=30).fit_transform(Z[idx])
        for k, nm in enumerate(names):
            m = y[idx] == k
            ax.scatter(P[m, 0], P[m, 1], s=8, color=PALETTE[k % len(PALETTE)], label=nm,
                       linewidths=0.3, edgecolors="white")
        ax.set_title(t, loc="left"); ax.set_xticks([]); ax.set_yticks([])
    axs[1].legend(markerscale=2, frameon=False, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_scores(metrics, ref, path):
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    models = metrics.model.unique()
    sets = [s for s in SET_COLORS if s in metrics.set.unique()]
    w = 0.8 / len(sets)
    for ax, col, t in zip(axs, ["accuracy", "macro_f1"], ["Blind-well accuracy", "Blind-well macro-F1"]):
        for i, s in enumerate(sets):
            v = [metrics[(metrics.model == m) & (metrics.set == s)][col].iloc[0] for m in models]
            x = np.arange(len(models)) + (i - (len(sets) - 1) / 2) * w
            ax.bar(x, v, w * 0.92, color=SET_COLORS[s], label=s)
            for xi, vi in zip(x, v):
                ax.text(xi, vi + 0.01, f"{vi:.2f}", ha="center", fontsize=7, color="#444")
        if ref is not None:
            ax.axhline(ref[col], color="#444", ls="--", lw=1, label="TabPFN as classifier")
        ax.set_xticks(range(len(models)), models); ax.set_ylim(0, 1.05); ax.set_title(t, loc="left")
        ax.grid(axis="y", color="#e5e5e5"); ax.set_axisbelow(True)
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False)
    fig.tight_layout(rect=[0, 0, 1, 0.92]); fig.savefig(path, dpi=200); plt.close(fig)


def fig_confusion(y, preds, titles, names, path):
    K = len(names)
    fig, axs = plt.subplots(1, len(preds), figsize=(4.6 * len(preds), 4.2))
    for ax, p, t in zip(np.atleast_1d(axs), preds, titles):
        cm = confusion_matrix(y, p, labels=range(K)).astype(float)
        cm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
        for i in range(K):
            for j in range(K):
                if cm[i, j] >= 0.05:
                    ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=6.5,
                            color="white" if cm[i, j] > 0.6 else "#222")
        ax.set_xticks(range(K), names, rotation=45); ax.set_yticks(range(K), names)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(t, loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_shap(glob, cls, feat_names, raw_names, class_names, path, top=20):
    order = np.argsort(glob)[::-1][:top]
    fig, axs = plt.subplots(1, 2, figsize=(11, 0.28 * top + 1.2),
                            gridspec_kw=dict(width_ratios=[1, 1.3]))
    ax = axs[0]
    cols = ["#2a78d6" if feat_names[i] in raw_names else "#eb6834" for i in order]
    ax.barh(range(len(order)), glob[order], color=cols)
    ax.set_yticks(range(len(order)), [feat_names[i] for i in order]); ax.invert_yaxis()
    ax.set_xlabel("Mean |SHAP|"); ax.set_title("(a) Global importance", loc="left")
    from matplotlib.patches import Patch
    ax.legend([Patch(color="#2a78d6"), Patch(color="#eb6834")], ["Original log", "TabPFN embedding"],
              frameon=False, loc="lower right")
    o2 = order[:12]
    H = cls[o2]
    im = axs[1].imshow(H / H.max(), cmap="Blues", aspect="auto")
    axs[1].set_yticks(range(len(o2)), [feat_names[i] for i in o2])
    axs[1].set_xticks(range(len(class_names)), class_names, rotation=45)
    axs[1].set_title("(b) Class-wise importance (normalized)", loc="left")
    fig.colorbar(im, ax=axs[1], shrink=0.8)
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_rfe(hist, path):
    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.plot(hist.n_features, hist.macro_f1, "-o", color="#2a78d6", lw=2, ms=5)
    b = hist.macro_f1.idxmax()
    ax.plot(hist.n_features[b], hist.macro_f1[b], "o", color="#eb6834", ms=9)
    ax.set_xscale("log"); ax.set_xlabel("Number of retained features")
    ax.set_ylabel("Grouped-CV macro-F1"); ax.set_title("SHAP-based feature elimination", loc="left")
    ax.grid(color="#e5e5e5")
    fig.tight_layout(); fig.savefig(path, dpi=200); plt.close(fig)


def fig_well(dfw, feats, depth, tracks, names, path):
    K = len(names)
    cmap = ListedColormap([PALETTE[k % len(PALETTE)] for k in range(K)])
    feats = feats[:5]
    fig, axs = plt.subplots(1, len(feats) + len(tracks), figsize=(1.3 * (len(feats) + len(tracks)) + 1.5, 7),
                            sharey=True)
    z = dfw[depth].to_numpy() if depth in dfw else np.arange(len(dfw))
    for ax, f in zip(axs, feats):
        ax.plot(dfw[f], z, color="#333", lw=0.7); ax.set_title(f, fontsize=8); ax.grid(color="#eee")
    for ax, (t, lab) in zip(axs[len(feats):], tracks.items()):
        ax.imshow(np.repeat(lab[:, None], 2, 1), aspect="auto", cmap=cmap, vmin=-0.5, vmax=K - 0.5,
                  extent=[0, 1, z.max(), z.min()], interpolation="nearest")
        ax.set_title(t, fontsize=8); ax.set_xticks([])
    axs[0].set_ylim(z.max(), z.min()); axs[0].set_ylabel(depth)
    handles = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[k % len(PALETTE)]) for k in range(K)]
    fig.legend(handles, names, loc="center right", frameon=False, fontsize=8)
    fig.tight_layout(rect=[0, 0, 0.88, 1]); fig.savefig(path, dpi=200); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    a = get_args()
    out = HERE / a.out
    (out / "figures").mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # ---- S1 data
    if a.demo:
        path = HERE / "data" / "facies_vectors.csv"
        if not path.exists():
            path.parent.mkdir(exist_ok=True)
            print("Downloading Hugoton-Panoma facies_vectors.csv ...")
            urllib.request.urlretrieve(DEMO_URL, path)
        a.data = [path]
    df = lt.load_tables(a.data, well_col=a.well)
    df = lt.clean_logs(df, a.features, a.label, a.well, a.depth)
    if a.demo:
        df[a.label] = df[a.label].map(DEMO_NAMES)

    if a.blind:
        test_mask = df[a.well].isin(a.blind).to_numpy()
        missing = set(a.blind) - set(df[a.well].unique())
        if missing:
            raise SystemExit(f"Blind wells not found: {missing}. Wells: {sorted(df[a.well].unique())}")
    else:
        _, te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=a.seed,
                                 stratify=df[a.label])
        test_mask = np.isin(np.arange(len(df)), te)
    dtr, dte = df[~test_mask].reset_index(drop=True), df[test_mask].reset_index(drop=True)
    le = LabelEncoder().fit(dtr[a.label])
    dte = dte[dte[a.label].isin(le.classes_)].reset_index(drop=True)   # drop unseen classes
    names = [str(c) for c in le.classes_]
    ytr, yte = le.transform(dtr[a.label]), le.transform(dte[a.label])
    groups = dtr[a.well].to_numpy() if a.well in dtr else None
    print(f"Train {len(dtr)} samples / {dtr[a.well].nunique()} wells, test {len(dte)} samples, "
          f"{len(names)} classes: {names}")

    sc = lt.Scaler().fit(dtr[a.features].to_numpy(float))
    Xtr, Xte = sc.transform(dtr[a.features].to_numpy(float)), sc.transform(dte[a.features].to_numpy(float))

    # ---- S2 TabPFN embeddings (out-of-fold) and S3 concatenation
    print("Extracting embeddings" + (" (quick stand-in, not TabPFN)" if a.quick else " with TabPFN"))
    oof_groups = groups if a.oof == "well" else None
    Etr, Ete, _, _ = lt.tabpfn_embeddings(Xtr, ytr, Xte, n_fold=a.n_fold, seed=a.seed, quick=a.quick,
                                          groups=oof_groups,
                                              version=a.tabpfn_version, device=a.device)
    emb_names = [f"E{i:03d}" for i in range(Etr.shape[1])]
    feat_names = list(a.features) + emb_names
    XtrE, XteE = np.hstack([Xtr, Etr]), np.hstack([Xte, Ete])
    print(f"Enhanced feature set: {len(a.features)} original + {Etr.shape[1]} embedding = {XtrE.shape[1]}")

    # ---- S5 SHAP and S6 selection
    glob, cls = lt.shap_importance(XtrE, ytr, seed=a.seed)
    pd.DataFrame({"feature": feat_names, "mean_abs_shap": glob}).sort_values(
        "mean_abs_shap", ascending=False).to_csv(out / "shap_global.csv", index=False)
    pd.DataFrame(cls, index=feat_names, columns=names).to_csv(out / "shap_classwise.csv")
    sets = {"Raw": (Xtr, Xte), "Raw+TabPFN": (XtrE, XteE)}
    if not a.no_rfe:
        print("SHAP-based recursive feature elimination (grouped CV by well)")
        sel, hist = lt.shap_rfe(XtrE, ytr, feat_names, groups=groups, seed=a.seed)
        hist.to_csv(out / "rfe_history.csv", index=False)
        sets["Selected"] = (XtrE[:, sel], XteE[:, sel])
        print(f"  selected {len(sel)} features: {[feat_names[i] for i in sel]}")
        fig_rfe(hist, out / "figures" / "fig7_shap_rfe.png")

    # ---- S4 classifiers on each feature set
    rows, preds = [], {}
    for sname, (A, B) in sets.items():
        for mname, model in lt.make_models(a.seed).items():
            p = lt.fit_predict(model, A, ytr, B)
            preds[(mname, sname)] = p
            rows.append(dict(model=mname, set=sname, n_features=A.shape[1],
                             accuracy=accuracy_score(yte, p), macro_f1=f1_score(yte, p, average="macro")))
            print(f"  {mname:8s} {sname:11s} acc {rows[-1]['accuracy']:.3f}  macro-F1 {rows[-1]['macro_f1']:.3f}")
    metrics = pd.DataFrame(rows)
    ref = None
    if not a.quick:                       # reference: TabPFN itself as the classifier (4 members)
        ref_clf = lt.make_tabpfn(a.tabpfn_version, a.device, n_estimators=4, seed=a.seed).fit(Xtr, ytr)
        pt = ref_clf.predict_proba(Xte).argmax(1)
        ref = dict(accuracy=accuracy_score(yte, pt), macro_f1=f1_score(yte, pt, average="macro"))
        metrics = pd.concat([metrics, pd.DataFrame([dict(model="TabPFN classifier", set="Raw",
                                                         n_features=Xtr.shape[1], **ref)])])
        print(f"  TabPFN as classifier       acc {ref['accuracy']:.3f}  macro-F1 {ref['macro_f1']:.3f}")
    metrics.to_csv(out / "metrics.csv", index=False)

    # ---- outputs
    m4 = metrics[metrics.model != "TabPFN classifier"]
    best = m4[m4.set != "Raw"].sort_values("macro_f1").iloc[-1]
    bm, bs = best.model, best.set
    pr = dte[[c for c in [a.well, a.depth, a.label] if c in dte]].copy()
    pr[f"pred_{bm}_Raw"] = le.inverse_transform(preds[(bm, "Raw")])
    pr[f"pred_{bm}_{bs}"] = le.inverse_transform(preds[(bm, bs)])
    pr.to_csv(out / "predictions.csv", index=False)

    F = out / "figures"
    fig_tsne(Xtr, Etr, ytr, names, F / "fig2_tsne_embeddings.png", a.seed)
    fig_scores(m4, ref, F / "fig3_accuracy_f1.png")
    fig_confusion(yte, [preds[(bm, "Raw")], preds[(bm, bs)]],
                  [f"(a) {bm}, original features", f"(b) {bm}, {bs}"], names, F / "fig4_confusion.png")
    fig_shap(glob, cls, feat_names, set(a.features), names, F / "fig5_shap_importance.png")
    for w in (a.blind or []):
        m = (dte[a.well] == w).to_numpy()
        tracks = {"Core": yte[m], f"{bm}\nRaw": preds[(bm, "Raw")][m], f"{bm}\n{bs}": preds[(bm, bs)][m]}
        fig_well(dte[m], a.features, a.depth, tracks, names, F / f"fig6_blind_well_{w}.png")

    print(f"\nBest: {bm} on {bs}: acc {best.accuracy:.3f}, macro-F1 {best.macro_f1:.3f}")
    print(f"Done in {time.time() - t0:.0f} s. Results in {out}")


if __name__ == "__main__":
    main()
