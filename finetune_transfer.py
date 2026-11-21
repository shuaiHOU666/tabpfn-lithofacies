"""
finetune_transfer.py
====================
Part 1 (Hugoton-Panoma, public): fine-tune TabPFN on public well logs and test it on blind wells.
Part 2 (SCAN, application): use the Hugoton-fine-tuned weights on the Dutch Dinantian carbonates,
with the SCAN training wells as in-context examples (leave-one-well-out).

Both datasets share three log features, so the fine-tuned model can be transferred:
  LOG_RDEP  log10 deep resistivity          (Hugoton: ILD_log10)
  PHIND     average neutron-density porosity, %   (SCAN: 100 * (NPHI + PHID) / 2)
  DeltaPHI  neutron minus density porosity, %     (SCAN: 100 * (NPHI - PHID))
with PHID = (2.71 - RHOB) / (2.71 - 1.0) (limestone matrix).

python finetune_transfer.py --part 1 --device cpu --out results_ft
python finetune_transfer.py --part 2 --scan data/scan_facies.csv --device cpu --out results_ft
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

import lith_tabpfn as lt

HERE = Path(__file__).resolve().parent
HUG_ALL = ["GR", "ILD_log10", "DeltaPHI", "PHIND", "PE", "NM_M", "RELPOS"]
COMMON = ["LOG_RDEP", "PHIND", "DeltaPHI"]
BLIND = ["SHANKLE", "NEWBY"]
VAL_WELL = "CHURCHMAN BIBLE"         # held out for early stopping; the only real well with all 9 facies


def hugoton():
    df = pd.read_csv(HERE / "data" / "facies_vectors.csv")
    df = lt.clean_logs(df, HUG_ALL, "Facies", "Well Name", "Depth")
    df["LOG_RDEP"] = df["ILD_log10"]
    return df.rename(columns={"Well Name": "WELL", "Facies": "FACIES", "Depth": "DEPTH"})


def scan(path, min_class=20):
    df = pd.read_csv(path)
    df = df[df.WELL.isin(["BHG-01", "KTG-01", "O18-01"])].dropna(subset=["RHOB", "NPHI", "LOG_RDEP"])
    phid = (2.71 - df.RHOB) / 1.71
    df["PHIND"] = 100 * (df.NPHI + phid) / 2
    df["DeltaPHI"] = 100 * (df.NPHI - phid)
    vc = df.FACIES.value_counts()
    return df[df.FACIES.isin(vc[vc >= min_class].index)].reset_index(drop=True)


def finetune(X, y, Xv, yv, device, epochs, seed, out, early_stopping=True):
    from tabpfn.constants import ModelVersion
    from tabpfn.finetuning import FinetunedTabPFNClassifier
    ft = FinetunedTabPFNClassifier(
        device=device, epochs=epochs, learning_rate=1e-5, random_state=seed,
        n_finetune_ctx_plus_query_samples=1000, finetune_ctx_query_split_ratio=0.2,
        n_estimators_finetune=1, n_estimators_validation=1, n_estimators_final_inference=4,
        early_stopping=early_stopping, early_stopping_patience=5,
        validation_split_ratio=0.1 if early_stopping else 0.0, eval_metric="log_loss",
        save_checkpoint_interval=None, model_version=ModelVersion.V2,
        extra_classifier_kwargs=dict(inference_precision=torch.float32, ignore_pretraining_limits=True))
    t = time.time()
    ft.fit(X, y, X_val=Xv if early_stopping else None, y_val=yv if early_stopping else None, output_dir=out)
    print(f"  fine-tuned in {time.time() - t:.0f} s")
    return ft


def with_weights(state, X, y, device, seed):
    """Plain TabPFN classifier fitted on (X, y) as context, carrying fine-tuned weights."""
    clf = lt.make_tabpfn(device=device, n_estimators=4, seed=seed).fit(X, y)
    for m in getattr(clf, "models_", [clf.model_]):
        m.load_state_dict(state)
    return clf


def score(rows, tag, yt, p, **kw):
    rows.append(dict(model=tag, accuracy=accuracy_score(yt, p), macro_f1=f1_score(yt, p, average="macro"), **kw))
    print(f"  {tag:34s} acc {rows[-1]['accuracy']:.3f}  macro-F1 {rows[-1]['macro_f1']:.3f}")


def part1(a, out):
    df = hugoton()
    rows = []
    for feats, fname in [(HUG_ALL, "all7"), (COMMON, "common3")]:
        te = df.WELL.isin(BLIND).to_numpy()
        va = (df.WELL == VAL_WELL).to_numpy()
        dtr, dva, dte = df[~te & ~va], df[va], df[te]
        le = LabelEncoder().fit(df[~te].FACIES)
        sc = lt.Scaler().fit(dtr[feats].to_numpy(float))
        X, Xv, Xt = (sc.transform(d[feats].to_numpy(float)) for d in (dtr, dva, dte))
        y, yv, yt = (le.transform(d.FACIES) for d in (dtr, dva, dte))
        print(f"\nPart 1, features {fname}: train {len(X)}, val {len(Xv)} ({VAL_WELL}), blind {len(Xt)}")
        Xall, yall = np.vstack([X, Xv]), np.r_[y, yv]          # context for inference: all non-blind wells
        base = lt.make_tabpfn(device=a.device, n_estimators=4, seed=a.seed).fit(Xall, yall)
        score(rows, "TabPFN (pretrained)", yt, base.predict(Xt), features=fname)
        ft = finetune(X, y, Xv, yv, a.device, a.epochs, a.seed, out / f"ckpt_hug_{fname}", not a.no_early_stop)
        state = ft.finetuned_estimator_.model_.state_dict()
        torch.save(state, out / f"tabpfn_ft_hugoton_{fname}.pt")
        clf = with_weights(state, Xall, yall, a.device, a.seed)
        score(rows, "TabPFN (fine-tuned on Hugoton)", yt, clf.predict(Xt), features=fname)
        for m, model in lt.make_models(a.seed).items():
            score(rows, f"{m} (original logs)", yt, lt.fit_predict(model, Xall, yall, Xt), features=fname)
        pd.DataFrame(rows).to_csv(out / "part1_hugoton_blind.csv", index=False)


def part2(a, out):
    hug = hugoton()
    sc_h = lt.Scaler().fit(hug[COMMON].to_numpy(float))
    path = out / "tabpfn_ft_hugoton_all_common3.pt"
    if not path.exists():                                       # fine-tune on all Hugoton wells
        va = (hug.WELL == VAL_WELL).to_numpy()
        le = LabelEncoder().fit(hug.FACIES)
        X = sc_h.transform(hug[~va][COMMON].to_numpy(float)); y = le.transform(hug[~va].FACIES)
        Xv = sc_h.transform(hug[va][COMMON].to_numpy(float)); yv = le.transform(hug[va].FACIES)
        print("Fine-tuning on all Hugoton wells (common features)")
        ft = finetune(X, y, Xv, yv, a.device, a.epochs, a.seed, out / "ckpt_hug_all_common3", not a.no_early_stop)
        torch.save(ft.finetuned_estimator_.model_.state_dict(), path)
    state = torch.load(path)
    df = scan(a.scan)
    print(df.groupby(["WELL", "FACIES"]).size().unstack(fill_value=0))
    rows, preds = [], []
    for w in df.WELL.unique():
        te = (df.WELL == w).to_numpy()
        dtr, dte = df[~te], df[te]
        le = LabelEncoder().fit(dtr.FACIES)
        dte = dte[dte.FACIES.isin(le.classes_)]
        y, yt = le.transform(dtr.FACIES), le.transform(dte.FACIES)
        print(f"\nPart 2, blind {w}: train {len(y)}, test {len(yt)}")
        for feats, fname in [(COMMON, "common3"), (["LOG_RDEP", "RHOB", "NPHI", "DT"], "scan4")]:
            sc = lt.Scaler().fit(dtr[feats].to_numpy(float))
            X, Xt = sc.transform(dtr[feats].to_numpy(float)), sc.transform(dte[feats].to_numpy(float))
            res = {"TabPFN (pretrained)": lt.make_tabpfn(device=a.device, n_estimators=4, seed=a.seed).fit(X, y).predict(Xt)}
            if fname == "common3":
                res["TabPFN (fine-tuned on Hugoton)"] = with_weights(state, X, y, a.device, a.seed).predict(Xt)
            for m, model in lt.make_models(a.seed).items():
                res[f"{m} (original logs)"] = lt.fit_predict(model, X, y, Xt)
            for tag, p in res.items():
                score(rows, tag, yt, p, features=fname, well=w)
                preds.append(pd.DataFrame(dict(well=w, features=fname, model=tag, true=le.inverse_transform(yt),
                                               pred=le.inverse_transform(p))))
        pd.DataFrame(rows).to_csv(out / "part2_scan_lowo.csv", index=False)
    P = pd.concat(preds)
    P.to_csv(out / "part2_scan_predictions.csv", index=False)
    pooled = P.groupby(["features", "model"]).apply(lambda g: pd.Series(dict(
        accuracy=accuracy_score(g.true, g.pred), macro_f1=f1_score(g.true, g.pred, average="macro"))),
        include_groups=False).round(4)
    pooled.to_csv(out / "part2_scan_pooled.csv")
    print("\nPooled over SCAN wells:\n", pooled)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, choices=[1, 2], required=True)
    ap.add_argument("--scan", default=str(HERE / "data" / "scan_facies.csv"))
    ap.add_argument("--device", default="auto")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-early-stop", action="store_true",
                    help="fine-tune for exactly --epochs epochs without validation-based early stopping")
    ap.add_argument("--out", default="results_ft")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    (part1 if a.part == 1 else part2)(a, out)


if __name__ == "__main__":
    main()
