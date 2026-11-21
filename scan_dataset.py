"""
scan_dataset.py
===============
Build a training table for run_lithology.py from SCAN wells:
  core facies intervals (core_labels.py output)  +  NLOG LAS logs  ->  data/scan_facies.csv

LAS files are found under --las-dir. A well's files go either in their own folder
(data/las/LTG-01/*.las) or at any level with the well code as file-name prefix (BHG-01_Final_Evaluation.las).
Per well, a "*Final_Evaluation*.las" (SCAN petrophysics) is preferred, then "PUBLIC-COMP*.las",
then all other LAS files of that well merged.

Core depth -> log depth: if a "*Core_data*.las" with *_O (original) and *_DS (depth-shifted) core curves
exists, the shift is taken per cored interval from it; otherwise 0 and the well is flagged.
Shifts can be overridden in data/core_shift.csv (WELL, core_top, core_base, shift_m).

python scan_dataset.py --las-dir data --cores data/core_facies.csv --out data/scan_facies.csv
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import lasio
import numpy as np
import pandas as pd

# curve aliases -> (standard name, unit conversion to target)
ALIASES = {
    "GR": ["GR", "GR_EDTC", "GRC", "SGR", "CGR", "GRD"],
    "RDEP": ["DEEP_IND_RESISTIVITY", "DEEP_RESISTIVITY", "RES_DEP", "ILD", "LLD", "RD", "RT", "RILD", "RLLD", "HDRS", "AT90"],
    "RHOB": ["DENSITY_ED", "DEN", "RHOB", "RHOZ", "DENSITY"],
    "NPHI": ["NEUTRON", "NEU", "NPHI", "TNPH", "NPOR"],
    "DT": ["SONIC_ED", "DT", "DTCO", "DTC", "SON", "AC"],
    "PE": ["PE", "PEF", "PEFZ"],
}
FEATURES = ["GR", "LOG_RDEP", "RHOB", "NPHI", "DT", "PE"]


def well_code(path: Path, las_dir: Path):
    rel = path.relative_to(las_dir).parts
    if len(rel) > 1 and re.match(r"^[A-Z0-9]+-\d+", rel[-2]):
        return rel[-2]
    m = re.match(r"^([A-Z0-9]+-\d+(?:-S\d)?)[_ ]", path.name.upper())
    return m.group(1) if m else None


def read_curves(path: Path) -> pd.DataFrame:
    las = lasio.read(str(path), ignore_header_errors=True)
    df = las.df()
    df = df.mask(df <= -900)                         # undeclared nulls (-999, -999.25)
    depth_unit = str(las.index_unit or las.curves[0].unit).upper()
    if depth_unit.startswith("F"):
        df.index = df.index * 0.3048
    out = {}
    units = {c.mnemonic.upper(): str(c.unit).upper() for c in las.curves}
    cols = {c.upper(): c for c in df.columns}
    for std, names in ALIASES.items():
        for n in names:
            if n in cols:
                s = df[cols[n]].astype(float)
                u = units.get(n, "")
                if std == "RHOB" and (u.startswith("KG") or s.median() > 100):
                    s = s / 1000.0
                if std == "NPHI" and (u in ("%", "PU") or s.median() > 1.5):
                    s = s / 100.0
                if std == "DT" and ("/F" in u or (("/M" not in u) and s.median() < 140)):
                    s = s * 3.28084                                  # us/ft -> us/m
                out[std] = s
                break
    res = pd.DataFrame(out)
    res.index.name = "DEPTH"
    return res[~res.index.duplicated()]


def well_logs(files):
    up = lambda f: f.name.upper()
    pref = [f for f in files if "FINAL_EVALUATION" in up(f)] or \
           [f for f in files if "PUBLIC" in up(f)] or \
           [f for f in files if "COMPOSITE" in up(f) or up(f).endswith("_COMP.LAS")]
    skip = ("CORE", "WELLPATH", "TEMP", "DEVI", "TEST_INTERVAL")
    rest = [f for f in files if f not in pref and not any(k in up(f) for k in skip)]
    logs = None
    for f in pref + rest:                            # preferred file first, gaps filled from the others
        try:
            d = read_curves(f)
        except Exception:                            # noqa: BLE001  unreadable LAS
            continue
        if d.empty:
            continue
        if logs is None:
            logs = d
        else:                                        # resample onto the preferred depth grid before filling
            d = d.reindex(logs.index.union(d.index)).interpolate(limit=2).reindex(logs.index)
            logs = logs.combine_first(d)
    return logs


def core_shifts(files, intervals):
    """Shift per core interval from *_O / *_DS core curves (log = core + shift)."""
    cf = [f for f in files if "CORE_DATA" in f.name.upper()]
    if not cf:
        return None
    d = lasio.read(str(cf[0]), ignore_header_errors=True).df()
    o = [c for c in d.columns if c.upper().endswith("_O")]
    s = [c for c in d.columns if c.upper().endswith("_DS")]
    if not o or not s:
        return None
    o, s = d[o[0]].dropna(), d[s[0]].dropna()
    out = []
    for top, base in intervals:
        oo = o[(o.index >= top - 0.5) & (o.index <= base + 0.5)]
        sh = []
        for z, v in oo.items():
            k = s[(s - v).abs() < 1e-6]
            if len(k):
                sh.append(k.index[np.argmin(abs(k.index - z))] - z)
        out.append(float(np.median(sh)) if sh else np.nan)
    return out


def porosity_shift(files, logs, top, base, max_shift=12.0):
    """Fallback: shift that best correlates core porosity with log density porosity."""
    cf = [f for f in files if "CORE_DATA" in f.name.upper()]
    if not cf or "RHOB" not in logs:
        return np.nan, 0.0
    d = lasio.read(str(cf[0]), ignore_header_errors=True).df()
    col = [c for c in d.columns if c.upper().startswith("POROSITY_CORE")]
    if not col:
        return np.nan, 0.0
    cp = d[col[0]].dropna()
    cp = cp[(cp.index >= top - 1) & (cp.index <= base + 1)]
    if len(cp) < 8:
        return np.nan, 0.0
    phid = ((2.71 - logs.RHOB) / 1.71).dropna()
    best = (np.nan, 0.0)
    for sh in np.arange(-max_shift, max_shift + 0.01, 0.1):
        v = phid.reindex(cp.index + sh, method="nearest", tolerance=0.2).to_numpy()
        ok = ~np.isnan(v)
        if ok.sum() >= 8 and np.std(v[ok]) > 0:
            r = np.corrcoef(cp.to_numpy()[ok], v[ok])[0, 1]
            if r > best[1]:
                best = (sh, r)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--las-dir", default="data")
    ap.add_argument("--cores", default="data/core_facies.csv")
    ap.add_argument("--shifts", default="data/core_shift.csv")
    ap.add_argument("--out", default="data/scan_facies.csv")
    ap.add_argument("--min-feature-coverage", type=float, default=0.8,
                    help="drop a feature if it is missing in more than this share of labelled rows")
    a = ap.parse_args()
    las_dir = Path(a.las_dir)
    cores = pd.read_csv(a.cores)
    files = {}
    for f in las_dir.rglob("*"):
        if f.suffix.lower() == ".las":
            w = well_code(f, las_dir)
            if w:
                files.setdefault(w, []).append(f)
    manual = pd.read_csv(a.shifts) if Path(a.shifts).exists() else pd.DataFrame(
        columns=["WELL", "core_top", "core_base", "shift_m"])

    rows, report = [], []
    for well, iv in cores.groupby("WELL"):
        if well not in files:
            report.append(dict(WELL=well, status="no LAS found", core_m=round((iv.base - iv.top).sum(), 1)))
            continue
        logs = well_logs(files[well])
        # cored runs = groups of intervals with gaps < 3 m
        iv = iv.sort_values("top").reset_index(drop=True)
        run = (iv.top - iv.base.shift()).fillna(0).gt(3).cumsum()
        runs = [(g.top.min(), g.base.max()) for _, g in iv.groupby(run)]
        auto = core_shifts(files[well], runs)
        shifts = []
        for k, (t, b) in enumerate(runs):
            m = manual[(manual.WELL == well) & (manual.core_top <= t + 0.5) & (manual.core_base >= b - 0.5)]
            if len(m):
                shifts.append((float(m.shift_m.iloc[0]), "manual"))
            elif auto and not np.isnan(auto[k]):
                shifts.append((auto[k], "core_data DS"))
            else:
                sh, r = porosity_shift(files[well], logs, t, b)
                if not np.isnan(sh) and r >= 0.4:
                    shifts.append((sh, f"core-log porosity corr r={r:.2f}"))
                else:
                    shifts.append((0.0, "none - check!"))
        for k, ((t, b), (sh, src)) in enumerate(zip(runs, shifts)):
            sub = iv[(iv.top >= t - 1e-6) & (iv.base <= b + 1e-6)]
            seg = logs[(logs.index >= t + sh) & (logs.index <= b + sh)]
            fac = []                                    # samples in uncored gaps stay unlabelled
            for z in seg.index - sh:
                hit = sub[(sub.top <= z) & (sub.base >= z)]
                fac.append(hit.FACIES.iloc[0] if len(hit) else None)
            seg = seg.assign(WELL=well, CORE_DEPTH=seg.index - sh, FACIES=fac).dropna(subset=["FACIES"])
            rows.append(seg.reset_index())
            report.append(dict(WELL=well, run=k + 1, core_top=round(t, 2), core_base=round(b, 2),
                               shift_m=round(sh, 2), shift_source=src, samples=len(seg)))
    rep = pd.DataFrame(report)
    print(rep.to_string(index=False))
    if not rows:
        raise SystemExit("No labelled samples: put LAS files under --las-dir (one folder per well).")
    df = pd.concat(rows, ignore_index=True)
    if "RDEP" in df:
        df["LOG_RDEP"] = np.log10(df.RDEP.clip(lower=0.01))
    feats = [f for f in FEATURES if f in df and df[f].notna().mean() >= a.min_feature_coverage]
    df = df[["WELL", "DEPTH", "CORE_DEPTH", "FACIES"] + feats]
    df.to_csv(a.out, index=False)
    rep.to_csv(Path(a.out).with_name("scan_dataset_report.csv"), index=False)
    print(f"\nWrote {a.out}: {len(df)} samples, {df.WELL.nunique()} wells, features {feats}")
    print(df.groupby(["WELL", "FACIES"]).size().unstack(fill_value=0))
    print("\nNext:  python run_lithology.py --data", a.out, "--label FACIES --well WELL --depth DEPTH --features",
          " ".join(feats), "--blind <WELL> ...   and   python lowo.py-style leave-one-well-out")


if __name__ == "__main__":
    main()
