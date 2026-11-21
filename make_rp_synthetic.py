"""
make_rp_synthetic.py
====================
Synthetic facies dataset for the RP project (SCAN, Dinantian carbonates + Triassic clastics),
built with the rock-physics models in rockphysics.py (DEM + Gassmann).

Six facies, defined by porosity, crack porosity, mineralogy and clay content:
  TightLst      tight limestone, phi 0.5-3 %, no cracks
  VuggyLst      vuggy/moldic limestone, phi 6-15 %
  FracturedLst  fractured limestone, phi 2-7 %, crack porosity 0.3-1 %   (the geothermal target)
  Dolostone     dolomite 70-100 %, phi 3-12 %
  Sandstone     Triassic-type sandstone, phi 10-25 %
  Shale         clay-rich, phi 5-15 %

For each synthetic well a layered column (0.5 m sampling) is drawn, the model gives Vp, Vs, rho,
and the logs are degraded the way seismic-inversion results are: vertical smoothing,
relative noise (AI 6 %, Vp/Vs 5 %, density 0.05 g/cc) and a
per-well calibration bias (3 %). Features written:
  GR, AI (km/s * g/cc), VPVS, RHOB     + truth columns PHIT, PHI_CRACK (not used as features)

The point is the RP question: can elastic attributes (AI, Vp/Vs, density) from inversion
separate fractured carbonate from porous carbonate? Replace this file with SCAN well data
(digitised core facies + logs) when available.

Usage:  python make_rp_synthetic.py          -> data/rp_synthetic_facies.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                  # rockphysics.py sits next to this script
from rockphysics import carbonate_model, sandstone_model  # noqa: E402

rng = np.random.default_rng(42)
def U(*a):
    return rng.uniform(*a)

# facies: (probability, sampler returning (vp, vs, rho, phi, crack, GR mean))
def tight():
    p = U(0.005, 0.03); return (*carbonate_model(p, 0.0), p, 0.0, 15)
def vuggy():
    p = U(0.06, 0.15); c = U(0, 0.0005)
    return (*carbonate_model(p, c, alpha_stiff=U(0.5, 1.0)), p, c, 15)
def fractured():
    p = U(0.02, 0.07); c = U(0.003, 0.01); return (*carbonate_model(p, c), p, c, 18)
def dolo():
    p = U(0.03, 0.12); c = U(0, 0.001)
    return (*carbonate_model(p, c, dolomite_frac=U(0.7, 1.0)), p, c, 12)
def sand():
    p = U(0.10, 0.25); return (*sandstone_model(p, U(0, 0.0005), clay_frac=U(0.02, 0.15)), p, 0.0, 45)
def shale():
    p = U(0.05, 0.15); return (*sandstone_model(p, 0.0, clay_frac=U(0.6, 0.8), alpha_pore=0.05), p, 0.0, 110)

FACIES = {"TightLst": (0.30, tight), "VuggyLst": (0.15, vuggy), "FracturedLst": (0.07, fractured),
          "Dolostone": (0.10, dolo), "Sandstone": (0.15, sand), "Shale": (0.23, shale)}


def smooth(x, n=9):
    return np.convolve(np.pad(x, n // 2, mode="edge"), np.ones(n) / n, mode="valid")


def make_well(name, top=2000.0, thickness=300.0, dz=0.5, bias_all=None):
    z = np.arange(top, top + thickness, dz)
    names, probs = list(FACIES), np.array([v[0] for v in FACIES.values()])
    rows, d = [], top
    while d < z[-1]:
        h = max(1.0, rng.exponential(6.0))                       # layer thickness, m
        f = rng.choice(names, p=probs)
        for zz in z[(z >= d) & (z < d + h)]:
            vp, vs, rho, phi, crack, gr = FACIES[f][1]()          # small within-layer variation
            rows.append((zz, f, vp, vs, rho, phi, crack, rng.normal(gr, 0.25 * gr)))
        d += h
    df = pd.DataFrame(rows, columns=["DEPTH", "FACIES", "VP", "VS", "RHOB", "PHIT", "PHI_CRACK", "GR"])
    # degrade like inverted / log data: smoothing, noise, per-well calibration bias
    bias = 1 + rng.normal(0, 0.03)
    ai = smooth(df.VP * df.RHOB) * bias
    vpvs = smooth(df.VP / df.VS)
    rho = smooth(df.RHOB)
    df["AI"] = ai * (1 + rng.normal(0, 0.06, len(df)))
    df["VPVS"] = vpvs * (1 + rng.normal(0, 0.05, len(df)))
    df["RHOB"] = rho + rng.normal(0, 0.05, len(df))
    df["GR"] = smooth(df.GR.to_numpy(), 3)
    if bias_all is not None:                     # controlled well-to-well shift on every attribute
        df["AI"] *= 1 + rng.normal(0, bias_all)
        df["VPVS"] *= 1 + rng.normal(0, bias_all)
        df["RHOB"] *= 1 + rng.normal(0, bias_all / 2)
        df["GR"] *= 1 + rng.normal(0, 3 * bias_all)
    df.insert(0, "WELL", name)
    return df[["WELL", "DEPTH", "FACIES", "GR", "AI", "VPVS", "RHOB", "PHIT", "PHI_CRACK"]]


def main():
    import argparse
    global rng
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-wells", type=int, default=8)
    ap.add_argument("--bias", type=float, default=None,
                    help="extra per-well calibration shift (relative sd) on all attributes, e.g. 0.05")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(HERE / "data" / "rp_synthetic_facies.csv"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    out = Path(a.out)
    out.parent.mkdir(exist_ok=True, parents=True)
    df = pd.concat([make_well(f"W{i:02d}", bias_all=a.bias) for i in range(1, a.n_wells + 1)], ignore_index=True)
    df.round(4).to_csv(out, index=False)
    print(f"Wrote {out} ({len(df)} samples, {a.n_wells} wells)")
    print(df.FACIES.value_counts().to_string())


if __name__ == "__main__":
    main()
