"""
rockphysics.py
==============
Small, dependency-light rock-physics toolbox used by the SCAN geothermal scripts.

Units: moduli in GPa, density in g/cc, velocity in km/s.
Impedance AI = rho * Vp  (g/cc * km/s), SI = rho * Vs.

Contents
--------
- vrh                    : Voigt-Reuss-Hill mineral mixing
- berryman_pq            : Berryman (1980) P, Q geometric factors for spheroidal inclusions
- dem                    : Differential Effective Medium with several inclusion types
                           (e.g. stiff vugs/moldic pores + compliant cracks)
- gassmann               : Gassmann fluid substitution
- velocities             : Vp, Vs from moduli and density
- schoenberg_hti         : HTI stiffness of a background with aligned vertical fractures
                           (Schoenberg linear-slip, via normal/tangential weaknesses)
- carbonate_model / sandstone_model : convenience forward models used in the templates

References
----------
Berryman, J.G. (1980) Long-wavelength propagation in composite elastic media II.
Mavko, G., Mukerji, T., Dvorkin, J. (2020) The Rock Physics Handbook, 3rd ed.
Schoenberg, M., Sayers, C. (1995) Seismic anisotropy of fractured rock. Geophysics 60.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

# ---------------------------------------------------------------------------
# Mineral and fluid properties (K, mu in GPa; rho in g/cc), Mavko et al. (2020)
# ---------------------------------------------------------------------------
MINERALS = {
    "calcite":  dict(K=76.8, mu=32.0, rho=2.71),
    "dolomite": dict(K=94.9, mu=45.0, rho=2.87),
    "quartz":   dict(K=37.0, mu=44.0, rho=2.65),
    "clay":     dict(K=21.0, mu=7.0,  rho=2.58),
}
# Saline formation water at ~2-4 km depth (approximate, Batzle & Wang type values)
BRINE = dict(K=2.6, rho=1.05)


def vrh(fractions, K, mu):
    """Voigt-Reuss-Hill average of mineral moduli."""
    f = np.asarray(fractions, float)
    f = f / f.sum()
    K, mu = np.asarray(K, float), np.asarray(mu, float)
    Kv, Kr = np.sum(f * K), 1.0 / np.sum(f / K)
    mv, mr = np.sum(f * mu), 1.0 / np.sum(f / mu)
    return 0.5 * (Kv + Kr), 0.5 * (mv + mr)


def berryman_pq(Km, mum, Ki, mui, alpha):
    """
    Berryman's P and Q factors for spheroidal inclusions (Ki, mui, aspect ratio alpha)
    embedded in a background (Km, mum). Valid for oblate spheroids (alpha < 1).
    """
    alpha = min(float(alpha), 0.9999)
    a2 = alpha * alpha
    theta = alpha / (1.0 - a2) ** 1.5 * (np.arccos(alpha) - alpha * np.sqrt(1.0 - a2))
    f = a2 / (1.0 - a2) * (3.0 * theta - 2.0)
    R = 3.0 * mum / (3.0 * Km + 4.0 * mum)
    A = mui / mum - 1.0
    B = (Ki / Km - mui / mum) / 3.0

    F1 = 1 + A * (1.5 * (f + theta) - R * (1.5 * f + 2.5 * theta - 4.0 / 3.0))
    F2 = (1 + A * (1 + 1.5 * (f + theta) - 0.5 * R * (3 * f + 5 * theta))
          + B * (3 - 4 * R)
          + 0.5 * A * (A + 3 * B) * (3 - 4 * R) * (f + theta - R * (f - theta + 2 * theta ** 2)))
    F3 = 1 + A * (1 - (f + 1.5 * theta) + R * (f + theta))
    F4 = 1 + 0.25 * A * (f + 3 * theta - R * (f - theta))
    F5 = A * (-f + R * (f + theta - 4.0 / 3.0)) + B * theta * (3 - 4 * R)
    F6 = 1 + A * (1 + f - R * (f + theta)) + B * (1 - theta) * (3 - 4 * R)
    F7 = 2 + 0.25 * A * (3 * f + 9 * theta - R * (3 * f + 5 * theta)) + B * theta * (3 - 4 * R)
    F8 = A * (1 - 2 * R + 0.5 * f * (R - 1) + 0.5 * theta * (5 * R - 3)) + B * (1 - theta) * (3 - 4 * R)
    F9 = A * ((R - 1) * f - R * theta) + B * theta * (3 - 4 * R)

    Tiijj = 3.0 * F1 / F2
    Tijij_minus = 2.0 / F3 + 1.0 / F4 + (F4 * F5 + F6 * F7 - F8 * F9) / (F2 * F4)
    P = Tiijj / 3.0
    Q = Tijij_minus / 5.0
    return P, Q


def dem(Km, mum, phi, inclusions):
    """
    Differential Effective Medium (Berryman 1992) for a mix of inclusion types.

    Parameters
    ----------
    Km, mum : host (mineral) moduli
    phi     : total inclusion (pore) volume fraction
    inclusions : list of dicts {"frac": share of the pore space, "alpha": aspect ratio,
                 "K": inclusion bulk modulus, "mu": inclusion shear modulus}
                 Shares must sum to 1. Use K=mu=0 for dry pores.
    Returns effective (K, mu).
    """
    if phi <= 0:
        return Km, mum
    shares = np.array([inc["frac"] for inc in inclusions], float)
    shares = shares / shares.sum()

    def rhs(y, km):
        K, mu = max(km[0], 1e-6), max(km[1], 1e-6)
        dK = dmu = 0.0
        for s, inc in zip(shares, inclusions):
            P, Q = berryman_pq(K, mu, inc["K"], inc["mu"], inc["alpha"])
            dK += s * (inc["K"] - K) * P
            dmu += s * (inc["mu"] - mu) * Q
        return [dK / (1.0 - y), dmu / (1.0 - y)]

    sol = solve_ivp(rhs, (0.0, phi), [Km, mum], rtol=1e-6, atol=1e-8, method="RK45")
    return float(sol.y[0, -1]), float(sol.y[1, -1])


def gassmann(Kdry, Kmin, Kfl, phi):
    """Gassmann saturated bulk modulus (shear modulus unchanged)."""
    if phi <= 0:
        return Kmin
    num = (1.0 - Kdry / Kmin) ** 2
    den = phi / Kfl + (1.0 - phi) / Kmin - Kdry / Kmin ** 2
    return Kdry + num / den


def velocities(K, mu, rho):
    """Vp, Vs in km/s for K, mu in GPa and rho in g/cc."""
    vp = np.sqrt((K + 4.0 / 3.0 * mu) / rho)
    vs = np.sqrt(mu / rho)
    return vp, vs


def schoenberg_hti(K, mu, delta_n, delta_t):
    """
    6x6 Voigt stiffness (GPa) of an isotropic background (K, mu) containing one set of
    vertical fractures with normals along x1, described by the normal and tangential
    weaknesses delta_n, delta_t in [0, 1) (Schoenberg & Sayers 1995).
    """
    lam = K - 2.0 / 3.0 * mu
    M = lam + 2 * mu
    r = lam / M
    C = np.zeros((6, 6))
    C[0, 0] = M * (1 - delta_n)
    C[0, 1] = C[0, 2] = C[1, 0] = C[2, 0] = lam * (1 - delta_n)
    C[1, 1] = C[2, 2] = M * (1 - delta_n * r ** 2)
    C[1, 2] = C[2, 1] = lam * (1 - delta_n * r)
    C[3, 3] = mu
    C[4, 4] = C[5, 5] = mu * (1 - delta_t)
    return C


# ---------------------------------------------------------------------------
# Convenience forward models used for templates and the inversion demo
# ---------------------------------------------------------------------------
def carbonate_model(phi, crack_phi, alpha_stiff=0.8, alpha_crack=0.01,
                    dolomite_frac=0.0, Kfl=BRINE["K"], rhofl=BRINE["rho"]):
    """
    Brine-saturated limestone/dolostone: stiff (vuggy/moldic/intercrystalline) pores plus
    compliant cracks. phi = total porosity, crack_phi = crack porosity (<= phi).
    Returns Vp, Vs (km/s), rho (g/cc).
    """
    crack_phi = min(crack_phi, phi)
    Km, mum = vrh([1 - dolomite_frac, dolomite_frac],
                  [MINERALS["calcite"]["K"], MINERALS["dolomite"]["K"]],
                  [MINERALS["calcite"]["mu"], MINERALS["dolomite"]["mu"]])
    rhom = (1 - dolomite_frac) * MINERALS["calcite"]["rho"] + dolomite_frac * MINERALS["dolomite"]["rho"]
    if phi <= 0:
        return (*velocities(Km, mum, rhom), rhom)
    incs = [dict(frac=(phi - crack_phi) / phi, alpha=alpha_stiff, K=0.0, mu=0.0),
            dict(frac=crack_phi / phi, alpha=alpha_crack, K=0.0, mu=0.0)]
    incs = [i for i in incs if i["frac"] > 0]
    Kd, mud = dem(Km, mum, phi, incs)
    Ks = gassmann(Kd, Km, Kfl, phi)
    rho = (1 - phi) * rhom + phi * rhofl
    vp, vs = velocities(Ks, mud, rho)
    return vp, vs, rho


def sandstone_model(phi, crack_phi, clay_frac=0.1, alpha_pore=0.12, alpha_crack=0.01,
                    Kfl=BRINE["K"], rhofl=BRINE["rho"]):
    """Brine-saturated quartz sandstone with intergranular pores plus cracks."""
    crack_phi = min(crack_phi, phi)
    Km, mum = vrh([1 - clay_frac, clay_frac],
                  [MINERALS["quartz"]["K"], MINERALS["clay"]["K"]],
                  [MINERALS["quartz"]["mu"], MINERALS["clay"]["mu"]])
    rhom = (1 - clay_frac) * MINERALS["quartz"]["rho"] + clay_frac * MINERALS["clay"]["rho"]
    if phi <= 0:
        return (*velocities(Km, mum, rhom), rhom)
    incs = [dict(frac=(phi - crack_phi) / phi, alpha=alpha_pore, K=0.0, mu=0.0),
            dict(frac=crack_phi / phi, alpha=alpha_crack, K=0.0, mu=0.0)]
    incs = [i for i in incs if i["frac"] > 0]
    Kd, mud = dem(Km, mum, phi, incs)
    Ks = gassmann(Kd, Km, Kfl, phi)
    rho = (1 - phi) * rhom + phi * rhofl
    vp, vs = velocities(Ks, mud, rho)
    return vp, vs, rho


# ---------------------------------------------------------------------------
# Self-tests: python rockphysics.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # 1) Sphere limit of P, Q for dry pores must match the closed form
    Km, mum = 76.8, 32.0
    P, Q = berryman_pq(Km, mum, 0.0, 0.0, 0.9999)
    P_ref = (Km + 4 / 3 * mum) / (4 / 3 * mum)
    zeta = mum / 6 * (9 * Km + 8 * mum) / (Km + 2 * mum)
    Q_ref = (mum + zeta) / zeta
    print(f"Sphere limit  P={P:.4f} (ref {P_ref:.4f})   Q={Q:.4f} (ref {Q_ref:.4f})")
    assert abs(P - P_ref) / P_ref < 1e-3 and abs(Q - Q_ref) / Q_ref < 1e-3

    # 2) Inclusion identical to host -> no change
    assert np.allclose(dem(Km, mum, 0.2, [dict(frac=1, alpha=0.1, K=Km, mu=mum)]), (Km, mum), rtol=1e-6)

    # 3) Cracks soften much more than stiff pores at equal porosity
    vp_stiff, vs_stiff, _ = carbonate_model(0.10, 0.0)
    vp_crk, vs_crk, _ = carbonate_model(0.10, 0.005)
    print(f"Limestone phi=10%: Vp {vp_stiff:.2f} -> {vp_crk:.2f} km/s with 0.5% cracks")
    assert vp_crk < vp_stiff

    # 4) Zero weakness -> isotropic stiffness
    C = schoenberg_hti(Km, mum, 0.0, 0.0)
    assert np.isclose(C[0, 0], C[2, 2]) and np.isclose(C[3, 3], C[4, 4])
    print("All rock-physics self-tests passed.")
