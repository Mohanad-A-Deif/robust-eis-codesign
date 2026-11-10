"""Loading and modelling of the measured printed-electrode impedance spectra
(Wade et al., Apollo - University of Cambridge Repository, doi:10.17863/CAM.109929, CC BY 4.0)."""
import json, re
import numpy as np
from scipy.optimize import least_squares

EPS0 = 8.8541878128e-12
EPS_W = 78.4
FMAX = 10e6
BAD_CH = {("160", "2-4")}
SKIP_C = {0.6, 0.7, 0.8, 0.9}


def load(path="../data/cam_eis_binned.json"):
    D = json.load(open(path))
    rec = []
    for d in D:
        n = d["n"]
        b = np.array(d["b"], float)
        b = b[1:-1]                                 # drop first and last bins
        b = b[b[:, 0] <= FMAX]
        ch = re.search(r"ADAPTER 1910 ([0-9]+-[0-9]+)", n).group(1)
        r = dict(name=n, f=b[:, 0], Z=b[:, 1] + 1j * b[:, 2], sdr=b[:, 3], sdi=b[:, 4], npts=b[:, 5], ch=ch)
        p = n.split("/")
        if n.startswith("Nyquist Electrode Gap"):
            r.update(study="gap", geom=p[2].split()[0], c=float(p[3]))
        elif n.startswith("[SI] Electrode length"):
            r.update(study="elen", geom=p[2].split()[0], c=float(p[3]))
        elif n.startswith("[SI] Finger length"):
            r.update(study="flen", geom=p[2].split()[0], c=float(p[3]))
        else:
            r.update(study="s2s", geom=p[2] + "/" + p[3], c=np.nan, t=p[4])
        if r["study"] != "s2s" and (r["c"] in SKIP_C or (r["geom"], ch) in BAD_CH and r["study"] == "gap"):
            continue
        rec.append(r)
    return rec


# x = [ln Gsol, ln Q, beta, ln C, ln Rs, ln Ls]
XN = ["ln_Gsol", "ln_Q", "beta", "ln_C", "ln_Rs", "ln_Ls"]
LB = np.array([-25, -30, 0.3, -32, 0, -25]); UB = np.array([5, 0, 1.0, -18, 9, -8])


def zmodel(x, f):
    w = 2 * np.pi * f
    Gs, Q, be, Cp, Rs, Ls = np.exp(x[0]), np.exp(x[1]), x[2], np.exp(x[3]), np.exp(x[4]), np.exp(x[5])
    Zep = 1.0 / (Q * (1j * w) ** be)
    return Rs + 1j * w * Ls + 1.0 / (1.0 / (1.0 / Gs + Zep) + 1j * w * Cp)


def x_init(f, Z):
    hi = Z[np.argmax(f)].real
    R = max(np.abs(Z[np.argmin(np.abs(f - 2e4))]) - hi, 50.0)
    return np.array([np.log(1 / R), np.log(1e-6), 0.6, np.log(2e-11), np.log(max(hi * 0.8, 2)), np.log(1e-7)])


def fit(f, Z, sd, x0=None, prior=None, idx=None, multistart=True):
    """Weighted least squares (optionally MAP with Gaussian prior (mu, sd) on some coordinates)."""
    if idx is not None:
        f, Z, sd = f[idx], Z[idx], sd[idx]
    def res(x):
        r = (zmodel(x, f) - Z) / sd
        out = [r.real, r.imag]
        if prior is not None:
            mu, ps, mask = prior
            out.append(((x - mu) / ps)[mask])
        return np.concatenate(out)
    starts = [x_init(f, Z) if x0 is None else x0]
    if multistart and x0 is None:
        for dl in [-1.5, 1.5]:
            s = starts[0].copy(); s[0] += dl; starts.append(s)
    best = None
    lb, ub = LB, UB
    for s in starts:
        s = np.clip(s, lb + 1e-6, ub - 1e-6)
        try:
            r = least_squares(res, s, bounds=(lb, ub), x_scale="jac", max_nfev=400)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best


def jac(x, f, h=1e-6):
    cols = []
    for k in range(len(x)):
        xp = x.copy(); xm = x.copy(); xp[k] += h; xm[k] -= h
        cols.append((zmodel(xp, f) - zmodel(xm, f)) / (2 * h))
    return np.stack(cols, 1)
