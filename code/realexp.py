"""Experimental validation on measured printed-electrode spectra (electrode-gap study).

Tasks
 (1) noise / model-adequacy calibration
 (2) frequency-subset design for ln G (solution conductance) with nuisance marginalisation:
     robust (worst case over concentrations), nominal (single concentration), log-uniform,
     sensitivity-band heuristic; evaluated by (a) deviation from the full-spectrum estimate
     and (b) cross-validated concentration prediction (leave-one-channel-out calibration)
 (3) reference operator: single vs matched-reference differential vs mismatched reference
 (4) geometry: model-predicted worst-case CRLB ranking of electrode gaps vs observed error
"""
import json, pickle, itertools
import numpy as np
from scipy.optimize import least_squares
from scipy.stats import spearmanr
import realdata as R

rng = np.random.default_rng(0)
CONC = [0.1, 0.5, 1.0, 2.0, 4.0]
CREF = 1.0
GAPS = ["80", "120", "160", "200", "240"]
EPS_MODEL = 1.1e-2       # model-discrepancy floor (relative) = median full-spectrum residual


def sd_eff(r):
    return np.sqrt(np.maximum(r["sdr"], r["sdi"]) ** 2 + (EPS_MODEL * np.abs(r["Z"])) ** 2)


# ------------------------------------------------------------------ generic efficient-information tools
def G_blocks(x, f, sd):
    J = R.jac(x, f)
    Jr = np.stack([J.real, J.imag], 1) / sd[:, None, None]
    return np.einsum("mai,maj->mij", Jr, Jr)


def eff(A, P, p=1):
    Att, Atn, Ann = A[..., :p, :p], A[..., :p, p:], A[..., p:, p:] + P
    return Att - Atn @ np.linalg.solve(Ann, np.swapaxes(Atn, -1, -2))


def subset_val(Gs, P, idx):
    A = Gs[:, list(idx)].sum(1)
    return eff(A, P)[:, 0, 0].min()


def proj(y, m):
    lo, hi = y.min() - 1, y.max()
    for _ in range(80):
        t = 0.5 * (lo + hi)
        if np.clip(y - t, 0, 1).sum() > m: lo = t
        else: hi = t
    return np.clip(y - 0.5 * (lo + hi), 0, 1)


def robust_design(Gs, P, m, iters=400):
    """Maximise min_t I_eff,t(w) on the capped simplex (concave), then top-m + swap."""
    T, Mg, q, _ = Gs.shape
    w = np.full(Mg, m / Mg); step = None; best = (-np.inf, w)
    for it in range(iters):
        A = np.einsum("l,tlij->tij", w, Gs)
        Ann = A[:, 1:, 1:] + P
        X = np.linalg.solve(Ann, A[:, 1:, :1])            # (T, r, 1)
        val_t = (A[:, :1, :1] - A[:, :1, 1:] @ X)[:, 0, 0]
        lm = val_t.min(); mu = 30 / max(lm, 1e-30)
        pi = np.exp(-mu * (val_t - lm)); pi /= pi.sum()
        u = np.concatenate([np.ones((T, 1, 1)), -X], 1)[:, :, 0]   # (T, q)
        g = np.einsum("tq,tlqr,tr,t->l", u, Gs, u, pi)
        if lm > best[0]: best = (lm, w.copy())
        if step is None: step = 0.5 * m / np.abs(g).max()
        w = proj(w + step * g / (1 + 0.02 * it), m)
    w = best[1]
    idx = list(np.argsort(-w)[:m])
    cur = subset_val(Gs, P, idx)
    for _ in range(3):
        imp = False
        for pos in range(m):
            for c in range(Mg):
                if c in idx: continue
                tr = idx.copy(); tr[pos] = c
                v = subset_val(Gs, P, tr)
                if v > cur * (1 + 1e-9): cur, idx, imp = v, tr, True
        if not imp: break
    return tuple(sorted(idx)), cur


# ------------------------------------------------------------------ estimation from a subset
def fit_subset(f, Z, sd, idx, prior_mu, prior_sd, x0=None, ref=None):
    """MAP estimate of x (ln G free; nuisances with Gaussian prior). If ref=(xr, Zr_meas_idx, sdr),
    the observable is dZ = Z - Zr and the series terms (Rs, Ls) of the reference are shared
    (common-mode) up to a residual dRs with its own prior."""
    f, Z, sd = f[list(idx)], Z[list(idx)], sd[list(idx)]
    mask = np.ones(6, bool); mask[0] = False
    if ref is None:
        def res(x):
            r = (R.zmodel(x, f) - Z) / sd
            return np.concatenate([r.real, r.imag, ((x - prior_mu) / prior_sd)[mask]])
        x0 = prior_mu.copy() if x0 is None else x0
        starts = [x0 + np.r_[d, 0, 0, 0, 0, 0] for d in (0.0, -1.5, 1.5)]
        best = None
        for s in starts:
            s = np.clip(s, R.LB + 1e-6, R.UB - 1e-6)
            rr = least_squares(res, s, bounds=(R.LB, R.UB), x_scale="jac", max_nfev=300)
            if best is None or rr.cost < best.cost: best = rr
        return best.x
    xr, Zr, sdr, dRs_sd = ref
    Zr = Zr[list(idx)]; sdr = sdr[list(idx)]
    dZ = Z - Zr; sdd = np.sqrt(sd ** 2 + sdr ** 2)
    # unknowns: ln G, ln Q, beta, ln C (sample), dRs (series residual); Ls shared exactly
    mu = np.r_[prior_mu[:4], 0.0]; ps = np.r_[prior_sd[:4], dRs_sd]
    lb = np.r_[R.LB[:4], -5 * dRs_sd]; ub = np.r_[R.UB[:4], 5 * dRs_sd]
    def model(y):
        xs = np.r_[y[:4], np.log(np.exp(xr[4]) + y[4]) if np.exp(xr[4]) + y[4] > 1 else 0.0, xr[5]]
        return R.zmodel(xs, f) - R.zmodel(xr, f)
    def res(y):
        r = (model(y) - dZ) / sdd
        return np.concatenate([r.real, r.imag, ((y - mu) / ps)[1:]])
    best = None
    for d in (0.0, -1.5, 1.5):
        s = np.clip(mu + np.r_[d, 0, 0, 0, 0], lb + 1e-6, ub - 1e-6)
        rr = least_squares(res, s, bounds=(lb, ub), x_scale="jac", max_nfev=300)
        if best is None or rr.cost < best.cost: best = rr
    return best.x


def main():
    fits = pickle.load(open("../results/gapfits.pkl", "rb"))
    F = fits[0]["f"]
    assert all(len(r["f"]) == len(F) and np.allclose(r["f"], F, rtol=1e-6) for r in fits)
    out = dict(freq=F.tolist())
    # ---------------- (1) noise & adequacy
    relnoise = [np.median(np.maximum(r["sdr"], r["sdi"]) / np.abs(r["Z"])) for r in fits]
    out["noise_rel_median"] = float(np.median(relnoise))
    out["noise_rel_p10_p90"] = np.percentile(relnoise, [10, 90]).tolist()
    out["fit_rel_rms_median"] = float(np.median([r["rel"] for r in fits]))
    out["fit_rel_rms_p90"] = float(np.percentile([r["rel"] for r in fits], 90))
    # channel-to-channel spread of fitted conductance (same gap, same solution)
    cv = []
    for g in GAPS:
        for c in CONC:
            G = [np.exp(r["xfull"][0]) for r in fits if r["geom"] == g and r["c"] == c]
            cv.append(np.std(G, ddof=1) / np.mean(G))
    out["channel_cv_G_median"] = float(np.median(cv)); out["channel_cv_G_p90"] = float(np.percentile(cv, 90))

    res_rows = []
    pred_rows = []
    for g in GAPS:
        recs = [r for r in fits if r["geom"] == g]
        chans = sorted(set(r["ch"] for r in recs))
        for test_ch in chans:
            train = [r for r in recs if r["ch"] != test_ch and r["c"] in CONC]
            test = [r for r in recs if r["ch"] == test_ch and r["c"] in CONC]
            X = np.array([r["xfull"] for r in train])
            pmu = X.mean(0); psd = np.maximum(X.std(0, ddof=1), np.array([1, 0.2, 0.03, 0.1, 0.05, 0.5]))
            # wider nuisance prior: include between-concentration variation
            P = np.diag(1 / psd[1:] ** 2)
            # scenarios: training-channel fits at each concentration
            Gs = np.array([G_blocks(r["xfull"], F, sd_eff(r)) for r in train])
            Gs_nom = np.array([G_blocks(r["xfull"], F, sd_eff(r)) for r in train if r["c"] == CREF])
            # reference (matched: same channel at CREF; mismatched: another channel at CREF)
            ref_m = [r for r in recs if r["ch"] == test_ch and r["c"] == CREF][0]
            other = [r for r in recs if r["ch"] != test_ch and r["c"] == CREF]
            ref_x = other[0]
            dRs = [np.exp(r["xfull"][4]) - np.exp([q for q in train if q["ch"] == r["ch"] and q["c"] == CREF][0]["xfull"][4]) for r in train]
            dRs_sd = max(np.std(dRs, ddof=1) + np.abs(np.mean(dRs)), 5.0)
            for m in [4, 5, 6]:
                designs = {
                    "log-uniform": tuple(np.unique(np.round(np.linspace(0, len(F) - 1, m)).astype(int))),
                }
                sc = np.sqrt(np.einsum("tlii->tl", Gs[:, :, :1, :1])).mean(0)
                band = np.where(sc >= 0.1 * sc.max())[0]
                designs["heuristic"] = tuple(np.unique(np.round(np.linspace(band.min(), band.max(), m)).astype(int)))
                designs["nominal"], _ = robust_design(Gs_nom, P, m)
                designs["robust"], _ = robust_design(Gs, P, m)
                for dn, idx in designs.items():
                    pv = subset_val(Gs, P, idx)
                    for r in test:
                        s = sd_eff(r)
                        xs = fit_subset(r["f"], r["Z"], s, idx, pmu, psd)
                        row = dict(gap=g, ch=test_ch, m=m, design=dn, op="single", c=r["c"],
                                   lnG_full=float(r["xfull"][0]), lnG=float(xs[0]), pred_wc_var=float(1 / pv))
                        res_rows.append(row)
                        if dn == "robust":
                            for opn, refr in [("matched-ref", ref_m), ("mismatched-ref", ref_x)]:
                                if opn == "matched-ref" and r["c"] == CREF:
                                    continue
                                y = fit_subset(r["f"], r["Z"], s, idx, pmu, psd,
                                               ref=(refr["xfull"], refr["Z"], sd_eff(refr), dRs_sd))
                                res_rows.append(dict(gap=g, ch=test_ch, m=m, design=dn, op=opn, c=r["c"],
                                                     lnG_full=float(r["xfull"][0]), lnG=float(y[0]), pred_wc_var=float(1 / pv)))
            print(g, test_ch, "done", flush=True)
    out["rows"] = res_rows
    json.dump(out, open("../results/realdata_results.json", "w"))


if __name__ == "__main__":
    main()
