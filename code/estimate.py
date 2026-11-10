"""Monte Carlo estimation study with a higher-fidelity data generator (finite sample height
from 2-D FEM, unmodelled secondary dispersion, nonlinear fabrication errors), MAP estimation
with the design model, and stress tests (prior amplification, reference mismatch)."""
import json, sys, time
import numpy as np
from scipy.optimize import least_squares
import model as M, design as D
from fem_cell import flux_per_gap
from scipy.special import ellipk

H_SAMPLE_UM = 500.0     # sample-layer (well) height used only by the generator
EXTRA_KAPPA, EXTRA_FC = 0.01, 1.0e4      # unmodelled low-frequency dispersion (relative increment, corner)


def phi_an(w, s):
    k = np.cos(np.pi * w / (2 * (w + s))); m = k * k
    return ellipk(1 - m) / (2 * ellipk(m))


def kcorr_true(d):
    w, s = d[1] * 1e6, d[2] * 1e6
    return phi_an(w, s) / flux_per_gap(round(w, 2), round(s, 2), H_SAMPLE_UM)


def generate(omega, d, th, nu, op, rng, kc, ref_bias=0.0, ideal=False):
    sig0 = np.exp(th[0])
    tau2 = 1 / (2 * np.pi * EXTRA_FC)
    extra = None if ideal else (EXTRA_KAPPA * sig0 * tau2 / M.EPS0, tau2)
    P = dict(M.DEFAULT)
    Zs = M.Z_sample(omega, d, th, nu, P, kcorr=kc, extra=extra)
    sd_s = M.NOISE["eta_rel"] * np.abs(Zs) + M.NOISE["eta_abs"]
    if op == "single":
        return Zs + sd_s * (rng.standard_normal(len(omega)) + 1j * rng.standard_normal(len(omega)))
    P2 = dict(P); P2["sigma_r"] = P["sigma_r"] * (1 + ref_bias)
    Zr = M.Z_ref(omega, d, nu, P2, kcorr=kc)
    if M.NOISE["readout"] == "digital":
        sd_r = M.NOISE["eta_rel"] * np.abs(Zr) + M.NOISE["eta_abs"]
        n = lambda s: s * (rng.standard_normal(len(omega)) + 1j * rng.standard_normal(len(omega)))
        return (Zs + n(sd_s)) - (Zr + n(sd_r))
    dZ = Zs - Zr
    sd = M.NOISE["eta_rel"] * np.abs(dZ) + M.NOISE["eta_cm"] * 0.5 * (np.abs(Zs) + np.abs(Zr)) + M.NOISE["eta_abs"]
    return dZ + sd * (rng.standard_normal(len(omega)) + 1j * rng.standard_normal(len(omega)))


def map_estimate(omega, d, y, op, act, nu0, nusd, starts):
    """MAP: theta in box (flat prior), active nuisances with Gaussian prior."""
    p = 4
    def unpack(x):
        nu = nu0.copy(); nu[act] = x[p:]
        return x[:p], nu
    def res(x):
        th, nu = unpack(x)
        mu = M.observe(omega, d, th, nu, op)
        sd = M.noise_sd(omega, d, th, nu, op)
        r = (mu - y) / sd
        return np.concatenate([r.real, r.imag, (x[p:] - nu0[act]) / nusd[act]])
    lb = np.r_[M.THETA_BOX[:, 0], nu0[act] - 6 * nusd[act]]
    ub = np.r_[M.THETA_BOX[:, 1], nu0[act] + 6 * nusd[act]]
    best = None
    for th0 in starts:
        x0 = np.clip(np.r_[th0, nu0[act]], lb + 1e-9, ub - 1e-9)
        try:
            r = least_squares(res, x0, bounds=(lb, ub), x_scale="jac", max_nfev=200, method="trf")
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
    return best.x[:p]


# deterministic multi-start (independent of the true parameters)
STARTS = [M.THETA_C.copy()] + list(D.sobol_pool(4, 7))


def run_mc(design, n_theta=60, seed=0, prior_amp=1.0, ref_bias_sd=0.0, ideal=False):
    op = design["op"]; M.NOISE["readout"] = design["ro"]
    d = tuple(design["d"]); idx = design["idx"]
    omega = D.OMEGA[idx]
    act = np.array(M.active_nuisance(op))
    nu0 = M.nu_vector(); nusd = M.nu_sd_vector()
    rng = np.random.default_rng(1000 + seed)
    thetas = D.sobol_pool(n_theta, 500 + seed)
    kc = 1.0 if ideal else kcorr_true(d)
    errs = []
    for th in thetas:
        nu = nu0.copy()
        nu[act] = nu0[act] + prior_amp * nusd[act] * rng.standard_normal(len(act))
        rb = ref_bias_sd * rng.standard_normal() if op == "diff" else 0.0
        y = generate(omega, d, th, nu, op, rng, kc, rb, ideal)
        starts = STARTS
        est = map_estimate(omega, d, y, op, act, nu0, nusd, starts)
        errs.append((est - th) / M.THETA_TOL)
    return np.array(errs)


def summarize(E):
    rmse = np.sqrt(np.mean(E ** 2, 0))
    p95 = np.percentile(np.abs(E), 95, axis=0)
    comp = np.mean(np.all(np.abs(E) <= 2.0, axis=1))
    return dict(rmse=rmse.tolist(), p95=p95.tolist(), worst_rmse=float(rmse.max()),
                worst_p95=float(p95.max()), compliance=float(comp))


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "main"
    designs = {(r["name"], r["m"]): r for r in json.load(open("../results/designs.json"))}
    out = {}
    names = ["B1", "B2", "B3", "B5", "B6", "B7", "Ours", "Ours-dig", "B3-dig"]
    if which == "main":
        for nm in names:
            key = (nm, 6)
            if key not in designs:
                continue
            res = []
            for s in range(5):
                t = time.time(); E = run_mc(designs[key], seed=s); res.append(summarize(E))
            out[nm] = res
            print(nm, np.mean([r["worst_rmse"] for r in res]), np.mean([r["compliance"] for r in res]), flush=True)
        json.dump(out, open("../results/mc_main.json", "w"), indent=1)
    elif which == "ideal":
        for nm in ["Ours", "Ours-dig", "B7"]:
            out[nm] = [summarize(run_mc(designs[(nm, 6)], n_theta=40, seed=s, ideal=True)) for s in range(2)]
            print(nm, np.mean([r["worst_rmse"] for r in out[nm]]), np.mean([r["compliance"] for r in out[nm]]), flush=True)
        json.dump(out, open("../results/mc_ideal.json", "w"), indent=1)
    elif which == "stress":
        for nm in ["Ours", "B7", "B6", "B3"]:
            out[nm] = {}
            for amp in [1.0, 1.5, 2.0, 3.0]:
                out[nm][f"amp{amp}"] = [summarize(run_mc(designs[(nm, 6)], n_theta=30, seed=s, prior_amp=amp)) for s in range(2)]
            if designs[(nm, 6)]["op"] == "diff":
                for rb in [0.0, 0.01, 0.02, 0.03]:
                    out[nm][f"ref{rb}"] = [summarize(run_mc(designs[(nm, 6)], n_theta=30, seed=s, ref_bias_sd=rb)) for s in range(2)]
            print(nm, "done", flush=True)
            json.dump(out, open("../results/mc_stress.json", "w"), indent=1)


def run_mc_fixed(design, th, n=60, seed=0):
    """Repeated estimation at one fixed medium (e.g. the design's worst-case corner)."""
    op = design["op"]; M.NOISE["readout"] = design["ro"]
    d = tuple(design["d"]); idx = design["idx"]; omega = D.OMEGA[idx]
    act = np.array(M.active_nuisance(op)); nu0 = M.nu_vector(); nusd = M.nu_sd_vector()
    rng = np.random.default_rng(3000 + seed); kc = kcorr_true(d); errs = []
    for _ in range(n):
        nu = nu0.copy(); nu[act] = nu0[act] + nusd[act] * rng.standard_normal(len(act))
        y = generate(omega, d, th, nu, op, rng, kc)
        est = map_estimate(omega, d, y, op, act, nu0, nusd, STARTS)
        errs.append((est - th) / M.THETA_TOL)
    return np.array(errs)
